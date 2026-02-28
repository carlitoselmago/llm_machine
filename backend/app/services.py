from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field

import httpx
from huggingface_hub import snapshot_download

from .config import Config
from .docker_manager import DockerManager, ManagedContainerRecord
from .gpu_manager import GPUManager
from .model_registry import ModelRegistry, sanitize_model_id
from .schemas import DownloadModelRequest, GpuInfo, ModelInfo, StartModelRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AppServices:
    config: Config
    registry: ModelRegistry
    gpu_manager: GPUManager
    docker_manager: DockerManager
    http_client: httpx.AsyncClient
    _ops_lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def create(cls) -> "AppServices":
        config = Config.from_env()
        return cls(
            config=config,
            registry=ModelRegistry(config.models_dir),
            gpu_manager=GPUManager(),
            docker_manager=DockerManager(config),
            http_client=httpx.AsyncClient(timeout=config.request_timeout_seconds),
        )

    def startup(self) -> None:
        self.registry.ensure_models_dir()
        try:
            self.docker_manager.connect()
        except RuntimeError as exc:
            logger.warning("Docker unavailable; running in degraded mode: %s", exc)
        try:
            self.gpu_manager.refresh()
        except RuntimeError as exc:
            logger.warning("GPU detection unavailable; continuing without GPUs: %s", exc)
        self.registry.sync_downloaded_from_disk()
        if self.docker_manager.is_connected():
            self.reconcile_runtime_state()

    def reconcile_runtime_state(self) -> None:
        for rec in self.docker_manager.list_managed_containers():
            if not rec.model_id or rec.status != "running" or rec.host_port is None:
                continue
            if rec.gpu_id:
                try:
                    self.gpu_manager.reserve_existing(rec.gpu_id, rec.model_id)
                except RuntimeError as exc:
                    logger.warning("Failed to reserve GPU for %s: %s", rec.container_id, exc)
            self.registry.mark_started(
                rec.model_id,
                container_id=rec.container_id,
                gpu_id=rec.gpu_id or "",
                port=rec.host_port,
                endpoint=f"http://127.0.0.1:{rec.host_port}",
                served_model_name=rec.served_model_name,
                repo_id=rec.repo_id or rec.model_id,
            )

    async def shutdown(self) -> None:
        await self.http_client.aclose()
        self.docker_manager.close()

    def health(self) -> tuple[bool, int]:
        return self.docker_manager.is_connected(), len(self.gpu_manager.list_gpus())

    def list_gpus(self) -> list[GpuInfo]:
        return self.gpu_manager.list_gpus()

    def list_models(self) -> list[ModelInfo]:
        self.registry.sync_downloaded_from_disk()
        return self.registry.list_models()

    def list_managed_containers(self) -> list[dict[str, object]]:
        if not self.docker_manager.is_connected():
            return []
        return [
            {
                "container_id": rec.container_id,
                "name": rec.name,
                "status": rec.status,
                "repo_id": rec.repo_id,
                "model_id": rec.model_id,
                "gpu_id": rec.gpu_id,
                "host_port": rec.host_port,
                "served_model_name": rec.served_model_name,
            }
            for rec in self.docker_manager.list_managed_containers()
        ]

    def download_model(self, req: DownloadModelRequest) -> ModelInfo:
        with self._ops_lock:
            self.registry.mark_downloading(req.repo_id)
        model_id = sanitize_model_id(req.repo_id)
        local_dir = os.path.join(self.config.models_dir, model_id)
        try:
            os.makedirs(self.config.models_dir, exist_ok=True)
            snapshot_download(
                repo_id=req.repo_id,
                revision=req.revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=req.allow_patterns,
                ignore_patterns=req.ignore_patterns,
                token=self.config.hf_token,
            )
            return self.registry.mark_downloaded(req.repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Download failed for repo=%s", req.repo_id)
            self.registry.mark_download_failed(req.repo_id, str(exc))
            raise

    @staticmethod
    def _validate_model_directory_for_vllm(local_path: str) -> None:
        config_json = os.path.join(local_path, "config.json")
        params_json = os.path.join(local_path, "params.json")
        if os.path.isfile(config_json) or os.path.isfile(params_json):
            return

        try:
            entries = os.listdir(local_path)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect model directory {local_path}: {exc}") from exc

        has_gguf = any(name.lower().endswith(".gguf") for name in entries)
        if has_gguf:
            raise RuntimeError(
                "Model format appears to be GGUF, which is not supported by this vLLM workflow. "
                "Use a Hugging Face Transformers model directory with config.json (or params.json)."
            )
        raise RuntimeError(
            "Model directory is missing config.json/params.json required by vLLM. "
            "Use a compatible Hugging Face Transformers model."
        )

    def start_model(self, model_id: str, options: StartModelRequest) -> ModelInfo:
        with self._ops_lock:
            if not self.docker_manager.is_connected():
                raise RuntimeError("Docker daemon is unavailable")
            self.registry.sync_downloaded_from_disk()
            state = self.registry.get_state(model_id)
            if state is None or not os.path.isdir(state.local_path):
                raise FileNotFoundError(f"Model {model_id} is not downloaded")
            if state.running:
                raise RuntimeError(f"Model {model_id} is already running")
            try:
                self._validate_model_directory_for_vllm(state.local_path)
            except RuntimeError as exc:
                self.registry.mark_start_failed(model_id, str(exc))
                raise

            gpu_id = self.gpu_manager.allocate(model_id)
            used_ports = {m.port for m in self.registry.running_models() if m.port is not None}
            host_port = self.docker_manager.find_free_host_port({int(p) for p in used_ports})
            gpu_memory_utilization = options.gpu_memory_utilization
            if gpu_memory_utilization is None:
                gpu_memory_utilization = self.config.default_gpu_memory_utilization
            if gpu_memory_utilization is None:
                gpu_memory_utilization = 0.75
            max_num_seqs = options.max_num_seqs
            if max_num_seqs is None:
                max_num_seqs = self.config.default_max_num_seqs
            if max_num_seqs is None:
                max_num_seqs = 64
            effective_served_model_name = options.served_model_name or model_id
            started: ManagedContainerRecord | None = None
            try:
                self.docker_manager.pull_image_if_needed()
                started = self.docker_manager.start_vllm_container(
                    model_id=model_id,
                    repo_id=state.repo_id,
                    gpu_id=gpu_id,
                    host_port=host_port,
                    served_model_name=effective_served_model_name,
                    trust_remote_code=options.trust_remote_code,
                    dtype=options.dtype,
                    max_model_len=options.max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                )
                self.docker_manager.wait_for_model_ready(host_port, container_id=started.container_id)
                return self.registry.mark_started(
                    model_id,
                    container_id=started.container_id,
                    gpu_id=gpu_id,
                    port=host_port,
                    endpoint=f"http://127.0.0.1:{host_port}",
                    served_model_name=effective_served_model_name,
                    repo_id=state.repo_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start model %s", model_id)
                if started is not None:
                    try:
                        self.docker_manager.stop_and_remove_container(started.container_id)
                    except Exception:  # noqa: BLE001
                        logger.exception("Cleanup failed after start error for %s", model_id)
                self.gpu_manager.release(gpu_id)
                self.registry.mark_start_failed(model_id, str(exc))
                raise

    def stop_model(self, model_id: str) -> ModelInfo:
        with self._ops_lock:
            if not self.docker_manager.is_connected():
                raise RuntimeError("Docker daemon is unavailable")
            state = self.registry.get_state(model_id)
            if state is None:
                raise FileNotFoundError(f"Model {model_id} not found")
            if not state.running or not state.container_id:
                raise RuntimeError(f"Model {model_id} is not running")
            self.docker_manager.stop_and_remove_container(state.container_id)
            if state.gpu_id:
                self.gpu_manager.release(state.gpu_id)
            return self.registry.mark_stopped(model_id)

    def delete_model(self, model_id: str) -> None:
        with self._ops_lock:
            state = self.registry.get_state(model_id)
            if state is None:
                raise FileNotFoundError(f"Model {model_id} not found")
            if state.running:
                raise RuntimeError(f"Model {model_id} is running and cannot be deleted")
            if os.path.isdir(state.local_path):
                shutil.rmtree(state.local_path)
            self.registry.delete_model(model_id)
