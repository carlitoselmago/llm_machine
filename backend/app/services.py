from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field

import httpx
from huggingface_hub import HfApi, snapshot_download

from .config import Config
from .gpu_manager import GPUManager
from .model_registry import ModelRegistry, sanitize_model_id
from .process_manager import ManagedProcessRecord, ProcessManager
from .schemas import DownloadModelRequest, GpuInfo, ModelInfo, StartModelRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ModelLaunchSpec:
    model_ref: str
    tokenizer_ref: str | None = None
    hf_config_path: str | None = None


@dataclass(slots=True)
class AppServices:
    config: Config
    registry: ModelRegistry
    gpu_manager: GPUManager
    process_manager: ProcessManager
    http_client: httpx.AsyncClient
    _ops_lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def create(cls) -> "AppServices":
        config = Config.from_env()
        return cls(
            config=config,
            registry=ModelRegistry(config.models_dir),
            gpu_manager=GPUManager(),
            process_manager=ProcessManager(config),
            http_client=httpx.AsyncClient(timeout=config.request_timeout_seconds),
        )

    def startup(self) -> None:
        self.registry.ensure_models_dir()
        self.process_manager.connect()
        try:
            self.gpu_manager.refresh()
        except RuntimeError as exc:
            logger.warning("GPU detection unavailable; continuing without GPUs: %s", exc)
        self.registry.sync_downloaded_from_disk()
        self.reconcile_runtime_state()

    def reconcile_runtime_state(self) -> None:
        for rec in self.process_manager.list_managed_processes():
            if not rec.model_id or rec.status != "running" or rec.host_port is None:
                continue
            if rec.gpu_id:
                try:
                    self.gpu_manager.reserve_existing(rec.gpu_id, rec.model_id)
                except RuntimeError as exc:
                    logger.warning("Failed to reserve GPU for %s: %s", rec.process_id, exc)
            self.registry.mark_started(
                rec.model_id,
                runtime_id=rec.process_id,
                gpu_id=rec.gpu_id or "",
                port=rec.host_port,
                endpoint=f"http://127.0.0.1:{rec.host_port}",
                served_model_name=rec.served_model_name,
                repo_id=rec.repo_id or rec.model_id,
            )

    async def shutdown(self) -> None:
        await self.http_client.aclose()
        self.process_manager.close()

    def health(self) -> tuple[bool, int]:
        return self.process_manager.is_connected(), len(self.gpu_manager.list_gpus())

    def list_gpus(self) -> list[GpuInfo]:
        return self.gpu_manager.list_gpus()

    def list_models(self) -> list[ModelInfo]:
        self.registry.sync_downloaded_from_disk()
        return self.registry.list_models()

    def set_model_nickname(self, model_id: str, nickname: str | None) -> ModelInfo:
        with self._ops_lock:
            self.registry.sync_downloaded_from_disk()
            return self.registry.set_nickname(model_id, nickname)

    def list_managed_runtimes(self) -> list[dict[str, object]]:
        return [
            {
                "runtime_id": rec.process_id,
                "pid": rec.pid,
                "name": rec.name,
                "status": rec.status,
                "repo_id": rec.repo_id,
                "model_id": rec.model_id,
                "gpu_id": rec.gpu_id,
                "host_port": rec.host_port,
                "served_model_name": rec.served_model_name,
                "log_file": rec.log_file,
            }
            for rec in self.process_manager.list_managed_processes()
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

    def list_repo_files(self, repo_id: str, revision: str | None = None) -> list[str]:
        api = HfApi(token=self.config.hf_token)
        try:
            files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="model")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to list files for repo {repo_id}: {exc}") from exc
        return sorted(files)

    @staticmethod
    def _resolve_model_launch_spec(*, model_id: str, repo_id: str, local_path: str) -> _ModelLaunchSpec:
        config_json = os.path.join(local_path, "config.json")
        params_json = os.path.join(local_path, "params.json")
        if os.path.isfile(config_json) or os.path.isfile(params_json):
            return _ModelLaunchSpec(model_ref=local_path)

        try:
            entries = os.listdir(local_path)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect model directory {local_path}: {exc}") from exc

        gguf_files = sorted(
            name
            for name in entries
            if name.lower().endswith(".gguf") and os.path.isfile(os.path.join(local_path, name))
        )
        if len(gguf_files) >= 1:
            if len(gguf_files) == 1:
                gguf_name = gguf_files[0]
            else:
                sized_gguf_files: list[tuple[int, str]] = []
                for name in gguf_files:
                    full_path = os.path.join(local_path, name)
                    try:
                        sized_gguf_files.append((os.path.getsize(full_path), name))
                    except OSError as exc:
                        raise RuntimeError(f"Failed to read GGUF file size for {name}: {exc}") from exc
                sized_gguf_files.sort(key=lambda item: (item[0], item[1]), reverse=True)
                gguf_name = sized_gguf_files[0][1]
                logger.info(
                    "Multiple GGUF files found for %s; selected largest file: %s",
                    model_id,
                    gguf_name,
                )

            model_dir_ref = local_path
            model_ref = os.path.join(model_dir_ref, gguf_name)
            tokenizer_markers = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
            tokenizer_ref = model_dir_ref if any(os.path.isfile(os.path.join(local_path, marker)) for marker in tokenizer_markers) else None
            hf_config_path = repo_id if repo_id and "/" in repo_id else None
            return _ModelLaunchSpec(
                model_ref=model_ref,
                tokenizer_ref=tokenizer_ref,
                hf_config_path=hf_config_path,
            )
        raise RuntimeError(
            "Model directory is missing config.json/params.json required by vLLM. "
            "or GGUF files. Use a compatible Hugging Face Transformers model."
        )

    def start_model(self, model_id: str, options: StartModelRequest) -> ModelInfo:
        with self._ops_lock:
            self.registry.sync_downloaded_from_disk()
            state = self.registry.get_state(model_id)
            if state is None or not state.downloaded or not os.path.isdir(state.local_path):
                raise FileNotFoundError(f"Model {model_id} is not downloaded")
            if state.download_status == "downloading":
                raise RuntimeError(f"Model {model_id} is still downloading")
            if state.download_status == "loading":
                raise RuntimeError(f"Model {model_id} is already loading")
            if state.download_status == "unloading":
                raise RuntimeError(f"Model {model_id} is unloading")
            if state.running:
                raise RuntimeError(f"Model {model_id} is already running")

            self.registry.mark_loading(model_id)
            gpu_id: str | None = None
            started: ManagedProcessRecord | None = None
            try:
                launch_spec = self._resolve_model_launch_spec(
                    model_id=model_id,
                    repo_id=state.repo_id,
                    local_path=state.local_path,
                )

                gpu_id = self.gpu_manager.allocate(model_id)
                used_ports = {m.port for m in self.registry.running_models() if m.port is not None}
                host_port = self.process_manager.find_free_host_port({int(p) for p in used_ports})
                gpu_memory_utilization = options.gpu_memory_utilization
                if gpu_memory_utilization is None:
                    gpu_memory_utilization = self.config.default_gpu_memory_utilization
                max_num_seqs = options.max_num_seqs
                if max_num_seqs is None:
                    max_num_seqs = self.config.default_max_num_seqs
                max_model_len = options.max_model_len
                if max_model_len is None:
                    max_model_len = self.config.default_max_model_len
                effective_served_model_name = options.served_model_name or state.nickname or model_id
                started = self.process_manager.start_vllm_process(
                    model_id=model_id,
                    repo_id=state.repo_id,
                    gpu_id=gpu_id,
                    host_port=host_port,
                    model_ref=launch_spec.model_ref,
                    tokenizer_ref=launch_spec.tokenizer_ref,
                    hf_config_path=launch_spec.hf_config_path,
                    served_model_name=effective_served_model_name,
                    trust_remote_code=options.trust_remote_code,
                    dtype=options.dtype,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_num_seqs=max_num_seqs,
                )
                loading_info = self.registry.mark_loading_runtime(
                    model_id,
                    runtime_id=started.process_id,
                    gpu_id=gpu_id,
                    port=host_port,
                    endpoint=f"http://127.0.0.1:{host_port}",
                    served_model_name=effective_served_model_name,
                )
                self._start_readiness_watcher(
                    model_id=model_id,
                    repo_id=state.repo_id,
                    runtime_id=started.process_id,
                    gpu_id=gpu_id,
                    host_port=host_port,
                    served_model_name=effective_served_model_name,
                )
                return loading_info
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to start model %s", model_id)
                if started is not None:
                    try:
                        self.process_manager.stop_process(started.process_id)
                    except Exception:  # noqa: BLE001
                        logger.exception("Cleanup failed after start error for %s", model_id)
                if gpu_id is not None:
                    self.gpu_manager.release(gpu_id)
                self.registry.mark_start_failed(model_id, str(exc))
                raise

    def _start_readiness_watcher(
        self,
        *,
        model_id: str,
        repo_id: str,
        runtime_id: str,
        gpu_id: str,
        host_port: int,
        served_model_name: str | None,
    ) -> None:
        watcher = threading.Thread(
            target=self._await_model_ready_worker,
            kwargs={
                "model_id": model_id,
                "repo_id": repo_id,
                "runtime_id": runtime_id,
                "gpu_id": gpu_id,
                "host_port": host_port,
                "served_model_name": served_model_name,
            },
            daemon=True,
        )
        watcher.start()

    def _await_model_ready_worker(
        self,
        *,
        model_id: str,
        repo_id: str,
        runtime_id: str,
        gpu_id: str,
        host_port: int,
        served_model_name: str | None,
    ) -> None:
        try:
            self.process_manager.wait_for_model_ready(host_port, runtime_id=runtime_id)
            with self._ops_lock:
                state = self.registry.get_state(model_id)
                if state is None:
                    return
                if state.runtime_id != runtime_id:
                    return
                self.registry.mark_started(
                    model_id,
                    runtime_id=runtime_id,
                    gpu_id=gpu_id,
                    port=host_port,
                    endpoint=f"http://127.0.0.1:{host_port}",
                    served_model_name=served_model_name,
                    repo_id=repo_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model readiness failed for %s", model_id)
            with self._ops_lock:
                try:
                    self.process_manager.stop_process(runtime_id)
                except Exception:  # noqa: BLE001
                    logger.exception("Failed stopping runtime after readiness failure for %s", model_id)
                self.gpu_manager.release(gpu_id)
                self.registry.mark_start_failed(model_id, str(exc))

    def stop_model(self, model_id: str) -> ModelInfo:
        with self._ops_lock:
            state = self.registry.get_state(model_id)
            if state is None:
                raise FileNotFoundError(f"Model {model_id} not found")
            if state.download_status == "unloading":
                raise RuntimeError(f"Model {model_id} is already unloading")
            if not state.running or not state.runtime_id:
                raise RuntimeError(f"Model {model_id} is not running")
            self.registry.mark_unloading(model_id)
            try:
                self.process_manager.stop_process(state.runtime_id)
                if state.gpu_id:
                    self.gpu_manager.release(state.gpu_id)
                return self.registry.mark_stopped(model_id)
            except Exception as exc:  # noqa: BLE001
                self.registry.mark_stop_failed(model_id, str(exc))
                raise

    def delete_model(self, model_id: str) -> None:
        with self._ops_lock:
            state = self.registry.get_state(model_id)
            if state is None:
                raise FileNotFoundError(f"Model {model_id} not found")
            if state.download_status in {"downloading", "loading", "unloading"}:
                raise RuntimeError(f"Model {model_id} is busy ({state.download_status}) and cannot be deleted")
            if state.running:
                raise RuntimeError(f"Model {model_id} is running and cannot be deleted")
            if os.path.isdir(state.local_path):
                shutil.rmtree(state.local_path)
            self.registry.delete_model(model_id)
