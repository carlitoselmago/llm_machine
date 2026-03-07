from __future__ import annotations

import logging
import os
import re
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
    _repo_files_cache: dict[str, set[str] | None] = field(default_factory=dict)

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
        # Refresh before returning so the admin UI reflects current VRAM usage.
        try:
            self.gpu_manager.refresh()
        except RuntimeError as exc:
            logger.warning("GPU refresh failed while listing GPUs: %s", exc)
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
                "log_tail": self.process_manager.get_log_tail(rec.process_id, max_lines=25),
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
    def _extract_base_model_from_readme(local_path: str) -> str | None:
        readme_path = os.path.join(local_path, "README.md")
        if not os.path.isfile(readme_path):
            return None
        try:
            with open(readme_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(128 * 1024)
        except OSError:
            return None

        single_line = re.search(
            r"(?im)^\s*base_model\s*:\s*['\"]?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]?\s*$",
            text,
        )
        if single_line:
            return single_line.group(1).strip()

        block = re.search(r"(?ims)^\s*base_model\s*:\s*\n(?P<body>(?:\s*-\s*.+\n)+)", text)
        if block:
            first_item = re.search(
                r"(?im)^\s*-\s*['\"]?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]?\s*$",
                block.group("body"),
            )
            if first_item:
                return first_item.group(1).strip()
        return None

    def _infer_base_model_from_repo_metadata(self, repo_id: str) -> str | None:
        def _fetch_with_token(token: str | None):
            api = HfApi(token=token)
            try:
                return api.model_info(repo_id=repo_id, repo_type="model")
            except TypeError:
                return api.model_info(repo_id=repo_id)

        try:
            info = _fetch_with_token(self.config.hf_token)
        except Exception:
            try:
                info = _fetch_with_token(None)
            except Exception:
                return None

        card_data = getattr(info, "card_data", None)
        if card_data is None:
            return None
        if hasattr(card_data, "to_dict"):
            try:
                card_data = card_data.to_dict()
            except Exception:
                return None
        if not isinstance(card_data, dict):
            return None

        for key in ("base_model", "base_models", "baseModel"):
            value = card_data.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
        return None

    @staticmethod
    def _base_model_candidates_from_repo_name(repo_id: str) -> list[str]:
        owner, _, name = repo_id.partition("/")
        base_name = name or owner
        stripped = re.sub(r"(?i)[._-]?gguf.*$", "", base_name).strip("._- ")
        if not stripped:
            return []

        candidates: list[str] = []
        if owner and name and stripped != name:
            candidates.append(f"{owner}/{stripped}")
        candidates.append(stripped)
        return candidates

    @staticmethod
    def _add_unique(items: list[str], value: str | None) -> None:
        if not value:
            return
        candidate = value.strip()
        if not candidate or candidate in items:
            return
        items.append(candidate)

    @staticmethod
    def _looks_like_local_path(value: str) -> bool:
        if not value:
            return False
        if os.path.isabs(value):
            return True
        if re.match(r"^[A-Za-z]:[\\/]", value):
            return True
        if value.startswith("./") or value.startswith(".\\") or value.startswith("../") or value.startswith("..\\"):
            return True
        if "\\" in value:
            return True
        return os.path.exists(value)

    @staticmethod
    def _is_hf_repo_id(value: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$", value))

    @staticmethod
    def _tokenizer_file_markers() -> tuple[str, ...]:
        return (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "merges.txt",
        )

    def _local_has_tokenizer_files(self, local_path: str) -> bool:
        try:
            files = {name.lower() for name in os.listdir(local_path) if os.path.isfile(os.path.join(local_path, name))}
        except OSError:
            return False
        markers = set(self._tokenizer_file_markers())
        if files.intersection(markers):
            return True
        if "vocab.json" in files and "merges.txt" in files:
            return True
        return False

    def _list_repo_files_cached(self, repo_id: str) -> set[str] | None:
        cached = self._repo_files_cache.get(repo_id)
        if cached is not None or repo_id in self._repo_files_cache:
            return cached

        def _list_with_token(token: str | None) -> list[str]:
            api = HfApi(token=token)
            try:
                return api.list_repo_files(repo_id=repo_id, repo_type="model")
            except TypeError:
                return api.list_repo_files(repo_id=repo_id)

        files: set[str] | None = None
        try:
            listed = _list_with_token(self.config.hf_token)
            files = {os.path.basename(path).lower() for path in listed}
        except Exception:
            try:
                listed = _list_with_token(None)
                files = {os.path.basename(path).lower() for path in listed}
            except Exception:
                files = None

        self._repo_files_cache[repo_id] = files
        return files

    def _repo_has_tokenizer_files(self, repo_id: str) -> bool:
        files = self._list_repo_files_cached(repo_id)
        if not files:
            return False
        markers = set(self._tokenizer_file_markers())
        if files.intersection(markers):
            return True
        if "vocab.json" in files and "merges.txt" in files:
            return True
        return False

    def _expand_tokenizer_candidates(self, initial_candidates: list[str]) -> list[str]:
        expanded: list[str] = []
        for candidate in initial_candidates:
            self._add_unique(expanded, candidate)
            if self._is_hf_repo_id(candidate):
                self._add_unique(expanded, candidate.split("/", 1)[1])

        seed_values = list(expanded)
        for candidate in seed_values:
            if self._looks_like_local_path(candidate):
                continue
            if "\\" in candidate or ":" in candidate:
                continue
            slug = candidate.split("/", 1)[1] if self._is_hf_repo_id(candidate) else candidate
            parts = [part for part in re.split(r"[-_]+", slug) if part]
            for end_idx in range(len(parts), 0, -1):
                self._add_unique(expanded, "-".join(parts[:end_idx]))

        lowered = [item.lower() for item in expanded]
        if any("gpt2-xl" in item for item in lowered):
            self._add_unique(expanded, "gpt2-xl")
            self._add_unique(expanded, "openai-community/gpt2-xl")
        if any("gpt2" in item for item in lowered):
            self._add_unique(expanded, "gpt2")
            self._add_unique(expanded, "openai-community/gpt2")

        return expanded

    def _select_tokenizer_ref(
        self,
        *,
        model_id: str,
        repo_id: str,
        local_path: str,
        tokenizer_override: str | None = None,
    ) -> str:
        candidates: list[str] = []
        if tokenizer_override:
            self._add_unique(candidates, tokenizer_override)

        self._add_unique(candidates, local_path)
        self._add_unique(candidates, self._extract_base_model_from_readme(local_path))
        self._add_unique(candidates, self._infer_base_model_from_repo_metadata(repo_id))
        for candidate in self._base_model_candidates_from_repo_name(repo_id):
            self._add_unique(candidates, candidate)

        self._add_unique(candidates, repo_id)
        if "/" not in repo_id and "_" in repo_id:
            self._add_unique(candidates, repo_id.replace("_", "/", 1))

        candidates = self._expand_tokenizer_candidates(candidates)

        if not candidates:
            raise RuntimeError(f"Could not determine tokenizer reference for model {model_id}")

        validated: list[str] = []
        for ref in candidates:
            if self._looks_like_local_path(ref):
                if self._local_has_tokenizer_files(ref):
                    validated.append(ref)
                continue
            if self._is_hf_repo_id(ref) and self._repo_has_tokenizer_files(ref):
                validated.append(ref)

        if validated:
            selected = validated[0]
            logger.info("Tokenizer candidates for %s: %s", model_id, candidates)
            logger.info("Selected tokenizer for %s: %s", model_id, selected)
            return selected

        # Best-effort fallback: only choose valid HF repo ids here. Bare slugs
        # like "Ninja-v1-NSFW" are not valid model identifiers and just create
        # noisier downstream failures.
        for ref in candidates:
            if self._is_hf_repo_id(ref):
                logger.warning(
                    "No tokenizer artifact match for %s; falling back to repo candidate: %s",
                    model_id,
                    ref,
                )
                return ref

        # Last fallback to keep behavior deterministic.
        selected = candidates[0]
        logger.warning(
            "No tokenizer artifact match for %s; falling back to first candidate: %s",
            model_id,
            selected,
        )
        return selected

    def _resolve_model_launch_spec(
        self,
        *,
        model_id: str,
        repo_id: str,
        local_path: str,
        preferred_gguf_file: str | None = None,
        tokenizer_override: str | None = None,
    ) -> _ModelLaunchSpec:
        config_json = os.path.join(local_path, "config.json")
        params_json = os.path.join(local_path, "params.json")
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
            if preferred_gguf_file:
                if preferred_gguf_file not in gguf_files:
                    raise RuntimeError(
                        f"Requested GGUF file '{preferred_gguf_file}' not found in model directory"
                    )
                gguf_name = preferred_gguf_file
            elif len(gguf_files) == 1:
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
            tokenizer_ref = self._select_tokenizer_ref(
                model_id=model_id,
                repo_id=repo_id,
                local_path=local_path,
                tokenizer_override=tokenizer_override,
            )
            hf_config_path = model_dir_ref if (os.path.isfile(config_json) or os.path.isfile(params_json)) else None
            return _ModelLaunchSpec(
                model_ref=model_ref,
                tokenizer_ref=tokenizer_ref,
                hf_config_path=hf_config_path,
            )
        if os.path.isfile(config_json) or os.path.isfile(params_json):
            return _ModelLaunchSpec(model_ref=local_path, tokenizer_ref=tokenizer_override)
        raise RuntimeError(
            "Model directory is missing config.json/params.json required by vLLM. "
            "or GGUF files. Use a compatible Hugging Face Transformers model."
        )

    def start_model(self, model_id: str, options: StartModelRequest) -> ModelInfo:
        with self._ops_lock:
            # Refresh right before allocation to avoid stale free-memory checks.
            self.gpu_manager.refresh()
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
                preferred_gguf_file = options.gguf_file.strip() if options.gguf_file else None
                tokenizer_override = options.tokenizer_id.strip() if options.tokenizer_id else None
                launch_spec = self._resolve_model_launch_spec(
                    model_id=model_id,
                    repo_id=state.repo_id,
                    local_path=state.local_path,
                    preferred_gguf_file=preferred_gguf_file,
                    tokenizer_override=tokenizer_override,
                )

                gpu_memory_utilization = options.gpu_memory_utilization
                if gpu_memory_utilization is None:
                    gpu_memory_utilization = self.config.default_gpu_memory_utilization
                max_num_seqs = options.max_num_seqs
                if max_num_seqs is None:
                    max_num_seqs = self.config.default_max_num_seqs
                max_model_len = options.max_model_len
                if max_model_len is None:
                    max_model_len = self.config.default_max_model_len
                gpu_id = self.gpu_manager.allocate(model_id, min_free_ratio=gpu_memory_utilization)
                used_ports = {m.port for m in self.registry.running_models() if m.port is not None}
                host_port = self.process_manager.find_free_host_port({int(p) for p in used_ports})
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
