from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass

from .schemas import ModelInfo

MODEL_META_FILENAME = ".llm_orchestrator.json"


def sanitize_model_id(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "model"


@dataclass(slots=True)
class _ModelState:
    repo_id: str
    model_id: str
    local_path: str
    nickname: str | None = None
    downloaded: bool = False
    download_status: str = "not_downloaded"
    running: bool = False
    gpu_id: str | None = None
    runtime_id: str | None = None
    port: int | None = None
    endpoint: str | None = None
    served_model_name: str | None = None
    error: str | None = None

    def to_model_info(self) -> ModelInfo:
        return ModelInfo(
            repo_id=self.repo_id,
            model_id=self.model_id,
            local_path=self.local_path,
            nickname=self.nickname,
            downloaded=self.downloaded,
            download_status=self.download_status,
            running=self.running,
            gpu_id=self.gpu_id,
            runtime_id=self.runtime_id,
            port=self.port,
            endpoint=self.endpoint,
            served_model_name=self.served_model_name,
            error=self.error,
        )


class ModelRegistry:
    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self._lock = threading.RLock()
        self._models: dict[str, _ModelState] = {}

    def ensure_models_dir(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)

    def sync_downloaded_from_disk(self) -> list[ModelInfo]:
        self.ensure_models_dir()
        with self._lock:
            seen: set[str] = set()
            for name in os.listdir(self.models_dir):
                full_path = os.path.join(self.models_dir, name)
                if not os.path.isdir(full_path):
                    continue
                model_id = sanitize_model_id(name)
                persisted_repo_id = self._read_persisted_repo_id(full_path)
                state = self._models.get(model_id)
                if state is None:
                    state = _ModelState(
                        repo_id=persisted_repo_id or name,
                        model_id=model_id,
                        local_path=full_path,
                    )
                    self._models[model_id] = state
                elif persisted_repo_id:
                    state.repo_id = persisted_repo_id
                state.local_path = full_path
                if state.download_status == "downloading":
                    state.downloaded = False
                else:
                    state.downloaded = True
                    if state.running:
                        if state.download_status != "unloading":
                            state.download_status = "ready"
                    elif state.download_status in {"not_downloaded", "failed", "ready"}:
                        state.download_status = "downloaded"
                seen.add(model_id)

            for model_id, state in list(self._models.items()):
                if model_id in seen:
                    continue
                if not state.running:
                    state.downloaded = False
                    if state.download_status not in {"downloading", "loading", "unloading"}:
                        state.download_status = "not_downloaded"
            return self.list_models()

    def _get_or_create(self, repo_id: str) -> _ModelState:
        model_id = sanitize_model_id(repo_id)
        state = self._models.get(model_id)
        if state is None:
            state = _ModelState(
                repo_id=repo_id,
                model_id=model_id,
                local_path=os.path.join(self.models_dir, model_id),
            )
            self._models[model_id] = state
        else:
            state.repo_id = repo_id
            state.local_path = os.path.join(self.models_dir, model_id)
        return state

    def mark_downloading(self, repo_id: str) -> ModelInfo:
        with self._lock:
            state = self._get_or_create(repo_id)
            if state.download_status == "downloading":
                raise RuntimeError(f"Model {state.model_id} is already downloading")
            if state.download_status in {"loading", "unloading"}:
                raise RuntimeError(f"Model {state.model_id} is busy ({state.download_status})")
            if state.running:
                raise RuntimeError(f"Model {state.model_id} is running")
            state.download_status = "downloading"
            state.error = None
            return state.to_model_info()

    def mark_downloaded(self, repo_id: str) -> ModelInfo:
        with self._lock:
            state = self._get_or_create(repo_id)
            state.downloaded = True
            state.download_status = "downloaded"
            state.error = None
            self._persist_repo_id(state.local_path, state.repo_id)
            return state.to_model_info()

    def mark_download_failed(self, repo_id: str, error: str) -> ModelInfo:
        with self._lock:
            state = self._get_or_create(repo_id)
            state.downloaded = state.downloaded and os.path.isdir(state.local_path)
            state.download_status = "downloaded" if state.downloaded else "failed"
            state.error = error
            return state.to_model_info()

    def set_nickname(self, model_id: str, nickname: str | None) -> ModelInfo:
        with self._lock:
            state = self._models.get(model_id)
            if state is None:
                raise FileNotFoundError(f"Model {model_id} not found")
            desired = (nickname or "").strip() or None
            if desired and len(desired) > 120:
                raise RuntimeError("Nickname must be 120 characters or fewer")
            if desired:
                for candidate in self._models.values():
                    if candidate.model_id == model_id:
                        continue
                    if desired in {
                        candidate.model_id,
                        candidate.nickname,
                        candidate.served_model_name,
                    }:
                        raise RuntimeError(f"Nickname '{desired}' is already used by another model")
            state.nickname = desired
            self._persist_repo_id(state.local_path, state.repo_id)
            return state.to_model_info()

    def mark_loading(self, model_id: str) -> ModelInfo:
        with self._lock:
            state = self._models[model_id]
            if not state.downloaded:
                raise RuntimeError(f"Model {model_id} is not downloaded")
            if state.running:
                raise RuntimeError(f"Model {model_id} is already running")
            if state.download_status == "loading":
                raise RuntimeError(f"Model {model_id} is already loading")
            if state.download_status == "unloading":
                raise RuntimeError(f"Model {model_id} is unloading")
            state.download_status = "loading"
            state.error = None
            return state.to_model_info()

    def mark_loading_runtime(
        self,
        model_id: str,
        *,
        runtime_id: str,
        gpu_id: str,
        port: int,
        endpoint: str,
        served_model_name: str | None = None,
    ) -> ModelInfo:
        with self._lock:
            state = self._models[model_id]
            state.download_status = "loading"
            state.running = False
            state.runtime_id = runtime_id
            state.gpu_id = gpu_id
            state.port = port
            state.endpoint = endpoint
            state.served_model_name = served_model_name
            state.error = None
            return state.to_model_info()

    def mark_started(
        self,
        model_id: str,
        *,
        runtime_id: str,
        gpu_id: str | None,
        port: int,
        endpoint: str,
        served_model_name: str | None = None,
        repo_id: str | None = None,
    ) -> ModelInfo:
        with self._lock:
            state = self._models.get(model_id)
            if state is None:
                state = self._get_or_create(repo_id or model_id)
            if repo_id:
                state.repo_id = repo_id
            self._persist_repo_id(state.local_path, state.repo_id)
            state.downloaded = True
            state.download_status = "ready"
            state.running = True
            state.runtime_id = runtime_id
            state.gpu_id = gpu_id
            state.port = port
            state.endpoint = endpoint
            state.served_model_name = served_model_name
            state.error = None
            return state.to_model_info()

    def mark_stopped(self, model_id: str) -> ModelInfo:
        with self._lock:
            state = self._models[model_id]
            state.running = False
            state.runtime_id = None
            state.gpu_id = None
            state.port = None
            state.endpoint = None
            state.served_model_name = None
            state.downloaded = os.path.isdir(state.local_path)
            state.download_status = "downloaded" if state.downloaded else "not_downloaded"
            state.error = None
            return state.to_model_info()

    def mark_unloading(self, model_id: str) -> ModelInfo:
        with self._lock:
            state = self._models[model_id]
            if not state.running:
                raise RuntimeError(f"Model {model_id} is not running")
            state.download_status = "unloading"
            state.error = None
            return state.to_model_info()

    def mark_start_failed(self, model_id: str, error: str) -> ModelInfo | None:
        with self._lock:
            state = self._models.get(model_id)
            if state is None:
                return None
            state.running = False
            state.runtime_id = None
            state.gpu_id = None
            state.port = None
            state.endpoint = None
            state.served_model_name = None
            state.downloaded = os.path.isdir(state.local_path)
            state.download_status = "downloaded" if state.downloaded else "not_downloaded"
            state.error = error
            return state.to_model_info()

    def mark_stop_failed(self, model_id: str, error: str) -> ModelInfo | None:
        with self._lock:
            state = self._models.get(model_id)
            if state is None:
                return None
            if state.running:
                state.download_status = "ready"
            else:
                state.downloaded = os.path.isdir(state.local_path)
                state.download_status = "downloaded" if state.downloaded else "not_downloaded"
            state.error = error
            return state.to_model_info()

    def delete_model(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)

    def list_models(self) -> list[ModelInfo]:
        with self._lock:
            return [s.to_model_info() for s in sorted(self._models.values(), key=lambda x: x.model_id.lower())]

    def running_models(self) -> list[ModelInfo]:
        with self._lock:
            return [
                s.to_model_info()
                for s in sorted(self._models.values(), key=lambda x: x.model_id.lower())
                if s.running
            ]

    def get_model(self, model_id: str) -> ModelInfo | None:
        with self._lock:
            state = self._models.get(model_id)
            return state.to_model_info() if state else None

    def get_state(self, model_id: str) -> _ModelState | None:
        with self._lock:
            return self._models.get(model_id)

    def resolve_for_request(self, requested_model: str) -> _ModelState | None:
        with self._lock:
            state = self._models.get(requested_model)
            if state:
                return state
            for candidate in self._models.values():
                if candidate.served_model_name == requested_model:
                    return candidate
                if candidate.nickname == requested_model:
                    return candidate
            return None

    @staticmethod
    def _persist_repo_id(local_path: str, repo_id: str) -> None:
        if not local_path or not os.path.isdir(local_path):
            return
        meta_path = os.path.join(local_path, MODEL_META_FILENAME)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"repo_id": repo_id}, f)
        except OSError:
            return

    @staticmethod
    def _read_persisted_repo_id(local_path: str) -> str | None:
        meta_path = os.path.join(local_path, MODEL_META_FILENAME)
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        repo_id = payload.get("repo_id")
        if not isinstance(repo_id, str):
            return None
        repo_id = repo_id.strip()
        return repo_id or None
