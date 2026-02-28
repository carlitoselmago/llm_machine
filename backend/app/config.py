from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return float(value)


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return int(value)


@dataclass(slots=True)
class Config:
    host: str = "0.0.0.0"
    port: int = 8080
    models_dir: str = "/models"
    vllm_image: str = "vllm/vllm-openai:latest"
    vllm_internal_port: int = 8000
    host_port_start: int = 8001
    host_port_end: int = 8999
    container_name_prefix: str = "llmorch"
    managed_label_key: str = "llm_orchestrator.managed"
    label_repo_id: str = "llm_orchestrator.repo_id"
    label_model_id: str = "llm_orchestrator.model_id"
    label_gpu_id: str = "llm_orchestrator.gpu_id"
    label_host_port: str = "llm_orchestrator.host_port"
    label_served_model_name: str = "llm_orchestrator.served_model_name"
    admin_username: str = "admin"
    admin_password: str = "workshop"
    hf_token: str | None = None
    request_timeout_seconds: float = 600.0
    vllm_startup_timeout_seconds: float = 600.0
    vllm_health_poll_seconds: float = 2.0
    default_gpu_memory_utilization: float | None = None
    default_max_num_seqs: int | None = None

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            models_dir=os.getenv("MODELS_DIR", "/models"),
            vllm_image=os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest"),
            vllm_internal_port=int(os.getenv("VLLM_INTERNAL_PORT", "8000")),
            host_port_start=int(os.getenv("HOST_PORT_START", "8001")),
            host_port_end=int(os.getenv("HOST_PORT_END", "8999")),
            container_name_prefix=os.getenv("CONTAINER_NAME_PREFIX", "llmorch"),
            admin_username=os.getenv("ADMIN_USERNAME", "admin"),
            admin_password=os.getenv("ADMIN_PASSWORD", "workshop"),
            hf_token=os.getenv("HF_TOKEN") or None,
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600")),
            vllm_startup_timeout_seconds=float(os.getenv("VLLM_STARTUP_TIMEOUT_SECONDS", "600")),
            vllm_health_poll_seconds=float(os.getenv("VLLM_HEALTH_POLL_SECONDS", "2")),
            default_gpu_memory_utilization=_optional_float_env("DEFAULT_GPU_MEMORY_UTILIZATION"),
            default_max_num_seqs=_optional_int_env("DEFAULT_MAX_NUM_SEQS"),
        )
