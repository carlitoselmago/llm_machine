from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    message: str
    type: str = "error"
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DownloadModelRequest(BaseModel):
    repo_id: str = Field(min_length=1)
    revision: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


class StartModelRequest(BaseModel):
    model_id: str | None = None
    served_model_name: str | None = None
    trust_remote_code: bool = False
    dtype: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None


class ModelInfo(BaseModel):
    repo_id: str
    model_id: str
    local_path: str
    downloaded: bool = False
    download_status: str = "not_downloaded"
    running: bool = False
    gpu_id: str | None = None
    container_id: str | None = None
    port: int | None = None
    endpoint: str | None = None
    served_model_name: str | None = None
    error: str | None = None


class GpuInfo(BaseModel):
    gpu_id: str
    name: str
    allocated: bool = False
    allocated_model_id: str | None = None


class AdminHealthResponse(BaseModel):
    status: str
    docker_connected: bool
    gpu_count: int


class AdminContainersResponse(BaseModel):
    containers: list[dict[str, Any]]


class OpenAIModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llm-orchestrator"


class OpenAIModelListResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIModelCard]
