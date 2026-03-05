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
    gguf_file: str | None = None
    tokenizer_id: str | None = None
    trust_remote_code: bool = False
    dtype: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None


class ModelNicknameRequest(BaseModel):
    nickname: str | None = None


class ModelInfo(BaseModel):
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


class GpuInfo(BaseModel):
    gpu_id: str
    name: str
    allocated: bool = False
    allocated_model_id: str | None = None


class AdminHealthResponse(BaseModel):
    status: str
    runtime_connected: bool
    runtime_mode: str = "process"
    gpu_count: int


class AdminRuntimesResponse(BaseModel):
    runtimes: list[dict[str, Any]]


class AdminRepoFilesResponse(BaseModel):
    repo_id: str
    files: list[str]
    gguf_files: list[str]


class OpenAIModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llm-orchestrator"
    display_name: str | None = None
    size_gb: float | None = None


class OpenAIModelListResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIModelCard]
