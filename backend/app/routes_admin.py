from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .auth import get_services, require_admin
from .model_registry import sanitize_model_id
from .schemas import (
    AdminHealthResponse,
    AdminRepoFilesResponse,
    AdminRuntimesResponse,
    DownloadModelRequest,
    GpuInfo,
    ModelNicknameRequest,
    ModelInfo,
    StartModelRequest,
)
from .services import AppServices

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])
templates = Jinja2Templates(directory="app/templates")


def _http_error_from_exc(exc: Exception) -> HTTPException:
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=msg)
    if isinstance(exc, RuntimeError):
        detail_lower = msg.lower()
        if "nvidia-smi not found" in detail_lower:
            return HTTPException(status_code=503, detail=msg)
        if "runtime unavailable" in detail_lower:
            return HTTPException(status_code=503, detail=msg)
        if "missing config.json/params.json required by vllm" in detail_lower:
            return HTTPException(status_code=400, detail=msg)
        if "free memory on device" in detail_lower or "gpu memory utilization" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "out of memory" in detail_lower or "cuda out of memory" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "max_num_seqs" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "runtime is restarting repeatedly" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "still downloading" in detail_lower or "loading" in detail_lower or "unloading" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "busy (" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "already" in detail_lower or "not running" in detail_lower or "cannot be deleted" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
        if "no free gpu" in detail_lower or "no free host port" in detail_lower:
            return HTTPException(status_code=409, detail=msg)
    return HTTPException(status_code=500, detail=msg)


@router.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def admin_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "LLM Orchestrator Admin",
        },
    )


@router.get(
    "/api/admin/health",
    response_model=AdminHealthResponse,
    dependencies=[Depends(require_admin)],
)
async def admin_health(services: AppServices = Depends(get_services)) -> AdminHealthResponse:
    runtime_ok, gpu_count = services.health()
    return AdminHealthResponse(
        status="ok" if runtime_ok else "degraded",
        runtime_connected=runtime_ok,
        runtime_mode="process",
        gpu_count=gpu_count,
    )


@router.get(
    "/api/admin/gpus",
    response_model=list[GpuInfo],
    dependencies=[Depends(require_admin)],
)
async def admin_gpus(services: AppServices = Depends(get_services)) -> list[GpuInfo]:
    return services.list_gpus()


@router.get(
    "/api/admin/models",
    response_model=list[ModelInfo],
    dependencies=[Depends(require_admin)],
)
async def admin_models(services: AppServices = Depends(get_services)) -> list[ModelInfo]:
    return await run_in_threadpool(services.list_models)


@router.get(
    "/api/admin/runtimes",
    response_model=AdminRuntimesResponse,
    dependencies=[Depends(require_admin)],
)
async def admin_runtimes(services: AppServices = Depends(get_services)) -> AdminRuntimesResponse:
    runtimes = await run_in_threadpool(services.list_managed_runtimes)
    return AdminRuntimesResponse(runtimes=runtimes)


@router.get(
    "/api/admin/containers",
    response_model=AdminRuntimesResponse,
    dependencies=[Depends(require_admin)],
)
async def admin_containers_compat(services: AppServices = Depends(get_services)) -> AdminRuntimesResponse:
    runtimes = await run_in_threadpool(services.list_managed_runtimes)
    return AdminRuntimesResponse(runtimes=runtimes)


@router.post(
    "/api/admin/models/download",
    response_model=ModelInfo,
    dependencies=[Depends(require_admin)],
)
async def download_model(
    payload: DownloadModelRequest,
    services: AppServices = Depends(get_services),
) -> ModelInfo:
    try:
        return await run_in_threadpool(services.download_model, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Download request failed")
        raise _http_error_from_exc(exc) from exc


@router.get(
    "/api/admin/repos/files",
    response_model=AdminRepoFilesResponse,
    dependencies=[Depends(require_admin)],
)
async def repo_files(
    repo_id: str,
    revision: str | None = None,
    services: AppServices = Depends(get_services),
) -> AdminRepoFilesResponse:
    repo_id = repo_id.strip()
    if not repo_id:
        raise HTTPException(status_code=400, detail="repo_id is required")
    try:
        files = await run_in_threadpool(services.list_repo_files, repo_id, revision)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Repo files request failed for %s", repo_id)
        raise _http_error_from_exc(exc) from exc
    gguf_files = [name for name in files if name.lower().endswith(".gguf")]
    return AdminRepoFilesResponse(repo_id=repo_id, files=files, gguf_files=gguf_files)


@router.post(
    "/api/admin/models/{model_id}/start",
    response_model=ModelInfo,
    dependencies=[Depends(require_admin)],
)
async def start_model(
    model_id: str,
    payload: Annotated[StartModelRequest | None, Body()] = None,
    services: AppServices = Depends(get_services),
) -> ModelInfo:
    options = payload or StartModelRequest(model_id=model_id)
    if options.model_id and sanitize_model_id(options.model_id) != sanitize_model_id(model_id):
        raise HTTPException(status_code=400, detail="Body model_id does not match path model_id")
    try:
        return await run_in_threadpool(services.start_model, sanitize_model_id(model_id), options)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Start request failed for %s", model_id)
        raise _http_error_from_exc(exc) from exc


@router.post(
    "/api/admin/models/{model_id}/stop",
    response_model=ModelInfo,
    dependencies=[Depends(require_admin)],
)
async def stop_model(model_id: str, services: AppServices = Depends(get_services)) -> ModelInfo:
    try:
        return await run_in_threadpool(services.stop_model, sanitize_model_id(model_id))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stop request failed for %s", model_id)
        raise _http_error_from_exc(exc) from exc


@router.put(
    "/api/admin/models/{model_id}/nickname",
    response_model=ModelInfo,
    dependencies=[Depends(require_admin)],
)
async def update_model_nickname(
    model_id: str,
    payload: ModelNicknameRequest,
    services: AppServices = Depends(get_services),
) -> ModelInfo:
    try:
        return await run_in_threadpool(
            services.set_model_nickname,
            sanitize_model_id(model_id),
            payload.nickname,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Nickname update failed for %s", model_id)
        raise _http_error_from_exc(exc) from exc


@router.delete(
    "/api/admin/models/{model_id}",
    dependencies=[Depends(require_admin)],
)
async def delete_model(model_id: str, services: AppServices = Depends(get_services)) -> dict[str, str]:
    try:
        await run_in_threadpool(services.delete_model, sanitize_model_id(model_id))
        return {"status": "deleted", "model_id": sanitize_model_id(model_id)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Delete request failed for %s", model_id)
        raise _http_error_from_exc(exc) from exc
