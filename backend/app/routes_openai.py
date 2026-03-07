from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import get_services
from .schemas import ErrorDetail, ErrorResponse, OpenAIModelCard, OpenAIModelListResponse
from .services import AppServices

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai"])
CONTEXT_SAFETY_TOKENS = 32


def openai_error(status_code: int, message: str, *, error_type: str = "invalid_request_error", code: str | None = None) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(message=message, type=error_type, code=code))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    hop_by_hop = {"host", "connection", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


def _extract_error_message(body: bytes) -> str | None:
    if not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return body.decode("utf-8", errors="replace")
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
    if isinstance(data, str):
        return data
    return body.decode("utf-8", errors="replace")


def _extract_context_limit(message: str | None) -> int | None:
    if not message:
        return None
    patterns = (
        r"max_total_tokens=(\d+)",
        r"max_model_len[^0-9]*(\d+)",
        r"context length is only\s+(\d+)\s+tokens",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            try:
                limit = int(match.group(1))
            except ValueError:
                return None
            return limit if limit > 0 else None
    return None


def _adjust_max_tokens(payload: dict[str, Any], limit: int) -> dict[str, Any] | None:
    requested = payload.get("max_tokens")
    if not isinstance(requested, int) or requested < 1:
        return None
    adjusted = max(1, limit - CONTEXT_SAFETY_TOKENS)
    if adjusted >= requested:
        return None
    retried = dict(payload)
    retried["max_tokens"] = adjusted
    return retried


def _directory_size_bytes(path: str) -> int | None:
    if not os.path.isdir(path):
        return None
    total = 0
    try:
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    total += os.path.getsize(file_path)
                except OSError:
                    continue
    except OSError:
        return None
    return total


async def forward_openai_request(
    services: AppServices,
    path: str,
    payload: dict[str, Any],
    incoming_headers: dict[str, str] | None = None,
) -> Response:
    requested_model = payload.get("model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        return openai_error(400, "Missing required field: model")

    state = services.registry.resolve_for_request(requested_model)
    if state is None:
        return openai_error(404, f"Model '{requested_model}' is not known", code="model_not_found")
    if not state.running or not state.endpoint:
        return openai_error(409, f"Model '{requested_model}' is downloaded but not running", code="model_not_running")

    upstream_model = state.served_model_name or f"/models/{state.model_id}"
    upstream_payload = dict(payload)
    upstream_payload["model"] = upstream_model

    url = f"{state.endpoint}/v1{path}"
    headers = _filtered_headers(incoming_headers or {})
    headers.setdefault("content-type", "application/json")

    async def send_upstream(request_payload: dict[str, Any]) -> httpx.Response | None:
        try:
            request = services.http_client.build_request("POST", url, json=request_payload, headers=headers)
            return await services.http_client.send(request, stream=True)
        except httpx.TimeoutException:
            return None
        except httpx.HTTPError as exc:
            logger.exception("Upstream request failed for model %s", requested_model)
            raise RuntimeError(f"Upstream model request failed: {exc}") from exc

    try:
        upstream = await send_upstream(upstream_payload)
        if upstream is None:
            return openai_error(504, "Upstream model request timed out", error_type="timeout_error")
    except RuntimeError as exc:
        return openai_error(502, str(exc), error_type="upstream_error")

    if upstream.status_code == 400:
        error_body = await upstream.aread()
        error_message = _extract_error_message(error_body)
        await upstream.aclose()
        context_limit = _extract_context_limit(error_message)
        adjusted_payload = _adjust_max_tokens(upstream_payload, context_limit) if context_limit is not None else None
        if adjusted_payload is not None:
            logger.info(
                "Retrying %s for model %s with reduced max_tokens=%s after context limit error",
                path,
                requested_model,
                adjusted_payload.get("max_tokens"),
            )
            try:
                upstream = await send_upstream(adjusted_payload)
                if upstream is None:
                    return openai_error(504, "Upstream model request timed out", error_type="timeout_error")
            except RuntimeError as exc:
                return openai_error(502, str(exc), error_type="upstream_error")
        else:
            return Response(content=error_body, status_code=400, media_type="application/json")
    elif upstream is None:
        return openai_error(504, "Upstream model request timed out", error_type="timeout_error")

    if payload.get("stream") is True:
        response_headers = {}
        content_type = upstream.headers.get("content-type")
        if content_type:
            response_headers["content-type"] = content_type
        if "cache-control" in upstream.headers:
            response_headers["cache-control"] = upstream.headers["cache-control"]

        async def streamer():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            streamer(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=response_headers.get("content-type"),
        )

    body = await upstream.aread()
    media_type = upstream.headers.get("content-type", "application/json")
    status_code = upstream.status_code
    await upstream.aclose()
    return Response(content=body, status_code=status_code, media_type=media_type)


@router.get("/v1/models", response_model=OpenAIModelListResponse)
async def list_models(services: AppServices = Depends(get_services)) -> OpenAIModelListResponse:
    cards: list[OpenAIModelCard] = []
    for m in services.registry.running_models():
        model_name = m.nickname or m.served_model_name or m.model_id
        size_bytes = _directory_size_bytes(m.local_path)
        size_gb = round(size_bytes / (1024**3), 2) if size_bytes is not None else None
        display_name = f"{model_name} ({size_gb:.2f} GB)" if size_gb is not None else model_name
        cards.append(
            OpenAIModelCard(
                id=model_name,
                display_name=display_name,
                size_gb=size_gb,
            )
        )
    return OpenAIModelListResponse(data=cards)


@router.post("/v1/completions")
async def completions(request: Request, services: AppServices = Depends(get_services)) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict):
        return openai_error(400, "JSON body must be an object")
    return await forward_openai_request(services, "/completions", payload, dict(request.headers))


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, services: AppServices = Depends(get_services)) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict):
        return openai_error(400, "JSON body must be an object")
    return await forward_openai_request(services, "/chat/completions", payload, dict(request.headers))
