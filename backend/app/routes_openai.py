from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .auth import get_services
from .schemas import ErrorDetail, ErrorResponse, OpenAIModelCard, OpenAIModelListResponse
from .services import AppServices

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai"])


def openai_error(status_code: int, message: str, *, error_type: str = "invalid_request_error", code: str | None = None) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(message=message, type=error_type, code=code))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _filtered_headers(headers: dict[str, str]) -> dict[str, str]:
    hop_by_hop = {"host", "connection", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in hop_by_hop}


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

    try:
        request = services.http_client.build_request("POST", url, json=upstream_payload, headers=headers)
        upstream = await services.http_client.send(request, stream=True)
    except httpx.TimeoutException:
        return openai_error(504, "Upstream model request timed out", error_type="timeout_error")
    except httpx.HTTPError as exc:
        logger.exception("Upstream request failed for model %s", requested_model)
        return openai_error(502, f"Upstream model request failed: {exc}", error_type="upstream_error")

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
    cards = [
        OpenAIModelCard(id=m.served_model_name or m.model_id)
        for m in services.registry.running_models()
    ]
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
