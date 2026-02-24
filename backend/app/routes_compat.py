from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from .auth import get_services
from .routes_openai import forward_openai_request
from .services import AppServices

router = APIRouter(tags=["compat"])


@router.get("/api/models")
async def compat_models(services: AppServices = Depends(get_services)) -> list[str]:
    return [m.served_model_name or m.model_id for m in services.registry.running_models()]


@router.post("/api/complete")
async def compat_complete(request: Request, services: AppServices = Depends(get_services)) -> Response:
    payload: Any = await request.json()
    if not isinstance(payload, dict):
        return Response(status_code=400, content='{"error":"JSON body must be object"}', media_type="application/json")
    payload.setdefault("stream", True)
    return await forward_openai_request(services, "/completions", payload, dict(request.headers))
