from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .logging_config import setup_logging
from .routes_admin import router as admin_router
from .routes_compat import router as compat_router
from .routes_openai import router as openai_router
from .services import AppServices


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    services = AppServices.create()
    app.state.services = services
    services.startup()
    try:
        yield
    finally:
        await services.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(title="LLM Orchestrator", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(admin_router)
    app.include_router(openai_router)
    app.include_router(compat_router)

    static_front_dir = os.getenv("FRONT_STATIC_DIR", "app/static/front")
    app.mount("/", StaticFiles(directory=static_front_dir, html=True), name="static-front")
    return app


app = create_app()
