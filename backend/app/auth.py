from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .services import AppServices

security = HTTPBasic()


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def require_admin(
    credentials: HTTPBasicCredentials = Depends(security),
    services: AppServices = Depends(get_services),
) -> str:
    username_ok = secrets.compare_digest(credentials.username, services.config.admin_username)
    password_ok = secrets.compare_digest(credentials.password, services.config.admin_password)
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
