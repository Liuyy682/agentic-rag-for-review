from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status

from agentic_rag import config


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    role: str = "user"
    email: str = ""
    token_id: str = ""
    expires_at: int = 0


def validate_auth_config() -> None:
    if len(config.AUTH_JWT_SECRET) < 32:
        raise RuntimeError("AUTH_JWT_SECRET must be at least 32 characters")
    if not config.AUTH_TENANT_ID:
        raise RuntimeError("AUTH_TENANT_ID is required")
    if config.AUTH_TOKEN_TTL_SECONDS <= 0:
        raise RuntimeError("AUTH_TOKEN_TTL_SECONDS must be positive")
    if config.APP_ENV == "production" and not config.AUTH_COOKIE_SECURE:
        raise RuntimeError("AUTH_COOKIE_SECURE must be enabled in production")
    if bool(config.INITIAL_ADMIN_EMAIL) != bool(config.INITIAL_ADMIN_PASSWORD):
        raise RuntimeError("INITIAL_ADMIN_EMAIL and INITIAL_ADMIN_PASSWORD must be set together")


def _unauthorized(detail: str = "Invalid authentication credentials") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _redis_unavailable() -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication is temporarily unavailable")


@lru_cache(maxsize=1)
def _redis_client() -> Any:
    try:
        import redis
        return redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=config.REDIS_SOCKET_TIMEOUT_SECONDS,
            health_check_interval=30,
        )
    except Exception as exc:
        raise RuntimeError("Redis authentication client is unavailable") from exc


def _revocation_key(token_id: str) -> str:
    return f"agentic-rag:auth:revoked:{token_id}"


def issue_token(user: dict) -> str:
    now = int(time.time())
    expires_at = now + config.AUTH_TOKEN_TTL_SECONDS
    return jwt.encode(
        {
            "sub": str(user["user_id"]),
            "tenant_id": str(user["tenant_id"]),
            "email": str(user["email"]),
            "role": str(user["role"]),
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": expires_at,
        },
        config.AUTH_JWT_SECRET,
        algorithm="HS256",
    )


def _decode_token(token: str) -> Principal:
    try:
        claims = jwt.decode(token, config.AUTH_JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc
    user_id = str(claims.get("sub") or "").strip()
    tenant_id = str(claims.get("tenant_id") or "").strip()
    role = str(claims.get("role") or "").strip()
    email = str(claims.get("email") or "").strip()
    token_id = str(claims.get("jti") or "").strip()
    expires_at = claims.get("exp")
    if not user_id or not tenant_id or role not in {"user", "admin"} or not email or not token_id:
        raise _unauthorized("Required identity claims are missing")
    if tenant_id != config.AUTH_TENANT_ID or not isinstance(expires_at, int):
        raise _unauthorized()
    try:
        if _redis_client().exists(_revocation_key(token_id)):
            raise _unauthorized("Authentication token has been revoked")
    except HTTPException:
        raise
    except Exception as exc:
        raise _redis_unavailable() from exc
    return Principal(tenant_id, user_id, role, email, token_id, expires_at)


def get_principal(request: Request) -> Principal:
    token = request.cookies.get(config.AUTH_COOKIE_NAME)
    if not token:
        raise _unauthorized("Authentication is required")
    return _decode_token(token)


def get_admin(principal: Principal = Depends(get_principal)) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access is required")
    return principal


def revoke_token(principal: Principal) -> None:
    remaining = principal.expires_at - int(time.time())
    if remaining <= 0:
        return
    try:
        _redis_client().set(_revocation_key(principal.token_id), "1", ex=remaining)
    except Exception as exc:
        raise _redis_unavailable() from exc
