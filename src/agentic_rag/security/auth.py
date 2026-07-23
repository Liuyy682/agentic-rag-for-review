from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from agentic_rag import config


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str


_bearer = HTTPBearer(auto_error=False)


def validate_auth_config() -> None:
    if config.AUTH_MODE == "dev":
        if not config.DEV_TENANT_ID or not config.DEV_USER_ID:
            raise RuntimeError("DEV_TENANT_ID and DEV_USER_ID are required in dev auth mode")
        return
    if config.AUTH_MODE != "oidc":
        raise RuntimeError("AUTH_MODE must be either 'dev' or 'oidc'")

    required = {
        "OIDC_ISSUER": config.OIDC_ISSUER,
        "OIDC_AUDIENCE": config.OIDC_AUDIENCE,
        "OIDC_JWKS_URL": config.OIDC_JWKS_URL,
        "OIDC_TENANT_CLAIM": config.OIDC_TENANT_CLAIM,
        "OIDC_USER_CLAIM": config.OIDC_USER_CLAIM,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing OIDC configuration: {', '.join(missing)}")
    if not config.OIDC_ALGORITHMS:
        raise RuntimeError("OIDC_ALGORITHMS must contain at least one allowed algorithm")


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    return PyJWKClient(config.OIDC_JWKS_URL, cache_keys=True)


def _unauthorized(detail: str = "Invalid authentication credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if config.AUTH_MODE == "dev":
        return Principal(config.DEV_TENANT_ID, config.DEV_USER_ID)

    if config.AUTH_MODE != "oidc":
        raise HTTPException(status_code=500, detail="Authentication is not configured")
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Bearer token is required")

    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(credentials.credentials)
        claims = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=list(config.OIDC_ALGORITHMS),
            audience=config.OIDC_AUDIENCE,
            issuer=config.OIDC_ISSUER,
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized() from exc
    except Exception as exc:
        raise _unauthorized("Unable to validate bearer token") from exc

    tenant_id = str(claims.get(config.OIDC_TENANT_CLAIM) or "").strip()
    user_id = str(claims.get(config.OIDC_USER_CLAIM) or "").strip()
    if not tenant_id or not user_id:
        raise _unauthorized("Required identity claims are missing")
    return Principal(tenant_id=tenant_id, user_id=user_id)
