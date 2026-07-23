from types import SimpleNamespace
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from agentic_rag import config
from agentic_rag.security import auth


def test_dev_mode_returns_configured_identity(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "dev")
    monkeypatch.setattr(config, "DEV_TENANT_ID", "tenant-a")
    monkeypatch.setattr(config, "DEV_USER_ID", "user-a")

    principal = auth.get_principal(None)

    assert principal.tenant_id == "tenant-a"
    assert principal.user_id == "user-a"


def test_oidc_mode_requires_bearer_token(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "oidc")

    with pytest.raises(HTTPException) as exc_info:
        auth.get_principal(None)

    assert exc_info.value.status_code == 401


def test_oidc_mode_validates_token_and_extracts_owner(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "oidc")
    monkeypatch.setattr(config, "OIDC_AUDIENCE", "rag-api")
    monkeypatch.setattr(config, "OIDC_ISSUER", "https://issuer.example/")
    monkeypatch.setattr(config, "OIDC_TENANT_CLAIM", "tenant_id")
    monkeypatch.setattr(config, "OIDC_USER_CLAIM", "sub")
    monkeypatch.setattr(config, "OIDC_ALGORITHMS", ("RS256",))
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="signed-token")
    jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key="public-key")
    )

    with patch.object(auth, "_jwks_client", return_value=jwks), patch.object(
        auth.jwt,
        "decode",
        return_value={"tenant_id": "tenant-a", "sub": "user-a"},
    ) as decode:
        principal = auth.get_principal(credentials)

    assert principal == auth.Principal("tenant-a", "user-a")
    decode.assert_called_once_with(
        "signed-token",
        "public-key",
        algorithms=["RS256"],
        audience="rag-api",
        issuer="https://issuer.example/",
    )


def test_oidc_mode_rejects_missing_identity_claims(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "oidc")
    monkeypatch.setattr(config, "OIDC_AUDIENCE", "rag-api")
    monkeypatch.setattr(config, "OIDC_ISSUER", "https://issuer.example/")
    monkeypatch.setattr(config, "OIDC_TENANT_CLAIM", "tenant_id")
    monkeypatch.setattr(config, "OIDC_USER_CLAIM", "sub")
    monkeypatch.setattr(config, "OIDC_ALGORITHMS", ("RS256",))
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="signed-token")
    jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key="public-key")
    )

    with patch.object(auth, "_jwks_client", return_value=jwks), patch.object(
        auth.jwt,
        "decode",
        return_value={"sub": "user-a"},
    ):
        with pytest.raises(HTTPException) as exc_info:
            auth.get_principal(credentials)

    assert exc_info.value.status_code == 401


def test_oidc_mode_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "oidc")
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
    jwks = SimpleNamespace(
        get_signing_key_from_jwt=lambda token: SimpleNamespace(key="public-key")
    )

    with patch.object(auth, "_jwks_client", return_value=jwks), patch.object(
        auth.jwt,
        "decode",
        side_effect=jwt.InvalidTokenError("bad"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            auth.get_principal(credentials)

    assert exc_info.value.status_code == 401


def test_validate_auth_config_rejects_missing_oidc_settings(monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "oidc")
    monkeypatch.setattr(config, "OIDC_ISSUER", "")
    monkeypatch.setattr(config, "OIDC_AUDIENCE", "")
    monkeypatch.setattr(config, "OIDC_JWKS_URL", "")

    with pytest.raises(RuntimeError, match="Missing OIDC configuration"):
        auth.validate_auth_config()
