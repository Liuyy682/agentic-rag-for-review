from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from agentic_rag import config
from agentic_rag.security import auth


class FakeRedis:
    def __init__(self):
        self.values = {}

    def exists(self, key):
        return int(key in self.values)

    def set(self, key, value, ex):
        self.values[key] = (value, ex)


def _request(token=None):
    headers = [] if token is None else [(b"cookie", f"{config.AUTH_COOKIE_NAME}={token}".encode())]
    return Request({"type": "http", "method": "GET", "path": "/api/auth/me", "headers": headers})


@pytest.fixture(autouse=True)
def auth_config(monkeypatch):
    monkeypatch.setattr(config, "AUTH_JWT_SECRET", "test-secret-that-is-longer-than-thirty-two-characters")
    monkeypatch.setattr(config, "AUTH_TENANT_ID", "tenant-a")
    monkeypatch.setattr(config, "AUTH_TOKEN_TTL_SECONDS", 3600)
    auth._redis_client.cache_clear()


def test_token_uses_expected_identity_claims():
    user = {"user_id": "user-a", "tenant_id": "tenant-a", "email": "a@example.com", "role": "user"}
    with patch.object(auth, "_redis_client", return_value=FakeRedis()):
        principal = auth.get_principal(_request(auth.issue_token(user)))

    assert principal.user_id == "user-a"
    assert principal.tenant_id == "tenant-a"
    assert principal.role == "user"


def test_missing_or_invalid_cookie_is_unauthorized():
    with pytest.raises(HTTPException) as missing:
        auth.get_principal(_request())
    assert missing.value.status_code == 401

    with patch.object(auth, "_redis_client", return_value=FakeRedis()):
        with pytest.raises(HTTPException) as invalid:
            auth.get_principal(_request("not-a-token"))
    assert invalid.value.status_code == 401


def test_revoked_token_is_unauthorized_and_logout_records_remaining_ttl():
    redis = FakeRedis()
    user = {"user_id": "user-a", "tenant_id": "tenant-a", "email": "a@example.com", "role": "user"}
    with patch.object(auth, "_redis_client", return_value=redis):
        token = auth.issue_token(user)
        principal = auth.get_principal(_request(token))
        auth.revoke_token(principal)
        with pytest.raises(HTTPException) as revoked:
            auth.get_principal(_request(token))

    assert revoked.value.status_code == 401
    assert redis.values[auth._revocation_key(principal.token_id)][1] > 0


def test_missing_required_claims_are_rejected():
    token = jwt.encode(
        {"sub": "user-a", "tenant_id": "tenant-a", "exp": 9999999999},
        config.AUTH_JWT_SECRET,
        algorithm="HS256",
    )
    with patch.object(auth, "_redis_client", return_value=FakeRedis()):
        with pytest.raises(HTTPException) as exc_info:
            auth.get_principal(_request(token))
    assert exc_info.value.status_code == 401


def test_redis_failure_fails_closed():
    user = {"user_id": "user-a", "tenant_id": "tenant-a", "email": "a@example.com", "role": "user"}
    with patch.object(auth, "_redis_client", side_effect=RuntimeError("down")):
        with pytest.raises(HTTPException) as exc_info:
            auth.get_principal(_request(auth.issue_token(user)))
    assert exc_info.value.status_code == 503


def test_auth_config_requires_a_strong_secret(monkeypatch):
    monkeypatch.setattr(config, "AUTH_JWT_SECRET", "short")
    with pytest.raises(RuntimeError, match="AUTH_JWT_SECRET"):
        auth.validate_auth_config()
