from __future__ import annotations

import re
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from psycopg2.extras import RealDictCursor

from agentic_rag.storage.postgres import transaction

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_password_hasher = PasswordHasher()


class InvalidCredentialsError(ValueError):
    pass


def normalize_email(email: str) -> str:
    value = (email or "").strip().casefold()
    if not _EMAIL_RE.fullmatch(value):
        raise ValueError("A valid email address is required")
    return value


def validate_password(password: str) -> str:
    if len(password or "") < 12:
        raise ValueError("Password must be at least 12 characters")
    return password


class UserRepository:
    def create_user(self, *, email: str, password: str, tenant_id: str, role: str = "user") -> dict:
        email = normalize_email(email)
        password = validate_password(password)
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO app_users (user_id, tenant_id, email, password_hash, role)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING user_id, tenant_id, email, role, is_active
                    """,
                    (str(uuid.uuid4()), tenant_id, email, _password_hasher.hash(password), role),
                )
                row = cur.fetchone()
        return dict(row)

    def find_by_email(self, email: str) -> dict | None:
        email = normalize_email(email)
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT user_id, tenant_id, email, password_hash, role, is_active
                    FROM app_users WHERE email = %s
                    """,
                    (email,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def authenticate(self, *, email: str, password: str) -> dict:
        user = self.find_by_email(email)
        if user is None or not user["is_active"]:
            raise InvalidCredentialsError("Invalid email or password")
        try:
            valid = _password_hasher.verify(user["password_hash"], password or "")
        except (InvalidHashError, VerifyMismatchError):
            valid = False
        if not valid:
            raise InvalidCredentialsError("Invalid email or password")
        if _password_hasher.check_needs_rehash(user["password_hash"]):
            with transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE app_users SET password_hash = %s, updated_at = now() WHERE user_id = %s",
                        (_password_hasher.hash(password), user["user_id"]),
                    )
        return user
