import pytest

from agentic_rag.security.users import normalize_email, validate_password


def test_email_is_normalized_and_password_is_not_accepted_when_weak():
    assert normalize_email("  USER@Example.COM ") == "user@example.com"
    with pytest.raises(ValueError, match="valid email"):
        normalize_email("not-an-email")
    with pytest.raises(ValueError, match="12 characters"):
        validate_password("too-short")
