import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agentic_rag.api import routes
from agentic_rag.security.auth import Principal


class FakeSessionStore:
    def __init__(self):
        self.calls = []
        self.session = {"id": "memory-id"}

    def list_sessions(self, course_name, *, owner):
        self.calls.append(("list", course_name, owner))
        return []

    def create_session(self, course_name, *, owner):
        self.calls.append(("create", course_name, owner))
        return "memory-id"

    def get_session(self, session_id, *, owner):
        self.calls.append(("get", session_id, owner))
        return self.session

    def get_session_turns(self, session_id, *, owner):
        self.calls.append(("turns", session_id, owner))
        return []

    def delete_session(self, session_id, *, owner):
        self.calls.append(("delete", session_id, owner))
        return self.session is not None


def _app(store):
    rag_system = SimpleNamespace(reset_thread=lambda session_id: None)
    chat_interface = SimpleNamespace(session_memory=store, rag_system=rag_system)
    return SimpleNamespace(chat_interface=chat_interface)


def test_session_routes_forward_validated_owner(monkeypatch):
    store = FakeSessionStore()
    principal = Principal("tenant-a", "user-a")
    monkeypatch.setattr(routes, "get_rag_app", lambda: _app(store))

    result = asyncio.run(routes.list_sessions("Math", principal))
    created = asyncio.run(routes.create_session(routes.CreateSessionRequest(course_name="Math"), principal))

    assert result == {"sessions": []}
    assert created["session_id"] == "memory-id"
    assert store.calls == [
        ("list", "Math", principal),
        ("create", "Math", principal),
    ]


def test_session_lookup_hides_missing_or_unowned_session(monkeypatch):
    store = FakeSessionStore()
    store.session = None
    principal = Principal("tenant-b", "user-b")
    monkeypatch.setattr(routes, "get_rag_app", lambda: _app(store))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.get_session_turns("stolen-memory-id", principal))

    assert exc_info.value.status_code == 404
    assert store.calls == [("get", "stolen-memory-id", principal)]


def test_delete_hides_missing_or_unowned_session(monkeypatch):
    store = FakeSessionStore()
    store.session = None
    principal = Principal("tenant-b", "user-b")
    monkeypatch.setattr(routes, "get_rag_app", lambda: _app(store))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(routes.delete_session("stolen-memory-id", principal))

    assert exc_info.value.status_code == 404
