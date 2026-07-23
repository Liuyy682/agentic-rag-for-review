import os

import pytest

from agentic_rag.chat.pg_session_memory import PgSessionMemoryStore
from agentic_rag.security.auth import Principal
from agentic_rag.storage import postgres


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PG_STORAGE_TESTS") != "1",
    reason="Set RUN_PG_STORAGE_TESTS=1 with a local PostgreSQL database to run storage integration tests.",
)


@pytest.fixture
def store():
    postgres.reset_pool_for_tests()
    result = PgSessionMemoryStore()
    yield result
    with postgres.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE tenant_id LIKE 'test-tenant-%'")
    postgres.reset_pool_for_tests()


def test_sessions_and_turns_are_owner_scoped(store):
    owner = Principal("test-tenant-a", "user-a")
    other_user = Principal("test-tenant-a", "user-b")
    other_tenant = Principal("test-tenant-b", "user-a")
    session_id = store.create_session("Math", owner=owner)
    assert store.append_turn(session_id, "question", "answer", "Math", owner=owner)

    assert store.get_session(session_id, owner=owner)["id"] == session_id
    assert store.get_session_turns(session_id, owner=owner)[0]["assistant_final"] == "answer"
    assert store.get_session(session_id, owner=other_user) is None
    assert store.get_session(session_id, owner=other_tenant) is None
    assert store.get_session_turns(session_id, owner=other_user) == []
    assert store.delete_session(session_id, owner=other_user) is False
    assert store.get_session(session_id, owner=owner) is not None


def test_delete_cascades_turns_for_the_owner(store):
    owner = Principal("test-tenant-delete", "user-a")
    session_id = store.create_session(owner=owner)
    store.append_turn(session_id, "question", "answer", owner=owner)

    assert store.delete_session(session_id, owner=owner) is True
    assert store.get_session(session_id, owner=owner) is None
    assert store.get_session_turns(session_id, owner=owner) == []
