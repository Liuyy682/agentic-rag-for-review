import json

import pytest

from agentic_rag.chat.hot_session_memory import HotSessionMemoryStore
from agentic_rag.chat.redis_memory import MemoryBackendUnavailable, RedisSessionMemoryCache
from agentic_rag.security.auth import Principal


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = []
        self.available = True
        self.closed = False

    def ping(self):
        if not self.available:
            raise ConnectionError("redis down")
        return True

    def get(self, key):
        if not self.available:
            raise ConnectionError("redis down")
        return self.values.get(key)

    def set(self, key, value, ex):
        if not self.available:
            raise ConnectionError("redis down")
        self.values[key] = value
        self.expirations.append((key, ex))

    def expire(self, key, seconds):
        if not self.available:
            raise ConnectionError("redis down")
        self.expirations.append((key, seconds))
        return key in self.values

    def delete(self, key):
        if not self.available:
            raise ConnectionError("redis down")
        self.values.pop(key, None)

    def close(self):
        self.closed = True


class FakeDurableStore:
    def __init__(self):
        self.sessions = {
            "session-a": {"id": "session-a", "rolling_summary": "Earlier preference: concise answers."},
        }
        self.turns = {
            "session-a": [
                {"turn_index": 1, "user_original": "first", "assistant_final": "one"},
                {"turn_index": 2, "user_original": "second", "assistant_final": "two"},
            ]
        }
        self.recent_calls = 0
        self.appended = []

    def get_session(self, session_id, *, owner):
        return self.sessions.get(session_id)

    def get_recent_turns(self, session_id, limit, *, owner):
        self.recent_calls += 1
        return self.turns.get(session_id, [])[-limit:]

    def append_turn(self, session_id, user, assistant, course_name=None, *, owner):
        self.appended.append((session_id, user, assistant, course_name, owner))
        self.turns.setdefault(session_id, []).append(
            {"turn_index": len(self.turns.get(session_id, [])) + 1, "user_original": user, "assistant_final": assistant}
        )
        return True

    def delete_session(self, session_id, *, owner):
        return self.sessions.pop(session_id, None) is not None

    @staticmethod
    def format_recent_turns(turns):
        return "\n".join(f"User: {turn['user_original']}" for turn in turns)


def test_cache_uses_memory_id_key_and_refreshes_ttl_on_read():
    client = FakeRedis()
    cache = RedisSessionMemoryCache(client, ttl_seconds=60)

    cache.put("session-a", rolling_summary="summary", turns=[{"user_original": "a"}])
    snapshot = cache.get("session-a")

    assert snapshot.rolling_summary == "summary"
    assert snapshot.turns == [{"user_original": "a"}]
    assert client.expirations == [
        ("agentic-rag:memory:session-a:snapshot", 60),
        ("agentic-rag:memory:session-a:snapshot", 60),
    ]
    assert "agentic-rag:memory:session-b:snapshot" not in client.values


def test_hot_store_hydrates_postgres_on_cache_miss_then_serves_cache():
    durable = FakeDurableStore()
    client = FakeRedis()
    store = HotSessionMemoryStore(durable, RedisSessionMemoryCache(client, ttl_seconds=60), recent_turns=2)
    owner = Principal("tenant-a", "user-a")

    context = store.get_memory_context("session-a", owner=owner)
    cached_context = store.get_memory_context("session-a", owner=owner)

    assert "Earlier preference" in context
    assert "User: first" in context
    assert cached_context == context
    assert durable.recent_calls == 1
    payload = json.loads(client.values["agentic-rag:memory:session-a:snapshot"])
    assert len(payload["turns"]) == 2


def test_append_refreshes_active_snapshot_from_durable_history():
    durable = FakeDurableStore()
    client = FakeRedis()
    store = HotSessionMemoryStore(durable, RedisSessionMemoryCache(client, ttl_seconds=60), recent_turns=2)
    owner = Principal("tenant-a", "user-a")

    assert store.append_turn("session-a", "third", "three", owner=owner)

    payload = json.loads(client.values["agentic-rag:memory:session-a:snapshot"])
    assert [turn["user_original"] for turn in payload["turns"]] == ["second", "third"]
    assert durable.appended[0][-1] == owner


def test_redis_unavailable_raises_instead_of_using_process_memory():
    client = FakeRedis()
    client.available = False
    cache = RedisSessionMemoryCache(client, ttl_seconds=60)

    with pytest.raises(MemoryBackendUnavailable):
        cache.ensure_available()
    with pytest.raises(MemoryBackendUnavailable):
        cache.get("session-a")


def test_delete_keeps_durable_delete_authoritative_when_redis_is_down():
    durable = FakeDurableStore()
    client = FakeRedis()
    store = HotSessionMemoryStore(durable, RedisSessionMemoryCache(client, ttl_seconds=60), recent_turns=2)
    client.available = False

    assert store.delete_session("session-a", owner=Principal("tenant-a", "user-a"))
    assert "session-a" not in durable.sessions
