import os
import uuid

import pytest
import redis
from langgraph.checkpoint.base import empty_checkpoint

from agentic_rag.agent.redis_checkpoint import create_redis_checkpointer
from agentic_rag.chat.distributed_lock import RedisMemoryLockManager
from agentic_rag.chat.redis_memory import MemorySnapshot, RedisSessionMemoryCache


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REDIS_INTEGRATION_TESTS") != "1",
    reason="Set RUN_REDIS_INTEGRATION_TESTS=1 with Redis 8 to run Redis integration tests.",
)


@pytest.fixture
def redis_client():
    client = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    client.ping()
    try:
        yield client
    finally:
        client.close()


def test_session_memory_round_trip_refreshes_ttl_and_deletes(redis_client):
    session_id = f"ci-{uuid.uuid4()}"
    cache = RedisSessionMemoryCache(redis_client, ttl_seconds=60)
    key = cache._key(session_id)
    expected = MemorySnapshot(
        rolling_summary="此前讨论了 Redis。",
        turns=[{"role": "user", "content": "继续"}],
    )

    try:
        cache.put(
            session_id,
            rolling_summary=expected.rolling_summary,
            turns=expected.turns,
        )
        assert 0 < redis_client.ttl(key) <= 60

        redis_client.expire(key, 10)
        assert cache.get(session_id) == expected
        assert redis_client.ttl(key) > 10

        cache.delete(session_id)
        assert cache.get(session_id) is None
    finally:
        redis_client.delete(key)


def test_distributed_locks_are_scoped_renewable_and_owner_safe(redis_client):
    session_id = f"ci-{uuid.uuid4()}"
    other_session_id = f"ci-{uuid.uuid4()}"
    manager = RedisMemoryLockManager(
        redis_client,
        wait_seconds=0,
        lease_seconds=30,
        renew_interval_seconds=60,
    )
    lease = manager.acquire(session_id)
    other_lease = manager.acquire(other_session_id)

    try:
        assert lease is not None
        assert other_lease is not None
        assert manager.acquire(session_id) is None
        assert lease._renew_once()

        redis_client.set(lease.key, "another-owner", px=30_000)
        lease.release()
        assert redis_client.get(lease.key) == b"another-owner"
    finally:
        if other_lease is not None:
            other_lease.release()
        redis_client.delete(
            RedisMemoryLockManager._key(session_id),
            RedisMemoryLockManager._key(other_session_id),
        )


def test_shallow_checkpointer_round_trip_uses_redis_modules():
    thread_id = f"ci-{uuid.uuid4()}"
    saver = create_redis_checkpointer()
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    try:
        stored_config = saver.put(config, empty_checkpoint(), {}, {})
        restored = saver.get_tuple(stored_config)

        assert restored is not None
        assert restored.config["configurable"]["thread_id"] == thread_id

        saver.delete_thread(thread_id)
        assert saver.get_tuple(config) is None
    finally:
        saver.delete_thread(thread_id)
        close = getattr(saver, "close", None)
        if close:
            close()
        else:
            saver._redis.close()
