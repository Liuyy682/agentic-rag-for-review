from __future__ import annotations

import math
from typing import Any

from agentic_rag import config


def create_redis_checkpointer() -> Any:
    """Create the process-wide shallow Redis checkpointer for LangGraph.

    Checkpoints are hot state only: PostgreSQL remains the durable transcript
    source.  The saver must be backed by RedisJSON and RediSearch (Redis 8 in
    the development compose service) and deliberately fails startup when those
    capabilities are missing.
    """
    try:
        from langgraph.checkpoint.redis.shallow import ShallowRedisSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-redis to enable Redis LangGraph checkpoints"
        ) from exc

    ttl_minutes = max(1, math.ceil(config.MEMORY_TTL_SECONDS / 60))
    saver = ShallowRedisSaver(
        redis_url=config.REDIS_URL,
        connection_args={
            "socket_connect_timeout": config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            "socket_timeout": config.REDIS_SOCKET_TIMEOUT_SECONDS,
        },
        ttl={"default_ttl": ttl_minutes, "refresh_on_read": True},
    )
    saver.setup()
    return saver
