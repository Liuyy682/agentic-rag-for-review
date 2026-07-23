from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentic_rag import config


class MemoryBackendUnavailable(RuntimeError):
    """Raised when Redis hot memory cannot safely serve a chat request."""


@dataclass(frozen=True)
class MemorySnapshot:
    rolling_summary: str
    turns: list[dict]


class RedisSessionMemoryCache:
    """Redis cache for the active context of one opaque MemoryId."""

    def __init__(self, client: Any, *, ttl_seconds: int = config.MEMORY_TTL_SECONDS):
        if ttl_seconds <= 0:
            raise ValueError("MEMORY_TTL_SECONDS must be positive")
        self._client = client
        self._ttl_seconds = ttl_seconds

    @classmethod
    def from_config(cls) -> "RedisSessionMemoryCache":
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("Install the 'redis' package to enable Redis session memory") from exc

        client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
            socket_timeout=config.REDIS_SOCKET_TIMEOUT_SECONDS,
            health_check_interval=30,
        )
        return cls(client)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agentic-rag:memory:{session_id}:snapshot"

    def ensure_available(self) -> None:
        try:
            if not self._client.ping():
                raise MemoryBackendUnavailable("Redis ping returned false")
        except MemoryBackendUnavailable:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailable("Redis hot memory is unavailable") from exc

    def get(self, session_id: str) -> MemorySnapshot | None:
        key = self._key(session_id)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            payload = json.loads(raw)
            turns = payload.get("turns")
            if not isinstance(turns, list):
                return None
            self._client.expire(key, self._ttl_seconds)
            return MemorySnapshot(
                rolling_summary=str(payload.get("rolling_summary") or ""),
                turns=[turn for turn in turns if isinstance(turn, dict)],
            )
        except Exception as exc:
            raise MemoryBackendUnavailable("Redis hot memory read failed") from exc

    def put(self, session_id: str, *, rolling_summary: str, turns: list[dict]) -> None:
        payload = json.dumps(
            {"rolling_summary": rolling_summary or "", "turns": turns},
            ensure_ascii=False,
            default=str,
        )
        try:
            self._client.set(self._key(session_id), payload, ex=self._ttl_seconds)
        except Exception as exc:
            raise MemoryBackendUnavailable("Redis hot memory write failed") from exc

    def delete(self, session_id: str) -> None:
        try:
            self._client.delete(self._key(session_id))
        except Exception as exc:
            raise MemoryBackendUnavailable("Redis hot memory delete failed") from exc

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()
