from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .redis_memory import MemoryBackendUnavailable


_RENEW_IF_OWNER = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_IF_OWNER = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisMemoryLockManager:
    """A token-owned, renewable mutex scoped to one opaque MemoryId."""

    def __init__(
        self,
        client: Any,
        *,
        wait_seconds: float,
        lease_seconds: float,
        renew_interval_seconds: float,
    ):
        self._client = client
        self._wait_seconds = wait_seconds
        self._lease_ms = max(1, int(lease_seconds * 1000))
        self._renew_interval_seconds = renew_interval_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agentic-rag:memory:{session_id}:lock"

    def acquire(self, session_id: str) -> "MemoryLockLease | None":
        key = self._key(session_id)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + self._wait_seconds
        while True:
            try:
                acquired = self._client.set(key, token, nx=True, px=self._lease_ms)
            except Exception as exc:
                raise MemoryBackendUnavailable("Redis distributed lock is unavailable") from exc
            if acquired:
                lease = MemoryLockLease(
                    client=self._client,
                    key=key,
                    token=token,
                    lease_ms=self._lease_ms,
                    renew_interval_seconds=self._renew_interval_seconds,
                )
                lease.start_renewal()
                return lease
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


@dataclass
class MemoryLockLease:
    client: Any
    key: str
    token: str
    lease_ms: int
    renew_interval_seconds: float

    def __post_init__(self) -> None:
        self._released = threading.Event()
        self._renew_thread: threading.Thread | None = None

    def start_renewal(self) -> None:
        self._renew_thread = threading.Thread(target=self._renew_loop, daemon=True)
        self._renew_thread.start()

    def _renew_loop(self) -> None:
        while not self._released.wait(self.renew_interval_seconds):
            try:
                if not self._renew_once():
                    return
            except Exception:
                # The TTL still bounds lock lifetime if Redis becomes unavailable.
                return

    def _renew_once(self) -> bool:
        return bool(self.client.eval(_RENEW_IF_OWNER, 1, self.key, self.token, self.lease_ms))

    def release(self) -> None:
        if self._released.is_set():
            return
        self._released.set()
        try:
            self.client.eval(_RELEASE_IF_OWNER, 1, self.key, self.token)
        except Exception:
            # The token and TTL prevent another owner from being deleted later.
            pass
