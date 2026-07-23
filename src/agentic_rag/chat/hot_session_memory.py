from __future__ import annotations

from agentic_rag.security.auth import Principal

from .redis_memory import MemoryBackendUnavailable, MemorySnapshot, RedisSessionMemoryCache


class HotSessionMemoryStore:
    """PostgreSQL-backed session memory with an expiring Redis active-context cache."""

    def __init__(self, durable_store, cache: RedisSessionMemoryCache, *, recent_turns: int):
        if recent_turns <= 0:
            raise ValueError("MEMORY_RECENT_TURNS must be positive")
        self._durable_store = durable_store
        self._cache = cache
        self._recent_turns = recent_turns

    def ensure_available(self) -> None:
        self._cache.ensure_available()

    def create_session(self, *args, **kwargs):
        return self._durable_store.create_session(*args, **kwargs)

    def list_sessions(self, *args, **kwargs):
        return self._durable_store.list_sessions(*args, **kwargs)

    def get_session(self, *args, **kwargs):
        return self._durable_store.get_session(*args, **kwargs)

    def update_session_title(self, *args, **kwargs) -> None:
        self._durable_store.update_session_title(*args, **kwargs)

    def rename_course_in_sessions(self, *args, **kwargs) -> None:
        self._durable_store.rename_course_in_sessions(*args, **kwargs)

    def delete_session(self, session_id: str, *, owner: Principal | None = None) -> bool:
        deleted = self._durable_store.delete_session(session_id, owner=owner)
        if deleted:
            try:
                self._cache.delete(session_id)
            except MemoryBackendUnavailable:
                # PostgreSQL deletion is authoritative. The inaccessible cache entry expires
                # under its configured TTL and cannot be read without the durable owner check.
                pass
        return deleted

    def append_turn(self, session_id: str, *args, owner: Principal | None = None, **kwargs) -> bool:
        appended = self._durable_store.append_turn(session_id, *args, owner=owner, **kwargs)
        if appended:
            self._refresh_snapshot(session_id, owner=owner)
        return appended

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
        *,
        owner: Principal | None = None,
    ) -> list[dict]:
        snapshot = self._load_snapshot(session_id, owner=owner)
        return snapshot.turns[-limit:] if limit > 0 else []

    def get_session_turns(self, *args, **kwargs):
        return self._durable_store.get_session_turns(*args, **kwargs)

    def get_memory_context(self, session_id: str, *, owner: Principal | None = None) -> str:
        snapshot = self._load_snapshot(session_id, owner=owner)
        recent = self._durable_store.format_recent_turns(snapshot.turns)
        if not snapshot.rolling_summary:
            return recent
        summary = f"Long-term conversation summary from this session.\n{snapshot.rolling_summary}"
        return f"{summary}\n\n{recent}" if recent else summary

    def format_recent_turns(self, turns) -> str:
        return self._durable_store.format_recent_turns(turns)

    def close(self) -> None:
        self._cache.close()

    def _load_snapshot(self, session_id: str, *, owner: Principal | None) -> MemorySnapshot:
        snapshot = self._cache.get(session_id)
        if snapshot is not None:
            return snapshot
        return self._hydrate_snapshot(session_id, owner=owner)

    def _hydrate_snapshot(self, session_id: str, *, owner: Principal | None) -> MemorySnapshot:
        session = self._durable_store.get_session(session_id, owner=owner)
        if session is None:
            return MemorySnapshot(rolling_summary="", turns=[])
        turns = self._durable_store.get_recent_turns(
            session_id,
            limit=self._recent_turns,
            owner=owner,
        )
        snapshot = MemorySnapshot(
            rolling_summary=session.get("rolling_summary") or "",
            turns=turns,
        )
        self._cache.put(
            session_id,
            rolling_summary=snapshot.rolling_summary,
            turns=snapshot.turns,
        )
        return snapshot

    def _refresh_snapshot(self, session_id: str, *, owner: Principal | None) -> None:
        self._hydrate_snapshot(session_id, owner=owner)
