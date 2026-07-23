from __future__ import annotations

from typing import Callable

from agentic_rag import config
from agentic_rag.security.auth import Principal

from .redis_memory import MemoryBackendUnavailable, MemorySnapshot, RedisSessionMemoryCache


class HotSessionMemoryStore:
    """PostgreSQL-backed session memory with an expiring Redis active-context cache."""

    def __init__(self, durable_store, cache: RedisSessionMemoryCache, *, recent_turns: int, lock_manager=None):
        if recent_turns <= 0:
            raise ValueError("MEMORY_RECENT_TURNS must be positive")
        self._durable_store = durable_store
        self._cache = cache
        self._recent_turns = recent_turns
        self._lock_manager = lock_manager

    def ensure_available(self) -> None:
        self._cache.ensure_available()

    def acquire_lock(self, session_id: str):
        if self._lock_manager is None:
            return None
        return self._lock_manager.acquire(session_id)

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

    def compact_context(
        self,
        session_id: str,
        summarizer: Callable[[str, list[dict]], str],
        *,
        token_budget: int,
        min_recent_turns: int,
        owner: Principal | None = None,
    ) -> bool:
        """Summarize only old, not-yet-summarized turns after archival succeeds."""
        session = self._durable_store.get_session(session_id, owner=owner)
        if session is None:
            return False
        turns = self._durable_store.get_session_turns(session_id, owner=owner)
        through = int(session.get("summarized_through_turn") or 0)
        pending = [turn for turn in turns if int(turn.get("turn_index") or 0) > through]
        summary = session.get("rolling_summary") or ""
        if self._estimate_tokens(summary) + self._estimate_turns(pending) <= token_budget:
            self._refresh_snapshot(session_id, owner=owner)
            return False
        candidates = pending[:-min_recent_turns] if len(pending) > min_recent_turns else []
        if not candidates:
            self._refresh_snapshot(session_id, owner=owner)
            return False
        new_summary = summarizer(summary, candidates)
        if not new_summary or not new_summary.strip():
            raise RuntimeError("Conversation summarizer returned an empty summary")
        summarized_through = int(candidates[-1]["turn_index"])
        updated = self._durable_store.update_rolling_summary(
            session_id,
            new_summary,
            summarized_through,
            owner=owner,
        )
        if updated:
            self._refresh_snapshot(session_id, owner=owner)
        return updated

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
        turns = self._context_turns(session_id, session, owner=owner)
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
        session = self._durable_store.get_session(session_id, owner=owner)
        if session is None:
            return
        turns = self._context_turns(session_id, session, owner=owner)
        self._cache.put(
            session_id,
            rolling_summary=session.get("rolling_summary") or "",
            turns=turns,
        )

    def _context_turns(self, session_id: str, session: dict, *, owner: Principal | None) -> list[dict]:
        turns = self._durable_store.get_session_turns(session_id, owner=owner)
        through = int(session.get("summarized_through_turn") or 0)
        pending = [turn for turn in turns if int(turn.get("turn_index") or 0) > through]
        summary_tokens = self._estimate_tokens(session.get("rolling_summary") or "")
        selected: list[dict] = []
        used = summary_tokens
        for turn in reversed(pending):
            turn_tokens = self._estimate_turns([turn])
            if selected and len(selected) >= self._recent_turns and used + turn_tokens > config.MEMORY_CONTEXT_TOKEN_BUDGET:
                break
            selected.append(turn)
            used += turn_tokens
        return list(reversed(selected))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # A deliberately conservative, dependency-free approximation for budget governance.
        return (len(text or "") + 3) // 4

    @classmethod
    def _estimate_turns(cls, turns: list[dict]) -> int:
        return sum(
            cls._estimate_tokens(str(turn.get("user_original") or ""))
            + cls._estimate_tokens(str(turn.get("assistant_final") or ""))
            + 8
            for turn in turns
        )
