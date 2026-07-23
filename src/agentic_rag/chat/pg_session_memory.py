from __future__ import annotations

import uuid
from typing import Iterable

from psycopg2.extras import RealDictCursor

from agentic_rag.security.auth import Principal
from agentic_rag.storage.postgres import ensure_schema, transaction


def _memory_uuid(session_id: str) -> str | None:
    try:
        return str(uuid.UUID(str(session_id)))
    except (TypeError, ValueError, AttributeError):
        return None


class PgSessionMemoryStore:
    """Tenant-scoped durable chat history stored in PostgreSQL."""

    def __init__(self):
        ensure_schema()

    @staticmethod
    def _owner(owner: Principal | None) -> Principal:
        if owner is None:
            raise ValueError("A validated memory owner is required")
        return owner

    def create_session(
        self,
        course_name: str = "",
        title: str | None = None,
        *,
        owner: Principal | None = None,
    ) -> str:
        owner = self._owner(owner)
        memory_id = str(uuid.uuid4())
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (memory_id, tenant_id, user_id, course_name, title)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (memory_id, owner.tenant_id, owner.user_id, course_name or "", title),
                )
        return memory_id

    def list_sessions(self, course_name: str = "", *, owner: Principal | None = None) -> list[dict]:
        owner = self._owner(owner)
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        s.memory_id AS id,
                        s.course_name,
                        s.title,
                        s.created_at,
                        s.updated_at,
                        COUNT(t.id) AS turn_count
                    FROM chat_sessions s
                    LEFT JOIN chat_turns t ON t.memory_id = s.memory_id
                    WHERE s.tenant_id = %s AND s.user_id = %s AND s.course_name = %s
                    GROUP BY s.memory_id
                    ORDER BY s.updated_at DESC
                    """,
                    (owner.tenant_id, owner.user_id, course_name or ""),
                )
                rows = cur.fetchall()
        return [self._serialize_session(row) for row in rows]

    def get_session(self, session_id: str, *, owner: Principal | None = None) -> dict | None:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        if memory_id is None:
            return None
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT memory_id AS id, course_name, title, rolling_summary,
                           summarized_through_turn, created_at, updated_at
                    FROM chat_sessions
                    WHERE memory_id = %s AND tenant_id = %s AND user_id = %s
                    """,
                    (memory_id, owner.tenant_id, owner.user_id),
                )
                row = cur.fetchone()
        return self._serialize_session(row) if row else None

    def update_session_title(
        self,
        session_id: str,
        title: str,
        *,
        owner: Principal | None = None,
    ) -> None:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        if memory_id is None or not title:
            return
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions SET title = %s, updated_at = now()
                    WHERE memory_id = %s AND tenant_id = %s AND user_id = %s
                    """,
                    (title.strip(), memory_id, owner.tenant_id, owner.user_id),
                )

    def delete_session(self, session_id: str, *, owner: Principal | None = None) -> bool:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        if memory_id is None:
            return False
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE memory_id = %s AND tenant_id = %s AND user_id = %s
                    """,
                    (memory_id, owner.tenant_id, owner.user_id),
                )
                return cur.rowcount == 1

    def rename_course_in_sessions(self, old_name: str, new_name: str) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_sessions SET course_name = %s, updated_at = now() WHERE course_name = %s",
                    (new_name, old_name),
                )

    def append_turn(
        self,
        session_id: str,
        user_original: str,
        assistant_final: str,
        course_name: str | None = None,
        *,
        owner: Principal | None = None,
    ) -> bool:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        user_text = (user_original or "").strip()
        assistant_text = (assistant_final or "").strip()
        if memory_id is None or not user_text or not assistant_text:
            return False

        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT title FROM chat_sessions
                    WHERE memory_id = %s AND tenant_id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (memory_id, owner.tenant_id, owner.user_id),
                )
                session = cur.fetchone()
                if session is None:
                    return False
                cur.execute(
                    "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM chat_turns WHERE memory_id = %s",
                    (memory_id,),
                )
                turn_index = int(cur.fetchone()[0])
                cur.execute(
                    """
                    INSERT INTO chat_turns (
                        memory_id, turn_index, user_original, assistant_final
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (memory_id, turn_index, user_text, assistant_text),
                )
                title = user_text[:40] + ("..." if len(user_text) > 40 else "")
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET title = COALESCE(title, %s),
                        course_name = COALESCE(%s, course_name),
                        updated_at = now()
                    WHERE memory_id = %s
                    """,
                    (title, course_name, memory_id),
                )
        return True

    def get_recent_turns(
        self,
        session_id: str,
        limit: int = 5,
        *,
        owner: Principal | None = None,
    ) -> list[dict]:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        if memory_id is None or limit <= 0:
            return []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT t.id, t.memory_id AS session_id, t.turn_index,
                           t.user_original, t.assistant_final, s.course_name, t.created_at
                    FROM (
                        SELECT * FROM chat_turns
                        WHERE memory_id = %s
                        ORDER BY turn_index DESC
                        LIMIT %s
                    ) t
                    JOIN chat_sessions s ON s.memory_id = t.memory_id
                    WHERE s.tenant_id = %s AND s.user_id = %s
                    ORDER BY t.turn_index ASC
                    """,
                    (memory_id, limit, owner.tenant_id, owner.user_id),
                )
                rows = cur.fetchall()
        return [self._serialize_turn(row) for row in rows]

    def get_session_turns(self, session_id: str, *, owner: Principal | None = None) -> list[dict]:
        owner = self._owner(owner)
        memory_id = _memory_uuid(session_id)
        if memory_id is None:
            return []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT t.id, t.memory_id AS session_id, t.turn_index,
                           t.user_original, t.assistant_final, s.course_name, t.created_at
                    FROM chat_turns t
                    JOIN chat_sessions s ON s.memory_id = t.memory_id
                    WHERE t.memory_id = %s AND s.tenant_id = %s AND s.user_id = %s
                    ORDER BY t.turn_index ASC
                    """,
                    (memory_id, owner.tenant_id, owner.user_id),
                )
                rows = cur.fetchall()
        return [self._serialize_turn(row) for row in rows]

    @staticmethod
    def format_recent_turns(turns: Iterable[dict]) -> str:
        turns = list(turns)
        if not turns:
            return ""
        lines = [
            "Recent conversation memory from this session.",
            "Use it to understand references, user intent, and continuity.",
            "Do not treat prior assistant answers as knowledge-base evidence.",
        ]
        for index, turn in enumerate(turns, start=1):
            lines.extend(
                [
                    "",
                    f"Turn {index}",
                    f"User: {turn.get('user_original', '')}",
                    f"Assistant: {turn.get('assistant_final', '')}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _serialize_session(row) -> dict:
        result = dict(row)
        result["id"] = str(result["id"])
        return result

    @staticmethod
    def _serialize_turn(row) -> dict:
        result = dict(row)
        result["session_id"] = str(result["session_id"])
        return result
