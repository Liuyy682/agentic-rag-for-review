import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config


class SessionMemoryStore:
    def __init__(self, db_path: str | os.PathLike = config.SESSION_MEMORY_PATH):
        self.db_path = Path(db_path)
        self.ensure_schema()

    def ensure_schema(self):
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    user_original TEXT NOT NULL,
                    assistant_final TEXT NOT NULL,
                    course_name TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_turns_session_turn
                ON session_turns (session_id, turn_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_turns_session_created
                ON session_turns (session_id, created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    course_name TEXT NOT NULL DEFAULT '',
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_course
                ON sessions (course_name, updated_at DESC)
                """
            )
            # Migrate existing session_turns into sessions table
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (id, course_name, title, created_at, updated_at)
                SELECT
                    st.session_id,
                    COALESCE(MIN(st.course_name), ''),
                    NULL,
                    MIN(st.created_at),
                    MAX(st.created_at)
                FROM session_turns st
                WHERE st.session_id NOT IN (SELECT id FROM sessions)
                GROUP BY st.session_id
                """
            )

    # ── Session CRUD ────────────────────────────────────────────────────────

    def create_session(self, course_name: str = "", title: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, course_name, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, course_name or "", title, now, now),
            )
        return session_id

    def list_sessions(self, course_name: str = "") -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.id,
                    s.course_name,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(st.id) AS turn_count
                FROM sessions s
                LEFT JOIN session_turns st ON s.id = st.session_id
                WHERE s.course_name = ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """,
                (course_name or "",),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, course_name, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str):
        """Unconditionally set the session title (used by LLM title generation)."""
        if not session_id or not title:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), now, session_id),
            )

    def delete_session(self, session_id: str):
        if not session_id:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def rename_course_in_sessions(self, old_name: str, new_name: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET course_name = ? WHERE course_name = ?",
                (new_name, old_name),
            )
            conn.execute(
                "UPDATE session_turns SET course_name = ? WHERE course_name = ?",
                (new_name, old_name),
            )

    # ── Turn management ─────────────────────────────────────────────────────

    def append_turn(self, session_id: str, user_original: str, assistant_final: str, course_name: str | None = None):
        user_text = (user_original or "").strip()
        assistant_text = (assistant_final or "").strip()
        if not session_id or not user_text or not assistant_text:
            return

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM session_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_index = int(row[0])
            conn.execute(
                """
                INSERT INTO session_turns (
                    session_id,
                    turn_index,
                    user_original,
                    assistant_final,
                    course_name,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, turn_index, user_text, assistant_text, course_name, created_at),
            )
            # Ensure session row exists (handles legacy session_ids)
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, course_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, course_name or "", created_at, created_at),
            )
            # Auto-title from first message
            if turn_index == 1:
                title = user_text[:40] + ("..." if len(user_text) > 40 else "")
                conn.execute(
                    "UPDATE sessions SET title = COALESCE(title, ?), updated_at = ? WHERE id = ?",
                    (title, created_at, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (created_at, session_id),
                )

    def get_recent_turns(self, session_id: str, limit: int = 5) -> list[dict]:
        if not session_id or limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    turn_index,
                    user_original,
                    assistant_final,
                    course_name,
                    created_at
                FROM (
                    SELECT *
                    FROM session_turns
                    WHERE session_id = ?
                    ORDER BY turn_index DESC
                    LIMIT ?
                )
                ORDER BY turn_index ASC
                """,
                (session_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_session_turns(self, session_id: str) -> list[dict]:
        """Get all turns for a session (for loading history in UI)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def format_recent_turns(self, turns: Iterable[dict]) -> str:
        turns = list(turns)
        if not turns:
            return ""

        lines = [
            "Recent conversation memory from this session.",
            "Use it to understand references, user intent, and continuity.",
            "Do not treat prior assistant answers as knowledge-base evidence.",
        ]
        for idx, turn in enumerate(turns, start=1):
            lines.extend(
                [
                    "",
                    f"Turn {idx}",
                    f"User: {turn.get('user_original', '')}",
                    f"Assistant: {turn.get('assistant_final', '')}",
                ]
            )
        return "\n".join(lines)

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
