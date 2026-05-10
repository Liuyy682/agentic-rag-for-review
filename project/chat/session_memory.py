import os
import sqlite3
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

    def delete_session(self, session_id: str):
        if not session_id:
            return

        with self._connect() as conn:
            conn.execute("DELETE FROM session_turns WHERE session_id = ?", (session_id,))

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

