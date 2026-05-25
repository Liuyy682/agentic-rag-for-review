from __future__ import annotations

import re
from typing import Dict, List

from psycopg2.extras import Json, RealDictCursor

from storage.postgres import ensure_schema, transaction


class PgParentStoreManager:
    """PostgreSQL parent chunk storage."""

    def __init__(self):
        ensure_schema()

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        metadata = dict(metadata or {})
        doc_id = metadata.get("doc_id") or metadata.get("source_file") or metadata.get("source") or ""
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO parent_chunks (parent_id, doc_id, page_content, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (parent_id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id,
                        page_content = EXCLUDED.page_content,
                        metadata = EXCLUDED.metadata
                    """,
                    (parent_id, doc_id, content, Json(metadata)),
                )

    def save_many(self, parents: List) -> None:
        if not parents:
            return
        with transaction() as conn:
            with conn.cursor() as cur:
                for parent_id, doc in parents:
                    metadata = dict(doc.metadata or {})
                    doc_id = metadata.get("doc_id") or metadata.get("source_file") or metadata.get("source") or ""
                    cur.execute(
                        """
                        INSERT INTO parent_chunks (parent_id, doc_id, page_content, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (parent_id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            page_content = EXCLUDED.page_content,
                            metadata = EXCLUDED.metadata
                        """,
                        (parent_id, doc_id, doc.page_content, Json(metadata)),
                    )

    def load(self, parent_id: str) -> Dict:
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT page_content, metadata FROM parent_chunks WHERE parent_id = %s",
                    (parent_id,),
                )
                row = cur.fetchone()
        if not row:
            return {}
        return {"page_content": row["page_content"], "metadata": row["metadata"] or {}}

    def load_content(self, parent_id: str) -> Dict:
        data = self.load(parent_id)
        if not data:
            return {}
        return {"content": data["page_content"], "parent_id": parent_id, "metadata": data["metadata"]}

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique_ids = list(dict.fromkeys(parent_ids))
        if not unique_ids:
            return []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT parent_id, page_content, metadata
                    FROM parent_chunks
                    WHERE parent_id = ANY(%s)
                    """,
                    (unique_ids,),
                )
                rows = cur.fetchall()

        key_fn = self._get_sort_key
        return sorted(
            (
                {"content": row["page_content"], "parent_id": row["parent_id"], "metadata": row["metadata"] or {}}
                for row in rows
            ),
            key=lambda item: key_fn(item["parent_id"]),
        )

    def delete_many(self, parent_ids: List[str]) -> None:
        if not parent_ids:
            return
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM parent_chunks WHERE parent_id = ANY(%s)", (list(parent_ids),))

    def delete_by_source_file(self, source_file: str) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM parent_chunks WHERE doc_id = %s OR metadata->>'source_file' = %s OR metadata->>'source' = %s",
                    (source_file, source_file, source_file),
                )

    def clear_store(self) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM parent_chunks")

    @staticmethod
    def _get_sort_key(id_str):
        match = re.search(r"_parent_(\d+)$", id_str)
        return int(match.group(1)) if match else 0
