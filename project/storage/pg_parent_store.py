import re
from typing import Dict, List

from project.database.engine import SessionLocal
from project.database.models.chunk import ParentChunk, ChildChunk


class PgParentStoreManager:
    """PG replacement for ParentStoreManager. Same public API, reads/writes parent_chunks table."""

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        session = SessionLocal()
        try:
            existing = session.query(ParentChunk).filter(ParentChunk.parent_id == parent_id).first()
            if existing:
                existing.page_content = content
                existing.metadata_ = metadata
            else:
                doc_id = metadata.get("doc_id") or metadata.get("source_file", "")
                session.add(ParentChunk(parent_id=parent_id, doc_id=doc_id, page_content=content, metadata_=metadata))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_many(self, parents: List) -> None:
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def load(self, parent_id: str) -> Dict:
        session = SessionLocal()
        try:
            row = session.query(ParentChunk).filter(ParentChunk.parent_id == parent_id).first()
            if not row:
                return {}
            return {"page_content": row.page_content, "metadata": row.metadata_ or {}}
        finally:
            session.close()

    def load_content(self, parent_id: str) -> Dict:
        data = self.load(parent_id)
        if not data:
            return {}
        return {"content": data["page_content"], "parent_id": parent_id, "metadata": data["metadata"]}

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique_ids = list(dict.fromkeys(parent_ids))
        if not unique_ids:
            return []
        session = SessionLocal()
        try:
            rows = session.query(ParentChunk).filter(ParentChunk.parent_id.in_(unique_ids)).all()
        finally:
            session.close()
        key_fn = self._get_sort_key
        return sorted(
            (
                {"content": row.page_content, "parent_id": row.parent_id, "metadata": row.metadata_ or {}}
                for row in rows
            ),
            key=lambda d: key_fn(d["parent_id"]),
        )

    def delete_many(self, parent_ids: List[str]) -> None:
        if not parent_ids:
            return
        session = SessionLocal()
        try:
            session.query(ChildChunk).filter(ChildChunk.parent_id.in_(parent_ids)).delete(synchronize_session=False)
            session.query(ParentChunk).filter(ParentChunk.parent_id.in_(parent_ids)).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_by_source_file(self, source_file: str) -> None:
        session = SessionLocal()
        try:
            session.query(ParentChunk).filter(ParentChunk.doc_id == source_file).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def clear_store(self) -> None:
        session = SessionLocal()
        try:
            session.query(ParentChunk).delete()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _get_sort_key(id_str):
        match = re.search(r"_parent_(\d+)$", id_str)
        return int(match.group(1)) if match else 0
