from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import func, text

import config
from database.engine import SessionLocal
from database.models.chunk import ParentChunk, ChildChunk
from retrieval.fusion import reciprocal_rank_fusion


def _make_ts_query(text: str) -> str:
    import jieba

    tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
    if not tokens:
        return text.strip().replace(" ", " & ")
    return " & ".join(tokens)


class _PgCollection:
    """Adapter that mimics Langchain QdrantVectorStore for add_documents and similarity_search."""

    def __init__(self, manager: "PgVectorManager", collection_name: str):
        self._manager = manager
        self._collection_name = collection_name

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed documents and insert into child_chunks table."""
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        embeddings = self._manager._dense_embeddings.embed_documents(texts)

        session = SessionLocal()
        ids = []
        try:
            for doc, emb in zip(documents, embeddings):
                meta = doc.metadata or {}
                child = ChildChunk(
                    child_id=meta.get("chunk_id", ""),
                    parent_id=meta.get("parent_id", ""),
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=meta.get("chunk_index"),
                    source=meta.get("source"),
                    source_file=meta.get("source_file"),
                    page_numbers=meta.get("page_numbers"),
                    slide_title=meta.get("slide_title"),
                    content=doc.page_content,
                    embedding=emb,
                    content_tsv=func.to_tsvector("simple", _make_ts_query(doc.page_content)),
                    metadata_=meta,
                )
                session.add(child)
                ids.append(child.child_id)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return ids

    def similarity_search(self, query: str, k: int = 4, score_threshold: float | None = None) -> List[Document]:
        """Dense-only similarity search. Used by qdrant_hybrid mode."""
        return self._manager.dense_search(self._collection_name, query, k)


class PgVectorManager:
    """Pgvector replacement for VectorDbManager. Same public API."""

    def __init__(self):
        import os

        self._dense_embeddings = HuggingFaceEmbeddings(
            model_name=config.DENSE_MODEL,
            cache_folder=getattr(config, "HF_CACHE_DIR", None),
            model_kwargs={
                "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
            },
        )

    # ── lifecycle ──────────────────────────────────────────────────

    def create_collection(self, collection_name: str) -> None:
        pass  # tables already exist via migration

    def delete_collection(self, collection_name: str) -> None:
        session = SessionLocal()
        try:
            session.query(ChildChunk).delete()
            session.query(ParentChunk).delete()
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── delete ──────────────────────────────────────────────────────

    def delete_by_parent_ids(self, collection_name: str, parent_ids: List[str]) -> None:
        if not parent_ids:
            return
        session = SessionLocal()
        try:
            session.query(ChildChunk).filter(ChildChunk.parent_id.in_(parent_ids)).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_by_source_file(self, collection_name: str, source_file: str) -> None:
        session = SessionLocal()
        try:
            session.query(ChildChunk).filter(ChildChunk.source_file == source_file).delete(synchronize_session=False)
            session.query(ParentChunk).filter(ParentChunk.doc_id == source_file).delete(synchronize_session=False)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── collection adapter ──────────────────────────────────────────

    def get_collection(self, collection_name, retrieval_mode=None) -> _PgCollection:
        return _PgCollection(self, collection_name)

    # ── search ──────────────────────────────────────────────────────

    def dense_search(self, collection_name: str, query: str, k: int) -> List[Document]:
        embedding = self._dense_embeddings.embed_query(query)
        vec_str = "'[" + ",".join(str(v) for v in embedding) + "]'"
        session = SessionLocal()
        try:
            result = session.execute(
                text(
                    "SELECT child_id, parent_id, doc_id, content, source, source_file, "
                    "page_numbers, slide_title, metadata, "
                    "1 - (embedding <=> " + vec_str + "::vector) AS score "
                    "FROM child_chunks "
                    "ORDER BY embedding <=> " + vec_str + "::vector "
                    "LIMIT :k"
                ),
                {"k": k},
            )
            rows = result.fetchall()
            return [_text_row_to_doc(row) for row in rows]
        finally:
            session.close()

    def sparse_search(self, collection_name: str, query: str, k: int) -> List[Document]:
        ts_query = _make_ts_query(query)
        session = SessionLocal()
        try:
            result = session.execute(
                text(
                    "SELECT child_id, parent_id, doc_id, content, source, source_file, "
                    "page_numbers, slide_title, metadata, "
                    "ts_rank(content_tsv, to_tsquery('simple', :tsq)) AS score "
                    "FROM child_chunks "
                    "WHERE content_tsv @@ to_tsquery('simple', :tsq) "
                    "ORDER BY score DESC "
                    "LIMIT :k"
                ),
                {"tsq": ts_query, "k": k},
            )
            rows = result.fetchall()
            return [_text_row_to_doc(row) for row in rows]
        finally:
            session.close()

    def rrf_search(
        self,
        collection_name: str,
        query: str,
        dense_k: int,
        sparse_k: int,
        fused_k: int,
        rrf_k: int,
    ) -> List[Document]:
        dense_results = self.dense_search(collection_name, query, dense_k)
        sparse_results = self.sparse_search(collection_name, query, sparse_k)
        return reciprocal_rank_fusion(
            rankings=[dense_results, sparse_results],
            k=rrf_k,
            top_k=fused_k,
        )


def _row_to_doc(row) -> Document:
    child_id, parent_id, doc_id, content, source, source_file, page_numbers, slide_title, metadata_, score = row
    meta = dict(metadata_ or {})
    meta.setdefault("chunk_id", child_id)
    meta.setdefault("parent_id", parent_id)
    meta.setdefault("doc_id", doc_id)
    meta.setdefault("source", source)
    meta.setdefault("source_file", source_file)
    if page_numbers is not None:
        meta.setdefault("page_numbers", page_numbers)
    if slide_title:
        meta.setdefault("slide_title", slide_title)
    if score is not None:
        meta["score"] = float(score)
    return Document(page_content=content, metadata=meta)


def _text_row_to_doc(row) -> Document:
    """Convert a text() query row to a Langchain Document."""
    data = row._mapping
    meta = dict(data.get("metadata") or {})
    meta["chunk_id"] = data["child_id"]
    meta["parent_id"] = data["parent_id"]
    meta["doc_id"] = data["doc_id"]
    meta["source"] = data["source"]
    meta["source_file"] = data["source_file"]
    if data.get("page_numbers"):
        meta["page_numbers"] = data["page_numbers"]
    if data.get("slide_title"):
        meta["slide_title"] = data["slide_title"]
    score = data.get("score")
    if score is not None:
        meta["score"] = float(score)
    return Document(page_content=data["content"], metadata=meta)
