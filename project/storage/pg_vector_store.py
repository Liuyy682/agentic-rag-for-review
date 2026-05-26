from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from psycopg2.extras import Json, RealDictCursor

from retrieval.embeddings import DenseEmbeddingModel
from retrieval.fusion import reciprocal_rank_fusion
from storage.postgres import ensure_schema, transaction


_TSQUERY_PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~")

def _make_ts_query(text: str) -> str:
    import jieba

    tokens = [
        token.strip() for token in jieba.cut(text or "")
        if token.strip() and token.strip() not in _TSQUERY_PUNCTUATION
    ]
    if not tokens:
        return str(text or "").strip().replace(" ", " & ")
    return " & ".join(tokens)


def _make_tsvector_text(text: str) -> str:
    import jieba

    tokens = [token.strip() for token in jieba.cut(text or "") if token.strip()]
    return " ".join(tokens) if tokens else str(text or "")


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class PgVectorManager:
    """PostgreSQL/pgvector child chunk storage and retrieval."""

    def __init__(self):
        ensure_schema()
        self._dense_embeddings = DenseEmbeddingModel()

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        embeddings = self._dense_embeddings.embed_documents(texts)
        ids: List[str] = []

        with transaction() as conn:
            with conn.cursor() as cur:
                for index, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    metadata = dict(doc.metadata or {})
                    child_id = metadata.get("chunk_id") or metadata.get("child_id") or f"child_{index}"
                    parent_id = metadata.get("parent_id") or ""
                    doc_id = metadata.get("doc_id") or metadata.get("source_file") or metadata.get("source") or ""
                    source = metadata.get("source")
                    source_file = metadata.get("source_file")
                    page_numbers = metadata.get("page_numbers")
                    slide_title = metadata.get("slide_title")
                    cur.execute(
                        """
                        INSERT INTO child_chunks (
                            child_id, parent_id, doc_id, chunk_index, source, source_file,
                            page_numbers, slide_title, content, embedding, content_tsv, metadata
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s::vector, to_tsvector('simple', %s), %s
                        )
                        ON CONFLICT (child_id) DO UPDATE SET
                            parent_id = EXCLUDED.parent_id,
                            doc_id = EXCLUDED.doc_id,
                            chunk_index = EXCLUDED.chunk_index,
                            source = EXCLUDED.source,
                            source_file = EXCLUDED.source_file,
                            page_numbers = EXCLUDED.page_numbers,
                            slide_title = EXCLUDED.slide_title,
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            content_tsv = EXCLUDED.content_tsv,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            child_id,
                            parent_id,
                            doc_id,
                            metadata.get("chunk_index"),
                            source,
                            source_file,
                            page_numbers,
                            slide_title,
                            doc.page_content,
                            _vector_literal(embedding),
                            _make_tsvector_text(doc.page_content),
                            Json(metadata),
                        ),
                    )
                    ids.append(str(child_id))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> List[Document]:
        return self.dense_search(query, k=k)

    def clear_store(self) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM child_chunks")

    def delete_by_parent_ids(self, parent_ids: List[str]) -> None:
        if not parent_ids:
            return
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM child_chunks WHERE parent_id = ANY(%s)", (list(parent_ids),))

    def delete_by_source_file(self, source_file: str) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM child_chunks WHERE source_file = %s OR source = %s",
                    (source_file, source_file),
                )

    def dense_search(self, query: str, k: int) -> List[Document]:
        embedding = self._dense_embeddings.embed_query(query)
        vector = _vector_literal(embedding)
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT child_id, parent_id, doc_id, content, source, source_file,
                           page_numbers, slide_title, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM child_chunks
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, vector, k),
                )
                rows = cur.fetchall()
        return [_row_to_doc(row) for row in rows]

    def sparse_search(self, query: str, k: int) -> List[Document]:
        ts_query = _make_ts_query(query)
        if not ts_query:
            return []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT child_id, parent_id, doc_id, content, source, source_file,
                           page_numbers, slide_title, metadata,
                           ts_rank(content_tsv, to_tsquery('simple', %s)) AS score
                    FROM child_chunks
                    WHERE content_tsv @@ to_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (ts_query, ts_query, k),
                )
                rows = cur.fetchall()
        return [_row_to_doc(row) for row in rows]

    def load_child_neighbors(self, anchors: List[dict], window: int = 1) -> List[dict]:
        if not anchors:
            return []

        normalized = []
        for anchor in anchors:
            parent_id = str(anchor.get("parent_id") or "")
            try:
                chunk_index = int(anchor.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            if parent_id:
                normalized.append((parent_id, chunk_index))
        if not normalized:
            return []

        seen_child_ids = set()
        rows_by_key = []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for request_order, (parent_id, chunk_index) in enumerate(normalized):
                    radius = max(0, int(window))
                    cur.execute(
                        """
                        SELECT id, child_id, parent_id, doc_id, chunk_index, source,
                               source_file, content, metadata
                        FROM child_chunks
                        WHERE parent_id = %s
                          AND chunk_index >= %s
                          AND chunk_index <= %s
                        ORDER BY chunk_index ASC, id ASC
                        """,
                        (parent_id, chunk_index - radius, chunk_index + radius),
                    )
                    for row in cur.fetchall():
                        if row["child_id"] in seen_child_ids:
                            continue
                        seen_child_ids.add(row["child_id"])
                        metadata = dict(row.get("metadata") or {})
                        metadata.update(
                            {
                                "chunk_id": row["child_id"],
                                "parent_id": row["parent_id"],
                                "doc_id": row["doc_id"],
                                "chunk_index": row["chunk_index"],
                                "source": row["source"],
                                "source_file": row["source_file"],
                            }
                        )
                        rows_by_key.append(
                            (
                                request_order,
                                row["chunk_index"] if row["chunk_index"] is not None else 0,
                                {
                                    "content": row["content"],
                                    "parent_id": row["parent_id"],
                                    "metadata": metadata,
                                },
                            )
                        )

        return [item for _, _, item in sorted(rows_by_key, key=lambda row: (row[0], row[1]))]

    def rrf_search(
        self,
        query: str,
        dense_k: int,
        sparse_k: int,
        fused_k: int,
        rrf_k: int,
    ) -> List[Document]:
        dense_results = self.dense_search(query, dense_k)
        sparse_results = self.sparse_search(query, sparse_k)
        return reciprocal_rank_fusion(
            rankings=[dense_results, sparse_results],
            k=rrf_k,
            top_k=fused_k,
        )


def _row_to_doc(row) -> Document:
    metadata = dict(row.get("metadata") or {})
    metadata["chunk_id"] = row["child_id"]
    metadata["parent_id"] = row["parent_id"]
    metadata["doc_id"] = row["doc_id"]
    metadata["source"] = row["source"]
    metadata["source_file"] = row["source_file"]
    if row.get("page_numbers"):
        metadata["page_numbers"] = row["page_numbers"]
    if row.get("slide_title"):
        metadata["slide_title"] = row["slide_title"]
    score = row.get("score")
    if score is not None:
        metadata["score"] = float(score)
    return Document(page_content=row["content"], metadata=metadata)
