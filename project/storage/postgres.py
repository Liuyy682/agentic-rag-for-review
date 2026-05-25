from __future__ import annotations

from contextlib import contextmanager
import re
from threading import Lock
from typing import Iterator

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

import config


class SchemaMismatchError(RuntimeError):
    """Raised when an existing pgvector schema cannot support the configured model."""


_POOL: ThreadedConnectionPool | None = None
_POOL_LOCK = Lock()
_SCHEMA_READY = False
_SCHEMA_LOCK = Lock()


def _pool() -> ThreadedConnectionPool:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=config.DATABASE_URL,
                )
    return _POOL


@contextmanager
def connection():
    conn = _pool().getconn()
    try:
        yield conn
    finally:
        _pool().putconn(conn)


@contextmanager
def transaction():
    with connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def ensure_schema() -> None:
    """Create the minimal runtime schema used by the RAG storage layer."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS parent_chunks (
                        id SERIAL PRIMARY KEY,
                        parent_id VARCHAR(255) UNIQUE NOT NULL,
                        doc_id VARCHAR(255) NOT NULL,
                        page_content TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS child_chunks (
                        id SERIAL PRIMARY KEY,
                        child_id VARCHAR(255) UNIQUE NOT NULL,
                        parent_id VARCHAR(255) NOT NULL,
                        doc_id VARCHAR(255) NOT NULL,
                        chunk_index INTEGER,
                        source VARCHAR(500),
                        source_file VARCHAR(500),
                        page_numbers INTEGER[],
                        slide_title VARCHAR(500),
                        content TEXT NOT NULL,
                        embedding vector({int(config.DENSE_EMBEDDING_DIMENSION)}),
                        content_tsv TSVECTOR,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT now()
                    )
                    """
                )
                _validate_embedding_dimension(cur)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc ON parent_chunks (doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_child_chunks_parent ON child_chunks (parent_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_child_chunks_doc ON child_chunks (doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_child_chunks_tsv ON child_chunks USING GIN (content_tsv)")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding
                    ON child_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200)
                    """
                )
        _SCHEMA_READY = True


def _validate_embedding_dimension(cur) -> None:
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        WHERE attrelid = 'child_chunks'::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
        """
    )
    row = cur.fetchone()
    if not row:
        return
    type_name = str(row[0] or "")
    match = re.search(r"vector\((\d+)\)", type_name)
    if not match:
        return
    existing_dim = int(match.group(1))
    expected_dim = int(config.DENSE_EMBEDDING_DIMENSION)
    if existing_dim != expected_dim:
        raise SchemaMismatchError(
            "child_chunks.embedding dimension is "
            f"{existing_dim}, but DENSE_EMBEDDING_DIMENSION is {expected_dim}. "
            "Rebuild the PostgreSQL data volume and re-index documents."
        )


def reset_pool_for_tests() -> None:
    global _POOL, _SCHEMA_READY
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.closeall()
            _POOL = None
    _SCHEMA_READY = False
