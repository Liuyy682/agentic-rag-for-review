import os
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PG_STORAGE_TESTS") != "1",
    reason="Set RUN_PG_STORAGE_TESTS=1 with a local pgvector PostgreSQL database to run storage integration tests.",
)

from langchain_core.documents import Document

import config
from storage import postgres
from storage.pg_parent_store import PgParentStoreManager
from storage.pg_vector_store import PgVectorManager


def _vec(index: int) -> list[float]:
    values = [0.0] * config.DENSE_EMBEDDING_DIMENSION
    values[index] = 1.0
    return values


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        return [_vec(0 if "alpha" in text else 1) for text in texts]

    def embed_query(self, query):
        return _vec(0 if "alpha" in query else 1)


@pytest.fixture(autouse=True)
def pg_storage(monkeypatch):
    monkeypatch.setattr("storage.pg_vector_store.DenseEmbeddingModel", FakeEmbeddingModel)
    postgres.reset_pool_for_tests()
    vector = PgVectorManager()
    parent = PgParentStoreManager()
    vector.clear_store()
    parent.clear_store()
    yield vector, parent
    vector.clear_store()
    parent.clear_store()
    postgres.reset_pool_for_tests()


def test_ensure_schema_creates_runtime_tables_and_indexes(pg_storage):
    with postgres.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public'
                """
            )
            tables = {row[0] for row in cur.fetchall()}
            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
            indexes = {row[0] for row in cur.fetchall()}

    assert {"parent_chunks", "child_chunks"} <= tables
    assert {"idx_child_chunks_embedding", "idx_child_chunks_tsv", "idx_child_chunks_parent"} <= indexes


def test_parent_store_crud(pg_storage):
    _, parent = pg_storage

    parent.save("parent_1", "first", {"source_file": "doc.md", "doc_id": "doc"})
    assert parent.load_content("parent_1")["content"] == "first"

    parent.save("parent_1", "updated", {"source_file": "doc.md", "doc_id": "doc"})
    assert parent.load("parent_1")["page_content"] == "updated"

    parent.save_many([
        ("doc_page_1_parent_2", Document(page_content="two", metadata={"source_file": "doc.md"})),
        ("doc_page_1_parent_1", Document(page_content="one", metadata={"source_file": "doc.md"})),
    ])
    assert [row["parent_id"] for row in parent.load_content_many(["doc_page_1_parent_2", "doc_page_1_parent_1"])] == [
        "doc_page_1_parent_1",
        "doc_page_1_parent_2",
    ]

    parent.delete_many(["doc_page_1_parent_1"])
    assert parent.load_content("doc_page_1_parent_1") == {}

    parent.clear_store()
    assert parent.load_content("parent_1") == {}


def test_child_store_search_and_delete(pg_storage):
    vector, _ = pg_storage
    docs = [
        Document(
            page_content="alpha 机器学习 evidence",
            metadata={
                "chunk_id": "child_alpha",
                "parent_id": "parent_1",
                "doc_id": "doc",
                "chunk_index": 1,
                "source": "doc.md",
                "source_file": "doc.md",
                "page_numbers": [1],
            },
        ),
        Document(
            page_content="beta database evidence",
            metadata={
                "chunk_id": "child_beta",
                "parent_id": "parent_1",
                "doc_id": "doc",
                "chunk_index": 2,
                "source": "doc.md",
                "source_file": "doc.md",
                "page_numbers": [1],
            },
        ),
    ]

    assert vector.add_documents(docs) == ["child_alpha", "child_beta"]
    assert vector.dense_search("alpha query", k=1)[0].metadata["chunk_id"] == "child_alpha"
    assert vector.sparse_search("机器学习", k=2)[0].metadata["chunk_id"] == "child_alpha"
    assert [row["metadata"]["chunk_id"] for row in vector.load_child_neighbors([{"parent_id": "parent_1", "chunk_index": 1}], window=1)] == [
        "child_alpha",
        "child_beta",
    ]

    vector.delete_by_source_file("doc.md")
    assert vector.dense_search("alpha query", k=1) == []


def test_embedding_dimension_mismatch_is_clear(monkeypatch):
    with postgres.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS child_chunks")
            cur.execute("DROP TABLE IF EXISTS parent_chunks")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE child_chunks (
                    id SERIAL PRIMARY KEY,
                    child_id VARCHAR(255) UNIQUE NOT NULL,
                    parent_id VARCHAR(255) NOT NULL,
                    doc_id VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(3)
                )
                """
            )
    postgres.reset_pool_for_tests()

    with pytest.raises(postgres.SchemaMismatchError, match="DENSE_EMBEDDING_DIMENSION"):
        postgres.ensure_schema()

    with postgres.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS child_chunks")
            cur.execute("DROP TABLE IF EXISTS parent_chunks")
    postgres.reset_pool_for_tests()
    postgres.ensure_schema()
