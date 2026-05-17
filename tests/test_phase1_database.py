"""Phase 1 verification tests per docs/database-design.md Section 9.

Requires: docker compose up -d postgres && alembic upgrade head
"""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from project.config import DATABASE_URL
from project.database.engine import engine, SessionLocal
from project.database.models import (
    User,
    KnowledgeBase,
    Document,
    ParentChunk,
    ChildChunk,
    ConversationMessage,
    EvalResult,
)


# ── 1.1-1.2: config + connection ────────────────────────────────────

def test_database_url_configured():
    """1.2 DATABASE_URL is set."""
    assert DATABASE_URL is not None
    assert DATABASE_URL.startswith("postgresql://")


def test_engine_connects():
    """1.3 engine connection pool works."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


# ── 1.4: tables exist ───────────────────────────────────────────────

EXPECTED_TABLES = [
    "users",
    "knowledge_bases",
    "documents",
    "parent_chunks",
    "child_chunks",
    "conversation_messages",
    "eval_results",
]


def test_all_tables_exist():
    """1.4 all 7 tables created."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public' ORDER BY tablename"
            )
        )
        tables = [row[0] for row in result]
    for name in EXPECTED_TABLES:
        assert name in tables, f"Table '{name}' not found in {tables}"


# ── 1.5-1.7: vector + HNSW + GIN indexes ───────────────────────────

def test_vector_type_available():
    """1.5 vector type works."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT '[1,2,3]'::vector"))
        assert result.scalar() is not None


def test_hnsw_index_exists():
    """1.6 HNSW index on child_chunks.embedding."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE indexname = 'idx_child_chunks_embedding'"
            )
        )
        assert result.scalar() == "idx_child_chunks_embedding"


def test_gin_index_exists():
    """1.7 GIN index on child_chunks.content_tsv."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE indexname = 'idx_child_chunks_tsv'"
            )
        )
        assert result.scalar() == "idx_child_chunks_tsv"


# ── 1.8: vector insert + cosine search ──────────────────────────────

def test_vector_insert_and_search():
    """1.8 write a ChildChunk with embedding, then cosine_distance search hits it."""
    import numpy as np

    session = SessionLocal()
    try:
        embedding = np.random.randn(768).astype(np.float32).tolist()

        chunk = ChildChunk(
            child_id="test_1_8_child",
            parent_id="test_1_8_parent",
            doc_id="test_1_8_doc",
            content="vector test chunk",
            embedding=embedding,
        )
        session.add(chunk)
        session.commit()

        # Search: use <=> operator with literal vector cast
        vec_str = "'[" + ",".join(str(v) for v in embedding) + "]'"
        result = session.execute(
            text(
                "SELECT child_id, 1 - (embedding <=> " + vec_str + "::vector) AS sim "
                "FROM child_chunks "
                "WHERE child_id = 'test_1_8_child' "
                "ORDER BY embedding <=> " + vec_str + "::vector LIMIT 1"
            ),
        ).first()
        assert result is not None
        assert result[0] == "test_1_8_child"
        assert result[1] > 0.99

        # cleanup
        session.execute(text("DELETE FROM child_chunks WHERE child_id = 'test_1_8_child'"))
        session.commit()
    finally:
        session.close()


# ── 1.9: jieba tsvector Chinese search ──────────────────────────────

def test_jieba_tsvector_chinese_search():
    """1.9 jieba tokenized tsvector + ts_rank Chinese search."""
    import jieba
    from sqlalchemy import func as sa_func

    session = SessionLocal()
    try:
        content = "深度学习是机器学习的一个重要分支"
        tokens = " ".join(jieba.cut(content))

        chunk = ChildChunk(
            child_id="test_1_9_child",
            parent_id="test_1_9_parent",
            doc_id="test_1_9_doc",
            content=content,
            content_tsv=sa_func.to_tsvector("simple", tokens),
        )
        session.add(chunk)
        session.commit()

        query = "深度学习"
        query_tokens = " & ".join(jieba.cut(query))

        result = session.execute(
            text(
                "SELECT child_id, ts_rank(content_tsv, to_tsquery('simple', :q)) AS rank "
                "FROM child_chunks "
                "WHERE content_tsv @@ to_tsquery('simple', :q) "
                "ORDER BY rank DESC LIMIT 5"
            ),
            {"q": query_tokens},
        ).first()
        assert result is not None
        assert result[0] == "test_1_9_child"
        assert result[1] > 0

        # cleanup
        session.execute(text("DELETE FROM child_chunks WHERE child_id = 'test_1_9_child'"))
        session.commit()
    finally:
        session.close()


# ── Model CRUD smoke tests ──────────────────────────────────────────

def test_user_crud():
    session = SessionLocal()
    try:
        u = User(username="test_user", email="test@example.com")
        session.add(u)
        session.commit()
        assert u.id is not None
        session.delete(u)
        session.commit()
    finally:
        session.close()


def test_knowledge_base_crud():
    session = SessionLocal()
    try:
        kb = KnowledgeBase(
            name="Test KB",
            document_ids=["doc1", "doc2"],
            sections=[{"section_id": "s1", "title": "Intro"}],
        )
        session.add(kb)
        session.commit()
        assert kb.id is not None
        session.delete(kb)
        session.commit()
    finally:
        session.close()


def test_document_crud():
    session = SessionLocal()
    try:
        doc = Document(
            doc_id="test_doc",
            source_file="test.pdf",
            status="pending",
            pages={"1": {"page_hash": "abc123"}},
        )
        session.add(doc)
        session.commit()
        assert doc.id is not None
        session.delete(doc)
        session.commit()
    finally:
        session.close()


def test_parent_chunk_crud():
    session = SessionLocal()
    try:
        pc = ParentChunk(
            parent_id="test_parent_0",
            doc_id="test_doc",
            page_content="Some page content here.",
        )
        session.add(pc)
        session.commit()
        assert pc.id is not None
        session.delete(pc)
        session.commit()
    finally:
        session.close()


def test_conversation_message_crud():
    session = SessionLocal()
    try:
        msg = ConversationMessage(
            session_id="sess-001",
            turn_index=0,
            role="user",
            content="Hello",
        )
        session.add(msg)
        session.commit()
        assert msg.id is not None
        session.delete(msg)
        session.commit()
    finally:
        session.close()


def test_eval_result_crud():
    session = SessionLocal()
    try:
        er = EvalResult(
            run_id="run-001",
            question_id="q1",
            question="What is RAG?",
        )
        session.add(er)
        session.commit()
        assert er.id is not None
        session.delete(er)
        session.commit()
    finally:
        session.close()


# ── Model-only tests (no DB required) ───────────────────────────────

def test_models_tablenames():
    """Verify the 7 models map to the correct table names."""
    assert User.__tablename__ == "users"
    assert KnowledgeBase.__tablename__ == "knowledge_bases"
    assert Document.__tablename__ == "documents"
    assert ParentChunk.__tablename__ == "parent_chunks"
    assert ChildChunk.__tablename__ == "child_chunks"
    assert ConversationMessage.__tablename__ == "conversation_messages"
    assert EvalResult.__tablename__ == "eval_results"


def test_child_chunk_has_vector_column():
    """child_chunks must have an embedding column of type vector."""
    col = ChildChunk.__table__.columns["embedding"]
    assert col is not None


def test_child_chunk_has_tsv_column():
    """child_chunks must have a content_tsv column."""
    col = ChildChunk.__table__.columns["content_tsv"]
    assert col is not None
