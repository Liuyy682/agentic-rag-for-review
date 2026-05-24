"""switch chunks to BGE zh embeddings

Revision ID: 002
Revises: 001
Create Date: 2026-05-24

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    _drop_chunk_tables()
    _create_parent_chunks()
    _create_child_chunks(1024)


def downgrade() -> None:
    _drop_chunk_tables()
    _create_parent_chunks()
    _create_child_chunks(768)


def _drop_chunk_tables() -> None:
    op.execute("DROP INDEX IF EXISTS idx_child_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS idx_child_chunks_tsv")
    op.execute("DROP INDEX IF EXISTS idx_child_chunks_parent")
    op.execute("DROP INDEX IF EXISTS idx_child_chunks_doc")
    op.execute("DROP INDEX IF EXISTS idx_parent_chunks_doc")
    op.execute("DROP TABLE IF EXISTS child_chunks")
    op.execute("DROP TABLE IF EXISTS parent_chunks")


def _create_parent_chunks() -> None:
    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("parent_id", sa.String(255), unique=True, nullable=False),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("page_content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_parent_chunks_doc", "parent_chunks", ["doc_id"])


def _create_child_chunks(embedding_dimension: int) -> None:
    op.create_table(
        "child_chunks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("child_id", sa.String(255), unique=True, nullable=False),
        sa.Column("parent_id", sa.String(255), nullable=False),
        sa.Column("doc_id", sa.String(255), nullable=False),
        sa.Column("chunk_index", sa.Integer()),
        sa.Column("source", sa.String(500)),
        sa.Column("source_file", sa.String(500)),
        sa.Column("page_numbers", postgresql.ARRAY(sa.Integer())),
        sa.Column("slide_title", sa.String(500)),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(embedding_dimension)),
        sa.Column("content_tsv", postgresql.TSVECTOR()),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        "CREATE INDEX idx_child_chunks_embedding "
        "ON child_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 200)"
    )
    op.execute("CREATE INDEX idx_child_chunks_tsv ON child_chunks USING GIN (content_tsv)")
    op.create_index("idx_child_chunks_parent", "child_chunks", ["parent_id"])
    op.create_index("idx_child_chunks_doc", "child_chunks", ["doc_id"])
