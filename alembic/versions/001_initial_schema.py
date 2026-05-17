"""initial schema: 7 tables

Revision ID: 001
Revises:
Create Date: 2026-05-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(255)),
        sa.Column("email", sa.String(255)),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column("document_ids", postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("sections", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("schema_version", sa.Integer(), server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_kb_document_ids", "knowledge_bases", ["document_ids"], postgresql_using="GIN")

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.String(255), unique=True, nullable=False),
        sa.Column("source_file", sa.String(500), nullable=False),
        sa.Column("original_file", sa.String(500)),
        sa.Column("original_extension", sa.String(50)),
        sa.Column("source_path", sa.Text()),
        sa.Column("converter", sa.String(100)),
        sa.Column("markdown_path", sa.Text()),
        sa.Column("raw_file_hash", sa.String(64)),
        sa.Column("markdown_hash", sa.String(64)),
        sa.Column("document_hash", sa.String(64)),
        sa.Column("status", sa.String(50), server_default=sa.text("'pending'")),
        sa.Column("parent_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("child_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("pages", postgresql.JSONB()),
        sa.Column("index_config", postgresql.JSONB()),
        sa.Column("last_result", postgresql.JSONB()),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_documents_status", "documents", ["status"])
    op.create_index("idx_documents_source", "documents", ["source_file"])

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
        sa.Column("embedding", sa.NullType()),  # pgvector vector(768)
        sa.Column("content_tsv", postgresql.TSVECTOR()),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute("CREATE INDEX idx_child_chunks_embedding ON child_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)")
    op.execute("CREATE INDEX idx_child_chunks_tsv ON child_chunks USING GIN (content_tsv)")
    op.create_index("idx_child_chunks_parent", "child_chunks", ["parent_id"])
    op.create_index("idx_child_chunks_doc", "child_chunks", ["doc_id"])

    op.create_table(
        "conversation_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("kb_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id")),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_conv_msg_session_turn", "conversation_messages", ["session_id", "turn_index"])
    op.create_index("idx_conv_msg_session_created", "conversation_messages", ["session_id", "created_at"])
    op.create_index("idx_conv_msg_user", "conversation_messages", ["user_id"], postgresql_where=sa.text("user_id IS NOT NULL"))

    op.create_table(
        "eval_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(255), nullable=False),
        sa.Column("question_id", sa.String(255), nullable=False),
        sa.Column("dataset_name", sa.String(500)),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text()),
        sa.Column("source_file", sa.String(500)),
        sa.Column("gold_parent_ids", postgresql.ARRAY(sa.String())),
        sa.Column("gold_child_ids", postgresql.ARRAY(sa.String())),
        sa.Column("gold_evidence_text", sa.Text()),
        sa.Column("question_type", sa.String(100)),
        sa.Column("difficulty", sa.String(50)),
        sa.Column("tags", postgresql.ARRAY(sa.String())),
        sa.Column("rag_output", sa.Text()),
        sa.Column("retrieval_metrics", postgresql.JSONB()),
        sa.Column("rag_metrics", postgresql.JSONB()),
        sa.Column("error_info", sa.Text()),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_eval_results_run", "eval_results", ["run_id"])
    op.create_index("idx_eval_results_dataset", "eval_results", ["dataset_name"])


def downgrade() -> None:
    op.drop_table("eval_results")
    op.drop_table("conversation_messages")
    op.drop_table("child_chunks")
    op.drop_table("parent_chunks")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
    op.drop_table("users")
