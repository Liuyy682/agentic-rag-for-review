from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.sql import func

import config
from database.base import Base


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_id = Column(String(255), unique=True, nullable=False)
    doc_id = Column(String(255), nullable=False)
    page_content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime, server_default=func.now())


class ChildChunk(Base):
    __tablename__ = "child_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    child_id = Column(String(255), unique=True, nullable=False)
    parent_id = Column(String(255), nullable=False)
    doc_id = Column(String(255), nullable=False)
    chunk_index = Column(Integer)
    source = Column(String(500))
    source_file = Column(String(500))
    page_numbers = Column(ARRAY(Integer))
    slide_title = Column(String(500))
    content = Column(Text, nullable=False)
    embedding = Column(Vector(config.DENSE_EMBEDDING_DIMENSION))
    content_tsv = Column(TSVECTOR)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime, server_default=func.now())
