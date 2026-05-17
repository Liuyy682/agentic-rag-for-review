from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from project.database.base import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(255), unique=True, nullable=False)
    source_file = Column(String(500), nullable=False)
    original_file = Column(String(500))
    original_extension = Column(String(50))
    source_path = Column(Text)
    converter = Column(String(100))
    markdown_path = Column(Text)
    raw_file_hash = Column(String(64))
    markdown_hash = Column(String(64))
    document_hash = Column(String(64))
    status = Column(String(50), default="pending")
    parent_count = Column(Integer, default=0)
    child_count = Column(Integer, default=0)
    pages = Column(JSONB)
    index_config = Column(JSONB)
    last_result = Column(JSONB)
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
