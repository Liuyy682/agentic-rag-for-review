from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from database.base import Base


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    turn_index = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB)
    created_at = Column(DateTime, server_default=func.now())
