from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.sql import func

from project.database.base import Base


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(255), nullable=False)
    question_id = Column(String(255), nullable=False)
    dataset_name = Column(String(500))
    question = Column(Text, nullable=False)
    reference_answer = Column(Text)
    source_file = Column(String(500))
    gold_parent_ids = Column(ARRAY(String))
    gold_child_ids = Column(ARRAY(String))
    gold_evidence_text = Column(Text)
    question_type = Column(String(100))
    difficulty = Column(String(50))
    tags = Column(ARRAY(String))
    rag_output = Column(Text)
    retrieval_metrics = Column(JSONB)
    rag_metrics = Column(JSONB)
    error_info = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
