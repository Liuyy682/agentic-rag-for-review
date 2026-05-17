from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config as _config

DATABASE_URL = getattr(_config, "DATABASE_URL", "postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
