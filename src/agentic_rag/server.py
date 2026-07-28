import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag.api.deps import close_rag_app
from agentic_rag.api.routes import router
from agentic_rag.api.tasks import task_store
from agentic_rag import config
from agentic_rag.security.auth import validate_auth_config
from agentic_rag.security.users import UserRepository
from agentic_rag.storage.postgres import ensure_schema

STATIC_DIR = Path(__file__).parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_config()
    ensure_schema()
    if config.INITIAL_ADMIN_EMAIL:
        users = UserRepository()
        if users.find_by_email(config.INITIAL_ADMIN_EMAIL) is None:
            users.create_user(
                email=config.INITIAL_ADMIN_EMAIL,
                password=config.INITIAL_ADMIN_PASSWORD,
                tenant_id=config.AUTH_TENANT_ID,
                role="admin",
            )
    cleanup_task = asyncio.create_task(task_store._cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        close_rag_app()


app = FastAPI(title="Agentic RAG", lifespan=lifespan)

app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
