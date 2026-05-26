import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.deps import get_rag_app
from api.stream import stream_chat
from api.tasks import task_store

router = APIRouter()


# ── Pydantic models ──────────────────────────────────────────────────────────

class RenameCourseRequest(BaseModel):
    current_name: str
    new_name: str

class RenameSectionRequest(BaseModel):
    course_name: str
    current_section: str
    new_section: str

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    course_name: Optional[str] = None
    session_id: str

class ClearChatRequest(BaseModel):
    session_id: str


# ── Document endpoints ───────────────────────────────────────────────────────

@router.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    course_names: str = Form(default=""),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    rag_app = get_rag_app()
    doc_manager = rag_app.document_manager

    # Save uploaded files to a temp directory
    tmpdir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    saved_paths = []
    for f in files:
        safe_name = Path(f.filename or "upload").name
        dest = tmpdir / safe_name
        content = await f.read()
        dest.write_bytes(content)
        saved_paths.append(str(dest))

    task_id = task_store.create()

    def _run_ingestion():
        try:
            def progress_callback(progress: float, desc: str):
                task_store.update(task_id, progress=progress, description=desc)

            result = doc_manager.add_documents_detailed(
                saved_paths,
                progress_callback=progress_callback,
                course_names=course_names if course_names else None,
            )
            task_store.update(
                task_id,
                status="completed",
                progress=1.0,
                description="Done",
                result={
                    "added": result.added,
                    "skipped": result.skipped,
                    "failed": result.failed,
                    "course_updated": result.course_updated,
                },
            )
        except Exception as e:
            task_store.update(task_id, status="failed", error=str(e))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_ingestion)

    return {"task_id": task_id, "status": "processing"}


@router.get("/documents/tasks/{task_id}")
async def get_task(task_id: str):
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/documents/files")
async def list_files():
    doc_manager = get_rag_app().document_manager
    files = await asyncio.to_thread(doc_manager.get_markdown_files)
    return {"files": files}


@router.get("/documents/courses")
async def list_courses():
    doc_manager = get_rag_app().document_manager
    choices = await asyncio.to_thread(doc_manager.get_course_choices)
    formatted = await asyncio.to_thread(doc_manager.get_course_list)
    return {"choices": choices, "formatted": formatted}


@router.post("/documents/clear")
async def clear_documents():
    doc_manager = get_rag_app().document_manager
    await asyncio.to_thread(doc_manager.clear_all)
    return {"status": "ok"}


@router.post("/documents/courses/rename")
async def rename_course(body: RenameCourseRequest):
    if not body.current_name or not body.new_name:
        raise HTTPException(status_code=400, detail="Both current_name and new_name are required")
    doc_manager = get_rag_app().document_manager
    ok = await asyncio.to_thread(doc_manager.rename_course, body.current_name, body.new_name)
    if not ok:
        return {"success": False, "error": f"Failed to rename '{body.current_name}'"}
    choices = await asyncio.to_thread(doc_manager.get_course_choices)
    formatted = await asyncio.to_thread(doc_manager.get_course_list)
    return {"success": True, "choices": choices, "formatted": formatted}


@router.post("/documents/sections/rename")
async def rename_section(body: RenameSectionRequest):
    if not body.course_name or not body.current_section or not body.new_section:
        raise HTTPException(status_code=400, detail="All fields are required")
    doc_manager = get_rag_app().document_manager
    ok = await asyncio.to_thread(
        doc_manager.rename_section,
        body.course_name,
        body.current_section,
        body.new_section,
    )
    if not ok:
        return {"success": False, "error": f"Failed to rename section '{body.current_section}'"}
    formatted = await asyncio.to_thread(doc_manager.get_course_list)
    return {"success": True, "formatted": formatted}


# ── Chat endpoints ───────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    rag_app = get_rag_app()
    chat_interface = rag_app.chat_interface

    async def event_generator():
        async for sse_str in stream_chat(
            chat_interface=chat_interface,
            message=body.message,
            history=body.history,
            course_name=body.course_name,
            session_id=body.session_id,
        ):
            # Check for client disconnect
            if await request.is_disconnected():
                break
            yield sse_str

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/clear")
async def clear_chat(body: ClearChatRequest):
    rag_app = get_rag_app()
    chat_interface = rag_app.chat_interface
    # Use the chat_interface's clear_session, but we need to set the session_id first
    await asyncio.to_thread(chat_interface.clear_session, body.session_id)
    return {"status": "ok"}
