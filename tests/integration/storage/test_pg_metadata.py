import os
import tempfile
import uuid
from pathlib import Path

import pytest

from agentic_rag.storage.metadata_repository import PgCourseStructureStore, PgDocumentRepository
from agentic_rag.storage.postgres import reset_pool_for_tests


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PG_STORAGE_TESTS") != "1",
    reason="Set RUN_PG_STORAGE_TESTS=1 with a local PostgreSQL database.",
)


@pytest.fixture
def stores():
    reset_pool_for_tests()
    documents = PgDocumentRepository()
    courses = PgCourseStructureStore()
    documents.clear()
    yield documents, courses
    documents.clear()
    reset_pool_for_tests()


def test_document_and_course_metadata_are_shared_in_postgres(stores):
    documents, courses = stores
    document_id = str(uuid.uuid4())
    documents.create_processing(
        document_id,
        "notes.md",
        "notes.md",
        f"documents/{document_id}/original/notes.md",
    )
    documents.mark_success(
        document_id,
        raw_file_hash="raw",
        markdown_hash="markdown",
        markdown_object_key=f"documents/{document_id}/derived/document.md",
        image_object_keys=[],
        parent_count=1,
        child_count=2,
        index_config={"version": 1},
        last_result={"status": "added"},
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        markdown = Path(temp_dir) / "document.md"
        markdown.write_text("# Chapter\n\n- Knowledge point", encoding="utf-8")
        courses.assign_document_to_courses(
            document_id,
            ["Course A"],
            markdown,
            "notes.md",
        )

    reloaded_documents = PgDocumentRepository()
    reloaded_courses = PgCourseStructureStore()
    row = reloaded_documents.get_document(document_id)
    assert row["status"] == "success"
    assert row["markdown_object_key"].endswith("/derived/document.md")
    assert reloaded_courses.source_files_for_course("Course A") == [document_id]
    assert reloaded_documents.course_names_for_document(document_id) == ["Course A"]

    assert reloaded_courses.rename_course("Course A", "Course B")
    assert reloaded_courses.source_files_for_course("Course B") == [document_id]
    assert reloaded_courses.remove_document(document_id) == ["Course B"]
    reloaded_documents.delete_document(document_id)
    assert reloaded_documents.get_document(document_id) is None
