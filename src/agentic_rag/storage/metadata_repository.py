from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from psycopg2.extras import Json, RealDictCursor

from agentic_rag.ingestion.course_structure import CourseStructureStore, parse_course_names
from agentic_rag.storage.postgres import ensure_schema, transaction


def _course_id(name: str) -> str:
    normalized = re.sub(r"\s+", "-", name.strip().lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "", normalized).strip("-_") or "course"
    digest = hashlib.sha1(name.casefold().encode("utf-8")).hexdigest()[:8]
    return f"course_{normalized}_{digest}"


class PgDocumentRepository:
    def __init__(self):
        ensure_schema()

    def create_processing(
        self,
        document_id: str,
        original_file: str,
        source_file: str,
        raw_object_key: str,
    ) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_documents (
                        document_id, original_file, source_file, original_extension,
                        status, raw_object_key
                    )
                    VALUES (%s, %s, %s, %s, 'processing', %s)
                    """,
                    (
                        document_id,
                        original_file,
                        source_file,
                        Path(original_file).suffix.lower(),
                        raw_object_key,
                    ),
                )

    def mark_success(
        self,
        document_id: str,
        *,
        raw_file_hash: str,
        markdown_hash: str,
        markdown_object_key: str,
        image_object_keys: list[str],
        parent_count: int,
        child_count: int,
        index_config: dict,
        last_result: dict,
    ) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_documents
                    SET status = 'success',
                        raw_file_hash = %s,
                        markdown_hash = %s,
                        markdown_object_key = %s,
                        image_object_keys = %s,
                        parent_count = %s,
                        child_count = %s,
                        index_config = %s,
                        last_result = %s,
                        last_error = NULL,
                        updated_at = now()
                    WHERE document_id = %s
                    """,
                    (
                        raw_file_hash,
                        markdown_hash,
                        markdown_object_key,
                        Json(image_object_keys),
                        parent_count,
                        child_count,
                        Json(index_config),
                        Json(last_result),
                        document_id,
                    ),
                )

    def mark_failed(self, document_id: str, error: str, last_result: dict | None = None) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_documents
                    SET status = 'failed', last_error = %s, last_result = %s, updated_at = now()
                    WHERE document_id = %s
                    """,
                    (error, Json(last_result or {}), document_id),
                )

    def set_status(self, document_id: str, status: str, error: str | None = None) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE rag_documents
                    SET status = %s, last_error = %s, updated_at = now()
                    WHERE document_id = %s
                    """,
                    (status, error, document_id),
                )

    def get_document(self, identifier: str) -> dict | None:
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM rag_documents
                    WHERE document_id::text = %s OR source_file = %s OR original_file = %s
                    ORDER BY created_at
                    LIMIT 1
                    """,
                    (identifier, identifier, identifier),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def list_documents(self, statuses: Iterable[str] | None = None) -> list[dict]:
        params: tuple = ()
        where = ""
        if statuses:
            where = "WHERE status = ANY(%s)"
            params = (list(statuses),)
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"SELECT * FROM rag_documents {where} ORDER BY created_at, document_id", params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def course_names_for_document(self, document_id: str) -> list[str]:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.name
                    FROM courses c
                    JOIN course_documents cd ON cd.course_id = c.course_id
                    WHERE cd.document_id = %s
                    ORDER BY lower(c.name)
                    """,
                    (document_id,),
                )
                rows = cur.fetchall()
        return [row[0] for row in rows]

    def delete_document(self, document_id: str) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_documents WHERE document_id = %s", (document_id,))

    def clear(self) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_documents")
                cur.execute("DELETE FROM courses")


class PgCourseStructureStore:
    """PostgreSQL-backed course metadata with row-locked updates."""

    def __init__(self):
        ensure_schema()

    def list_courses(self) -> list[dict]:
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT c.course_id, c.name, c.summary, c.sections, c.created_at, c.updated_at,
                           COALESCE(
                               array_agg(cd.document_id::text ORDER BY cd.document_id)
                               FILTER (WHERE cd.document_id IS NOT NULL),
                               ARRAY[]::text[]
                           ) AS documents
                    FROM courses c
                    LEFT JOIN course_documents cd ON cd.course_id = c.course_id
                    GROUP BY c.course_id
                    ORDER BY lower(c.name)
                    """
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def get_course(self, course_id: str) -> dict | None:
        return next((course for course in self.list_courses() if course["course_id"] == course_id), None)

    def get_course_by_name(self, name: str) -> dict | None:
        wanted = name.strip().casefold()
        return next((course for course in self.list_courses() if course["name"].casefold() == wanted), None)

    def ensure_course(self, name: str, conn=None) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Course name cannot be empty")
        course_id = _course_id(name)

        def execute(connection):
            with connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO courses (course_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (course_id, name),
                )
                cur.execute("SELECT course_id FROM courses WHERE lower(name) = lower(%s)", (name,))
                return cur.fetchone()[0]

        if conn is not None:
            return execute(conn)
        with transaction() as owned:
            return execute(owned)

    def assign_document_to_courses(
        self,
        document_id: str,
        course_names: Iterable[str],
        markdown_path: str | Path,
        source_file: str,
    ) -> list[str]:
        extractor = CourseStructureStore.__new__(CourseStructureStore)
        sections = extractor._extract_sections(Path(markdown_path), source_file)
        for section in sections:
            section["document_id"] = document_id
            for point in section.get("knowledge_points", []):
                point["document_id"] = document_id

        course_ids = []
        with transaction() as conn:
            for name in parse_course_names(course_names):
                course_id = self.ensure_course(name, conn=conn)
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT sections FROM courses WHERE course_id = %s FOR UPDATE", (course_id,))
                    row = cur.fetchone() or {"sections": []}
                    merged = [
                        item for item in list(row.get("sections") or [])
                        if item.get("document_id") != document_id
                    ]
                    merged.extend(sections)
                    summary = self._summary(merged, self._document_count(conn, course_id, document_id))
                    cur.execute(
                        """
                        INSERT INTO course_documents (course_id, document_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (course_id, document_id),
                    )
                    cur.execute(
                        """
                        UPDATE courses
                        SET sections = %s, summary = %s, updated_at = now()
                        WHERE course_id = %s
                        """,
                        (Json(merged), summary, course_id),
                    )
                course_ids.append(course_id)
        return course_ids

    def rename_course(self, course_id_or_name: str, new_name: str) -> bool:
        if not new_name.strip():
            return False
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE courses SET name = %s, updated_at = now()
                    WHERE course_id = %s OR lower(name) = lower(%s)
                    """,
                    (new_name.strip(), course_id_or_name, course_id_or_name),
                )
                return cur.rowcount > 0

    def rename_section(self, course_id_or_name: str, section_id_or_title: str, new_title: str) -> bool:
        course = self.get_course(course_id_or_name) or self.get_course_by_name(course_id_or_name)
        if not course or not section_id_or_title.strip() or not new_title.strip():
            return False
        changed = False
        sections = list(course.get("sections") or [])
        for section in sections:
            if (
                section.get("section_id") == section_id_or_title
                or section.get("title", "").casefold() == section_id_or_title.casefold()
            ):
                section["title"] = new_title.strip()
                section["user_edited"] = True
                changed = True
                break
        if changed:
            with transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE courses SET sections = %s, updated_at = now() WHERE course_id = %s",
                        (Json(sections), course["course_id"]),
                    )
        return changed

    def source_files_for_course(self, course_id_or_name: str | None) -> list[str]:
        if not course_id_or_name:
            return []
        course = self.get_course(course_id_or_name) or self.get_course_by_name(course_id_or_name)
        return sorted(course.get("documents", [])) if course else []

    def remove_document(self, document_id: str) -> list[str]:
        affected = []
        with transaction() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT c.course_id, c.name, c.sections
                    FROM courses c
                    JOIN course_documents cd ON cd.course_id = c.course_id
                    WHERE cd.document_id = %s
                    FOR UPDATE
                    """,
                    (document_id,),
                )
                courses = cur.fetchall()
                cur.execute("DELETE FROM course_documents WHERE document_id = %s", (document_id,))
                for course in courses:
                    sections = [
                        item for item in list(course.get("sections") or [])
                        if item.get("document_id") != document_id
                    ]
                    cur.execute(
                        "UPDATE courses SET sections = %s, summary = %s, updated_at = now() WHERE course_id = %s",
                        (Json(sections), self._summary(sections, self._document_count(conn, course["course_id"])), course["course_id"]),
                    )
                    affected.append(course["name"])
        return affected

    def clear(self) -> None:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM courses")

    def format_course_list(self) -> str:
        courses = self.list_courses()
        if not courses:
            return "暂无课程。上传资料时填写课程名称后会自动创建课程。"
        return "\n".join(
            f"- {course['name']}：{len(course.get('documents', []))} 份资料，"
            f"{len(course.get('sections', []))} 个复习单元"
            for course in courses
        )

    @staticmethod
    def _document_count(conn, course_id: str, pending_document_id: str | None = None) -> int:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM course_documents WHERE course_id = %s", (course_id,))
            count = int(cur.fetchone()[0])
            if pending_document_id:
                cur.execute(
                    "SELECT 1 FROM course_documents WHERE course_id = %s AND document_id = %s",
                    (course_id, pending_document_id),
                )
                if cur.fetchone() is None:
                    count += 1
            return count

    @staticmethod
    def _summary(sections: list[dict], document_count: int) -> str:
        if not sections:
            return "尚未生成课程结构。"
        titles = "、".join(section.get("title", "") for section in sections[:5])
        suffix = "等" if len(sections) > 5 else ""
        return f"已整理 {document_count} 份资料，识别 {len(sections)} 个复习单元：{titles}{suffix}。"
