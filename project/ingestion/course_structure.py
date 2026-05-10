from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config
from ingestion.cleaning import parse_pages


COURSE_SCHEMA_VERSION = 1
MAX_SECTIONS_PER_DOCUMENT = 20
MAX_KNOWLEDGE_POINTS_PER_SECTION = 10


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    normalized = re.sub(r"\s+", "-", value.strip().lower())
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "", normalized)
    return normalized.strip("-_") or "course"


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{_slug(value)}_{digest}"


def _clean_title(value: str) -> str:
    title = re.sub(r"^#+\s*", "", value or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title


class CourseStructureStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or config.COURSE_STRUCTURE_PATH)
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Course structure file is invalid JSON: {self.path}") from exc
        data.setdefault("schema_version", COURSE_SCHEMA_VERSION)
        data.setdefault("courses", {})
        return data

    def _empty(self) -> dict:
        return {
            "schema_version": COURSE_SCHEMA_VERSION,
            "courses": {},
            "updated_at": _utc_now(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["schema_version"] = COURSE_SCHEMA_VERSION
        self.data["updated_at"] = _utc_now()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def clear(self) -> None:
        self.data = self._empty()
        self.save()

    def list_courses(self) -> list[dict]:
        return sorted(self.data.get("courses", {}).values(), key=lambda item: item.get("name", "").lower())

    def get_course(self, course_id: str) -> dict | None:
        return self.data.get("courses", {}).get(course_id)

    def get_course_by_name(self, name: str) -> dict | None:
        normalized = name.strip().casefold()
        if not normalized:
            return None
        for course in self.data.get("courses", {}).values():
            if course.get("name", "").casefold() == normalized:
                return course
        return None

    def ensure_course(self, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Course name cannot be empty")

        existing = self.get_course_by_name(name)
        if existing:
            return existing["course_id"]

        course_id = _stable_id("course", name)
        suffix = 2
        while course_id in self.data["courses"]:
            course_id = f"{_stable_id('course', name)}_{suffix}"
            suffix += 1

        self.data["courses"][course_id] = {
            "course_id": course_id,
            "name": name,
            "documents": [],
            "summary": "",
            "sections": [],
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self.save()
        return course_id

    def assign_document_to_courses(self, source_file: str, course_names: Iterable[str], markdown_dir: str | Path | None = None) -> list[str]:
        course_ids = []
        for name in course_names:
            name = name.strip()
            if not name:
                continue
            course_id = self.ensure_course(name)
            course = self.data["courses"][course_id]
            if source_file not in course["documents"]:
                course["documents"].append(source_file)
            course["documents"] = sorted(set(course["documents"]))
            course["updated_at"] = _utc_now()
            self.rebuild_course(course_id, markdown_dir=markdown_dir, save=False)
            course_ids.append(course_id)
        self.save()
        return course_ids

    def rename_course(self, course_id_or_name: str, new_name: str) -> bool:
        course = self.get_course(course_id_or_name) or self.get_course_by_name(course_id_or_name)
        if not course or not new_name.strip():
            return False
        duplicate = self.get_course_by_name(new_name)
        if duplicate and duplicate["course_id"] != course["course_id"]:
            raise ValueError(f"Course already exists: {new_name}")
        course["name"] = new_name.strip()
        course["updated_at"] = _utc_now()
        self.save()
        return True

    def rename_section(self, course_id_or_name: str, section_id_or_title: str, new_title: str) -> bool:
        course = self.get_course(course_id_or_name) or self.get_course_by_name(course_id_or_name)
        if not course or not section_id_or_title.strip() or not new_title.strip():
            return False

        wanted = section_id_or_title.strip()
        for section in course.get("sections", []):
            if section.get("section_id") == wanted or section.get("title", "").casefold() == wanted.casefold():
                section["title"] = new_title.strip()
                section["user_edited"] = True
                course["updated_at"] = _utc_now()
                self.save()
                return True
        return False

    def source_files_for_course(self, course_id_or_name: str | None) -> list[str]:
        if not course_id_or_name:
            return []
        course = self.get_course(course_id_or_name) or self.get_course_by_name(course_id_or_name)
        if not course:
            return []
        return sorted(set(course.get("documents", [])))

    def rebuild_course(self, course_id: str, markdown_dir: str | Path | None = None, save: bool = True) -> bool:
        course = self.get_course(course_id)
        if not course:
            return False

        markdown_dir = Path(markdown_dir or config.MARKDOWN_DIR)
        existing_titles = {
            (section.get("source_file"), section.get("original_title")): section
            for section in course.get("sections", [])
            if section.get("user_edited")
        }

        sections = []
        for source_file in course.get("documents", []):
            md_path = markdown_dir / f"{Path(source_file).stem}.md"
            if not md_path.exists():
                continue
            document_sections = self._extract_sections(md_path, source_file)
            for section in document_sections:
                edited = existing_titles.get((section["source_file"], section["original_title"]))
                if edited:
                    section["title"] = edited.get("title", section["title"])
                    section["user_edited"] = True
                sections.append(section)

        course["sections"] = sections[:MAX_SECTIONS_PER_DOCUMENT * max(len(course.get("documents", [])), 1)]
        course["summary"] = self._build_course_summary(course)
        course["updated_at"] = _utc_now()
        if save:
            self.save()
        return True

    def _extract_sections(self, md_path: Path, source_file: str) -> list[dict]:
        markdown_text = md_path.read_text(encoding="utf-8")
        pages = parse_pages(markdown_text, source_file=source_file)
        sections: list[dict] = []
        current: dict | None = None

        for page in pages:
            page_number = page.page_number or 1
            for line in page.raw_lines:
                heading = re.match(r"^\s{0,3}(#{1,4})\s+(.+?)\s*$", line)
                if heading:
                    level = len(heading.group(1))
                    title = _clean_title(heading.group(2))
                    if not title:
                        continue
                    if level <= 2 or current is None:
                        current = self._new_section(title, source_file, page_number)
                        sections.append(current)
                    else:
                        self._add_knowledge_point(current, title, source_file, page_number)
                    continue

                if current and self._looks_like_candidate_point(line):
                    self._add_knowledge_point(current, _clean_title(line), source_file, page_number)

        if not sections:
            title = Path(source_file).stem
            section = self._new_section(title, source_file, 1)
            section["summary"] = "未在资料中识别到明确章节标题，已按整份资料作为复习单元。"
            sections.append(section)

        return sections[:MAX_SECTIONS_PER_DOCUMENT]

    def _new_section(self, title: str, source_file: str, page_number: int) -> dict:
        section_key = f"{source_file}:{title}:{page_number}"
        return {
            "section_id": _stable_id("section", section_key),
            "title": title,
            "original_title": title,
            "summary": f"围绕“{title}”展开的课程内容。",
            "source_file": source_file,
            "page_numbers": [page_number],
            "knowledge_points": [],
            "user_edited": False,
        }

    def _add_knowledge_point(self, section: dict, title: str, source_file: str, page_number: int) -> None:
        title = re.sub(r"^[-*+\d.)\s]+", "", title).strip()
        if not title or len(title) > 80:
            return
        existing = {item.get("name", "").casefold() for item in section.get("knowledge_points", [])}
        if title.casefold() in existing:
            return
        if len(section["knowledge_points"]) >= MAX_KNOWLEDGE_POINTS_PER_SECTION:
            return
        section["knowledge_points"].append({
            "knowledge_point_id": _stable_id("kp", f"{source_file}:{section['section_id']}:{title}"),
            "name": title,
            "source_file": source_file,
            "page_numbers": [page_number],
        })
        if page_number not in section["page_numbers"]:
            section["page_numbers"].append(page_number)

    def _looks_like_candidate_point(self, line: str) -> bool:
        stripped = line.strip()
        if not re.match(r"^([-*+]|\d+[.)])\s+\S", stripped):
            return False
        stripped = re.sub(r"^([-*+]|\d+[.)])\s+", "", stripped)
        return 4 <= len(stripped) <= 80 and not stripped.startswith("http")

    def _build_course_summary(self, course: dict) -> str:
        sections = course.get("sections", [])
        if not sections:
            return "尚未生成课程结构。"
        titles = "、".join(section["title"] for section in sections[:5])
        suffix = "等" if len(sections) > 5 else ""
        return f"已整理 {len(course.get('documents', []))} 份资料，识别 {len(sections)} 个复习单元：{titles}{suffix}。"

    def format_course_list(self) -> str:
        courses = self.list_courses()
        if not courses:
            return "暂无课程。上传资料时填写课程名称后会自动创建课程。"

        lines = []
        for course in courses:
            lines.append(
                f"- {course['name']}：{len(course.get('documents', []))} 份资料，"
                f"{len(course.get('sections', []))} 个复习单元"
            )
        return "\n".join(lines)


def parse_course_names(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,，\n;；]+", value)
    else:
        raw_items = list(value)
    names = []
    seen = set()
    for item in raw_items:
        name = str(item).strip()
        key = name.casefold()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names
