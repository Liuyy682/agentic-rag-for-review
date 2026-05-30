from __future__ import annotations

import json
import time
from pathlib import Path

import config
from ingestion.cleaning import CleanedMarkdown, clean_markdown_text, parse_pages
from ingestion.conversion import (
    clear_directory_contents,
    convert_document_to_markdown,
    is_supported_document,
)
from ingestion.course_structure import CourseStructureStore, parse_course_names
from ingestion.file_integrity import compute_file_hash
from ingestion.index_manifest import (
    IndexManifest,
    build_document_record,
    build_page_hashes,
    text_hash,
)
from ingestion.models import (
    AddDocumentsResult,
    DeleteDocumentResult,
    DocumentDetail,
    DocumentInfo,
    DocumentIngestionResult,
    IngestionDocument,
    IngestionStageResult,
    KnowledgeBaseStats,
)


class DocumentManager:

    def __init__(self, rag_system, course_store=None):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANED_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANING_LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.MARKDOWN_CLEANING_DIFF_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.DOCUMENT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        Path(config.INGESTION_LOG_DIR).mkdir(parents=True, exist_ok=True)
        self.manifest = IndexManifest()
        self.course_store = course_store or CourseStructureStore()

    def add_documents(self, document_paths, progress_callback=None, course_names=None):
        summary = self.add_documents_detailed(
            document_paths,
            progress_callback=progress_callback,
            course_names=course_names,
        )
        return summary.added, summary.skipped + summary.failed

    def add_documents_detailed(self, document_paths, progress_callback=None, course_names=None) -> AddDocumentsResult:
        if not document_paths:
            return AddDocumentsResult()

        document_paths = [document_paths] if isinstance(document_paths, str) else list(document_paths)
        document_paths = [path for path in document_paths if path]
        course_names = parse_course_names(course_names)
        results = []

        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), f"Processing {Path(doc_path).name}")
            results.append(self._process_document(doc_path, course_names))

        summary = AddDocumentsResult(documents=results)
        summary.added = sum(1 for result in results if result.status == "added")
        summary.skipped = sum(1 for result in results if result.status == "skipped")
        summary.failed = sum(1 for result in results if result.status == "failed")
        summary.course_updated = sum(1 for result in results if result.course_updated)
        return summary

    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name for p in self.markdown_dir.glob("*.md")])

    def list_documents(self) -> list[DocumentInfo]:
        documents = []
        for document in sorted(self.manifest.list_documents(), key=lambda item: item.get("source_file", "")):
            source_file = document.get("source_file") or ""
            documents.append(self._document_info(document, self._course_names_for_source(source_file)))
        return documents

    def get_document_detail(self, source_file: str) -> DocumentDetail | None:
        source_file = self._normalize_source_file(source_file)
        document = self.manifest.get_document(source_file)
        if not document:
            return None
        course_names = self._course_names_for_source(source_file)
        return DocumentDetail(
            info=self._document_info(document, course_names),
            markdown_path=document.get("markdown_path"),
            parent_ids=list(document.get("parent_ids", [])),
            child_ids=list(document.get("child_ids", [])),
            stage_stats=list(document.get("last_result", {}).get("stages", [])),
            last_error=document.get("last_error") or document.get("last_result", {}).get("error"),
            course_names=course_names,
        )

    def get_collection_stats(self) -> KnowledgeBaseStats:
        documents = self.manifest.list_documents()
        return KnowledgeBaseStats(
            document_count=len(documents),
            parent_count=sum(int(document.get("parent_count", 0)) for document in documents),
            child_count=sum(int(document.get("child_count", 0)) for document in documents),
            course_count=len(self.course_store.list_courses()),
        )

    def get_course_list(self):
        return self.course_store.format_course_list()

    def get_course_choices(self):
        return [course["name"] for course in self.course_store.list_courses()]

    def rename_course(self, current_name, new_name):
        return self.course_store.rename_course(current_name, new_name)

    def rename_section(self, course_name, current_section, new_section):
        return self.course_store.rename_section(course_name, current_section, new_section)

    def delete_document(self, source_file: str) -> DeleteDocumentResult:
        source_file = self._normalize_source_file(source_file)
        result = DeleteDocumentResult(success=False, source_file=source_file)
        document = self.manifest.get_document(source_file)

        try:
            self.rag_system.vector_db.delete_by_source_file(source_file)
            result.vector_deleted = True
        except Exception as exc:
            result.errors.append(f"vector delete failed: {exc}")

        try:
            self.rag_system.parent_store.delete_by_source_file(source_file)
            result.parent_deleted = True
        except Exception as exc:
            result.errors.append(f"parent delete failed: {exc}")

        try:
            removed = self.manifest.remove_document(source_file)
            if removed is not None:
                self.manifest.save()
                result.manifest_deleted = True
        except Exception as exc:
            result.errors.append(f"manifest delete failed: {exc}")

        markdown_path = Path(document.get("markdown_path")) if document and document.get("markdown_path") else self.markdown_dir / source_file
        try:
            self._unlink_if_exists(markdown_path)
            result.markdown_deleted = True
        except Exception as exc:
            result.errors.append(f"markdown delete failed: {exc}")

        try:
            self._delete_cleaning_outputs(source_file)
            result.cleaning_outputs_deleted = True
        except Exception as exc:
            result.errors.append(f"cleaning output delete failed: {exc}")

        try:
            affected_courses = self.course_store.remove_document(source_file, markdown_dir=self.markdown_dir)
            result.course_updated = bool(affected_courses)
        except Exception as exc:
            result.errors.append(f"course update failed: {exc}")

        result.success = not result.errors
        return result

    def _process_document(self, document_path, course_names: list[str]) -> DocumentIngestionResult:
        result = DocumentIngestionResult(source_path=str(document_path))
        try:
            doc_path = self._validate_document_path(document_path, result)
            source_file = self._normalize_source_file(doc_path.name)
            result.source_file = source_file
            result.original_file = doc_path.name

            raw_file_hash = self._hash_document(doc_path, result)
            result.raw_file_hash = raw_file_hash

            should_skip, reason = self._should_skip_indexing(source_file, raw_file_hash)
            self._add_stage(
                result,
                "skip_check",
                "skipped" if should_skip else "success",
                details={"reason": reason},
            )
            if should_skip:
                existing = self.manifest.get_document(source_file) or {}
                result.status = "skipped"
                result.reason = reason
                result.parent_count = int(existing.get("parent_count", 0))
                result.child_count = int(existing.get("child_count", 0))
                self._mark_index_stages_skipped(result, reason)
                self._bind_courses_if_needed(result, source_file, course_names)
                self._update_manifest_result(source_file, result)
                self._write_stage_logs(result)
                return result

            document = self._build_ingestion_document(doc_path, source_file, raw_file_hash, result)
            page_hashes = self._clean_document(document, result)
            parent_chunks, child_chunks = self._chunk_document(document, result)
            self._delete_old_index(document.source_file, result)
            self._write_vector_chunks(child_chunks, result)
            self._write_parent_chunks(parent_chunks, result)

            result.status = "added"
            result.reason = "indexed"
            result.indexed = True
            result.parent_count = len(parent_chunks)
            result.child_count = len(child_chunks)
            self._save_document_manifest(document, page_hashes, parent_chunks, child_chunks, result)
            self._bind_courses_if_needed(result, source_file, course_names)
            self._update_manifest_result(source_file, result)
            self._write_stage_logs(result)
            return result
        except Exception as exc:
            result.status = "failed"
            result.reason = result.reason or "processing_failed"
            result.error = str(exc)
            self._write_stage_logs(result)
            return result

    def _validate_document_path(self, document_path, result: DocumentIngestionResult) -> Path:
        start = time.perf_counter()
        try:
            doc_path = Path(document_path)
            if not doc_path.is_file():
                raise ValueError(f"Document is not a local file: {doc_path}")
            if not is_supported_document(doc_path):
                raise ValueError(f"Unsupported document type: {doc_path.suffix}")
            self._add_stage(
                result,
                "validate",
                "success",
                started_at=start,
                details={"extension": doc_path.suffix.lower()},
            )
            return doc_path
        except Exception as exc:
            self._add_stage(result, "validate", "failed", started_at=start, error=str(exc))
            result.reason = "validation_failed"
            raise

    def _hash_document(self, doc_path: Path, result: DocumentIngestionResult) -> str:
        return self._run_stage(
            result,
            "hash",
            lambda: compute_file_hash(doc_path),
            detail_builder=lambda digest: {"raw_file_hash": digest},
        )

    def _build_ingestion_document(
        self,
        doc_path: Path,
        source_file: str,
        raw_file_hash: str,
        result: DocumentIngestionResult,
    ) -> IngestionDocument:
        start = time.perf_counter()
        try:
            md_path = convert_document_to_markdown(doc_path, self.markdown_dir, overwrite=True)
            markdown_text = md_path.read_text(encoding="utf-8")

            markdown_text = self._enhance_images(markdown_text, md_path)
            if markdown_text != md_path.read_text(encoding="utf-8"):
                md_path.write_text(markdown_text, encoding="utf-8")

            markdown_hash = text_hash(markdown_text)
            document = IngestionDocument(
                doc_id=Path(source_file).stem,
                source_file=source_file,
                original_file=doc_path.name,
                source_path=doc_path,
                markdown_path=md_path,
                original_extension=doc_path.suffix.lower(),
                raw_file_hash=raw_file_hash,
                markdown_hash=markdown_hash,
                text=markdown_text,
                metadata={"converter": getattr(config, "DOCUMENT_CONVERTER", "markitdown")},
            )
            self._add_stage(
                result,
                "convert",
                "success",
                started_at=start,
                details={
                    "markdown_path": str(md_path),
                    "markdown_hash": markdown_hash,
                    "converter": document.metadata["converter"],
                },
            )
            return document
        except Exception as exc:
            self._add_stage(result, "convert", "failed", started_at=start, error=str(exc))
            result.reason = "conversion_failed"
            raise

    def _enhance_images(self, markdown_text: str, md_path: Path) -> str:
        """对 Markdown 中的本地图片引用执行 OCR/VLM 分析并嵌入结果。

        当 IMAGE_ANALYSIS_ENGINE 为 "none" 时直接返回原文。
        """
        engine = getattr(config, "IMAGE_ANALYSIS_ENGINE", "none")
        if engine == "none":
            return markdown_text

        from ingestion.image_describer import enhance_markdown_image_references, create_image_describer

        describe_fn = create_image_describer()
        if describe_fn is None:
            return markdown_text

        image_root = Path(config.DOCUMENT_IMAGE_DIR)
        enhanced = enhance_markdown_image_references(
            markdown_text,
            markdown_dir=md_path.parent,
            image_root=image_root,
            describe_image=describe_fn,
        )
        return enhanced

    def _clean_document(self, document: IngestionDocument, result: DocumentIngestionResult) -> dict[str, str]:
        cleaned = self._run_stage(
            result,
            "clean",
            lambda: self._clean_for_hash(document.text, document.source_file),
            detail_builder=lambda value: {
                "page_count": len(value.pages),
                "event_count": len(value.events),
            },
        )
        page_hashes = build_page_hashes(cleaned.pages)
        if not page_hashes:
            result.reason = "empty_cleaned_markdown"
            raise ValueError(f"Cleaned Markdown is empty: {document.source_file}")
        return page_hashes

    def _chunk_document(self, document: IngestionDocument, result: DocumentIngestionResult):
        parent_chunks, child_chunks = self._run_stage(
            result,
            "chunk",
            lambda: self.rag_system.chunker.create_chunks_single(document.markdown_path),
            detail_builder=lambda chunks: {
                "parent_count": len(chunks[0]),
                "child_count": len(chunks[1]),
            },
        )
        if not child_chunks:
            result.reason = "no_chunks"
            raise ValueError(f"No chunks were produced: {document.source_file}")
        return parent_chunks, child_chunks

    def _delete_old_index(self, source_file: str, result: DocumentIngestionResult) -> None:
        def delete_old():
            self.rag_system.vector_db.delete_by_source_file(source_file)
            self.rag_system.parent_store.delete_by_source_file(source_file)

        self._run_stage(result, "delete_old", delete_old, details={"source_file": source_file})

    def _write_vector_chunks(self, child_chunks: list, result: DocumentIngestionResult) -> None:
        def write_vectors():
            self.rag_system.vector_db.add_documents(child_chunks)

        self._run_stage(result, "write_vector", write_vectors, details={"child_count": len(child_chunks)})

    def _write_parent_chunks(self, parent_chunks: list, result: DocumentIngestionResult) -> None:
        self._run_stage(
            result,
            "write_parent",
            lambda: self.rag_system.parent_store.save_many(parent_chunks),
            details={"parent_count": len(parent_chunks)},
        )

    def _save_document_manifest(
        self,
        document: IngestionDocument,
        page_hashes: dict[str, str],
        parent_chunks: list,
        child_chunks: list,
        result: DocumentIngestionResult,
    ) -> None:
        start = time.perf_counter()
        try:
            previous = self.manifest.get_document(document.source_file) or {}
            record = build_document_record(
                doc_id=document.doc_id,
                source_file=document.source_file,
                markdown_path=document.markdown_path,
                markdown_text=document.text,
                page_hashes=page_hashes,
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                original_file=document.original_file,
                source_path=document.source_path,
                raw_file_hash=document.raw_file_hash,
                markdown_hash=document.markdown_hash,
                original_extension=document.original_extension,
                created_at=previous.get("created_at"),
            )
            self.manifest.set_document(document.source_file, record)
            self.manifest.save()
            self._add_stage(
                result,
                "manifest",
                "success",
                started_at=start,
                details={"source_file": document.source_file},
            )
        except Exception as exc:
            self._add_stage(result, "manifest", "failed", started_at=start, error=str(exc))
            result.reason = "manifest_failed"
            raise

    def _should_skip_indexing(self, source_file: str, raw_file_hash: str) -> tuple[bool, str]:
        if not getattr(config, "INGESTION_SKIP_UNCHANGED_FILES", True):
            return False, "skip_disabled"
        if not self.manifest.is_config_compatible():
            return False, "index_config_changed"
        document = self.manifest.get_document(source_file)
        if not document:
            return False, "new_document"
        if document.get("status") != "success":
            return False, "previous_index_not_successful"
        if document.get("raw_file_hash") != raw_file_hash:
            return False, "file_changed"
        markdown_path = Path(document.get("markdown_path") or self.markdown_dir / source_file)
        if not markdown_path.exists():
            return False, "markdown_missing"
        return True, "unchanged_file"

    def _clean_for_hash(self, markdown_text: str, source_file: str):
        """Mirror the chunker cleaning path so page hashes match indexed content."""
        if config.MARKDOWN_CLEANING_ENABLED:
            return clean_markdown_text(
                markdown_text,
                source_file=source_file,
                scan_lines=config.HEADER_FOOTER_SCAN_LINES,
                min_repeat_pages=config.MIN_REPEAT_PAGES,
                min_repeat_ratio=config.MIN_REPEAT_RATIO,
            )

        pages = parse_pages(markdown_text, source_file=source_file)
        for page in pages:
            page.cleaned_text = page.raw_text
            page.cleaned_lines = page.raw_lines
        return CleanedMarkdown(
            source_file=source_file,
            cleaned_text="\n\n".join(page.cleaned_text for page in pages if page.cleaned_text.strip()),
            pages=pages,
            events=[],
            candidates=[],
        )

    def _bind_courses_if_needed(
        self,
        result: DocumentIngestionResult,
        source_file: str,
        course_names: list[str],
    ) -> None:
        if not course_names:
            return

        start = time.perf_counter()
        try:
            before = set(self._course_names_for_source(source_file))
            course_ids = self.course_store.assign_document_to_courses(
                source_file=source_file,
                course_names=course_names,
                markdown_dir=self.markdown_dir,
            )
            after = set(self._course_names_for_source(source_file))
            result.course_updated = after != before
            self._add_stage(
                result,
                "course_bind",
                "success",
                started_at=start,
                details={"course_ids": course_ids, "updated": result.course_updated},
            )
        except Exception as exc:
            self._add_stage(result, "course_bind", "failed", started_at=start, error=str(exc))
            result.reason = "course_bind_failed"
            raise

    def _document_info(self, document: dict, course_names: list[str]) -> DocumentInfo:
        return DocumentInfo(
            source_file=document.get("source_file", ""),
            original_file=document.get("original_file") or document.get("source_file", ""),
            original_extension=document.get("original_extension", ""),
            raw_file_hash=document.get("raw_file_hash"),
            markdown_hash=document.get("markdown_hash") or document.get("document_hash"),
            parent_count=int(document.get("parent_count", len(document.get("parent_ids", [])))),
            child_count=int(document.get("child_count", len(document.get("child_ids", [])))),
            courses=course_names,
            updated_at=document.get("updated_at"),
            status=document.get("status", "unknown"),
        )

    def _course_names_for_source(self, source_file: str) -> list[str]:
        return sorted(
            course.get("name", "")
            for course in self.course_store.list_courses()
            if source_file in course.get("documents", [])
        )

    def _normalize_source_file(self, source_file: str) -> str:
        return Path(str(source_file)).with_suffix(".md").name

    def _mark_index_stages_skipped(self, result: DocumentIngestionResult, reason: str) -> None:
        for stage_name in ("convert", "clean", "chunk", "delete_old", "write_vector", "write_parent", "manifest"):
            self._add_stage(result, stage_name, "skipped", details={"reason": reason})

    def _update_manifest_result(self, source_file: str, result: DocumentIngestionResult) -> None:
        if self.manifest.update_document_result(source_file, self._manifest_result(result)):
            self.manifest.save()

    def _manifest_result(self, result: DocumentIngestionResult) -> dict:
        return {
            "status": result.status,
            "reason": result.reason,
            "indexed": result.indexed,
            "course_updated": result.course_updated,
            "parent_count": result.parent_count,
            "child_count": result.child_count,
            "stages": [
                {
                    "name": stage.name,
                    "status": stage.status,
                    "elapsed_ms": stage.elapsed_ms,
                    "details": stage.details,
                    "error": stage.error,
                }
                for stage in result.stages
            ],
            "error": result.error,
        }

    def _run_stage(
        self,
        result: DocumentIngestionResult,
        name: str,
        action,
        details: dict | None = None,
        detail_builder=None,
    ):
        start = time.perf_counter()
        try:
            value = action()
            stage_details = detail_builder(value) if detail_builder else details
            self._add_stage(result, name, "success", started_at=start, details=stage_details)
            return value
        except Exception as exc:
            self._add_stage(result, name, "failed", started_at=start, error=str(exc))
            raise

    def _add_stage(
        self,
        result: DocumentIngestionResult,
        name: str,
        status: str,
        started_at: float | None = None,
        details: dict | None = None,
        error: str | None = None,
    ) -> None:
        elapsed_ms = 0.0
        if started_at is not None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        result.stages.append(
            IngestionStageResult(
                name=name,
                status=status,
                elapsed_ms=elapsed_ms,
                details=details or {},
                error=error,
            )
        )

    def _write_stage_logs(self, result: DocumentIngestionResult) -> None:
        if not getattr(config, "INGESTION_STAGE_LOG_ENABLED", True):
            return
        log_dir = Path(config.INGESTION_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "ingestion.jsonl"
        with log_path.open("a", encoding="utf-8") as log_file:
            for stage in result.stages:
                log_file.write(json.dumps({
                    "source_path": result.source_path,
                    "source_file": result.source_file,
                    "status": result.status,
                    "reason": result.reason,
                    "stage": stage.name,
                    "stage_status": stage.status,
                    "elapsed_ms": stage.elapsed_ms,
                    "details": stage.details,
                    "error": stage.error,
                }, ensure_ascii=False) + "\n")

    def _delete_cleaning_outputs(self, source_file: str) -> None:
        stem = Path(source_file).stem
        for path in (
            Path(config.MARKDOWN_CLEANED_DIR) / f"{stem}.md",
            Path(config.MARKDOWN_CLEANING_LOG_DIR) / f"{stem}.jsonl",
            Path(config.MARKDOWN_CLEANING_DIFF_DIR) / f"{stem}.diff",
        ):
            self._unlink_if_exists(path)

    @staticmethod
    def _unlink_if_exists(path: Path) -> None:
        if path.exists():
            path.unlink()

    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)
        clear_directory_contents(Path(config.MARKDOWN_CLEANED_DIR))
        clear_directory_contents(Path(config.MARKDOWN_CLEANING_LOG_DIR))
        clear_directory_contents(Path(config.MARKDOWN_CLEANING_DIFF_DIR))
        clear_directory_contents(Path(config.DOCUMENT_IMAGE_DIR))
        clear_directory_contents(Path(config.INGESTION_LOG_DIR))

        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.clear_store()
        self.manifest = IndexManifest()
        self.manifest.save()
        self.course_store.clear()
