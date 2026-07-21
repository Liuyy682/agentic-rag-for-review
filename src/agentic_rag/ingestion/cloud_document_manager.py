from __future__ import annotations

import mimetypes
import tempfile
import uuid
from pathlib import Path

from agentic_rag import config
from agentic_rag.ingestion.conversion import convert_document_to_markdown, is_supported_document
from agentic_rag.ingestion.file_integrity import compute_file_hash
from agentic_rag.ingestion.index_manifest import current_index_config, text_hash
from agentic_rag.ingestion.models import (
    AddDocumentsResult,
    DeleteDocumentResult,
    DocumentDetail,
    DocumentInfo,
    DocumentIngestionResult,
    KnowledgeBaseStats,
)
from agentic_rag.storage.metadata_repository import PgCourseStructureStore, PgDocumentRepository
from agentic_rag.storage.object_storage import MinioObjectStorage, document_object_keys


class CloudDocumentManager:
    """Document ingestion whose durable state lives in MinIO and PostgreSQL."""

    def __init__(
        self,
        rag_system,
        *,
        object_storage: MinioObjectStorage,
        repository: PgDocumentRepository,
        course_store: PgCourseStructureStore,
    ):
        self.rag_system = rag_system
        self.object_storage = object_storage
        self.repository = repository
        self.course_store = course_store

    def add_documents(self, document_paths, progress_callback=None, course_names=None):
        summary = self.add_documents_detailed(
            document_paths,
            progress_callback=progress_callback,
            course_names=course_names,
        )
        return summary.added, summary.skipped + summary.failed

    def add_documents_detailed(
        self,
        document_paths,
        progress_callback=None,
        course_names=None,
    ) -> AddDocumentsResult:
        if not document_paths:
            return AddDocumentsResult()
        paths = [document_paths] if isinstance(document_paths, str) else list(document_paths)
        paths = [path for path in paths if path]
        results = []
        for index, document_path in enumerate(paths):
            if progress_callback:
                progress_callback((index + 1) / len(paths), f"Processing {Path(document_path).name}")
            results.append(self._process_document(Path(document_path), course_names))
        return AddDocumentsResult(
            added=sum(result.status == "added" for result in results),
            skipped=0,
            failed=sum(result.status == "failed" for result in results),
            course_updated=sum(bool(result.course_updated) for result in results),
            documents=results,
        )

    def _process_document(self, upload_path: Path, course_names) -> DocumentIngestionResult:
        original_file = Path(upload_path.name).name or "upload"
        source_file = original_file
        document_id = str(uuid.uuid4())
        result = DocumentIngestionResult(
            source_path=str(upload_path),
            source_file=source_file,
            original_file=original_file,
            document_id=document_id,
        )
        keys = document_object_keys(document_id, original_file)
        derived_keys: list[str] = []
        metadata_created = False

        try:
            if not upload_path.is_file():
                raise ValueError(f"Document is not a local file: {upload_path}")
            if not is_supported_document(upload_path):
                raise ValueError(f"Unsupported document type: {upload_path.suffix}")

            content_type = mimetypes.guess_type(original_file)[0] or "application/octet-stream"
            self.object_storage.upload_file(keys["raw"], upload_path, content_type)
            try:
                self.repository.create_processing(
                    document_id,
                    original_file,
                    source_file,
                    keys["raw"],
                )
                metadata_created = True
            except Exception:
                self.object_storage.delete_object(keys["raw"])
                raise

            with tempfile.TemporaryDirectory(prefix=f"rag_ingest_{document_id}_") as temp_dir:
                workspace = Path(temp_dir)
                local_source = workspace / f"{document_id}{upload_path.suffix.lower()}"
                markdown_dir = workspace / "markdown"
                image_dir = workspace / "images"
                self.object_storage.download_file(keys["raw"], local_source)

                raw_file_hash = compute_file_hash(local_source)
                result.raw_file_hash = raw_file_hash
                markdown_path = convert_document_to_markdown(
                    local_source,
                    markdown_dir,
                    overwrite=True,
                    image_output_dir=image_dir,
                )
                markdown_text = markdown_path.read_text(encoding="utf-8")
                markdown_text = self._enhance_images(markdown_text, markdown_path, image_dir)
                markdown_path.write_text(markdown_text, encoding="utf-8")
                markdown_hash = text_hash(markdown_text)

                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(markdown_path)
                self._rewrite_chunk_identity(
                    document_id,
                    source_file,
                    parent_chunks,
                    child_chunks,
                )
                if not child_chunks:
                    raise ValueError(f"No chunks were produced: {source_file}")

                self._delete_index(document_id)
                self.rag_system.vector_db.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)

                self.object_storage.upload_file(keys["markdown"], markdown_path, "text/markdown")
                derived_keys.append(keys["markdown"])
                image_object_keys = self._upload_images(keys["images_prefix"], image_dir)
                derived_keys.extend(image_object_keys)

                if course_names:
                    self.course_store.assign_document_to_courses(
                        document_id,
                        course_names,
                        markdown_path,
                        source_file,
                    )
                    result.course_updated = True

                result.status = "added"
                result.reason = "indexed"
                result.indexed = True
                result.parent_count = len(parent_chunks)
                result.child_count = len(child_chunks)
                self.repository.mark_success(
                    document_id,
                    raw_file_hash=raw_file_hash,
                    markdown_hash=markdown_hash,
                    markdown_object_key=keys["markdown"],
                    image_object_keys=image_object_keys,
                    parent_count=len(parent_chunks),
                    child_count=len(child_chunks),
                    index_config=current_index_config(),
                    last_result=result.to_dict(),
                )
                self._cleanup_local_cleaning(document_id)
                return result
        except Exception as exc:
            result.status = "failed"
            result.reason = result.reason or "processing_failed"
            result.error = str(exc)
            try:
                self._delete_index(document_id)
                self.course_store.remove_document(document_id)
                self.object_storage.delete_objects(derived_keys)
            except Exception as cleanup_exc:
                result.error = f"{result.error}; cleanup failed: {cleanup_exc}"
            if metadata_created:
                self.repository.mark_failed(document_id, result.error, result.to_dict())
            self._cleanup_local_cleaning(document_id)
            return result

    def _enhance_images(self, markdown_text: str, markdown_path: Path, image_root: Path) -> str:
        if getattr(config, "IMAGE_ANALYSIS_ENGINE", "none") == "none":
            return markdown_text
        from agentic_rag.ingestion.image_describer import (
            create_image_describer,
            enhance_markdown_image_references,
        )

        describe_image = create_image_describer()
        if describe_image is None:
            return markdown_text
        return enhance_markdown_image_references(
            markdown_text,
            markdown_dir=markdown_path.parent,
            image_root=image_root,
            describe_image=describe_image,
        )

    @staticmethod
    def _rewrite_chunk_identity(document_id, source_file, parent_chunks, child_chunks) -> None:
        for _, document in parent_chunks:
            document.metadata.update(
                {"doc_id": document_id, "source": source_file, "source_file": source_file}
            )
        for document in child_chunks:
            document.metadata.update(
                {"doc_id": document_id, "source": source_file, "source_file": source_file}
            )

    def _upload_images(self, prefix: str, image_dir: Path) -> list[str]:
        object_keys = []
        if not image_dir.exists():
            return object_keys
        for image_path in sorted(path for path in image_dir.rglob("*") if path.is_file()):
            relative = image_path.name
            object_key = f"{prefix}/{relative}"
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self.object_storage.upload_file(object_key, image_path, content_type)
            object_keys.append(object_key)
        return object_keys

    def _delete_index(self, document_id: str) -> None:
        delete_vectors = getattr(self.rag_system.vector_db, "delete_by_document_id", None)
        delete_parents = getattr(self.rag_system.parent_store, "delete_by_document_id", None)
        if delete_vectors:
            delete_vectors(document_id)
        if delete_parents:
            delete_parents(document_id)

    @staticmethod
    def _cleanup_local_cleaning(document_id: str) -> None:
        for path in (
            Path(config.MARKDOWN_CLEANED_DIR) / f"{document_id}.md",
            Path(config.MARKDOWN_CLEANING_LOG_DIR) / f"{document_id}.jsonl",
            Path(config.MARKDOWN_CLEANING_DIFF_DIR) / f"{document_id}.diff",
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def get_markdown_files(self) -> list[str]:
        return [row["original_file"] for row in self.repository.list_documents(["success"])]

    def list_documents(self) -> list[DocumentInfo]:
        documents = []
        for row in self.repository.list_documents():
            document_id = str(row["document_id"])
            documents.append(
                DocumentInfo(
                    source_file=row["source_file"],
                    original_file=row["original_file"],
                    original_extension=row.get("original_extension") or "",
                    raw_file_hash=row.get("raw_file_hash"),
                    markdown_hash=row.get("markdown_hash"),
                    parent_count=int(row.get("parent_count") or 0),
                    child_count=int(row.get("child_count") or 0),
                    courses=self.repository.course_names_for_document(document_id),
                    updated_at=row.get("updated_at").isoformat() if row.get("updated_at") else None,
                    status=row["status"],
                    document_id=document_id,
                )
            )
        return documents

    def get_document_detail(self, identifier: str) -> DocumentDetail | None:
        row = self.repository.get_document(identifier)
        if not row:
            return None
        document_id = str(row["document_id"])
        info = next(item for item in self.list_documents() if item.document_id == document_id)
        last_result = row.get("last_result") or {}
        return DocumentDetail(
            info=info,
            markdown_path=row.get("markdown_object_key"),
            parent_ids=[],
            child_ids=[],
            stage_stats=list(last_result.get("stages", [])),
            last_error=row.get("last_error"),
            course_names=info.courses,
        )

    def get_collection_stats(self) -> KnowledgeBaseStats:
        documents = self.repository.list_documents(["success"])
        return KnowledgeBaseStats(
            document_count=len(documents),
            parent_count=sum(int(row.get("parent_count") or 0) for row in documents),
            child_count=sum(int(row.get("child_count") or 0) for row in documents),
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

    def delete_document(self, identifier: str) -> DeleteDocumentResult:
        row = self.repository.get_document(identifier)
        if not row:
            return DeleteDocumentResult(success=True, source_file=identifier)
        document_id = str(row["document_id"])
        result = DeleteDocumentResult(success=False, source_file=row["source_file"])
        self.repository.set_status(document_id, "deleting")
        try:
            self._delete_index(document_id)
            result.vector_deleted = True
            result.parent_deleted = True
            result.course_updated = bool(self.course_store.remove_document(document_id))
            object_keys = [
                row.get("raw_object_key"),
                row.get("markdown_object_key"),
                *list(row.get("image_object_keys") or []),
            ]
            self.object_storage.delete_objects(object_keys)
            result.markdown_deleted = True
            result.cleaning_outputs_deleted = True
            self.repository.delete_document(document_id)
            result.manifest_deleted = True
            result.success = True
        except Exception as exc:
            result.errors.append(str(exc))
            self.repository.set_status(document_id, "delete_failed", str(exc))
        return result

    def clear_all(self):
        errors = []
        for row in self.repository.list_documents():
            result = self.delete_document(str(row["document_id"]))
            errors.extend(result.errors)
        if errors:
            raise RuntimeError("; ".join(errors))
        self.rag_system.parent_store.clear_store()
        self.rag_system.vector_db.clear_store()
        self.course_store.clear()
