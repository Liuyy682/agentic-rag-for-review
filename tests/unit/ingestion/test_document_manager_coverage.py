import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


import config
from ingestion.chunking import DocumentChunker
from ingestion.document_manager import DocumentManager
from ingestion.models import DocumentIngestionResult, IngestionDocument


class FakeVectorDb:
    def __init__(self):
        self.documents = []
        self.deleted = []
        self.cleared = False

    def add_documents(self, documents):
        self.documents.extend(documents)
        return [d.metadata.get("chunk_id") for d in documents]

    def delete_by_source_file(self, source_file):
        self.deleted.append(source_file)

    def clear_store(self):
        self.cleared = True
        self.documents = []


class FakeParentStore:
    def __init__(self):
        self.saved = []
        self.deleted = []
        self.cleared = False

    def save_many(self, parents):
        self.saved.extend(parents)

    def delete_by_source_file(self, source_file):
        self.deleted.append(source_file)

    def clear_store(self):
        self.cleared = True
        self.saved = []


class FakeRagSystem:
    def __init__(self):
        self.vector_db = FakeVectorDb()
        self.parent_store = FakeParentStore()
        self.chunker = DocumentChunker()


class ConfigPatch:
    def __init__(self, **values):
        self.values = values
        self.old = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.old[name] = getattr(config, name)
            setattr(config, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in self.old.items():
            setattr(config, name, value)


def build_test_config(temp_path: Path, **overrides) -> dict:
    base = {
        "MARKDOWN_DIR": str(temp_path / "markdown"),
        "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
        "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
        "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
        "DOCUMENT_IMAGE_DIR": str(temp_path / "images"),
        "INGESTION_LOG_DIR": str(temp_path / "ingestion_logs"),
        "INDEX_STATE_DIR": str(temp_path / "index_state"),
        "COURSE_STRUCTURE_PATH": str(temp_path / "course_structure.json"),
        "MARKDOWN_CLEANING_ENABLED": False,
        "INGESTION_SKIP_UNCHANGED_FILES": True,
        "INGESTION_STAGE_LOG_ENABLED": False,
        "IMAGE_ANALYSIS_ENGINE": "none",
        "MIN_PARENT_SIZE": 1,
        "MAX_PARENT_SIZE": 2000,
        "CHILD_CHUNK_SIZE": 1000,
        "CHILD_CHUNK_OVERLAP": 0,
    }
    base.update(overrides)
    return base


MARKDOWN_BODY = "# Notes\n\nThis is the body of the notes document.\n"


class QueryMethodsTest(unittest.TestCase):
    def _manager(self, temp_path):
        return DocumentManager(FakeRagSystem())

    def test_add_documents_detailed_empty_returns_empty_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ConfigPatch(**build_test_config(Path(temp_dir))):
                manager = DocumentManager(FakeRagSystem())
                summary = manager.add_documents_detailed([])
                self.assertEqual((summary.added, summary.skipped, summary.failed), (0, 0, 0))
                self.assertEqual(summary.documents, [])

    def test_add_documents_tuple_and_progress_callback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())

                progress = []
                added, failed = manager.add_documents(
                    str(source),
                    progress_callback=lambda frac, msg: progress.append((frac, msg)),
                )

                self.assertEqual((added, failed), (1, 0))
                self.assertEqual(len(progress), 1)
                self.assertEqual(progress[0][0], 1.0)
                self.assertIn("notes.md", progress[0][1])

    def test_get_markdown_files_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager = DocumentManager(FakeRagSystem())
                # Remove the markdown dir created by __init__.
                shutil.rmtree(manager.markdown_dir)
                self.assertEqual(manager.get_markdown_files(), [])

    def test_get_markdown_files_lists_sorted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager = DocumentManager(FakeRagSystem())
                (manager.markdown_dir / "b.md").write_text("b", encoding="utf-8")
                (manager.markdown_dir / "a.md").write_text("a", encoding="utf-8")
                self.assertEqual(manager.get_markdown_files(), ["a.md", "b.md"])

    def test_list_documents_and_collection_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                manager.add_documents_detailed([str(source)], course_names="Course A")

                docs = manager.list_documents()
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0].source_file, "notes.md")
                self.assertEqual(docs[0].courses, ["Course A"])

                stats = manager.get_collection_stats()
                self.assertEqual(stats.document_count, 1)
                self.assertGreaterEqual(stats.parent_count, 1)
                self.assertGreaterEqual(stats.child_count, 1)
                self.assertEqual(stats.course_count, 1)

    def test_get_document_detail_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with ConfigPatch(**build_test_config(Path(temp_dir))):
                manager = DocumentManager(FakeRagSystem())
                self.assertIsNone(manager.get_document_detail("nope.md"))

    def test_course_passthrough_methods(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                manager.add_documents_detailed([str(source)], course_names="Course A")

                self.assertIn("Course A", manager.get_course_list())
                self.assertEqual(manager.get_course_choices(), ["Course A"])
                self.assertTrue(manager.rename_course("Course A", "Course B"))
                self.assertEqual(manager.get_course_choices(), ["Course B"])
                # rename_section returns False for a section that does not exist.
                self.assertFalse(manager.rename_section("Course B", "ghost", "new"))


class DeleteAndClearTest(unittest.TestCase):
    def _seed(self, temp_path):
        source = temp_path / "notes.md"
        source.write_text(MARKDOWN_BODY, encoding="utf-8")
        rag = FakeRagSystem()
        manager = DocumentManager(rag)
        manager.add_documents_detailed([str(source)], course_names="Course A")
        return manager, rag

    def test_delete_document_records_all_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, rag = self._seed(temp_path)

                # Make each storage/manifest/course operation fail.
                rag.vector_db.delete_by_source_file = lambda sf: (_ for _ in ()).throw(RuntimeError("vec boom"))
                rag.parent_store.delete_by_source_file = lambda sf: (_ for _ in ()).throw(RuntimeError("par boom"))
                manager.manifest.remove_document = lambda sf: (_ for _ in ()).throw(RuntimeError("man boom"))
                manager.course_store.remove_document = lambda sf, markdown_dir=None: (_ for _ in ()).throw(RuntimeError("course boom"))

                result = manager.delete_document("notes.md")

                self.assertFalse(result.success)
                joined = " ".join(result.errors)
                self.assertIn("vector delete failed", joined)
                self.assertIn("parent delete failed", joined)
                self.assertIn("manifest delete failed", joined)
                self.assertIn("course update failed", joined)

    def test_delete_document_markdown_delete_error_recorded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, rag = self._seed(temp_path)

                with patch.object(DocumentManager, "_unlink_if_exists", side_effect=RuntimeError("unlink boom")):
                    result = manager.delete_document("notes.md")

                joined = " ".join(result.errors)
                self.assertIn("markdown delete failed", joined)
                self.assertIn("cleaning output delete failed", joined)
                self.assertFalse(result.success)

    def test_delete_document_unknown_uses_default_markdown_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager = DocumentManager(FakeRagSystem())
                # No manifest entry → markdown_path falls back to markdown_dir / source_file.
                result = manager.delete_document("ghost.md")
                self.assertTrue(result.success)
                self.assertTrue(result.markdown_deleted)

    def test_clear_all_resets_stores_and_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, rag = self._seed(temp_path)
                (manager.markdown_dir / "extra.md").write_text("x", encoding="utf-8")

                manager.clear_all()

                self.assertTrue(rag.parent_store.cleared)
                self.assertTrue(rag.vector_db.cleared)
                self.assertEqual(manager.get_markdown_files(), [])
                self.assertEqual(manager.get_course_choices(), [])
                # NOTE (suspected business bug): clear_all() rebuilds the manifest via
                # IndexManifest(), which reloads the persisted manifest file because
                # INDEX_STATE_DIR is never cleared. As a result document records survive
                # clear_all(). Asserting the actual current behavior here.
                self.assertEqual(len(manager.list_documents()), 1)


class StageLogTest(unittest.TestCase):
    def test_stage_logs_written_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path, INGESTION_STAGE_LOG_ENABLED=True)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                manager.add_documents_detailed([str(source)])

                log_path = Path(config.INGESTION_LOG_DIR) / "ingestion.jsonl"
                self.assertTrue(log_path.exists())
                self.assertIn("notes.md", log_path.read_text(encoding="utf-8"))


class DocumentDetailTest(unittest.TestCase):
    def test_get_document_detail_for_indexed_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                manager.add_documents_detailed([str(source)], course_names="Course A")

                detail = manager.get_document_detail("notes.md")
                self.assertIsNotNone(detail)
                self.assertEqual(detail.info.source_file, "notes.md")
                self.assertEqual(detail.course_names, ["Course A"])
                self.assertTrue(detail.parent_ids)
                self.assertTrue(detail.child_ids)
                self.assertTrue(detail.markdown_path)


class SkipIndexingTest(unittest.TestCase):
    def _index_once(self, temp_path):
        source = temp_path / "notes.md"
        source.write_text(MARKDOWN_BODY, encoding="utf-8")
        manager = DocumentManager(FakeRagSystem())
        manager.add_documents_detailed([str(source)])
        return manager, source

    def test_skip_disabled_reindexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path, INGESTION_SKIP_UNCHANGED_FILES=False)):
                manager, source = self._index_once(temp_path)
                summary = manager.add_documents_detailed([str(source)])
                # Skip disabled → indexed again, not skipped.
                self.assertEqual(summary.added, 1)

    def test_unchanged_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, source = self._index_once(temp_path)
                summary = manager.add_documents_detailed([str(source)])
                self.assertEqual(summary.skipped, 1)
                result = summary.documents[0]
                self.assertEqual(result.reason, "unchanged_file")
                statuses = {s.name: s.status for s in result.stages}
                self.assertEqual(statuses["convert"], "skipped")

    def test_file_changed_reindexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, source = self._index_once(temp_path)
                source.write_text(MARKDOWN_BODY + "\nmore content\n", encoding="utf-8")
                summary = manager.add_documents_detailed([str(source)])
                self.assertEqual(summary.added, 1)

    def test_markdown_missing_reindexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, source = self._index_once(temp_path)
                # Remove the generated markdown so skip check forces a reindex.
                (manager.markdown_dir / "notes.md").unlink()
                # Use the stored raw hash so the file_changed check passes and we
                # reach the markdown_missing branch.
                stored_hash = manager.manifest.get_document("notes.md")["raw_file_hash"]
                should_skip, reason = manager._should_skip_indexing("notes.md", stored_hash)
                self.assertFalse(should_skip)
                self.assertEqual(reason, "markdown_missing")

    def test_previous_index_not_successful_reindexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, source = self._index_once(temp_path)
                doc = manager.manifest.get_document("notes.md")
                stored_hash = doc["raw_file_hash"]
                doc["status"] = "failed"
                should_skip, reason = manager._should_skip_indexing("notes.md", stored_hash)
                self.assertFalse(should_skip)
                self.assertEqual(reason, "previous_index_not_successful")

    def test_index_config_changed_reindexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager, source = self._index_once(temp_path)
                with patch.object(manager.manifest, "is_config_compatible", return_value=False):
                    should_skip, reason = manager._should_skip_indexing("notes.md", "x")
                self.assertFalse(should_skip)
                self.assertEqual(reason, "index_config_changed")


class FailureBranchTest(unittest.TestCase):
    def _source(self, temp_path):
        source = temp_path / "notes.md"
        source.write_text(MARKDOWN_BODY, encoding="utf-8")
        return source

    def test_validate_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                manager = DocumentManager(FakeRagSystem())
                # Supported suffix but file does not exist → "not a local file".
                summary = manager.add_documents_detailed([str(temp_path / "ghost.md")])
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "validation_failed")

    def test_unsupported_extension_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                bad = temp_path / "data.zip"
                bad.write_bytes(b"x")
                manager = DocumentManager(FakeRagSystem())
                summary = manager.add_documents_detailed([str(bad)])
                self.assertEqual(summary.failed, 1)
                self.assertEqual(summary.documents[0].reason, "validation_failed")

    def test_conversion_failure_marks_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = self._source(temp_path)
                manager = DocumentManager(FakeRagSystem())
                with patch(
                    "ingestion.document_manager.convert_document_to_markdown",
                    side_effect=RuntimeError("convert boom"),
                ):
                    summary = manager.add_documents_detailed([str(source)])
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "conversion_failed")
                self.assertEqual(result.error, "convert boom")
                statuses = {s.name: s.status for s in result.stages}
                self.assertEqual(statuses["convert"], "failed")

    def test_empty_cleaned_markdown_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = self._source(temp_path)
                manager = DocumentManager(FakeRagSystem())
                from ingestion.cleaning import CleanedMarkdown
                empty = CleanedMarkdown(
                    source_file="notes.md", cleaned_text="", pages=[], events=[], candidates=[]
                )
                with patch.object(DocumentManager, "_clean_for_hash", return_value=empty):
                    summary = manager.add_documents_detailed([str(source)])
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "empty_cleaned_markdown")

    def test_no_chunks_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = self._source(temp_path)
                rag = FakeRagSystem()
                rag.chunker.create_chunks_single = lambda path: ([], [])
                manager = DocumentManager(rag)
                summary = manager.add_documents_detailed([str(source)])
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "no_chunks")
                statuses = {s.name: s.status for s in result.stages}
                # The chunk stage itself succeeds (returns an empty tuple); the
                # no_chunks failure is raised afterward, so no failed stage is added.
                self.assertEqual(statuses["chunk"], "success")

    def test_manifest_save_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = self._source(temp_path)
                manager = DocumentManager(FakeRagSystem())
                with patch.object(
                    manager.manifest, "set_document", side_effect=RuntimeError("manifest boom")
                ):
                    summary = manager.add_documents_detailed([str(source)])
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "manifest_failed")

    def test_course_bind_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = self._source(temp_path)
                manager = DocumentManager(FakeRagSystem())
                manager.course_store.assign_document_to_courses = (
                    lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bind boom"))
                )
                summary = manager.add_documents_detailed([str(source)], course_names="Course A")
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "course_bind_failed")


class CleaningEnabledAndImageTest(unittest.TestCase):
    def test_clean_for_hash_with_cleaning_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path, MARKDOWN_CLEANING_ENABLED=True)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                summary = manager.add_documents_detailed([str(source)])
                self.assertEqual(summary.added, 1)

    def test_enhance_images_rewrites_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path, IMAGE_ANALYSIS_ENGINE="paddleocr")):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())

                # Stub the image describer module so the enhance path runs without OCR.
                def fake_enhance(markdown_text, markdown_dir, image_root, describe_image):
                    return markdown_text + "\n\n<!-- image analysis -->\n"

                with patch(
                    "ingestion.image_describer.create_image_describer",
                    return_value=lambda *a, **k: "desc",
                ), patch(
                    "ingestion.image_describer.enhance_markdown_image_references",
                    side_effect=fake_enhance,
                ):
                    summary = manager.add_documents_detailed([str(source)])

                self.assertEqual(summary.added, 1)
                rewritten = (manager.markdown_dir / "notes.md").read_text(encoding="utf-8")
                self.assertIn("image analysis", rewritten)

    def test_enhance_images_no_describer_returns_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path, IMAGE_ANALYSIS_ENGINE="paddleocr")):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                manager = DocumentManager(FakeRagSystem())
                with patch(
                    "ingestion.image_describer.create_image_describer",
                    return_value=None,
                ):
                    summary = manager.add_documents_detailed([str(source)])
                self.assertEqual(summary.added, 1)


class RunStageFailureTest(unittest.TestCase):
    def test_write_vector_failure_propagates_through_run_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text(MARKDOWN_BODY, encoding="utf-8")
                rag = FakeRagSystem()
                rag.vector_db.add_documents = lambda docs: (_ for _ in ()).throw(
                    RuntimeError("write boom")
                )
                manager = DocumentManager(rag)

                summary = manager.add_documents_detailed([str(source)])

                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.error, "write boom")
                statuses = {s.name: s.status for s in result.stages}
                # _run_stage records the write_vector stage as failed before re-raising.
                self.assertEqual(statuses["write_vector"], "failed")


if __name__ == "__main__":
    unittest.main()
