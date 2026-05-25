import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from ingestion.document_manager import DocumentManager
from ingestion.index_manifest import IndexManifest
from ingestion.chunking import DocumentChunker


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.add_calls = []

    def add_documents(self, documents):
        self.add_calls.append(list(documents))
        self.documents.extend(documents)


class FakeVectorDb:
    def __init__(self):
        self.collection = FakeCollection()
        self.deleted_source_files = []

    def add_documents(self, documents):
        return self.collection.add_documents(documents)

    def delete_by_source_file(self, source_file):
        self.deleted_source_files.append(source_file)
        self.collection.documents = [
            doc for doc in self.collection.documents
            if doc.metadata.get("source_file") != source_file
        ]

    def clear_store(self):
        self.collection.documents = []


class FakeParentStore:
    def __init__(self):
        self.saved = []
        self.deleted_source_files = []
        self.cleared = False

    def save_many(self, parents):
        self.saved.extend(parents)

    def delete_by_source_file(self, source_file):
        self.deleted_source_files.append(source_file)
        self.saved = [
            (parent_id, doc)
            for parent_id, doc in self.saved
            if doc.metadata.get("source_file") != source_file
        ]

    def clear_store(self):
        self.cleared = True
        self.saved = []


class FakeRagSystem:
    def __init__(self, state_dir=None):
        self.vector_db = FakeVectorDb()
        self.parent_store = FakeParentStore()
        self.chunker = DocumentChunker()


class ConfigPatch:
    def __init__(self, **values):
        self.values = values
        self.old_values = {}

    def __enter__(self):
        for name, value in self.values.items():
            self.old_values[name] = getattr(config, name)
            setattr(config, name, value)

    def __exit__(self, exc_type, exc, tb):
        for name, value in self.old_values.items():
            setattr(config, name, value)


def build_test_config(temp_path: Path) -> dict:
    return {
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
        "MIN_PARENT_SIZE": 1,
        "MAX_PARENT_SIZE": 2000,
        "CHILD_CHUNK_SIZE": 1000,
        "CHILD_CHUNK_OVERLAP": 0,
    }


class TestIngestionPipelineResult(unittest.TestCase):
    def test_unchanged_same_source_skips_conversion_but_updates_new_course(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text("# Notes\n\nBody", encoding="utf-8")
                rag_system = FakeRagSystem(config.INDEX_STATE_DIR)
                manager = DocumentManager(rag_system)

                first = manager.add_documents_detailed([str(source)], course_names="Course A")

                self.assertEqual((first.added, first.skipped, first.failed), (1, 0, 0))
                self.assertTrue(first.documents[0].indexed)
                self.assertEqual(first.documents[0].reason, "indexed")
                self.assertTrue(first.documents[0].course_updated)

                with patch("ingestion.document_manager.convert_document_to_markdown") as convert:
                    convert.side_effect = AssertionError("conversion should be skipped")
                    second = manager.add_documents_detailed([str(source)], course_names="Course B")

                self.assertEqual((second.added, second.skipped, second.failed), (0, 1, 0))
                self.assertEqual(second.course_updated, 1)
                result = second.documents[0]
                self.assertEqual(result.status, "skipped")
                self.assertEqual(result.reason, "unchanged_file")
                self.assertFalse(result.indexed)
                self.assertGreater(result.parent_count, 0)
                self.assertGreater(result.child_count, 0)
                self.assertEqual(
                    manager.course_store.source_files_for_course("Course B"),
                    ["notes.md"],
                )

                stage_statuses = {stage.name: stage.status for stage in result.stages}
                self.assertEqual(stage_statuses["convert"], "skipped")
                self.assertEqual(stage_statuses["chunk"], "skipped")
                self.assertEqual(rag_system.vector_db.deleted_source_files, ["notes.md"])

                detail = manager.get_document_detail("notes.md")
                self.assertIsNotNone(detail)
                self.assertEqual(detail.info.original_file, "notes.md")
                self.assertEqual(set(detail.course_names), {"Course A", "Course B"})
                self.assertTrue(detail.info.raw_file_hash)

    def test_same_hash_different_source_files_are_indexed_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                first = temp_path / "first.md"
                second = temp_path / "second.md"
                first.write_text("# Same\n\nBody", encoding="utf-8")
                second.write_text("# Same\n\nBody", encoding="utf-8")
                manager = DocumentManager(FakeRagSystem(config.INDEX_STATE_DIR))

                summary = manager.add_documents_detailed([str(first), str(second)])

                self.assertEqual((summary.added, summary.skipped, summary.failed), (2, 0, 0))
                manifest = IndexManifest()
                self.assertIsNotNone(manifest.get_document("first.md"))
                self.assertIsNotNone(manifest.get_document("second.md"))
                self.assertEqual(
                    manifest.get_document("first.md")["raw_file_hash"],
                    manifest.get_document("second.md")["raw_file_hash"],
                )

    def test_unsupported_document_returns_failed_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "archive.zip"
                source.write_bytes(b"fake")
                manager = DocumentManager(FakeRagSystem(config.INDEX_STATE_DIR))

                summary = manager.add_documents_detailed([str(source)])

                self.assertEqual((summary.added, summary.skipped, summary.failed), (0, 0, 1))
                result = summary.documents[0]
                self.assertEqual(result.status, "failed")
                self.assertEqual(result.reason, "validation_failed")
                self.assertEqual(result.stages[0].name, "validate")
                self.assertEqual(result.stages[0].status, "failed")

    def test_delete_document_removes_index_artifacts_and_course_binding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with ConfigPatch(**build_test_config(temp_path)):
                source = temp_path / "notes.md"
                source.write_text("# Notes\n\nBody", encoding="utf-8")
                rag_system = FakeRagSystem(config.INDEX_STATE_DIR)
                manager = DocumentManager(rag_system)
                manager.add_documents_detailed([str(source)], course_names="Course A")

                cleaned = Path(config.MARKDOWN_CLEANED_DIR) / "notes.md"
                log = Path(config.MARKDOWN_CLEANING_LOG_DIR) / "notes.jsonl"
                diff = Path(config.MARKDOWN_CLEANING_DIFF_DIR) / "notes.diff"
                for path in (cleaned, log, diff):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("artifact", encoding="utf-8")

                result = manager.delete_document("notes.md")

                self.assertTrue(result.success)
                self.assertTrue(result.vector_deleted)
                self.assertTrue(result.parent_deleted)
                self.assertTrue(result.manifest_deleted)
                self.assertTrue(result.markdown_deleted)
                self.assertTrue(result.cleaning_outputs_deleted)
                self.assertTrue(result.course_updated)
                self.assertEqual(rag_system.vector_db.collection.documents, [])
                self.assertEqual(rag_system.parent_store.deleted_source_files, ["notes.md", "notes.md"])
                self.assertIsNone(IndexManifest().get_document("notes.md"))
                self.assertFalse((Path(config.MARKDOWN_DIR) / "notes.md").exists())
                self.assertFalse(cleaned.exists())
                self.assertFalse(log.exists())
                self.assertFalse(diff.exists())
                self.assertEqual(manager.course_store.source_files_for_course("Course A"), [])


if __name__ == "__main__":
    unittest.main()
