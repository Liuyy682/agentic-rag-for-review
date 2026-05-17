import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from ingestion.document_manager import DocumentManager
from ingestion.cleaning import clean_markdown_text
from ingestion.index_manifest import (
    IndexManifest,
    build_page_hashes,
    current_index_config,
)
from storage.parent_store import ParentStoreManager
from ingestion.chunking import DocumentChunker


def paged_markdown(page_3_text="page three"):
    return f"""# Page 1
page one
--- end of page.page_number=1 ---
# Page 2
page two
--- end of page.page_number=2 ---
# Page 3
{page_3_text}
--- end of page.page_number=3 ---
# Page 4
page four
--- end of page.page_number=4 ---
# Page 5
page five
--- end of page.page_number=5 ---
"""


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
        self.deleted_parent_ids = []
        self.deleted_source_files = []

    def get_collection(self, collection_name):
        return self.collection

    def delete_by_parent_ids(self, collection_name, parent_ids):
        self.deleted_parent_ids.extend(parent_ids)
        parent_ids = set(parent_ids)
        self.collection.documents = [
            doc for doc in self.collection.documents
            if doc.metadata.get("parent_id") not in parent_ids
        ]

    def delete_by_source_file(self, collection_name, source_file):
        self.deleted_source_files.append(source_file)
        self.collection.documents = [
            doc for doc in self.collection.documents
            if doc.metadata.get("source_file") != source_file
        ]

    def delete_collection(self, collection_name):
        self.collection.documents = []

    def create_collection(self, collection_name):
        pass


class FakeRagSystem:
    def __init__(self, parent_store_path):
        self.collection_name = "test_collection"
        self.vector_db = FakeVectorDb()
        self.parent_store = ParentStoreManager(parent_store_path)
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


class TestPageIncrementalIndex(unittest.TestCase):
    def test_manifest_detects_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = IndexManifest(Path(temp_dir) / "index_manifest.json")
            manifest.data["documents"]["doc.md"] = {"pages": {}}
            manifest.data["index_config"] = current_index_config()
            manifest.save()

            loaded = IndexManifest(Path(temp_dir) / "index_manifest.json")
            self.assertTrue(loaded.is_config_compatible())

            loaded.data["index_config"]["chunker_config_hash"] = "different"
            self.assertFalse(loaded.is_config_compatible())

    def test_parent_store_delete_many_removes_only_selected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParentStoreManager(temp_dir)
            store.save("p1", "one", {"source_file": "a.md"})
            store.save("p2", "two", {"source_file": "a.md"})

            store.delete_many(["p1"])

            self.assertFalse((Path(temp_dir) / "p1.json").exists())
            self.assertTrue((Path(temp_dir) / "p2.json").exists())

    def test_same_name_markdown_update_rebuilds_whole_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            values = {
                "MARKDOWN_DIR": str(temp_path / "markdown"),
                "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
                "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
                "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
                "DOCUMENT_IMAGE_DIR": str(temp_path / "images"),
                "INGESTION_LOG_DIR": str(temp_path / "ingestion_logs"),
                "PARENT_STORE_PATH": str(temp_path / "parents"),
                "COURSE_STRUCTURE_PATH": str(temp_path / "course_structure.json"),
                "MARKDOWN_CLEANING_ENABLED": False,
                "INGESTION_SKIP_UNCHANGED_FILES": True,
                "INGESTION_STAGE_LOG_ENABLED": False,
                "MIN_PARENT_SIZE": 1,
                "MAX_PARENT_SIZE": 2000,
                "CHILD_CHUNK_SIZE": 1000,
                "CHILD_CHUNK_OVERLAP": 0,
            }
            with ConfigPatch(**values):
                source = temp_path / "slides.md"
                source.write_text(paged_markdown(), encoding="utf-8")
                rag_system = FakeRagSystem(config.PARENT_STORE_PATH)
                manager = DocumentManager(rag_system)

                added, skipped = manager.add_documents([str(source)])
                self.assertEqual((added, skipped), (1, 0))
                first_parent_ids = {
                    doc.metadata["parent_id"]
                    for doc in rag_system.vector_db.collection.documents
                }
                self.assertIn("slides_page_1_parent_0", first_parent_ids)
                self.assertIn("slides_page_5_parent_0", first_parent_ids)

                source.write_text(paged_markdown("page three changed"), encoding="utf-8")
                added, skipped = manager.add_documents([str(source)])

                self.assertEqual((added, skipped), (1, 0))
                self.assertEqual(rag_system.vector_db.deleted_parent_ids, [])
                self.assertEqual(rag_system.vector_db.deleted_source_files, ["slides.md", "slides.md"])
                current_parent_ids = {
                    doc.metadata["parent_id"]
                    for doc in rag_system.vector_db.collection.documents
                }
                self.assertIn("slides_page_1_parent_0", current_parent_ids)
                self.assertIn("slides_page_3_parent_0", current_parent_ids)
                self.assertIn("slides_page_5_parent_0", current_parent_ids)
                current_text = "\n".join(
                    doc.page_content for doc in rag_system.vector_db.collection.documents
                )
                self.assertIn("page three changed", current_text)
                self.assertNotIn("page three\n", current_text)

    def test_docx_upload_converts_to_markdown_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            values = {
                "MARKDOWN_DIR": str(temp_path / "markdown"),
                "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
                "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
                "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
                "DOCUMENT_IMAGE_DIR": str(temp_path / "images"),
                "INGESTION_LOG_DIR": str(temp_path / "ingestion_logs"),
                "PARENT_STORE_PATH": str(temp_path / "parents"),
                "COURSE_STRUCTURE_PATH": str(temp_path / "course_structure.json"),
                "MARKDOWN_CLEANING_ENABLED": False,
                "INGESTION_SKIP_UNCHANGED_FILES": True,
                "INGESTION_STAGE_LOG_ENABLED": False,
                "MIN_PARENT_SIZE": 1,
                "MAX_PARENT_SIZE": 2000,
                "CHILD_CHUNK_SIZE": 1000,
                "CHILD_CHUNK_OVERLAP": 0,
            }
            with ConfigPatch(**values):
                source = temp_path / "notes.docx"
                source.write_bytes(b"fake")
                rag_system = FakeRagSystem(config.PARENT_STORE_PATH)
                manager = DocumentManager(rag_system)

                with patch("ingestion.conversion._convert_with_markitdown", return_value="# Notes\n\nBody"):
                    added, skipped = manager.add_documents([str(source)], course_names="Database Systems")

                self.assertEqual((added, skipped), (1, 0))
                self.assertEqual(manager.get_markdown_files(), ["notes.md"])
                self.assertEqual(manager.course_store.source_files_for_course("Database Systems"), ["notes.md"])
                self.assertTrue((Path(config.MARKDOWN_DIR) / "notes.md").exists())
                self.assertEqual(rag_system.vector_db.collection.documents[0].metadata["source_file"], "notes.md")

                manifest = IndexManifest()
                document = manifest.get_document("notes.md")
                self.assertEqual(document["source_file"], "notes.md")
                self.assertEqual(document["original_file"], "notes.docx")

    def test_cross_page_duplicate_heading_does_not_dirty_unchanged_page_hash(self):
        old_markdown = """# Intro
page one
--- end of page.page_number=1 ---
# Methods
page two
--- end of page.page_number=2 ---
"""
        new_markdown = """# Methods
page one changed
--- end of page.page_number=1 ---
# Methods
page two
--- end of page.page_number=2 ---
"""

        old_hashes = build_page_hashes(clean_markdown_text(old_markdown, source_file="slides.md").pages)
        new_hashes = build_page_hashes(clean_markdown_text(new_markdown, source_file="slides.md").pages)

        self.assertNotEqual(old_hashes["1"], new_hashes["1"])
        self.assertEqual(old_hashes["2"], new_hashes["2"])

    def test_page_rebuild_keeps_cross_page_repeated_heading_as_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            values = {
                "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
                "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
                "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
                "MARKDOWN_CLEANING_ENABLED": True,
                "MIN_PARENT_SIZE": 1,
                "MAX_PARENT_SIZE": 2000,
                "CHILD_CHUNK_SIZE": 1000,
                "CHILD_CHUNK_OVERLAP": 0,
            }
            with ConfigPatch(**values):
                source = temp_path / "slides.md"
                source.write_text(
                    """# Shared Section
page one
--- end of page.page_number=1 ---
# Shared Section
page two
--- end of page.page_number=2 ---
""",
                    encoding="utf-8",
                )

                parent_chunks, _ = DocumentChunker().create_chunks_single(source, page_numbers=[2])

                self.assertEqual(len(parent_chunks), 1)
                self.assertTrue(parent_chunks[0][1].page_content.startswith("# Shared Section"))
                self.assertEqual(parent_chunks[0][1].metadata["page_numbers"], [2])

    def test_markdown_without_page_markers_omits_page_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            values = {
                "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
                "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
                "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
                "MARKDOWN_CLEANING_ENABLED": False,
                "MIN_PARENT_SIZE": 1,
                "MAX_PARENT_SIZE": 2000,
                "CHILD_CHUNK_SIZE": 1000,
                "CHILD_CHUNK_OVERLAP": 0,
            }
            with ConfigPatch(**values):
                source = temp_path / "notes.md"
                source.write_text("# Notes\n\nNo physical page number.", encoding="utf-8")

                parent_chunks, child_chunks = DocumentChunker().create_chunks_single(source)

                self.assertTrue(parent_chunks)
                metadata = parent_chunks[0][1].metadata
                self.assertEqual(metadata["source_file"], "notes.md")
                self.assertNotIn("page_number", metadata)
                self.assertNotIn("page_numbers", metadata)
                self.assertEqual(child_chunks[0].metadata["source_file"], "notes.md")


if __name__ == "__main__":
    unittest.main()
