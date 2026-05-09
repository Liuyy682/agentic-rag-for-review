import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from ingestion.document_manager import DocumentManager
from ingestion.index_manifest import (
    IndexManifest,
    changed_pages,
    close_rebuild_scope,
    current_index_config,
    expand_with_neighbors,
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
    def test_changed_pages_expand_and_close_over_parent_pages(self):
        document = {
            "pages": {
                "1": {"page_hash": "a"},
                "2": {"page_hash": "b"},
                "3": {"page_hash": "c"},
                "4": {"page_hash": "d"},
                "5": {"page_hash": "e"},
            },
            "parent_pages": {
                "p1": [1],
                "p2": [2, 3],
                "p3": [4],
                "p4": [5],
            },
        }
        new_hashes = {"1": "a", "2": "b", "3": "changed", "4": "d", "5": "e"}

        changed = changed_pages(document, new_hashes)
        seed = expand_with_neighbors(changed, {1, 2, 3, 4, 5})
        rebuild_pages, stale_parent_ids = close_rebuild_scope(document, seed, {1, 2, 3, 4, 5})

        self.assertEqual(changed, {3})
        self.assertEqual(seed, {2, 3, 4})
        self.assertEqual(rebuild_pages, {2, 3, 4})
        self.assertEqual(stale_parent_ids, {"p2", "p3"})

    def test_manifest_detects_config_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = IndexManifest(Path(temp_dir) / "index_manifest.json")
            manifest.data["documents"]["doc.pdf"] = {"pages": {}}
            manifest.data["index_config"] = current_index_config()
            manifest.save()

            loaded = IndexManifest(Path(temp_dir) / "index_manifest.json")
            self.assertTrue(loaded.is_config_compatible())

            loaded.data["index_config"]["chunker_config_hash"] = "different"
            self.assertFalse(loaded.is_config_compatible())

    def test_parent_store_delete_many_removes_only_selected_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ParentStoreManager(temp_dir)
            store.save("p1", "one", {"source_file": "a.pdf"})
            store.save("p2", "two", {"source_file": "a.pdf"})

            store.delete_many(["p1"])

            self.assertFalse((Path(temp_dir) / "p1.json").exists())
            self.assertTrue((Path(temp_dir) / "p2.json").exists())

    def test_same_name_markdown_update_rebuilds_changed_page_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            values = {
                "MARKDOWN_DIR": str(temp_path / "markdown"),
                "MARKDOWN_CLEANED_DIR": str(temp_path / "cleaned"),
                "MARKDOWN_CLEANING_LOG_DIR": str(temp_path / "logs"),
                "MARKDOWN_CLEANING_DIFF_DIR": str(temp_path / "diffs"),
                "DOCUMENT_IMAGE_DIR": str(temp_path / "images"),
                "PARENT_STORE_PATH": str(temp_path / "parents"),
                "MARKDOWN_CLEANING_ENABLED": False,
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
                self.assertEqual(
                    set(rag_system.vector_db.deleted_parent_ids),
                    {
                        "slides_page_2_parent_0",
                        "slides_page_3_parent_0",
                        "slides_page_4_parent_0",
                    },
                )
                current_parent_ids = {
                    doc.metadata["parent_id"]
                    for doc in rag_system.vector_db.collection.documents
                }
                self.assertIn("slides_page_1_parent_0", current_parent_ids)
                self.assertIn("slides_page_5_parent_0", current_parent_ids)


if __name__ == "__main__":
    unittest.main()
