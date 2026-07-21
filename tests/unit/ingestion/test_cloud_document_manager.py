import tempfile
import unittest
from pathlib import Path

from agentic_rag import config
from agentic_rag.ingestion.chunking import DocumentChunker
from agentic_rag.ingestion.cloud_document_manager import CloudDocumentManager
from agentic_rag.storage.object_storage import MinioObjectStorage, document_object_keys


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


class FakeObjectStorage:
    def __init__(self):
        self.objects = {}

    def upload_file(self, key, path, content_type=None):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, key, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])

    def delete_object(self, key):
        self.objects.pop(key, None)

    def delete_objects(self, keys):
        for key in keys:
            if key:
                self.objects.pop(key, None)


class FakeRepository:
    def __init__(self):
        self.documents = {}

    def create_processing(self, document_id, original_file, source_file, raw_object_key):
        self.documents[document_id] = {
            "document_id": document_id,
            "original_file": original_file,
            "source_file": source_file,
            "original_extension": Path(original_file).suffix,
            "status": "processing",
            "raw_object_key": raw_object_key,
            "image_object_keys": [],
            "parent_count": 0,
            "child_count": 0,
        }

    def mark_success(self, document_id, **values):
        self.documents[document_id].update(values)
        self.documents[document_id]["status"] = "success"

    def mark_failed(self, document_id, error, last_result=None):
        self.documents[document_id].update(
            {"status": "failed", "last_error": error, "last_result": last_result or {}}
        )

    def set_status(self, document_id, status, error=None):
        self.documents[document_id].update({"status": status, "last_error": error})

    def get_document(self, identifier):
        return next(
            (
                row for row in self.documents.values()
                if identifier in {row["document_id"], row["source_file"], row["original_file"]}
            ),
            None,
        )

    def list_documents(self, statuses=None):
        rows = list(self.documents.values())
        if statuses:
            rows = [row for row in rows if row["status"] in statuses]
        return rows

    def course_names_for_document(self, document_id):
        return []

    def delete_document(self, document_id):
        self.documents.pop(document_id, None)


class FakeCourseStore:
    def __init__(self):
        self.bindings = {}

    def assign_document_to_courses(self, document_id, names, markdown_path, source_file):
        self.bindings[document_id] = list(names) if not isinstance(names, str) else [names]
        return ["course"]

    def remove_document(self, document_id):
        return self.bindings.pop(document_id, [])

    def list_courses(self):
        return []

    def format_course_list(self):
        return "暂无课程。"

    def clear(self):
        self.bindings.clear()


class FakeVectorStore:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)

    def delete_by_document_id(self, document_id):
        self.documents = [
            doc for doc in self.documents if doc.metadata.get("doc_id") != document_id
        ]

    def clear_store(self):
        self.documents.clear()


class FakeParentStore:
    def __init__(self):
        self.documents = []

    def save_many(self, documents):
        self.documents.extend(documents)

    def delete_by_document_id(self, document_id):
        self.documents = [
            item for item in self.documents if item[1].metadata.get("doc_id") != document_id
        ]

    def clear_store(self):
        self.documents.clear()


class FakeRagSystem:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.vector_db = FakeVectorStore()
        self.parent_store = FakeParentStore()


class BrokenChunker:
    def create_chunks_single(self, path):
        raise RuntimeError("chunking failed")


class ObjectStorageTest(unittest.TestCase):
    def test_document_keys_use_unique_prefix_and_safe_name(self):
        keys = document_object_keys("doc-id", "../../notes.pdf")
        self.assertEqual(keys["raw"], "documents/doc-id/original/notes.pdf")
        self.assertEqual(keys["markdown"], "documents/doc-id/derived/document.md")

    def test_missing_configuration_fails_fast(self):
        with ConfigPatch(
            MINIO_ENDPOINT="",
            MINIO_ACCESS_KEY="",
            MINIO_SECRET_KEY="",
            MINIO_BUCKET="",
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing required MinIO configuration"):
                MinioObjectStorage.from_config()


class CloudDocumentManagerTest(unittest.TestCase):
    def _config(self, root):
        return ConfigPatch(
            MARKDOWN_CLEANED_DIR=str(root / "cleaned"),
            MARKDOWN_CLEANING_LOG_DIR=str(root / "logs"),
            MARKDOWN_CLEANING_DIFF_DIR=str(root / "diffs"),
            MARKDOWN_CLEANING_ENABLED=False,
            IMAGE_ANALYSIS_ENGINE="none",
            MIN_PARENT_SIZE=1,
            MAX_PARENT_SIZE=2000,
            CHILD_CHUNK_SIZE=1000,
            CHILD_CHUNK_OVERLAP=0,
        )

    def test_same_name_uploads_get_distinct_ids_and_object_prefixes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "one" / "notes.md"
            second = root / "two" / "notes.md"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("# First\n\nFirst body", encoding="utf-8")
            second.write_text("# Second\n\nSecond body", encoding="utf-8")
            storage = FakeObjectStorage()
            repository = FakeRepository()
            courses = FakeCourseStore()
            with self._config(root):
                manager = CloudDocumentManager(
                    FakeRagSystem(),
                    object_storage=storage,
                    repository=repository,
                    course_store=courses,
                )
                summary = manager.add_documents_detailed([str(first), str(second)])

            self.assertEqual((summary.added, summary.failed), (2, 0))
            self.assertEqual(len(repository.documents), 2)
            ids = list(repository.documents)
            self.assertNotEqual(ids[0], ids[1])
            self.assertEqual(manager.get_markdown_files(), ["notes.md", "notes.md"])
            raw_keys = [row["raw_object_key"] for row in repository.documents.values()]
            self.assertEqual(len(set(raw_keys)), 2)
            self.assertTrue(all(key.endswith("/original/notes.md") for key in raw_keys))
            self.assertEqual(
                {doc.metadata["source_file"] for doc in manager.rag_system.vector_db.documents},
                {"notes.md"},
            )
            self.assertEqual(
                {doc.metadata["doc_id"] for doc in manager.rag_system.vector_db.documents},
                set(ids),
            )

    def test_failure_keeps_raw_object_and_records_failed_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "notes.md"
            source.write_text("# Notes\n\nBody", encoding="utf-8")
            storage = FakeObjectStorage()
            repository = FakeRepository()
            courses = FakeCourseStore()
            rag = FakeRagSystem()
            rag.chunker = BrokenChunker()
            with self._config(root):
                manager = CloudDocumentManager(
                    rag,
                    object_storage=storage,
                    repository=repository,
                    course_store=courses,
                )
                summary = manager.add_documents_detailed([str(source)])

            self.assertEqual((summary.added, summary.failed), (0, 1))
            row = next(iter(repository.documents.values()))
            self.assertEqual(row["status"], "failed")
            self.assertIn("chunking failed", row["last_error"])
            self.assertIn(row["raw_object_key"], storage.objects)
            self.assertFalse(any("/derived/" in key for key in storage.objects))


if __name__ == "__main__":
    unittest.main()
