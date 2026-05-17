import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from ingestion.conversion import (
    convert_document_to_markdown,
    documents_to_markdowns,
    is_supported_document,
)


class TestDocumentConversion(unittest.TestCase):
    def test_supported_extensions_are_limited_to_first_batch(self):
        self.assertTrue(is_supported_document("lecture.pdf"))
        self.assertTrue(is_supported_document("lecture.md"))
        self.assertTrue(is_supported_document("lecture.docx"))
        self.assertTrue(is_supported_document("lecture.pptx"))
        self.assertFalse(is_supported_document("lecture.zip"))
        self.assertFalse(is_supported_document("lecture.xlsx"))

    def test_markdown_file_is_written_to_markdown_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "notes.md"
            output_dir = temp_path / "markdown"
            source.write_text("# Notes\n\nBody", encoding="utf-8")

            md_path = convert_document_to_markdown(source, output_dir, overwrite=True)

            self.assertEqual(md_path, output_dir / "notes.md")
            self.assertEqual(md_path.read_text(encoding="utf-8"), "# Notes\n\nBody")

    def test_binary_documents_use_markitdown_local_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "markdown"

            for suffix in [".pdf", ".docx", ".pptx"]:
                source = temp_path / f"lecture{suffix}"
                source.write_bytes(b"fake")
                with patch("ingestion.conversion._convert_with_markitdown", return_value="# Converted\n\nBody") as convert:
                    md_path = convert_document_to_markdown(source, output_dir, overwrite=True)

                convert.assert_called_once_with(source)
                self.assertEqual(md_path, output_dir / "lecture.md")
                self.assertEqual(md_path.read_text(encoding="utf-8"), "# Converted\n\nBody")

    def test_rejects_url_zip_directory_and_empty_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "archive.zip"
            zip_path.write_bytes(b"fake")

            with self.assertRaises(ValueError):
                convert_document_to_markdown("https://example.com/file.pdf", temp_path)
            with self.assertRaises(ValueError):
                convert_document_to_markdown(zip_path, temp_path)
            with self.assertRaises(ValueError):
                convert_document_to_markdown(temp_path, temp_path)

            source = temp_path / "empty.pdf"
            source.write_bytes(b"fake")
            with patch("ingestion.conversion._convert_with_markitdown", return_value="   "):
                with self.assertRaises(ValueError):
                    convert_document_to_markdown(source, temp_path, overwrite=True)

    def test_documents_to_markdowns_accepts_multiple_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.md"
            second = temp_path / "second.md"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")

            paths = documents_to_markdowns([first, second], overwrite=True)

            self.assertEqual([path.name for path in paths], ["first.md", "second.md"])


if __name__ == "__main__":
    unittest.main()
