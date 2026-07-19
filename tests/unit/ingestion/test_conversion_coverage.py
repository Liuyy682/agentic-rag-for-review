import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


from agentic_rag import config
from agentic_rag.ingestion import conversion


class TestClearDirectoryContents(unittest.TestCase):
    def test_removes_files_and_subdirs_but_keeps_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "file.txt").write_text("x", encoding="utf-8")
            sub = root / "sub"
            sub.mkdir()
            (sub / "inner.txt").write_text("y", encoding="utf-8")

            conversion.clear_directory_contents(root)

            self.assertTrue(root.is_dir())
            self.assertEqual(list(root.iterdir()), [])

    def test_non_directory_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "does_not_exist"
            # Should not raise
            conversion.clear_directory_contents(missing)


class TestConvertWithMarkitdown(unittest.TestCase):
    def test_uses_injected_markitdown_module(self):
        fake_module = types.ModuleType("markitdown")

        class FakeResult:
            text_content = "# Hello\n\nbody"

        class FakeMarkItDown:
            def convert_local(self, path):
                return FakeResult()

        fake_module.MarkItDown = FakeMarkItDown
        with patch.dict(sys.modules, {"markitdown": fake_module}):
            out = conversion._convert_with_markitdown(Path("whatever.docx"))
        self.assertEqual(out, "# Hello\n\nbody")


class TestConvertPdfWithPymupdf4llm(unittest.TestCase):
    def test_calls_to_markdown_with_image_extraction(self):
        fake_module = types.ModuleType("pymupdf4llm")
        captured = {}

        def to_markdown(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return "# PDF\n\ntext"

        fake_module.to_markdown = to_markdown
        with patch.dict(sys.modules, {"pymupdf4llm": fake_module}):
            out = conversion._convert_pdf_with_pymupdf4llm(Path("doc.pdf"), Path("/tmp/img"))

        self.assertEqual(out, "# PDF\n\ntext")
        self.assertTrue(captured["kwargs"]["write_images"])
        self.assertTrue(captured["kwargs"]["page_separators"])


class TestConvertReturnsExistingMarkdown(unittest.TestCase):
    def test_existing_markdown_returned_without_reconversion(self):
        with tempfile.TemporaryDirectory() as d:
            temp = Path(d)
            output_dir = temp / "markdown"
            output_dir.mkdir()
            # Pre-create the target markdown
            existing = output_dir / "notes.md"
            existing.write_text("already converted", encoding="utf-8")

            source = temp / "notes.md"
            source.write_text("# fresh\n\nbody", encoding="utf-8")

            md_path = conversion.convert_document_to_markdown(
                source, output_dir, overwrite=False
            )
            # Returns the existing file unchanged (overwrite=False short-circuit)
            self.assertEqual(md_path, existing)
            self.assertEqual(md_path.read_text(encoding="utf-8"), "already converted")


class TestDocumentsToMarkdownsGlob(unittest.TestCase):
    def test_string_glob_pattern_is_expanded(self):
        orig_md_dir = config.MARKDOWN_DIR
        with tempfile.TemporaryDirectory() as d:
            try:
                src_dir = Path(d) / "src"
                src_dir.mkdir()
                out_dir = Path(d) / "out"
                config.MARKDOWN_DIR = str(out_dir)

                (src_dir / "one.md").write_text("# one\n\na", encoding="utf-8")
                (src_dir / "two.md").write_text("# two\n\nb", encoding="utf-8")

                paths = conversion.documents_to_markdowns(
                    str(src_dir / "*.md"), overwrite=True
                )
                self.assertEqual(
                    sorted(p.name for p in paths), ["one.md", "two.md"]
                )
            finally:
                config.MARKDOWN_DIR = orig_md_dir


class TestEstimateContextTokens(unittest.TestCase):
    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Encoding:
        def encode(self, text):
            return list(text)

    def test_counts_tokens_across_messages(self):
        messages = [self._Msg("hello world"), self._Msg("another message")]
        with patch.object(conversion.tiktoken, "encoding_for_model", return_value=self._Encoding()):
            total = conversion.estimate_context_tokens(messages)
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0)

    def test_skips_messages_without_content(self):
        class NoContent:
            pass

        messages = [self._Msg(""), NoContent(), self._Msg("text")]
        with patch.object(conversion.tiktoken, "encoding_for_model", return_value=self._Encoding()):
            total = conversion.estimate_context_tokens(messages)
        self.assertGreater(total, 0)

    def test_falls_back_to_cl100k_when_model_lookup_fails(self):
        with patch.object(
            conversion.tiktoken, "encoding_for_model", side_effect=KeyError("no model")
        ), patch.object(
            conversion.tiktoken, "get_encoding", return_value=self._Encoding()
        ):
            total = conversion.estimate_context_tokens([self._Msg("fallback text")])
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()
