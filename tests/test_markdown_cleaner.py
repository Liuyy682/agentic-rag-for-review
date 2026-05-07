import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from document_chunker import DocumentChuncker
from markdown_cleaner import clean_markdown_text


SAMPLE_MARKDOWN = """# Agentic RAG Overview

- Multi-step reasoning
- Tool calling
- State management

Company Confidential
--- end of page.page_number=1 ---

# Retrieval Pipeline

1. Query rewrite
2. Hybrid retrieval
3. Rerank

Company Confidential
--- end of page.page_number=2 ---

# Evaluation

第 1 步：构造评测集
第 2 步：运行检索评测

3 / 3
Company Confidential
--- end of page.page_number=3 ---
"""


class TestMarkdownCleaner(unittest.TestCase):
    def test_removes_repeated_footer_and_page_number_without_dropping_content(self):
        cleaned = clean_markdown_text(SAMPLE_MARKDOWN, source_file="slides.pdf")

        self.assertNotIn("Company Confidential", cleaned.cleaned_text)
        self.assertNotIn("3 / 3", cleaned.cleaned_text)
        self.assertIn("# Agentic RAG Overview", cleaned.cleaned_text)
        self.assertIn("- Tool calling", cleaned.cleaned_text)
        self.assertIn("1. Query rewrite", cleaned.cleaned_text)
        self.assertIn("第 1 步：构造评测集", cleaned.cleaned_text)

        reasons = [event.reason for event in cleaned.events]
        self.assertIn("repeated_header_footer", reasons)
        self.assertIn("page_number", reasons)
        self.assertEqual([page.slide_title for page in cleaned.pages], [
            "Agentic RAG Overview",
            "Retrieval Pipeline",
            "Evaluation",
        ])

    def test_keeps_repeated_markdown_heading_as_protected_content(self):
        markdown = """# Shared Section

First page body
--- end of page.page_number=1 ---

# Shared Section

Second page body
--- end of page.page_number=2 ---

# Shared Section

Third page body
--- end of page.page_number=3 ---
"""

        cleaned = clean_markdown_text(markdown, source_file="slides.pdf")

        self.assertEqual(cleaned.cleaned_text.count("# Shared Section"), 3)
        self.assertEqual(len(cleaned.candidates), 3)
        self.assertTrue(all(candidate.action == "kept" for candidate in cleaned.candidates))

    def test_chunker_adds_page_metadata_and_writes_cleaning_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cleaned_dir = config.MARKDOWN_CLEANED_DIR
            old_log_dir = config.MARKDOWN_CLEANING_LOG_DIR
            old_diff_dir = config.MARKDOWN_CLEANING_DIFF_DIR
            old_min_parent = config.MIN_PARENT_SIZE
            old_max_parent = config.MAX_PARENT_SIZE
            try:
                config.MARKDOWN_CLEANED_DIR = str(Path(temp_dir) / "cleaned")
                config.MARKDOWN_CLEANING_LOG_DIR = str(Path(temp_dir) / "logs")
                config.MARKDOWN_CLEANING_DIFF_DIR = str(Path(temp_dir) / "diffs")
                config.MIN_PARENT_SIZE = 1
                config.MAX_PARENT_SIZE = 4000

                md_path = Path(temp_dir) / "slides.md"
                md_path.write_text(SAMPLE_MARKDOWN, encoding="utf-8")

                parent_chunks, child_chunks = DocumentChuncker().create_chunks_single(md_path)

                self.assertTrue(parent_chunks)
                self.assertTrue(child_chunks)
                self.assertTrue((Path(config.MARKDOWN_CLEANED_DIR) / "slides.md").exists())
                self.assertTrue((Path(config.MARKDOWN_CLEANING_LOG_DIR) / "slides.jsonl").exists())
                diff_path = Path(config.MARKDOWN_CLEANING_DIFF_DIR) / "slides.diff"
                self.assertTrue(diff_path.exists())
                self.assertIn("-Company Confidential", diff_path.read_text(encoding="utf-8"))

                first_parent = parent_chunks[0][1]
                self.assertEqual(first_parent.metadata["source_file"], "slides.pdf")
                self.assertEqual(first_parent.metadata["page_number"], 1)
                self.assertEqual(first_parent.metadata["page_numbers"], [1])
                self.assertEqual(first_parent.metadata["slide_title"], "Agentic RAG Overview")

                first_child = child_chunks[0]
                self.assertEqual(first_child.metadata["source_file"], "slides.pdf")
                self.assertIn("chunk_index", first_child.metadata)
            finally:
                config.MARKDOWN_CLEANED_DIR = old_cleaned_dir
                config.MARKDOWN_CLEANING_LOG_DIR = old_log_dir
                config.MARKDOWN_CLEANING_DIFF_DIR = old_diff_dir
                config.MIN_PARENT_SIZE = old_min_parent
                config.MAX_PARENT_SIZE = old_max_parent


if __name__ == "__main__":
    unittest.main()
