import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import ingestion.cleaning as cleaning
from ingestion.cleaning import (
    clean_markdown_text,
    write_cleaning_log,
    _sanitize_pua_characters,
    _parse_pages_by_form_feed,
    _parse_image_analysis_fields,
    _is_low_value_picture_text,
)


class TestPuaSanitization(unittest.TestCase):
    def test_no_pua_returns_unmodified(self):
        line, modified = _sanitize_pua_characters("normal text")
        self.assertEqual(line, "normal text")
        self.assertFalse(modified)

    def test_pua_with_known_replacement_and_drop(self):
        # Build a line with PUA chars: one mapped, one unmapped (dropped).
        mapped_cp = next(iter(cleaning._PUA_REPLACEMENTS))
        mapped = chr(mapped_cp)
        unmapped = "" if mapped_cp != 0xE000 else ""
        line, modified = _sanitize_pua_characters(f"a{mapped}{unmapped}b")
        self.assertTrue(modified)
        self.assertIn(cleaning._PUA_REPLACEMENTS[mapped_cp], line)
        self.assertNotIn(unmapped, line)


class TestFormFeedParsing(unittest.TestCase):
    def test_splits_on_form_feed(self):
        pages = _parse_pages_by_form_feed("page one\x0cpage two", "doc.md")
        self.assertEqual([p.page_number for p in pages], [1, 2])
        self.assertEqual(pages[0].raw_text, "page one")

    def test_blank_document_returns_single_empty_page(self):
        pages = _parse_pages_by_form_feed("   \x0c  ", "doc.md")
        self.assertEqual(len(pages), 1)
        self.assertIsNone(pages[0].page_number)
        self.assertEqual(pages[0].raw_text, "")


class TestImageAnalysisFields(unittest.TestCase):
    def test_parses_and_continues_multiline_value(self):
        block = "OCR: first\nsecond line\nRAG_SUMMARY: a summary\nKEY_TERMS: x, y"
        fields = _parse_image_analysis_fields(block)
        self.assertEqual(fields["OCR"], "first second line")
        self.assertEqual(fields["RAG_SUMMARY"], "a summary")
        self.assertEqual(fields["KEY_TERMS"], "x, y")


class TestLowValuePictureText(unittest.TestCase):
    def test_empty_block_is_low_value(self):
        self.assertTrue(_is_low_value_picture_text("   \n  "))

    def test_only_noise_lines_is_low_value(self):
        self.assertTrue(_is_low_value_picture_text("Figure 1\nPage 3\nBJUT"))

    def test_short_non_data_text_is_low_value(self):
        self.assertTrue(_is_low_value_picture_text("a small caption"))

    def test_data_like_text_is_kept(self):
        text = "attenuation 25 dB at 100 MHz across the frequency horizon of the link budget table"
        self.assertFalse(_is_low_value_picture_text(text))


class TestSmallHelpers(unittest.TestCase):
    def test_html_breaks_to_text(self):
        self.assertEqual(
            cleaning._html_breaks_to_text("a<br>b<br/>**c**<span>d</span>"),
            "a\nb\ncd",
        )

    def test_compact_text_truncates(self):
        out = cleaning._compact_text("x" * 300, limit=10)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(len(out), 10)

    def test_protected_content_line(self):
        self.assertFalse(cleaning._is_protected_content_line("   "))
        self.assertTrue(cleaning._is_protected_content_line("# Heading"))
        self.assertTrue(cleaning._is_protected_content_line("- bullet item"))

    def test_extract_slide_title_from_heading(self):
        self.assertEqual(cleaning._extract_slide_title(["## My Topic", "body"]), "My Topic")

    def test_extract_slide_title_first_plain_line(self):
        # No heading; first non-protected line within first 5 becomes title.
        self.assertEqual(cleaning._extract_slide_title(["Plain Title", "- x"]), "Plain Title")

    def test_extract_slide_title_none(self):
        self.assertIsNone(cleaning._extract_slide_title(["- only", "- bullets"]))


class TestParsePagesFallback(unittest.TestCase):
    def test_plain_markdown_single_page(self):
        pages = cleaning.parse_pages("# Title\nbody", "doc.md")
        self.assertEqual(len(pages), 1)

    def test_form_feed_dispatch(self):
        pages = cleaning.parse_pages("one\x0ctwo", "doc.md")
        self.assertEqual(len(pages), 2)


class TestCleanWithCodeBlockAndLog(unittest.TestCase):
    def test_code_block_lines_preserved_and_log_written(self):
        md = (
            "# Heading\n"
            "```python\n"
            "Company Confidential\n"
            "x = 1\n"
            "```\n"
            "Company Confidential\n"
            "--- end of page.page_number=1 ---\n"
            "# Heading\n"
            "Company Confidential\n"
            "--- end of page.page_number=2 ---\n"
            "# Heading\n"
            "Company Confidential\n"
            "--- end of page.page_number=3 ---\n"
        )
        cleaned = clean_markdown_text(md, source_file="doc.md", min_repeat_pages=2, min_repeat_ratio=0.1)
        # Code-block content must survive even though it matches a repeated edge line.
        self.assertIn("x = 1", cleaned.cleaned_text)
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "log.jsonl"
            write_cleaning_log(cleaned, log_path)
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                json.loads(line)  # each line is valid JSON



if __name__ == "__main__":
    unittest.main()
