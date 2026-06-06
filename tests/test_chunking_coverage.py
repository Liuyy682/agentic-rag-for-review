import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from langchain_core.documents import Document
from ingestion.chunking import DocumentChunker
from ingestion.cleaning import PageBlock


def doc(text, metadata=None):
    return Document(page_content=text, metadata=dict(metadata or {}))


class _FakeSplitter:
    """Splitter stub whose split_text always returns an empty list."""

    def split_text(self, text):
        return []


class TestMergeSmallParents(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.chunker._DocumentChunker__min_parent_size = 10
        self.merge = self.chunker._DocumentChunker__merge_small_parents

    def test_empty_returns_empty(self):
        self.assertEqual(self.merge([]), [])

    def test_small_chunks_merge_until_min_size(self):
        chunks = [
            doc("aaa", {"page_number": 1, "page_numbers": [1]}),
            doc("bbb", {"page_number": 2, "page_numbers": [2]}),
            doc("cccccc", {"page_number": 3, "page_numbers": [3]}),
        ]
        merged = self.merge(chunks)
        self.assertEqual(len(merged), 1)
        self.assertIn("aaa", merged[0].page_content)
        self.assertIn("ccc", merged[0].page_content)
        # metadata merged across pages
        self.assertEqual(merged[0].metadata["page_numbers"], [1, 2, 3])

    def test_leftover_appended_to_last_merged(self):
        chunks = [
            doc("aaaaaaaaaa", {"page_number": 1, "page_numbers": [1]}),  # reaches min
            doc("tiny", {"page_number": 2, "page_numbers": [2]}),         # leftover
        ]
        merged = self.merge(chunks)
        self.assertEqual(len(merged), 1)
        self.assertIn("tiny", merged[0].page_content)
        self.assertEqual(merged[0].metadata["page_numbers"], [1, 2])

    def test_single_small_chunk_only_leftover(self):
        merged = self.merge([doc("tiny", {"page_number": 5})])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].page_content, "tiny")


class TestSplitLargeParents(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.chunker._DocumentChunker__max_parent_size = 100
        self.split = self.chunker._DocumentChunker__split_large_parents

    def test_small_chunk_passes_through(self):
        chunks = [doc("short", {"a": 1})]
        out = self.split(chunks)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].page_content, "short")

    def test_large_chunk_is_split(self):
        chunks = [doc("x" * 500, {"a": 1})]
        out = self.split(chunks)
        self.assertGreater(len(out), 1)
        for c in out:
            self.assertLessEqual(len(c.page_content), 100)


class TestCleanSmallChunks(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.chunker._DocumentChunker__min_parent_size = 10
        self.clean = self.chunker._DocumentChunker__clean_small_chunks

    def test_small_chunk_merges_into_previous_cleaned(self):
        chunks = [
            doc("aaaaaaaaaa", {"page_number": 1, "page_numbers": [1]}),  # big -> cleaned
            doc("tiny", {"page_number": 2, "page_numbers": [2]}),         # small -> merge back
        ]
        out = self.clean(chunks)
        self.assertEqual(len(out), 1)
        self.assertIn("tiny", out[0].page_content)
        self.assertEqual(out[0].metadata["page_numbers"], [1, 2])

    def test_leading_small_chunk_prepended_to_next(self):
        chunks = [
            doc("tiny", {"page_number": 1, "page_numbers": [1]}),
            doc("bbbbbbbbbb", {"page_number": 2, "page_numbers": [2]}),
        ]
        out = self.clean(chunks)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].page_content.startswith("tiny"))
        # prepend=True keeps page 1 first
        self.assertEqual(out[0].metadata["page_numbers"], [1, 2])

    def test_single_small_chunk_kept_as_is(self):
        chunks = [doc("tiny", {"page_number": 1})]
        out = self.clean(chunks)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].page_content, "tiny")

    def test_large_chunk_kept(self):
        chunks = [doc("aaaaaaaaaa", {"page_number": 1})]
        out = self.clean(chunks)
        self.assertEqual(len(out), 1)


class TestMergeMetadata(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.merge = self.chunker._DocumentChunker__merge_metadata

    def test_page_numbers_append_and_prepend_with_dedup(self):
        target = {"page_numbers": [1, 2]}
        self.merge(target, {"page_numbers": [2, 3]})
        self.assertEqual(target["page_numbers"], [1, 2, 3])

        target = {"page_numbers": [2, 3]}
        self.merge(target, {"page_numbers": [1, 2]}, prepend=True)
        self.assertEqual(target["page_numbers"], [1, 2, 3])

    def test_page_numbers_existing_not_a_list_and_none_filtered(self):
        target = {"page_numbers": 1}
        self.merge(target, {"page_numbers": [None, 3]})
        self.assertEqual(target["page_numbers"], [1, 3])

    def test_page_number_key_builds_page_numbers_and_sets_first(self):
        target = {}
        self.merge(target, {"page_number": 2})
        self.assertEqual(target["page_numbers"], [2])
        self.assertEqual(target["page_number"], 2)

    def test_page_number_existing_pages_not_list(self):
        target = {"page_numbers": 1}
        self.merge(target, {"page_number": 2})
        self.assertEqual(target["page_numbers"], [1, 2])
        self.assertEqual(target["page_number"], 1)

    def test_page_number_none_value_leaves_pages_empty(self):
        target = {}
        self.merge(target, {"page_number": None})
        self.assertEqual(target["page_numbers"], [])
        self.assertNotIn("page_number", target)

    def test_slide_title_empty_value_is_skipped(self):
        target = {"slide_title": "Existing"}
        self.merge(target, {"slide_title": ""})
        self.assertEqual(target["slide_title"], "Existing")

    def test_slide_title_set_when_absent(self):
        target = {}
        self.merge(target, {"slide_title": "Intro"})
        self.assertEqual(target["slide_title"], "Intro")

    def test_slide_title_append_and_prepend_chain(self):
        target = {"slide_title": "A"}
        self.merge(target, {"slide_title": "B"})
        self.assertEqual(target["slide_title"], "A -> B")

        target = {"slide_title": "A"}
        self.merge(target, {"slide_title": "B"}, prepend=True)
        self.assertEqual(target["slide_title"], "B -> A")

    def test_slide_title_duplicate_in_chain_not_appended(self):
        target = {"slide_title": "A -> B"}
        self.merge(target, {"slide_title": "B"})
        self.assertEqual(target["slide_title"], "A -> B")

    def test_generic_key_conflict_joins_with_arrow(self):
        target = {"source": "x"}
        self.merge(target, {"source": "y"})
        self.assertEqual(target["source"], "x -> y")

        target = {"source": "x"}
        self.merge(target, {"source": "y"}, prepend=True)
        self.assertEqual(target["source"], "y -> x")

    def test_generic_new_key_and_equal_value(self):
        target = {}
        self.merge(target, {"new": "v"})
        self.assertEqual(target["new"], "v")

        target = {"same": "v"}
        self.merge(target, {"same": "v"})
        self.assertEqual(target["same"], "v")


class TestContiguousPageGroups(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker()
        self.fn = self.chunker._DocumentChunker__contiguous_page_groups

    def _page(self, n, text="body"):
        pb = PageBlock(source_file="d.md", page_number=n, raw_text=text, raw_lines=[text])
        pb.cleaned_text = text
        pb.cleaned_lines = [text]
        return pb

    def test_no_selected_pages_returns_empty(self):
        pages = [self._page(1)]
        self.assertEqual(self.fn(pages, selected_pages={99}), [])

    def test_contiguous_pages_stay_in_one_group(self):
        pages = [self._page(1), self._page(2), self._page(3)]
        groups = self.fn(pages, selected_pages=None)
        self.assertEqual(len(groups), 1)
        self.assertEqual([p.page_number for p in groups[0]], [1, 2, 3])

    def test_gap_splits_into_separate_groups(self):
        pages = [self._page(1), self._page(3)]
        groups = self.fn(pages, selected_pages=None)
        self.assertEqual(len(groups), 2)

    def test_none_page_number_treated_as_one(self):
        pb = PageBlock(source_file="d.md", page_number=None, raw_text="b", raw_lines=["b"])
        groups = self.fn([pb], selected_pages=None)
        self.assertEqual(len(groups), 1)

    def test_selected_pages_filter(self):
        pages = [self._page(1), self._page(2), self._page(3)]
        groups = self.fn(pages, selected_pages={2})
        self.assertEqual(len(groups), 1)
        self.assertEqual([p.page_number for p in groups[0]], [2])


class TestCreateChunksSingle(unittest.TestCase):
    def _override(self):
        self._orig = {
            "MIN_PARENT_SIZE": config.MIN_PARENT_SIZE,
            "MAX_PARENT_SIZE": config.MAX_PARENT_SIZE,
            "MARKDOWN_CLEANING_ENABLED": config.MARKDOWN_CLEANING_ENABLED,
        }

    def _restore(self):
        for k, v in self._orig.items():
            setattr(config, k, v)

    def test_cleaning_disabled_path_uses_raw_text(self):
        self._override()
        try:
            config.MARKDOWN_CLEANING_ENABLED = False
            config.MIN_PARENT_SIZE = 1
            config.MAX_PARENT_SIZE = 4000
            with tempfile.TemporaryDirectory() as d:
                md = Path(d) / "raw.md"
                md.write_text(
                    "# Title\n\nbody text here\n--- end of page.page_number=1 ---\n",
                    encoding="utf-8",
                )
                parents, children = DocumentChunker().create_chunks_single(md)
                self.assertTrue(parents)
                self.assertTrue(children)
                self.assertIn("page_1_parent_0", parents[0][0])
        finally:
            self._restore()

    def test_large_parent_split_and_small_merge_combined(self):
        self._override()
        try:
            config.MARKDOWN_CLEANING_ENABLED = False
            config.MIN_PARENT_SIZE = 50
            config.MAX_PARENT_SIZE = 200
            with tempfile.TemporaryDirectory() as d:
                md = Path(d) / "big.md"
                body = "word " * 200
                md.write_text(
                    f"# Big Section\n\n{body}\n--- end of page.page_number=1 ---\n",
                    encoding="utf-8",
                )
                parents, children = DocumentChunker().create_chunks_single(md)
                self.assertGreater(len(parents), 1)
                # split produced multiple parents; cleaning may re-merge a
                # trailing fragment back so sizes are not strictly bounded.
                self.assertTrue(all(p.page_content for _, p in parents))
        finally:
            self._restore()

    def test_page_numbers_anchor_parent_ids(self):
        self._override()
        try:
            config.MARKDOWN_CLEANING_ENABLED = False
            config.MIN_PARENT_SIZE = 1
            config.MAX_PARENT_SIZE = 4000
            with tempfile.TemporaryDirectory() as d:
                md = Path(d) / "multi.md"
                md.write_text(
                    "# P1\n\naaa\n--- end of page.page_number=1 ---\n"
                    "# P2\n\nbbb\n--- end of page.page_number=2 ---\n",
                    encoding="utf-8",
                )
                # selected page rebuild: only page 2
                parents, _ = DocumentChunker().create_chunks_single(md, page_numbers=[2])
                self.assertTrue(parents)
                self.assertIn("page_2_parent_0", parents[0][0])
        finally:
            self._restore()


class TestSplitPagesEdgeCases(unittest.TestCase):
    def test_blank_page_text_skipped(self):
        chunker = DocumentChunker()
        blank = PageBlock(source_file="d.md", page_number=1, raw_text="   ", raw_lines=["   "])
        blank.cleaned_text = "   "
        groups = chunker._DocumentChunker__split_pages_into_parent_groups(
            [blank], Path("d.md"), selected_pages=None, use_cleaned_text=True
        )
        self.assertEqual(groups, [])

    def test_empty_page_chunks_skipped(self):
        chunker = DocumentChunker()
        chunker._DocumentChunker__parent_splitter = _FakeSplitter()
        page = PageBlock(source_file="d.md", page_number=1, raw_text="content", raw_lines=["content"])
        page.cleaned_text = "content"
        groups = chunker._DocumentChunker__split_pages_into_parent_groups(
            [page], Path("d.md"), selected_pages=None, use_cleaned_text=True
        )
        self.assertEqual(groups, [])


class TestCreateChunksDirectory(unittest.TestCase):
    def test_iterates_markdown_files_in_dir(self):
        orig = {
            "MIN_PARENT_SIZE": config.MIN_PARENT_SIZE,
            "MAX_PARENT_SIZE": config.MAX_PARENT_SIZE,
            "MARKDOWN_CLEANING_ENABLED": config.MARKDOWN_CLEANING_ENABLED,
        }
        try:
            config.MARKDOWN_CLEANING_ENABLED = False
            config.MIN_PARENT_SIZE = 1
            config.MAX_PARENT_SIZE = 4000
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "a.md").write_text(
                    "# A\n\naaa\n--- end of page.page_number=1 ---\n", encoding="utf-8"
                )
                (Path(d) / "b.md").write_text(
                    "# B\n\nbbb\n--- end of page.page_number=1 ---\n", encoding="utf-8"
                )
                parents, children = DocumentChunker().create_chunks(path_dir=d)
                self.assertTrue(parents)
                self.assertTrue(children)
                docs = {pid.split("_page_")[0] for pid, _ in parents}
                self.assertEqual(docs, {"a", "b"})
        finally:
            for k, v in orig.items():
                setattr(config, k, v)


if __name__ == "__main__":
    unittest.main()