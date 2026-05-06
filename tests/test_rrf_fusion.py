import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from retrieval.fusion import get_doc_key, reciprocal_rank_fusion


def doc(chunk_id, content=None, parent_id="p1"):
    return Document(
        page_content=content or f"content {chunk_id}",
        metadata={"chunk_id": chunk_id, "parent_id": parent_id, "source": "test.pdf"},
    )


class TestReciprocalRankFusion(unittest.TestCase):
    def test_single_ranking_preserves_order(self):
        results = reciprocal_rank_fusion([[doc("d1"), doc("d2"), doc("d3")]], k=60, top_k=3)

        self.assertEqual([d.metadata["chunk_id"] for d in results], ["d1", "d2", "d3"])

    def test_overlapping_rankings_boost_shared_documents(self):
        dense = [doc("d1"), doc("d2"), doc("d3")]
        sparse = [doc("d2"), doc("d4"), doc("d1")]

        results = reciprocal_rank_fusion([dense, sparse], k=60, top_k=4)
        ordered_ids = [d.metadata["chunk_id"] for d in results]

        self.assertEqual(ordered_ids[0], "d2")
        self.assertLess(ordered_ids.index("d1"), ordered_ids.index("d3"))
        self.assertLess(ordered_ids.index("d1"), ordered_ids.index("d4"))

    def test_deduplicates_by_chunk_id(self):
        dense_doc = doc("same", "dense content")
        sparse_doc = doc("same", "sparse content")

        results = reciprocal_rank_fusion([[dense_doc], [sparse_doc]], k=60, top_k=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["chunk_id"], "same")
        self.assertEqual(results[0].metadata["rrf_rank_details"], {"rank_0": 1, "rank_1": 1})

    def test_top_k_truncates_results(self):
        ranking = [doc(f"d{i}") for i in range(10)]

        results = reciprocal_rank_fusion([ranking], k=60, top_k=3)

        self.assertEqual(len(results), 3)

    def test_fallback_key_is_stable(self):
        document = Document(page_content="same content", metadata={"parent_id": "parent"})

        self.assertEqual(get_doc_key(document), get_doc_key(document))


if __name__ == "__main__":
    unittest.main()
