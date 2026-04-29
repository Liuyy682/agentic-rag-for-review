import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from rag_agent.reranker import CrossEncoderReranker


def build_fake_cross_encoder(scores):
    class FakeCrossEncoder:
        def __init__(self, model_name, device=None, max_length=None):
            self.model_name = model_name
            self.device = device
            self.max_length = max_length

        def predict(self, pairs, batch_size=None):
            return scores

    return FakeCrossEncoder


def make_docs(count):
    docs = []
    for i in range(count):
        docs.append(
            Document(
                page_content=f"content {i}",
                metadata={"chunk_id": f"c{i}", "parent_id": "p1"},
            )
        )
    return docs


class TestCrossEncoderReranker(unittest.TestCase):
    def test_rerank_orders_by_score(self):
        scores = [0.2, 0.9, 0.4]
        fake_encoder = build_fake_cross_encoder(scores)

        with patch("rag_agent.reranker.CrossEncoder", fake_encoder):
            reranker = CrossEncoderReranker("fake-model", device="cpu", batch_size=2, max_length=32)
            results = reranker.rerank("query", make_docs(3), top_k=3)

        ordered_ids = [doc.metadata["chunk_id"] for doc in results]
        self.assertEqual(ordered_ids, ["c1", "c2", "c0"])
        self.assertEqual(results[0].metadata["rerank_rank"], 1)
        self.assertAlmostEqual(results[0].metadata["rerank_score"], 0.9)

    def test_rerank_applies_top_k_and_threshold(self):
        scores = [0.2, 0.9, 0.4]
        fake_encoder = build_fake_cross_encoder(scores)

        with patch("rag_agent.reranker.CrossEncoder", fake_encoder):
            reranker = CrossEncoderReranker("fake-model", device="cpu", batch_size=2, max_length=32)
            results = reranker.rerank("query", make_docs(3), top_k=2, score_threshold=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
