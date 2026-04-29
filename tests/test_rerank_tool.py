import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from rag_agent.tools import ToolFactory


def make_doc(idx):
    return Document(
        page_content=f"content {idx}",
        metadata={"chunk_id": f"c{idx}", "parent_id": "parent_1", "source": "source.pdf"},
    )


class FakeVectorDb:
    def __init__(self):
        self.calls = []

    def dense_search(self, collection_name, query, k):
        self.calls.append(("dense", collection_name, query, k))
        return [make_doc(i) for i in range(5)]


class FakeCollection:
    pass


class TestRerankToolIntegration(unittest.TestCase):
    def setUp(self):
        self.vector_db = FakeVectorDb()
        self.tool_factory = ToolFactory(
            FakeCollection(),
            vector_db=self.vector_db,
            collection_name="child_collection",
        )

    def test_search_child_chunks_does_not_rerank(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RERANKER_ENABLED", True), \
            patch("config.RERANKER_TOP_N", 3):
            output = self.tool_factory._search_child_chunks("query", 2)

        self.assertNotIn("Rerank Score:", output)
        self.assertEqual(self.vector_db.calls[0][-1], 3)

    def test_search_child_chunks_uses_limit_when_reranker_disabled(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RERANKER_ENABLED", False):
            output = self.tool_factory._search_child_chunks("query", 1)

        self.assertIn("Content: content 0", output)

    def test_search_child_chunks_expands_candidate_pool_for_graph_rerank(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RERANKER_ENABLED", True), \
            patch("config.RERANKER_TOP_N", 7):
            self.tool_factory._search_child_chunks("query", 2)

        self.assertEqual(self.vector_db.calls[0][-1], 7)


if __name__ == "__main__":
    unittest.main()
