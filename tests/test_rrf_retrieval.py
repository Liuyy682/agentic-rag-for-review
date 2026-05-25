import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from rag_agent.tools import ToolFactory


class FakeVectorDb:
    def __init__(self):
        self.calls = []

    def dense_search(self, query, k):
        self.calls.append(("dense", query, k))
        return [self._doc("dense_child")]

    def sparse_search(self, query, k):
        self.calls.append(("sparse", query, k))
        return [self._doc("sparse_child")]

    def rrf_search(self, query, dense_k, sparse_k, fused_k, rrf_k):
        self.calls.append(("rrf", query, dense_k, sparse_k, fused_k, rrf_k))
        result = self._doc("rrf_child")
        result.metadata["rrf_score"] = 0.032
        result.metadata["rrf_rank_details"] = {"rank_0": 1, "rank_1": 2}
        return [result]

    @staticmethod
    def _doc(chunk_id):
        return Document(
            page_content=f"{chunk_id} content",
            metadata={"chunk_id": chunk_id, "parent_id": "parent_1", "source": "source.pdf"},
        )


class FakeParentStore:
    def load_content_many(self, parent_ids):
        return []

    def load_content(self, parent_id):
        return {}


class TestRrfRetrievalTool(unittest.TestCase):
    def setUp(self):
        self.vector_db = FakeVectorDb()
        self.tool_factory = ToolFactory(vector_db=self.vector_db, parent_store_manager=FakeParentStore())

    def test_dense_mode_returns_compatible_output(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"):
            output = self.tool_factory._search_child_chunks("query", 5)

        self.assertIn("Parent ID: parent_1", output)
        self.assertIn("File Name: source.pdf", output)
        self.assertIn("Content: dense_child content", output)

    def test_sparse_mode_returns_compatible_output(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "sparse"):
            output = self.tool_factory._search_child_chunks("query", 5)

        self.assertIn("Parent ID: parent_1", output)
        self.assertIn("File Name: source.pdf", output)
        self.assertIn("Content: sparse_child content", output)

    def test_rrf_mode_returns_compatible_output(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "rrf"):
            output = self.tool_factory._search_child_chunks("query", 5)

        self.assertIn("Parent ID: parent_1", output)
        self.assertIn("File Name: source.pdf", output)
        self.assertIn("Content: rrf_child content", output)

    def test_rrf_debug_output_is_optional(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "rrf"), patch("config.RETRIEVAL_DEBUG", False):
            output = self.tool_factory._search_child_chunks("query", 5)
        self.assertNotIn("RRF Score:", output)

        with patch("config.RETRIEVAL_FUSION_MODE", "rrf"), patch("config.RETRIEVAL_DEBUG", True):
            output = self.tool_factory._search_child_chunks("query", 5)
        self.assertIn("RRF Score:", output)
        self.assertIn("RRF Rank Details:", output)


if __name__ == "__main__":
    unittest.main()
