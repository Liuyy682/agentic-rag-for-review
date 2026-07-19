import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock


from rag_agent.tools import ToolFactory


class FakeVectorDb:
    def dense_search(self, query, k):
        return []


class FakeParentStore:
    def load_content_many(self, parent_ids):
        return []

    def load_content(self, parent_id):
        return {}


class TestToolFactoryDelegation(unittest.TestCase):
    def setUp(self):
        self.tool_factory = ToolFactory(
            vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore()
        )
        # Replace the real pipeline with a recording mock so we exercise the
        # thin delegating wrappers without touching real retrieval.
        self.pipeline = MagicMock()
        self.tool_factory.pipeline = self.pipeline

    def test_format_child_chunk_results_delegates(self):
        self.pipeline.format_child_chunk_results.return_value = "formatted"
        result = self.tool_factory._format_child_chunk_results(["doc"])
        self.assertEqual(result, "formatted")
        self.pipeline.format_child_chunk_results.assert_called_once_with(["doc"])

    def test_search_child_chunk_documents_delegates(self):
        self.pipeline.search_child_chunk_documents.return_value = ["d1"]
        result = self.tool_factory._search_child_chunk_documents("query", 3)
        self.assertEqual(result, ["d1"])
        self.pipeline.search_child_chunk_documents.assert_called_once_with("query", 3)

    def test_search_child_chunks_delegates(self):
        self.pipeline.search_child_chunks.return_value = "chunk text"
        result = self.tool_factory._search_child_chunks("query", 2)
        self.assertEqual(result, "chunk text")
        self.pipeline.search_child_chunks.assert_called_once_with("query", 2)

    def test_rerank_child_documents_delegates(self):
        self.pipeline.rerank_child_documents.return_value = ["ranked"]
        result = self.tool_factory._rerank_child_documents("query", ["doc"])
        self.assertEqual(result, ["ranked"])
        self.pipeline.rerank_child_documents.assert_called_once_with("query", ["doc"])

    def test_context_from_child_doc_delegates(self):
        self.pipeline.context_from_child_doc.return_value = {"parent_id": "p1"}
        result = self.tool_factory._context_from_child_doc("doc")
        self.assertEqual(result, {"parent_id": "p1"})
        self.pipeline.context_from_child_doc.assert_called_once_with("doc")

    def test_parent_contexts_delegates(self):
        self.pipeline.parent_contexts.return_value = [{"parent_id": "p1"}]
        result = self.tool_factory._parent_contexts(["p1"], ["doc"])
        self.assertEqual(result, [{"parent_id": "p1"}])
        self.pipeline.parent_contexts.assert_called_once_with(["p1"], ["doc"])

    def test_rag_research_delegates_with_all_args(self):
        self.pipeline.rag_research.return_value = "{}"
        result = self.tool_factory._rag_research(
            "query",
            focus="focus",
            keep_parent_ids=["p1"],
            exclude_parent_ids=["p2"],
            retry_reason="weak",
        )
        self.assertEqual(result, "{}")
        self.pipeline.rag_research.assert_called_once_with(
            query="query",
            focus="focus",
            keep_parent_ids=["p1"],
            exclude_parent_ids=["p2"],
            retry_reason="weak",
        )

    def test_set_allowed_source_files_delegates(self):
        self.tool_factory.set_allowed_source_files(["source.pdf"])
        self.pipeline.set_allowed_source_files.assert_called_once_with(["source.pdf"])

    def test_retrieve_many_parent_chunks_delegates(self):
        self.pipeline.retrieve_many_parent_chunks.return_value = "many"
        result = self.tool_factory._retrieve_many_parent_chunks(["p1", "p2"])
        self.assertEqual(result, "many")
        self.pipeline.retrieve_many_parent_chunks.assert_called_once_with(["p1", "p2"])

    def test_retrieve_parent_chunks_delegates(self):
        self.pipeline.retrieve_parent_chunks.return_value = "one"
        result = self.tool_factory._retrieve_parent_chunks("p1")
        self.assertEqual(result, "one")
        self.pipeline.retrieve_parent_chunks.assert_called_once_with("p1")

    def test_create_tools_exposes_rag_research(self):
        tools = self.tool_factory.create_tools()
        self.assertEqual([t.name for t in tools], ["rag_research"])


if __name__ == "__main__":
    unittest.main()
