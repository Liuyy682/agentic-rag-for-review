import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from rag_agent.tools import ToolFactory


class FakeVectorDb:
    def dense_search(self, collection_name, query, k):
        return [
            Document(
                page_content="child evidence",
                metadata={"chunk_id": "child_1", "parent_id": "parent_1", "source": "source.pdf"},
            )
        ]


class MultiSourceFakeVectorDb:
    def dense_search(self, collection_name, query, k):
        return [
            Document(
                page_content="child evidence",
                metadata={"chunk_id": "child_1", "parent_id": "parent_1", "source": "source.pdf"},
            ),
            Document(
                page_content="other evidence",
                metadata={"chunk_id": "child_2", "parent_id": "parent_2", "source": "other.pdf"},
            ),
        ]


class FakeCollection:
    pass


class TestRagResearchTool(unittest.TestCase):
    def setUp(self):
        self.tool_factory = ToolFactory(
            FakeCollection(),
            vector_db=FakeVectorDb(),
            collection_name="child_collection",
        )

    def test_create_tools_exposes_high_level_rag_tool(self):
        tools = self.tool_factory.create_tools()

        self.assertEqual([tool.name for tool in tools], ["rag_research"])

    def test_rag_research_returns_structured_result(self):
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RETRIEVAL_CONTEXT_POLICY", "parent"), \
            patch("config.RERANKER_ENABLED", False), \
            patch.object(
                self.tool_factory.parent_store_manager,
                "load_content_many",
                return_value=[
                    {
                        "parent_id": "parent_1",
                        "metadata": {"source": "source.pdf"},
                        "content": "parent evidence",
                    }
                ],
            ):
            result = json.loads(self.tool_factory._rag_research("query"))

        self.assertEqual(result["query"], "query")
        self.assertEqual(result["parent_ids"], ["parent_1"])
        self.assertEqual(result["sources"], ["source.pdf"])
        self.assertEqual(result["contexts"][0]["content"], "parent evidence")
        self.assertEqual(result["gaps"], [])

    def test_rag_research_respects_course_source_scope(self):
        self.tool_factory = ToolFactory(
            FakeCollection(),
            vector_db=MultiSourceFakeVectorDb(),
            collection_name="child_collection",
        )
        self.tool_factory.set_allowed_source_files(["source.pdf"])
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RETRIEVAL_CONTEXT_POLICY", "parent"), \
            patch("config.RERANKER_ENABLED", False), \
            patch.object(
                self.tool_factory.parent_store_manager,
                "load_content_many",
                return_value=[
                    {
                        "parent_id": "parent_1",
                        "metadata": {"source": "source.pdf"},
                        "content": "parent evidence",
                    },
                    {
                        "parent_id": "parent_2",
                        "metadata": {"source": "other.pdf"},
                        "content": "other evidence",
                    },
                ],
            ):
            result = json.loads(self.tool_factory._rag_research("query"))

        self.assertEqual(result["parent_ids"], ["parent_1"])
        self.assertEqual(result["sources"], ["source.pdf"])
        self.assertEqual(result["contexts"][0]["content"], "parent evidence")


if __name__ == "__main__":
    unittest.main()
