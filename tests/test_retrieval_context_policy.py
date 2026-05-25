import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from retrieval.pipeline import RetrievalPipeline


class FakeVectorDb:
    def __init__(self, docs, neighbors=None, recorder=None):
        self.docs = docs
        self.neighbors = neighbors or []
        self.recorder = recorder

    def dense_search(self, query, k):
        return self.docs[:k]

    def load_child_neighbors(self, anchors, window=1):
        if self.recorder:
            self.recorder.neighbor_calls += 1
            self.recorder.last_anchors = list(anchors)
            self.recorder.last_window = window
        return list(self.neighbors)


class RecordingParentStore:
    def __init__(self, parents=None, neighbors=None):
        self.parents = parents or {}
        self.neighbors = neighbors or []
        self.parent_calls = 0
        self.neighbor_calls = 0
        self.last_parent_ids = None
        self.last_anchors = None
        self.last_window = None

    def load_content_many(self, parent_ids):
        self.parent_calls += 1
        self.last_parent_ids = list(parent_ids)
        return [self.parents[parent_id] for parent_id in parent_ids if parent_id in self.parents]


def child_doc(
    content="child evidence",
    chunk_id="child_1",
    parent_id="parent_1",
    chunk_index=1,
    source="source.pdf",
    source_file=None,
):
    metadata = {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "source": source,
        "source_file": source_file or source,
    }
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    return Document(page_content=content, metadata=metadata)


def parent_row(parent_id="parent_1", content="parent evidence", source="source.pdf"):
    return {
        "parent_id": parent_id,
        "metadata": {"source": source, "source_file": source},
        "content": content,
    }


def neighbor_row(content, chunk_id, parent_id="parent_1", chunk_index=1, source="source.pdf"):
    return {
        "parent_id": parent_id,
        "content": content,
        "metadata": {
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "chunk_index": chunk_index,
            "source": source,
            "source_file": source,
        },
    }


class TestRetrievalContextPolicy(unittest.TestCase):
    def run_pipeline(self, docs, parent_store, query="query", keep_parent_ids=None):
        pipeline = RetrievalPipeline(
            vector_db=FakeVectorDb(docs, neighbors=parent_store.neighbors, recorder=parent_store),
            parent_store_manager=parent_store,
        )
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RERANKER_ENABLED", False):
            return json.loads(pipeline.rag_research(query, keep_parent_ids=keep_parent_ids))

    def test_parent_policy_keeps_existing_parent_backfill(self):
        store = RecordingParentStore(
            parents={"parent_1": parent_row(content="expanded parent evidence")}
        )

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "parent"):
            result = self.run_pipeline([child_doc()], store)

        self.assertEqual(result["diagnostics"]["context_policy"], "parent")
        self.assertEqual(result["contexts"][0]["content"], "expanded parent evidence")
        self.assertEqual(result["contexts"][0]["context_type"], "parent")
        self.assertEqual(store.parent_calls, 1)

    def test_child_policy_does_not_read_parent_store(self):
        store = RecordingParentStore(parents={"parent_1": parent_row()})

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "child"):
            result = self.run_pipeline([child_doc(content="only child")], store)

        self.assertEqual(result["diagnostics"]["context_policy"], "child")
        self.assertEqual(result["contexts"][0]["content"], "only child")
        self.assertEqual(result["contexts"][0]["context_type"], "child")
        self.assertEqual(store.parent_calls, 0)
        self.assertEqual(store.neighbor_calls, 0)

    def test_neighbor_policy_returns_same_parent_window(self):
        store = RecordingParentStore(
            neighbors=[
                neighbor_row("before", "child_1", chunk_index=1),
                neighbor_row("hit", "child_2", chunk_index=2),
                neighbor_row("after", "child_3", chunk_index=3),
            ]
        )

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "neighbor"), \
            patch("config.RETRIEVAL_NEIGHBOR_WINDOW", 1):
            result = self.run_pipeline(
                [child_doc(content="hit", chunk_id="child_2", chunk_index=2)],
                store,
            )

        self.assertEqual(result["diagnostics"]["context_policy"], "neighbor")
        self.assertEqual(result["diagnostics"]["neighbor_window"], 1)
        self.assertEqual([context["content"] for context in result["contexts"]], ["before", "hit", "after"])
        self.assertEqual(store.last_anchors, [{"parent_id": "parent_1", "chunk_index": 2}])
        self.assertEqual(store.last_window, 1)
        self.assertEqual(store.parent_calls, 0)

    def test_neighbor_policy_without_chunk_index_falls_back_to_child(self):
        store = RecordingParentStore(neighbors=[])

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "neighbor"):
            result = self.run_pipeline(
                [child_doc(content="no index child", chunk_id="child_missing", chunk_index=None)],
                store,
            )

        self.assertEqual(result["diagnostics"]["context_policy"], "neighbor")
        self.assertEqual(result["contexts"][0]["content"], "no index child")
        self.assertEqual(result["contexts"][0]["context_type"], "child")

    def test_adaptive_fact_query_selects_child(self):
        store = RecordingParentStore(parents={"parent_1": parent_row()})

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "adaptive"):
            result = self.run_pipeline(
                [child_doc(content="nine antigens")],
                store,
                query="How many antigens could be detected?",
            )

        self.assertEqual(result["diagnostics"]["context_policy"], "child")
        self.assertEqual(result["diagnostics"]["context_policy_reason"], "fact_query")
        self.assertEqual(store.parent_calls, 0)

    def test_adaptive_explanation_query_selects_parent(self):
        store = RecordingParentStore(
            parents={"parent_1": parent_row(content="mechanism parent context")}
        )

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "adaptive"):
            result = self.run_pipeline(
                [child_doc(content="mechanism child")],
                store,
                query="Why does viral persistence cause inflammation?",
            )

        self.assertEqual(result["diagnostics"]["context_policy"], "parent")
        self.assertEqual(result["diagnostics"]["context_policy_reason"], "query_requires_broad_context")
        self.assertEqual(result["contexts"][0]["content"], "mechanism parent context")

    def test_keep_parent_ids_force_parent_even_when_configured_child(self):
        store = RecordingParentStore(
            parents={
                "parent_keep": parent_row("parent_keep", "kept parent context"),
                "parent_1": parent_row(content="retrieved parent context"),
            }
        )

        with patch("config.RETRIEVAL_CONTEXT_POLICY", "child"):
            result = self.run_pipeline(
                [child_doc(content="child evidence")],
                store,
                keep_parent_ids=["parent_keep"],
            )

        self.assertEqual(result["diagnostics"]["context_policy"], "parent")
        self.assertEqual(result["diagnostics"]["context_policy_reason"], "keep_parent_ids_requested")
        self.assertEqual(result["parent_ids"], ["parent_keep", "parent_1"])
        self.assertEqual([context["context_type"] for context in result["contexts"]], ["parent", "parent"])

    def test_allowed_source_files_filter_neighbor_contexts(self):
        store = RecordingParentStore(
            neighbors=[
                neighbor_row("allowed before", "child_1", chunk_index=1, source="source.pdf"),
                neighbor_row("blocked neighbor", "child_2", chunk_index=2, source="other.pdf"),
                neighbor_row("allowed hit", "child_3", chunk_index=3, source="source.pdf"),
            ]
        )
        pipeline = RetrievalPipeline(
            vector_db=FakeVectorDb(
                [child_doc(content="allowed hit", chunk_id="child_3", chunk_index=3)],
                neighbors=store.neighbors,
                recorder=store,
            ),
            parent_store_manager=store,
        )
        pipeline.set_allowed_source_files(["source.pdf"])

        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
            patch("config.RETRIEVAL_CONTEXT_POLICY", "neighbor"), \
            patch("config.RERANKER_ENABLED", False):
            result = json.loads(pipeline.rag_research("query"))

        self.assertEqual([context["source"] for context in result["contexts"]], ["source.pdf", "source.pdf"])
        self.assertNotIn("blocked neighbor", [context["content"] for context in result["contexts"]])


if __name__ == "__main__":
    unittest.main()
