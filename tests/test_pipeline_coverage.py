import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.documents import Document
from retrieval.pipeline import RetrievalPipeline
from retrieval.reranker import RerankerUnavailable


def child_doc(content="child evidence", chunk_id="child_1", parent_id="parent_1",
              chunk_index=1, source="source.pdf", source_file=None, extra=None):
    metadata = {
        "chunk_id": chunk_id,
        "parent_id": parent_id,
        "source": source,
        "source_file": source_file or source,
    }
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    if extra:
        metadata.update(extra)
    return Document(page_content=content, metadata=metadata)


class FakeVectorDb:
    def __init__(self, docs=None, neighbors=None, neighbor_error=False):
        self.docs = docs or []
        self.neighbors = neighbors or []
        self.neighbor_error = neighbor_error

    def rrf_search(self, query, dense_k, sparse_k, fused_k, rrf_k):
        return self.docs

    def dense_search(self, query, k):
        return self.docs[:k]

    def sparse_search(self, query, k):
        return self.docs[:k]

    def load_child_neighbors(self, anchors, window=1):
        if self.neighbor_error:
            raise RuntimeError("neighbor boom")
        return list(self.neighbors)


class FakeParentStore:
    def __init__(self, parents=None, error=False):
        self.parents = parents or {}
        self.error = error

    def load_content_many(self, parent_ids):
        if self.error:
            raise RuntimeError("parent boom")
        return [self.parents[pid] for pid in parent_ids if pid in self.parents]

    def load_content(self, parent_id):
        if self.error:
            raise RuntimeError("parent boom")
        return self.parents.get(parent_id)


def parent_row(parent_id="parent_1", content="parent evidence", source="source.pdf"):
    return {
        "parent_id": parent_id,
        "metadata": {"source": source, "source_file": source},
        "content": content,
    }


class TestFormatChildChunkResults(unittest.TestCase):
    def test_rerank_score_formatting_and_invalid_value(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())
        good = child_doc(extra={"rerank_score": 0.5, "rerank_rank": 1})
        bad = child_doc(extra={"rerank_score": "not-a-number"})
        with patch("config.RETRIEVAL_DEBUG", True):
            output = pipeline.format_child_chunk_results([good, bad])
        self.assertIn("Rerank Score: 0.500000", output)
        self.assertIn("Rerank Rank: 1", output)
        self.assertIn("Rerank Score: not-a-number", output)


class TestSearchChildChunkDocuments(unittest.TestCase):
    def _pipeline(self, docs):
        return RetrievalPipeline(vector_db=FakeVectorDb(docs), parent_store_manager=FakeParentStore())

    def test_dense_mode(self):
        pipeline = self._pipeline([child_doc()])
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), patch("config.RERANKER_ENABLED", False):
            results = pipeline.search_child_chunk_documents("q", 5)
        self.assertEqual(len(results), 1)

    def test_sparse_mode(self):
        pipeline = self._pipeline([child_doc()])
        with patch("config.RETRIEVAL_FUSION_MODE", "sparse"), patch("config.RERANKER_ENABLED", False):
            results = pipeline.search_child_chunk_documents("q", 5)
        self.assertEqual(len(results), 1)

    def test_unsupported_mode_raises(self):
        pipeline = self._pipeline([child_doc()])
        with patch("config.RETRIEVAL_FUSION_MODE", "bogus"):
            with self.assertRaises(ValueError):
                pipeline.search_child_chunk_documents("q", 5)

    def test_rrf_requires_vector_db(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())
        pipeline.vector_db = None
        with patch("config.RETRIEVAL_FUSION_MODE", "rrf"):
            with self.assertRaises(ValueError):
                pipeline.search_child_chunk_documents("q", 5)

    def test_dense_requires_vector_db(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())
        pipeline.vector_db = None
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"):
            with self.assertRaises(ValueError):
                pipeline.search_child_chunk_documents("q", 5)

    def test_sparse_requires_vector_db(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())
        pipeline.vector_db = None
        with patch("config.RETRIEVAL_FUSION_MODE", "sparse"):
            with self.assertRaises(ValueError):
                pipeline.search_child_chunk_documents("q", 5)

    def test_reranker_enabled_expands_retrieval_limit(self):
        pipeline = self._pipeline([child_doc()])
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", True), \
                patch("config.RERANKER_TOP_N", 40):
            results = pipeline.search_child_chunk_documents("q", 5)
        self.assertEqual(len(results), 1)


class TestSearchChildChunks(unittest.TestCase):
    def test_no_relevant_chunks(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb([]), parent_store_manager=FakeParentStore())
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), patch("config.RERANKER_ENABLED", False):
            self.assertEqual(pipeline.search_child_chunks("q", 5), "NO_RELEVANT_CHUNKS")

    def test_retrieval_error_branch(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb([]), parent_store_manager=FakeParentStore())
        with patch("config.RETRIEVAL_FUSION_MODE", "bogus"):
            result = pipeline.search_child_chunks("q", 5)
        self.assertTrue(result.startswith("RETRIEVAL_ERROR:"))

    def test_returns_formatted_results(self):
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb([child_doc()]), parent_store_manager=FakeParentStore())
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), patch("config.RERANKER_ENABLED", False):
            result = pipeline.search_child_chunks("q", 5)
        self.assertIn("Parent ID: parent_1", result)


class FakeReranker:
    def __init__(self, error=None):
        self.error = error

    def rerank(self, query, documents, top_k, score_threshold=None):
        if self.error:
            raise self.error
        docs = list(documents)[:top_k]
        for rank, doc in enumerate(docs, start=1):
            doc.metadata["rerank_score"] = 0.9
            doc.metadata["rerank_rank"] = rank
        return docs


class TestRerankChildDocuments(unittest.TestCase):
    def _pipeline(self):
        return RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())

    def test_disabled_truncates_to_final_top_k(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id=f"c{i}", parent_id=f"p{i}") for i in range(5)]
        with patch("config.RERANKER_ENABLED", False), patch("config.RERANKER_FINAL_TOP_K", 2):
            result = pipeline.rerank_child_documents("q", docs)
        self.assertEqual(len(result), 2)

    def test_empty_docs_returns_truncated(self):
        pipeline = self._pipeline()
        with patch("config.RERANKER_ENABLED", True), patch("config.RERANKER_FINAL_TOP_K", 3):
            result = pipeline.rerank_child_documents("q", [])
        self.assertEqual(result, [])

    def test_top_k_zero_returns_empty(self):
        pipeline = self._pipeline()
        docs = [child_doc()]
        with patch("config.RERANKER_ENABLED", True), patch("config.RERANKER_TOP_N", 40), \
                patch("config.RERANKER_FINAL_TOP_K", 0):
            result = pipeline.rerank_child_documents("q", docs)
        self.assertEqual(result, [])

    def test_successful_rerank(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id=f"c{i}", parent_id=f"p{i}") for i in range(3)]
        with patch("config.RERANKER_ENABLED", True), patch("config.RERANKER_TOP_N", 40), \
                patch("config.RERANKER_FINAL_TOP_K", 2), \
                patch("config.RERANKER_SCORE_THRESHOLD", 0.1), \
                patch("retrieval.pipeline.get_reranker", return_value=FakeReranker()):
            result = pipeline.rerank_child_documents("q", docs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].metadata["rerank_rank"], 1)

    def test_reranker_unavailable_falls_back(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id=f"c{i}", parent_id=f"p{i}") for i in range(3)]
        with patch("config.RERANKER_ENABLED", True), patch("config.RERANKER_TOP_N", 40), \
                patch("config.RERANKER_FINAL_TOP_K", 2), \
                patch("config.RERANKER_SCORE_THRESHOLD", 0.1), \
                patch("retrieval.pipeline.get_reranker",
                      return_value=FakeReranker(error=RerankerUnavailable("nope"))):
            result = pipeline.rerank_child_documents("q", docs)
        self.assertEqual(len(result), 2)
        self.assertNotIn("rerank_score", result[0].metadata)

    def test_generic_exception_falls_back(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id=f"c{i}", parent_id=f"p{i}") for i in range(3)]
        with patch("config.RERANKER_ENABLED", True), patch("config.RERANKER_TOP_N", 40), \
                patch("config.RERANKER_FINAL_TOP_K", 2), \
                patch("config.RERANKER_SCORE_THRESHOLD", 0.1), \
                patch("retrieval.pipeline.get_reranker",
                      return_value=FakeReranker(error=RuntimeError("boom"))):
            result = pipeline.rerank_child_documents("q", docs)
        self.assertEqual(len(result), 2)


class TestContextFromChildDoc(unittest.TestCase):
    def _pipeline(self):
        return RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())

    def test_invalid_score_becomes_none(self):
        pipeline = self._pipeline()
        doc = child_doc(extra={"rrf_score": "bad"})
        ctx = pipeline.context_from_child_doc(doc)
        self.assertIsNone(ctx["score"])

    def test_rerank_score_preferred(self):
        pipeline = self._pipeline()
        doc = child_doc(extra={"rerank_score": 0.7, "rrf_score": 0.1})
        ctx = pipeline.context_from_child_doc(doc)
        self.assertAlmostEqual(ctx["score"], 0.7)


class TestChildContexts(unittest.TestCase):
    def _pipeline(self):
        return RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())

    def test_deduplicates_by_chunk_id(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id="c1"), child_doc(chunk_id="c1")]
        contexts = pipeline.child_contexts(docs)
        self.assertEqual(len(contexts), 1)

    def test_dedup_key_uses_parent_and_content_when_no_chunk_id(self):
        pipeline = self._pipeline()
        d1 = child_doc(chunk_id="", content="same")
        d2 = child_doc(chunk_id="", content="same")
        contexts = pipeline.child_contexts([d1, d2])
        self.assertEqual(len(contexts), 1)

    def test_allowed_source_filter_skips_disallowed(self):
        pipeline = self._pipeline()
        pipeline.set_allowed_source_files(["allowed.pdf"])
        docs = [child_doc(chunk_id="c1", source="other.pdf", source_file="other.pdf")]
        contexts = pipeline.child_contexts(docs)
        self.assertEqual(contexts, [])


class TestNeighborContexts(unittest.TestCase):
    def test_neighbor_error_falls_back_to_child(self):
        vector_db = FakeVectorDb(neighbor_error=True)
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=FakeParentStore())
        docs = [child_doc(content="fallback child", chunk_id="c1", chunk_index=2,
                          extra={"rerank_score": 0.8})]
        contexts = pipeline.neighbor_contexts(docs, window=1)
        self.assertEqual(contexts[0]["content"], "fallback child")

    def test_neighbor_dedup_and_score_propagation(self):
        neighbors = [
            {"parent_id": "parent_1", "content": "n1",
             "metadata": {"chunk_id": "c1", "chunk_index": 1, "source": "source.pdf",
                          "source_file": "source.pdf"}},
            {"parent_id": "parent_1", "content": "dup",
             "metadata": {"chunk_id": "c1", "chunk_index": 1, "source": "source.pdf",
                          "source_file": "source.pdf"}},
        ]
        vector_db = FakeVectorDb(neighbors=neighbors)
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=FakeParentStore())
        docs = [child_doc(chunk_id="c1", parent_id="parent_1", chunk_index=1,
                          extra={"rerank_score": 0.6})]
        contexts = pipeline.neighbor_contexts(docs, window=1)
        neighbor_ctxs = [c for c in contexts if c["context_type"] == "neighbor_child"]
        self.assertEqual(len(neighbor_ctxs), 1)
        self.assertAlmostEqual(neighbor_ctxs[0]["score"], 0.6)

    def test_neighbor_uses_parent_score_when_no_child_id(self):
        neighbors = [
            {"parent_id": "parent_1", "content": "n-noid",
             "metadata": {"chunk_id": "", "chunk_index": 5, "source": "source.pdf",
                          "source_file": "source.pdf"}},
        ]
        vector_db = FakeVectorDb(neighbors=neighbors)
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=FakeParentStore())
        docs = [child_doc(chunk_id="anchor", parent_id="parent_1", chunk_index=1,
                          extra={"rerank_score": 0.42})]
        contexts = pipeline.neighbor_contexts(docs, window=1)
        noid = [c for c in contexts if c["content"] == "n-noid"][0]
        self.assertAlmostEqual(noid["score"], 0.42)

    def test_neighbor_source_filter_skips_disallowed(self):
        neighbors = [
            {"parent_id": "parent_1", "content": "blocked",
             "metadata": {"chunk_id": "cb", "chunk_index": 1, "source": "other.pdf",
                          "source_file": "other.pdf"}},
        ]
        vector_db = FakeVectorDb(neighbors=neighbors)
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=FakeParentStore())
        pipeline.set_allowed_source_files(["source.pdf"])
        docs = [child_doc(chunk_id="c1", chunk_index=1)]
        contexts = pipeline.neighbor_contexts(docs, window=1)
        self.assertNotIn("blocked", [c["content"] for c in contexts])


class TestParentContexts(unittest.TestCase):
    def test_parent_store_error_falls_back(self):
        store = FakeParentStore(error=True)
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        docs = [child_doc(parent_id="parent_1", extra={"rerank_score": 0.5})]
        contexts = pipeline.parent_contexts(["parent_1"], docs)
        self.assertEqual(contexts[0]["context_type"], "child")

    def test_skips_empty_and_duplicate_parent_ids(self):
        store = FakeParentStore(parents={"parent_1": parent_row()})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        store.parents["dup"] = parent_row(parent_id="dup")
        raw = [parent_row(parent_id=""), parent_row(parent_id="parent_1"),
               parent_row(parent_id="parent_1")]
        store.load_content_many = lambda ids: raw
        contexts = pipeline.parent_contexts(["parent_1"], [])
        self.assertEqual(len([c for c in contexts if c["parent_id"] == "parent_1"]), 1)

    def test_allowed_source_filter_skips_parent(self):
        store = FakeParentStore(parents={"parent_1": parent_row(source="other.pdf")})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        pipeline.set_allowed_source_files(["source.pdf"])
        contexts = pipeline.parent_contexts(["parent_1"], [])
        self.assertEqual(contexts, [])

    def test_fallback_for_parent_id_missing_from_store(self):
        store = FakeParentStore(parents={})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        docs = [child_doc(parent_id="parent_x", extra={"rerank_score": 0.5})]
        contexts = pipeline.parent_contexts(["parent_x"], docs)
        self.assertEqual(contexts[0]["context_type"], "child")


class TestSelectContextPolicy(unittest.TestCase):
    def _pipeline(self):
        return RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=FakeParentStore())

    def test_invalid_config_returns_parent(self):
        pipeline = self._pipeline()
        with patch("config.RETRIEVAL_CONTEXT_POLICY", "bogus"):
            policy, reason = pipeline.select_context_policy("q", [], [])
        self.assertEqual(policy, "parent")
        self.assertTrue(reason.startswith("invalid_config:"))

    def test_multiple_child_hits_same_parent(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id="a", parent_id="p1"), child_doc(chunk_id="b", parent_id="p1")]
        with patch("config.RETRIEVAL_CONTEXT_POLICY", "adaptive"), \
                patch("config.RETRIEVAL_PARENT_EXPAND_MIN_HITS", 2):
            policy, reason = pipeline.select_context_policy("plain query", docs, [])
        self.assertEqual(policy, "parent")
        self.assertEqual(reason, "multiple_child_hits_same_parent")

    def test_default_neighbor_context(self):
        pipeline = self._pipeline()
        docs = [child_doc(chunk_id="a", parent_id="p1")]
        with patch("config.RETRIEVAL_CONTEXT_POLICY", "adaptive"), \
                patch("config.RETRIEVAL_PARENT_EXPAND_MIN_HITS", 2):
            policy, reason = pipeline.select_context_policy("plain statement", docs, [])
        self.assertEqual(policy, "neighbor")
        self.assertEqual(reason, "default_neighbor_context")


class TestRagResearch(unittest.TestCase):
    def test_low_score_evidence_status(self):
        # child_docs present but contexts empty: doc has empty parent_id so under
        # parent policy no parent_ids are produced and parent_contexts returns [].
        vector_db = FakeVectorDb([child_doc(parent_id="")])
        store = FakeParentStore(parents={})
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=store)
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", True), \
                patch("config.RERANKER_TOP_N", 40), \
                patch("config.RERANKER_FINAL_TOP_K", 3), \
                patch("config.RERANKER_SCORE_THRESHOLD", 0.6), \
                patch("config.RETRIEVAL_CONTEXT_POLICY", "parent"), \
                patch("retrieval.pipeline.get_reranker",
                      return_value=FakeReranker(error=RerankerUnavailable("x"))):
            result = json.loads(pipeline.rag_research("question"))
        self.assertEqual(result["diagnostics"]["evidence_status"], "low_score")

    def test_insufficient_evidence_status(self):
        vector_db = FakeVectorDb([])
        store = FakeParentStore(parents={})
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=store)
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", False):
            result = json.loads(pipeline.rag_research("question"))
        self.assertEqual(result["diagnostics"]["evidence_status"], "insufficient")
        self.assertEqual(result["gaps"], ["No relevant document context was retrieved."])

    def test_sufficient_with_keep_parent_ids(self):
        store = FakeParentStore(parents={"parent_keep": parent_row("parent_keep", "kept")})
        vector_db = FakeVectorDb([child_doc(parent_id="parent_1")])
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=store)
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", False):
            result = json.loads(pipeline.rag_research("q", keep_parent_ids=["parent_keep"]))
        self.assertEqual(result["diagnostics"]["evidence_status"], "sufficient")
        self.assertIn("parent_keep", result["parent_ids"])

    def test_exclude_parent_ids_and_focus(self):
        store = FakeParentStore(parents={"parent_1": parent_row()})
        vector_db = FakeVectorDb([
            child_doc(chunk_id="c1", parent_id="parent_1"),
            child_doc(chunk_id="c2", parent_id="parent_excluded"),
        ])
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=store)
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", False), \
                patch("config.RETRIEVAL_CONTEXT_POLICY", "parent"):
            result = json.loads(pipeline.rag_research(
                "q", focus="myfocus", exclude_parent_ids=["parent_excluded"],
                retry_reason="retry1"))
        self.assertEqual(result["focus"], "myfocus")
        self.assertNotIn("parent_excluded", result["parent_ids"])
        self.assertEqual(result["diagnostics"]["retry_reason"], "retry1")

    def test_rag_research_error_path(self):
        store = FakeParentStore()
        vector_db = FakeVectorDb([child_doc()])
        pipeline = RetrievalPipeline(vector_db=vector_db, parent_store_manager=store)
        with patch("config.RETRIEVAL_FUSION_MODE", "dense"), \
                patch("config.RERANKER_ENABLED", False), \
                patch.object(pipeline, "select_context_policy", side_effect=RuntimeError("boom")):
            result = json.loads(pipeline.rag_research("q", keep_parent_ids=["pk"]))
        self.assertEqual(result["diagnostics"]["evidence_status"], "error")
        self.assertTrue(result["gaps"][0].startswith("RAG_RESEARCH_ERROR:"))
        self.assertEqual(result["parent_ids"], ["pk"])


class TestRetrieveParentChunks(unittest.TestCase):
    def test_retrieve_many_string_input(self):
        store = FakeParentStore(parents={"p1": parent_row("p1", "content one")})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        output = pipeline.retrieve_many_parent_chunks("p1")
        self.assertIn("Parent ID: p1", output)
        self.assertIn("content one", output)

    def test_retrieve_many_empty_returns_sentinel(self):
        store = FakeParentStore(parents={})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        self.assertEqual(pipeline.retrieve_many_parent_chunks(["px"]), "NO_PARENT_DOCUMENTS")

    def test_retrieve_many_error(self):
        store = FakeParentStore(error=True)
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        result = pipeline.retrieve_many_parent_chunks(["p1"])
        self.assertTrue(result.startswith("PARENT_RETRIEVAL_ERROR:"))

    def test_retrieve_single_parent(self):
        store = FakeParentStore(parents={"p1": parent_row("p1", "single content")})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        output = pipeline.retrieve_parent_chunks("p1")
        self.assertIn("single content", output)

    def test_retrieve_single_missing_returns_sentinel(self):
        store = FakeParentStore(parents={})
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        self.assertEqual(pipeline.retrieve_parent_chunks("px"), "NO_PARENT_DOCUMENT")

    def test_retrieve_single_error(self):
        store = FakeParentStore(error=True)
        pipeline = RetrievalPipeline(vector_db=FakeVectorDb(), parent_store_manager=store)
        result = pipeline.retrieve_parent_chunks("p1")
        self.assertTrue(result.startswith("PARENT_RETRIEVAL_ERROR:"))


if __name__ == "__main__":
    unittest.main()
