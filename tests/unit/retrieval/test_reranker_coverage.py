import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


from langchain_core.documents import Document
import retrieval.reranker as reranker_module
from retrieval.reranker import (
    CrossEncoderReranker,
    RerankerUnavailable,
    _resolve_local_model_path,
    _shorten_query,
    get_reranker,
    resolve_device,
)


def make_docs(count):
    return [
        Document(page_content=f"content {i}", metadata={"chunk_id": f"c{i}"})
        for i in range(count)
    ]


def fake_torch(cuda_avail, mps_avail, has_mps=True):
    m = MagicMock()
    m.cuda.is_available.return_value = cuda_avail
    if has_mps:
        m.backends.mps.is_available.return_value = mps_avail
    else:
        del m.backends.mps
    return m


class TestShortenQuery(unittest.TestCase):
    def test_none_query_returns_empty(self):
        self.assertEqual(_shorten_query(None), "")

    def test_long_query_is_truncated(self):
        long_query = "a" * 300
        result = _shorten_query(long_query, max_len=200)
        self.assertEqual(len(result), 200)
        self.assertTrue(result.endswith("..."))


class TestResolveDevice(unittest.TestCase):
    def test_cuda_requested_but_unavailable_falls_back_to_cpu(self):
        with patch("retrieval.reranker.torch", fake_torch(False, False)):
            self.assertEqual(resolve_device("cuda"), "cpu")

    def test_mps_requested_but_unavailable_falls_back_to_cpu(self):
        with patch("retrieval.reranker.torch", fake_torch(False, False)):
            self.assertEqual(resolve_device("mps"), "cpu")

    def test_mps_requested_and_available_returns_mps(self):
        with patch("retrieval.reranker.torch", fake_torch(False, True)):
            self.assertEqual(resolve_device("mps"), "mps")

    def test_auto_prefers_cuda_when_available(self):
        with patch("retrieval.reranker.torch", fake_torch(True, False)):
            self.assertEqual(resolve_device("auto"), "cuda")

    def test_auto_prefers_mps_when_cuda_absent(self):
        with patch("retrieval.reranker.torch", fake_torch(False, True)):
            self.assertEqual(resolve_device("auto"), "mps")

    def test_auto_falls_back_to_cpu(self):
        with patch("retrieval.reranker.torch", fake_torch(False, False, has_mps=False)):
            self.assertEqual(resolve_device("auto"), "cpu")


class FakeCrossEncoder:
    def __init__(self, model_name, device=None, max_length=None, **kwargs):
        self.model_name = model_name

    def predict(self, pairs, batch_size=None):
        return [0.5 for _ in pairs]


class TestRerankEarlyReturns(unittest.TestCase):
    def test_empty_documents_returns_empty(self):
        with patch("retrieval.reranker.CrossEncoder", FakeCrossEncoder):
            reranker = CrossEncoderReranker("fake-model", device="cpu")
            self.assertEqual(reranker.rerank("q", [], top_k=5), [])

    def test_non_positive_top_k_returns_empty(self):
        with patch("retrieval.reranker.CrossEncoder", FakeCrossEncoder):
            reranker = CrossEncoderReranker("fake-model", device="cpu")
            self.assertEqual(reranker.rerank("q", make_docs(3), top_k=0), [])


class FakeTokenizer:
    def __call__(self, queries, passages, padding=None, truncation=None,
                 max_length=None, return_tensors=None):
        return {"input_ids": torch.ones((len(queries), 3), dtype=torch.long)}


class FakeTorchModel:
    def __init__(self, logits):
        self._logits = logits

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        out = MagicMock()
        out.logits = self._logits
        return out


class TestTransformersSequenceClassifier(unittest.TestCase):
    def _build(self, logits):
        tokenizer = FakeTokenizer()
        model = FakeTorchModel(logits)
        with patch("retrieval.reranker.AutoTokenizer") as mock_tok, \
                patch("retrieval.reranker.AutoModelForSequenceClassification") as mock_model, \
                patch("retrieval.reranker._resolve_local_model_path", return_value=None), \
                patch("config.RERANKER_LOCAL_FILES_ONLY", True):
            mock_tok.from_pretrained.return_value = tokenizer
            mock_model.from_pretrained.return_value = model
            # model_name starting with BAAI/bge-reranker triggers transformers loader path
            return CrossEncoderReranker(
                "BAAI/bge-reranker-base", device="cpu", batch_size=2, max_length=16
            )

    def test_transformers_loader_single_logit_column(self):
        logits = torch.tensor([[0.9], [0.1], [0.5]])
        reranker = self._build(logits)
        self.assertIsInstance(reranker.model, reranker_module._TransformersSequenceClassifier)
        results = reranker.rerank("query", make_docs(3), top_k=3)
        self.assertEqual([d.metadata["chunk_id"] for d in results], ["c0", "c2", "c1"])

    def test_transformers_loader_multi_logit_column(self):
        logits = torch.tensor([[0.1, 0.8], [0.2, 0.3], [0.0, 0.5]])
        reranker = self._build(logits)
        scores = reranker.model.predict([("q", "a"), ("q", "b"), ("q", "c")], batch_size=None)
        self.assertEqual(len(scores), 3)
        for actual, expected in zip(scores, [0.8, 0.3, 0.5]):
            self.assertAlmostEqual(actual, expected, places=5)


class TestResolveLocalModelPath(unittest.TestCase):
    def test_no_cache_dir_returns_none(self):
        with patch("config.HF_CACHE_DIR", ""):
            self.assertIsNone(_resolve_local_model_path("BAAI/bge-reranker-base"))

    def test_missing_snapshots_dir_returns_none(self, ):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with patch("config.HF_CACHE_DIR", tmp):
                self.assertIsNone(_resolve_local_model_path("BAAI/bge-reranker-base"))

    def test_resolves_via_refs_main(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "models--BAAI--bge-reranker-base"
            snap = base / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            refs = base / "refs"
            refs.mkdir(parents=True)
            (refs / "main").write_text("abc123", encoding="utf-8")
            with patch("config.HF_CACHE_DIR", tmp):
                resolved = _resolve_local_model_path("BAAI/bge-reranker-base")
            self.assertEqual(resolved, str(snap))

    def test_falls_back_to_latest_snapshot(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "models--BAAI--bge-reranker-base"
            (base / "snapshots" / "aaa").mkdir(parents=True)
            (base / "snapshots" / "bbb").mkdir(parents=True)
            with patch("config.HF_CACHE_DIR", tmp):
                resolved = _resolve_local_model_path("BAAI/bge-reranker-base")
            self.assertEqual(resolved, str(base / "snapshots" / "bbb"))


class TestGetReranker(unittest.TestCase):
    def tearDown(self):
        reranker_module._reranker = None
        reranker_module._reranker_load_error = None

    def test_returns_cached_instance(self):
        with patch("retrieval.reranker.CrossEncoder", FakeCrossEncoder), \
                patch("config.RERANKER_MODEL", "fake-model"), \
                patch("config.RERANKER_DEVICE", "cpu"):
            first = get_reranker()
            second = get_reranker()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
