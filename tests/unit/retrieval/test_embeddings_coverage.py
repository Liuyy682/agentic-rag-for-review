import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


import agentic_rag.retrieval.embeddings as embeddings_module
from agentic_rag.retrieval.embeddings import DenseEmbeddingModel, resolve_embedding_device


def fake_torch(cuda_avail, mps_avail, has_mps=True):
    m = MagicMock()
    m.cuda.is_available.return_value = cuda_avail
    if has_mps:
        m.backends.mps.is_available.return_value = mps_avail
    else:
        del m.backends.mps
    return m


class TestSentenceTransformerCls(unittest.TestCase):
    def test_returns_sentence_transformer_class(self):
        cls = embeddings_module._sentence_transformer_cls()
        from sentence_transformers import SentenceTransformer
        self.assertIs(cls, SentenceTransformer)


class TestResolveEmbeddingDevice(unittest.TestCase):
    def test_cuda_requested_but_unavailable_falls_back(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False, False)}):
            self.assertEqual(resolve_embedding_device("cuda"), "cpu")

    def test_cuda_requested_and_available(self):
        with patch.dict(sys.modules, {"torch": fake_torch(True, False)}):
            self.assertEqual(resolve_embedding_device("cuda"), "cuda")

    def test_mps_requested_but_unavailable_falls_back(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False, False)}):
            self.assertEqual(resolve_embedding_device("mps"), "cpu")

    def test_mps_requested_and_available(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False, True)}):
            self.assertEqual(resolve_embedding_device("mps"), "mps")

    def test_auto_prefers_cuda(self):
        with patch.dict(sys.modules, {"torch": fake_torch(True, False)}):
            self.assertEqual(resolve_embedding_device("auto"), "cuda")

    def test_auto_prefers_mps_when_cuda_absent(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False, True)}):
            self.assertEqual(resolve_embedding_device("auto"), "mps")

    def test_auto_falls_back_to_cpu(self):
        with patch.dict(sys.modules, {"torch": fake_torch(False, False, has_mps=False)}):
            self.assertEqual(resolve_embedding_device("auto"), "cpu")


class FakeSentenceTransformer:
    def __init__(self, model_name, cache_folder=None, device=None, local_files_only=None):
        self.model_name = model_name

    def encode(self, texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None):
        return np.ones((len(texts), 3), dtype=np.float32)


class TestPrefixQuery(unittest.TestCase):
    def _model(self, query_instruction):
        with patch("agentic_rag.retrieval.embeddings._sentence_transformer_cls", return_value=FakeSentenceTransformer):
            return DenseEmbeddingModel(
                model_name="fake-model",
                device="cpu",
                batch_size=4,
                query_instruction=query_instruction,
                normalize_embeddings=False,
                local_files_only=True,
            )

    def test_empty_instruction_returns_query_unchanged(self):
        model = self._model(query_instruction="")
        self.assertEqual(model._prefix_query("hello"), "hello")

    def test_none_query_with_empty_instruction(self):
        model = self._model(query_instruction="")
        self.assertEqual(model._prefix_query(None), "")

    def test_instruction_is_prepended(self):
        model = self._model(query_instruction="检索：")
        self.assertEqual(model._prefix_query("索引"), "检索：索引")


if __name__ == "__main__":
    unittest.main()
