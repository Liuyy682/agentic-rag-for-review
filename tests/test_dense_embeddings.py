import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from retrieval.embeddings import DenseEmbeddingModel


class FakeSentenceTransformer:
    calls = []

    def __init__(self, model_name, cache_folder=None, device=None, local_files_only=None):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.local_files_only = local_files_only

    def encode(self, texts, batch_size=None, show_progress_bar=None, normalize_embeddings=None):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return np.ones((len(texts), 3), dtype=np.float32)


class TestDenseEmbeddingModel(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.calls = []

    def test_documents_are_normalized_without_query_instruction(self):
        with patch("retrieval.embeddings._sentence_transformer_cls", return_value=FakeSentenceTransformer):
            model = DenseEmbeddingModel(
                model_name="fake-model",
                device="cpu",
                batch_size=7,
                query_instruction="检索：",
                normalize_embeddings=True,
                local_files_only=True,
            )
            embeddings = model.embed_documents(["文档内容"])

        self.assertEqual(embeddings, [[1.0, 1.0, 1.0]])
        self.assertEqual(FakeSentenceTransformer.calls[0]["texts"], ["文档内容"])
        self.assertEqual(FakeSentenceTransformer.calls[0]["batch_size"], 7)
        self.assertTrue(FakeSentenceTransformer.calls[0]["normalize_embeddings"])

    def test_query_uses_instruction_and_normalization(self):
        with patch("retrieval.embeddings._sentence_transformer_cls", return_value=FakeSentenceTransformer):
            model = DenseEmbeddingModel(
                model_name="fake-model",
                device="cpu",
                batch_size=7,
                query_instruction="检索：",
                normalize_embeddings=True,
                local_files_only=True,
            )
            embedding = model.embed_query("数据库索引是什么")

        self.assertEqual(embedding, [1.0, 1.0, 1.0])
        self.assertEqual(FakeSentenceTransformer.calls[0]["texts"], ["检索：数据库索引是什么"])
        self.assertTrue(FakeSentenceTransformer.calls[0]["normalize_embeddings"])


if __name__ == "__main__":
    unittest.main()
