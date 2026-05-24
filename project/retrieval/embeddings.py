from __future__ import annotations

from typing import Iterable

import numpy as np

import config


def _sentence_transformer_cls():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


def resolve_embedding_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested != "auto":
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        if requested == "mps":
            import torch

            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                return "cpu"
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseEmbeddingModel:
    """Project embedding adapter with BGE query-instruction support."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_folder: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        query_instruction: str | None = None,
        normalize_embeddings: bool | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name or config.DENSE_MODEL
        self.cache_folder = cache_folder if cache_folder is not None else getattr(config, "HF_CACHE_DIR", None)
        self.device = resolve_embedding_device(device or config.DENSE_EMBEDDING_DEVICE)
        self.batch_size = batch_size or config.DENSE_EMBEDDING_BATCH_SIZE
        self.query_instruction = (
            config.DENSE_QUERY_INSTRUCTION if query_instruction is None else query_instruction
        )
        self.normalize_embeddings = (
            config.DENSE_NORMALIZE_EMBEDDINGS if normalize_embeddings is None else normalize_embeddings
        )
        self.local_files_only = (
            config.DENSE_LOCAL_FILES_ONLY if local_files_only is None else local_files_only
        )

        sentence_transformer = _sentence_transformer_cls()
        self.model = sentence_transformer(
            self.model_name,
            cache_folder=self.cache_folder,
            device=self.device,
            local_files_only=self.local_files_only,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.encode_documents(texts).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.encode_queries([query])[0].tolist()

    def encode_documents(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        return self._encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def encode_queries(
        self,
        queries: Iterable[str],
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prefixed = [self._prefix_query(query) for query in queries]
        return self._encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def _encode(
        self,
        texts: list[str],
        batch_size: int | None,
        show_progress_bar: bool,
    ) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _prefix_query(self, query: str) -> str:
        query_text = "" if query is None else str(query)
        if not self.query_instruction:
            return query_text
        return f"{self.query_instruction}{query_text}"
