from __future__ import annotations

import logging
import time
from copy import deepcopy
from typing import Iterable

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


def _shorten_query(query: str, max_len: int = 200) -> str:
    if query is None:
        return ""
    cleaned = str(query).replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 3]}..."


def resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU")
            return "cpu"
        if requested == "mps":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                logger.warning("MPS requested but not available; falling back to CPU")
                return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def rerank(
        self,
        query: str,
        documents: Iterable[Document],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[Document]:
        docs = list(documents)
        if not docs or not top_k or top_k <= 0:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        start = time.perf_counter()
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(
            zip(docs, [float(score) for score in scores]),
            key=lambda item: item[1],
            reverse=True,
        )

        if score_threshold is not None:
            ranked = [item for item in ranked if item[1] >= score_threshold]

        ranked = ranked[:top_k]

        reranked_docs: list[Document] = []
        for rank, (doc, score) in enumerate(ranked, start=1):
            cloned = deepcopy(doc)
            cloned.metadata = dict(cloned.metadata or {})
            cloned.metadata["rerank_score"] = float(score)
            cloned.metadata["rerank_rank"] = rank
            reranked_docs.append(cloned)

        latency_ms = (time.perf_counter() - start) * 1000
        top_scores = [round(score, 6) for _, score in ranked[:5]]
        logger.info(
            "Reranker run: model=%s device=%s input=%d output=%d latency_ms=%.2f query=%s top_scores=%s",
            self.model_name,
            self.device,
            len(docs),
            len(reranked_docs),
            latency_ms,
            _shorten_query(query),
            top_scores,
        )
        return reranked_docs


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=config.RERANKER_MODEL,
            device=config.RERANKER_DEVICE,
            batch_size=config.RERANKER_BATCH_SIZE,
            max_length=config.RERANKER_MAX_LENGTH,
        )
    return _reranker
