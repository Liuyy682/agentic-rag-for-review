from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Callable, Sequence

from langchain_core.documents import Document


def get_doc_key(doc: Document) -> str:
    """Return a stable key for chunk-level deduplication."""
    metadata = doc.metadata or {}
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    if metadata.get("id"):
        return str(metadata["id"])

    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return f"{metadata.get('parent_id', '')}:{content_hash}"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Document]],
    k: int = 60,
    top_k: int = 10,
    key_fn: Callable[[Document], str] | None = None,
) -> list[Document]:
    """Fuse multiple ranked document lists using Reciprocal Rank Fusion."""
    key_fn = key_fn or get_doc_key

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    rank_details: dict[str, dict[str, int]] = {}

    for ranking_index, ranking in enumerate(rankings):
        for rank, doc in enumerate(ranking, start=1):
            key = key_fn(doc)
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            rank_details.setdefault(key, {})[f"rank_{ranking_index}"] = rank

    sorted_keys = sorted(scores, key=lambda key: scores[key], reverse=True)

    fused_docs: list[Document] = []
    for key in sorted_keys[:top_k]:
        doc = deepcopy(docs[key])
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["rrf_score"] = scores[key]
        doc.metadata["rrf_rank_details"] = rank_details.get(key, {})
        fused_docs.append(doc)

    return fused_docs
