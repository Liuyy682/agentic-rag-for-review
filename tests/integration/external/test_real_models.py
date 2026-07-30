import os

import numpy as np
import pytest
from langchain_core.documents import Document

from agentic_rag import config
from agentic_rag.retrieval.embeddings import DenseEmbeddingModel
from agentic_rag.retrieval.reranker import CrossEncoderReranker


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 after downloading the configured models.",
)


def test_real_dense_embedding_is_finite_and_normalized():
    model = DenseEmbeddingModel(device="cpu", local_files_only=True)

    query = model.encode_queries(["什么是持续集成？"])
    document = model.encode_documents(["持续集成会在代码提交后自动运行测试。"])

    assert query.shape == document.shape == (1, config.DENSE_EMBEDDING_DIMENSION)
    assert np.isfinite(query).all()
    assert np.isfinite(document).all()
    assert np.linalg.norm(query[0]) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(document[0]) == pytest.approx(1.0, abs=1e-4)


def test_real_reranker_places_relevant_document_first():
    reranker = CrossEncoderReranker(
        model_name=config.RERANKER_MODEL,
        device="cpu",
        batch_size=2,
        max_length=128,
    )
    documents = [
        Document(page_content="香蕉是一种常见的热带水果。", metadata={"id": "unrelated"}),
        Document(
            page_content="持续集成会在每次代码提交后自动构建并运行测试。",
            metadata={"id": "relevant"},
        ),
    ]

    ranked = reranker.rerank("持续集成有什么作用？", documents, top_k=2)

    assert [item.metadata["rerank_rank"] for item in ranked] == [1, 2]
    assert all(np.isfinite(item.metadata["rerank_score"]) for item in ranked)
    assert ranked[0].metadata["id"] == "relevant"
