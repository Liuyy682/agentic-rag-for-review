from typing import Any, Dict


def build_ragbench_local_eval_metadata(
    subset: str,
    split: str,
    limit: int,
    offset: int,
) -> Dict[str, Any]:
    return {
        "evaluation_type": "synthetic_ragbench_document_retrieval",
        "ragbench_subset": subset,
        "ragbench_split": split,
        "ragbench_limit": limit,
        "ragbench_offset": offset,
        "uses_project_retriever": True,
        "uses_synthetic_document_chunks": True,
        "notes": "Each RAGBench document is indexed as one chunk; this evaluates retrieval strategy, not the project's production ingestion chunking.",
    }
