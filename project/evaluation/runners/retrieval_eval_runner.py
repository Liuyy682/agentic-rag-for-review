import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from db.vector_db_manager import VectorDbManager
from evaluation.data import load_eval_questions
from evaluation.io import config_snapshot, make_run_id, write_jsonl, write_metrics_csv
from evaluation.metrics.retrieval_metrics import (
    DEFAULT_K_VALUES,
    build_retrieval_error_cases,
    compute_retrieval_metrics,
)
from evaluation.reports import write_retrieval_report


def run_retrieval_eval(
    dataset_path: str,
    output_dir: str,
    run_label: str,
    top_k: int,
    collection_name: str,
    dataset_version: str,
    score_threshold: float | None,
) -> Dict[str, Any]:
    questions = load_eval_questions(dataset_path)
    run_id = make_run_id(run_label)
    run_dir = Path(output_dir) / "eval_runs" / run_id
    reports_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    vector_db = VectorDbManager()
    vector_db.create_collection(collection_name)
    collection = vector_db.get_collection(collection_name)

    results = [
        {
            "question_id": item.question_id,
            "query": item.question,
            "retrieved_chunks": retrieve_chunks(
                collection,
                item.question,
                top_k,
                score_threshold,
                vector_db=vector_db,
                collection_name=collection_name,
            ),
        }
        for item in questions
    ]

    k_values = sorted(set(DEFAULT_K_VALUES + [top_k]))
    metrics, per_question = compute_retrieval_metrics(questions, results, k_values=k_values)
    error_cases = build_retrieval_error_cases(questions, results, per_question, top_k=top_k)
    metadata = config_snapshot(run_id, dataset_path, dataset_version, top_k, score_threshold)

    write_jsonl(run_dir / "retrieval_results.jsonl", results)
    write_jsonl(run_dir / "retrieval_per_question_metrics.jsonl", per_question)
    write_jsonl(run_dir / "retrieval_error_cases.jsonl", error_cases)
    write_metrics_csv(run_dir / "retrieval_metrics_summary.csv", metrics)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    write_jsonl(reports_dir / "retrieval_results.jsonl", results)
    write_jsonl(reports_dir / "retrieval_error_cases.jsonl", error_cases)
    write_metrics_csv(reports_dir / "retrieval_metrics_summary.csv", metrics)
    write_retrieval_report(reports_dir / "retrieval_report.md", metadata, questions, metrics, error_cases)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(reports_dir / "retrieval_report.md"),
        "metrics": metrics,
    }


def retrieve_chunks(
    collection: Any,
    query: str,
    top_k: int,
    score_threshold: float | None = None,
    vector_db: VectorDbManager | None = None,
    collection_name: str | None = None,
) -> List[Dict[str, Any]]:
    candidate_k = top_k
    if config.RERANKER_ENABLED:
        candidate_k = max(candidate_k, config.RERANKER_TOP_N)

    docs_with_scores = _retrieve_candidate_docs(
        collection=collection,
        query=query,
        candidate_k=candidate_k,
        score_threshold=score_threshold,
        vector_db=vector_db,
        collection_name=collection_name,
    )

    docs_with_scores = _rerank_candidates(query, docs_with_scores, top_k)

    chunks = []
    for rank, (doc, score) in enumerate(docs_with_scores, start=1):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        parent_id = metadata.get("parent_id")
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("child_id")
            or metadata.get("_id")
            or metadata.get("id")
            or f"{parent_id or 'unknown_parent'}::rank_{rank}"
        )
        chunks.append(
            {
                "rank": rank,
                "chunk_id": str(chunk_id),
                "parent_id": str(parent_id or ""),
                "source_file": str(metadata.get("source") or metadata.get("source_file") or ""),
                "score_dense": None,
                "score_sparse": None,
                "score_fused": _float_or_none(metadata.get("rrf_score")) or _float_or_none(score),
                "score_rerank": _float_or_none(metadata.get("rerank_score")),
                "metadata": metadata,
                "text": getattr(doc, "page_content", "").strip(),
            }
        )
    return chunks


def _retrieve_candidate_docs(
    collection: Any,
    query: str,
    candidate_k: int,
    score_threshold: float | None,
    vector_db: VectorDbManager | None,
    collection_name: str | None,
) -> List[tuple[Any, float | None]]:
    mode = config.RETRIEVAL_FUSION_MODE
    if mode == "rrf" and vector_db and collection_name:
        docs = vector_db.rrf_search(
            collection_name=collection_name,
            query=query,
            dense_k=config.DENSE_TOP_K,
            sparse_k=config.SPARSE_TOP_K,
            fused_k=candidate_k,
            rrf_k=config.RRF_K,
        )
        return [(doc, doc.metadata.get("rrf_score") if doc.metadata else None) for doc in docs]
    if mode == "dense" and vector_db and collection_name:
        return [(doc, None) for doc in vector_db.dense_search(collection_name, query, k=candidate_k)]
    if mode == "sparse" and vector_db and collection_name:
        return [(doc, None) for doc in vector_db.sparse_search(collection_name, query, k=candidate_k)]

    kwargs = {"score_threshold": score_threshold} if score_threshold is not None else {}
    try:
        raw_results = collection.similarity_search_with_score(query, k=candidate_k, **kwargs)
        return [(doc, score) for doc, score in raw_results]
    except Exception:
        docs = collection.similarity_search(query, k=candidate_k, **kwargs)
        return [(doc, None) for doc in docs]


def _rerank_candidates(
    query: str,
    docs_with_scores: List[tuple[Any, float | None]],
    top_k: int,
) -> List[tuple[Any, float | None]]:
    if not config.RERANKER_ENABLED:
        return docs_with_scores[:top_k]

    docs = [doc for doc, _ in docs_with_scores[: config.RERANKER_TOP_N]]
    try:
        from rag_agent.reranker import get_reranker

        reranked = get_reranker().rerank(
            query=query,
            documents=docs,
            top_k=min(top_k, config.RERANKER_FINAL_TOP_K, len(docs)),
            score_threshold=config.RERANKER_SCORE_THRESHOLD,
        )
    except Exception:
        return docs_with_scores[:top_k]

    from rag_agent.retrieval_fusion import get_doc_key

    original_scores = {get_doc_key(doc): score for doc, score in docs_with_scores}
    return [(doc, original_scores.get(get_doc_key(doc))) for doc in reranked]


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for the current RAG retriever.")
    parser.add_argument("--dataset", default=str(PROJECT_DIR / "evaluation" / "datasets" / "eval_questions.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "evaluation" / "reports"))
    parser.add_argument("--run-label", default="baseline")
    parser.add_argument("--dataset-version", default="eval_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--collection", default=config.CHILD_COLLECTION)
    args = parser.parse_args()

    result = run_retrieval_eval(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        run_label=args.run_label,
        top_k=args.top_k,
        collection_name=args.collection,
        dataset_version=args.dataset_version,
        score_threshold=args.score_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
