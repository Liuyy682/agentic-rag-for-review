from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from core.rag_system import RAGSystem
from evaluation.io import config_snapshot, make_run_id, write_jsonl, write_metrics_csv
from evaluation.ragbench_keys import document_id_from_sentence_key
from evaluation.validation import build_validity_summary, make_warning, write_validation_outputs
from ingestion.document_manager import DocumentManager
from retrieval.pipeline import RetrievalPipeline


DEFAULT_K_VALUES = [1, 3, 5, 10, 20]


def run_ingestion_retrieval_eval(
    source_contexts: str,
    output_dir: str,
    run_label: str,
    limit: int,
    top_k: int,
    k_values: Sequence[int],
    use_reranker: bool,
    reranker_final_top_k: int | None,
) -> Dict[str, Any]:
    run_id = make_run_id(run_label)
    run_dir = Path(output_dir) / "eval_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(source_contexts)[:limit]
    _configure_isolated_runtime(run_dir)
    config.RERANKER_ENABLED = use_reranker
    if reranker_final_top_k is not None:
        config.RERANKER_FINAL_TOP_K = reranker_final_top_k

    source_docs = _write_ragbench_markdown_docs(rows, run_dir / "input_docs")
    rag_system = RAGSystem()
    manager = DocumentManager(rag_system)
    manager.clear_all()
    ingestion_results = manager.add_documents_detailed(
        [str(doc["path"]) for doc in source_docs],
        course_names=["ragbench"],
    )
    ingestion_rows = [_ingestion_result_to_dict(item) for item in ingestion_results.documents]

    pipeline = RetrievalPipeline(
        vector_db=rag_system.vector_db,
        parent_store_manager=rag_system.parent_store,
    )
    sentence_index = _build_sentence_index(rows)
    k_values = sorted({int(k) for k in k_values if int(k) > 0} | {int(top_k)})

    retrieval_rows = []
    per_question = []
    rag_outputs = []
    for row in rows:
        query = row.get("question", "")
        child_docs = pipeline.search_child_chunk_documents(query, top_k)
        reranked_docs = pipeline.rerank_child_documents(query, child_docs)[:top_k]
        retrieved_chunks = [_doc_to_retrieved_chunk(doc, rank) for rank, doc in enumerate(reranked_docs, start=1)]
        contexts = _contexts_for_docs(pipeline, query, reranked_docs)
        context_texts = [context.get("content", "") for context in contexts]

        retrieval_row = {
            "question_id": row["question_id"],
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "contexts": contexts,
        }
        retrieval_rows.append(retrieval_row)
        per_question.append(_score_question(row, retrieved_chunks, contexts, sentence_index, k_values))
        rag_outputs.append(
            {
                "question_id": row["question_id"],
                "question": query,
                "user_input": query,
                "answer": row.get("reference_response", ""),
                "response": row.get("reference_response", ""),
                "contexts": context_texts,
                "retrieved_contexts": context_texts,
                "reference": row.get("reference_response", ""),
                "ground_truth": row.get("reference_response", ""),
                "answer_source": "reference",
                "answer_model": None,
                "retrieved_metadata": [
                    {
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "parent_id": chunk["parent_id"],
                        "source_file": chunk["source_file"],
                        "source_doc_id": chunk["source_doc_id"],
                        "score_fused": chunk.get("score_fused"),
                        "score_rerank": chunk.get("score_rerank"),
                    }
                    for chunk in retrieved_chunks
                ],
                "retrieval_context_policy": contexts[0].get("context_type", "") if contexts else "",
            }
        )

    metrics = _aggregate(per_question, k_values)
    warnings = _validation_warnings(rows, ingestion_rows, retrieval_rows, per_question, k_values)
    validity_summary = build_validity_summary(
        rows=len(rows),
        warnings=warnings,
        evaluation_type="ragbench_project_ingestion_retrieval_eval",
    )

    metadata = config_snapshot(str(run_id), source_contexts, "ragbench_project_ingestion", top_k, None)
    metadata.update(
        {
            "evaluation_type": "ragbench_project_ingestion_retrieval_eval",
            "ragbench_rows": len(rows),
            "source_contexts_input": source_contexts,
            "uses_project_ingestion": True,
            "uses_project_retriever": True,
            "uses_synthetic_document_chunks": False,
            "generated_input_documents": len(source_docs),
            "reranker_enabled": config.RERANKER_ENABLED,
            "reranker_final_top_k": config.RERANKER_FINAL_TOP_K,
            "retrieval_context_policy": config.RETRIEVAL_CONTEXT_POLICY,
            "k_values": k_values,
            "validity_summary": validity_summary,
        }
    )

    write_jsonl(run_dir / "ingestion_results.jsonl", ingestion_rows)
    write_jsonl(run_dir / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(run_dir / "retrieval_per_question_metrics.jsonl", per_question)
    write_jsonl(run_dir / "rag_outputs.jsonl", rag_outputs)
    write_metrics_csv(run_dir / "retrieval_metrics_summary.csv", metrics)
    write_validation_outputs(run_dir, warnings, validity_summary)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_dir = Path(output_dir) / "ragbench_project_ingestion_latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "metrics": metrics,
        "validity_summary": validity_summary,
    }


def _configure_isolated_runtime(run_dir: Path) -> None:
    runtime = run_dir / "ingestion_runtime"
    config.INDEX_STATE_DIR = str(runtime / "index_state")
    config.COURSE_STRUCTURE_PATH = str(Path(config.INDEX_STATE_DIR) / "course_structure.json")
    config.MARKDOWN_DIR = str(runtime / "markdown_docs")
    config.MARKDOWN_CLEANED_DIR = str(runtime / "markdown_docs_cleaned")
    config.MARKDOWN_CLEANING_LOG_DIR = str(runtime / "markdown_cleaning_logs")
    config.MARKDOWN_CLEANING_DIFF_DIR = str(runtime / "markdown_cleaning_diffs")
    config.DOCUMENT_IMAGE_DIR = str(runtime / "document_images")
    config.INGESTION_LOG_DIR = str(runtime / "ingestion_logs")


def _write_ragbench_markdown_docs(rows: Sequence[Dict[str, Any]], output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for row in rows:
        question_id = row["question_id"]
        for index, text in enumerate(row.get("documents") or []):
            doc_id = f"{question_id}_doc_{index}"
            path = output_dir / f"{doc_id}.md"
            path.write_text(str(text).strip() + "\n", encoding="utf-8")
            docs.append({"doc_id": doc_id, "path": path})
    return docs


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ingestion_result_to_dict(result: Any) -> Dict[str, Any]:
    return {
        "source_path": result.source_path,
        "source_file": result.source_file,
        "status": result.status,
        "reason": result.reason,
        "indexed": result.indexed,
        "parent_count": result.parent_count,
        "child_count": result.child_count,
        "error": result.error,
        "stages": [
            {
                "name": stage.name,
                "status": stage.status,
                "elapsed_ms": stage.elapsed_ms,
                "details": stage.details,
                "error": stage.error,
            }
            for stage in result.stages
        ],
    }


def _doc_to_retrieved_chunk(doc: Any, rank: int) -> Dict[str, Any]:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source_file = str(metadata.get("source_file") or metadata.get("source") or "")
    return {
        "rank": rank,
        "chunk_id": str(metadata.get("chunk_id") or metadata.get("child_id") or ""),
        "parent_id": str(metadata.get("parent_id") or ""),
        "source_file": source_file,
        "source_doc_id": Path(source_file).stem,
        "score_fused": _float_or_none(metadata.get("rrf_score")),
        "score_rerank": _float_or_none(metadata.get("rerank_score")),
        "metadata": metadata,
        "text": str(getattr(doc, "page_content", "") or "").strip(),
    }


def _contexts_for_docs(pipeline: RetrievalPipeline, query: str, docs: Sequence[Any]) -> List[Dict[str, Any]]:
    parent_ids = []
    for doc in docs:
        parent_id = (getattr(doc, "metadata", {}) or {}).get("parent_id", "")
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)
    policy, _reason = pipeline.select_context_policy(query, list(docs), [])
    return pipeline.contexts_for_policy(policy, parent_ids, list(docs))


def _build_sentence_index(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        question_id = row["question_id"]
        for doc_index, sentence_pairs in enumerate(row.get("documents_sentences") or []):
            source_doc_id = f"{question_id}_doc_{doc_index}"
            index[source_doc_id] = [
                {"key": f"{source_doc_id}:{key}", "text": str(text)}
                for key, text in sentence_pairs
                if str(text).strip()
            ]
    return index


def _score_question(
    row: Dict[str, Any],
    retrieved_chunks: Sequence[Dict[str, Any]],
    contexts: Sequence[Dict[str, Any]],
    sentence_index: Dict[str, List[Dict[str, str]]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    raw_gold_sentence_keys = [str(key) for key in (row.get("all_relevant_sentence_keys") or []) if str(key)]
    gold_doc_ids = {
        f"{row['question_id']}_doc_{document_id_from_sentence_key(key)}"
        for key in raw_gold_sentence_keys
        if document_id_from_sentence_key(key)
    }
    gold_sentence_keys = {
        f"{row['question_id']}_doc_{document_id_from_sentence_key(key)}:{key}"
        for key in raw_gold_sentence_keys
        if document_id_from_sentence_key(key)
    }
    scored = bool(gold_doc_ids)
    result: Dict[str, Any] = {
        "question_id": row["question_id"],
        "question": row.get("question", ""),
        "scored": scored,
        "gold_doc_count": len(gold_doc_ids),
        "gold_sentence_count": len(gold_sentence_keys),
        "mrr": _reciprocal_rank(retrieved_chunks, gold_doc_ids),
        "warnings": [] if scored else ["missing_primary_gold"],
    }
    for k in k_values:
        top = list(retrieved_chunks[:k])
        hit_docs = {chunk["source_doc_id"] for chunk in top if chunk["source_doc_id"] in gold_doc_ids}
        relevant_chunk_hits = [chunk for chunk in top if chunk["source_doc_id"] in gold_doc_ids]
        top_contexts = _contexts_matching_top(contexts, top)
        context_keys = _sentence_keys_for_contexts(top_contexts, sentence_index)
        sentence_hits = context_keys & gold_sentence_keys

        result[f"actual_results@{k}"] = float(len(top))
        result[f"hitrate@{k}"] = 1.0 if hit_docs else 0.0
        result[f"source_hitrate@{k}"] = 1.0 if hit_docs else 0.0
        result[f"recall@{k}"] = _ratio(len(hit_docs), len(gold_doc_ids))
        result[f"doc_recall@{k}"] = result[f"recall@{k}"]
        result[f"precision@{k}"] = len(relevant_chunk_hits) / k if k else 0.0
        result[f"doc_precision@{k}"] = result[f"precision@{k}"]
        result[f"sentence_recall@{k}"] = _ratio(len(sentence_hits), len(gold_sentence_keys))
        result[f"sentence_precision@{k}"] = _ratio(len(sentence_hits), len(context_keys))
        result[f"context_count@{k}"] = float(len(top_contexts))
        result[f"context_chars@{k}"] = float(sum(len(context.get("content", "")) for context in top_contexts))
        result[f"ndcg@{k}"] = _ndcg(top, gold_doc_ids, k)
    return result


def _contexts_matching_top(contexts: Sequence[Dict[str, Any]], top_chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_parent_ids = {chunk.get("parent_id") for chunk in top_chunks if chunk.get("parent_id")}
    allowed_child_ids = {chunk.get("chunk_id") for chunk in top_chunks if chunk.get("chunk_id")}
    matched = []
    for context in contexts:
        parent_id = context.get("parent_id")
        child_id = context.get("child_id")
        if parent_id in allowed_parent_ids or child_id in allowed_child_ids:
            matched.append(context)
    return matched


def _sentence_keys_for_contexts(
    contexts: Sequence[Dict[str, Any]],
    sentence_index: Dict[str, List[Dict[str, str]]],
) -> set[str]:
    keys: set[str] = set()
    for context in contexts:
        source = str(context.get("source") or "")
        source_doc_id = Path(source).stem
        normalized_content = _normalize_text(context.get("content", ""))
        for sentence in sentence_index.get(source_doc_id, []):
            if _normalize_text(sentence["text"]) in normalized_content:
                keys.add(sentence["key"])
    return keys


def _aggregate(per_question: Sequence[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
    metrics = ["mrr"]
    for k in k_values:
        metrics.extend(
            [
                f"actual_results@{k}",
                f"context_chars@{k}",
                f"context_count@{k}",
                f"doc_precision@{k}",
                f"doc_recall@{k}",
                f"hitrate@{k}",
                f"ndcg@{k}",
                f"precision@{k}",
                f"recall@{k}",
                f"sentence_precision@{k}",
                f"sentence_recall@{k}",
                f"source_hitrate@{k}",
            ]
        )
    scored_rows = [row for row in per_question if row.get("scored")]
    aggregate = {metric: _mean(row.get(metric, 0.0) for row in scored_rows) for metric in metrics}
    aggregate.update(
        {
            "rows": float(len(per_question)),
            "scored_rows": float(len(scored_rows)),
            "unscored_rows": float(len(per_question) - len(scored_rows)),
            "warning_count": float(sum(len(row.get("warnings", [])) for row in per_question)),
            "evaluation_valid": 1.0 if per_question and scored_rows else 0.0,
        }
    )
    return aggregate


def _validation_warnings(
    rows: Sequence[Dict[str, Any]],
    ingestion_rows: Sequence[Dict[str, Any]],
    retrieval_rows: Sequence[Dict[str, Any]],
    per_question: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    if not rows:
        warnings.append(make_warning("empty_dataset", "No RAGBench rows were available.", severity="error"))
    failed_ingestion = [row for row in ingestion_rows if row.get("status") == "failed"]
    if failed_ingestion:
        warnings.append(
            make_warning(
                "ingestion_failed",
                "One or more generated Markdown documents failed project ingestion.",
                severity="error",
                details={"failed_count": len(failed_ingestion)},
            )
        )
    for row in per_question:
        if not row.get("scored"):
            warnings.append(
                make_warning(
                    "missing_primary_gold",
                    "Question has no RAGBench relevant sentence keys; retrieval metrics exclude this row.",
                    question_id=row.get("question_id"),
                )
            )
    for result in retrieval_rows:
        count = len(result.get("retrieved_chunks") or [])
        for k in k_values:
            if count < k:
                warnings.append(
                    make_warning(
                        "insufficient_results_for_k",
                        "Retrieved result depth is smaller than a declared metric cutoff.",
                        question_id=result.get("question_id"),
                        details={"k": int(k), "actual_results": count},
                    )
                )
    return warnings


def _reciprocal_rank(chunks: Sequence[Dict[str, Any]], gold_doc_ids: set[str]) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get("source_doc_id") in gold_doc_ids:
            return 1 / rank
    return 0.0


def _ndcg(chunks: Sequence[Dict[str, Any]], gold_doc_ids: set[str], k: int) -> float:
    gains = [1 if chunk.get("source_doc_id") in gold_doc_ids else 0 for chunk in chunks[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = [1] * min(len(gold_doc_ids), k)
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGBench retrieval eval through the project's ingestion pipeline.")
    parser.add_argument("--source-contexts", default=str(PROJECT_DIR / "evaluation" / "datasets" / "ragbench_covidqa_test_200_source_contexts.jsonl"))
    parser.add_argument("--output-dir", default=str(Path(config.EVALUATION_REPORTS_DIR) / "ragbench_project_ingestion"))
    parser.add_argument("--run-label", default="ragbench_project_ingestion")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker-final-top-k", type=int, default=None)
    args = parser.parse_args()

    if args.reranker is not None:
        use_reranker = args.reranker
    else:
        use_reranker = config.RERANKER_ENABLED

    result = run_ingestion_retrieval_eval(
        source_contexts=args.source_contexts,
        output_dir=args.output_dir,
        run_label=args.run_label,
        limit=args.limit,
        top_k=args.top_k,
        k_values=[int(item) for item in args.k_values.split(",") if item.strip()],
        use_reranker=use_reranker,
        reranker_final_top_k=args.reranker_final_top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
