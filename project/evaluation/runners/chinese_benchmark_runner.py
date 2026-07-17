from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from core.rag_system import RAGSystem
from evaluation.chinese_benchmarks import (
    BenchmarkDocument,
    BenchmarkQuestion,
    load_crud_benchmark,
    load_t2_benchmark,
)
from evaluation.io import config_snapshot, make_run_id, write_jsonl, write_metrics_csv
from evaluation.llm_config import answer_model as resolve_answer_model
from evaluation.llm_config import judge_model as resolve_judge_model
from evaluation.metrics.ragas_metrics import (
    build_ragas_error_cases,
    run_ragas_metrics,
    summarize_ragas_rows,
)
from evaluation.runners.ragbench_ingestion_retrieval_runner import (
    _configure_isolated_runtime,
    _contexts_for_docs,
    _doc_to_retrieved_chunk,
    _ingestion_result_to_dict,
)
from evaluation.runners.ragbench_local_rag_runner import (
    _AgentGraphAnswerGenerator,
    _AnswerGenerator,
)
from evaluation.validation import (
    build_validity_summary,
    make_warning,
    validate_ragas_rows,
    write_validation_outputs,
)
from ingestion.document_manager import DocumentManager
from retrieval.pipeline import RetrievalPipeline
from storage.postgres import reset_pool_for_tests


DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


@contextmanager
def experiment_config(
    retrieval_mode: str | None = None,
    reranker_enabled: bool | None = None,
    reranker_final_top_k: int | None = None,
    reranker_score_threshold: float | None = None,
    context_policy: str | None = None,
):
    overrides = {
        "RETRIEVAL_FUSION_MODE": retrieval_mode,
        "RERANKER_ENABLED": reranker_enabled,
        "RERANKER_FINAL_TOP_K": reranker_final_top_k,
        "RERANKER_SCORE_THRESHOLD": reranker_score_threshold,
        "RETRIEVAL_CONTEXT_POLICY": context_policy,
    }
    original = {name: getattr(config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is not None:
                setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def configure_eval_database(environ: dict[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    eval_url = str(values.get("EVAL_DATABASE_URL") or "").strip()
    runtime_url = str(values.get("DATABASE_URL") or config.DATABASE_URL or "").strip()
    if not eval_url:
        raise RuntimeError(
            "EVAL_DATABASE_URL is required because this evaluation clears and rebuilds its PostgreSQL index."
        )
    if eval_url == runtime_url:
        raise RuntimeError("EVAL_DATABASE_URL must not equal DATABASE_URL; refusing to clear the runtime knowledge base.")
    config.DATABASE_URL = eval_url
    reset_pool_for_tests()
    return eval_url


def run_chinese_benchmark(
    dataset: str,
    limit: int,
    offset: int,
    distractor_docs: int,
    top_k: int,
    output_dir: str,
    crud_root: str | None = None,
    skip_ragas: bool = False,
    answer_mode: str | None = None,
    run_label: str | None = None,
    retrieval_mode: str | None = None,
    reranker_enabled: bool | None = None,
    reranker_final_top_k: int | None = None,
    reranker_score_threshold: float | None = None,
    context_policy: str | None = None,
    context_top_k: int = 3,
) -> dict[str, Any]:
    if retrieval_mode not in {None, "dense", "rrf", "sparse"}:
        raise ValueError("retrieval_mode must be dense, rrf, or sparse")
    if context_policy not in {None, "child", "neighbor", "parent", "adaptive"}:
        raise ValueError("context_policy must be child, neighbor, parent, or adaptive")
    if context_top_k <= 0:
        raise ValueError("context_top_k must be positive")
    if reranker_final_top_k is not None and reranker_final_top_k <= 0:
        raise ValueError("reranker_final_top_k must be positive")
    with experiment_config(
        retrieval_mode=retrieval_mode,
        reranker_enabled=reranker_enabled,
        reranker_final_top_k=reranker_final_top_k,
        reranker_score_threshold=reranker_score_threshold,
        context_policy=context_policy,
    ):
        return _run_chinese_benchmark(
            dataset=dataset,
            limit=limit,
            offset=offset,
            distractor_docs=distractor_docs,
            top_k=top_k,
            output_dir=output_dir,
            crud_root=crud_root,
            skip_ragas=skip_ragas,
            answer_mode=answer_mode,
            run_label=run_label,
            context_top_k=context_top_k,
        )


def _run_chinese_benchmark(
    dataset: str,
    limit: int,
    offset: int,
    distractor_docs: int,
    top_k: int,
    output_dir: str,
    crud_root: str | None,
    skip_ragas: bool,
    answer_mode: str | None,
    run_label: str | None,
    context_top_k: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if limit <= 0:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if distractor_docs < 0:
        raise ValueError("distractor_docs must be non-negative; use 0 for the full corpus")

    if dataset == "t2_retrieval":
        if answer_mode not in {None, "reference"}:
            raise ValueError("T2Retrieval is retrieval-only; --answer-mode must be omitted or set to reference.")
        resolved_answer_mode = "reference"
        questions, documents = load_t2_benchmark(limit, offset, distractor_docs)
        dataset_version = "C-MTEB/T2Retrieval:dev"
        dataset_source = "C-MTEB/T2Retrieval + C-MTEB/T2Retrieval-qrels"
        evaluation_type = "t2_project_ingestion_retrieval_eval"
    elif dataset == "crud_rag":
        if not crud_root:
            raise ValueError("--crud-root is required for CRUD-RAG evaluation")
        resolved_answer_mode = answer_mode or "agent"
        questions, documents = load_crud_benchmark(crud_root, limit, offset, distractor_docs)
        dataset_version = "IAAR-Shanghai/CRUD_RAG:quest_answer"
        dataset_source = str(Path(crud_root).resolve())
        evaluation_type = "crud_rag_project_ingestion_end_to_end_eval"
    else:
        raise ValueError(f"Unsupported Chinese benchmark: {dataset}")

    effective_label = run_label or f"{dataset}_{resolved_answer_mode}"
    run_id = make_run_id(effective_label)
    output = Path(output_dir)
    run_dir = output / "eval_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_isolated_runtime(run_dir)
    eval_database_url = configure_eval_database()

    materialized_docs, source_name_to_id = materialize_documents(documents, run_dir / "input_docs")
    rag_system = RAGSystem()
    manager = DocumentManager(rag_system)
    manager.clear_all()
    ingestion_started_at = time.perf_counter()
    ingestion_result = manager.add_documents_detailed(
        [str(item["path"]) for item in materialized_docs],
        course_names=[dataset],
    )
    ingestion_seconds = time.perf_counter() - ingestion_started_at
    ingestion_rows = [_ingestion_result_to_dict(item) for item in ingestion_result.documents]

    pipeline = RetrievalPipeline(
        vector_db=rag_system.vector_db,
        parent_store_manager=rag_system.parent_store,
    )
    answer_generator: Any = None
    resolved_model: str | None = None
    if dataset == "crud_rag" and resolved_answer_mode != "reference":
        resolved_model = resolve_answer_model(None)
        answer_generator = (
            _AgentGraphAnswerGenerator(rag_system.vector_db, rag_system.parent_store, resolved_model)
            if resolved_answer_mode == "agent"
            else _AnswerGenerator(resolved_model)
        )

    k_values = sorted({k for k in DEFAULT_K_VALUES if k <= top_k} | {top_k})
    retrieval_rows: list[dict[str, Any]] = []
    per_question: list[dict[str, Any]] = []
    rag_outputs: list[dict[str, Any]] = []
    runtime_warnings: list[dict[str, Any]] = []
    partial_path = run_dir / "rag_outputs.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question.question_id}", flush=True)
        diagnostics: dict[str, Any] = {}
        try:
            retrieval_started_at = time.perf_counter()
            child_docs = pipeline.search_child_chunk_documents(question.question, top_k)
            reranked_docs = pipeline.rerank_child_documents(question.question, child_docs)[:top_k]
            retrieval_latency_ms = (time.perf_counter() - retrieval_started_at) * 1000.0
            chunks = [_doc_to_retrieved_chunk(doc, rank) for rank, doc in enumerate(reranked_docs, start=1)]
            for chunk in chunks:
                chunk["source_doc_id"] = source_name_to_id.get(chunk["source_doc_id"], chunk["source_doc_id"])
            contexts = _contexts_for_docs(pipeline, question.question, reranked_docs[:context_top_k])
            context_texts = [str(item.get("content") or "").strip() for item in contexts if item.get("content")]
        except Exception as exc:
            chunks = []
            context_texts = []
            retrieval_latency_ms = 0.0
            diagnostics.update(retrieval_error_type=type(exc).__name__, retrieval_error=str(exc))
            runtime_warnings.append(
                make_warning(
                    "retrieval_failed",
                    "Project retrieval failed for a benchmark question.",
                    severity="error",
                    question_id=question.question_id,
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )

        retrieval_row = {
            "question_id": question.question_id,
            "query": question.question,
            "retrieved_chunks": chunks,
        }
        retrieval_rows.append(retrieval_row)
        question_metrics = score_retrieval_question(question, chunks, k_values)
        question_metrics.update(
            {
                "retrieval_latency_ms": retrieval_latency_ms,
                "context_count": float(len(context_texts)),
                "context_chars": float(sum(len(item) for item in context_texts)),
            }
        )
        per_question.append(question_metrics)

        if dataset == "crud_rag":
            try:
                generation_started_at = time.perf_counter()
                answer, answer_contexts, answer_diagnostics = generate_answer(
                    question,
                    context_texts,
                    resolved_answer_mode,
                    answer_generator,
                )
                diagnostics["generation_latency_ms"] = (time.perf_counter() - generation_started_at) * 1000.0
                diagnostics.update(answer_diagnostics)
            except Exception as exc:
                answer = ""
                answer_contexts = context_texts
                diagnostics.update(answer_error_type=type(exc).__name__, answer_error=str(exc))
                runtime_warnings.append(
                    make_warning(
                        "answer_generation_failed",
                        "Answer generation failed for a CRUD-RAG question.",
                        severity="error",
                        question_id=question.question_id,
                        details={"error_type": type(exc).__name__, "error": str(exc)},
                    )
                )
            output_row = {
                "question_id": question.question_id,
                "question": question.question,
                "user_input": question.question,
                "answer": answer,
                "response": answer,
                "contexts": answer_contexts,
                "retrieved_contexts": answer_contexts,
                "reference": question.reference,
                "ground_truth": question.reference,
                "question_type": question.task,
                "gold_doc_ids": list(question.gold_doc_ids),
                "answer_source": resolved_answer_mode,
                "answer_model": resolved_model,
                "diagnostics": diagnostics,
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
                    for chunk in chunks
                ],
            }
            rag_outputs.append(output_row)
            append_jsonl(partial_path, output_row)

    retrieval_metrics = aggregate_retrieval_metrics(per_question, k_values)
    retrieval_by_group = grouped_retrieval_metrics(questions, per_question, k_values)
    retrieval_error_cases = build_retrieval_error_cases(questions, retrieval_rows, per_question, top_k)
    validation_warnings = build_run_warnings(
        questions,
        ingestion_rows,
        retrieval_rows,
        per_question,
        k_values,
    ) + runtime_warnings
    validity_summary = build_validity_summary(
        rows=len(questions),
        warnings=validation_warnings,
        evaluation_type=evaluation_type,
    )

    metadata = config_snapshot(run_id, dataset_source, dataset_version, top_k, None)
    metadata.update(
        {
            "evaluation_type": evaluation_type,
            "dataset": dataset,
            "dataset_language": "zh",
            "dataset_version": dataset_version,
            "dataset_source": dataset_source,
            "run_label": effective_label,
            "limit": limit,
            "offset": offset,
            "distractor_docs": distractor_docs,
            "candidate_document_count": len(documents),
            "uses_full_corpus": distractor_docs == 0,
            "uses_project_ingestion": True,
            "uses_project_retriever": True,
            "retrieval_only": dataset == "t2_retrieval",
            "answer_mode": resolved_answer_mode,
            "answer_model": resolved_model,
            "answer_temperature": config.LLM_TEMPERATURE,
            "ragas_enabled": dataset == "crud_rag" and not skip_ragas,
            "ragas_judge_model": resolve_judge_model() if dataset == "crud_rag" and not skip_ragas else None,
            "eval_database_isolated": True,
            "eval_database": redact_database_url(eval_database_url),
            "reranker_enabled": config.RERANKER_ENABLED,
            "reranker_model": config.RERANKER_MODEL if config.RERANKER_ENABLED else None,
            "reranker_top_n": config.RERANKER_TOP_N,
            "reranker_final_top_k": config.RERANKER_FINAL_TOP_K,
            "reranker_score_threshold": config.RERANKER_SCORE_THRESHOLD,
            "retrieval_fusion_mode": config.RETRIEVAL_FUSION_MODE,
            "dense_model": config.DENSE_MODEL,
            "dense_top_k": config.DENSE_TOP_K,
            "sparse_top_k": config.SPARSE_TOP_K,
            "rrf_top_k": config.RRF_TOP_K,
            "rrf_k": config.RRF_K,
            "child_chunk_size": config.CHILD_CHUNK_SIZE,
            "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
            "retrieval_context_policy": config.RETRIEVAL_CONTEXT_POLICY,
            "context_top_k": context_top_k,
            "git_commit": current_git_commit(),
            "question_fingerprint": stable_fingerprint(question.question_id for question in questions),
            "candidate_document_fingerprint": stable_fingerprint(document.doc_id for document in documents),
            "ingestion_seconds": ingestion_seconds,
            "k_values": k_values,
            "task_counts": task_counts(questions),
            "validity_summary": validity_summary,
        }
    )

    write_jsonl(run_dir / "ingestion_results.jsonl", ingestion_rows)
    write_jsonl(run_dir / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(run_dir / "retrieval_per_question_metrics.jsonl", per_question)
    write_jsonl(run_dir / "retrieval_error_cases.jsonl", retrieval_error_cases)
    write_jsonl(run_dir / "rag_outputs.jsonl", rag_outputs)
    write_metrics_csv(run_dir / "retrieval_metrics_summary.csv", retrieval_metrics)
    (run_dir / "retrieval_metrics_by_group.json").write_text(
        json.dumps(retrieval_by_group, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_validation_outputs(run_dir, validation_warnings, validity_summary)

    ragas_metrics: dict[str, float] | None = None
    ragas_by_group: dict[str, dict[str, float]] = {}
    if dataset == "crud_rag" and not skip_ragas:
        ragas_results, ragas_metrics = run_ragas_metrics(rag_outputs)
        ragas_error_cases = build_ragas_error_cases(ragas_results)
        ragas_by_group = grouped_ragas_metrics(ragas_results)
        ragas_warnings = validate_ragas_rows(ragas_results, expected_rows=len(rag_outputs))
        validation_warnings.extend(ragas_warnings)
        validity_summary = build_validity_summary(
            rows=len(questions),
            warnings=validation_warnings,
            evaluation_type=evaluation_type,
        )
        metadata["validity_summary"] = validity_summary
        metadata["ragas_metric_coverage"] = {
            key: value for key, value in ragas_metrics.items() if key.endswith("_rows")
        }
        write_jsonl(run_dir / "ragas_results.jsonl", ragas_results)
        write_jsonl(run_dir / "ragas_error_cases.jsonl", ragas_error_cases)
        write_metrics_csv(run_dir / "ragas_metrics_summary.csv", ragas_metrics)
        (run_dir / "ragas_metrics_by_group.json").write_text(
            json.dumps(ragas_by_group, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_validation_outputs(run_dir, validation_warnings, validity_summary)

    metadata["total_runtime_seconds"] = time.perf_counter() - started_at

    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        run_dir / "report.md",
        metadata,
        retrieval_metrics,
        retrieval_by_group,
        ragas_metrics,
        ragas_by_group,
        validation_warnings,
    )
    latest_dir = output / f"{dataset}_latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "retrieval_metrics": retrieval_metrics,
        "ragas_metrics": ragas_metrics,
        "validity_summary": validity_summary,
    }


def materialize_documents(
    documents: Sequence[BenchmarkDocument], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    source_name_to_id = {}
    for index, document in enumerate(documents):
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", document.doc_id).strip("._-")[:80] or "doc"
        digest = hashlib.sha1(document.doc_id.encode("utf-8")).hexdigest()[:10]
        source_name = f"bench_{index:06d}_{safe}_{digest}"
        path = output_dir / f"{source_name}.md"
        path.write_text(document.text.strip() + "\n", encoding="utf-8")
        rows.append({"doc_id": document.doc_id, "path": path, "source_name": source_name})
        source_name_to_id[source_name] = document.doc_id
    return rows, source_name_to_id


def generate_answer(
    question: BenchmarkQuestion,
    contexts: list[str],
    answer_mode: str,
    generator: Any,
) -> tuple[str, list[str], dict[str, Any]]:
    if answer_mode == "reference":
        return question.reference, contexts, {}
    result = generator.generate(question.question, contexts)
    if isinstance(result, str):
        return result, contexts, {}
    return result["answer"], list(result.get("contexts") or contexts), dict(result.get("diagnostics") or {})


def score_retrieval_question(
    question: BenchmarkQuestion,
    chunks: Sequence[dict[str, Any]],
    k_values: Sequence[int],
) -> dict[str, Any]:
    gold = set(question.gold_doc_ids)
    result: dict[str, Any] = {
        "question_id": question.question_id,
        "question": question.question,
        "question_type": question.task,
        "gold_doc_count": len(gold),
        "scored": bool(gold),
        "mrr": reciprocal_rank(chunks, gold),
        "warnings": [] if gold else ["missing_primary_gold"],
    }
    for k in k_values:
        top = list(chunks[:k])
        hit_docs = {chunk["source_doc_id"] for chunk in top if chunk.get("source_doc_id") in gold}
        result[f"actual_results@{k}"] = float(len(top))
        result[f"hitrate@{k}"] = 1.0 if hit_docs else 0.0
        result[f"source_hitrate@{k}"] = result[f"hitrate@{k}"]
        result[f"recall@{k}"] = ratio(len(hit_docs), len(gold))
        result[f"precision@{k}"] = len(hit_docs) / k if k else 0.0
        result[f"ndcg@{k}"] = ndcg(top, gold, k)
    return result


def aggregate_retrieval_metrics(
    rows: Sequence[dict[str, Any]], k_values: Sequence[int]
) -> dict[str, float]:
    metric_names = ["mrr", "retrieval_latency_ms", "context_count", "context_chars"]
    for k in k_values:
        metric_names.extend(
            [f"actual_results@{k}", f"hitrate@{k}", f"source_hitrate@{k}", f"recall@{k}", f"precision@{k}", f"ndcg@{k}"]
        )
    scored = [row for row in rows if row.get("scored")]
    metrics = {name: mean(row.get(name, 0.0) for row in scored) for name in metric_names}
    metrics.update(
        {
            "rows": float(len(rows)),
            "scored_rows": float(len(scored)),
            "unscored_rows": float(len(rows) - len(scored)),
            "warning_count": float(sum(len(row.get("warnings") or []) for row in rows)),
            "evaluation_valid": 1.0 if rows and scored else 0.0,
        }
    )
    return metrics


def grouped_retrieval_metrics(
    questions: Sequence[BenchmarkQuestion],
    rows: Sequence[dict[str, Any]],
    k_values: Sequence[int],
) -> dict[str, dict[str, float]]:
    task_by_id = {question.question_id: question.task for question in questions}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(task_by_id[row["question_id"]], []).append(row)
    return {task: aggregate_retrieval_metrics(group, k_values) for task, group in sorted(groups.items())}


def grouped_ragas_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("question_type") or "unknown"), []).append(row)
    return {task: summarize_ragas_rows(group) for task, group in sorted(groups.items())}


def build_retrieval_error_cases(
    questions: Sequence[BenchmarkQuestion],
    retrieval_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    retrieval_by_id = {row["question_id"]: row for row in retrieval_rows}
    score_by_id = {row["question_id"]: row for row in score_rows}
    cases = []
    for question in questions:
        score = score_by_id[question.question_id]
        if score.get(f"hitrate@{top_k}", 0.0) > 0:
            continue
        chunks = retrieval_by_id[question.question_id].get("retrieved_chunks") or []
        cases.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "question_type": question.task,
                "expected_source_doc_ids": list(question.gold_doc_ids),
                "retrieved_source_doc_ids": [chunk.get("source_doc_id") for chunk in chunks[:top_k]],
                "failure_type": "missed_source",
            }
        )
    return cases


def build_run_warnings(
    questions: Sequence[BenchmarkQuestion],
    ingestion_rows: Sequence[dict[str, Any]],
    retrieval_rows: Sequence[dict[str, Any]],
    score_rows: Sequence[dict[str, Any]],
    k_values: Sequence[int],
) -> list[dict[str, Any]]:
    warnings = []
    if not questions:
        warnings.append(make_warning("empty_dataset", "No benchmark questions were selected.", severity="error"))
    failed = [row for row in ingestion_rows if row.get("status") == "failed"]
    if failed:
        warnings.append(
            make_warning(
                "ingestion_failed",
                "One or more benchmark documents failed project ingestion.",
                severity="error",
                details={"failed_count": len(failed)},
            )
        )
    for row in score_rows:
        if not row.get("scored"):
            warnings.append(
                make_warning(
                    "missing_primary_gold",
                    "Question has no gold document IDs.",
                    question_id=row.get("question_id"),
                )
            )
    for row in retrieval_rows:
        actual = len(row.get("retrieved_chunks") or [])
        for k in k_values:
            if actual < k:
                warnings.append(
                    make_warning(
                        "insufficient_results_for_k",
                        "Retrieved result depth is smaller than a declared metric cutoff.",
                        question_id=row.get("question_id"),
                        details={"k": k, "actual_results": actual},
                    )
                )
    return warnings


def task_counts(questions: Sequence[BenchmarkQuestion]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.task] = counts.get(question.task, 0) + 1
    return counts


def reciprocal_rank(chunks: Sequence[dict[str, Any]], gold: set[str]) -> float:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.get("source_doc_id") in gold:
            return 1.0 / rank
    return 0.0


def ndcg(chunks: Sequence[dict[str, Any]], gold: set[str], k: int) -> float:
    seen = set()
    gains = []
    for chunk in chunks[:k]:
        doc_id = chunk.get("source_doc_id")
        relevant = doc_id in gold and doc_id not in seen
        gains.append(1 if relevant else 0)
        if relevant:
            seen.add(doc_id)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    idcg = sum(1 / math.log2(index + 2) for index in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def redact_database_url(url: str) -> str:
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", url)


def stable_fingerprint(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_report(
    path: Path,
    metadata: dict[str, Any],
    retrieval_metrics: dict[str, float],
    retrieval_by_group: dict[str, dict[str, float]],
    ragas_metrics: dict[str, float] | None,
    ragas_by_group: dict[str, dict[str, float]],
    warnings: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# Chinese RAG Benchmark Report",
        "",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Evaluation type: `{metadata['evaluation_type']}`",
        f"- Rows: `{metadata['limit']}` (offset `{metadata['offset']}`)",
        f"- Candidate documents: `{metadata['candidate_document_count']}`",
        f"- Full corpus: `{metadata['uses_full_corpus']}`",
        f"- Answer mode: `{metadata['answer_mode']}`",
        f"- Retrieval only: `{metadata['retrieval_only']}`",
        f"- Run label: `{metadata['run_label']}`",
        f"- Retrieval mode: `{metadata['retrieval_fusion_mode']}`",
        f"- Reranker enabled: `{metadata['reranker_enabled']}`",
        f"- Context policy/top-k: `{metadata['retrieval_context_policy']}` / `{metadata['context_top_k']}`",
        "",
        "## Retrieval Metrics",
        "",
    ]
    lines.extend(f"- `{key}`: {value:.6f}" for key, value in sorted(retrieval_metrics.items()))
    lines.extend(["", "## Retrieval Metrics By Task", "", "```json", json.dumps(retrieval_by_group, ensure_ascii=False, indent=2), "```"])
    if ragas_metrics is not None:
        lines.extend(["", "## RAGAS Metrics", ""])
        lines.extend(f"- `{key}`: {value:.6f}" for key, value in sorted(ragas_metrics.items()))
        lines.extend(["", "## RAGAS Metrics By Task", "", "```json", json.dumps(ragas_by_group, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## Warnings", "", f"Total warnings: `{len(warnings)}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chinese T2Retrieval or CRUD-RAG through project ingestion and retrieval.")
    parser.add_argument("--dataset", choices=("t2_retrieval", "crud_rag"), required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--distractor-docs", type=int, default=120, help="Use 0 for the complete corpus.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", default=str(Path(config.EVALUATION_REPORTS_DIR) / "chinese_benchmarks"))
    parser.add_argument("--crud-root")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--answer-mode", choices=("agent", "direct", "reference"), default=None)
    parser.add_argument("--run-label")
    parser.add_argument("--retrieval-mode", choices=("dense", "rrf", "sparse"))
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker-final-top-k", type=int)
    parser.add_argument("--reranker-score-threshold", type=float)
    parser.add_argument("--context-policy", choices=("child", "neighbor", "parent", "adaptive"))
    parser.add_argument("--context-top-k", type=int, default=3)
    args = parser.parse_args()
    result = run_chinese_benchmark(
        dataset=args.dataset,
        limit=args.limit,
        offset=args.offset,
        distractor_docs=args.distractor_docs,
        top_k=args.top_k,
        output_dir=args.output_dir,
        crud_root=args.crud_root,
        skip_ragas=args.skip_ragas,
        answer_mode=args.answer_mode,
        run_label=args.run_label,
        retrieval_mode=args.retrieval_mode,
        reranker_enabled=args.reranker,
        reranker_final_top_k=args.reranker_final_top_k,
        reranker_score_threshold=args.reranker_score_threshold,
        context_policy=args.context_policy,
        context_top_k=args.context_top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
