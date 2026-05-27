from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from evaluation.io import config_snapshot, make_run_id, read_jsonl, write_jsonl, write_metrics_csv
from evaluation.llm_config import answer_model as resolve_answer_model
from evaluation.llm_config import api_key, base_url
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.reports import write_ragas_report
from evaluation.validation import build_validity_summary, make_warning, validate_ragas_rows, write_validation_outputs


def run_generation_from_rag_outputs(
    rag_outputs_path: str,
    output_dir: str,
    run_label: str,
    limit: int | None = None,
    offset: int = 0,
    answer_model: str | None = None,
    skip_ragas: bool = False,
    ragas_timeout: int = 180,
    ragas_max_retries: int = 2,
    ragas_max_workers: int = 2,
    ragas_batch_size: int | None = 1,
    answer_max_retries: int = 2,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    source_path = Path(rag_outputs_path)
    source_rows = read_jsonl(source_path)
    selected_rows = source_rows[offset : offset + limit if limit is not None else None]

    run_id = make_run_id(run_label)
    output = Path(output_dir)
    run_dir = output / "eval_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "rag_outputs.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")

    source_metadata = _read_source_metadata(source_path)
    resolved_answer_model = resolve_answer_model(answer_model)
    generator = ExistingContextAnswerGenerator(resolved_answer_model)
    outputs: List[Dict[str, Any]] = []
    generation_warnings: List[Dict[str, Any]] = []

    for index, row in enumerate(selected_rows, start=1):
        question = str(row.get("question") or row.get("user_input") or "").strip()
        contexts = [str(item) for item in (row.get("retrieved_contexts") or row.get("contexts") or [])]
        question_id = str(row.get("question_id") or f"row_{offset + index}")
        print(f"[{index}/{len(selected_rows)}] Generating answer for {question_id}...", flush=True)
        diagnostics: Dict[str, Any] = {
            "source_answer_source": row.get("answer_source"),
            "source_answer_model": row.get("answer_model"),
            "source_context_count": len(contexts),
            "source_context_chars": sum(len(context) for context in contexts),
        }
        try:
            answer = ""
            attempts = max(1, answer_max_retries + 1)
            for attempt in range(1, attempts + 1):
                answer = generator.generate(question, contexts)
                diagnostics["answer_generation_attempts"] = attempt
                if answer.strip():
                    break
        except Exception as exc:
            if fail_fast:
                raise
            answer = ""
            diagnostics.update(
                {
                    "answer_generation_failed": True,
                    "answer_error_type": type(exc).__name__,
                    "answer_error": str(exc),
                }
            )
            generation_warnings.append(
                make_warning(
                    "answer_generation_failed",
                    "Answer generation failed for a row; RAGAS metrics for this row may be unusable.",
                    question_id=question_id,
                    details={"error_type": type(exc).__name__, "error": str(exc)[:500]},
                )
            )
            print(f"[{index}/{len(selected_rows)}] Answer generation failed for {question_id}: {exc}", flush=True)
        if not answer.strip() and not diagnostics.get("answer_generation_failed"):
            generation_warnings.append(
                make_warning(
                    "empty_generated_answer",
                    "Answer generation returned empty content after retries; RAGAS metrics for this row may be unusable.",
                    question_id=question_id,
                    details={"attempts": diagnostics.get("answer_generation_attempts", 0)},
                )
            )

        output_row = dict(row)
        output_row.update(
            {
                "question_id": question_id,
                "question": question,
                "user_input": question,
                "answer": answer,
                "response": answer,
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "reference": row.get("reference") or row.get("ground_truth") or "",
                "ground_truth": row.get("ground_truth") or row.get("reference") or "",
                "answer_source": "generated_from_existing_contexts",
                "answer_model": resolved_answer_model,
                "source_answer": row.get("answer"),
                "source_response": row.get("response"),
                "generation_diagnostics": diagnostics,
            }
        )
        outputs.append(output_row)
        _append_jsonl(partial_path, output_row)
        print(f"[{index}/{len(selected_rows)}] Done {question_id}", flush=True)

    context_counts = [len(row.get("retrieved_contexts") or []) for row in outputs]
    context_chars = [sum(len(context) for context in (row.get("retrieved_contexts") or [])) for row in outputs]
    metadata = config_snapshot(
        run_id=run_id,
        dataset_path=str(source_path),
        dataset_version="existing_retrieval_rag_outputs",
        top_k=int(source_metadata.get("top_k") or (max(context_counts) if context_counts else 0)),
        score_threshold=None,
    )
    metadata.update(
        {
            "evaluation_type": "generation_from_existing_retrieval_contexts",
            "source_rag_outputs": str(source_path),
            "source_run_metadata": str(_source_metadata_path(source_path)) if _source_metadata_path(source_path).exists() else "",
            "source_run_id": source_metadata.get("run_id"),
            "source_evaluation_type": source_metadata.get("evaluation_type"),
            "source_reranker_enabled": source_metadata.get("reranker_enabled"),
            "source_reranker": source_metadata.get("reranker"),
            "source_reranker_final_top_k": source_metadata.get("reranker_final_top_k"),
            "source_top_k": source_metadata.get("top_k"),
            "source_retrieval_context_policy": source_metadata.get("retrieval_context_policy"),
            "source_answer_source_counts": _count_values(row.get("answer_source") for row in selected_rows),
            "rows": len(outputs),
            "source_rows": len(source_rows),
            "offset": offset,
            "limit": limit,
            "answer_source": "generated_from_existing_contexts",
            "answer_model": resolved_answer_model,
            "context_count_min": min(context_counts) if context_counts else 0,
            "context_count_max": max(context_counts) if context_counts else 0,
            "context_count_avg": sum(context_counts) / len(context_counts) if context_counts else 0,
            "context_chars_min": min(context_chars) if context_chars else 0,
            "context_chars_max": max(context_chars) if context_chars else 0,
            "context_chars_avg": sum(context_chars) / len(context_chars) if context_chars else 0,
            "ragas_enabled": not skip_ragas,
        }
    )

    validation_warnings = list(generation_warnings)
    if not outputs:
        validation_warnings.append(
            make_warning(
                "empty_dataset",
                "No rows were available for generation-side evaluation.",
                severity="error",
            )
        )

    write_jsonl(run_dir / "rag_outputs.jsonl", outputs)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    result: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "rag_outputs": str(run_dir / "rag_outputs.jsonl"),
    }

    if skip_ragas or not outputs:
        validity_summary = build_validity_summary(
            rows=len(outputs),
            warnings=validation_warnings,
            evaluation_type=metadata["evaluation_type"],
        )
        metadata["validity_summary"] = validity_summary
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        write_validation_outputs(run_dir, validation_warnings, validity_summary)
        _refresh_latest(run_dir, output)
        result["validity_summary"] = validity_summary
        return result

    ragas_results, ragas_metrics = run_ragas_metrics(
        outputs,
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_workers=ragas_max_workers,
        batch_size=ragas_batch_size,
    )
    ragas_error_cases = build_ragas_error_cases(ragas_results)
    validation_warnings.extend(validate_ragas_rows(ragas_results, expected_rows=len(outputs)))
    validity_summary = build_validity_summary(
        rows=len(outputs),
        warnings=validation_warnings,
        evaluation_type=metadata["evaluation_type"],
    )
    metadata["validity_summary"] = validity_summary
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(run_dir / "ragas_error_cases.jsonl", ragas_error_cases)
    write_metrics_csv(run_dir / "ragas_metrics_summary.csv", ragas_metrics)
    write_validation_outputs(run_dir, validation_warnings, validity_summary)
    write_ragas_report(
        run_dir / "ragas_report.md",
        metadata,
        ragas_metrics,
        ragas_error_cases,
        validation_warnings=validation_warnings,
        validity_summary=validity_summary,
    )
    latest_dir = _refresh_latest(run_dir, output)
    result.update(
        {
            "latest_dir": str(latest_dir),
            "report_path": str(run_dir / "ragas_report.md"),
            "ragas_metrics": ragas_metrics,
            "validity_summary": validity_summary,
        }
    )
    return result


class ExistingContextAnswerGenerator:
    def __init__(self, model: str) -> None:
        self.model = model
        self._client = None

    def generate(self, question: str, contexts: List[str]) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=api_key(), base_url=base_url())
        context = "\n\n".join(f"[Context {index}]\n{text}" for index, text in enumerate(contexts, start=1))
        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved contexts:\n{context}\n\n"
            "Answer the question using only the retrieved contexts. Keep the answer concise, preferably 1-3 sentences. "
            "If the retrieved contexts do not contain enough evidence, say that the context is insufficient."
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=int(os.environ.get("ANSWER_MAX_TOKENS", "512")),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful RAG answer generator. Answer only from the provided contexts. "
                        "If the contexts do not contain enough evidence, say that the context is insufficient."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def _read_source_metadata(source_path: Path) -> Dict[str, Any]:
    metadata_path = _source_metadata_path(source_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _source_metadata_path(source_path: Path) -> Path:
    return source_path.resolve().parent / "run_metadata.json"


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_values(values: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _refresh_latest(run_dir: Path, output_dir: Path) -> Path:
    latest_dir = output_dir / "rag_outputs_generation_latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    return latest_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate answers from an existing rag_outputs.jsonl context set and score generation metrics."
    )
    parser.add_argument("--rag-outputs", required=True, help="Existing rag_outputs.jsonl with retrieved_contexts.")
    parser.add_argument("--output-dir", default=str(Path(config.EVALUATION_REPORTS_DIR) / "rag_outputs_generation"))
    parser.add_argument("--run-label", default="rag_outputs_generation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=2)
    parser.add_argument("--ragas-max-workers", type=int, default=2)
    parser.add_argument("--ragas-batch-size", type=int, default=1)
    parser.add_argument("--ragas-max-tokens", default=None)
    parser.add_argument("--enable-answer-relevancy", action="store_true")
    parser.add_argument("--answer-max-retries", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.ragas_max_tokens is not None:
        os.environ["RAGAS_MAX_TOKENS"] = args.ragas_max_tokens
    if args.enable_answer_relevancy:
        os.environ["RAGAS_ENABLE_ANSWER_RELEVANCY"] = "true"

    result = run_generation_from_rag_outputs(
        rag_outputs_path=args.rag_outputs,
        output_dir=args.output_dir,
        run_label=args.run_label,
        limit=args.limit,
        offset=args.offset,
        answer_model=args.answer_model,
        skip_ragas=args.skip_ragas,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        answer_max_retries=args.answer_max_retries,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
