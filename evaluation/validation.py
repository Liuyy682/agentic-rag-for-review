import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from evaluation.data import EvalQuestion
from evaluation.io import write_jsonl


RAGAS_METRIC_GROUPS = {
    "faithfulness": ["faithfulness"],
    "context_precision": ["context_precision", "llm_context_precision_with_reference"],
    "context_recall": ["context_recall"],
}


def make_warning(
    code: str,
    message: str,
    *,
    severity: str = "warning",
    question_id: str | None = None,
    details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    warning: Dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if question_id:
        warning["question_id"] = question_id
    if details:
        warning["details"] = details
    return warning


def build_validity_summary(
    rows: int,
    warnings: Sequence[Dict[str, Any]],
    evaluation_type: str,
) -> Dict[str, Any]:
    warning_count = len(warnings)
    error_count = sum(1 for warning in warnings if warning.get("severity") == "error")
    codes = Counter(str(warning.get("code", "unknown")) for warning in warnings)
    return {
        "evaluation_type": evaluation_type,
        "rows": rows,
        "warning_count": warning_count,
        "error_count": error_count,
        "evaluation_valid": error_count == 0,
        "warning_codes": dict(sorted(codes.items())),
    }


def validate_retrieval_inputs(
    questions: Sequence[EvalQuestion],
    result_rows: Sequence[Dict[str, Any]],
    k_values: Iterable[int],
    evaluation_type: str,
    configured_top_k: int | None = None,
    score_threshold: float | None = None,
    retrieval_mode: str | None = None,
) -> list[Dict[str, Any]]:
    warnings: list[Dict[str, Any]] = []
    if not questions:
        warnings.append(
            make_warning(
                "empty_dataset",
                "Evaluation dataset has no questions; metric values are not usable for conclusions.",
                severity="error",
                details={"configured_top_k": configured_top_k},
            )
        )
        return warnings

    question_ids = [question.question_id for question in questions]
    duplicate_question_ids = sorted([item for item, count in Counter(question_ids).items() if count > 1])
    if duplicate_question_ids:
        warnings.append(
            make_warning(
                "duplicate_question_id",
                "Evaluation dataset contains duplicate question_id values.",
                severity="error",
                details={"question_ids": duplicate_question_ids},
            )
        )

    results_by_id = {row.get("question_id"): row for row in result_rows}
    result_ids = [row.get("question_id") for row in result_rows]
    duplicate_result_ids = sorted([str(item) for item, count in Counter(result_ids).items() if item and count > 1])
    if duplicate_result_ids:
        warnings.append(
            make_warning(
                "duplicate_result_question_id",
                "Retrieval results contain duplicate question_id values.",
                severity="error",
                details={"question_ids": duplicate_result_ids},
            )
        )

    unknown_result_ids = sorted(str(item) for item in result_ids if item and item not in set(question_ids))
    if unknown_result_ids:
        warnings.append(
            make_warning(
                "result_without_question",
                "Retrieval results contain question_id values not present in the evaluation dataset.",
                details={"question_ids": unknown_result_ids},
            )
        )

    if score_threshold is not None and retrieval_mode in {"rrf", "dense", "sparse"}:
        warnings.append(
            make_warning(
                "score_threshold_ignored",
                "score_threshold is recorded but not applied by this retrieval mode.",
                details={"retrieval_mode": retrieval_mode, "score_threshold": score_threshold},
            )
        )

    sorted_k_values = sorted({int(k) for k in k_values if int(k) > 0})
    scored_rows = 0
    observed_primary_overlap = False
    saw_retrieved_chunks = False
    for question in questions:
        row = results_by_id.get(question.question_id)
        if row is None:
            warnings.append(
                make_warning(
                    "missing_result_row",
                    "No retrieval result row was produced for this question.",
                    question_id=question.question_id,
                )
            )
            retrieved_chunks: Sequence[Dict[str, Any]] = []
        else:
            retrieved_chunks = row.get("retrieved_chunks") or []
        saw_retrieved_chunks = saw_retrieved_chunks or bool(retrieved_chunks)

        if not question.gold_child_ids and not question.gold_parent_ids:
            warnings.append(
                make_warning(
                    "missing_primary_gold",
                    "Question has no child or parent gold ids; primary retrieval metrics exclude this row.",
                    question_id=question.question_id,
                )
            )
        else:
            scored_rows += 1
            if question.gold_child_ids:
                gold = {str(item) for item in question.gold_child_ids}
                observed = {str(item.get("chunk_id")) for item in retrieved_chunks}
            else:
                gold = {str(item) for item in question.gold_parent_ids}
                observed = {str(item.get("parent_id")) for item in retrieved_chunks}
            observed_primary_overlap = observed_primary_overlap or bool(gold & observed)

        for k in sorted_k_values:
            if len(retrieved_chunks) < k:
                warnings.append(
                    make_warning(
                        "insufficient_results_for_k",
                        "Retrieved result depth is smaller than a declared metric cutoff.",
                        question_id=question.question_id,
                        details={"k": k, "actual_results": len(retrieved_chunks)},
                    )
                )
    if scored_rows == 0:
        warnings.append(
            make_warning(
                "no_scored_rows",
                "No questions have child or parent gold ids; primary retrieval metrics are invalid.",
                severity="error",
            )
        )
    elif saw_retrieved_chunks and not observed_primary_overlap:
        warnings.append(
            make_warning(
                "gold_result_id_no_overlap",
                "No retrieved child/parent ids overlap any primary gold ids; check gold id alignment or retrieval quality.",
                details={"scored_rows": scored_rows},
            )
        )
    return warnings


def validate_ragas_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    expected_rows: int | None = None,
    metric_groups: Dict[str, Sequence[str]] = RAGAS_METRIC_GROUPS,
) -> list[Dict[str, Any]]:
    warnings: list[Dict[str, Any]] = []
    if expected_rows is not None and len(rows) != expected_rows:
        warnings.append(
            make_warning(
                "ragas_row_count_mismatch",
                "RAGAS result row count does not match the expected output row count.",
                severity="error",
                details={"expected_rows": expected_rows, "actual_rows": len(rows)},
            )
        )
    if not rows:
        warnings.append(
            make_warning(
                "empty_ragas_results",
                "RAGAS returned no rows; generated metric values are not usable.",
                severity="error",
            )
        )
        return warnings

    for row in rows:
        present_any_metric = False
        for source_keys in metric_groups.values():
            if any(_to_float(row.get(key)) is not None for key in source_keys):
                present_any_metric = True
                break
        if not present_any_metric:
            warnings.append(
                make_warning(
                    "ragas_metric_missing",
                    "RAGAS row has no usable metric values.",
                    question_id=str(row.get("question_id") or ""),
                )
            )
            continue

        for metric_name, source_keys in metric_groups.items():
            if not any(_to_float(row.get(key)) is not None for key in source_keys):
                warnings.append(
                    make_warning(
                        "ragas_metric_missing",
                        "RAGAS row is missing a required usable metric value.",
                        question_id=str(row.get("question_id") or ""),
                        details={"metric": metric_name},
                    )
                )
                continue
            for key in source_keys:
                if key in row and row.get(key) is not None and _to_float(row.get(key)) is None:
                    warnings.append(
                        make_warning(
                            "ragas_metric_invalid",
                            "RAGAS metric value is present but not numeric.",
                            question_id=str(row.get("question_id") or ""),
                            details={"metric": key, "value": str(row.get(key))},
                        )
                    )
    return warnings


def write_validation_outputs(
    output_dir: str | Path,
    warnings: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "evaluation_warnings.jsonl", warnings)
    (output / "validity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validation_markdown_section(
    warnings: Sequence[Dict[str, Any]],
    summary: Dict[str, Any] | None = None,
    *,
    max_warnings: int = 20,
) -> list[str]:
    lines = ["", "## Evaluation Validity"]
    if summary:
        lines.extend(
            [
                f"- evaluation_type: `{summary.get('evaluation_type', '')}`",
                f"- evaluation_valid: `{str(summary.get('evaluation_valid', False)).lower()}`",
                f"- warnings: {summary.get('warning_count', 0)}",
            ]
        )
    if not warnings:
        lines.append("- No validation warnings.")
        return lines

    lines.append("")
    lines.append("| Severity | Code | Question | Message |")
    lines.append("|---|---|---|---|")
    for warning in list(warnings)[:max_warnings]:
        question = warning.get("question_id", "")
        message = str(warning.get("message", "")).replace("|", "\\|")
        lines.append(
            f"| {warning.get('severity', 'warning')} | `{warning.get('code', '')}` | `{question}` | {message} |"
        )
    return lines


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None
