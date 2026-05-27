from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from evaluation.io import read_jsonl, write_jsonl, write_metrics_csv
from evaluation.llm_config import api_key, base_url, judge_model
from evaluation.validation import build_validity_summary, make_warning, write_validation_outputs


METRIC_KEYS = [
    "judge_faithfulness",
    "judge_answer_correctness",
    "judge_context_recall",
    "judge_context_precision",
]


def run_llm_judge_outputs(
    rag_outputs_path: str,
    output_dir: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    model: str | None = None,
    timeout: int = 180,
    max_retries: int = 2,
    max_workers: int = 4,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    source_path = Path(rag_outputs_path)
    rows = read_jsonl(source_path)
    selected_rows = rows[offset : offset + limit if limit is not None else None]
    output = Path(output_dir) if output_dir else source_path.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "llm_judge_results.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")

    resolved_model = model or judge_model()
    print(f"Judging {len(selected_rows)} rows with {resolved_model}...", flush=True)
    results: List[Dict[str, Any] | None] = [None] * len(selected_rows)
    warnings: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                judge_output,
                row,
                model=resolved_model,
                timeout=timeout,
                max_retries=max_retries,
                max_tokens=max_tokens,
            ): index
            for index, row in enumerate(selected_rows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            source_row = selected_rows[index]
            try:
                score = future.result()
            except Exception as exc:
                score = {
                    "judge_error": str(exc),
                    "judge_error_type": type(exc).__name__,
                    **{key: math.nan for key in METRIC_KEYS},
                }
                warnings.append(
                    make_warning(
                        "llm_judge_failed",
                        "LLM judge failed for a row; metric values are missing.",
                        question_id=str(source_row.get("question_id") or ""),
                        details={"error_type": type(exc).__name__, "error": str(exc)[:500]},
                    )
                )
            row = dict(source_row)
            row.update(score)
            results[index] = row
            _append_jsonl(partial_path, row)
            print(f"[{completed}/{len(selected_rows)}] judged {source_row.get('question_id')}", flush=True)

    ordered_results = [row for row in results if row is not None]
    metrics = summarize_judge_rows(ordered_results)
    warnings.extend(validate_judge_rows(ordered_results, expected_rows=len(selected_rows)))
    validity_summary = build_validity_summary(
        rows=len(ordered_results),
        warnings=warnings,
        evaluation_type="llm_judge_generation_eval",
    )
    error_cases = build_judge_error_cases(ordered_results)

    write_jsonl(output / "llm_judge_results.jsonl", ordered_results)
    write_jsonl(output / "llm_judge_error_cases.jsonl", error_cases)
    write_metrics_csv(output / "llm_judge_metrics_summary.csv", metrics)
    write_validation_outputs(output, warnings, validity_summary)
    write_report(output / "llm_judge_report.md", metrics, error_cases, warnings, validity_summary)
    _update_metadata(output / "run_metadata.json", resolved_model, metrics, validity_summary)
    return {
        "rows": len(ordered_results),
        "output_dir": str(output),
        "metrics": metrics,
        "validity_summary": validity_summary,
    }


def judge_output(
    output: Dict[str, Any],
    *,
    model: str,
    timeout: int,
    max_retries: int,
    max_tokens: int,
) -> Dict[str, Any]:
    from openai import OpenAI

    contexts = "\n\n".join(
        f"[Context {index + 1}]\n{text}"
        for index, text in enumerate(output.get("retrieved_contexts") or output.get("contexts") or [])
    )
    prompt = (
        "Score this retrieval-augmented generation answer. Return JSON only with keys "
        "faithfulness, answer_correctness, context_recall, context_precision, explanation. "
        "Each numeric score must be between 0 and 1.\n\n"
        f"Question:\n{output.get('question') or output.get('user_input') or ''}\n\n"
        f"Reference answer:\n{output.get('reference') or output.get('ground_truth') or ''}\n\n"
        f"Retrieved contexts:\n{contexts}\n\n"
        f"Generated answer:\n{output.get('answer') or output.get('response') or ''}\n"
    )
    client = OpenAI(
        api_key=api_key(),
        base_url=base_url(),
        timeout=timeout,
        max_retries=max_retries,
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are a strict evaluator for retrieval-augmented generation."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = parse_json_object(raw)
    return {
        "judge_faithfulness": clamp_float(parsed.get("faithfulness")),
        "judge_answer_correctness": clamp_float(parsed.get("answer_correctness")),
        "judge_context_recall": clamp_float(parsed.get("context_recall")),
        "judge_context_precision": clamp_float(parsed.get("context_precision")),
        "judge_explanation": str(parsed.get("explanation") or raw)[:1000],
        "judge_model": model,
    }


def parse_json_object(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(number):
        return math.nan
    return max(0.0, min(1.0, number))


def summarize_judge_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    summary: Dict[str, float] = {"rows": float(len(rows))}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if _is_number(row.get(key))]
        output_key = key.removeprefix("judge_")
        if values:
            summary[output_key] = sum(values) / len(values)
        summary[f"{output_key}_rows"] = float(len(values))
        summary[f"{output_key}_missing_rows"] = float(len(rows) - len(values))
    return summary


def validate_judge_rows(rows: List[Dict[str, Any]], expected_rows: int) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []
    if len(rows) != expected_rows:
        warnings.append(
            make_warning(
                "llm_judge_row_count_mismatch",
                "LLM judge result row count does not match the expected output row count.",
                severity="error",
                details={"expected_rows": expected_rows, "actual_rows": len(rows)},
            )
        )
    for row in rows:
        missing = [key for key in METRIC_KEYS if not _is_number(row.get(key))]
        if missing:
            warnings.append(
                make_warning(
                    "llm_judge_metric_missing",
                    "LLM judge row is missing one or more usable metric values.",
                    question_id=str(row.get("question_id") or ""),
                    details={"metrics": missing},
                )
            )
    return warnings


def build_judge_error_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    thresholds = {
        "judge_faithfulness": 0.75,
        "judge_answer_correctness": 0.75,
        "judge_context_recall": 0.70,
        "judge_context_precision": 0.65,
    }
    cases = []
    for row in rows:
        low = [key for key, threshold in thresholds.items() if _is_number(row.get(key)) and float(row[key]) < threshold]
        missing = [key for key in thresholds if not _is_number(row.get(key))]
        if not low and not missing:
            continue
        cases.append(
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "failure_type": "judge_metric_missing" if missing else "low_generation_quality",
                "low_metrics": low,
                "missing_metrics": missing,
                "answer": row.get("answer"),
                "judge_explanation": row.get("judge_explanation", ""),
            }
        )
    return cases


def write_report(
    path: Path,
    metrics: Dict[str, float],
    error_cases: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    validity_summary: Dict[str, Any],
) -> None:
    lines = [
        "# LLM Judge Generation Report",
        "",
        "## Metrics",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(metrics.items()))
    lines.extend(["", "## Top Failure Cases"])
    if error_cases:
        for case in error_cases[:20]:
            lines.append(f"- `{case['question_id']}` {case['failure_type']}: {str(case.get('question') or '')[:140]}")
    else:
        lines.append("- No failure cases detected by the current rules.")
    lines.extend(
        [
            "",
            "## Evaluation Validity",
            f"- evaluation_type: `{validity_summary.get('evaluation_type')}`",
            f"- evaluation_valid: `{str(validity_summary.get('evaluation_valid', False)).lower()}`",
            f"- warnings: {validity_summary.get('warning_count', 0)}",
        ]
    )
    if warnings:
        lines.extend(["", "| Severity | Code | Question | Message |", "|---|---|---|---|"])
        for warning in warnings[:20]:
            lines.append(
                f"| {warning.get('severity', 'warning')} | `{warning.get('code', '')}` | "
                f"`{warning.get('question_id', '')}` | {str(warning.get('message', '')).replace('|', '\\|')} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_metadata(
    metadata_path: Path,
    model: str,
    metrics: Dict[str, float],
    validity_summary: Dict[str, Any],
) -> None:
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "llm_judge_model": model,
            "llm_judge_metrics": metrics,
            "llm_judge_validity_summary": validity_summary,
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated RAG outputs with a direct LLM judge.")
    parser.add_argument("--rag-outputs", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    result = run_llm_judge_outputs(
        rag_outputs_path=args.rag_outputs,
        output_dir=args.output_dir,
        limit=args.limit,
        offset=args.offset,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
