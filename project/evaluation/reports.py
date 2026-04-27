from pathlib import Path
from typing import Any, Dict, Iterable, List

from evaluation.data import dataset_stats, EvalQuestion


def write_retrieval_report(
    path: str | Path,
    run_metadata: Dict[str, Any],
    questions: Iterable[EvalQuestion],
    metrics: Dict[str, float],
    error_cases: List[Dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = dataset_stats(questions)
    lines = [
        "# Retrieval Evaluation Report",
        "",
        "## Run Metadata",
        f"- run_id: `{run_metadata.get('run_id')}`",
        f"- git_commit: `{run_metadata.get('git_commit')}`",
        f"- dataset_version: `{run_metadata.get('dataset_version')}`",
        f"- dataset_path: `{run_metadata.get('dataset_path')}`",
        f"- retrieval_mode: `{run_metadata.get('retrieval_mode')}`",
        f"- top_k: `{run_metadata.get('top_k')}`",
        "",
        "## Dataset",
        f"- total: {stats['total']}",
        f"- parent id coverage: {stats['parent_coverage']:.2%}",
        f"- child id coverage: {stats['child_coverage']:.2%}",
        f"- evidence coverage: {stats['evidence_coverage']:.2%}",
        "",
        "## Metrics",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(metrics.items()))
    lines.extend(["", "## Top Failure Cases"])
    if error_cases:
        for case in error_cases[:20]:
            lines.append(
                f"- `{case['question_id']}` {case['failure_type']}: "
                f"{case['question'][:140]}"
            )
    else:
        lines.append("- No retrieval failure cases detected by the current rules.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_ragas_report(
    path: str | Path,
    run_metadata: Dict[str, Any],
    metrics: Dict[str, float],
    error_cases: List[Dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAGAS Evaluation Report",
        "",
        "## Run Metadata",
        f"- run_id: `{run_metadata.get('run_id')}`",
        f"- git_commit: `{run_metadata.get('git_commit')}`",
        f"- dataset_version: `{run_metadata.get('dataset_version')}`",
        f"- llm_model: `{run_metadata.get('llm_model')}`",
        "",
        "## Metrics",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(metrics.items()))
    lines.extend(["", "## Top Failure Cases"])
    if error_cases:
        for case in error_cases[:20]:
            lines.append(f"- `{case['question_id']}` {case['failure_type']}: {case['question'][:140]}")
    else:
        lines.append("- No RAGAS failure cases detected by the current rules.")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_compare_report(
    path: str | Path,
    baseline: Dict[str, float],
    current: Dict[str, float],
    baseline_label: str = "Baseline",
    current_label: str = "Current",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set(baseline) | set(current))
    lines = [
        "# RAG Evaluation Compare Report",
        "",
        f"| Metric | {baseline_label} | {current_label} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in keys:
        base = baseline.get(key, 0.0)
        curr = current.get(key, 0.0)
        lines.append(f"| {key} | {base:.4f} | {curr:.4f} | {curr - base:+.4f} |")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")

