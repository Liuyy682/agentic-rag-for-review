from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.runners.chinese_benchmark_compare import percentile, read_json, read_jsonl


def select_final_candidate(
    t2_comparisons: dict[str, str | Path],
    crud_comparisons: dict[str, str | Path],
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    labels = sorted(set(t2_comparisons) & set(crud_comparisons))
    if not labels:
        raise ValueError("At least one candidate label must have both T2 and CRUD comparisons")
    candidates = []
    baseline_runs: dict[str, str] = {}
    for index, label in enumerate(labels):
        t2 = read_json(Path(t2_comparisons[label]))
        crud = read_json(Path(crud_comparisons[label]))
        if t2.get("dataset") != "t2_retrieval" or crud.get("dataset") != "crud_rag":
            raise ValueError(f"Candidate {label} must pair a T2 comparison with a CRUD comparison")
        if t2.get("candidate_label") != label or crud.get("candidate_label") != label:
            raise ValueError(f"Comparison candidate labels do not match {label}")
        for dataset, summary in (("t2", t2), ("crud", crud)):
            baseline_run = str(Path(summary["baseline_run"]).resolve())
            if dataset in baseline_runs and baseline_runs[dataset] != baseline_run:
                raise ValueError(f"All {dataset} candidates must use the same baseline run")
            baseline_runs[dataset] = baseline_run
        t2_metric = t2["retrieval"][t2["primary_metric"]]
        crud_metric = crud["retrieval"][crud["primary_metric"]]
        combined_baseline = (t2_metric["baseline"] + crud_metric["baseline"]) / 2.0
        combined_candidate = (t2_metric["candidate"] + crud_metric["candidate"]) / 2.0
        lower, upper = combined_bootstrap_ci(t2, crud, bootstrap_samples, seed + index)
        ineligibility_reasons = crud_group_regressions(crud)
        if t2_metric["absolute_delta"] < -0.01:
            ineligibility_reasons.append("T2 primary metric regressed by more than 0.01")
        if crud_metric["absolute_delta"] < -0.01:
            ineligibility_reasons.append("CRUD primary metric regressed by more than 0.01")
        eligible = not ineligibility_reasons
        candidates.append(
            {
                "label": label,
                "eligible": eligible,
                "ineligibility_reasons": ineligibility_reasons,
                "t2_ndcg": t2_metric["candidate"],
                "crud_recall": crud_metric["candidate"],
                "combined_baseline": combined_baseline,
                "combined_candidate": combined_candidate,
                "combined_delta": combined_candidate - combined_baseline,
                "combined_ci95_lower": lower,
                "combined_ci95_upper": upper,
                "context_chars": average_candidate_metric(t2, crud, "context_chars"),
                "retrieval_latency_ms": average_candidate_metric(t2, crud, "retrieval_latency_ms"),
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    winner = choose_winner(eligible)
    material_improvement = bool(
        winner and winner["combined_delta"] >= 0.02 and winner["combined_ci95_lower"] > 0.0
    )
    result = {
        "selection_rule": "highest combined primary score; within 0.01 prefer fewer context chars then lower latency",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "candidates": candidates,
        "selected_candidate": winner["label"] if winner else None,
        "retrieval_material_improvement": material_improvement,
        "verdict": (
            "检索得到有效提升"
            if material_improvement
            else "没有候选通过约束" if not winner else "未验证出显著检索提升"
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output / "final_selection.md", result)
    return result


def combined_bootstrap_ci(
    t2_summary: dict[str, Any],
    crud_summary: dict[str, Any],
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    t2_deltas = primary_deltas(t2_summary)
    crud_deltas = primary_deltas(crud_summary)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        t2_mean = sum(t2_deltas[rng.randrange(len(t2_deltas))] for _ in t2_deltas) / len(t2_deltas)
        crud_mean = sum(crud_deltas[rng.randrange(len(crud_deltas))] for _ in crud_deltas) / len(crud_deltas)
        values.append((t2_mean + crud_mean) / 2.0)
    values.sort()
    return percentile(values, 0.025), percentile(values, 0.975)


def primary_deltas(summary: dict[str, Any]) -> list[float]:
    metric = summary["primary_metric"]
    baseline = read_jsonl(Path(summary["baseline_run"]) / "retrieval_per_question_metrics.jsonl")
    candidate = read_jsonl(Path(summary["candidate_run"]) / "retrieval_per_question_metrics.jsonl")
    if [row["question_id"] for row in baseline] != [row["question_id"] for row in candidate]:
        raise ValueError("Comparison source runs no longer contain matching question IDs")
    return [float(right[metric]) - float(left[metric]) for left, right in zip(baseline, candidate)]


def crud_group_regressions(summary: dict[str, Any]) -> list[str]:
    reasons = []
    metric = summary["primary_metric"]
    for group in ("questanswer_2docs", "questanswer_3docs"):
        row = (summary.get("retrieval_by_group", {}).get(group) or {}).get(metric)
        if not row:
            reasons.append(f"{group} {metric} is missing")
        elif row["absolute_delta"] < -0.02:
            reasons.append(f"{group} {metric} regressed by more than 0.02")
    return reasons


def average_candidate_metric(t2: dict[str, Any], crud: dict[str, Any], metric: str) -> float:
    values = []
    for summary in (t2, crud):
        row = summary.get("retrieval", {}).get(metric)
        if row:
            values.append(float(row["candidate"]))
    if len(values) != 2:
        raise ValueError(f"Both T2 and CRUD comparisons must contain {metric}")
    return sum(values) / len(values)


def choose_winner(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    best_score = max(row["combined_candidate"] for row in candidates)
    contenders = [row for row in candidates if best_score - row["combined_candidate"] < 0.01]
    return min(contenders, key=lambda row: (row["context_chars"], row["retrieval_latency_ms"], row["label"]))


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Chinese RAG Final Candidate Selection",
        "",
        f"- Selected candidate: `{result['selected_candidate']}`",
        f"- Verdict: **{result['verdict']}**",
        "- Scope: sampled candidate pools; not an official full-corpus benchmark score.",
        "",
        "| Candidate | Eligible | Combined | Delta | 95% CI | Context chars | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            f"| {row['label']} | {row['eligible']} | {row['combined_candidate']:.6f} | "
            f"{row['combined_delta']:+.6f} | [{row['combined_ci95_lower']:+.6f}, "
            f"{row['combined_ci95_upper']:+.6f}] | {row['context_chars']:.1f} | "
            f"{row['retrieval_latency_ms']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_comparisons(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path or label in result:
            raise ValueError("Comparison arguments must be unique LABEL=/path/to/experiment_summary.json values")
        result[label] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the final Chinese RAG candidate across T2 and CRUD.")
    parser.add_argument("--t2-comparison", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--crud-comparison", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = select_final_candidate(
        parse_comparisons(args.t2_comparison),
        parse_comparisons(args.crud_comparison),
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
