from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


COMPATIBILITY_FIELDS = (
    "dataset",
    "dataset_version",
    "limit",
    "offset",
    "distractor_docs",
    "candidate_document_count",
    "uses_full_corpus",
    "question_fingerprint",
    "candidate_document_fingerprint",
    "dense_model",
    "child_chunk_size",
    "child_chunk_overlap",
    "answer_mode",
    "answer_model",
    "answer_temperature",
    "ragas_judge_model",
)
RAGAS_METRICS = ("faithfulness", "context_precision", "context_recall")


def compare_runs(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    baseline_metadata = read_json(baseline_path / "run_metadata.json")
    candidate_metadata = read_json(candidate_path / "run_metadata.json")
    validate_compatible_runs(baseline_metadata, candidate_metadata)

    baseline_rows = read_jsonl(baseline_path / "retrieval_per_question_metrics.jsonl")
    candidate_rows = read_jsonl(candidate_path / "retrieval_per_question_metrics.jsonl")
    paired = pair_rows(baseline_rows, candidate_rows)
    metric_names = retrieval_metric_names(paired)
    retrieval = compare_metrics(paired, metric_names, bootstrap_samples, seed)
    retrieval_by_group = compare_by_group(paired, metric_names, bootstrap_samples, seed)

    ragas: dict[str, dict[str, float | None]] = {}
    ragas_by_group: dict[str, dict[str, dict[str, float | None]]] = {}
    baseline_ragas_path = baseline_path / "ragas_results.jsonl"
    candidate_ragas_path = candidate_path / "ragas_results.jsonl"
    if baseline_ragas_path.exists() and candidate_ragas_path.exists():
        ragas_pairs = pair_rows(read_jsonl(baseline_ragas_path), read_jsonl(candidate_ragas_path))
        ragas = compare_metrics(ragas_pairs, RAGAS_METRICS, bootstrap_samples, seed)
        ragas_by_group = compare_by_group(ragas_pairs, RAGAS_METRICS, bootstrap_samples, seed)

    dataset = baseline_metadata["dataset"]
    top_k = int(baseline_metadata.get("top_k") or max(baseline_metadata.get("k_values") or [10]))
    primary_metric = f"ndcg@{top_k}" if dataset == "t2_retrieval" else f"recall@{top_k}"
    primary = retrieval.get(primary_metric) or {}
    retrieval_improved = bool(
        primary
        and float(primary.get("absolute_delta") or 0.0) >= 0.02
        and float(primary.get("ci95_lower") or 0.0) > 0.0
    )
    ragas_positive = sum(
        1 for metric in RAGAS_METRICS if float((ragas.get(metric) or {}).get("ci95_lower") or 0.0) > 0.0
    )
    ragas_non_regressing = all(
        float((ragas.get(metric) or {}).get("absolute_delta") or 0.0) >= -0.01 for metric in RAGAS_METRICS
    ) if ragas else False
    ragas_coverage = {
        metric: float((ragas.get(metric) or {}).get("rows") or 0.0) / len(paired)
        for metric in RAGAS_METRICS
    } if ragas and paired else {}
    ragas_coverage_valid = bool(ragas_coverage) and all(value >= 0.95 for value in ragas_coverage.values())

    summary = {
        "baseline_run": str(baseline_path.resolve()),
        "candidate_run": str(candidate_path.resolve()),
        "baseline_label": baseline_metadata.get("run_label"),
        "candidate_label": candidate_metadata.get("run_label"),
        "dataset": dataset,
        "sampled_candidate_pool": not bool(baseline_metadata.get("uses_full_corpus")),
        "paired_rows": len(paired),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "primary_metric": primary_metric,
        "retrieval_material_improvement": retrieval_improved,
        "ragas_material_improvement": bool(
            ragas and ragas_coverage_valid and ragas_positive >= 2 and ragas_non_regressing
        ),
        "ragas_coverage": ragas_coverage,
        "warning_count": {
            "baseline": count_jsonl(baseline_path / "evaluation_warnings.jsonl"),
            "candidate": count_jsonl(candidate_path / "evaluation_warnings.jsonl"),
        },
        "retrieval": retrieval,
        "retrieval_by_group": retrieval_by_group,
        "ragas": ragas,
        "ragas_by_group": ragas_by_group,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(output / "experiment_summary.csv", retrieval, ragas)
    write_jsonl(output / "paired_retrieval_deltas.jsonl", paired_delta_rows(paired, metric_names))
    write_report(output / "experiment_report.md", summary)
    return summary


def validate_compatible_runs(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    mismatches = {
        field: {"baseline": baseline.get(field), "candidate": candidate.get(field)}
        for field in COMPATIBILITY_FIELDS
        if baseline.get(field) != candidate.get(field)
    }
    if mismatches:
        raise ValueError(f"Runs are not comparable: {json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}")


def pair_rows(
    baseline_rows: Sequence[dict[str, Any]], candidate_rows: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_ids = [str(row.get("question_id") or "") for row in baseline_rows]
    candidate_ids = [str(row.get("question_id") or "") for row in candidate_rows]
    if not baseline_ids or baseline_ids != candidate_ids or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("Runs must contain the same unique question IDs in the same order")
    return list(zip(baseline_rows, candidate_rows))


def retrieval_metric_names(pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> list[str]:
    if not pairs:
        return []
    prefixes = (
        "mrr",
        "actual_results@",
        "hitrate@",
        "source_hitrate@",
        "recall@",
        "precision@",
        "ndcg@",
        "retrieval_latency_ms",
        "context_count",
        "context_chars",
    )
    common = set(pairs[0][0]) & set(pairs[0][1])
    return sorted(name for name in common if name == "mrr" or name.startswith(prefixes[1:]))


def compare_metrics(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    metric_names: Iterable[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for index, metric in enumerate(metric_names):
        values = [
            (float(baseline[metric]), float(candidate[metric]))
            for baseline, candidate in pairs
            if is_number(baseline.get(metric)) and is_number(candidate.get(metric))
        ]
        if not values:
            continue
        baseline_mean = mean(item[0] for item in values)
        candidate_mean = mean(item[1] for item in values)
        delta = candidate_mean - baseline_mean
        lower, upper = bootstrap_delta_ci(values, bootstrap_samples, seed + index)
        result[metric] = {
            "rows": float(len(values)),
            "baseline": baseline_mean,
            "candidate": candidate_mean,
            "absolute_delta": delta,
            "relative_delta": delta / baseline_mean if baseline_mean else None,
            "ci95_lower": lower,
            "ci95_upper": upper,
        }
    return result


def compare_by_group(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    metric_names: Iterable[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float | None]]]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in pairs:
        group = str(pair[0].get("question_type") or "unknown")
        if group != str(pair[1].get("question_type") or "unknown"):
            raise ValueError(f"Question type mismatch for {pair[0].get('question_id')}")
        groups.setdefault(group, []).append(pair)
    return {
        group: compare_metrics(group_pairs, metric_names, bootstrap_samples, seed)
        for group, group_pairs in sorted(groups.items())
    }


def bootstrap_delta_ci(
    values: Sequence[tuple[float, float]], samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    deltas = [candidate - baseline for baseline, candidate in values]
    rng = random.Random(seed)
    bootstrapped = []
    size = len(deltas)
    for _ in range(samples):
        bootstrapped.append(sum(deltas[rng.randrange(size)] for _ in range(size)) / size)
    bootstrapped.sort()
    return percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def paired_delta_rows(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]], metric_names: Iterable[str]
) -> list[dict[str, Any]]:
    names = list(metric_names)
    rows = []
    for baseline, candidate in pairs:
        deltas = {
            metric: float(candidate[metric]) - float(baseline[metric])
            for metric in names
            if is_number(baseline.get(metric)) and is_number(candidate.get(metric))
        }
        rows.append(
            {
                "question_id": baseline["question_id"],
                "question_type": baseline.get("question_type"),
                "deltas": deltas,
            }
        )
    return rows


def percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def write_summary_csv(
    path: Path,
    retrieval: dict[str, dict[str, float | None]],
    ragas: dict[str, dict[str, float | None]],
) -> None:
    fields = ["category", "metric", "rows", "baseline", "candidate", "absolute_delta", "relative_delta", "ci95_lower", "ci95_upper"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for category, metrics in (("retrieval", retrieval), ("ragas", ragas)):
            for metric, values in sorted(metrics.items()):
                writer.writerow({"category": category, "metric": metric, **values})


def write_report(path: Path, summary: dict[str, Any]) -> None:
    verdict = "检索得到有效提升" if summary["retrieval_material_improvement"] else "未验证出显著检索提升"
    lines = [
        "# Chinese RAG Baseline Comparison",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Baseline: `{summary['baseline_label']}`",
        f"- Candidate: `{summary['candidate_label']}`",
        f"- Paired rows: `{summary['paired_rows']}`",
        f"- Sampled candidate pool: `{summary['sampled_candidate_pool']}`",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Verdict: **{verdict}**",
        "",
        "## Retrieval deltas",
        "",
        "| Metric | Baseline | Candidate | Delta | Relative | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, row in sorted(summary["retrieval"].items()):
        relative = row["relative_delta"]
        relative_text = "n/a" if relative is None else f"{relative:.2%}"
        lines.append(
            f"| {metric} | {row['baseline']:.6f} | {row['candidate']:.6f} | "
            f"{row['absolute_delta']:+.6f} | {relative_text} | "
            f"[{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}] |"
        )
    if len(summary["retrieval_by_group"]) > 1:
        lines.extend(["", "## Retrieval primary metric by group", ""])
        for group, metrics in summary["retrieval_by_group"].items():
            row = metrics.get(summary["primary_metric"])
            if row:
                lines.append(
                    f"- `{group}`: {row['baseline']:.6f} -> {row['candidate']:.6f}; "
                    f"delta {row['absolute_delta']:+.6f}; CI [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}]"
                )
    if summary["ragas"]:
        ragas_verdict = "端到端质量得到提升" if summary["ragas_material_improvement"] else "未验证出显著端到端提升"
        lines.extend(["", "## RAGAS deltas", "", f"Verdict: **{ragas_verdict}**", ""])
        for metric, row in sorted(summary["ragas"].items()):
            lines.append(
                f"- `{metric}`: {row['baseline']:.6f} -> {row['candidate']:.6f}; "
                f"delta {row['absolute_delta']:+.6f}; CI [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}]"
            )
        lines.extend(["", "RAGAS coverage: `" + json.dumps(summary["ragas_coverage"], sort_keys=True) + "`"])
    lines.extend(["", "Results from sampled candidate pools are not official full-corpus benchmark scores."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path)) if path.exists() else 0


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two compatible Chinese benchmark runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare_runs(
        args.baseline,
        args.candidate,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
