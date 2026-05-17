import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import config


def make_run_id(label: str = "baseline") -> str:
    now = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    clean_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return f"run_{now}_{clean_label}"


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(path)
    rows: List[Dict[str, Any]] = []
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_metrics_csv(path: str | Path, metrics: Dict[str, float]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "score": f"{metrics[key]:.6f}"})


def read_metrics_csv(path: str | Path) -> Dict[str, float]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {input_path}")
    with input_path.open("r", encoding="utf-8", newline="") as f:
        return {row["metric"]: float(row["score"]) for row in csv.DictReader(f)}


def config_snapshot(
    run_id: str,
    dataset_path: str,
    dataset_version: str,
    top_k: int,
    score_threshold: float | None = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "git_commit": _git_commit(),
        "dataset_path": dataset_path,
        "dataset_version": dataset_version,
        "document_converter": getattr(config, "DOCUMENT_CONVERTER", "markitdown"),
        "markdown_cleaner": False,
        "chunk_size": config.CHILD_CHUNK_SIZE,
        "chunk_overlap": config.CHILD_CHUNK_OVERLAP,
        "parent_min_size": config.MIN_PARENT_SIZE,
        "parent_max_size": config.MAX_PARENT_SIZE,
        "dense_model": config.DENSE_MODEL,
        "sparse_model": config.SPARSE_MODEL,
        "retrieval_mode": "hybrid",
        "top_k": top_k,
        "score_threshold": score_threshold,
        "reranker": None,
        "llm_model": config.LLM_MODEL,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
