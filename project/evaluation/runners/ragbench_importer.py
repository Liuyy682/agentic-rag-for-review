import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.io import write_jsonl
from evaluation.ragbench_keys import document_id_from_sentence_key


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
RAGBENCH_DATASET = "galileo-ai/ragbench"
RAGBENCH_SUBSETS = [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]


def import_ragbench(
    subset: str,
    split: str,
    limit: int,
    output_dataset: str,
    output_contexts: str,
    offset: int = 0,
    page_size: int = 25,
) -> Dict[str, Any]:
    rows = fetch_ragbench_rows(subset=subset, split=split, limit=limit, offset=offset, page_size=page_size)
    eval_rows = [to_eval_question(row, subset, split) for row in rows]
    context_rows = [to_context_record(row, subset, split) for row in rows]
    write_jsonl(output_dataset, eval_rows)
    write_jsonl(output_contexts, context_rows)
    return {
        "subset": subset,
        "split": split,
        "rows": len(rows),
        "output_dataset": output_dataset,
        "output_contexts": output_contexts,
    }


def fetch_ragbench_rows(
    subset: str,
    split: str,
    limit: int,
    offset: int = 0,
    page_size: int = 25,
    max_attempts: int = 5,
    request_timeout: int = 60,
) -> List[Dict[str, Any]]:
    if subset not in RAGBENCH_SUBSETS:
        raise ValueError(f"Unknown RAGBench subset: {subset}. Choose one of: {', '.join(RAGBENCH_SUBSETS)}")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test")
    if limit <= 0:
        raise ValueError("limit must be positive")

    rows: List[Dict[str, Any]] = []
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        params = {
            "dataset": RAGBENCH_DATASET,
            "config": subset,
            "split": split,
            "offset": offset + len(rows),
            "length": length,
        }
        response = _get_with_retries(
            HF_ROWS_URL,
            params=params,
            timeout=request_timeout,
            max_attempts=max_attempts,
        )
        payload = response.json()
        page_rows = [item["row"] for item in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
    return rows[:limit]


def _get_with_retries(
    url: str,
    params: Dict[str, Any],
    timeout: int,
    max_attempts: int,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"RAGBench fetch failed (attempt {attempt}/{max_attempts}, "
                f"offset={params.get('offset')}, length={params.get('length')}): {exc}. "
                f"Retrying in {delay}s...",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Could not fetch RAGBench rows after {max_attempts} attempts "
        f"(offset={params.get('offset')}, length={params.get('length')})."
    ) from last_error


def to_eval_question(row: Dict[str, Any], subset: str, split: str) -> Dict[str, Any]:
    question_id = make_question_id(row, subset, split)
    relevant_sentence_keys = list(row.get("all_relevant_sentence_keys") or [])
    relevant_doc_ids = sorted({document_id_from_sentence_key(key) for key in relevant_sentence_keys if key})
    gold_parent_ids = [f"{question_id}_doc_{doc_id}" for doc_id in relevant_doc_ids]
    gold_child_ids = [f"{question_id}_sent_{key}" for key in relevant_sentence_keys]
    return {
        "question_id": question_id,
        "question": row.get("question", ""),
        "reference_answer": row.get("response", ""),
        "source_file": f"ragbench/{subset}/{split}/{row.get('id')}",
        "gold_parent_ids": gold_parent_ids,
        "gold_child_ids": gold_child_ids,
        "gold_evidence_text": extract_evidence_text(row, relevant_sentence_keys),
        "question_type": subset,
        "difficulty": "unknown",
        "tags": ["ragbench", subset, split],
    }


def to_context_record(row: Dict[str, Any], subset: str, split: str) -> Dict[str, Any]:
    question_id = make_question_id(row, subset, split)
    return {
        "question_id": question_id,
        "subset": subset,
        "split": split,
        "ragbench_id": row.get("id"),
        "question": row.get("question", ""),
        "documents": row.get("documents") or [],
        "documents_sentences": row.get("documents_sentences") or [],
        "reference_response": row.get("response", ""),
        "all_relevant_sentence_keys": row.get("all_relevant_sentence_keys") or [],
        "all_utilized_sentence_keys": row.get("all_utilized_sentence_keys") or [],
        "unsupported_response_sentence_keys": row.get("unsupported_response_sentence_keys") or [],
        "adherence_score": row.get("adherence_score"),
        "relevance_score": row.get("relevance_score"),
        "utilization_score": row.get("utilization_score"),
        "completeness_score": row.get("completeness_score"),
        "ragas_faithfulness": row.get("ragas_faithfulness"),
        "ragas_context_relevance": row.get("ragas_context_relevance"),
        "overall_supported_explanation": row.get("overall_supported_explanation"),
        "relevance_explanation": row.get("relevance_explanation"),
    }


def extract_evidence_text(row: Dict[str, Any], relevant_sentence_keys: Iterable[str]) -> List[str]:
    wanted = set(relevant_sentence_keys)
    evidence: List[str] = []
    for sentence_group in row.get("documents_sentences") or []:
        for key, text in sentence_group:
            if key in wanted:
                evidence.append(text)
    return evidence


def make_question_id(row: Dict[str, Any], subset: str, split: str) -> str:
    raw_id = str(row.get("id", "")).replace("/", "_")
    return f"ragbench_{subset}_{split}_{raw_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import rows from Hugging Face RAGBench into local eval JSONL files.")
    parser.add_argument("--subset", default="covidqa", choices=RAGBENCH_SUBSETS)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument(
        "--output-dataset",
        default=str(PROJECT_DIR / "evaluation" / "datasets" / "ragbench_eval_questions.jsonl"),
    )
    parser.add_argument(
        "--output-contexts",
        default=str(PROJECT_DIR / "evaluation" / "datasets" / "ragbench_contexts.jsonl"),
    )
    args = parser.parse_args()

    result = import_ragbench(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        output_dataset=args.output_dataset,
        output_contexts=args.output_contexts,
        offset=args.offset,
        page_size=args.page_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
