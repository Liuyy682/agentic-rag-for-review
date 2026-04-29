import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.io import read_jsonl, read_metrics_csv, write_jsonl, write_metrics_csv
from evaluation.llm_config import answer_model as resolve_answer_model
from evaluation.llm_config import api_key, base_url
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.runners.ragbench_importer import fetch_ragbench_rows


def run_ragbench_eval(
    subset: str,
    split: str,
    limit: int,
    output_dir: str,
    offset: int = 0,
    answer_model: str | None = None,
    ragas_timeout: int = 180,
    ragas_max_retries: int = 2,
    ragas_max_workers: int = 1,
    ragas_batch_size: int | None = 1,
    reuse_existing: bool = False,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = fetch_ragbench_rows(subset=subset, split=split, limit=limit, offset=offset)
    retrieval_metrics, retrieval_details = compute_ragbench_retrieval_metrics(rows)

    if reuse_existing and (output / "rag_outputs.jsonl").exists() and (output / "ragas_metrics_summary.csv").exists():
        outputs = read_jsonl(output / "rag_outputs.jsonl")
        ragas_results = read_jsonl(output / "ragas_results.jsonl") if (output / "ragas_results.jsonl").exists() else []
        ragas_metrics = read_metrics_csv(output / "ragas_metrics_summary.csv")
    else:
        outputs = [answer_ragbench_row(row, subset, split, answer_model) for row in rows]
        ragas_results, ragas_metrics = run_ragas_metrics(
            outputs,
            timeout=ragas_timeout,
            max_retries=ragas_max_retries,
            max_workers=ragas_max_workers,
            batch_size=ragas_batch_size,
        )
    error_cases = build_ragas_error_cases(ragas_results)

    write_jsonl(output / "rag_outputs.jsonl", outputs)
    write_jsonl(output / "ragas_results.jsonl", ragas_results)
    write_jsonl(output / "ragas_error_cases.jsonl", error_cases)
    write_jsonl(output / "retrieval_metrics_by_question.jsonl", retrieval_details)
    write_metrics_csv(output / "ragas_metrics_summary.csv", ragas_metrics)
    write_metrics_csv(output / "retrieval_metrics_summary.csv", retrieval_metrics)
    write_report(output / "ragbench_ragas_report.md", subset, split, limit, ragas_metrics, retrieval_metrics, error_cases)

    return {
        "subset": subset,
        "split": split,
        "rows": len(rows),
        "rag_outputs": str(output / "rag_outputs.jsonl"),
        "ragas_results": str(output / "ragas_results.jsonl"),
        "ragas_metrics": ragas_metrics,
        "retrieval_metrics": retrieval_metrics,
        "report": str(output / "ragbench_ragas_report.md"),
    }


def answer_ragbench_row(
    row: Dict[str, Any],
    subset: str,
    split: str,
    answer_model: str | None,
) -> Dict[str, Any]:
    question = row.get("question", "")
    documents = list(row.get("documents") or [])
    answer = generate_answer(question, documents, answer_model)
    question_id = f"ragbench_{subset}_{split}_{str(row.get('id', '')).replace('/', '_')}"
    return {
        "question_id": question_id,
        "question": question,
        "user_input": question,
        "answer": answer,
        "response": answer,
        "contexts": documents,
        "retrieved_contexts": documents,
        "reference": row.get("response", ""),
        "ground_truth": row.get("response", ""),
    }


def generate_answer(question: str, documents: List[str], answer_model: str | None) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key(),
        base_url=base_url(),
    )
    model = resolve_answer_model(answer_model)
    context = "\n\n".join(f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(documents))
    prompt = (
        f"Question:\n{question}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer the question using only the documents. Keep the answer concise, preferably 1-3 sentences. "
        "If the documents do not contain enough information, say so."
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=int(os.environ.get("ANSWER_MAX_TOKENS", "512")),
        messages=[
            {"role": "system", "content": "You are a precise RAG answer generator."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def compute_ragbench_retrieval_metrics(rows: List[Dict[str, Any]]) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    details = [score_ragbench_context_order(row) for row in rows]
    keys = sorted({key for row in details for key in row if key not in {"question_id", "question"}})
    summary = {"rows": float(len(details))}
    for key in keys:
        values = [row[key] for row in details if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = sum(values) / len(values)
    return summary, details


def score_ragbench_context_order(row: Dict[str, Any]) -> Dict[str, Any]:
    relevant_sentences = set(row.get("all_relevant_sentence_keys") or [])
    sentence_order = [sentence[0] for group in row.get("documents_sentences") or [] for sentence in group]
    document_order = [str(index) for index, _ in enumerate(row.get("documents") or [])]
    relevant_documents = sorted({_document_id_from_sentence_key(key) for key in relevant_sentences if key})

    result: Dict[str, Any] = {
        "question_id": row.get("id"),
        "question": row.get("question", ""),
        "sentence_mrr": reciprocal_rank(sentence_order, relevant_sentences),
        "document_mrr": reciprocal_rank(document_order, set(relevant_documents)),
    }
    for k in [1, 3, 5, 10, 20]:
        result.update(prefix_metrics("sentence", k, sentence_order, relevant_sentences))
        result.update(prefix_metrics("document", k, document_order, set(relevant_documents)))
    return result


def prefix_metrics(prefix: str, k: int, ranked_ids: List[str], relevant_ids: set[str]) -> Dict[str, float]:
    top_k = ranked_ids[:k]
    hits = [item for item in top_k if item in relevant_ids]
    return {
        f"{prefix}_recall@{k}": len(set(hits)) / len(relevant_ids) if relevant_ids else 0.0,
        f"{prefix}_precision@{k}": len(hits) / k if k else 0.0,
        f"{prefix}_hitrate@{k}": 1.0 if hits else 0.0,
        f"{prefix}_ndcg@{k}": ndcg_at_k(top_k, relevant_ids, k),
    }


def reciprocal_rank(ranked_ids: List[str], relevant_ids: set[str]) -> float:
    for index, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(top_k: List[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum((1.0 / math.log2(index + 2)) for index, item in enumerate(top_k[:k]) if item in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def _document_id_from_sentence_key(key: str) -> str:
    digits = []
    for char in str(key):
        if char.isdigit():
            digits.append(char)
        else:
            break
    return "".join(digits) if digits else ""


def write_report(
    path: Path,
    subset: str,
    split: str,
    limit: int,
    ragas_metrics: Dict[str, float],
    retrieval_metrics: Dict[str, float],
    error_cases: List[Dict[str, Any]],
) -> None:
    lines = [
        "# RAGBench + RAGAS Report",
        "",
        "## Dataset",
        f"- subset: `{subset}`",
        f"- split: `{split}`",
        f"- rows: {limit}",
        "",
        "## RAGAS Metrics",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(ragas_metrics.items()))
    lines.extend(["", "## Retrieval Metrics", "| Metric | Score |", "|---|---:|"])
    important_keys = [
        "document_recall@1",
        "document_recall@3",
        "document_recall@5",
        "document_precision@3",
        "document_hitrate@3",
        "document_mrr",
        "document_ndcg@3",
        "sentence_recall@5",
        "sentence_recall@10",
        "sentence_recall@20",
        "sentence_precision@10",
        "sentence_hitrate@10",
        "sentence_mrr",
        "sentence_ndcg@10",
    ]
    for key in important_keys:
        if key in retrieval_metrics:
            lines.append(f"| {key} | {retrieval_metrics[key]:.4f} |")
    lines.extend(["", "## Failure Cases"])
    if error_cases:
        for case in error_cases[:20]:
            lines.append(f"- `{case['question_id']}` {case['failure_type']}: {case['question'][:140]}")
    else:
        lines.append("- No failure cases detected by current thresholds.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAGBench subset by answering with question+documents, then scoring with RAGAS."
    )
    parser.add_argument("--subset", default="covidqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "evaluation" / "reports" / "ragbench"))
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--generate", action="store_true", help="Kept for compatibility; answer generation always runs.")
    parser.add_argument("--ragas", action="store_true", help="Kept for compatibility; RAGAS always runs in this runner.")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=2)
    parser.add_argument("--ragas-max-workers", type=int, default=1)
    parser.add_argument("--ragas-batch-size", type=int, default=1)
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing RAG/RAGAS outputs and refresh retrieval metrics/report.")
    args = parser.parse_args()

    try:
        result = run_ragbench_eval(
            subset=args.subset,
            split=args.split,
            limit=args.limit,
            output_dir=args.output_dir,
            offset=args.offset,
            answer_model=args.answer_model,
            ragas_timeout=args.ragas_timeout,
            ragas_max_retries=args.ragas_max_retries,
            ragas_max_workers=args.ragas_max_workers,
            ragas_batch_size=args.ragas_batch_size,
            reuse_existing=args.reuse_existing,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
