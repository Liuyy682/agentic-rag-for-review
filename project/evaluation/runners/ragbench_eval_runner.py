import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from evaluation.io import read_jsonl, write_jsonl, write_metrics_csv
from evaluation.reports import write_compare_report
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.runners.ragbench_importer import import_ragbench


def run_ragbench_eval(
    subset: str,
    split: str,
    limit: int,
    output_dir: str,
    generate: bool = False,
    ragas: bool = False,
    offset: int = 0,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / f"ragbench_{subset}_{split}_{limit}_eval_questions.jsonl"
    contexts_path = output / f"ragbench_{subset}_{split}_{limit}_contexts.jsonl"

    import_result = import_ragbench(
        subset=subset,
        split=split,
        limit=limit,
        output_dataset=str(dataset_path),
        output_contexts=str(contexts_path),
        offset=offset,
    )
    rows = read_jsonl(contexts_path)
    official_metrics = summarize_official_metrics(rows)
    write_metrics_csv(output / "ragbench_official_metrics_summary.csv", official_metrics)
    write_ragbench_report(output / "ragbench_report.md", import_result, official_metrics, [])

    result: Dict[str, Any] = {
        "dataset": str(dataset_path),
        "contexts": str(contexts_path),
        "official_metrics": official_metrics,
        "report": str(output / "ragbench_report.md"),
    }

    if ragas and not generate:
        ragas_outputs = to_ragas_outputs(rows, response_source="ragbench")
        ragas_results, ragas_metrics = run_ragas_metrics(ragas_outputs)
        ragas_error_cases = build_ragas_error_cases(ragas_results)
        write_jsonl(output / "ragbench_ragas_results.jsonl", ragas_results)
        write_jsonl(output / "ragbench_ragas_error_cases.jsonl", ragas_error_cases)
        write_metrics_csv(output / "ragbench_ragas_metrics_summary.csv", ragas_metrics)
        result["ragas_metrics"] = ragas_metrics

    if generate:
        generated_rows = generate_answers(rows)
        generated_metrics = summarize_generated_metrics(generated_rows)
        error_cases = build_generated_error_cases(generated_rows)
        write_jsonl(output / "ragbench_generated_outputs.jsonl", generated_rows)
        write_jsonl(output / "ragbench_generated_error_cases.jsonl", error_cases)
        write_metrics_csv(output / "ragbench_generated_metrics_summary.csv", generated_metrics)
        write_compare_report(
            output / "ragbench_generated_vs_reference_report.md",
            baseline={"reference_response_token_f1": 1.0},
            current=generated_metrics,
            baseline_label="Reference",
            current_label=config.LLM_MODEL,
        )
        write_ragbench_report(output / "ragbench_report.md", import_result, official_metrics, error_cases, generated_metrics)
        result["generated_metrics"] = generated_metrics

        if ragas:
            ragas_outputs = to_ragas_outputs(rows, response_source="generated", generated_rows=generated_rows)
            ragas_results, ragas_metrics = run_ragas_metrics(ragas_outputs)
            ragas_error_cases = build_ragas_error_cases(ragas_results)
            write_jsonl(output / "ragbench_generated_ragas_results.jsonl", ragas_results)
            write_jsonl(output / "ragbench_generated_ragas_error_cases.jsonl", ragas_error_cases)
            write_metrics_csv(output / "ragbench_generated_ragas_metrics_summary.csv", ragas_metrics)
            result["generated_ragas_metrics"] = ragas_metrics

    return result


def summarize_official_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    data = list(rows)
    metrics = {
        "rows": float(len(data)),
        "adherence_rate": _mean(1.0 if row.get("adherence_score") else 0.0 for row in data),
        "relevance_score": _mean(_num(row.get("relevance_score")) for row in data),
        "utilization_score": _mean(_num(row.get("utilization_score")) for row in data),
        "completeness_score": _mean(_num(row.get("completeness_score")) for row in data),
        "ragas_faithfulness": _mean(_num(row.get("ragas_faithfulness")) for row in data),
        "ragas_context_relevance": _mean(_num(row.get("ragas_context_relevance")) for row in data),
        "unsupported_sentence_rate": _mean(
            1.0 if row.get("unsupported_response_sentence_keys") else 0.0 for row in data
        ),
    }
    return metrics


def generate_answers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        context = "\n\n".join(
            f"[Document {index}]\n{document}" for index, document in enumerate(row.get("documents") or [])
        )
        prompt = (
            f"Question:\n{row['question']}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer the question using only the retrieved context. If the context is insufficient, say so."
        )
        response = llm.invoke(
            [
                SystemMessage(content="You are a careful RAG answer generator."),
                HumanMessage(content=prompt),
            ]
        )
        answer = str(response.content)
        reference = row.get("reference_response", "")
        outputs.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": answer,
                "reference": reference,
                "token_f1": token_f1(answer, reference),
                "documents": row.get("documents") or [],
            }
        )
    return outputs


def to_ragas_outputs(
    rows: List[Dict[str, Any]],
    response_source: str,
    generated_rows: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    generated_by_id = {row["question_id"]: row for row in generated_rows or []}
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        generated = generated_by_id.get(row["question_id"], {})
        response = generated.get("answer") if response_source == "generated" else row.get("reference_response", "")
        reference = row.get("reference_response", "")
        contexts = row.get("documents") or []
        outputs.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "user_input": row["question"],
                "answer": response,
                "response": response,
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "reference": reference,
                "ground_truth": reference,
                "retrieved_metadata": [
                    {
                        "rank": index + 1,
                        "chunk_id": f"{row['question_id']}_doc_{index}",
                        "parent_id": f"{row['question_id']}_doc_{index}",
                        "source_file": f"ragbench/{row['subset']}/{row['split']}/{row['ragbench_id']}",
                    }
                    for index, _ in enumerate(contexts)
                ],
            }
        )
    return outputs


def summarize_generated_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    data = list(rows)
    return {
        "rows": float(len(data)),
        "generated_token_f1": _mean(_num(row.get("token_f1")) for row in data),
    }


def build_generated_error_cases(rows: Iterable[Dict[str, Any]], threshold: float = 0.35) -> List[Dict[str, Any]]:
    cases = []
    for row in rows:
        if row.get("token_f1", 0.0) < threshold:
            cases.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "reference": row["reference"],
                    "token_f1": row["token_f1"],
                    "failure_type": "low_reference_overlap",
                }
            )
    return cases


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def write_ragbench_report(
    path: str | Path,
    import_result: Dict[str, Any],
    official_metrics: Dict[str, float],
    error_cases: List[Dict[str, Any]],
    generated_metrics: Dict[str, float] | None = None,
) -> None:
    lines = [
        "# RAGBench Evaluation Report",
        "",
        "## Dataset",
        f"- subset: `{import_result['subset']}`",
        f"- split: `{import_result['split']}`",
        f"- rows: {import_result['rows']}",
        f"- eval dataset: `{import_result['output_dataset']}`",
        f"- contexts: `{import_result['output_contexts']}`",
        "",
        "## Official RAGBench Labels",
        "| Metric | Score |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(official_metrics.items()))
    if generated_metrics:
        lines.extend(["", "## Generated Answer Metrics", "| Metric | Score |", "|---|---:|"])
        lines.extend(f"| {key} | {value:.4f} |" for key, value in sorted(generated_metrics.items()))
    lines.extend(["", "## Generated Failure Cases"])
    if error_cases:
        for case in error_cases[:20]:
            lines.append(f"- `{case['question_id']}` token_f1={case['token_f1']:.3f}: {case['question'][:140]}")
    else:
        lines.append("- No generated-answer failure cases recorded.")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RAGBench rows and run automatic benchmark summaries.")
    parser.add_argument("--subset", default="covidqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "evaluation" / "reports" / "ragbench"))
    parser.add_argument("--generate", action="store_true", help="Call the configured local LLM with RAGBench contexts.")
    parser.add_argument("--ragas", action="store_true", help="Recompute RAGAS metrics using current local RAGAS setup.")
    args = parser.parse_args()

    try:
        result = run_ragbench_eval(
            subset=args.subset,
            split=args.split,
            limit=args.limit,
            output_dir=args.output_dir,
            generate=args.generate,
            ragas=args.ragas,
            offset=args.offset,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
