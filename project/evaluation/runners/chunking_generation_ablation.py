import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluation.io import write_jsonl, write_metrics_csv
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.runners import chunking_ablation as retrieval_ablation
from evaluation.runners.ragbench_eval_runner import generate_answer
from evaluation.validation import (
    build_validity_summary,
    make_warning,
    validate_ragas_rows,
    validation_markdown_section,
    write_validation_outputs,
)
from retrieval.embeddings import DenseEmbeddingModel


DEFAULT_VARIANTS = "single_1200_240,pc_child,pc_neighbor,pc_parent,pc_adaptive"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generation-side quality for chunking variants on RAGBench source contexts."
    )
    parser.add_argument(
        "--source-contexts",
        default=(
            "runtime/evaluation_reports/eval_runs/"
            "run_2026_05_11_115156_covidqa200_rrf_rerank/"
            "ragbench_covidqa_test_200_source_contexts.jsonl"
        ),
    )
    parser.add_argument("--output-dir", default="runtime/evaluation_reports/chunking_generation_ablation")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--sparse-top-k", type=int, default=50)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--transport", choices=("openai", "powershell-deepseek"), default="openai")
    parser.add_argument("--judge-mode", choices=("ragas", "powershell", "none"), default="ragas")
    parser.add_argument("--powershell-proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--skip-ragas", action="store_true", help="Alias for --judge-mode none when using legacy commands.")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=2)
    parser.add_argument("--ragas-max-workers", type=int, default=2)
    parser.add_argument("--ragas-batch-size", type=int, default=2)
    args = parser.parse_args()

    rows = retrieval_ablation.read_jsonl(args.source_contexts)[: args.limit]
    source_docs = retrieval_ablation.build_source_docs(rows)
    variants = [retrieval_ablation.parse_variant(item.strip()) for item in args.variants.split(",") if item.strip()]

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_chunking_generation_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_warnings = []
    if not rows:
        validation_warnings.append(
            make_warning(
                "empty_dataset",
                "Chunking generation ablation received no rows; no metric values are usable for conclusions.",
                severity="error",
            )
        )
        validity_summary = build_validity_summary(
            rows=0,
            warnings=validation_warnings,
            evaluation_type="chunking_generation_ablation",
        )
        write_validation_outputs(output_dir, validation_warnings, validity_summary)
        write_jsonl(output_dir / "summary.jsonl", [])
        write_summary_csv(output_dir / "summary.csv", [])
        (output_dir / "report.md").write_text(
            "\n".join(
                ["# Chunking Generation Ablation Report", "", "No rows were available for evaluation."]
                + validation_markdown_section(validation_warnings, validity_summary)
            )
            + "\n",
            encoding="utf-8",
        )
        print(output_dir)
        return
    validity_summary = build_validity_summary(
        rows=len(rows),
        warnings=validation_warnings,
        evaluation_type="chunking_generation_ablation",
    )
    write_validation_outputs(output_dir, validation_warnings, validity_summary)

    questions = [row["question"] for row in rows]
    model = DenseEmbeddingModel(local_files_only=True)
    query_embeddings = model.encode_queries(questions, batch_size=32, show_progress_bar=True)

    summaries = []
    for variant in variants:
        print(f"Running generation ablation for {variant.name}...", flush=True)
        outputs, retrieval_metrics = run_variant(
            rows=rows,
            source_docs=source_docs,
            variant=variant,
            query_embeddings=query_embeddings,
            model=model,
            top_k=args.top_k,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            answer_model=args.answer_model,
            transport=args.transport,
            powershell_proxy=args.powershell_proxy,
        )
        variant_dir = output_dir / variant.name
        write_jsonl(variant_dir / "rag_outputs.jsonl", outputs)
        write_metrics_csv(variant_dir / "retrieval_metrics.csv", retrieval_metrics)

        ragas_metrics: dict[str, float] = {"rows": float(len(outputs))}
        judge_metrics: dict[str, float] = {"rows": float(len(outputs))}
        error_cases: list[dict[str, Any]] = []
        variant_warnings = []
        judge_mode = "none" if args.skip_ragas else args.judge_mode
        if judge_mode == "ragas":
            ragas_results, ragas_metrics = run_ragas_metrics(
                outputs,
                timeout=args.ragas_timeout,
                max_retries=args.ragas_max_retries,
                max_workers=args.ragas_max_workers,
                batch_size=args.ragas_batch_size,
            )
            error_cases = build_ragas_error_cases(ragas_results)
            variant_warnings.extend(validate_ragas_rows(ragas_results, expected_rows=len(outputs)))
            write_jsonl(variant_dir / "ragas_results.jsonl", ragas_results)
            write_jsonl(variant_dir / "ragas_error_cases.jsonl", error_cases)
            write_metrics_csv(variant_dir / "ragas_metrics_summary.csv", ragas_metrics)
        elif judge_mode == "powershell":
            judge_results, judge_metrics = run_powershell_judge(outputs, args.powershell_proxy, args.answer_model)
            write_jsonl(variant_dir / "llm_judge_results.jsonl", judge_results)
            write_metrics_csv(variant_dir / "llm_judge_metrics_summary.csv", judge_metrics)

        summary = {
            "variant": variant.name,
            "strategy": variant.strategy,
            **{f"retrieval_{key}": value for key, value in retrieval_metrics.items()},
            **{f"ragas_{key}": value for key, value in ragas_metrics.items()},
            **{f"judge_{key}": value for key, value in judge_metrics.items()},
            "ragas_error_cases": float(len(error_cases)),
        }
        summaries.append(summary)
        validation_warnings.extend(
            {
                **warning,
                "details": {
                    **(warning.get("details") or {}),
                    "variant": variant.name,
                },
            }
            for warning in variant_warnings
        )
        write_validation_outputs(
            variant_dir,
            variant_warnings,
            build_validity_summary(
                rows=len(outputs),
                warnings=variant_warnings,
                evaluation_type=f"chunking_generation_ablation:{variant.name}",
            ),
        )

    write_jsonl(output_dir / "summary.jsonl", summaries)
    write_summary_csv(output_dir / "summary.csv", summaries)
    validity_summary = build_validity_summary(
        rows=len(rows),
        warnings=validation_warnings,
        evaluation_type="chunking_generation_ablation",
    )
    write_validation_outputs(output_dir, validation_warnings, validity_summary)
    write_report(output_dir / "report.md", summaries, args, len(rows), validation_warnings, validity_summary)
    print(output_dir)


def run_variant(
    rows: list[dict],
    source_docs: list[retrieval_ablation.SourceDoc],
    variant: retrieval_ablation.Variant,
    query_embeddings,
    model: DenseEmbeddingModel,
    top_k: int,
    dense_top_k: int,
    sparse_top_k: int,
    answer_model: str | None,
    transport: str,
    powershell_proxy: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    chunks, parents = retrieval_ablation.build_variant_chunks(source_docs, variant)
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_embeddings = model.encode_documents(chunk_texts, batch_size=64, show_progress_bar=True)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    sparse_matrix = vectorizer.fit_transform(chunk_texts)
    query_sparse = vectorizer.transform(row["question"] for row in rows)

    outputs = []
    retrieval_rows = []
    for index, row in enumerate(rows, start=1):
        dense_scores = np.asarray(chunk_embeddings @ query_embeddings[index - 1].T).reshape(-1)
        sparse_scores = query_sparse[index - 1] @ sparse_matrix.T
        sparse_scores = np.asarray(sparse_scores.toarray()).reshape(-1)
        ranking = retrieval_ablation.rrf(
            dense_scores,
            sparse_scores,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )
        retrieved = [chunks[item] for item in ranking[:top_k]]
        contexts = retrieval_ablation.contexts_for_variant(
            variant,
            retrieved,
            parents,
            chunks,
            row.get("question", ""),
        )
        context_texts = [context.text for context in contexts]
        retrieval_rows.append(
            retrieval_ablation.score_question(row, variant, retrieved, parents, chunks, sorted({1, top_k}))
        )

        print(f"[{variant.name}] generating {index}/{len(rows)}", flush=True)
        answer = generate_with_transport(
            row["question"],
            context_texts,
            answer_model,
            transport=transport,
            powershell_proxy=powershell_proxy,
        )
        outputs.append(
            {
                "question_id": f"{row['question_id']}_{variant.name}",
                "source_question_id": row["question_id"],
                "variant": variant.name,
                "strategy": variant.strategy,
                "question": row["question"],
                "user_input": row["question"],
                "answer": answer,
                "response": answer,
                "contexts": context_texts,
                "retrieved_contexts": context_texts,
                "reference": row.get("reference_response") or row.get("response") or "",
                "ground_truth": row.get("reference_response") or row.get("response") or "",
                "context_chars": sum(len(text) for text in context_texts),
                "context_count": len(context_texts),
            }
        )

    retrieval_metrics = retrieval_ablation.aggregate_metrics(retrieval_rows, chunks, variant)
    return outputs, retrieval_metrics


def generate_with_transport(
    question: str,
    contexts: list[str],
    answer_model: str | None,
    transport: str,
    powershell_proxy: str,
) -> str:
    if transport == "openai":
        return generate_answer(question, contexts, answer_model)

    context = "\n\n".join(f"[Document {index + 1}]\n{text}" for index, text in enumerate(contexts))
    prompt = (
        f"Question:\n{question}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer the question using only the documents. Keep the answer concise, preferably 1-3 sentences. "
        "If the documents do not contain enough information, say so."
    )
    return powershell_deepseek_chat(
        system="You are a precise RAG answer generator.",
        prompt=prompt,
        max_tokens=int(os.environ.get("ANSWER_MAX_TOKENS", "512")),
        proxy=powershell_proxy,
        model=answer_model,
    )


def run_powershell_judge(
    outputs: list[dict[str, Any]],
    proxy: str,
    model: str | None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for index, output in enumerate(outputs, start=1):
        print(f"[{output['variant']}] judging {index}/{len(outputs)}", flush=True)
        score = judge_output(output, proxy=proxy, model=model)
        row = dict(output)
        row.update(score)
        rows.append(row)
    return rows, summarize_judge_rows(rows)


def judge_output(output: dict[str, Any], proxy: str, model: str | None) -> dict[str, float | str]:
    contexts = "\n\n".join(f"[Context {index + 1}]\n{text}" for index, text in enumerate(output["retrieved_contexts"]))
    prompt = (
        "Score this RAG answer. Return JSON only with keys "
        "faithfulness, answer_correctness, context_recall, context_precision, explanation. "
        "Each numeric score must be between 0 and 1.\n\n"
        f"Question:\n{output['question']}\n\n"
        f"Reference answer:\n{output['reference']}\n\n"
        f"Retrieved contexts:\n{contexts}\n\n"
        f"Generated answer:\n{output['answer']}\n"
    )
    raw = powershell_deepseek_chat(
        system="You are a strict evaluator for retrieval-augmented generation.",
        prompt=prompt,
        max_tokens=512,
        proxy=proxy,
        model=model,
    )
    parsed = parse_json_object(raw)
    return {
        "judge_faithfulness": clamp_float(parsed.get("faithfulness")),
        "judge_answer_correctness": clamp_float(parsed.get("answer_correctness")),
        "judge_context_recall": clamp_float(parsed.get("context_recall")),
        "judge_context_precision": clamp_float(parsed.get("context_precision")),
        "judge_explanation": str(parsed.get("explanation") or raw)[:1000],
    }


def powershell_deepseek_chat(
    system: str,
    prompt: str,
    max_tokens: int,
    proxy: str,
    model: str | None,
) -> str:
    payload = {
        "system": system,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": model or os.environ.get("ANSWER_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
        "base_url": os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
        "proxy": proxy,
        "api_key": os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    }
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$headers = @{
  Authorization = "Bearer $($payload.api_key)"
  "Content-Type" = "application/json"
}
$body = @{
  model = $payload.model
  temperature = 0
  max_tokens = [int]$payload.max_tokens
  messages = @(
    @{ role = "system"; content = $payload.system },
    @{ role = "user"; content = $payload.prompt }
  )
} | ConvertTo-Json -Depth 10 -Compress
$uri = ($payload.base_url.TrimEnd('/')) + "/chat/completions"
$response = Invoke-RestMethod -Uri $uri -Method Post -Headers $headers -Body $body -ContentType "application/json" -Proxy $payload.proxy -TimeoutSec 180
$response.choices[0].message.content
"""
    completed = subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile", "-Command", script],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "PowerShell DeepSeek call failed")
    return completed.stdout.strip()


def parse_json_object(raw: str) -> dict[str, Any]:
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
        return 0.0
    return max(0.0, min(1.0, number))


def summarize_judge_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "judge_faithfulness",
        "judge_answer_correctness",
        "judge_context_recall",
        "judge_context_precision",
    ]
    summary = {"rows": float(len(rows))}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key.removeprefix("judge_")] = sum(values) / len(values)
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "variant",
        "strategy",
        "retrieval_mrr",
        "retrieval_hitrate@1",
        "retrieval_hitrate@5",
        "retrieval_sentence_recall@5",
        "retrieval_sentence_precision@5",
        "retrieval_context_chars@5",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_context_recall",
        "judge_faithfulness",
        "judge_answer_correctness",
        "judge_context_recall",
        "judge_context_precision",
        "ragas_rows",
        "judge_rows",
        "ragas_error_cases",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(",".join(keys) + "\n")
        for row in rows:
            file.write(",".join(format_value(row.get(key, "")) for key in keys) + "\n")


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    args,
    limit: int,
    validation_warnings: list[dict] | None = None,
    validity_summary: dict | None = None,
) -> None:
    lines = [
        "# Chunking Generation Ablation Report",
        "",
        f"- Dataset: `{args.source_contexts}`",
        f"- Rows: {limit}",
        f"- Variants: `{args.variants}`",
        f"- Top-k contexts: {args.top_k}",
        "- Answer generation: same prompt/model, temperature=0",
        f"- Judge mode: `{args.judge_mode}`",
        "",
        "## Summary",
        "",
        "| Variant | Strategy | MRR | Hit@1 | Hit@5 | SentenceRecall@5 | Faithfulness | Correctness | ContextRecall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        faithfulness = row.get("ragas_faithfulness", row.get("judge_faithfulness", 0.0))
        correctness = row.get("judge_answer_correctness", row.get("ragas_context_precision", 0.0))
        context_recall = row.get("ragas_context_recall", row.get("judge_context_recall", 0.0))
        lines.append(
            "| {variant} | {strategy} | {mrr:.4f} | {hit1:.4f} | {hit5:.4f} | {sentrec:.4f} | "
            "{faith:.4f} | {correct:.4f} | {ctxrec:.4f} |".format(
                variant=row["variant"],
                strategy=row["strategy"],
                mrr=row.get("retrieval_mrr", 0.0),
                hit1=row.get("retrieval_hitrate@1", 0.0),
                hit5=row.get("retrieval_hitrate@5", 0.0),
                sentrec=row.get("retrieval_sentence_recall@5", 0.0),
                faith=faithfulness,
                correct=correctness,
                ctxrec=context_recall,
            )
        )
    if len(rows) >= 2:
        left, right = rows[0], rows[1]
        lines.extend(
            [
                "",
                "## Delta",
                "",
                f"- Compared `{right['variant']}` against `{left['variant']}`.",
            ]
        )
        for key in ["ragas_faithfulness", "judge_faithfulness", "judge_answer_correctness", "ragas_context_recall", "judge_context_recall"]:
            if key in left or key in right:
                lines.append(f"- `{key}`: {right.get(key, 0.0) - left.get(key, 0.0):+.4f}")
    lines.extend(validation_markdown_section(validation_warnings or [], validity_summary))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
