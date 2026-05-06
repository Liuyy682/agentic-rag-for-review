import os
import math
from typing import Any, Dict, List

import config
from evaluation.llm_config import api_key, base_url, judge_model


def run_ragas_metrics(
    outputs: List[Dict[str, Any]],
    timeout: int = 180,
    max_retries: int = 2,
    max_workers: int = 2,
    batch_size: int | None = 1,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import llm_factory
        from ragas.metrics import _AnswerRelevancy, _ContextPrecision, _ContextRecall, _Faithfulness
        from ragas.run_config import RunConfig
        from openai import APIStatusError, OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. Install `ragas` and `datasets`, "
            "or rerun without `--ragas` to only save/import benchmark outputs."
        ) from exc

    dataset = Dataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["reference"],
            }
            for row in outputs
        ]
    )
    client = OpenAI(
        api_key=api_key(),
        base_url=base_url(),
        timeout=timeout,
        max_retries=max_retries,
    )
    llm = llm_factory(
        judge_model(),
        client=client,
        max_tokens=int(os.environ.get("RAGAS_MAX_TOKENS", "4096")),
        temperature=0,
    )
    metrics = [
        _Faithfulness(llm=llm),
        _ContextPrecision(llm=llm),
        _ContextRecall(llm=llm),
    ]
    if os.environ.get("RAGAS_ENABLE_ANSWER_RELEVANCY", "false").lower() == "true":
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings

        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=api_key(),
                base_url=base_url(),
            )
        )
        metrics.append(_AnswerRelevancy(llm=llm, embeddings=embeddings))

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            raise_exceptions=False,
            run_config=RunConfig(timeout=timeout, max_retries=max_retries, max_workers=max_workers),
            batch_size=batch_size,
        )
    except APIStatusError as exc:
        raise RuntimeError(
            f"RAGAS judge request failed with HTTP {exc.status_code}. "
            f"Check DEEPSEEK_API_KEY/OPENAI_API_KEY, base URL ({base_url()}), "
            f"RAGAS_LLM_MODEL/DEEPSEEK_MODEL, and RAGAS_EMBEDDING_MODEL."
        ) from exc
    rows = result.to_pandas().to_dict(orient="records")
    metrics_summary = summarize_ragas_rows(rows)

    merged = []
    for source, score_row in zip(outputs, rows):
        merged_row = dict(source)
        merged_row.update(score_row)
        merged.append(merged_row)
    return merged, metrics_summary


def summarize_ragas_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    metric_keys = [
        "faithfulness",
        "answer_relevancy",
        "response_relevancy",
        "context_precision",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    summary: Dict[str, float] = {"rows": float(len(rows))}
    for key in metric_keys:
        values = [_to_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            output_key = "context_precision" if key == "llm_context_precision_with_reference" else key
            summary[output_key] = sum(values) / len(values)
    return summary


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


def build_ragas_error_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    thresholds = {
        "faithfulness": 0.75,
        "answer_relevancy": 0.75,
        "response_relevancy": 0.75,
        "context_precision": 0.65,
        "context_recall": 0.70,
    }
    for row in rows:
        low = [metric for metric, threshold in thresholds.items() if row.get(metric) is not None and row.get(metric) < threshold]
        if not low:
            continue
        failure_type = "hallucination" if "faithfulness" in low else "low_answer_quality"
        if "context_recall" in low:
            failure_type = "insufficient_context"
        cases.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": row["answer"],
                "failure_type": failure_type,
                "low_metrics": low,
                "retrieved_contexts": row["retrieved_contexts"],
            }
        )
    return cases
