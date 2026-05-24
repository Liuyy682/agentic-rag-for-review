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
    llm_kwargs = {
        "client": client,
        "temperature": 0,
    }
    max_tokens = _ragas_max_tokens()
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens
    llm = llm_factory(judge_model(), **llm_kwargs)
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
    metric_groups = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy"],
        "response_relevancy": ["response_relevancy"],
        "context_precision": ["context_precision", "llm_context_precision_with_reference"],
        "context_recall": ["context_recall"],
    }
    summary: Dict[str, float] = {"rows": float(len(rows))}
    for output_key, source_keys in metric_groups.items():
        values = [_first_float(row, source_keys) for row in rows]
        valid_values = [value for value in values if value is not None]
        if valid_values:
            summary[output_key] = sum(valid_values) / len(valid_values)
        metric_present = any(key in row for row in rows for key in source_keys)
        if rows and (output_key in {"faithfulness", "context_precision", "context_recall"} or valid_values or metric_present):
            summary[f"{output_key}_rows"] = float(len(valid_values))
            summary[f"{output_key}_missing_rows"] = float(len(rows) - len(valid_values))
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


def _first_float(row: Dict[str, Any], keys: List[str]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _ragas_max_tokens() -> int | None:
    raw = os.environ.get("RAGAS_MAX_TOKENS", "4096").strip().lower()
    if raw in {"", "none", "null", "unlimited", "0"}:
        return None
    return int(raw)


def build_ragas_error_cases(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cases = []
    required_thresholds = {
        "faithfulness": 0.75,
        "context_precision": 0.65,
        "context_recall": 0.70,
    }
    optional_thresholds = {
        "answer_relevancy": 0.75,
        "response_relevancy": 0.75,
    }
    for row in rows:
        metric_values = {
            "faithfulness": _first_float(row, ["faithfulness"]),
            "context_precision": _first_float(row, ["context_precision", "llm_context_precision_with_reference"]),
            "context_recall": _first_float(row, ["context_recall"]),
            "answer_relevancy": _first_float(row, ["answer_relevancy"]),
            "response_relevancy": _first_float(row, ["response_relevancy"]),
        }
        missing = [metric for metric in required_thresholds if metric_values.get(metric) is None]
        low = [
            metric
            for metric, threshold in required_thresholds.items()
            if metric_values.get(metric) is not None and metric_values[metric] < threshold
        ]
        low.extend(
            metric
            for metric, threshold in optional_thresholds.items()
            if metric in row
            and metric_values.get(metric) is not None
            and metric_values[metric] < threshold
        )
        if not low and not missing:
            continue
        failure_type = "judge_metric_missing" if missing else ("hallucination" if "faithfulness" in low else "low_answer_quality")
        if "context_recall" in low:
            failure_type = "insufficient_context"
        cases.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "answer": row["answer"],
                "failure_type": failure_type,
                "low_metrics": low,
                "missing_metrics": missing,
                "retrieved_contexts": row["retrieved_contexts"],
            }
        )
    return cases
