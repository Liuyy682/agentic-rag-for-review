import os
from typing import Any, Dict, List

import config


def run_ragas_metrics(outputs: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from langchain_huggingface import HuggingFaceEmbeddings
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. Install `ragas` and `datasets`, "
            "or rerun without `--ragas` to only save/import benchmark outputs."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "RAGAS 0.4.x requires a structured-output judge LLM. Set OPENAI_API_KEY "
            "or configure a RAGAS-compatible InstructorLLM before running with `--ragas`."
        )

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
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm = llm_factory(os.environ.get("RAGAS_LLM_MODEL", "gpt-4o-mini"), client=client)
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.DENSE_MODEL))
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
    )
    rows = result.to_pandas().to_dict(orient="records")
    metrics: Dict[str, float] = {}
    for key, value in result.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue

    merged = []
    for source, score_row in zip(outputs, rows):
        merged_row = dict(source)
        merged_row.update(score_row)
        merged.append(merged_row)
    return merged, metrics


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
