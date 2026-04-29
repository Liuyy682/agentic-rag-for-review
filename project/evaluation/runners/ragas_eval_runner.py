import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from core.rag_system import RAGSystem
from evaluation.data import load_eval_questions
from evaluation.io import config_snapshot, make_run_id, write_jsonl, write_metrics_csv
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.reports import write_ragas_report
from evaluation.runners.retrieval_eval_runner import retrieve_chunks
from langchain_core.messages import AIMessage, HumanMessage


def run_ragas_eval(
    dataset_path: str,
    output_dir: str,
    run_label: str,
    top_k: int,
    collection_name: str,
    dataset_version: str,
    score_threshold: float | None,
    skip_ragas: bool = False,
    ragas_timeout: int = 180,
    ragas_max_retries: int = 2,
    ragas_max_workers: int = 2,
    ragas_batch_size: int | None = 1,
) -> Dict[str, Any]:
    questions = load_eval_questions(dataset_path)
    run_id = make_run_id(run_label)
    run_dir = Path(output_dir) / "eval_runs" / run_id
    reports_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rag_system = RAGSystem(collection_name=collection_name)
    rag_system.initialize()
    collection = rag_system.vector_db.get_collection(collection_name)

    outputs: List[Dict[str, Any]] = []
    for item in questions:
        rag_system.reset_thread()
        retrieved = retrieve_chunks(
            collection,
            item.question,
            top_k,
            score_threshold,
            vector_db=rag_system.vector_db,
            collection_name=collection_name,
        )
        answer = invoke_rag_answer(rag_system, item.question)
        outputs.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "user_input": item.question,
                "answer": answer,
                "response": answer,
                "contexts": [chunk["text"] for chunk in retrieved],
                "retrieved_contexts": [chunk["text"] for chunk in retrieved],
                "reference": item.reference_answer,
                "ground_truth": item.reference_answer,
                "retrieved_metadata": [
                    {
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "parent_id": chunk["parent_id"],
                        "source_file": chunk["source_file"],
                    }
                    for chunk in retrieved
                ],
            }
        )

    metadata = config_snapshot(run_id, dataset_path, dataset_version, top_k, score_threshold)
    write_jsonl(run_dir / "rag_outputs.jsonl", outputs)
    write_jsonl(reports_dir / "rag_outputs.jsonl", outputs)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if skip_ragas:
        return {"run_id": run_id, "run_dir": str(run_dir), "rag_outputs": str(run_dir / "rag_outputs.jsonl")}

    ragas_results, metrics = run_ragas_metrics(
        outputs,
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_workers=ragas_max_workers,
        batch_size=ragas_batch_size,
    )
    error_cases = build_ragas_error_cases(ragas_results)
    write_jsonl(run_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(run_dir / "ragas_error_cases.jsonl", error_cases)
    write_metrics_csv(run_dir / "ragas_metrics_summary.csv", metrics)
    write_jsonl(reports_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(reports_dir / "ragas_error_cases.jsonl", error_cases)
    write_metrics_csv(reports_dir / "ragas_metrics_summary.csv", metrics)
    write_ragas_report(reports_dir / "ragas_report.md", metadata, metrics, error_cases)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(reports_dir / "ragas_report.md"),
        "metrics": metrics,
    }


def invoke_rag_answer(rag_system: RAGSystem, question: str) -> str:
    state = rag_system.agent_graph.invoke(
        {"messages": [HumanMessage(content=question.strip())]},
        config=rag_system.get_config(),
    )
    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end RAG output capture and optional RAGAS evaluation.")
    parser.add_argument("--dataset", default=str(PROJECT_DIR / "evaluation" / "datasets" / "eval_questions.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "evaluation" / "reports"))
    parser.add_argument("--run-label", default="baseline_ragas")
    parser.add_argument("--dataset-version", default="eval_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--collection", default=config.CHILD_COLLECTION)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=2)
    parser.add_argument("--ragas-max-workers", type=int, default=2)
    parser.add_argument("--ragas-batch-size", type=int, default=1)
    args = parser.parse_args()

    result = run_ragas_eval(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        run_label=args.run_label,
        top_k=args.top_k,
        collection_name=args.collection,
        dataset_version=args.dataset_version,
        score_threshold=args.score_threshold,
        skip_ragas=args.skip_ragas,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
