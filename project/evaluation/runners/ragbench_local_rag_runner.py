import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from db.vector_db_manager import VectorDbManager
from evaluation.data import EvalQuestion
from evaluation.io import config_snapshot, make_run_id, write_jsonl, write_metrics_csv
from evaluation.llm_config import answer_model as resolve_answer_model
from evaluation.llm_config import api_key, base_url
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, run_ragas_metrics
from evaluation.metrics.retrieval_metrics import DEFAULT_K_VALUES, build_retrieval_error_cases, compute_retrieval_metrics
from evaluation.runners.ragbench_importer import import_ragbench
from evaluation.runners.retrieval_eval_runner import retrieve_chunks
from langchain_core.documents import Document


def run_ragbench_local_rag_eval(
    subset: str,
    split: str,
    limit: int,
    output_dir: str,
    run_label: str,
    top_k: int,
    offset: int = 0,
    collection_name: str = "ragbench_eval_child_chunks",
    skip_ragas: bool = False,
    ragas_timeout: int = 180,
    ragas_max_retries: int = 2,
    ragas_max_workers: int = 4,
    ragas_batch_size: int | None = 4,
    generate_answers: bool = True,
    answer_model: str | None = None,
) -> Dict[str, Any]:
    run_id = make_run_id(run_label)
    output = Path(output_dir)
    run_dir = output / "eval_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / f"ragbench_{subset}_{split}_{limit}_eval_questions.jsonl"
    source_contexts_path = run_dir / f"ragbench_{subset}_{split}_{limit}_source_contexts.jsonl"
    import_result = import_ragbench(
        subset=subset,
        split=split,
        limit=limit,
        output_dataset=str(dataset_path),
        output_contexts=str(source_contexts_path),
        offset=offset,
    )
    rows = _read_jsonl(source_contexts_path)
    questions = [_context_row_to_eval_question(row, index + 1) for index, row in enumerate(rows)]

    config.QDRANT_DB_PATH = str(run_dir / "qdrant_db")
    vector_db = VectorDbManager()
    vector_db.delete_collection(collection_name)
    vector_db.create_collection(collection_name)
    collection = vector_db.get_collection(collection_name)
    collection.add_documents(_ragbench_documents(rows))

    retrieval_rows = []
    ragas_outputs = []
    resolved_answer_model = resolve_answer_model(answer_model) if generate_answers else None
    answer_generator = _AnswerGenerator(resolved_answer_model) if generate_answers else None
    for question, row in zip(questions, rows):
        chunks = retrieve_chunks(
            collection=collection,
            query=question.question,
            top_k=top_k,
            score_threshold=None,
            vector_db=vector_db,
            collection_name=collection_name,
        )
        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "query": question.question,
                "retrieved_chunks": chunks,
            }
        )
        contexts = [chunk["text"] for chunk in chunks]
        answer = (
            answer_generator.generate(question.question, contexts)
            if answer_generator
            else row.get("reference_response", "")
        )
        ragas_outputs.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "user_input": question.question,
                "answer": answer,
                "response": answer,
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "reference": row.get("reference_response", ""),
                "ground_truth": row.get("reference_response", ""),
                "answer_source": "generated" if generate_answers else "reference",
                "answer_model": resolved_answer_model,
                "retrieved_metadata": [
                    {
                        "rank": chunk["rank"],
                        "chunk_id": chunk["chunk_id"],
                        "parent_id": chunk["parent_id"],
                        "source_file": chunk["source_file"],
                        "score_fused": chunk["score_fused"],
                        "score_rerank": chunk["score_rerank"],
                    }
                    for chunk in chunks
                ],
            }
        )

    k_values = sorted(set(DEFAULT_K_VALUES + [top_k]))
    retrieval_metrics, per_question = compute_retrieval_metrics(questions, retrieval_rows, k_values=k_values)
    retrieval_error_cases = build_retrieval_error_cases(questions, retrieval_rows, per_question, top_k=top_k)
    metadata = config_snapshot(str(run_id), str(dataset_path), f"ragbench_{subset}_{split}", top_k, None)
    metadata.update(
        {
            "ragbench_subset": subset,
            "ragbench_split": split,
            "ragbench_limit": limit,
            "ragbench_offset": offset,
            "collection_name": collection_name,
            "retrieval_fusion_mode": config.RETRIEVAL_FUSION_MODE,
            "dense_top_k": config.DENSE_TOP_K,
            "sparse_top_k": config.SPARSE_TOP_K,
            "rrf_top_k": config.RRF_TOP_K,
            "rrf_k": config.RRF_K,
            "reranker": config.RERANKER_MODEL if config.RERANKER_ENABLED else None,
            "reranker_top_n": config.RERANKER_TOP_N,
            "reranker_final_top_k": config.RERANKER_FINAL_TOP_K,
            "answer_source": "generated" if generate_answers else "reference",
            "answer_model": resolved_answer_model,
        }
    )

    write_jsonl(run_dir / "retrieval_results.jsonl", retrieval_rows)
    write_jsonl(run_dir / "retrieval_per_question_metrics.jsonl", per_question)
    write_jsonl(run_dir / "retrieval_error_cases.jsonl", retrieval_error_cases)
    write_jsonl(run_dir / "rag_outputs.jsonl", ragas_outputs)
    write_metrics_csv(run_dir / "retrieval_metrics_summary.csv", retrieval_metrics)
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_dir = output / "ragbench_local_rag_latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)

    result: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "import_result": import_result,
        "retrieval_metrics": retrieval_metrics,
    }
    if skip_ragas:
        return result

    ragas_results, ragas_metrics = run_ragas_metrics(
        ragas_outputs,
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_workers=ragas_max_workers,
        batch_size=ragas_batch_size,
    )
    ragas_error_cases = build_ragas_error_cases(ragas_results)
    write_jsonl(run_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(run_dir / "ragas_error_cases.jsonl", ragas_error_cases)
    write_metrics_csv(run_dir / "ragas_metrics_summary.csv", ragas_metrics)
    write_jsonl(latest_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(latest_dir / "ragas_error_cases.jsonl", ragas_error_cases)
    write_metrics_csv(latest_dir / "ragas_metrics_summary.csv", ragas_metrics)
    result["ragas_metrics"] = ragas_metrics
    return result


def run_ragas_from_outputs(
    rag_outputs_path: str,
    output_dir: str,
    ragas_timeout: int = 180,
    ragas_max_retries: int = 2,
    ragas_max_workers: int = 4,
    ragas_batch_size: int | None = 4,
) -> Dict[str, Any]:
    run_dir = Path(rag_outputs_path).resolve().parent
    latest_dir = Path(output_dir) / "ragbench_local_rag_latest"
    ragas_outputs = _read_jsonl(rag_outputs_path)
    ragas_results, ragas_metrics = run_ragas_metrics(
        ragas_outputs,
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
        max_workers=ragas_max_workers,
        batch_size=ragas_batch_size,
    )
    ragas_error_cases = build_ragas_error_cases(ragas_results)
    write_jsonl(run_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(run_dir / "ragas_error_cases.jsonl", ragas_error_cases)
    write_metrics_csv(run_dir / "ragas_metrics_summary.csv", ragas_metrics)
    latest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(latest_dir / "ragas_results.jsonl", ragas_results)
    write_jsonl(latest_dir / "ragas_error_cases.jsonl", ragas_error_cases)
    write_metrics_csv(latest_dir / "ragas_metrics_summary.csv", ragas_metrics)
    return {
        "run_dir": str(run_dir),
        "latest_dir": str(latest_dir),
        "ragas_metrics": ragas_metrics,
    }


class _AnswerGenerator:
    def __init__(self, model: str) -> None:
        self.model = model
        self._openai_client = None
        self._ollama_llm = None

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(question, contexts)
        if self._has_remote_llm_config():
            return self._generate_openai(prompt)
        return self._generate_ollama(prompt)

    def _has_remote_llm_config(self) -> bool:
        try:
            return bool(api_key())
        except RuntimeError:
            return False

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI

        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=api_key(),
                base_url=base_url(),
            )
        response = self._openai_client.chat.completions.create(
            model=self.model,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful RAG answer generator. Answer only from the provided contexts. "
                        "If the contexts do not contain enough evidence, say that the context is insufficient."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _generate_ollama(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        if self._ollama_llm is None:
            self._ollama_llm = ChatOllama(model=self.model, temperature=config.LLM_TEMPERATURE)
        response = self._ollama_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful RAG answer generator. Answer only from the provided contexts. "
                        "If the contexts do not contain enough evidence, say that the context is insufficient."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        return str(response.content).strip()

    @staticmethod
    def _build_prompt(question: str, contexts: List[str]) -> str:
        context_text = "\n\n".join(
            f"[Context {index}]\n{context}" for index, context in enumerate(contexts, start=1)
        )
        return (
            f"Question:\n{question}\n\n"
            f"Retrieved contexts:\n{context_text}\n\n"
            "Answer the question using only the retrieved contexts."
        )


def _ragbench_documents(rows: List[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for row in rows:
        question_id = row["question_id"]
        source_file = f"ragbench/{row['subset']}/{row['split']}/{row['ragbench_id']}"
        for index, text in enumerate(row.get("documents") or []):
            docs.append(
                Document(
                    page_content=str(text),
                    metadata={
                        "chunk_id": f"{question_id}_doc_{index}",
                        "parent_id": f"{question_id}_doc_{index}",
                        "source": source_file,
                    },
                )
            )
    return docs


def _context_row_to_eval_question(row: Dict[str, Any], line_no: int) -> EvalQuestion:
    relevant_sentence_keys = list(row.get("all_relevant_sentence_keys") or [])
    relevant_doc_ids = sorted({key.split(" ")[0][0] for key in relevant_sentence_keys if key})
    raw = {
        "question_id": row["question_id"],
        "question": row.get("question", ""),
        "reference_answer": row.get("reference_response", ""),
        "source_file": f"ragbench/{row['subset']}/{row['split']}/{row['ragbench_id']}",
        "gold_parent_ids": [f"{row['question_id']}_doc_{doc_id}" for doc_id in relevant_doc_ids],
        "gold_child_ids": [],
        "gold_evidence_text": [],
        "question_type": row.get("subset", "ragbench"),
        "difficulty": "unknown",
        "tags": ["ragbench", row.get("subset", ""), row.get("split", "")],
    }
    return EvalQuestion.from_dict(raw, line_no=line_no)


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAGBench through local Qdrant + RRF + reranker.")
    parser.add_argument("--subset", default="covidqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "evaluation" / "reports"))
    parser.add_argument("--run-label", default="ragbench_local_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collection", default="ragbench_eval_child_chunks")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--use-reference-answer", action="store_true", help="Use RAGBench reference answers instead of generated answers.")
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--ragas-input", help="Existing rag_outputs.jsonl to score without rebuilding Qdrant.")
    parser.add_argument("--ragas-timeout", type=int, default=180)
    parser.add_argument("--ragas-max-retries", type=int, default=2)
    parser.add_argument("--ragas-max-workers", type=int, default=4)
    parser.add_argument("--ragas-batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.ragas_input:
        result = run_ragas_from_outputs(
            rag_outputs_path=args.ragas_input,
            output_dir=args.output_dir,
            ragas_timeout=args.ragas_timeout,
            ragas_max_retries=args.ragas_max_retries,
            ragas_max_workers=args.ragas_max_workers,
            ragas_batch_size=args.ragas_batch_size,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = run_ragbench_local_rag_eval(
        subset=args.subset,
        split=args.split,
        limit=args.limit,
        output_dir=args.output_dir,
        run_label=args.run_label,
        top_k=args.top_k,
        offset=args.offset,
        collection_name=args.collection,
        skip_ragas=args.skip_ragas,
        ragas_timeout=args.ragas_timeout,
        ragas_max_retries=args.ragas_max_retries,
        ragas_max_workers=args.ragas_max_workers,
        ragas_batch_size=args.ragas_batch_size,
        generate_answers=not args.use_reference_answer,
        answer_model=args.answer_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
