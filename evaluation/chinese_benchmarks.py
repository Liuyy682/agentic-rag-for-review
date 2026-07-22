from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from evaluation.runners.chunking_ablation import (
    T2_RETRIEVAL_QRELS_REPO,
    T2_RETRIEVAL_REPO,
    build_t2_retrieval_rows,
    read_hf_parquet_records,
)


CMEDQA_RETRIEVAL_REPO = "C-MTEB/CmedqaRetrieval"
CMEDQA_RETRIEVAL_QRELS_REPO = "C-MTEB/CmedqaRetrieval-qrels"


@dataclass(frozen=True)
class BenchmarkDocument:
    doc_id: str
    text: str


@dataclass(frozen=True)
class BenchmarkQuestion:
    question_id: str
    question: str
    reference: str
    gold_doc_ids: tuple[str, ...]
    task: str


def load_t2_benchmark(
    limit: int,
    offset: int,
    distractor_docs: int,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkDocument]]:
    corpus = read_hf_parquet_records(T2_RETRIEVAL_REPO, "corpus")
    queries = read_hf_parquet_records(T2_RETRIEVAL_REPO, "queries")
    qrels = read_hf_parquet_records(T2_RETRIEVAL_QRELS_REPO, "dev")
    rows, source_docs = build_t2_retrieval_rows(
        corpus, queries, qrels, limit=limit, offset=offset, distractor_docs=distractor_docs
    )
    questions = [
        BenchmarkQuestion(
            question_id=row["question_id"],
            question=row["question"],
            reference="",
            gold_doc_ids=tuple(doc_id.removeprefix("t2_doc_") for doc_id in row["gold_source_doc_ids"]),
            task="t2_retrieval",
        )
        for row in rows
    ]
    documents = [
        BenchmarkDocument(doc_id=doc.doc_id.removeprefix("t2_doc_"), text=doc.text)
        for doc in source_docs
    ]
    return questions, documents


def load_cmedqa_benchmark(
    limit: int,
    offset: int,
    distractor_docs: int,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkDocument]]:
    corpus_rows = read_hf_parquet_records(CMEDQA_RETRIEVAL_REPO, "corpus")
    query_rows = read_hf_parquet_records(CMEDQA_RETRIEVAL_REPO, "queries")
    qrels = read_hf_parquet_records(CMEDQA_RETRIEVAL_QRELS_REPO, "dev")
    corpus = {_record_id(row): _document_text(row) for row in corpus_rows}
    if not corpus or any(not text for text in corpus.values()):
        raise ValueError("CmedQA corpus contains missing IDs or empty documents.")
    positives: dict[str, list[tuple[str, float]]] = {}
    for row in qrels:
        score = _score(row)
        if score <= 0:
            continue
        query_id, document_id = _qrel_ids(row)
        if document_id not in corpus:
            raise ValueError(f"CmedQA qrels references missing corpus document {document_id!r}.")
        positives.setdefault(query_id, []).append((document_id, score))
    ordered_queries = []
    for row in query_rows:
        query_id = _record_id(row)
        question = _text(row, ("text", "query", "question"))
        if query_id and question and query_id in positives:
            ordered_queries.append((query_id, question))
    selected = ordered_queries[offset : offset + limit]
    if len(selected) < limit:
        raise ValueError(f"Only found {len(selected)} CmedQA rows at offset {offset}; requested {limit}.")

    questions = []
    all_gold: set[str] = set()
    for query_id, question in selected:
        ranked_gold = sorted(positives[query_id], key=lambda item: (-item[1], item[0]))
        gold_ids = tuple(document_id for document_id, _ in ranked_gold)
        all_gold.update(gold_ids)
        questions.append(
            BenchmarkQuestion(
                question_id=f"cmedqa_{query_id}",
                question=question,
                reference=corpus[gold_ids[0]],
                gold_doc_ids=gold_ids,
                task="cmedqa_retrieval",
            )
        )
    selected_ids = select_candidate_doc_ids(corpus, all_gold, distractor_docs)
    documents = [BenchmarkDocument(doc_id=document_id, text=corpus[document_id]) for document_id in sorted(selected_ids)]
    return questions, documents


def select_candidate_doc_ids(
    documents: dict[str, str], gold_doc_ids: set[str], distractor_docs: int
) -> set[str]:
    if distractor_docs <= 0:
        return set(documents)
    selected = set(gold_doc_ids)
    for document_id in sorted(documents):
        if document_id not in selected:
            selected.add(document_id)
        if len(selected) >= len(gold_doc_ids) + distractor_docs:
            break
    return selected


def _record_id(row: dict[str, Any]) -> str:
    return _text(row, ("_id", "id", "query_id", "qid"))


def _qrel_ids(row: dict[str, Any]) -> tuple[str, str]:
    query_id = _text(row, ("qid", "query_id", "query-id"))
    document_id = _text(row, ("pid", "corpus_id", "corpus-id", "doc_id"))
    if not query_id or not document_id:
        raise ValueError("CmedQA qrels row is missing query or corpus ID.")
    return query_id, document_id


def _score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score", row.get("relevance", 0)) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("CmedQA qrels score must be numeric.") from exc


def _document_text(row: dict[str, Any]) -> str:
    title = _text(row, ("title",))
    text = _text(row, ("text", "content", "document"))
    if title and text:
        return f"# {title}\n\n{text}"
    return text or title


def _text(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""
