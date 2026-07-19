from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from evaluation.runners.chunking_ablation import (
    build_t2_retrieval_rows,
    read_hf_parquet_records,
    T2_RETRIEVAL_QRELS_REPO,
    T2_RETRIEVAL_REPO,
)


CRUD_QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")


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
        corpus,
        queries,
        qrels,
        limit=limit,
        offset=offset,
        distractor_docs=distractor_docs,
    )
    questions = [
        BenchmarkQuestion(
            question_id=row["question_id"],
            question=row["question"],
            reference="",
            gold_doc_ids=tuple(
                doc_id.removeprefix("t2_doc_") for doc_id in row["gold_source_doc_ids"]
            ),
            task="t2_retrieval",
        )
        for row in rows
    ]
    documents = [
        BenchmarkDocument(doc_id=doc.doc_id.removeprefix("t2_doc_"), text=doc.text)
        for doc in source_docs
    ]
    return questions, documents


def load_crud_benchmark(
    crud_root: str | Path,
    limit: int,
    offset: int,
    distractor_docs: int,
) -> tuple[list[BenchmarkQuestion], list[BenchmarkDocument]]:
    root = Path(crud_root)
    split_path = _find_crud_split(root)
    docs_path = _find_crud_docs(root)
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    all_questions = interleave_crud_questions(build_crud_questions(raw))
    selected = all_questions[offset : offset + limit]
    if len(selected) < limit:
        raise ValueError(
            f"Only found {len(selected)} CRUD-RAG QA rows at offset {offset}; requested {limit}."
        )

    document_map = load_crud_documents(docs_path)
    missing = sorted(
        {
            doc_id
            for question in selected
            for doc_id in question.gold_doc_ids
            if doc_id not in document_map
        }
    )
    if missing:
        raise ValueError(
            "CRUD-RAG QA rows reference documents that are absent from 80000_docs: "
            + ", ".join(missing[:10])
        )

    selected_doc_ids = select_candidate_doc_ids(
        document_map,
        {doc_id for question in selected for doc_id in question.gold_doc_ids},
        distractor_docs,
    )
    documents = [
        BenchmarkDocument(doc_id=doc_id, text=document_map[doc_id])
        for doc_id in sorted(selected_doc_ids)
    ]
    return selected, documents


def build_crud_questions(raw: dict[str, Any]) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    seen_question_ids: set[str] = set()
    for task in CRUD_QA_TASKS:
        rows = raw.get(task)
        if not isinstance(rows, list):
            raise ValueError(f"CRUD-RAG split is missing list task {task!r}.")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"CRUD-RAG {task}[{index}] must be an object.")
            question = _required_text(row.get("questions"), task, index, "questions")
            reference = _required_text(row.get("answers"), task, index, "answers")
            doc_ids = tuple(_normalize_ids(row.get("IDs", row.get("ID"))))
            if not doc_ids:
                raise ValueError(f"CRUD-RAG {task}[{index}] has no ID/IDs gold documents.")
            if len(set(doc_ids)) != len(doc_ids):
                raise ValueError(f"CRUD-RAG {task}[{index}] contains duplicate document IDs.")
            expected_docs = int(task.removeprefix("questanswer_").removesuffix("docs").removesuffix("doc"))
            if len(doc_ids) != expected_docs:
                raise ValueError(
                    f"CRUD-RAG {task}[{index}] expected {expected_docs} gold documents, got {len(doc_ids)}."
                )
            raw_id = str(row.get("question_id") or row.get("qid") or f"{task}_{index}").strip()
            question_id = f"crud_{raw_id}"
            if question_id in seen_question_ids:
                raise ValueError(f"Duplicate CRUD-RAG question ID: {question_id}")
            seen_question_ids.add(question_id)
            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    question=question,
                    reference=reference,
                    gold_doc_ids=doc_ids,
                    task=task,
                )
            )
    return questions


def interleave_crud_questions(
    questions: Sequence[BenchmarkQuestion],
) -> list[BenchmarkQuestion]:
    groups = {
        task: [question for question in questions if question.task == task]
        for task in CRUD_QA_TASKS
    }
    ordered: list[BenchmarkQuestion] = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for task in CRUD_QA_TASKS:
            if index < len(groups[task]):
                ordered.append(groups[task][index])
    return ordered


def load_crud_documents(path: str | Path) -> dict[str, str]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"CRUD-RAG document directory not found: {root}")
    documents: dict[str, str] = {}
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            records = _read_document_records(file_path)
            for record in records:
                doc_id = _first_text(record, ("ID", "id", "_id", "doc_id"))
                text = _first_text(record, ("text", "content", "document"))
                title = _first_text(record, ("title",))
                if doc_id and (text or title):
                    _add_document(documents, doc_id, _join_title(title, text), file_path)
        elif suffix == ".zip":
            with zipfile.ZipFile(file_path) as archive:
                for member in sorted(name for name in archive.namelist() if not name.endswith("/")):
                    member_suffix = Path(member).suffix.lower()
                    content = archive.read(member).decode("utf-8")
                    if member_suffix in {".json", ".jsonl"}:
                        records = _records_from_text(content, member_suffix)
                        for record in records:
                            doc_id = _first_text(record, ("ID", "id", "_id", "doc_id"))
                            text = _first_text(record, ("text", "content", "document"))
                            title = _first_text(record, ("title",))
                            if doc_id and (text or title):
                                _add_document(documents, doc_id, _join_title(title, text), file_path)
                    elif member_suffix in {".txt", ".md"} and content.strip():
                        _add_document(documents, Path(member).stem, content.strip(), file_path)
        elif suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                _add_document(documents, file_path.stem, text, file_path)
    if not documents:
        raise ValueError(f"No CRUD-RAG documents could be loaded from {root}.")
    return documents


def select_candidate_doc_ids(
    documents: dict[str, str],
    gold_doc_ids: set[str],
    distractor_docs: int,
) -> set[str]:
    if distractor_docs <= 0:
        return set(documents)
    selected = set(gold_doc_ids)
    for doc_id in sorted(documents):
        if doc_id in selected:
            continue
        selected.add(doc_id)
        if len(selected) >= len(gold_doc_ids) + distractor_docs:
            break
    return selected


def _find_crud_split(root: Path) -> Path:
    candidates = (
        root / "data" / "crud_split" / "split_merged.json",
        root / "crud_split" / "split_merged.json",
        root / "split_merged.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"CRUD-RAG split_merged.json not found below {root}.")


def _find_crud_docs(root: Path) -> Path:
    candidates = (root / "data" / "80000_docs", root / "80000_docs")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"CRUD-RAG 80000_docs not found below {root}.")


def _read_document_records(path: Path) -> list[dict[str, Any]]:
    return _records_from_text(path.read_text(encoding="utf-8"), path.suffix.lower())


def _records_from_text(content: str, suffix: str) -> list[dict[str, Any]]:
    if suffix == ".jsonl":
        values = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        values = [json.loads(content)]
    return list(_walk_records(values))


def _walk_records(values: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for value in values:
        if isinstance(value, list):
            yield from _walk_records(value)
        elif isinstance(value, dict):
            if _first_text(value, ("ID", "id", "_id", "doc_id")) and _first_text(
                value, ("text", "content", "document")
            ):
                yield value
            else:
                yield from _walk_records(value.values())


def _normalize_ids(value: Any) -> list[str]:
    values: Sequence[Any] = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def _required_text(value: Any, task: str, index: int, field: str) -> str:
    if isinstance(value, (list, tuple)):
        text = "\n".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value or "").strip()
    if not text:
        raise ValueError(f"CRUD-RAG {task}[{index}] has empty {field!r}.")
    return text


def _first_text(record: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _join_title(title: str, text: str) -> str:
    if title and text:
        return f"# {title}\n\n{text}"
    return text or title


def _add_document(documents: dict[str, str], doc_id: str, text: str, source: Path) -> None:
    if doc_id in documents:
        raise ValueError(f"Duplicate CRUD-RAG document ID {doc_id!r} while reading {source}.")
    documents[doc_id] = text
