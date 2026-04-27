import math
from typing import Any, Dict, Iterable, List, Sequence, Set

from evaluation.data import EvalQuestion


DEFAULT_K_VALUES = [1, 3, 5, 10, 20]


def compute_retrieval_metrics(
    questions: Sequence[EvalQuestion],
    result_rows: Sequence[Dict[str, Any]],
    k_values: Iterable[int] = DEFAULT_K_VALUES,
) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    by_id = {row["question_id"]: row for row in result_rows}
    per_question: List[Dict[str, Any]] = []

    for question in questions:
        row = by_id.get(question.question_id, {"retrieved_chunks": []})
        per_question.append(_score_question(question, row.get("retrieved_chunks", []), k_values))

    aggregate = _aggregate(per_question, k_values)
    return aggregate, per_question


def _score_question(
    question: EvalQuestion,
    retrieved_chunks: Sequence[Dict[str, Any]],
    k_values: Iterable[int],
) -> Dict[str, Any]:
    child_gold = set(question.gold_child_ids)
    parent_gold = set(question.gold_parent_ids)
    source_gold = set(question.gold_source_files)
    relevant_gold = child_gold or parent_gold

    row: Dict[str, Any] = {
        "question_id": question.question_id,
        "question": question.question,
        "question_type": question.question_type,
        "mrr": _reciprocal_rank(retrieved_chunks, child_gold, parent_gold),
        "first_relevant_rank": _first_relevant_rank(retrieved_chunks, child_gold, parent_gold),
    }

    for k in k_values:
        top = list(retrieved_chunks[:k])
        child_hits = _hit_ids(top, "chunk_id", child_gold)
        parent_hits = _hit_ids(top, "parent_id", parent_gold)
        source_hits = _hit_ids(top, "source_file", source_gold)
        relevant_hits = child_hits if child_gold else parent_hits

        row[f"child_recall@{k}"] = _ratio(len(child_hits), len(child_gold))
        row[f"parent_recall@{k}"] = _ratio(len(parent_hits), len(parent_gold))
        row[f"source_hitrate@{k}"] = 1.0 if source_hits else 0.0
        row[f"recall@{k}"] = _ratio(len(relevant_hits), len(relevant_gold))
        row[f"hitrate@{k}"] = 1.0 if relevant_hits else 0.0
        row[f"precision@{k}"] = len(relevant_hits) / k if k else 0.0
        row[f"ndcg@{k}"] = _ndcg(top, child_gold, parent_gold, source_gold, k)

    return row


def _aggregate(per_question: Sequence[Dict[str, Any]], k_values: Iterable[int]) -> Dict[str, float]:
    metrics = ["mrr"]
    for k in k_values:
        metrics.extend(
            [
                f"recall@{k}",
                f"hitrate@{k}",
                f"precision@{k}",
                f"child_recall@{k}",
                f"parent_recall@{k}",
                f"source_hitrate@{k}",
                f"ndcg@{k}",
            ]
        )
    return {metric: _mean(row.get(metric, 0.0) for row in per_question) for metric in metrics}


def build_retrieval_error_cases(
    questions: Sequence[EvalQuestion],
    result_rows: Sequence[Dict[str, Any]],
    per_question: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    results_by_id = {row["question_id"]: row for row in result_rows}
    scores_by_id = {row["question_id"]: row for row in per_question}
    cases: List[Dict[str, Any]] = []

    for question in questions:
        scores = scores_by_id.get(question.question_id, {})
        result = results_by_id.get(question.question_id, {"retrieved_chunks": []})
        retrieved = result.get("retrieved_chunks", [])[:top_k]
        failure_type = None

        if question.source_file and scores.get(f"source_hitrate@{top_k}", 0.0) == 0.0:
            failure_type = "missed_source"
        elif question.gold_parent_ids and scores.get(f"parent_recall@{top_k}", 0.0) == 0.0:
            failure_type = "missed_parent"
        elif question.gold_child_ids and scores.get(f"child_recall@{top_k}", 0.0) == 0.0:
            failure_type = "missed_child"
        elif scores.get("first_relevant_rank") and scores["first_relevant_rank"] > 3:
            failure_type = "bad_ranking"

        if failure_type:
            cases.append(
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "expected_parent_ids": question.gold_parent_ids,
                    "expected_child_ids": question.gold_child_ids,
                    "expected_source_files": question.gold_source_files,
                    "retrieved_parent_ids": [item.get("parent_id") for item in retrieved],
                    "retrieved_child_ids": [item.get("chunk_id") for item in retrieved],
                    "retrieved_source_files": [item.get("source_file") for item in retrieved],
                    "failure_type": failure_type,
                    "first_relevant_rank": scores.get("first_relevant_rank"),
                    "notes": f"No sufficient hit in top {top_k}.",
                }
            )

    return cases


def _hit_ids(chunks: Sequence[Dict[str, Any]], field: str, gold: Set[str]) -> Set[str]:
    if not gold:
        return set()
    return {str(item.get(field)) for item in chunks if str(item.get(field)) in gold}


def _is_relevant(chunk: Dict[str, Any], child_gold: Set[str], parent_gold: Set[str]) -> bool:
    if child_gold:
        return str(chunk.get("chunk_id")) in child_gold
    return bool(parent_gold and str(chunk.get("parent_id")) in parent_gold)


def _first_relevant_rank(
    chunks: Sequence[Dict[str, Any]],
    child_gold: Set[str],
    parent_gold: Set[str],
) -> int | None:
    for index, chunk in enumerate(chunks, start=1):
        if _is_relevant(chunk, child_gold, parent_gold):
            return index
    return None


def _reciprocal_rank(
    chunks: Sequence[Dict[str, Any]],
    child_gold: Set[str],
    parent_gold: Set[str],
) -> float:
    rank = _first_relevant_rank(chunks, child_gold, parent_gold)
    return 1 / rank if rank else 0.0


def _ndcg(
    chunks: Sequence[Dict[str, Any]],
    child_gold: Set[str],
    parent_gold: Set[str],
    source_gold: Set[str],
    k: int,
) -> float:
    gains = [_relevance_grade(chunk, child_gold, parent_gold, source_gold) for chunk in chunks[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_gains = _ideal_gains(child_gold, parent_gold, source_gold, k)
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg else 0.0


def _ideal_gains(child_gold: Set[str], parent_gold: Set[str], source_gold: Set[str], k: int) -> List[int]:
    if child_gold:
        return [3] * min(len(child_gold), k)
    if parent_gold:
        return [2] * min(len(parent_gold), k)
    if source_gold:
        return [1] * min(len(source_gold), k)
    return []


def _relevance_grade(
    chunk: Dict[str, Any],
    child_gold: Set[str],
    parent_gold: Set[str],
    source_gold: Set[str],
) -> int:
    if child_gold and str(chunk.get("chunk_id")) in child_gold:
        return 3
    if parent_gold and str(chunk.get("parent_id")) in parent_gold:
        return 2
    if source_gold and str(chunk.get("source_file")) in source_gold:
        return 1
    return 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0
