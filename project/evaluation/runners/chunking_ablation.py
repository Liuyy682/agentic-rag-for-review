import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import config
from evaluation.io import write_jsonl, write_metrics_csv
from evaluation.ragbench_keys import document_id_from_sentence_key
from evaluation.validation import build_validity_summary, make_warning, validation_markdown_section, write_validation_outputs


DEFAULT_VARIANTS = [
    "single_300_60",
    "single_500_100",
    "single_800_160",
    "single_1200_240",
    "pc_150_40_800_160",
    "pc_300_60_800_160",
    "pc_500_100_1200_240",
    "pc_500_100_2000_400",
    "pc_800_160_2000_400",
    "pc_child",
    "pc_neighbor",
    "pc_parent",
    "pc_adaptive",
]

POLICY_VARIANT_CHILD_SIZE = 800
POLICY_VARIANT_CHILD_OVERLAP = 160
POLICY_VARIANT_PARENT_SIZE = 2000
POLICY_VARIANT_PARENT_OVERLAP = 400


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    text: str
    sentence_keys: tuple[str, ...]
    sentence_texts: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_id: str
    source_doc_id: str
    text: str
    sentence_keys: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    name: str
    strategy: str
    child_size: int
    child_overlap: int
    parent_size: int | None = None
    parent_overlap: int = 0
    context_policy: str = "parent"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a controlled parent-child vs single-granularity chunking ablation."
    )
    parser.add_argument(
        "--dataset",
        choices=("ragbench", "dapr_conditionalqa"),
        default="ragbench",
        help="Evaluation source. Use dapr_conditionalqa for UKPLab/dapr ConditionalQA.",
    )
    parser.add_argument(
        "--source-contexts",
        default=(
            "runtime/evaluation_reports/eval_runs/"
            "run_2026_05_11_115156_covidqa200_rrf_rerank/"
            "ragbench_covidqa_test_200_source_contexts.jsonl"
        ),
    )
    parser.add_argument("--output-dir", default="runtime/evaluation_reports/chunking_ablation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--sparse-top-k", type=int, default=50)
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument(
        "--dapr-distractor-docs",
        type=int,
        default=120,
        help="For DAPR, index all selected-query gold docs plus this many deterministic distractor docs. Use 0 for all docs.",
    )
    args = parser.parse_args()

    if args.dataset == "ragbench":
        rows = read_jsonl(args.source_contexts)[: args.limit]
        source_docs = build_source_docs(rows)
        dataset_label = args.source_contexts
    else:
        rows, source_docs = load_dapr_conditionalqa(args.limit, args.dapr_distractor_docs)
        dataset_label = (
            "UKPLab/dapr: ConditionalQA test, "
            f"gold docs + {args.dapr_distractor_docs or 'all'} distractor docs"
        )

    k_values = sorted({int(item) for item in args.k_values.split(",") if item.strip()})
    variants = [parse_variant(item.strip()) for item in args.variants.split(",") if item.strip()]

    output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_chunking_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_warnings = []
    if not rows:
        validation_warnings.append(
            make_warning(
                "empty_dataset",
                "Chunking ablation received no rows; no metric values are usable for conclusions.",
                severity="error",
            )
        )
        validity_summary = build_validity_summary(
            rows=0,
            warnings=validation_warnings,
            evaluation_type="chunking_ablation",
        )
        write_validation_outputs(output_dir, validation_warnings, validity_summary)
        write_jsonl(output_dir / "summary.jsonl", [])
        write_summary_csv(output_dir / "summary.csv", [])
        (output_dir / "report.md").write_text(
            "\n".join(
                ["# Chunking Ablation Report", "", "No rows were available for evaluation."]
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
        evaluation_type="chunking_ablation",
    )
    write_validation_outputs(output_dir, validation_warnings, validity_summary)

    questions = [row["question"] for row in rows]
    model = SentenceTransformer(
        config.DENSE_MODEL,
        cache_folder=getattr(config, "HF_CACHE_DIR", None),
        local_files_only=True,
    )
    query_embeddings = normalize(model.encode(questions, batch_size=32, show_progress_bar=True))

    summaries = []
    for variant in variants:
        print(f"Running {variant.name}...", flush=True)
        chunks, parents = build_variant_chunks(source_docs, variant)
        metrics, per_question = evaluate_variant(
            rows=rows,
            variant=variant,
            chunks=chunks,
            parents=parents,
            query_embeddings=query_embeddings,
            model=model,
            dense_top_k=args.dense_top_k,
            sparse_top_k=args.sparse_top_k,
            k_values=k_values,
        )
        summary = {"variant": variant.name, "strategy": variant.strategy, **metrics}
        summaries.append(summary)
        write_metrics_csv(output_dir / variant.name / "metrics.csv", metrics)
        write_jsonl(output_dir / variant.name / "per_question.jsonl", per_question)

    summaries = add_balanced_scores(summaries)
    write_jsonl(output_dir / "summary.jsonl", summaries)
    write_summary_csv(output_dir / "summary.csv", summaries)
    write_report(
        output_dir / "report.md",
        summaries,
        rows,
        source_docs,
        variants,
        args,
        dataset_label,
        validation_warnings,
        validity_summary,
    )
    print(output_dir)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_variant(raw: str) -> Variant:
    parts = raw.split("_")
    if parts[0] == "single" and len(parts) == 3:
        return Variant(
            name=raw,
            strategy="single",
            child_size=int(parts[1]),
            child_overlap=int(parts[2]),
        )
    if raw in {"pc_child", "pc_neighbor", "pc_parent", "pc_adaptive"}:
        return Variant(
            name=raw,
            strategy="parent_child",
            child_size=POLICY_VARIANT_CHILD_SIZE,
            child_overlap=POLICY_VARIANT_CHILD_OVERLAP,
            parent_size=POLICY_VARIANT_PARENT_SIZE,
            parent_overlap=POLICY_VARIANT_PARENT_OVERLAP,
            context_policy=raw.removeprefix("pc_"),
        )
    if parts[0] == "pc" and len(parts) == 5:
        return Variant(
            name=raw,
            strategy="parent_child",
            child_size=int(parts[1]),
            child_overlap=int(parts[2]),
            parent_size=int(parts[3]),
            parent_overlap=int(parts[4]),
        )
    raise ValueError(f"Unsupported variant: {raw}")


def build_source_docs(rows: list[dict]) -> list[SourceDoc]:
    docs = []
    for row in rows:
        question_id = row["question_id"]
        for doc_index, text in enumerate(row.get("documents") or []):
            sentence_pairs = row.get("documents_sentences", [])[doc_index]
            docs.append(
                SourceDoc(
                    doc_id=f"{question_id}_doc_{doc_index}",
                    text=str(text),
                    sentence_keys=tuple(str(item[0]) for item in sentence_pairs),
                    sentence_texts=tuple(str(item[1]) for item in sentence_pairs),
                )
            )
    return docs


def load_dapr_conditionalqa(limit: int, distractor_docs: int) -> tuple[list[dict], list[SourceDoc]]:
    corpus = read_hf_parquet("ConditionalQA/corpus/0000.parquet")
    queries = read_hf_parquet("ConditionalQA/queries/test.parquet")
    qrels = read_hf_parquet("ConditionalQA/qrels/test.parquet")

    corpus["_id"] = corpus["_id"].astype(str)
    corpus["doc_id"] = corpus["doc_id"].astype(str)
    corpus["text"] = corpus["text"].fillna("").astype(str)
    corpus["title"] = corpus["title"].fillna("").astype(str)
    queries["_id"] = queries["_id"].astype(str)
    qrels["query_id"] = qrels["query_id"].astype(str)
    qrels["corpus_id"] = qrels["corpus_id"].astype(str)
    qrels = qrels[qrels["score"].astype(float) > 0]

    passage_to_doc = dict(zip(corpus["_id"], corpus["doc_id"]))
    qrels_by_query: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels.itertuples(index=False):
        if qrel.corpus_id in passage_to_doc:
            qrels_by_query[qrel.query_id].append(qrel.corpus_id)

    selected_queries = []
    for query in queries.to_dict("records"):
        query_id = str(query["_id"])
        if query_id not in qrels_by_query:
            continue
        selected_queries.append(query)
        if len(selected_queries) >= limit:
            break

    if len(selected_queries) < limit:
        raise ValueError(f"Only found {len(selected_queries)} ConditionalQA queries with qrels, requested {limit}.")

    rows = []
    selected_gold_doc_ids = set()
    for query in selected_queries:
        query_id = str(query["_id"])
        gold_passage_ids = sorted(set(qrels_by_query[query_id]))
        selected_gold_doc_ids.update(passage_to_doc[item] for item in gold_passage_ids)
        rows.append(
            {
                "question_id": f"dapr_{query_id}",
                "question": str(query["text"]),
                "gold_sentence_keys": gold_passage_ids,
                "gold_source_doc_ids": sorted({f"dapr_doc_{passage_to_doc[item]}" for item in gold_passage_ids}),
            }
        )

    selected_doc_ids = select_dapr_doc_ids(corpus, selected_gold_doc_ids, distractor_docs)
    return rows, build_dapr_source_docs(corpus[corpus["doc_id"].isin(selected_doc_ids)])


def select_dapr_doc_ids(corpus: pd.DataFrame, gold_doc_ids: set[str], distractor_docs: int) -> set[str]:
    if distractor_docs <= 0:
        return set(corpus["doc_id"].unique())

    selected = set(gold_doc_ids)
    for doc_id in sorted(str(item) for item in corpus["doc_id"].unique()):
        if doc_id in selected:
            continue
        selected.add(doc_id)
        if len(selected) >= len(gold_doc_ids) + distractor_docs:
            break
    return selected


def read_hf_parquet(filename: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="UKPLab/dapr",
        filename=filename,
        repo_type="dataset",
        cache_dir=getattr(config, "HF_CACHE_DIR", None),
    )
    return pd.read_parquet(path)


def build_dapr_source_docs(corpus: pd.DataFrame) -> list[SourceDoc]:
    docs = []
    corpus = corpus.sort_values(["doc_id", "paragraph_no", "_id"])
    for doc_id, group in corpus.groupby("doc_id", sort=True):
        title = str(group["title"].iloc[0]) if len(group) else ""
        paragraphs = [str(text) for text in group["text"].tolist() if str(text).strip()]
        text = f"# {title}\n\n" + "\n\n".join(paragraphs)
        docs.append(
            SourceDoc(
                doc_id=f"dapr_doc_{doc_id}",
                text=text,
                sentence_keys=tuple(str(item) for item in group["_id"].tolist()),
                sentence_texts=tuple(str(item) for item in group["text"].tolist()),
            )
        )
    return docs


def build_variant_chunks(source_docs: list[SourceDoc], variant: Variant) -> tuple[list[Chunk], dict[str, Chunk]]:
    chunks: list[Chunk] = []
    parents: dict[str, Chunk] = {}

    for doc in source_docs:
        if variant.strategy == "single":
            for chunk_index, (text, keys) in enumerate(split_source_doc(doc, variant.child_size, variant.child_overlap)):
                chunk_id = f"{doc.doc_id}_single_{chunk_index}"
                chunk = Chunk(
                    chunk_id=chunk_id,
                    parent_id=chunk_id,
                    source_doc_id=doc.doc_id,
                    text=text,
                    sentence_keys=keys,
                )
                chunks.append(chunk)
                parents[chunk.parent_id] = chunk
            continue

        parent_size = variant.parent_size or len(doc.text)
        for parent_index, (parent_text, parent_keys) in enumerate(
            split_source_doc(doc, parent_size, variant.parent_overlap)
        ):
            parent_id = f"{doc.doc_id}_parent_{parent_index}"
            parent = Chunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                source_doc_id=doc.doc_id,
                text=parent_text,
                sentence_keys=parent_keys,
            )
            parents[parent_id] = parent
            parent_doc = SourceDoc(
                doc_id=doc.doc_id,
                text=parent_text,
                sentence_keys=parent_keys,
                sentence_texts=tuple(
                    sentence
                    for key, sentence in zip(doc.sentence_keys, doc.sentence_texts)
                    if key in set(parent_keys)
                ),
            )
            for child_index, (child_text, child_keys) in enumerate(
                split_source_doc(parent_doc, variant.child_size, variant.child_overlap)
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}_child_{child_index}",
                        parent_id=parent_id,
                        source_doc_id=doc.doc_id,
                        text=child_text,
                        sentence_keys=child_keys,
                    )
                )
    return chunks, parents


def split_source_doc(doc: SourceDoc, size: int, overlap: int) -> list[tuple[str, tuple[str, ...]]]:
    spans = split_spans(doc.text, size, overlap)
    sentence_spans = sentence_offsets(doc)
    result = []
    for start, end in spans:
        keys = tuple(
            key
            for key, sentence_start, sentence_end in sentence_spans
            if sentence_start < end and sentence_end > start
        )
        result.append((doc.text[start:end], keys))
    return result


def split_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    if len(text) <= size:
        return [(0, len(text))]
    step = max(1, size - overlap)
    spans = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        spans.append((start, end))
        if end == len(text):
            break
        start += step
    return spans


def sentence_offsets(doc: SourceDoc) -> list[tuple[str, int, int]]:
    offsets = []
    cursor = 0
    for key, sentence in zip(doc.sentence_keys, doc.sentence_texts):
        pos = doc.text.find(sentence, cursor)
        if pos < 0:
            pos = doc.text.find(sentence)
        if pos < 0:
            continue
        end = pos + len(sentence)
        offsets.append((key, pos, end))
        cursor = end
    return offsets


def evaluate_variant(
    rows: list[dict],
    variant: Variant,
    chunks: list[Chunk],
    parents: dict[str, Chunk],
    query_embeddings,
    model: SentenceTransformer,
    dense_top_k: int,
    sparse_top_k: int,
    k_values: list[int],
) -> tuple[dict[str, float], list[dict]]:
    chunk_texts = [chunk.text for chunk in chunks]
    chunk_embeddings = normalize(model.encode(chunk_texts, batch_size=64, show_progress_bar=True))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    sparse_matrix = vectorizer.fit_transform(chunk_texts)
    query_sparse = vectorizer.transform(row["question"] for row in rows)

    per_question = []
    for index, row in enumerate(rows):
        dense_scores = np.asarray(chunk_embeddings @ query_embeddings[index].T).reshape(-1)
        sparse_scores = query_sparse[index] @ sparse_matrix.T
        sparse_scores = np.asarray(sparse_scores.toarray()).reshape(-1)
        ranking = rrf(dense_scores, sparse_scores, dense_top_k=dense_top_k, sparse_top_k=sparse_top_k)
        retrieved = [chunks[item] for item in ranking[: max(k_values)]]
        per_question.append(score_question(row, variant, retrieved, parents, chunks, k_values))

    return aggregate_metrics(per_question, chunks, variant), per_question


def rrf(dense_scores, sparse_scores, dense_top_k: int, sparse_top_k: int, rrf_k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    dense_ranked = np.argsort(-dense_scores)[:dense_top_k]
    sparse_ranked = np.argsort(-sparse_scores)[:sparse_top_k]
    for ranking in (dense_ranked, sparse_ranked):
        for rank, item in enumerate(ranking, start=1):
            scores[int(item)] = scores.get(int(item), 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda item: scores[item], reverse=True)


def score_question(
    row: dict,
    variant: Variant,
    retrieved: list[Chunk],
    parents: dict[str, Chunk],
    all_chunks: list[Chunk],
    k_values: list[int],
) -> dict:
    gold_sentence_keys = set(row.get("all_relevant_sentence_keys") or row.get("gold_sentence_keys") or [])
    gold_doc_ids = set(row.get("gold_source_doc_ids") or [])
    if not gold_doc_ids:
        gold_doc_ids = {f"{row['question_id']}_doc_{document_id_from_sentence_key(key)}" for key in gold_sentence_keys if key}
    result = {
        "question_id": row["question_id"],
        "question": row["question"],
        "gold_doc_count": len(gold_doc_ids),
        "gold_sentence_count": len(gold_sentence_keys),
        "mrr": reciprocal_rank(retrieved, gold_doc_ids),
    }

    for k in k_values:
        top = retrieved[:k]
        hit_docs = {chunk.source_doc_id for chunk in top if chunk.source_doc_id in gold_doc_ids}
        contexts = contexts_for_variant(variant, top, parents, all_chunks, row.get("question", ""))
        context_keys = set()
        context_chars = 0
        for context in contexts:
            context_keys.update(context.sentence_keys)
            context_chars += len(context.text)
        sentence_hits = context_keys & gold_sentence_keys
        result[f"hitrate@{k}"] = 1.0 if hit_docs else 0.0
        result[f"doc_recall@{k}"] = ratio(len(hit_docs), len(gold_doc_ids))
        result[f"sentence_recall@{k}"] = ratio(len(sentence_hits), len(gold_sentence_keys))
        result[f"sentence_precision@{k}"] = ratio(len(sentence_hits), len(context_keys))
        result[f"context_chars@{k}"] = float(context_chars)
        result[f"context_count@{k}"] = float(len(contexts))
    return result


def contexts_for_variant(
    variant: Variant,
    retrieved: list[Chunk],
    parents: dict[str, Chunk],
    all_chunks: list[Chunk] | None = None,
    query: str = "",
) -> list[Chunk]:
    if variant.strategy == "single":
        return retrieved
    policy = variant.context_policy
    if policy == "adaptive":
        policy = select_context_policy(query, retrieved)
    if policy == "child":
        return dedupe_chunks(retrieved)
    if policy == "neighbor":
        return neighbor_contexts(retrieved, all_chunks or [])

    contexts = []
    seen = set()
    for chunk in retrieved:
        if chunk.parent_id in seen:
            continue
        parent = parents.get(chunk.parent_id)
        if parent:
            contexts.append(parent)
            seen.add(chunk.parent_id)
    return contexts


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    result = []
    seen = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        result.append(chunk)
    return result


def neighbor_contexts(retrieved: list[Chunk], all_chunks: list[Chunk], window: int = 1) -> list[Chunk]:
    by_parent: dict[str, list[Chunk]] = {}
    for chunk in all_chunks:
        by_parent.setdefault(chunk.parent_id, []).append(chunk)
    for siblings in by_parent.values():
        siblings.sort(key=lambda item: chunk_order(item.chunk_id))

    result = []
    seen = set()
    for chunk in retrieved:
        siblings = by_parent.get(chunk.parent_id) or [chunk]
        current_order = chunk_order(chunk.chunk_id)
        for sibling in siblings:
            if abs(chunk_order(sibling.chunk_id) - current_order) <= window and sibling.chunk_id not in seen:
                seen.add(sibling.chunk_id)
                result.append(sibling)
    return result or dedupe_chunks(retrieved)


def chunk_order(chunk_id: str) -> int:
    marker = "_child_"
    if marker not in chunk_id:
        return 0
    try:
        return int(chunk_id.rsplit(marker, 1)[1])
    except ValueError:
        return 0


def select_context_policy(query: str, retrieved: list[Chunk]) -> str:
    lowered = query.lower()
    parent_markers = {
        "why", "how do", "how does", "how can", "how to", "compare", "difference",
        "relationship", "mechanism", "process", "steps", "explain", "describe",
        "为什么", "如何", "比较", "区别", "联系", "机制", "流程", "步骤", "解释", "说明",
    }
    if any(marker in lowered for marker in parent_markers):
        return "parent"
    parent_counts: dict[str, int] = {}
    for chunk in retrieved:
        parent_counts[chunk.parent_id] = parent_counts.get(chunk.parent_id, 0) + 1
    if max(parent_counts.values(), default=0) >= 2:
        return "neighbor"
    fact_markers = {"who", "when", "where", "which", "what", "how many", "谁", "哪里", "多少", "哪"}
    if any(marker in lowered for marker in fact_markers):
        return "child"
    return "neighbor"


def reciprocal_rank(retrieved: list[Chunk], gold_doc_ids: set[str]) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source_doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(per_question: list[dict], chunks: list[Chunk], variant: Variant) -> dict[str, float]:
    if not per_question:
        return {
            "rows": 0.0,
            "index_chunks": float(len(chunks)),
            "avg_chunk_chars": mean(len(chunk.text) for chunk in chunks),
            "strategy_parent_child": 1.0 if variant.strategy == "parent_child" else 0.0,
            "child_size": float(variant.child_size),
            "child_overlap": float(variant.child_overlap),
            "parent_size": float(variant.parent_size or 0),
            "parent_overlap": float(variant.parent_overlap),
            "evaluation_valid": 0.0,
        }
    keys = [key for key in per_question[0] if key not in {"question_id", "question"}]
    metrics = {key: mean(row.get(key, 0.0) for row in per_question) for key in keys}
    metrics.update(
        {
            "rows": float(len(per_question)),
            "index_chunks": float(len(chunks)),
            "avg_chunk_chars": mean(len(chunk.text) for chunk in chunks),
            "strategy_parent_child": 1.0 if variant.strategy == "parent_child" else 0.0,
            "child_size": float(variant.child_size),
            "child_overlap": float(variant.child_overlap),
            "parent_size": float(variant.parent_size or 0),
            "parent_overlap": float(variant.parent_overlap),
        }
    )
    return metrics


def add_balanced_scores(rows: list[dict]) -> list[dict]:
    max_context_chars = max(row.get("context_chars@5", 0.0) for row in rows) or 1.0
    max_index_chunks = max(row.get("index_chunks", 0.0) for row in rows) or 1.0
    scored = []
    for row in rows:
        context_cost = row.get("context_chars@5", 0.0) / max_context_chars
        index_cost = row.get("index_chunks", 0.0) / max_index_chunks
        balanced = (
            0.30 * row.get("mrr", 0.0)
            + 0.25 * row.get("hitrate@5", 0.0)
            + 0.30 * row.get("sentence_recall@5", 0.0)
            + 0.15 * row.get("sentence_precision@5", 0.0)
            - 0.05 * context_cost
            - 0.03 * index_cost
        )
        next_row = dict(row)
        next_row["balanced_score"] = balanced
        scored.append(next_row)
    return sorted(scored, key=lambda item: item["balanced_score"], reverse=True)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = [
        "variant",
        "strategy",
        "balanced_score",
        "mrr",
        "hitrate@1",
        "hitrate@5",
        "doc_recall@5",
        "sentence_recall@5",
        "sentence_precision@5",
        "context_chars@5",
        "context_count@5",
        "index_chunks",
        "avg_chunk_chars",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(",".join(keys) + "\n")
        for row in rows:
            file.write(",".join(format_csv_value(row.get(key, "")) for key in keys) + "\n")


def write_report(
    path: Path,
    summaries: list[dict],
    rows: list[dict],
    source_docs: list[SourceDoc],
    variants: list[Variant],
    args,
    dataset_label: str,
    validation_warnings: list[dict] | None = None,
    validity_summary: dict | None = None,
) -> None:
    best_parent_child = next(row for row in summaries if row["strategy"] == "parent_child")
    best_single = next(row for row in summaries if row["strategy"] == "single")
    lines = [
        "# Chunking Ablation Report",
        "",
        f"- Dataset: `{dataset_label}`",
        f"- Rows: {len(rows)}",
        f"- Source documents: {len(source_docs)}",
        f"- Retrieval: all-mpnet-base-v2 dense + TF-IDF sparse, RRF fusion",
        f"- Reranker: disabled to isolate chunking strategy",
        f"- Variants: {', '.join(variant.name for variant in variants)}",
        "",
        "## Top Variants",
        "",
        "| Rank | Variant | Strategy | Balanced | MRR | Hit@5 | DocRecall@5 | SentenceRecall@5 | SentencePrecision@5 | CtxChars@5 | IndexChunks |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summaries, start=1):
        lines.append(
            "| {rank} | {variant} | {strategy} | {balanced:.4f} | {mrr:.4f} | {hit5:.4f} | "
            "{docrec5:.4f} | {sentrec5:.4f} | {sentprec5:.4f} | {ctx:.1f} | {chunks:.0f} |".format(
                rank=rank,
                variant=row["variant"],
                strategy=row["strategy"],
                balanced=row["balanced_score"],
                mrr=row["mrr"],
                hit5=row["hitrate@5"],
                docrec5=row["doc_recall@5"],
                sentrec5=row["sentence_recall@5"],
                sentprec5=row["sentence_precision@5"],
                ctx=row["context_chars@5"],
                chunks=row["index_chunks"],
            )
        )

    lines.extend(
        [
            "",
            "## Best Parent-Child vs Best Single",
            "",
            "| Metric | Best Parent-Child | Best Single | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in [
        "balanced_score",
        "mrr",
        "hitrate@1",
        "hitrate@5",
        "doc_recall@5",
        "sentence_recall@5",
        "sentence_precision@5",
        "context_chars@5",
        "index_chunks",
    ]:
        left = best_parent_child.get(key, 0.0)
        right = best_single.get(key, 0.0)
        lines.append(f"| {key} | {left:.4f} | {right:.4f} | {left - right:+.4f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `sentence_recall@5` is a deterministic proxy for context recall: the share of gold sentence/passage keys covered by the final contexts.",
            "- `sentence_precision@5` is the share of final-context sentence keys that are gold sentence keys.",
            "- Parent-child variants retrieve child chunks, then expand the returned context to the matching parent chunk.",
            "- Single variants retrieve and return the same fixed-size chunk.",
        ]
    )
    lines.extend(validation_markdown_section(validation_warnings or [], validity_summary))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_csv_value(value) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return ""
    return str(value)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


if __name__ == "__main__":
    main()
