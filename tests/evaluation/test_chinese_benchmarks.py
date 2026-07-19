import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from evaluation.chinese_benchmarks import (
    BenchmarkDocument,
    BenchmarkQuestion,
    build_crud_questions,
    interleave_crud_questions,
    load_crud_benchmark,
    load_crud_documents,
    load_t2_benchmark,
    select_candidate_doc_ids,
)
from evaluation.runners.chinese_benchmark_runner import (
    aggregate_retrieval_metrics,
    configure_eval_database,
    experiment_config,
    grouped_ragas_metrics,
    run_chinese_benchmark,
    score_retrieval_question,
)
from evaluation.runners.chinese_benchmark_compare import (
    bootstrap_delta_ci,
    compare_runs,
    validate_compatible_runs,
)
from evaluation.runners.chinese_benchmark_select import select_final_candidate


def crud_split():
    return {
        "questanswer_1doc": [
            {"question_id": "q1", "questions": "问题一", "answers": "答案一", "ID": "d1"}
        ],
        "questanswer_2docs": [
            {"question_id": "q2", "questions": "问题二", "answers": "答案二", "IDs": ["d1", "d2"]}
        ],
        "questanswer_3docs": [
            {"question_id": "q3", "questions": "问题三", "answers": "答案三", "ID": ["d1", "d2", "d3"]}
        ],
    }


class TestCrudAdapter(unittest.TestCase):
    def test_normalizes_all_qa_groups_and_scalar_or_list_ids(self):
        raw = crud_split()
        raw["questanswer_1doc"][0]["questions"] = ["问题一"]
        raw["questanswer_1doc"][0]["answers"] = ["答案", "补充"]
        questions = build_crud_questions(raw)

        self.assertEqual([item.question_id for item in questions], ["crud_q1", "crud_q2", "crud_q3"])
        self.assertEqual([len(item.gold_doc_ids) for item in questions], [1, 2, 3])
        self.assertEqual(questions[1].gold_doc_ids, ("d1", "d2"))
        self.assertEqual(questions[0].question, "问题一")
        self.assertEqual(questions[0].reference, "答案\n补充")

    def test_rejects_missing_fields_duplicate_ids_and_wrong_group_size(self):
        missing = crud_split()
        missing["questanswer_1doc"][0]["questions"] = ""
        with self.assertRaisesRegex(ValueError, "empty 'questions'"):
            build_crud_questions(missing)

        duplicate = crud_split()
        duplicate["questanswer_2docs"][0]["IDs"] = ["d1", "d1"]
        with self.assertRaisesRegex(ValueError, "duplicate document IDs"):
            build_crud_questions(duplicate)

        wrong_size = crud_split()
        wrong_size["questanswer_3docs"][0]["ID"] = ["d1", "d2"]
        with self.assertRaisesRegex(ValueError, "expected 3"):
            build_crud_questions(wrong_size)

    def test_interleaves_one_two_and_three_document_questions(self):
        questions = interleave_crud_questions(build_crud_questions(crud_split()))
        self.assertEqual(
            [question.task for question in questions],
            ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"],
        )

    def test_loads_json_jsonl_and_text_documents_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.json").write_text(
                json.dumps([{"ID": "d1", "title": "标题", "text": "正文"}], ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "b.jsonl").write_text(
                json.dumps({"id": "d2", "content": "内容二"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (root / "d3.txt").write_text("内容三", encoding="utf-8")
            with zipfile.ZipFile(root / "docs.zip", "w") as archive:
                archive.writestr("nested/d4.json", json.dumps({"ID": "d4", "text": "内容四"}))

            documents = load_crud_documents(root)

            self.assertEqual(documents["d1"], "# 标题\n\n正文")
            self.assertEqual(documents["d2"], "内容二")
            self.assertEqual(documents["d3"], "内容三")
            self.assertEqual(documents["d4"], "内容四")

            (root / "duplicate.md").write_text("not a duplicate id", encoding="utf-8")
            (root / "duplicate.json").write_text(
                json.dumps({"ID": "duplicate", "text": "duplicate"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate CRUD-RAG document ID"):
                load_crud_documents(root)

    def test_load_crud_selects_deterministic_candidates_and_validates_missing_gold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "data" / "crud_split"
            docs_dir = root / "data" / "80000_docs"
            split_dir.mkdir(parents=True)
            docs_dir.mkdir(parents=True)
            (split_dir / "split_merged.json").write_text(
                json.dumps(crud_split(), ensure_ascii=False), encoding="utf-8"
            )
            for doc_id in ("d1", "d2", "d3", "d4", "d5"):
                (docs_dir / f"{doc_id}.txt").write_text(f"文档 {doc_id}", encoding="utf-8")

            questions, documents = load_crud_benchmark(root, limit=1, offset=0, distractor_docs=2)

            self.assertEqual([item.question_id for item in questions], ["crud_q1"])
            self.assertEqual([item.doc_id for item in documents], ["d1", "d2", "d3"])

            (docs_dir / "d1.txt").unlink()
            with self.assertRaisesRegex(ValueError, "absent from 80000_docs"):
                load_crud_benchmark(root, limit=1, offset=0, distractor_docs=2)

    def test_candidate_selection_uses_zero_for_full_corpus(self):
        documents = {"d3": "3", "d1": "1", "d2": "2", "d4": "4"}
        self.assertEqual(select_candidate_doc_ids(documents, {"d3"}, 1), {"d1", "d3"})
        self.assertEqual(select_candidate_doc_ids(documents, {"d3"}, 0), set(documents))


class TestT2Adapter(unittest.TestCase):
    def test_maps_qrels_to_benchmark_gold_source_ids(self):
        corpus = [
            {"_id": "d1", "title": "标题", "text": "文档一"},
            {"_id": "d2", "title": "", "text": "文档二"},
            {"_id": "d3", "title": "", "text": "干扰文档"},
        ]
        queries = [{"_id": "q1", "text": "问题一"}, {"_id": "q2", "text": "问题二"}]
        qrels = [
            {"qid": "q1", "pid": "d1", "score": 1},
            {"qid": "q2", "pid": "d2", "score": 1},
            {"qid": "q2", "pid": "d3", "score": 0},
        ]
        with patch(
            "evaluation.chinese_benchmarks.read_hf_parquet_records",
            side_effect=[corpus, queries, qrels],
        ):
            questions, documents = load_t2_benchmark(limit=1, offset=1, distractor_docs=1)

        self.assertEqual(questions[0].question_id, "t2_q2")
        self.assertEqual(questions[0].gold_doc_ids, ("d2",))
        self.assertEqual([document.doc_id for document in documents], ["d1", "d2"])


class TestChineseBenchmarkMetricsAndSafety(unittest.TestCase):
    def test_experiment_config_applies_and_restores_overrides(self):
        from evaluation.runners import chinese_benchmark_runner as runner

        original = (
            runner.config.RETRIEVAL_FUSION_MODE,
            runner.config.RERANKER_ENABLED,
            runner.config.RETRIEVAL_CONTEXT_POLICY,
        )
        with experiment_config(retrieval_mode="dense", reranker_enabled=False, context_policy="child"):
            self.assertEqual(runner.config.RETRIEVAL_FUSION_MODE, "dense")
            self.assertFalse(runner.config.RERANKER_ENABLED)
            self.assertEqual(runner.config.RETRIEVAL_CONTEXT_POLICY, "child")
        self.assertEqual(
            (
                runner.config.RETRIEVAL_FUSION_MODE,
                runner.config.RERANKER_ENABLED,
                runner.config.RETRIEVAL_CONTEXT_POLICY,
            ),
            original,
        )

    def test_scores_source_document_retrieval(self):
        question = BenchmarkQuestion("q1", "问题", "答案", ("d1", "d2"), "questanswer_2docs")
        chunks = [
            {"source_doc_id": "noise"},
            {"source_doc_id": "d2"},
            {"source_doc_id": "d1"},
        ]

        row = score_retrieval_question(question, chunks, [1, 3])
        metrics = aggregate_retrieval_metrics([row], [1, 3])

        self.assertEqual(row["mrr"], 0.5)
        self.assertEqual(row["recall@3"], 1.0)
        self.assertEqual(metrics["scored_rows"], 1.0)

    def test_groups_ragas_by_crud_question_type(self):
        rows = [
            {"question_type": "questanswer_1doc", "faithfulness": 1.0, "context_precision": 0.5, "context_recall": 1.0},
            {"question_type": "questanswer_2docs", "faithfulness": 0.5, "context_precision": 1.0, "context_recall": 0.5},
        ]
        grouped = grouped_ragas_metrics(rows)
        self.assertEqual(grouped["questanswer_1doc"]["faithfulness"], 1.0)
        self.assertEqual(grouped["questanswer_2docs"]["context_recall"], 0.5)

    def test_database_guard_fails_before_resetting_pool(self):
        with patch("evaluation.runners.chinese_benchmark_runner.reset_pool_for_tests") as reset:
            with self.assertRaisesRegex(RuntimeError, "EVAL_DATABASE_URL is required"):
                configure_eval_database({"DATABASE_URL": "postgresql://runtime"})
            reset.assert_not_called()

            with self.assertRaisesRegex(RuntimeError, "must not equal"):
                configure_eval_database(
                    {"DATABASE_URL": "postgresql://same", "EVAL_DATABASE_URL": "postgresql://same"}
                )
            reset.assert_not_called()

    def test_database_guard_switches_only_to_distinct_eval_database(self):
        with patch("evaluation.runners.chinese_benchmark_runner.reset_pool_for_tests") as reset, patch(
            "evaluation.runners.chinese_benchmark_runner.config.DATABASE_URL", "postgresql://runtime"
        ):
            result = configure_eval_database(
                {"DATABASE_URL": "postgresql://runtime", "EVAL_DATABASE_URL": "postgresql://eval"}
            )
        self.assertEqual(result, "postgresql://eval")
        reset.assert_called_once_with()

    def test_t2_rejects_generation_mode_before_database_access(self):
        with patch("evaluation.runners.chinese_benchmark_runner.load_t2_benchmark") as load:
            with self.assertRaisesRegex(ValueError, "retrieval-only"):
                run_chinese_benchmark(
                    dataset="t2_retrieval",
                    limit=1,
                    offset=0,
                    distractor_docs=1,
                    top_k=1,
                    output_dir="unused",
                    answer_mode="agent",
                )
        load.assert_not_called()

    def test_runner_writes_t2_retrieval_only_outputs(self):
        question = BenchmarkQuestion("t2_q1", "问题", "", ("d1",), "t2_retrieval")
        document = BenchmarkDocument("d1", "文档内容")
        with tempfile.TemporaryDirectory() as tmpdir, ExitStack() as stack:
            stack.enter_context(
                patch(
                    "evaluation.runners.chinese_benchmark_runner.load_t2_benchmark",
                    return_value=([question], [document]),
                )
            )
            self._patch_runner_dependencies(stack, source_doc_id="d1")

            result = run_chinese_benchmark(
                dataset="t2_retrieval",
                limit=1,
                offset=0,
                distractor_docs=1,
                top_k=1,
                output_dir=tmpdir,
                skip_ragas=False,
                run_label="B0_dense",
                retrieval_mode="dense",
                reranker_enabled=False,
                reranker_final_top_k=10,
                context_policy="child",
            )

            run_dir = Path(result["run_dir"])
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["retrieval_only"])
            self.assertEqual(metadata["context_top_k"], 3)
            self.assertEqual(metadata["run_label"], "B0_dense")
            self.assertEqual(metadata["retrieval_fusion_mode"], "dense")
            self.assertFalse(metadata["reranker_enabled"])
            self.assertEqual(metadata["retrieval_context_policy"], "child")
            self.assertIn("question_fingerprint", metadata)
            self.assertIn("retrieval_latency_ms", result["retrieval_metrics"])
            self.assertEqual(result["retrieval_metrics"]["scored_rows"], 1.0)
            self.assertEqual((run_dir / "rag_outputs.jsonl").read_text(encoding="utf-8"), "")

    def test_runner_writes_crud_ragas_and_grouped_outputs(self):
        question = BenchmarkQuestion("crud_q1", "问题", "标准答案", ("d1",), "questanswer_1doc")
        document = BenchmarkDocument("d1", "文档内容")

        def fake_ragas(outputs, **_kwargs):
            rows = [
                {
                    **row,
                    "faithfulness": 1.0,
                    "context_precision": 1.0,
                    "context_recall": 1.0,
                }
                for row in outputs
            ]
            return rows, {
                "rows": 1.0,
                "faithfulness": 1.0,
                "faithfulness_rows": 1.0,
                "faithfulness_missing_rows": 0.0,
                "context_precision": 1.0,
                "context_precision_rows": 1.0,
                "context_precision_missing_rows": 0.0,
                "context_recall": 1.0,
                "context_recall_rows": 1.0,
                "context_recall_missing_rows": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir, ExitStack() as stack:
            stack.enter_context(
                patch(
                    "evaluation.runners.chinese_benchmark_runner.load_crud_benchmark",
                    return_value=([question], [document]),
                )
            )
            stack.enter_context(
                patch("evaluation.runners.chinese_benchmark_runner.run_ragas_metrics", side_effect=fake_ragas)
            )
            self._patch_runner_dependencies(stack, source_doc_id="d1")

            result = run_chinese_benchmark(
                dataset="crud_rag",
                limit=1,
                offset=0,
                distractor_docs=1,
                top_k=1,
                output_dir=tmpdir,
                crud_root="unused",
                answer_mode="reference",
            )

            run_dir = Path(result["run_dir"])
            grouped = json.loads((run_dir / "ragas_metrics_by_group.json").read_text(encoding="utf-8"))
            output = json.loads((run_dir / "rag_outputs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(output["reference"], "标准答案")
            self.assertEqual(output["retrieved_contexts"], ["召回上下文"])
            self.assertEqual(grouped["questanswer_1doc"]["faithfulness"], 1.0)
            self.assertEqual(result["ragas_metrics"]["context_recall"], 1.0)

    @staticmethod
    def _patch_runner_dependencies(stack, source_doc_id):
        fake_rag = SimpleNamespace(vector_db=object(), parent_store=object())
        fake_summary = SimpleNamespace(documents=[])
        fake_manager = SimpleNamespace(clear_all=lambda: None, add_documents_detailed=lambda *_a, **_k: fake_summary)
        fake_pipeline = SimpleNamespace(
            search_child_chunk_documents=lambda *_a, **_k: [object()],
            rerank_child_documents=lambda _q, docs: docs,
        )
        stack.enter_context(
            patch("evaluation.runners.chinese_benchmark_runner.configure_eval_database", return_value="postgresql://eval")
        )
        stack.enter_context(patch("evaluation.runners.chinese_benchmark_runner.RAGSystem", return_value=fake_rag))
        stack.enter_context(
            patch("evaluation.runners.chinese_benchmark_runner.DocumentManager", return_value=fake_manager)
        )
        stack.enter_context(
            patch("evaluation.runners.chinese_benchmark_runner.RetrievalPipeline", return_value=fake_pipeline)
        )
        stack.enter_context(
            patch(
                "evaluation.runners.chinese_benchmark_runner._doc_to_retrieved_chunk",
                return_value={
                    "rank": 1,
                    "chunk_id": "c1",
                    "parent_id": "p1",
                    "source_file": f"{source_doc_id}.md",
                    "source_doc_id": source_doc_id,
                    "score_fused": 1.0,
                    "score_rerank": 1.0,
                    "text": "召回上下文",
                },
            )
        )
        stack.enter_context(
            patch(
                "evaluation.runners.chinese_benchmark_runner._contexts_for_docs",
                return_value=[{"content": "召回上下文"}],
            )
        )


class TestChineseBenchmarkComparison(unittest.TestCase):
    def test_bootstrap_delta_is_deterministic(self):
        values = [(0.0, 0.1), (0.2, 0.4), (0.4, 0.7)]
        first = bootstrap_delta_ci(values, samples=200, seed=42)
        second = bootstrap_delta_ci(values, samples=200, seed=42)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)

    def test_rejects_incompatible_metadata(self):
        baseline = self._metadata("t2_retrieval", "same")
        candidate = self._metadata("t2_retrieval", "different")
        with self.assertRaisesRegex(ValueError, "not comparable"):
            validate_compatible_runs(baseline, candidate)

    def test_compare_runs_writes_paired_summary_and_verdict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            output = root / "comparison"
            baseline.mkdir()
            candidate.mkdir()
            baseline_meta = self._metadata("t2_retrieval", "same")
            candidate_meta = {**baseline_meta, "run_label": "B3"}
            baseline_meta["run_label"] = "B0"
            self._write_run(baseline, baseline_meta, [0.10, 0.20, 0.30])
            self._write_run(candidate, candidate_meta, [0.20, 0.35, 0.45])

            summary = compare_runs(baseline, candidate, output, bootstrap_samples=300, seed=42)

            self.assertTrue(summary["retrieval_material_improvement"])
            self.assertEqual(summary["primary_metric"], "ndcg@10")
            self.assertGreater(summary["retrieval"]["ndcg@10"]["ci95_lower"], 0.0)
            self.assertTrue((output / "experiment_summary.json").exists())
            self.assertTrue((output / "experiment_summary.csv").exists())
            self.assertIn("检索得到有效提升", (output / "experiment_report.md").read_text(encoding="utf-8"))

    def test_selects_candidate_from_t2_and_crud_comparisons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summaries = {}
            for dataset, metric_values in (
                ("t2_retrieval", ([0.10, 0.20, 0.30], [0.20, 0.35, 0.45])),
                ("crud_rag", ([0.20, 0.30, 0.40], [0.35, 0.45, 0.55])),
            ):
                baseline = root / f"{dataset}_baseline"
                candidate = root / f"{dataset}_candidate"
                comparison = root / f"{dataset}_comparison"
                baseline.mkdir()
                candidate.mkdir()
                baseline_meta = self._metadata(dataset, f"{dataset}_questions")
                candidate_meta = {**baseline_meta, "run_label": "B2"}
                baseline_meta["run_label"] = "B0"
                types = (
                    ["questanswer_1doc", "questanswer_2docs", "questanswer_3docs"]
                    if dataset == "crud_rag"
                    else None
                )
                self._write_run(baseline, baseline_meta, metric_values[0], types)
                self._write_run(candidate, candidate_meta, metric_values[1], types)
                compare_runs(baseline, candidate, comparison, bootstrap_samples=100, seed=42)
                summaries[dataset] = comparison / "experiment_summary.json"

            result = select_final_candidate(
                {"B2": summaries["t2_retrieval"]},
                {"B2": summaries["crud_rag"]},
                root / "selection",
                bootstrap_samples=100,
                seed=42,
            )

            self.assertEqual(result["selected_candidate"], "B2")
            self.assertTrue(result["retrieval_material_improvement"])
            self.assertTrue((root / "selection" / "final_selection.md").exists())

    @staticmethod
    def _metadata(dataset, fingerprint):
        return {
            "dataset": dataset,
            "dataset_version": "v1",
            "limit": 3,
            "offset": 0,
            "distractor_docs": 1000,
            "candidate_document_count": 1003,
            "uses_full_corpus": False,
            "question_fingerprint": fingerprint,
            "candidate_document_fingerprint": "docs",
            "dense_model": "dense",
            "child_chunk_size": 300,
            "child_chunk_overlap": 60,
            "top_k": 10,
            "k_values": [1, 3, 5, 10],
        }

    @staticmethod
    def _write_run(path, metadata, values, question_types=None):
        (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        rows = []
        for index, value in enumerate(values):
            rows.append(
                {
                    "question_id": f"q{index}",
                    "question_type": question_types[index] if question_types else "t2_retrieval",
                    "mrr": value,
                    "ndcg@10": value,
                    "recall@10": value,
                    "hitrate@10": value,
                    "actual_results@10": 10.0,
                    "retrieval_latency_ms": 10.0 + index,
                    "context_count": 3.0,
                    "context_chars": 300.0,
                }
            )
        (path / "retrieval_per_question_metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        (path / "evaluation_warnings.jsonl").write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
