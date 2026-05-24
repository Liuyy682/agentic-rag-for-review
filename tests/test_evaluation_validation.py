import math
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.data import EvalQuestion
from evaluation.metrics.ragas_metrics import build_ragas_error_cases, summarize_ragas_rows
from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics
from evaluation.validation import build_validity_summary, validate_retrieval_inputs


class TestEvaluationValidation(unittest.TestCase):
    def test_empty_dataset_produces_invalid_warning_not_silent_zero(self):
        warnings = validate_retrieval_inputs(
            questions=[],
            result_rows=[],
            k_values=[10],
            evaluation_type="local_retriever_eval",
            configured_top_k=10,
        )
        summary = build_validity_summary(rows=0, warnings=warnings, evaluation_type="local_retriever_eval")
        metrics, per_question = compute_retrieval_metrics([], [], k_values=[10])

        self.assertEqual(per_question, [])
        self.assertIn("empty_dataset", {warning["code"] for warning in warnings})
        self.assertFalse(summary["evaluation_valid"])
        self.assertEqual(metrics["rows"], 0.0)
        self.assertEqual(metrics["scored_rows"], 0.0)
        self.assertEqual(metrics["unscored_rows"], 0.0)

    def test_missing_primary_gold_does_not_pollute_retrieval_averages(self):
        no_gold = EvalQuestion(
            question_id="no_gold",
            question="No gold?",
            reference_answer="",
            source_file="",
        )
        parent_gold = EvalQuestion(
            question_id="parent_gold",
            question="Parent gold?",
            reference_answer="",
            source_file="source.pdf",
            gold_parent_ids=["p2"],
        )
        results = [
            {"question_id": "no_gold", "retrieved_chunks": [{"chunk_id": "c1", "parent_id": "p1"}]},
            {"question_id": "parent_gold", "retrieved_chunks": [{"chunk_id": "c2", "parent_id": "p2"}]},
        ]

        metrics, per_question = compute_retrieval_metrics([no_gold, parent_gold], results, k_values=[1])

        self.assertEqual(metrics["rows"], 2.0)
        self.assertEqual(metrics["scored_rows"], 1.0)
        self.assertEqual(metrics["unscored_rows"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertFalse(per_question[0]["scored"])
        self.assertIn("missing_primary_gold", per_question[0]["warnings"])

    def test_multiple_gold_source_files_are_preserved_and_scored(self):
        question = EvalQuestion.from_dict(
            {
                "question_id": "q1",
                "question": "Which source?",
                "reference_answer": "answer",
                "source_file": "a.pdf",
                "gold_source_files": ["a.pdf", "b.pdf"],
            },
            line_no=1,
        )
        results = [
            {
                "question_id": "q1",
                "retrieved_chunks": [{"chunk_id": "c1", "parent_id": "", "source_file": "b.pdf"}],
            }
        ]

        metrics, per_question = compute_retrieval_metrics([question], results, k_values=[1])

        self.assertEqual(question.gold_source_files, ["a.pdf", "b.pdf"])
        self.assertEqual(per_question[0]["source_hitrate@1"], 1.0)
        self.assertEqual(metrics["source_hitrate@1"], 1.0)

    def test_actual_result_depth_warning_is_exposed(self):
        question = EvalQuestion(
            question_id="q1",
            question="Need ten?",
            reference_answer="answer",
            source_file="source.pdf",
            gold_parent_ids=["p1"],
        )
        results = [
            {"question_id": "q1", "retrieved_chunks": [{"chunk_id": "c1", "parent_id": "p1"}]},
        ]

        metrics, per_question = compute_retrieval_metrics([question], results, k_values=[10])
        warnings = validate_retrieval_inputs(
            questions=[question],
            result_rows=results,
            k_values=[10],
            evaluation_type="local_retriever_eval",
            configured_top_k=10,
        )

        self.assertEqual(per_question[0]["actual_results@10"], 1)
        self.assertEqual(metrics["actual_results@10"], 1.0)
        self.assertIn("insufficient_results_for_k", {warning["code"] for warning in warnings})

    def test_gold_result_id_alignment_warning_when_no_primary_overlap_exists(self):
        question = EvalQuestion(
            question_id="q1",
            question="Mismatch?",
            reference_answer="answer",
            source_file="source.pdf",
            gold_parent_ids=["gold_parent"],
        )
        results = [
            {
                "question_id": "q1",
                "retrieved_chunks": [{"chunk_id": "retrieved_child", "parent_id": "retrieved_parent"}],
            }
        ]

        warnings = validate_retrieval_inputs(
            questions=[question],
            result_rows=results,
            k_values=[1],
            evaluation_type="local_retriever_eval",
            configured_top_k=1,
        )

        self.assertIn("gold_result_id_no_overlap", {warning["code"] for warning in warnings})

    def test_ragas_summary_records_metric_coverage_and_missing_metric_cases(self):
        rows = [
            {"question_id": "q1", "question": "q1", "answer": "a", "retrieved_contexts": [], "faithfulness": 1.0},
            {"question_id": "q2", "question": "q2", "answer": "a", "retrieved_contexts": [], "faithfulness": math.nan},
        ]

        summary = summarize_ragas_rows(rows)
        cases = build_ragas_error_cases(rows)

        self.assertEqual(summary["rows"], 2.0)
        self.assertEqual(summary["faithfulness_rows"], 1.0)
        self.assertEqual(summary["faithfulness_missing_rows"], 1.0)
        self.assertEqual(cases[0]["failure_type"], "judge_metric_missing")


if __name__ == "__main__":
    unittest.main()
