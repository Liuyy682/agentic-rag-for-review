import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.data import EvalQuestion
from evaluation.metrics.retrieval_metrics import compute_retrieval_metrics


class TestRetrievalMetrics(unittest.TestCase):
    def test_child_gold_takes_precedence_over_parent_gold(self):
        question = EvalQuestion(
            question_id="q1",
            question="child first",
            reference_answer="",
            source_file="source.pdf",
            gold_parent_ids=["p1"],
            gold_child_ids=["c2"],
        )
        results = [
            {
                "question_id": "q1",
                "retrieved_chunks": [
                    {"chunk_id": "c1", "parent_id": "p1"},
                    {"chunk_id": "c2", "parent_id": "p2"},
                ],
            }
        ]

        metrics, per_question = compute_retrieval_metrics([question], results, k_values=[1, 2])

        self.assertEqual(per_question[0]["first_relevant_rank"], 2)
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 1.0)

    def test_parent_only_gold_scores_parent_retrieval(self):
        question = EvalQuestion(
            question_id="q1",
            question="parent only",
            reference_answer="",
            source_file="source.pdf",
            gold_parent_ids=["p2"],
        )
        results = [
            {
                "question_id": "q1",
                "retrieved_chunks": [
                    {"chunk_id": "c1", "parent_id": "p1"},
                    {"chunk_id": "c2", "parent_id": "p2"},
                ],
            }
        ]

        metrics, per_question = compute_retrieval_metrics([question], results, k_values=[2])

        self.assertAlmostEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertGreater(per_question[0]["ndcg@2"], 0.0)

    def test_no_gold_rows_are_unscored(self):
        question = EvalQuestion(
            question_id="q1",
            question="no gold",
            reference_answer="",
            source_file="",
        )

        metrics, per_question = compute_retrieval_metrics([question], [{"question_id": "q1", "retrieved_chunks": []}], k_values=[1])

        self.assertFalse(per_question[0]["scored"])
        self.assertEqual(metrics["rows"], 1.0)
        self.assertEqual(metrics["scored_rows"], 0.0)
        self.assertEqual(metrics["unscored_rows"], 1.0)
        self.assertEqual(metrics["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
