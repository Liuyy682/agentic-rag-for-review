import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.data import EvalQuestion
from evaluation.metrics.retrieval_metrics import (
    build_retrieval_error_cases,
    compute_retrieval_metrics,
)


def _chunk(chunk_id=None, parent_id=None, source_file=None):
    return {"chunk_id": chunk_id, "parent_id": parent_id, "source_file": source_file}


class TestSourceOnlyGoldNdcg(unittest.TestCase):
    """Covers _ideal_gains (line 221) and _relevance_grade (line 236) for source-only gold."""

    def test_source_only_gold_produces_positive_ndcg(self):
        question = EvalQuestion(
            question_id="q1",
            question="source only",
            reference_answer="",
            source_file="source.pdf",
        )
        results = [
            {
                "question_id": "q1",
                "retrieved_chunks": [
                    _chunk(chunk_id="c1", parent_id="p1", source_file="source.pdf"),
                    _chunk(chunk_id="c2", parent_id="p2", source_file="other.pdf"),
                ],
            }
        ]

        metrics, per_question = compute_retrieval_metrics([question], results, k_values=[2])

        # Question is unscored (no child/parent gold) but per-question ndcg is still computed.
        self.assertFalse(per_question[0]["scored"])
        self.assertGreater(per_question[0]["ndcg@2"], 0.0)
        self.assertEqual(per_question[0]["source_hitrate@2"], 1.0)


class TestNdcgGrades(unittest.TestCase):
    def test_parent_match_grade_in_ndcg(self):
        # parent gold + parent hit exercises _relevance_grade returning 2.
        question = EvalQuestion(
            question_id="q1",
            question="parent grade",
            reference_answer="",
            source_file="",
            gold_parent_ids=["p1"],
        )
        results = [
            {"question_id": "q1", "retrieved_chunks": [_chunk(parent_id="p1")]}
        ]
        _, per_question = compute_retrieval_metrics([question], results, k_values=[1])
        self.assertEqual(per_question[0]["ndcg@1"], 1.0)

    def test_no_gold_yields_zero_ndcg(self):
        # No gold of any kind → _ideal_gains returns [] and ndcg is 0.0.
        question = EvalQuestion(
            question_id="q1",
            question="no gold",
            reference_answer="",
            source_file="",
        )
        results = [
            {"question_id": "q1", "retrieved_chunks": [_chunk(chunk_id="c1")]}
        ]
        _, per_question = compute_retrieval_metrics([question], results, k_values=[1])
        self.assertEqual(per_question[0]["ndcg@1"], 0.0)


class TestBuildRetrievalErrorCases(unittest.TestCase):
    def _per_question(self, questions, results, top_k):
        _, per_question = compute_retrieval_metrics(questions, results, k_values=[top_k])
        return per_question

    def test_missed_source_failure(self):
        question = EvalQuestion(
            question_id="q1",
            question="missed source",
            reference_answer="",
            source_file="wanted.pdf",
        )
        results = [{"question_id": "q1", "retrieved_chunks": [_chunk(source_file="other.pdf")]}]
        per_question = self._per_question([question], results, 3)

        cases = build_retrieval_error_cases([question], results, per_question, top_k=3)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["failure_type"], "missed_source")
        self.assertEqual(cases[0]["question_id"], "q1")
        self.assertEqual(cases[0]["expected_source_files"], ["wanted.pdf"])

    def test_missed_parent_failure(self):
        # source hit so it falls through to the parent branch.
        question = EvalQuestion(
            question_id="q2",
            question="missed parent",
            reference_answer="",
            source_file="doc.pdf",
            gold_parent_ids=["p_gold"],
        )
        results = [
            {
                "question_id": "q2",
                "retrieved_chunks": [_chunk(parent_id="p_other", source_file="doc.pdf")],
            }
        ]
        per_question = self._per_question([question], results, 3)

        cases = build_retrieval_error_cases([question], results, per_question, top_k=3)

        self.assertEqual(cases[0]["failure_type"], "missed_parent")
        self.assertEqual(cases[0]["expected_parent_ids"], ["p_gold"])

    def test_missed_child_failure(self):
        question = EvalQuestion(
            question_id="q3",
            question="missed child",
            reference_answer="",
            source_file="doc.pdf",
            gold_child_ids=["c_gold"],
        )
        results = [
            {
                "question_id": "q3",
                "retrieved_chunks": [_chunk(chunk_id="c_other", source_file="doc.pdf")],
            }
        ]
        per_question = self._per_question([question], results, 3)

        cases = build_retrieval_error_cases([question], results, per_question, top_k=3)

        self.assertEqual(cases[0]["failure_type"], "missed_child")
        self.assertEqual(cases[0]["expected_child_ids"], ["c_gold"])

    def test_bad_ranking_failure(self):
        # All recalls satisfied (child gold hit) but first relevant rank > 3.
        question = EvalQuestion(
            question_id="q4",
            question="bad ranking",
            reference_answer="",
            source_file="doc.pdf",
            gold_child_ids=["c_gold"],
        )
        retrieved = [
            _chunk(chunk_id="x1", source_file="doc.pdf"),
            _chunk(chunk_id="x2", source_file="doc.pdf"),
            _chunk(chunk_id="x3", source_file="doc.pdf"),
            _chunk(chunk_id="c_gold", source_file="doc.pdf"),
        ]
        results = [{"question_id": "q4", "retrieved_chunks": retrieved}]
        per_question = self._per_question([question], results, 5)

        cases = build_retrieval_error_cases([question], results, per_question, top_k=5)

        self.assertEqual(cases[0]["failure_type"], "bad_ranking")
        self.assertEqual(cases[0]["first_relevant_rank"], 4)

    def test_no_failure_yields_no_case(self):
        # Source, child all hit within top_k and at rank 1 → no failure recorded.
        question = EvalQuestion(
            question_id="q5",
            question="perfect",
            reference_answer="",
            source_file="doc.pdf",
            gold_child_ids=["c_gold"],
        )
        results = [
            {
                "question_id": "q5",
                "retrieved_chunks": [_chunk(chunk_id="c_gold", source_file="doc.pdf")],
            }
        ]
        per_question = self._per_question([question], results, 3)

        cases = build_retrieval_error_cases([question], results, per_question, top_k=3)

        self.assertEqual(cases, [])

    def test_missing_question_id_uses_empty_defaults(self):
        # No matching result row / per_question entry → defaults branch.
        question = EvalQuestion(
            question_id="missing",
            question="no data",
            reference_answer="",
            source_file="doc.pdf",
        )

        cases = build_retrieval_error_cases([question], [], [], top_k=3)

        # source_file set but source_hitrate defaults to 0.0 → missed_source.
        self.assertEqual(cases[0]["failure_type"], "missed_source")
        self.assertEqual(cases[0]["retrieved_parent_ids"], [])


if __name__ == "__main__":
    unittest.main()
