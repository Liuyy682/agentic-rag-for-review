import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from evaluation.ragbench_keys import document_id_from_sentence_key
from evaluation.ragbench_metadata import build_ragbench_local_eval_metadata
from evaluation.runners.ragbench_eval_runner import (
    build_ragbench_eval_metadata,
    score_ragbench_context_order,
    validate_reuse_existing_outputs,
)


class TestRagbenchEvaluationAccuracy(unittest.TestCase):
    def test_sentence_key_document_id_parses_multi_digit_document_ids(self):
        self.assertEqual(document_id_from_sentence_key("0c"), "0")
        self.assertEqual(document_id_from_sentence_key("3f"), "3")
        self.assertEqual(document_id_from_sentence_key("10a"), "10")

    def test_ragbench_context_order_uses_full_document_id(self):
        row = {
            "id": "sample",
            "question": "q",
            "documents": [f"doc {index}" for index in range(11)],
            "documents_sentences": [[(f"{index}a", f"sentence {index}")] for index in range(11)],
            "all_relevant_sentence_keys": ["10a"],
        }

        result = score_ragbench_context_order(row)

        self.assertAlmostEqual(result["document_mrr"], 1 / 11)
        self.assertEqual(result["document_recall@20"], 1.0)

    def test_oracle_context_metadata_is_explicit(self):
        metadata = build_ragbench_eval_metadata(subset="covidqa", split="test", rows=3)

        self.assertEqual(metadata["evaluation_type"], "oracle_context_generation_eval")
        self.assertFalse(metadata["uses_project_retriever"])

    def test_ragbench_local_metadata_marks_synthetic_document_retrieval(self):
        metadata = build_ragbench_local_eval_metadata(
            subset="covidqa",
            split="test",
            limit=3,
            offset=0,
        )

        self.assertEqual(metadata["evaluation_type"], "synthetic_ragbench_document_retrieval")
        self.assertTrue(metadata["uses_project_retriever"])
        self.assertTrue(metadata["uses_synthetic_document_chunks"])

    def test_reuse_existing_rejects_question_id_mismatch(self):
        rows = [{"id": "1"}, {"id": "2"}]
        outputs = [{"question_id": "ragbench_covidqa_test_1"}]

        reusable, warnings = validate_reuse_existing_outputs(rows, outputs, subset="covidqa", split="test")

        self.assertFalse(reusable)
        self.assertIn("reuse_existing_question_mismatch", {warning["code"] for warning in warnings})


if __name__ == "__main__":
    unittest.main()
