import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessage, ToolMessage

from rag_agent.edges import route_after_answer_evaluation
from rag_agent.nodes.evaluation import evaluate_answer
from rag_agent.nodes.execution import collect_answer, knowledge_fallback_answer


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.content)


class TestKnowledgeFallback(unittest.TestCase):
    def test_low_score_evidence_is_unsatisfactory_before_fallback(self):
        state = {
            "question": "What is the answer?",
            "final_answer": "The answer is unsupported.",
            "answer_mode": "rag_qa",
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [],
                            "gaps": ["No relevant document context was retrieved."],
                            "diagnostics": {
                                "evidence_status": "low_score",
                                "best_rerank_score": 0.1,
                            },
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_1",
                )
            ],
        }

        result = evaluate_answer(state, FakeLLM("should not be called"))

        self.assertFalse(result["answer_is_satisfactory"])
        self.assertEqual(result["retrieval_evidence_status"], "low_score")
        self.assertEqual(result["fallback_reason"], "rerank_score_below_threshold")

    def test_llm_judged_insufficient_routes_to_fallback_after_retries(self):
        state = {
            "answer_is_satisfactory": False,
            "answer_evaluation_count": 2,
            "answer_mode": "rag_qa",
            "iteration_count": 2,
            "tool_call_count": 2,
        }

        self.assertEqual(route_after_answer_evaluation(state), "knowledge_fallback")

    def test_knowledge_fallback_answer_uses_general_answer_mode(self):
        llm = FakeLLM("I did not find usable information in the knowledge base. General answer.")

        result = knowledge_fallback_answer(
            {
                "question": "Explain caching.",
                "fallback_reason": "no_relevant_document_context",
                "answer_evaluation_count": 2,
            },
            llm,
        )

        self.assertEqual(result["answer_mode"], "knowledge_fallback")
        self.assertFalse(result["used_knowledge_base"])
        self.assertTrue(result["fallback_triggered"])
        self.assertEqual(result["fallback_reason"], "no_relevant_document_context")

    def test_collect_answer_drops_sources_for_knowledge_fallback(self):
        state = {
            "question": "Explain caching.",
            "question_index": 0,
            "messages": [AIMessage(content="Knowledge base had no usable information.\n\nNo Sources here.")],
            "answer_mode": "knowledge_fallback",
            "used_knowledge_base": False,
            "fallback_reason": "retrieval_error",
        }

        result = collect_answer(state)
        task_result = result["task_results"][0]

        self.assertEqual(task_result["answer_mode"], "knowledge_fallback")
        self.assertFalse(task_result["used_knowledge_base"])
        self.assertEqual(task_result["sources"], [])

    def test_thresholded_context_score_is_low_score(self):
        state = {
            "question": "What is the answer?",
            "final_answer": "The answer is unsupported.",
            "answer_mode": "rag_qa",
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [{"content": "weak evidence", "score": 0.1}],
                            "gaps": [],
                            "diagnostics": {"evidence_status": "sufficient"},
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_1",
                )
            ],
        }

        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            result = evaluate_answer(state, FakeLLM("should not be called"))

        self.assertFalse(result["answer_is_satisfactory"])
        self.assertEqual(result["retrieval_evidence_status"], "low_score")
        self.assertEqual(result["best_rerank_score"], 0.1)

    def test_default_rerank_threshold_marks_negative_score_low(self):
        state = {
            "question": "What is the answer?",
            "final_answer": "The answer is unsupported.",
            "answer_mode": "rag_qa",
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [{"content": "weak evidence", "score": -0.1}],
                            "gaps": [],
                            "diagnostics": {"evidence_status": "sufficient"},
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_1",
                )
            ],
        }

        with patch("config.RERANKER_SCORE_THRESHOLD", None), patch("config.RERANKER_ENABLED", True):
            result = evaluate_answer(state, FakeLLM("should not be called"))

        self.assertFalse(result["answer_is_satisfactory"])
        self.assertEqual(result["retrieval_evidence_status"], "low_score")


if __name__ == "__main__":
    unittest.main()
