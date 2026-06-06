import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_agent.nodes.evaluation import (
    _collect_retrieved_context,
    _parse_answer_evaluation,
    _retrieval_evidence,
    evaluate_answer,
)


class ConfigurableLLM:
    def __init__(self, content):
        self.content = content
        self.invoked_with = None
        self.config_kwargs = None

    def with_config(self, **kwargs):
        self.config_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.invoked_with = messages
        return AIMessage(content=self.content)


def rag_tool_message(payload, tool_call_id="1"):
    return ToolMessage(
        content=json.dumps(payload), name="rag_research", tool_call_id=tool_call_id
    )


class TestCollectRetrievedContext(unittest.TestCase):
    def test_includes_summary_and_dedupes_tool_messages(self):
        state = {
            "context_summary": "compressed summary",
            "messages": [
                HumanMessage(content="ignored"),
                ToolMessage(content="dupe", name="rag_research", tool_call_id="1"),
                ToolMessage(content="dupe", name="rag_research", tool_call_id="2"),
                ToolMessage(content="other", name="rag_research", tool_call_id="3"),
            ],
        }
        ctx = _collect_retrieved_context(state)
        self.assertIn("Compressed Research Context", ctx)
        self.assertIn("compressed summary", ctx)
        self.assertEqual(ctx.count("Tool Result"), 2)

    def test_no_context_returns_placeholder(self):
        state = {"messages": [HumanMessage(content="hi")]}
        self.assertEqual(
            _collect_retrieved_context(state), "No retrieved context is available."
        )


class TestRetrievalEvidence(unittest.TestCase):
    def test_skips_non_rag_tool_messages(self):
        state = {
            "messages": [
                AIMessage(content="x"),
                ToolMessage(content="{}", name="other_tool", tool_call_id="1"),
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "insufficient")
        self.assertEqual(evidence["reason"], "no_retrieved_context")

    def test_unparseable_tool_result_marks_error(self):
        state = {
            "messages": [
                ToolMessage(content="not json", name="rag_research", tool_call_id="1")
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "error")
        self.assertEqual(evidence["reason"], "retrieval_error")

    def test_sufficient_contexts_above_threshold(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [{"content": "c", "score": 0.9}],
                        "gaps": [],
                        "diagnostics": {
                            "evidence_status": "sufficient",
                            "best_rerank_score": 0.95,
                        },
                    }
                )
            ]
        }
        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "sufficient")
        self.assertEqual(evidence["best_rerank_score"], 0.95)

    def test_invalid_scores_are_ignored(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [{"content": "c", "score": "not-a-number"}],
                        "gaps": [],
                        "diagnostics": {
                            "evidence_status": "sufficient",
                            "best_rerank_score": "bad",
                        },
                    }
                )
            ]
        }
        with patch("config.RERANKER_SCORE_THRESHOLD", None), patch(
            "config.RERANKER_ENABLED", False
        ):
            evidence = _retrieval_evidence(state)
        # contexts present, no usable score, threshold None -> sufficient
        self.assertEqual(evidence["status"], "sufficient")
        self.assertIsNone(evidence["best_rerank_score"])

    def test_contexts_below_explicit_threshold_marks_low_score(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [{"content": "weak", "score": 0.1}],
                        "gaps": [],
                        "diagnostics": {"evidence_status": "sufficient"},
                    }
                )
            ]
        }
        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "low_score")
        self.assertEqual(evidence["reason"], "best_rerank_score_below_threshold")
        self.assertEqual(evidence["best_rerank_score"], 0.1)

    def test_default_threshold_applied_when_reranker_enabled(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [{"content": "weak", "score": -0.2}],
                        "gaps": [],
                        "diagnostics": {"evidence_status": "sufficient"},
                    }
                )
            ]
        }
        with patch("config.RERANKER_SCORE_THRESHOLD", None), patch(
            "config.RERANKER_ENABLED", True
        ):
            evidence = _retrieval_evidence(state)
        # default threshold 0.0 -> -0.2 below it -> low_score
        self.assertEqual(evidence["status"], "low_score")

    def test_low_score_status_without_contexts(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [],
                        "gaps": [],
                        "diagnostics": {"evidence_status": "low_score"},
                    }
                )
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "low_score")
        self.assertEqual(evidence["reason"], "rerank_score_below_threshold")

    def test_diagnostics_error_marks_error(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [],
                        "gaps": [],
                        "diagnostics": {"error": "boom"},
                    }
                )
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "error")
        self.assertEqual(evidence["reason"], "retrieval_error")

    def test_gap_error_prefix_marks_error(self):
        state = {
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [],
                        "gaps": ["RAG_RESEARCH_ERROR: failed"],
                        "diagnostics": {},
                    }
                )
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "error")

    def test_no_contexts_no_errors_is_insufficient(self):
        state = {
            "messages": [
                rag_tool_message(
                    {"contexts": [], "gaps": ["nothing relevant"], "diagnostics": {}}
                )
            ]
        }
        evidence = _retrieval_evidence(state)
        self.assertEqual(evidence["status"], "insufficient")
        self.assertEqual(evidence["reason"], "no_relevant_document_context")


class TestEvaluateAnswer(unittest.TestCase):
    def _sufficient_state(self, answer="Final answer."):
        return {
            "question": "What is X?",
            "final_answer": answer,
            "answer_mode": "rag_qa",
            "context_summary": "",
            "messages": [
                rag_tool_message(
                    {
                        "contexts": [{"content": "evidence", "score": 0.9}],
                        "gaps": [],
                        "diagnostics": {
                            "evidence_status": "sufficient",
                            "best_rerank_score": 0.9,
                        },
                    }
                )
            ],
        }

    def test_knowledge_fallback_mode_short_circuits(self):
        result = evaluate_answer(
            {"answer_mode": "knowledge_fallback"}, ConfigurableLLM("unused")
        )
        self.assertTrue(result["answer_is_satisfactory"])
        self.assertFalse(result["used_knowledge_base"])
        self.assertEqual(result["answer_evaluation_count"], 1)

    def test_insufficient_evidence_short_circuits_before_llm(self):
        state = {
            "question": "What is X?",
            "final_answer": "answer",
            "answer_mode": "rag_qa",
            "messages": [],
        }
        llm = ConfigurableLLM("should not be called")
        result = evaluate_answer(state, llm)
        self.assertFalse(result["answer_is_satisfactory"])
        self.assertIsNone(llm.invoked_with)  # LLM not invoked

    def test_satisfactory_answer(self):
        llm = ConfigurableLLM(json.dumps({"is_satisfactory": True, "critique": ""}))
        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            result = evaluate_answer(self._sufficient_state(), llm)
        self.assertTrue(result["answer_is_satisfactory"])
        self.assertEqual(result["retrieval_evidence_status"], "sufficient")
        self.assertEqual(result["best_rerank_score"], 0.9)
        self.assertEqual(llm.config_kwargs, {"temperature": 0})
        self.assertIsInstance(llm.invoked_with[0], SystemMessage)

    def test_unsatisfactory_answer_with_details(self):
        llm = ConfigurableLLM(
            json.dumps(
                {
                    "is_satisfactory": False,
                    "critique": "Missing the year.",
                    "missing_information": ["the founding year"],
                    "suggested_search_queries": ["founding year of X"],
                }
            )
        )
        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            result = evaluate_answer(self._sufficient_state(), llm)
        self.assertFalse(result["answer_is_satisfactory"])
        self.assertEqual(result["retrieval_evidence_status"], "insufficient")
        self.assertEqual(result["fallback_reason"], "Missing the year.")
        feedback = result["messages"][0].content
        self.assertIn("the founding year", feedback)
        self.assertIn("founding year of X", feedback)

    def test_unsatisfactory_answer_uses_default_text_when_empty(self):
        llm = ConfigurableLLM(
            json.dumps(
                {
                    "is_satisfactory": False,
                    "critique": "",
                    "missing_information": [],
                    "suggested_search_queries": [],
                }
            )
        )
        with patch("config.RERANKER_SCORE_THRESHOLD", 0.5):
            result = evaluate_answer(self._sufficient_state(), llm)
        feedback = result["messages"][0].content
        self.assertIn("Unspecified gaps", feedback)
        self.assertIn("Rephrase the original question", feedback)
        self.assertEqual(result["fallback_reason"], "llm_judged_evidence_insufficient")


class TestParseAnswerEvaluation(unittest.TestCase):
    def test_clean_json(self):
        result = _parse_answer_evaluation(
            json.dumps(
                {
                    "is_satisfactory": True,
                    "critique": "good",
                    "missing_information": ["a"],
                    "suggested_search_queries": ["b"],
                }
            )
        )
        self.assertTrue(result.is_satisfactory)
        self.assertEqual(result.critique, "good")
        self.assertEqual(result.missing_information, ["a"])

    def test_regex_fallback_extracts_embedded_json(self):
        content = "Here is my verdict:\n{\"is_satisfactory\": false, \"critique\": \"x\"}\nThanks"
        result = _parse_answer_evaluation(content)
        self.assertFalse(result.is_satisfactory)
        self.assertEqual(result.critique, "x")

    def test_no_json_returns_defaults(self):
        result = _parse_answer_evaluation("no json here at all")
        self.assertFalse(result.is_satisfactory)
        self.assertEqual(result.critique, "")
        self.assertEqual(result.missing_information, [])
        self.assertEqual(result.suggested_search_queries, [])


if __name__ == "__main__":
    unittest.main()
