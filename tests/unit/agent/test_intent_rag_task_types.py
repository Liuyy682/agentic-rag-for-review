import json
import sys
import unittest
from pathlib import Path


from langchain_core.messages import AIMessage, HumanMessage

from rag_agent.edges import route_after_intent, route_after_task_planning
from rag_agent.nodes.intent import (
    _parse_intent_analysis,
    plan_rag_tasks,
    recognize_intent,
    rewrite_query,
)


class CaptureLLM:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def with_config(self, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


class TestIntentRagTaskTypes(unittest.TestCase):
    def test_parse_rag_task_type_summarization(self):
        result = _parse_intent_analysis(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "rag_task_type": "summarization",
                    "is_clear": True,
                    "original_query": "Summarize chapter 1",
                    "normalized_query": "Summarize chapter 1",
                    "clarification_needed": "",
                    "tasks": [],
                }
            ),
            "fallback",
        )

        self.assertEqual(result.intent_type, "rag_qa")
        self.assertEqual(result.rag_task_type, "summarization")

    def test_parse_rag_task_type_defaults_when_missing(self):
        result = _parse_intent_analysis(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "is_clear": True,
                    "original_query": "What is caching?",
                    "normalized_query": "What is caching?",
                    "clarification_needed": "",
                    "tasks": [],
                }
            ),
            "fallback",
        )

        self.assertEqual(result.rag_task_type, "fact_qa")

    def test_parse_rag_task_type_defaults_when_invalid(self):
        result = _parse_intent_analysis(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "rag_task_type": "unsupported_subtype",
                    "is_clear": True,
                    "original_query": "What is caching?",
                    "normalized_query": "What is caching?",
                    "clarification_needed": "",
                    "tasks": [],
                }
            ),
            "fallback",
        )

        self.assertEqual(result.rag_task_type, "fact_qa")

    def test_parse_unsupported_intent(self):
        result = _parse_intent_analysis(
            json.dumps(
                {
                    "intent_type": "unsupported",
                    "rag_task_type": "fact_qa",
                    "is_clear": True,
                    "original_query": "Book me a flight",
                    "normalized_query": "Book me a flight",
                    "clarification_needed": "",
                    "tasks": [],
                }
            ),
            "fallback",
        )

        self.assertEqual(result.intent_type, "unsupported")

    def test_route_after_intent_includes_unsupported(self):
        self.assertEqual(
            route_after_intent({"intent_type": "rag_qa", "questionIsClear": True}),
            "rewrite_query",
        )
        self.assertEqual(
            route_after_intent({"intent_type": "clarification", "questionIsClear": False}),
            "request_clarification",
        )
        self.assertEqual(
            route_after_intent({"intent_type": "chitchat", "questionIsClear": True}),
            "chitchat_response",
        )
        self.assertEqual(
            route_after_intent({"intent_type": "unsupported", "questionIsClear": True}),
            "unsupported_response",
        )

    def test_recognize_intent_returns_rag_task_type(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "rag_task_type": "comparison",
                    "is_clear": True,
                    "original_query": "Compare A and B",
                    "normalized_query": "Compare A and B",
                    "clarification_needed": "",
                    "tasks": [],
                }
            )
        )
        state = {"messages": [HumanMessage(content="Compare A and B")]}

        result = recognize_intent(state, llm)

        self.assertEqual(result["intent_type"], "rag_qa")
        self.assertEqual(result["rag_task_type"], "comparison")

    def test_rewrite_query_propagates_rag_task_type(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "is_clear": True,
                    "questions": ["Compare A and B from the documents"],
                    "clarification_needed": "",
                }
            )
        )
        state = {
            "messages": [HumanMessage(content="Compare them")],
            "originalQuery": "Compare them",
            "normalized_query": "Compare A and B",
            "rag_task_type": "comparison",
        }

        result = rewrite_query(state, llm)

        self.assertEqual(result["task_plan"][0]["rag_task_type"], "comparison")
        self.assertIn("RAG Task Type:\ncomparison", llm.messages[1].content)

    def test_plan_rag_tasks_propagates_rag_task_type(self):
        result = plan_rag_tasks(
            {
                "originalQuery": "Show me the setup steps",
                "normalized_query": "Show me the setup steps",
                "rag_task_type": "how_to",
            }
        )

        self.assertEqual(result["task_plan"][0]["rag_task_type"], "how_to")

    def test_route_after_task_planning_propagates_rag_task_type(self):
        sends = route_after_task_planning(
            {
                "originalQuery": "Summarize chapter 1",
                "task_plan": [
                    {
                        "task_id": "task_1",
                        "task_type": "rag_qa",
                        "rag_task_type": "summarization",
                        "query": "Summarize chapter 1",
                        "original_query": "Summarize chapter 1",
                    }
                ],
            }
        )

        self.assertEqual(sends[0].arg["rag_task_type"], "summarization")


if __name__ == "__main__":
    unittest.main()
