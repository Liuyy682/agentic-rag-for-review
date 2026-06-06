import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessage

from rag_agent.graph import (
    create_agent_graph,
    create_agent_subgraph,
    create_task_executor_subgraph,
)
from rag_agent.nodes.aggregation import aggregate_answers


class FakeLLM:
    """Minimal LLM stub supporting the methods graph assembly relies on."""

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def with_config(self, **kwargs):
        return self

    def invoke(self, messages):
        return AIMessage(content="stub answer")


class TestGraphAssembly(unittest.TestCase):
    def test_create_task_executor_subgraph_compiles(self):
        graph = create_task_executor_subgraph(FakeLLM(), [])
        self.assertIsNotNone(graph)
        node_names = set(graph.get_graph().nodes.keys())
        for expected in {
            "task_executor",
            "tools",
            "fallback_response",
            "knowledge_fallback",
            "collect_answer",
            "evaluate_answer",
        }:
            self.assertIn(expected, node_names)

    def test_create_agent_subgraph_alias(self):
        graph = create_agent_subgraph(FakeLLM(), [])
        self.assertIsNotNone(graph)

    def test_create_agent_graph_compiles(self):
        graph = create_agent_graph(FakeLLM(), [])
        self.assertIsNotNone(graph)
        node_names = set(graph.get_graph().nodes.keys())
        for expected in {
            "summarize_history",
            "recognize_intent",
            "rewrite_query",
            "request_clarification",
            "chitchat_response",
            "unsupported_response",
            "plan_rag_tasks",
            "task_executor",
            "aggregate_answers",
        }:
            self.assertIn(expected, node_names)


class TestAggregateAnswers(unittest.TestCase):
    def test_no_answers_returns_placeholder(self):
        result = aggregate_answers({}, FakeLLM())
        self.assertEqual(result["messages"][0].content, "No answers were generated.")

    def test_aggregates_sorted_answers(self):
        captured = {}

        class CaptureLLM(FakeLLM):
            def invoke(self, messages):
                captured["messages"] = messages
                return AIMessage(content="combined answer")

        state = {
            "originalQuery": "Tell me about X and Y",
            "conversation_memory": "prior",
            "task_results": [
                {
                    "index": 1,
                    "question": "What is Y?",
                    "answer": "Y answer",
                    "answer_mode": "rag_qa",
                    "rag_task_type": "fact_qa",
                    "sources": ["y.md"],
                },
                {
                    "index": 0,
                    "question": "What is X?",
                    "answer": "X answer",
                    "answer_mode": "knowledge_fallback",
                    "sources": [],
                },
            ],
        }

        result = aggregate_answers(state, CaptureLLM())

        self.assertEqual(result["messages"][0].content, "combined answer")
        user_content = captured["messages"][1].content
        # sorted by index -> X answer (0) before Y answer (1)
        self.assertLess(user_content.index("X answer"), user_content.index("Y answer"))
        self.assertIn("Conversation memory:", user_content)


if __name__ == "__main__":
    unittest.main()
