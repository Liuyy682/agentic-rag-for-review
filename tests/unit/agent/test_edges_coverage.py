import sys
import unittest
from pathlib import Path


from langchain_core.messages import AIMessage

from agentic_rag import config
from agentic_rag.agent.edges import (
    route_after_answer_evaluation,
    route_after_intent,
    route_after_rewrite,
    route_after_task_executor_call,
    route_after_task_planning,
)


class TestRouteAfterIntent(unittest.TestCase):
    def test_chitchat(self):
        self.assertEqual(route_after_intent({"intent_type": "chitchat"}), "chitchat_response")

    def test_unsupported(self):
        self.assertEqual(route_after_intent({"intent_type": "unsupported"}), "unsupported_response")

    def test_clarification(self):
        self.assertEqual(
            route_after_intent({"intent_type": "clarification", "questionIsClear": False}),
            "request_clarification",
        )

    def test_not_clear_falls_back_to_clarification(self):
        self.assertEqual(
            route_after_intent({"intent_type": "rag_qa", "questionIsClear": False}),
            "request_clarification",
        )

    def test_clear_rag_routes_to_rewrite(self):
        self.assertEqual(
            route_after_intent({"intent_type": "rag_qa", "questionIsClear": True}),
            "rewrite_query",
        )


class TestRouteAfterRewrite(unittest.TestCase):
    def test_not_clear_routes_to_clarification(self):
        self.assertEqual(route_after_rewrite({"questionIsClear": False}), "request_clarification")

    def test_clear_routes_to_plan(self):
        self.assertEqual(route_after_rewrite({"questionIsClear": True}), "plan_rag_tasks")

    def test_missing_flag_defaults_to_clarification(self):
        self.assertEqual(route_after_rewrite({}), "request_clarification")


class TestRouteAfterTaskPlanning(unittest.TestCase):
    def test_builds_sends_for_each_task_with_query(self):
        sends = route_after_task_planning(
            {
                "originalQuery": "orig",
                "rag_task_type": "fact_qa",
                "task_plan": [
                    {"task_id": "task_1", "query": "q1", "original_query": "orig"},
                    {"query": "q2"},
                    {"query": ""},  # skipped: no query
                ],
            }
        )
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0].node, "task_executor")
        self.assertEqual(sends[0].arg["task_id"], "task_1")
        self.assertEqual(sends[0].arg["question"], "q1")
        self.assertEqual(sends[0].arg["question_index"], 0)
        # second task has no explicit task_id -> derived from index
        self.assertEqual(sends[1].arg["task_id"], "task_2")
        self.assertEqual(sends[1].arg["rag_task_type"], "fact_qa")
        self.assertEqual(sends[1].arg["messages"], [])

    def test_empty_plan_returns_empty(self):
        self.assertEqual(route_after_task_planning({"task_plan": []}), [])


class TestRouteAfterTaskExecutorCall(unittest.TestCase):
    def test_iteration_limit_routes_to_fallback(self):
        state = {
            "iteration_count": config.MAX_ITERATIONS,
            "tool_call_count": 0,
            "messages": [AIMessage(content="x")],
        }
        self.assertEqual(route_after_task_executor_call(state), "fallback_response")

    def test_tool_limit_routes_to_fallback(self):
        state = {
            "iteration_count": 1,
            "tool_call_count": config.MAX_TOOL_CALLS + 1,
            "messages": [AIMessage(content="x")],
        }
        self.assertEqual(route_after_task_executor_call(state), "fallback_response")

    def test_no_tool_calls_routes_to_collect_answer(self):
        state = {
            "iteration_count": 1,
            "tool_call_count": 0,
            "messages": [AIMessage(content="final answer")],
        }
        self.assertEqual(route_after_task_executor_call(state), "collect_answer")

    def test_tool_calls_route_to_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "rag_research", "args": {"query": "q"}, "id": "1"}],
        )
        state = {"iteration_count": 1, "tool_call_count": 1, "messages": [msg]}
        self.assertEqual(route_after_task_executor_call(state), "tools")


class TestRouteAfterAnswerEvaluation(unittest.TestCase):
    def test_knowledge_fallback_mode_ends(self):
        self.assertEqual(
            route_after_answer_evaluation({"answer_mode": "knowledge_fallback"}), "__end__"
        )

    def test_satisfactory_ends(self):
        self.assertEqual(
            route_after_answer_evaluation({"answer_is_satisfactory": True}), "__end__"
        )

    def test_retry_limit_routes_to_knowledge_fallback(self):
        state = {
            "answer_is_satisfactory": False,
            "answer_evaluation_count": config.MAX_ANSWER_EVALUATION_RETRIES,
        }
        self.assertEqual(route_after_answer_evaluation(state), "knowledge_fallback")

    def test_iteration_limit_routes_to_knowledge_fallback(self):
        state = {
            "answer_is_satisfactory": False,
            "answer_evaluation_count": 0,
            "iteration_count": config.MAX_ITERATIONS,
            "tool_call_count": 0,
        }
        self.assertEqual(route_after_answer_evaluation(state), "knowledge_fallback")

    def test_default_routes_back_to_task_executor(self):
        state = {
            "answer_is_satisfactory": False,
            "answer_evaluation_count": 0,
            "iteration_count": 1,
            "tool_call_count": 1,
        }
        self.assertEqual(route_after_answer_evaluation(state), "task_executor")


if __name__ == "__main__":
    unittest.main()
