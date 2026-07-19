import json
import sys
import unittest
from pathlib import Path


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag_agent.nodes.intent import (
    _conversation_context,
    _parse_intent_analysis,
    _parse_query_analysis,
    chitchat_response,
    plan_rag_tasks,
    recognize_intent,
    request_clarification,
    rewrite_query,
    unsupported_response,
)


class CaptureLLM:
    def __init__(self, content):
        self.content = content
        self.messages = None
        self.config_kwargs = None

    def with_config(self, **kwargs):
        self.config_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


class TestParseIntentAnalysis(unittest.TestCase):
    def test_regex_fallback_when_not_pure_json(self):
        content = 'prefix {"intent_type": "rag_qa", "is_clear": true} suffix'
        result = _parse_intent_analysis(content, "fallback query")
        self.assertEqual(result.intent_type, "rag_qa")

    def test_no_json_defaults_to_clarification(self):
        result = _parse_intent_analysis("no json", "fallback query")
        self.assertEqual(result.intent_type, "clarification")
        self.assertEqual(result.normalized_query, "fallback query")

    def test_invalid_intent_type_defaults_clarification(self):
        result = _parse_intent_analysis(
            json.dumps({"intent_type": "weird", "is_clear": True}), "fb"
        )
        self.assertEqual(result.intent_type, "clarification")

    def test_tasks_non_list_coerced_to_empty(self):
        result = _parse_intent_analysis(
            json.dumps({"intent_type": "rag_qa", "is_clear": True, "tasks": "not-a-list"}),
            "fb",
        )
        self.assertEqual(result.tasks, [])

    def test_invalid_rag_task_type_defaults_fact_qa(self):
        result = _parse_intent_analysis(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "rag_task_type": "bogus_subtype",
                    "is_clear": True,
                }
            ),
            "fb",
        )
        self.assertEqual(result.rag_task_type, "fact_qa")


class TestParseQueryAnalysis(unittest.TestCase):
    def test_regex_fallback(self):
        content = 'noise {"is_clear": true, "questions": ["q1"]} trailing'
        result = _parse_query_analysis(content, "fb")
        self.assertEqual(result.questions, ["q1"])

    def test_no_json_uses_fallback_query(self):
        result = _parse_query_analysis("garbage", "fallback q")
        self.assertEqual(result.questions, ["fallback q"])

    def test_questions_as_string_wrapped_in_list(self):
        result = _parse_query_analysis(
            json.dumps({"is_clear": True, "questions": "single question"}), "fb"
        )
        self.assertEqual(result.questions, ["single question"])

    def test_questions_non_list_non_str_coerced(self):
        result = _parse_query_analysis(
            json.dumps({"is_clear": True, "questions": 123}), "fb query"
        )
        self.assertEqual(result.questions, ["fb query"])


class TestConversationContext(unittest.TestCase):
    def test_includes_memory_and_summary(self):
        ctx = _conversation_context(
            {"conversation_memory": "mem block", "conversation_summary": "sum block"}
        )
        self.assertIn("Conversation Memory:\nmem block", ctx)
        self.assertIn("Conversation Summary:\nsum block", ctx)

    def test_empty_when_nothing(self):
        self.assertEqual(_conversation_context({}), "")


class TestRecognizeIntent(unittest.TestCase):
    def test_clarification_followup_combines_original_query(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "rag_qa",
                    "rag_task_type": "fact_qa",
                    "is_clear": True,
                    "original_query": "ignored",
                    "normalized_query": "What is the API rate limit for plan A?",
                    "clarification_needed": "",
                    "tasks": [],
                }
            )
        )
        state = {
            "messages": [HumanMessage(content="Plan A")],
            "questionIsClear": False,
            "clarification_needed": "Which plan are you asking about?",
            "originalQuery": "What is the rate limit?",
            "conversation_summary": "discussing API plans",
        }

        result = recognize_intent(state, llm)

        self.assertEqual(result["intent_type"], "rag_qa")
        # original_query preserved from prior originalQuery during followup
        self.assertEqual(result["originalQuery"], "What is the rate limit?")
        prompt = llm.messages[1].content
        self.assertIn("asked for clarification", prompt)
        self.assertIn("What is the rate limit?", prompt)

    def test_unsupported_intent_branch(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "unsupported",
                    "is_clear": True,
                    "original_query": "Book a flight",
                    "normalized_query": "Book a flight",
                    "clarification_needed": "",
                    "tasks": [],
                }
            )
        )
        state = {"messages": [HumanMessage(content="Book a flight")]}

        result = recognize_intent(state, llm)

        self.assertEqual(result["intent_type"], "unsupported")
        self.assertEqual(result["rag_task_type"], "")
        self.assertTrue(result["questionIsClear"])
        self.assertEqual(result["task_results"], [{"__reset__": True}])

    def test_chitchat_intent_branch(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "chitchat",
                    "is_clear": True,
                    "original_query": "hello",
                    "normalized_query": "hello",
                    "clarification_needed": "",
                    "tasks": [],
                }
            )
        )
        state = {"messages": [HumanMessage(content="hello")]}

        result = recognize_intent(state, llm)

        self.assertEqual(result["intent_type"], "chitchat")
        self.assertEqual(result["rag_task_type"], "")

    def test_clarification_uses_long_llm_clarification(self):
        clar = "Could you tell me which specific document you are referring to?"
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "clarification",
                    "is_clear": False,
                    "original_query": "it",
                    "normalized_query": "it",
                    "clarification_needed": clar,
                    "tasks": [],
                }
            )
        )
        state = {"messages": [HumanMessage(content="it")]}

        result = recognize_intent(state, llm)

        self.assertFalse(result["questionIsClear"])
        self.assertEqual(result["intent_type"], "clarification")
        self.assertEqual(result["clarification_needed"], clar)
        self.assertEqual(result["messages"][0].content, clar)

    def test_clarification_uses_default_when_too_short(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "clarification",
                    "is_clear": False,
                    "original_query": "it",
                    "normalized_query": "it",
                    "clarification_needed": "huh?",
                    "tasks": [],
                }
            )
        )
        state = {"messages": [HumanMessage(content="it")]}

        result = recognize_intent(state, llm)

        self.assertEqual(
            result["clarification_needed"],
            "I need more information to understand your question.",
        )


class TestRewriteQuery(unittest.TestCase):
    def test_not_clear_requests_clarification(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "is_clear": False,
                    "questions": [],
                    "clarification_needed": "Which version do you mean?",
                }
            )
        )
        state = {"normalized_query": "the thing", "originalQuery": "the thing"}

        result = rewrite_query(state, llm)

        self.assertFalse(result["questionIsClear"])
        self.assertEqual(result["clarification_needed"], "Which version do you mean?")
        self.assertEqual(result["messages"][0].content, "Which version do you mean?")
        self.assertEqual(result["task_plan"], [])

    def test_not_clear_uses_default_clarification(self):
        llm = CaptureLLM(
            json.dumps({"is_clear": False, "questions": [], "clarification_needed": ""})
        )
        state = {"normalized_query": "x", "originalQuery": "x"}

        result = rewrite_query(state, llm)

        self.assertEqual(
            result["clarification_needed"],
            "I need more information to search the documents.",
        )

    def test_clear_builds_task_plan_limited_to_three(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "is_clear": True,
                    "questions": ["q1", "q2", "q3", "q4"],
                    "clarification_needed": "",
                }
            )
        )
        state = {
            "normalized_query": "multi",
            "originalQuery": "multi",
            "rag_task_type": "comparison",
            "conversation_summary": "ctx",
        }

        result = rewrite_query(state, llm)

        self.assertTrue(result["questionIsClear"])
        self.assertEqual(len(result["task_plan"]), 3)
        self.assertEqual(result["rewrittenQuestions"], ["q1", "q2", "q3"])
        self.assertEqual(result["task_plan"][0]["rag_task_type"], "comparison")
        self.assertEqual(result["task_plan"][0]["context"], "ctx")


class TestRequestClarification(unittest.TestCase):
    def test_returns_empty_dict(self):
        self.assertEqual(request_clarification({}), {})


class TestPlanRagTasks(unittest.TestCase):
    def test_existing_task_plan_extracts_questions(self):
        result = plan_rag_tasks(
            {"task_plan": [{"query": "q1"}, {"query": "q2"}, {"no_query": True}]}
        )
        self.assertEqual(result, {"rewrittenQuestions": ["q1", "q2"]})

    def test_no_query_returns_empty_plan(self):
        result = plan_rag_tasks({"task_plan": [], "normalized_query": "", "originalQuery": ""})
        self.assertEqual(result, {"task_plan": []})

    def test_builds_single_task_from_query(self):
        result = plan_rag_tasks(
            {
                "task_plan": [],
                "normalized_query": "What is X?",
                "originalQuery": "What is X?",
                "rag_task_type": "how_to",
            }
        )
        self.assertEqual(result["task_plan"][0]["rag_task_type"], "how_to")
        self.assertEqual(result["rewrittenQuestions"], ["What is X?"])


class TestChitchatResponse(unittest.TestCase):
    def test_responds_with_context(self):
        llm = CaptureLLM("Hi there!")
        state = {
            "normalized_query": "hello",
            "conversation_memory": "prior chat",
            "messages": [HumanMessage(content="hello")],
        }

        result = chitchat_response(state, llm)

        self.assertEqual(result["messages"][0].content, "Hi there!")
        self.assertIsInstance(llm.messages[0], SystemMessage)
        self.assertIn("Conversation Context", llm.messages[1].content)

    def test_responds_without_context_falls_back_to_last_message(self):
        llm = CaptureLLM("Hello!")
        state = {"messages": [HumanMessage(content="hi")]}

        result = chitchat_response(state, llm)

        self.assertEqual(result["messages"][0].content, "Hello!")
        self.assertNotIn("Conversation Context", llm.messages[1].content)


class TestUnsupportedResponse(unittest.TestCase):
    def test_responds(self):
        llm = CaptureLLM("This is unsupported.")
        state = {"normalized_query": "book flight", "messages": [HumanMessage(content="book flight")]}

        result = unsupported_response(state, llm)

        self.assertEqual(result["messages"][0].content, "This is unsupported.")
        self.assertIn("User Query:\nbook flight", llm.messages[1].content)

    def test_falls_back_to_last_message(self):
        llm = CaptureLLM("Unsupported.")
        state = {"messages": [HumanMessage(content="do magic")]}

        result = unsupported_response(state, llm)

        self.assertIn("do magic", llm.messages[1].content)


if __name__ == "__main__":
    unittest.main()
