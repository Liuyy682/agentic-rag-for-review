import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_agent.nodes.history import summarize_history


class ConfigurableLLM:
    """FakeLLM exposing with_config()/invoke() that records the call."""

    def __init__(self, content):
        self.content = content
        self.invoked_messages = None
        self.config_kwargs = None

    def with_config(self, **kwargs):
        self.config_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.invoked_messages = messages
        return AIMessage(content=self.content)


class TestSummarizeHistory(unittest.TestCase):
    def test_short_history_returns_empty_summary(self):
        state = {"messages": [HumanMessage(content="hi"), AIMessage(content="hello")]}
        result = summarize_history(state, ConfigurableLLM("unused"))
        self.assertEqual(result, {"conversation_summary": ""})

    def test_no_relevant_messages_returns_empty_summary(self):
        # Four messages but none are clean Human/AI text (all tool-related).
        tool_call_ai = AIMessage(
            content="",
            tool_calls=[{"name": "rag_research", "args": {}, "id": "1"}],
        )
        state = {
            "messages": [
                ToolMessage(content="t1", tool_call_id="1", name="rag_research"),
                ToolMessage(content="t2", tool_call_id="2", name="rag_research"),
                tool_call_ai,
                HumanMessage(content="latest"),  # excluded by [:-1] slice
            ]
        }
        result = summarize_history(state, ConfigurableLLM("unused"))
        self.assertEqual(result, {"conversation_summary": ""})

    def test_summarizes_and_resets_results(self):
        llm = ConfigurableLLM("Discussed caching strategies.")
        state = {
            "messages": [
                HumanMessage(content="What is caching?"),
                AIMessage(content="Caching stores data."),
                HumanMessage(content="Why use it?"),
                AIMessage(content="For speed."),
                HumanMessage(content="latest question"),
            ]
        }
        result = summarize_history(state, llm)

        self.assertEqual(result["conversation_summary"], "Discussed caching strategies.")
        self.assertEqual(result["task_results"], [{"__reset__": True}])
        self.assertEqual(result["agent_answers"], [{"__reset__": True}])
        # temperature passed through with_config
        self.assertEqual(llm.config_kwargs, {"temperature": 0.2})
        # System prompt + conversation human message
        self.assertIsInstance(llm.invoked_messages[0], SystemMessage)
        self.assertIsInstance(llm.invoked_messages[1], HumanMessage)
        self.assertIn("User: What is caching?", llm.invoked_messages[1].content)
        self.assertIn("Assistant: Caching stores data.", llm.invoked_messages[1].content)
        # The latest message (state["messages"][-1]) is excluded from the summary input.
        self.assertNotIn("latest question", llm.invoked_messages[1].content)


if __name__ == "__main__":
    unittest.main()
