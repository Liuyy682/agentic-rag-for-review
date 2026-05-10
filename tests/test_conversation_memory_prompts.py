import json
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessage, HumanMessage

from rag_agent.nodes.aggregation import aggregate_answers
from rag_agent.nodes.intent import recognize_intent, rewrite_query


class CaptureLLM:
    def __init__(self, content):
        self.content = content
        self.messages = None

    def with_config(self, **kwargs):
        return self

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.content)


class TestConversationMemoryPrompts(unittest.TestCase):
    def test_recognize_intent_includes_conversation_memory(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "intent_type": "follow_up",
                    "is_clear": True,
                    "original_query": "What about it?",
                    "normalized_query": "What about caching?",
                    "clarification_needed": "",
                    "follow_up_context": "caching",
                    "tasks": [],
                }
            )
        )
        state = {
            "messages": [HumanMessage(content="What about it?")],
            "conversation_memory": "MEMORY BLOCK",
            "conversation_summary": "",
        }

        result = recognize_intent(state, llm)

        self.assertEqual(result["intent_type"], "follow_up")
        self.assertIn("MEMORY BLOCK", llm.messages[1].content)

    def test_rewrite_query_includes_conversation_memory(self):
        llm = CaptureLLM(
            json.dumps(
                {
                    "is_clear": True,
                    "questions": ["Explain caching in this course"],
                    "clarification_needed": "",
                }
            )
        )
        state = {
            "messages": [HumanMessage(content="Explain it")],
            "originalQuery": "Explain it",
            "normalized_query": "Explain caching",
            "conversation_memory": "MEMORY BLOCK",
            "conversation_summary": "",
        }

        result = rewrite_query(state, llm)

        self.assertEqual(result["rewrittenQuestions"], ["Explain caching in this course"])
        self.assertIn("MEMORY BLOCK", llm.messages[1].content)

    def test_aggregation_includes_memory_without_sources(self):
        llm = CaptureLLM("final")
        state = {
            "originalQuery": "Explain it",
            "conversation_memory": "MEMORY BLOCK",
            "task_results": [
                {
                    "index": 0,
                    "question": "Explain caching",
                    "answer": "Caching stores reusable data.\n---\n**Sources:**\n- source.pdf",
                    "answer_mode": "rag_qa",
                    "used_knowledge_base": True,
                    "sources": ["source.pdf"],
                }
            ],
        }

        result = aggregate_answers(state, llm)

        self.assertEqual(result["messages"][0].content, "final")
        self.assertIn("MEMORY BLOCK", llm.messages[1].content)


if __name__ == "__main__":
    unittest.main()

