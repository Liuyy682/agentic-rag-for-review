import sys
import unittest
from pathlib import Path


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from rag_agent.nodes.execution import (
    _sources_from_answer,
    collect_answer,
    fallback_response,
    knowledge_fallback_answer,
    task_executor,
)


class FakeToolLLM:
    """Mimics an LLM bound with tools: invoke returns a preset AIMessage."""

    def __init__(self, response):
        self.response = response
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return self.response


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return AIMessage(content=self.content)


def tool_call(name="rag_research"):
    return {"name": name, "args": {"query": "q"}, "id": "call_1"}


class TestTaskExecutor(unittest.TestCase):
    def test_initial_research_forces_search(self):
        response = AIMessage(
            content="",
            tool_calls=[tool_call("rag_research"), tool_call("other_tool")],
        )
        llm = FakeToolLLM(response)
        state = {
            "question": "What is caching?",
            "messages": [],
            "context_summary": "prior summary",
            "task_context": "task ctx",
            "rag_task_type": "fact_qa",
        }

        result = task_executor(state, llm)

        # First message is the original question, second the LLM response.
        self.assertEqual(result["messages"][0].content, "What is caching?")
        self.assertIs(result["messages"][1], response)
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual(result["search_call_count"], 1)  # only rag_research counts
        self.assertEqual(result["iteration_count"], 1)
        # Force-search message must be included in the prompt.
        contents = [getattr(m, "content", "") for m in llm.invoked_with]
        self.assertTrue(any("YOU MUST CALL `rag_research`" in c for c in contents))
        self.assertTrue(any(isinstance(m, SystemMessage) for m in llm.invoked_with))
        self.assertTrue(any("COMPRESSED CONTEXT" in c for c in contents))
        self.assertTrue(any("[TASK CONTEXT]" in c for c in contents))

    def test_initial_research_without_summary_or_context(self):
        response = AIMessage(content="answer text", tool_calls=[])
        llm = FakeToolLLM(response)
        state = {"question": "Q?", "messages": []}

        result = task_executor(state, llm)

        self.assertEqual(result["tool_call_count"], 0)
        self.assertEqual(result["search_call_count"], 0)
        contents = [getattr(m, "content", "") for m in llm.invoked_with]
        self.assertFalse(any("COMPRESSED CONTEXT" in c for c in contents))
        self.assertFalse(any("[TASK CONTEXT]" in c for c in contents))

    def test_followup_iteration_uses_existing_messages(self):
        response = AIMessage(
            content="refined",
            tool_calls=[tool_call("rag_research")],
        )
        llm = FakeToolLLM(response)
        state = {
            "question": "Q?",
            "messages": [
                HumanMessage(content="Q?"),
                AIMessage(content="", tool_calls=[tool_call()]),
                ToolMessage(content="evidence", tool_call_id="call_1", name="rag_research"),
            ],
            "iteration_count": 1,
        }

        result = task_executor(state, llm)

        self.assertEqual(result["messages"], [response])
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["search_call_count"], 1)
        self.assertEqual(result["iteration_count"], 1)

    def test_followup_iteration_with_no_tool_calls(self):
        # Response object without tool_calls attribute path / empty content.
        response = AIMessage(content="", tool_calls=[])
        llm = FakeToolLLM(response)
        state = {
            "question": "Q?",
            "messages": [HumanMessage(content="Q?"), AIMessage(content="prev")],
        }

        result = task_executor(state, llm)

        self.assertEqual(result["tool_call_count"], 0)
        self.assertEqual(result["search_call_count"], 0)


class TestFallbackResponse(unittest.TestCase):
    def test_dedupes_tool_messages_and_includes_summary(self):
        llm = FakeLLM("fallback answer")
        state = {
            "question": "Explain X.",
            "context_summary": "compressed ctx",
            "messages": [
                HumanMessage(content="Explain X."),
                ToolMessage(content="dataA", tool_call_id="1", name="rag_research"),
                ToolMessage(content="dataA", tool_call_id="2", name="rag_research"),  # dup
                ToolMessage(content="dataB", tool_call_id="3", name="rag_research"),
            ],
        }

        result = fallback_response(state, llm)

        self.assertEqual(result["messages"][0].content, "fallback answer")
        human_msg = llm.invoked_with[1]
        self.assertIn("Compressed Research Context", human_msg.content)
        self.assertIn("compressed ctx", human_msg.content)
        self.assertIn("DATA SOURCE 1", human_msg.content)
        self.assertIn("DATA SOURCE 2", human_msg.content)
        self.assertNotIn("DATA SOURCE 3", human_msg.content)  # only 2 unique

    def test_no_data_no_summary(self):
        llm = FakeLLM("fallback answer")
        state = {"question": "Explain X.", "messages": [HumanMessage(content="Explain X.")]}

        result = fallback_response(state, llm)

        self.assertEqual(result["messages"][0].content, "fallback answer")
        human_msg = llm.invoked_with[1]
        self.assertIn("No data was retrieved from the documents.", human_msg.content)


class TestKnowledgeFallbackAnswer(unittest.TestCase):
    def test_uses_retrieval_evidence_status_when_no_fallback_reason(self):
        llm = FakeLLM("general answer")
        state = {
            "question": "Explain caching.",
            "retrieval_evidence_status": "insufficient",
            "answer_evaluation_count": 1,
        }

        result = knowledge_fallback_answer(state, llm)

        self.assertEqual(result["answer_mode"], "knowledge_fallback")
        self.assertFalse(result["used_knowledge_base"])
        self.assertTrue(result["fallback_triggered"])
        self.assertEqual(result["fallback_reason"], "insufficient")
        self.assertEqual(result["retry_count_before_fallback"], 1)

    def test_defaults_fallback_reason_when_nothing_set(self):
        llm = FakeLLM("general answer")
        result = knowledge_fallback_answer({"question": "Q?"}, llm)
        self.assertEqual(result["fallback_reason"], "knowledge_base_unavailable")
        self.assertEqual(result["retry_count_before_fallback"], 0)


class TestSourcesFromAnswer(unittest.TestCase):
    def test_no_sources_section(self):
        self.assertEqual(_sources_from_answer("just an answer"), [])

    def test_parses_and_dedupes_sorted(self):
        answer = (
            "Body text.\n\n"
            "**Sources:**\n"
            "- file_b.md\n"
            "- file_a.md\n"
            "- file_a.md\n"
            "- internal_chunk_id\n"  # no dot -> dropped
        )
        self.assertEqual(_sources_from_answer(answer), ["file_a.md", "file_b.md"])


class TestCollectAnswer(unittest.TestCase):
    def test_valid_rag_answer_with_sources(self):
        answer_text = "Answer body.\n\n**Sources:**\n- doc.md\n"
        state = {
            "question_index": 0,
            "question": "Q?",
            "messages": [AIMessage(content=answer_text)],
            "rag_task_type": "fact_qa",
        }

        result = collect_answer(state)

        self.assertEqual(result["final_answer"], answer_text)
        self.assertEqual(result["answer_mode"], "rag_qa")
        self.assertTrue(result["used_knowledge_base"])
        task_result = result["task_results"][0]
        self.assertEqual(task_result["task_id"], "task_1")
        self.assertEqual(task_result["sources"], ["doc.md"])
        self.assertEqual(result["agent_answers"][0]["sources"], ["doc.md"])

    def test_invalid_answer_when_last_message_has_tool_calls(self):
        state = {
            "question_index": 2,
            "question": "Q?",
            "task_id": "custom_task",
            "messages": [
                AIMessage(
                    content="partial",
                    tool_calls=[{"name": "rag_research", "args": {}, "id": "1"}],
                )
            ],
        }

        result = collect_answer(state)

        self.assertEqual(result["final_answer"], "Unable to generate an answer.")
        self.assertEqual(result["task_results"][0]["task_id"], "custom_task")

    def test_knowledge_fallback_answer_drops_sources(self):
        state = {
            "question_index": 0,
            "question": "Q?",
            "messages": [AIMessage(content="No usable info.\n\n**Sources:**\n- doc.md\n")],
            "answer_mode": "knowledge_fallback",
            "used_knowledge_base": False,
        }

        result = collect_answer(state)

        self.assertEqual(result["task_results"][0]["sources"], [])
        self.assertFalse(result["used_knowledge_base"])


if __name__ == "__main__":
    unittest.main()
