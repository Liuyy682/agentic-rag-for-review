import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


from langchain_core.messages import AIMessageChunk, ToolMessage

import agentic_rag.chat.chat_interface as ci
from agentic_rag.chat.chat_interface import (
    ChatInterface,
    make_message,
    find_msg_idx,
    parse_rewrite_json,
    format_rewrite_content,
    format_intent_content,
)


class TestHelpers(unittest.TestCase):
    def test_make_message_with_and_without_metadata(self):
        self.assertNotIn("metadata", make_message("hi"))
        self.assertEqual(make_message("hi", title="T", node="n")["metadata"], {"title": "T", "node": "n"})

    def test_find_msg_idx(self):
        msgs = [{"metadata": {"node": "a"}}, {"metadata": {"node": "b"}}]
        self.assertEqual(find_msg_idx(msgs, "b"), 1)
        self.assertIsNone(find_msg_idx(msgs, "z"))

    def test_parse_rewrite_json(self):
        self.assertEqual(parse_rewrite_json('x {"a": 1} y'), {"a": 1})
        self.assertIsNone(parse_rewrite_json("no json"))
        self.assertIsNone(parse_rewrite_json("{bad json}"))

    def test_format_rewrite_content_branches(self):
        self.assertIn("Rewriting", format_rewrite_content("garbage"))
        ready = format_rewrite_content(json.dumps({"is_clear": True, "questions": ["q1", "q2"]}))
        self.assertIn("ready", ready)
        self.assertIn("q1", ready)
        needs = format_rewrite_content(json.dumps({"is_clear": False, "clarification_needed": "which?"}))
        self.assertIn("clarification", needs.lower())

    def test_format_intent_content_branches(self):
        self.assertIn("Recognizing", format_intent_content("garbage"))
        full = format_intent_content(json.dumps({
            "intent_type": "rag_qa", "rag_task_type": "compare",
            "normalized_query": "nq", "tasks": [{"query": "t1"}],
        }))
        self.assertIn("rag_qa", full)
        self.assertIn("compare", full)
        self.assertIn("t1", full)
        clar = format_intent_content(json.dumps({
            "intent_type": "clarification", "clarification_needed": "which?",
        }))
        self.assertIn("which?", clar)


class FakeMemory:
    def __init__(self):
        self.saved = []
        self.deleted = []
        self.title = None
        self.fail = False

    def get_recent_turns(self, sid, limit=5):
        if self.fail:
            raise RuntimeError("boom")
        return [{"user_original": "u", "assistant_final": "a"}]

    def format_recent_turns(self, turns):
        return "MEMORY"

    def append_turn(self, session_id, user_original, assistant_final, course_name=None):
        if self.fail:
            raise RuntimeError("boom")
        self.saved.append((session_id, user_original, assistant_final, course_name))

    def get_session_turns(self, sid):
        return [{"user_original": "u", "assistant_final": "a"}]

    def update_session_title(self, sid, title):
        self.title = title

    def delete_session(self, sid):
        if self.fail:
            raise RuntimeError("boom")
        self.deleted.append(sid)


class FakeGraph:
    def __init__(self, chunks, next_after=False):
        self._chunks = chunks
        self._next_after = next_after
        self.updated = None

    def get_state(self, config):
        # First call (pre-stream) reports not resuming; controlled separately below.
        return SimpleNamespace(next=self._next_after)

    def stream(self, stream_input, config=None, stream_mode=None):
        yield from self._chunks

    def update_state(self, config, patch):
        self.updated = patch


class FakeRag:
    def __init__(self, graph, threading_ok=True):
        self.agent_graph = graph
        self.thread_id = "s1"
        self.course_scope = None
        self.deleted = None
        self.observability = SimpleNamespace(flush=lambda: None)
        self.llm = SimpleNamespace(invoke=lambda msgs: SimpleNamespace(content="A Nice Title Here"))

    def get_config(self, thread_id=None, *, source_files=None):
        self.course_scope = source_files
        return {"configurable": {"thread_id": thread_id or self.thread_id, "course_scope_sources": tuple(source_files or [])}}

    def reset_thread(self, thread_id=None):
        self.deleted = thread_id


def _ai(content, node):
    return AIMessageChunk(content=content), {"langgraph_node": node}


class TestChatGenerator(unittest.TestCase):
    def test_not_initialized(self):
        rag = FakeRag(None)
        rag.agent_graph = None
        chat = ChatInterface(rag, session_memory=FakeMemory())
        out = list(chat.chat("hi", history=[]))
        self.assertEqual(out, ["⚠️ System not initialized!"])

    def test_full_stream_saves_turn(self):
        # System node (intent), tool call + result, then a final LLM token.
        tool_chunk = SimpleNamespace(tool_calls=[{"id": "t1", "name": "search"}])
        chunks = [
            _ai(json.dumps({"intent_type": "rag_qa"}), "recognize_intent"),
            (tool_chunk, {"langgraph_node": "task_executor"}),
            (ToolMessage(content="x" * 400, tool_call_id="t1"), {"langgraph_node": "tools"}),
            _ai("final answer", "aggregate_answers"),
        ]
        mem = FakeMemory()
        chat = ChatInterface(FakeRag(FakeGraph(chunks)), session_memory=mem)
        out = list(chat.chat("question", history=[]))
        self.assertTrue(mem.saved)
        self.assertEqual(mem.saved[0][2], "final answer")
        # Tool result truncated with ellipsis fence.
        flat = out[-1]
        self.assertTrue(any("```" in m["content"] for m in flat))

    def test_resume_path_updates_state(self):
        graph = FakeGraph([_ai("answer", "aggregate_answers")])
        # Pre-stream snapshot says resuming; post-stream says done.
        states = iter([SimpleNamespace(next=True), SimpleNamespace(next=False)])
        graph.get_state = lambda config: next(states)
        chat = ChatInterface(FakeRag(graph), session_memory=FakeMemory())
        list(chat.chat("more", history=[]))
        self.assertIsNotNone(graph.updated)

    def test_interrupt_returns_early(self):
        graph = FakeGraph([_ai("part", "rewrite_query")])
        states = iter([SimpleNamespace(next=False), SimpleNamespace(next=("request_clarification",))])
        graph.get_state = lambda config: next(states)
        mem = FakeMemory()
        chat = ChatInterface(FakeRag(graph), session_memory=mem)
        list(chat.chat("ambiguous", history=[]))
        self.assertEqual(mem.saved, [])  # no turn saved on interrupt

    def test_clarification_surfaced(self):
        buf = json.dumps({"is_clear": False, "clarification_needed": "which course?"})
        chat = ChatInterface(FakeRag(FakeGraph([_ai(buf, "rewrite_query")])), session_memory=FakeMemory())
        out = list(chat.chat("q", history=[]))
        self.assertTrue(any(m.get("metadata", {}).get("node") == "clarification" for m in out[-1]))

    def test_exception_during_stream(self):
        class BoomGraph(FakeGraph):
            def stream(self, *a, **k):
                raise RuntimeError("kaboom")
                yield
        chat = ChatInterface(FakeRag(BoomGraph([])), session_memory=FakeMemory())
        out = list(chat.chat("q", history=[]))
        self.assertIn("Error", out[-1])

    def test_memory_load_and_save_failures_are_swallowed(self):
        mem = FakeMemory()
        mem.fail = True
        chat = ChatInterface(FakeRag(FakeGraph([_ai("final answer", "aggregate_answers")])), session_memory=mem)
        # Should not raise despite memory get/append throwing.
        list(chat.chat("q", history=[]))

    def test_generate_title_async_direct(self):
        mem = FakeMemory()
        chat = ChatInterface(FakeRag(FakeGraph([])), session_memory=mem)
        chat._generate_title_async("s1")
        self.assertEqual(mem.title, "A Nice Title Here")

    def test_clear_session(self):
        mem = FakeMemory()
        rag = FakeRag(FakeGraph([]))
        chat = ChatInterface(rag, session_memory=mem)
        chat.clear_session()
        self.assertEqual(mem.deleted, ["s1"])
        self.assertEqual(rag.deleted, "s1")

    def test_extract_final_response_skips_system_messages(self):
        chat = ChatInterface(FakeRag(FakeGraph([])), session_memory=FakeMemory())
        msgs = [
            {"role": "assistant", "content": "intent", "metadata": {"node": "recognize_intent"}},
            {"role": "assistant", "content": "the answer"},
        ]
        self.assertEqual(chat._extract_final_response(msgs), "the answer")
        self.assertEqual(chat._extract_final_response([]), "")

    def test_course_store_scopes_sources(self):
        course_store = SimpleNamespace(source_files_for_course=lambda name: ["a.md", "b.md"])
        rag = FakeRag(FakeGraph([_ai("final answer", "aggregate_answers")]))
        chat = ChatInterface(rag, course_store=course_store, session_memory=FakeMemory())
        list(chat.chat("q", history=[], course_name="Networks"))
        self.assertEqual(rag.course_scope, ["a.md", "b.md"])

    def test_title_truncated_to_eight_words(self):
        mem = FakeMemory()
        rag = FakeRag(FakeGraph([]))
        rag.llm = SimpleNamespace(
            invoke=lambda msgs: SimpleNamespace(content="one two three four five six seven eight nine ten")
        )
        chat = ChatInterface(rag, session_memory=mem)
        chat._generate_title_async("s1")
        self.assertEqual(len(mem.title.split()), 8)

    def test_title_async_no_turns(self):
        mem = FakeMemory()
        mem.get_session_turns = lambda sid: []
        chat = ChatInterface(FakeRag(FakeGraph([])), session_memory=mem)
        chat._generate_title_async("s1")
        self.assertIsNone(mem.title)

    def test_system_and_clarification_message_update_in_place(self):
        # Two chunks for the same node exercise the find-existing-index update branch.
        buf = json.dumps({"is_clear": False, "clarification_needed": "which?"})
        chunks = [_ai(buf, "rewrite_query"), _ai(buf, "rewrite_query")]
        chat = ChatInterface(FakeRag(FakeGraph(chunks)), session_memory=FakeMemory())
        out = list(chat.chat("q", history=[]))
        nodes = [m.get("metadata", {}).get("node") for m in out[-1]]
        self.assertEqual(nodes.count("rewrite_query"), 1)  # updated, not duplicated
        self.assertEqual(nodes.count("clarification"), 1)

    def test_delete_session_memory_failure_swallowed(self):
        mem = FakeMemory()
        mem.fail = True
        rag = FakeRag(FakeGraph([]))
        chat = ChatInterface(rag, session_memory=mem)
        chat.clear_session("x")  # delete raises internally but is swallowed
        self.assertEqual(rag.deleted, "x")


if __name__ == "__main__":
    unittest.main()
