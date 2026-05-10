import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import AIMessageChunk

from chat.chat_interface import ChatInterface
from chat.session_memory import SessionMemoryStore


class FakeGraph:
    def __init__(self):
        self.stream_input = None

    def get_state(self, config):
        return SimpleNamespace(next=False)

    def stream(self, stream_input, config=None, stream_mode=None):
        self.stream_input = stream_input
        yield AIMessageChunk(content="final answer"), {"langgraph_node": "aggregate_answers"}


class FakeRagSystem:
    def __init__(self):
        self.thread_id = "session_1"
        self.agent_graph = FakeGraph()
        self.deleted_thread = None
        self.course_scope = None
        self.observability = SimpleNamespace(flush=lambda: None)

    def set_course_scope(self, source_files=None):
        self.course_scope = source_files

    def get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def reset_thread(self):
        self.deleted_thread = self.thread_id
        self.thread_id = "session_2"


class FakeMemoryStore:
    def __init__(self):
        self.deleted_sessions = []
        self.saved_turns = []

    def get_recent_turns(self, session_id, limit=5):
        return [
            {
                "user_original": "previous question",
                "assistant_final": "previous answer",
            }
        ]

    def format_recent_turns(self, turns):
        return "MEMORY BLOCK" if list(turns) else ""

    def append_turn(self, session_id, user_original, assistant_final, course_name=None):
        self.saved_turns.append((session_id, user_original, assistant_final, course_name))

    def delete_session(self, session_id):
        self.deleted_sessions.append(session_id)


class TestSessionMemoryStore(unittest.TestCase):
    def test_recent_turns_returns_last_five_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionMemoryStore(Path(temp_dir) / "memory.sqlite3")
            for idx in range(7):
                store.append_turn("session_a", f"question {idx}", f"answer {idx}")

            turns = store.get_recent_turns("session_a", limit=5)

        self.assertEqual([turn["user_original"] for turn in turns], [f"question {idx}" for idx in range(2, 7)])
        self.assertEqual([turn["turn_index"] for turn in turns], [3, 4, 5, 6, 7])

    def test_sessions_are_isolated_and_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionMemoryStore(Path(temp_dir) / "memory.sqlite3")
            store.append_turn("session_a", "question a", "answer a")
            store.append_turn("session_b", "question b", "answer b")

            store.delete_session("session_a")

            self.assertEqual(store.get_recent_turns("session_a"), [])
            self.assertEqual(store.get_recent_turns("session_b")[0]["user_original"], "question b")

    def test_format_recent_turns_marks_memory_as_non_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionMemoryStore(Path(temp_dir) / "memory.sqlite3")
            memory = store.format_recent_turns(
                [
                    {
                        "user_original": "What is caching?",
                        "assistant_final": "Caching stores reusable data.",
                    }
                ]
            )

        self.assertIn("Do not treat prior assistant answers as knowledge-base evidence.", memory)
        self.assertIn("User: What is caching?", memory)
        self.assertIn("Assistant: Caching stores reusable data.", memory)


class TestChatInterfaceSessionMemory(unittest.TestCase):
    def test_chat_injects_memory_and_saves_final_visible_answer(self):
        rag_system = FakeRagSystem()
        memory_store = FakeMemoryStore()
        chat = ChatInterface(rag_system, session_memory=memory_store)

        outputs = list(chat.chat("current question", history=[]))

        self.assertEqual(outputs[-1][-1]["content"], "final answer")
        self.assertEqual(rag_system.agent_graph.stream_input["conversation_memory"], "MEMORY BLOCK")
        self.assertEqual(
            memory_store.saved_turns,
            [("session_1", "current question", "final answer", None)],
        )

    def test_clear_session_deletes_old_session_memory_before_reset(self):
        rag_system = FakeRagSystem()
        memory_store = FakeMemoryStore()
        chat = ChatInterface(rag_system, session_memory=memory_store)

        chat.clear_session()

        self.assertEqual(memory_store.deleted_sessions, ["session_1"])
        self.assertEqual(rag_system.deleted_thread, "session_1")
        self.assertEqual(rag_system.thread_id, "session_2")


if __name__ == "__main__":
    unittest.main()

