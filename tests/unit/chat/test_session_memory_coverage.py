import sys
import tempfile
import unittest
from pathlib import Path


from agentic_rag.chat.session_memory import SessionMemoryStore


class SessionMemoryCoverageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = SessionMemoryStore(Path(self._tmp.name) / "memory.sqlite3")

    # ── Session CRUD ────────────────────────────────────────────────────────

    def test_create_session_persists_and_is_retrievable(self):
        session_id = self.store.create_session(course_name="Math", title="Algebra")

        self.assertTrue(session_id)
        session = self.store.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session["course_name"], "Math")
        self.assertEqual(session["title"], "Algebra")
        self.assertEqual(session["created_at"], session["updated_at"])

    def test_create_session_defaults_blank_course(self):
        session_id = self.store.create_session()
        session = self.store.get_session(session_id)
        self.assertEqual(session["course_name"], "")
        self.assertIsNone(session["title"])

    def test_get_session_missing_returns_none(self):
        self.assertIsNone(self.store.get_session("does-not-exist"))

    def test_list_sessions_filters_by_course_and_counts_turns(self):
        math_id = self.store.create_session(course_name="Math")
        self.store.create_session(course_name="Physics")
        self.store.append_turn(math_id, "q1", "a1", course_name="Math")
        self.store.append_turn(math_id, "q2", "a2", course_name="Math")

        math_sessions = self.store.list_sessions("Math")
        physics_sessions = self.store.list_sessions("Physics")

        self.assertEqual(len(math_sessions), 1)
        self.assertEqual(math_sessions[0]["id"], math_id)
        self.assertEqual(math_sessions[0]["turn_count"], 2)
        self.assertEqual(len(physics_sessions), 1)
        self.assertEqual(physics_sessions[0]["turn_count"], 0)

    def test_list_sessions_default_blank_course(self):
        blank_id = self.store.create_session()
        self.store.create_session(course_name="Math")

        blank_sessions = self.store.list_sessions()

        self.assertEqual([s["id"] for s in blank_sessions], [blank_id])

    def test_update_session_title_sets_and_trims(self):
        session_id = self.store.create_session(course_name="Math")
        self.store.update_session_title(session_id, "  New Title  ")

        self.assertEqual(self.store.get_session(session_id)["title"], "New Title")

    def test_update_session_title_ignores_empty_inputs(self):
        session_id = self.store.create_session(title="Original")
        # Empty title is a no-op.
        self.store.update_session_title(session_id, "")
        self.assertEqual(self.store.get_session(session_id)["title"], "Original")
        # Empty session id is a no-op (must not raise).
        self.store.update_session_title("", "Something")

    def test_delete_session_removes_session_and_turns(self):
        session_id = self.store.create_session(course_name="Math")
        self.store.append_turn(session_id, "q", "a", course_name="Math")

        self.store.delete_session(session_id)

        self.assertIsNone(self.store.get_session(session_id))
        self.assertEqual(self.store.get_recent_turns(session_id), [])

    def test_delete_session_empty_id_is_noop(self):
        # Exercises the early return guard.
        self.store.delete_session("")
        self.store.delete_session(None)

    def test_rename_course_in_sessions_updates_both_tables(self):
        session_id = self.store.create_session(course_name="OldName")
        self.store.append_turn(session_id, "q", "a", course_name="OldName")

        self.store.rename_course_in_sessions("OldName", "NewName")

        self.assertEqual(self.store.get_session(session_id)["course_name"], "NewName")
        self.assertEqual(self.store.list_sessions("NewName")[0]["id"], session_id)
        turns = self.store.get_session_turns(session_id)
        self.assertEqual(turns[0]["course_name"], "NewName")

    # ── Turn management edge cases ──────────────────────────────────────────

    def test_append_turn_skips_blank_inputs(self):
        # Missing session id / empty text → no rows inserted.
        self.store.append_turn("", "q", "a")
        self.store.append_turn("s1", "   ", "a")
        self.store.append_turn("s1", "q", "   ")
        self.assertEqual(self.store.get_session_turns("s1"), [])

    def test_append_turn_creates_session_row_for_legacy_id(self):
        # No create_session() call first: INSERT OR IGNORE path builds the row,
        # and first turn auto-titles from the user text.
        self.store.append_turn("legacy", "a" * 50, "answer", course_name="Hist")

        session = self.store.get_session("legacy")
        self.assertIsNotNone(session)
        self.assertEqual(session["course_name"], "Hist")
        self.assertTrue(session["title"].endswith("..."))
        self.assertEqual(len(session["title"]), 43)  # 40 chars + "..."

    def test_append_turn_short_first_message_title_has_no_ellipsis(self):
        self.store.append_turn("s_short", "hi there", "answer")
        self.assertEqual(self.store.get_session("s_short")["title"], "hi there")

    def test_append_turn_second_turn_updates_timestamp_only(self):
        self.store.append_turn("s", "first message", "a1")
        first_title = self.store.get_session("s")["title"]
        self.store.append_turn("s", "second message", "a2")

        session = self.store.get_session("s")
        # Title preserved from the first turn (else-branch only updates updated_at).
        self.assertEqual(session["title"], first_title)
        turns = self.store.get_session_turns("s")
        self.assertEqual([t["turn_index"] for t in turns], [1, 2])

    def test_get_recent_turns_guards_empty_id_and_nonpositive_limit(self):
        self.store.append_turn("s", "q", "a")
        self.assertEqual(self.store.get_recent_turns(""), [])
        self.assertEqual(self.store.get_recent_turns("s", limit=0), [])
        self.assertEqual(self.store.get_recent_turns("s", limit=-1), [])

    def test_get_session_turns_returns_all_in_order(self):
        for i in range(3):
            self.store.append_turn("s", f"q{i}", f"a{i}")
        turns = self.store.get_session_turns("s")
        self.assertEqual([t["turn_index"] for t in turns], [1, 2, 3])

    def test_format_recent_turns_empty_returns_blank(self):
        self.assertEqual(self.store.format_recent_turns([]), "")

    def test_format_recent_turns_renders_numbered_turns(self):
        text = self.store.format_recent_turns(
            [
                {"user_original": "q one", "assistant_final": "a one"},
                {"user_original": "q two", "assistant_final": "a two"},
            ]
        )
        self.assertIn("Recent conversation memory from this session.", text)
        self.assertIn("Turn 1", text)
        self.assertIn("User: q one", text)
        self.assertIn("Assistant: a one", text)
        self.assertIn("Turn 2", text)
        self.assertIn("User: q two", text)


if __name__ == "__main__":
    unittest.main()
