import sys
import tempfile
import unittest
from pathlib import Path


from ingestion.course_structure import (
    MAX_KNOWLEDGE_POINTS_PER_SECTION,
    CourseStructureStore,
    parse_course_names,
)


class CourseStructureCoverageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.markdown_dir = self.root / "markdown"
        self.markdown_dir.mkdir()
        self.path = self.root / "course_structure.json"

    def _store(self):
        return CourseStructureStore(self.path)

    def _write_md(self, name, text):
        (self.markdown_dir / name).write_text(text, encoding="utf-8")

    # ── Loading ─────────────────────────────────────────────────────────────

    def test_load_invalid_json_raises_value_error(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            CourseStructureStore(self.path)
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_load_existing_json_backfills_defaults(self):
        self.path.write_text("{}", encoding="utf-8")
        store = CourseStructureStore(self.path)
        self.assertEqual(store.data["courses"], {})
        self.assertEqual(store.data["schema_version"], 1)

    # ── clear ────────────────────────────────────────────────────────────────

    def test_clear_resets_and_persists(self):
        store = self._store()
        store.ensure_course("Math")
        store.clear()
        self.assertEqual(store.list_courses(), [])
        # Reload to confirm persistence.
        self.assertEqual(self._store().list_courses(), [])

    # ── get_course_by_name ────────────────────────────────────────────────────

    def test_get_course_by_name_blank_returns_none(self):
        store = self._store()
        self.assertIsNone(store.get_course_by_name("   "))

    def test_get_course_by_name_is_case_insensitive(self):
        store = self._store()
        store.ensure_course("Database Systems")
        self.assertIsNotNone(store.get_course_by_name("database systems"))

    # ── ensure_course ──────────────────────────────────────────────────────────

    def test_ensure_course_empty_name_raises(self):
        store = self._store()
        with self.assertRaises(ValueError):
            store.ensure_course("   ")

    def test_ensure_course_returns_existing_id(self):
        store = self._store()
        first = store.ensure_course("Math")
        second = store.ensure_course("math")  # case-insensitive match
        self.assertEqual(first, second)

    def test_ensure_course_resolves_id_collision(self):
        store = self._store()
        first_id = store.ensure_course("Math")
        # Inject a fake course occupying the deterministic id for a different name
        # to force the while-loop suffixing path.
        from ingestion.course_structure import _stable_id

        colliding_id = _stable_id("course", "Physics")
        store.data["courses"][colliding_id] = {
            "course_id": colliding_id,
            "name": "Not Physics",
            "documents": [],
            "sections": [],
        }
        new_id = store.ensure_course("Physics")
        self.assertNotEqual(new_id, colliding_id)
        self.assertTrue(new_id.endswith("_2"))
        self.assertNotEqual(new_id, first_id)

    # ── assign_document_to_courses ─────────────────────────────────────────────

    def test_assign_skips_blank_names(self):
        store = self._store()
        self._write_md("db.md", "# Title\n\ncontent")
        ids = store.assign_document_to_courses(
            "db.md", ["", "   ", "Real Course"], markdown_dir=self.markdown_dir
        )
        self.assertEqual(len(ids), 1)
        self.assertEqual(store.source_files_for_course("Real Course"), ["db.md"])

    # ── rename_course ──────────────────────────────────────────────────────────

    def test_rename_course_missing_returns_false(self):
        store = self._store()
        self.assertFalse(store.rename_course("Nope", "New"))

    def test_rename_course_blank_new_name_returns_false(self):
        store = self._store()
        store.ensure_course("Math")
        self.assertFalse(store.rename_course("Math", "   "))

    def test_rename_course_to_existing_name_raises(self):
        store = self._store()
        store.ensure_course("Math")
        store.ensure_course("Physics")
        with self.assertRaises(ValueError):
            store.rename_course("Math", "Physics")

    def test_rename_course_to_same_name_is_allowed(self):
        store = self._store()
        store.ensure_course("Math")
        # duplicate resolves to same course → no ValueError, returns True.
        self.assertTrue(store.rename_course("Math", "Math"))

    # ── rename_section ─────────────────────────────────────────────────────────

    def test_rename_section_guards_blank_inputs(self):
        store = self._store()
        store.ensure_course("Math")
        self.assertFalse(store.rename_section("Math", "   ", "New"))
        self.assertFalse(store.rename_section("Math", "Sec", "   "))
        self.assertFalse(store.rename_section("Missing", "Sec", "New"))

    def test_rename_section_not_found_returns_false(self):
        store = self._store()
        self._write_md("db.md", "# Title\n\ncontent")
        store.assign_document_to_courses("db.md", ["Math"], markdown_dir=self.markdown_dir)
        self.assertFalse(store.rename_section("Math", "Nonexistent Section", "New"))

    # ── source_files_for_course ────────────────────────────────────────────────

    def test_source_files_for_course_guards(self):
        store = self._store()
        self.assertEqual(store.source_files_for_course(None), [])
        self.assertEqual(store.source_files_for_course(""), [])
        self.assertEqual(store.source_files_for_course("Unknown"), [])

    # ── remove_document ─────────────────────────────────────────────────────────

    def test_remove_document_no_match_returns_empty(self):
        store = self._store()
        self._write_md("db.md", "# Title\n\ncontent")
        store.assign_document_to_courses("db.md", ["Math"], markdown_dir=self.markdown_dir)
        # Removing a file not in any course leaves everything untouched.
        self.assertEqual(store.remove_document("other.md", markdown_dir=self.markdown_dir), [])
        self.assertEqual(store.source_files_for_course("Math"), ["db.md"])

    def test_remove_document_with_section_only_match(self):
        # A course whose document list no longer holds the file but whose sections
        # still reference it should be rebuilt (had_sections branch).
        store = self._store()
        self._write_md("db.md", "# Heading\n\ncontent")
        store.assign_document_to_courses("db.md", ["Math"], markdown_dir=self.markdown_dir)
        course = store.get_course_by_name("Math")
        # Manually drop the document but keep the section referencing db.md.
        course["documents"] = []
        store.save()

        affected = store.remove_document("db.md", markdown_dir=self.markdown_dir)

        self.assertEqual(affected, ["Math"])

    # ── rebuild_course ──────────────────────────────────────────────────────────

    def test_rebuild_course_missing_returns_false(self):
        store = self._store()
        self.assertFalse(store.rebuild_course("missing_id"))

    def test_rebuild_course_skips_missing_markdown_files(self):
        store = self._store()
        course_id = store.ensure_course("Math")
        store.data["courses"][course_id]["documents"] = ["ghost.md"]
        # File does not exist on disk → continue branch, no sections built.
        self.assertTrue(store.rebuild_course(course_id, markdown_dir=self.markdown_dir, save=True))
        self.assertEqual(store.get_course(course_id)["sections"], [])

    def test_rebuild_course_preserves_user_edited_titles(self):
        store = self._store()
        self._write_md("db.md", "# Topic A\n\ncontent\n\n## Topic B\n\nmore")
        store.assign_document_to_courses("db.md", ["Math"], markdown_dir=self.markdown_dir)
        course = store.get_course_by_name("Math")
        target = course["sections"][0]
        original_title = target["original_title"]
        store.rename_section("Math", target["title"], "Renamed Topic")

        # Rebuild should keep the user-edited title for the same source/original title.
        store.rebuild_course(course["course_id"], markdown_dir=self.markdown_dir, save=True)
        rebuilt = store.get_course_by_name("Math")
        edited = [s for s in rebuilt["sections"] if s["original_title"] == original_title]
        self.assertTrue(edited)
        self.assertEqual(edited[0]["title"], "Renamed Topic")
        self.assertTrue(edited[0]["user_edited"])

    # ── _extract_sections / _add_knowledge_point ──────────────────────────────────

    def test_extract_sections_fallback_when_no_headings(self):
        store = self._store()
        self._write_md("plain.md", "just some text\nwith no headings at all\n")
        store.assign_document_to_courses("plain.md", ["Misc"], markdown_dir=self.markdown_dir)
        course = store.get_course_by_name("Misc")
        self.assertEqual(len(course["sections"]), 1)
        self.assertIn("未在资料中识别到明确章节标题", course["sections"][0]["summary"])

    def test_deep_heading_becomes_knowledge_point(self):
        # A level-3 heading under a level-2 section becomes a knowledge point.
        store = self._store()
        self._write_md("db.md", "## Section One\n\ntext\n\n### Sub Point\n\nbody")
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        names = [kp["name"] for kp in section["knowledge_points"]]
        self.assertIn("Sub Point", names)

    def test_list_items_become_knowledge_points(self):
        store = self._store()
        self._write_md(
            "db.md",
            "## Topics\n\n- First bullet point here\n- Second bullet point here\n",
        )
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        names = [kp["name"] for kp in section["knowledge_points"]]
        self.assertIn("First bullet point here", names)
        self.assertIn("Second bullet point here", names)

    def test_add_knowledge_point_dedupes_and_caps(self):
        store = self._store()
        bullets = "\n".join(f"- Knowledge point number {i}" for i in range(MAX_KNOWLEDGE_POINTS_PER_SECTION + 5))
        # Duplicate first bullet to exercise the dedupe branch.
        body = "## Topics\n\n- Knowledge point number 0\n" + bullets
        self._write_md("db.md", body)
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        self.assertLessEqual(len(section["knowledge_points"]), MAX_KNOWLEDGE_POINTS_PER_SECTION)

    def test_overlong_list_item_is_ignored(self):
        store = self._store()
        long_item = "x" * 200
        self._write_md("db.md", f"## Topics\n\n- {long_item}\n")
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        self.assertEqual(section["knowledge_points"], [])

    # ── summaries and listing ─────────────────────────────────────────────────────

    def test_build_course_summary_with_many_sections_uses_suffix(self):
        store = self._store()
        headings = "\n\n".join(f"# Heading {i}\n\nbody {i}" for i in range(7))
        self._write_md("db.md", headings)
        store.assign_document_to_courses("db.md", ["Big Course"], markdown_dir=self.markdown_dir)
        summary = store.get_course_by_name("Big Course")["summary"]
        self.assertIn("等", summary)
        self.assertIn("复习单元", summary)

    def test_format_course_list_empty(self):
        store = self._store()
        self.assertIn("暂无课程", store.format_course_list())

    def test_format_course_list_with_courses(self):
        store = self._store()
        self._write_md("db.md", "# Heading\n\nbody")
        store.assign_document_to_courses("db.md", ["My Course"], markdown_dir=self.markdown_dir)
        listing = store.format_course_list()
        self.assertIn("My Course", listing)
        self.assertIn("份资料", listing)

    # ── parse_course_names ─────────────────────────────────────────────────────────

    def test_parse_course_names_none_and_iterable(self):
        self.assertEqual(parse_course_names(None), [])
        self.assertEqual(
            parse_course_names(["Math", "  Math  ", "Physics", ""]),
            ["Math", "Physics"],
        )

    def test_parse_course_names_string_splits_on_separators(self):
        self.assertEqual(
            parse_course_names("Math，Physics; Math\nChemistry；"),
            ["Math", "Physics", "Chemistry"],
        )

    def test_heading_with_empty_title_is_skipped(self):
        # Heading whose text collapses to empty after cleaning hits the `continue`.
        store = self._store()
        self._write_md("db.md", "## Real Section\n\ntext\n\n### #\n\nbody")
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        # The empty deep heading produced no knowledge point.
        self.assertEqual(section["knowledge_points"], [])

    def test_overlong_deep_heading_knowledge_point_ignored(self):
        store = self._store()
        long_heading = "y" * 120
        self._write_md("db.md", f"## Section\n\ntext\n\n### {long_heading}\n\nbody")
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        self.assertEqual(section["knowledge_points"], [])

    def test_knowledge_point_on_later_page_extends_page_numbers(self):
        # Section heading on page 1, deep heading on page 2 → page 2 appended.
        store = self._store()
        markdown = (
            "## Section One\n\nintro\n"
            "--- end of page.page_number=1 ---\n"
            "### Detail Point\n\nbody\n"
            "--- end of page.page_number=2 ---\n"
        )
        self._write_md("db.md", markdown)
        store.assign_document_to_courses("db.md", ["Course"], markdown_dir=self.markdown_dir)
        section = store.get_course_by_name("Course")["sections"][0]
        self.assertIn(2, section["page_numbers"])
        self.assertIn("Detail Point", [kp["name"] for kp in section["knowledge_points"]])


if __name__ == "__main__":
    unittest.main()
