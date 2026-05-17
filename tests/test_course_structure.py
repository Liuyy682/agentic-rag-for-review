import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from ingestion.course_structure import CourseStructureStore, parse_course_names


def sample_markdown():
    return """# Database Systems
intro
--- end of page.page_number=1 ---
## Transactions
### ACID
- Atomicity
- Isolation
--- end of page.page_number=2 ---
## Recovery
### Undo logging
--- end of page.page_number=3 ---
"""


class TestCourseStructureStore(unittest.TestCase):
    def test_parse_course_names_deduplicates_common_separators(self):
        self.assertEqual(
            parse_course_names("Database Systems， Computer Networks; Database Systems"),
            ["Database Systems", "Computer Networks"],
        )

    def test_document_can_belong_to_multiple_courses_and_persist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_dir = temp_path / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "db.md").write_text(sample_markdown(), encoding="utf-8")
            path = temp_path / "course_structure.json"

            store = CourseStructureStore(path)
            course_ids = store.assign_document_to_courses(
                "db.md",
                ["Database Systems", "Final Review"],
                markdown_dir=markdown_dir,
            )

            self.assertEqual(len(course_ids), 2)
            self.assertEqual(
                store.source_files_for_course("Database Systems"),
                ["db.md"],
            )
            self.assertEqual(
                store.source_files_for_course("Final Review"),
                ["db.md"],
            )
            self.assertGreaterEqual(len(store.get_course_by_name("Database Systems")["sections"]), 2)

            reloaded = CourseStructureStore(path)
            self.assertEqual(reloaded.source_files_for_course("Final Review"), ["db.md"])

    def test_rename_course_and_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_dir = temp_path / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "db.md").write_text(sample_markdown(), encoding="utf-8")

            store = CourseStructureStore(temp_path / "course_structure.json")
            store.assign_document_to_courses("db.md", ["Database Systems"], markdown_dir=markdown_dir)

            self.assertTrue(store.rename_course("Database Systems", "DB Final"))
            self.assertIsNotNone(store.get_course_by_name("DB Final"))
            self.assertTrue(store.rename_section("DB Final", "Transactions", "Transaction Management"))

            section_titles = [section["title"] for section in store.get_course_by_name("DB Final")["sections"]]
            self.assertIn("Transaction Management", section_titles)

    def test_remove_document_rebuilds_affected_courses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            markdown_dir = temp_path / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "db.md").write_text(sample_markdown(), encoding="utf-8")
            (markdown_dir / "net.md").write_text("# Networks\n\nRouting", encoding="utf-8")

            store = CourseStructureStore(temp_path / "course_structure.json")
            store.assign_document_to_courses("db.md", ["Final Review"], markdown_dir=markdown_dir)
            store.assign_document_to_courses("net.md", ["Final Review"], markdown_dir=markdown_dir)

            affected = store.remove_document("db.md", markdown_dir=markdown_dir)

            self.assertEqual(affected, ["Final Review"])
            self.assertEqual(store.source_files_for_course("Final Review"), ["net.md"])
            course = store.get_course_by_name("Final Review")
            self.assertTrue(course["sections"])
            self.assertNotIn("db.md", {section["source_file"] for section in course["sections"]})


if __name__ == "__main__":
    unittest.main()
