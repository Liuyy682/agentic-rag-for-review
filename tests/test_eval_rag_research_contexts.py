import json
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

try:
    from langchain_core.messages import ToolMessage
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("langchain_core is not installed in this environment") from exc

from evaluation.runners.ragbench_local_rag_runner import (
    _rag_research_diagnostics_from_state,
    _tool_contexts_from_state,
)


class TestEvalRagResearchContextExtraction(unittest.TestCase):
    def test_extracts_contexts_from_rag_research_json(self):
        state = {
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [
                                {"content": "evidence one"},
                                {"content": "evidence two"},
                                {"content": "evidence one"},
                            ],
                            "parent_ids": ["p1", "p2"],
                            "gaps": [],
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_1",
                )
            ]
        }

        self.assertEqual(_tool_contexts_from_state(state), ["evidence one", "evidence two"])

    def test_summarizes_rag_research_diagnostics(self):
        state = {
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [{"content": "one"}, {"content": "two"}],
                            "parent_ids": ["p1", "p2"],
                            "gaps": ["missing detail"],
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_1",
                ),
                ToolMessage(
                    content=json.dumps(
                        {
                            "contexts": [{"content": "three"}],
                            "parent_ids": ["p2", "p3"],
                            "gaps": [],
                        }
                    ),
                    name="rag_research",
                    tool_call_id="call_2",
                ),
            ]
        }

        self.assertEqual(
            _rag_research_diagnostics_from_state(state),
            {
                "rag_research_call_count": 2,
                "parent_id_count": 3,
                "context_count": 3,
                "gap_count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
