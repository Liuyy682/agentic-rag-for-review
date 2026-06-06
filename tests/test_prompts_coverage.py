import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from rag_agent import prompts


class TestPromptsCoverage(unittest.TestCase):
    def test_all_prompt_builders_return_non_empty_strings(self):
        builders = [
            prompts.get_conversation_summary_prompt,
            prompts.get_intent_recognition_prompt,
            prompts.get_rewrite_query_prompt,
            prompts.get_task_executor_prompt,
            prompts.get_chitchat_prompt,
            prompts.get_unsupported_prompt,
            prompts.get_fallback_response_prompt,
            prompts.get_knowledge_fallback_prompt,
            prompts.get_context_compression_prompt,
            prompts.get_answer_evaluation_prompt,
            prompts.get_aggregation_prompt,
        ]
        for builder in builders:
            value = builder()
            self.assertIsInstance(value, str)
            self.assertTrue(value.strip(), f"{builder.__name__} returned empty string")

    def test_prompts_carry_expected_keywords(self):
        self.assertIn("summar", prompts.get_conversation_summary_prompt().lower())
        self.assertIn("intent", prompts.get_intent_recognition_prompt().lower())
        self.assertIn("rewrite", prompts.get_rewrite_query_prompt().lower())
        self.assertIn("rag_research", prompts.get_task_executor_prompt())
        self.assertIn("Sources", prompts.get_fallback_response_prompt())
        self.assertIn("knowledge base", prompts.get_knowledge_fallback_prompt().lower())
        self.assertIn("compress", prompts.get_context_compression_prompt().lower())
        self.assertIn("is_satisfactory", prompts.get_answer_evaluation_prompt())
        self.assertIn("aggregat", prompts.get_aggregation_prompt().lower())


if __name__ == "__main__":
    unittest.main()
