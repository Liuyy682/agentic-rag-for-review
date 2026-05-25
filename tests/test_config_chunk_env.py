import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config


class TestChunkConfigEnvironment(unittest.TestCase):
    def tearDown(self):
        for name in ("CHILD_CHUNK_SIZE", "CHILD_CHUNK_OVERLAP", "MIN_PARENT_SIZE", "MAX_PARENT_SIZE"):
            os.environ.pop(name, None)
        importlib.reload(config)

    def test_chunk_defaults_are_preserved_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in ("CHILD_CHUNK_SIZE", "CHILD_CHUNK_OVERLAP", "MIN_PARENT_SIZE", "MAX_PARENT_SIZE"):
                os.environ.pop(name, None)
            importlib.reload(config)

        self.assertEqual(config.CHILD_CHUNK_SIZE, 500)
        self.assertEqual(config.CHILD_CHUNK_OVERLAP, 100)
        self.assertEqual(config.MIN_PARENT_SIZE, 2000)
        self.assertEqual(config.MAX_PARENT_SIZE, 4000)

    def test_chunk_sizes_can_be_overridden_by_env(self):
        with patch.dict(
            os.environ,
            {
                "CHILD_CHUNK_SIZE": "320",
                "CHILD_CHUNK_OVERLAP": "64",
                "MIN_PARENT_SIZE": "1600",
                "MAX_PARENT_SIZE": "6400",
            },
        ):
            importlib.reload(config)

        self.assertEqual(config.CHILD_CHUNK_SIZE, 320)
        self.assertEqual(config.CHILD_CHUNK_OVERLAP, 64)
        self.assertEqual(config.MIN_PARENT_SIZE, 1600)
        self.assertEqual(config.MAX_PARENT_SIZE, 6400)


class TestHuggingFaceCacheConfig(unittest.TestCase):
    def tearDown(self):
        for name in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
            os.environ.pop(name, None)
        importlib.reload(config)

    def test_defaults_use_project_cache(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in ("HF_HOME", "HF_HUB_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
                os.environ.pop(name, None)
            importlib.reload(config)

        self.assertEqual(config.HF_HOME, config.HF_CACHE_DIR)
        self.assertEqual(config.HF_HUB_CACHE, config.HF_CACHE_DIR)
        self.assertEqual(config.SENTENCE_TRANSFORMERS_HOME, config.HF_CACHE_DIR)

    def test_explicit_hf_cache_env_is_preserved(self):
        with patch.dict(
            os.environ,
            {
                "HF_HOME": "/tmp/hf-home",
                "HF_HUB_CACHE": "/tmp/hf-hub",
                "SENTENCE_TRANSFORMERS_HOME": "/tmp/st",
            },
        ):
            importlib.reload(config)

        self.assertEqual(config.HF_HOME, "/tmp/hf-home")
        self.assertEqual(config.HF_HUB_CACHE, "/tmp/hf-hub")
        self.assertEqual(config.SENTENCE_TRANSFORMERS_HOME, "/tmp/st")


if __name__ == "__main__":
    unittest.main()
