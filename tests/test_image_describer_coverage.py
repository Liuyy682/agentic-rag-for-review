import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

import config
from ingestion import image_describer
from ingestion.image_describer import (
    LocalVLMImageDescriber,
    PaddleOcrImageDescriber,
    create_image_describer,
    _resolve_default_describer,
    _default_max_per_doc,
    _default_min_width,
    _default_min_height,
    _default_worker_count,
)


class TestLocalVLMDescribeImage(unittest.TestCase):
    def test_posts_to_vlm_and_returns_content(self, ):
        with patch("ingestion.image_describer.httpx.post") as post:
            resp = MagicMock()
            resp.json.return_value = {
                "choices": [{"message": {"content": "  OCR: x\nRAG_SUMMARY: y\nKEY_TERMS: z  "}}]
            }
            resp.raise_for_status.return_value = None
            post.return_value = resp

            describer = LocalVLMImageDescriber(
                base_url="http://vlm.local/v1/",
                api_key="secret",
                model="m",
                timeout_seconds=5.0,
                max_tokens=100,
            )
            # base_url trailing slash should be stripped
            self.assertEqual(describer.base_url, "http://vlm.local/v1")

            import tempfile
            with tempfile.TemporaryDirectory() as d:
                img = Path(d) / "fig.png"
                img.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
                out = describer.describe_image(img, context_text="nearby text")

            self.assertEqual(out, "OCR: x\nRAG_SUMMARY: y\nKEY_TERMS: z")
            # Endpoint and auth header passed correctly
            called_url = post.call_args[0][0]
            self.assertEqual(called_url, "http://vlm.local/v1/chat/completions")
            headers = post.call_args.kwargs["headers"]
            self.assertEqual(headers["Authorization"], "Bearer secret")
            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "m")
            self.assertEqual(payload["max_tokens"], 100)

    def test_empty_context_becomes_placeholder(self):
        with patch("ingestion.image_describer.httpx.post") as post:
            resp = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": "SKIP_IMAGE"}}]}
            resp.raise_for_status.return_value = None
            post.return_value = resp

            describer = LocalVLMImageDescriber()
            import tempfile
            with tempfile.TemporaryDirectory() as d:
                img = Path(d) / "fig.png"
                img.write_bytes(b"data")
                out = describer.describe_image(img, context_text="   ")

            self.assertEqual(out, "SKIP_IMAGE")
            prompt_text = post.call_args.kwargs["json"]["messages"][0]["content"][0]["text"]
            self.assertIn("无", prompt_text)


class _FakeOcr:
    """Stand-in for a PaddleOCR instance with a programmable result."""

    def __init__(self, result=None, raise_exc=False):
        self._result = result
        self._raise = raise_exc

    def ocr(self, path):
        if self._raise:
            raise RuntimeError("ocr engine failure")
        return self._result


class TestPaddleOcrDescribeImage(unittest.TestCase):
    def _describer_with(self, fake):
        d = PaddleOcrImageDescriber()
        d._ocr = fake  # bypass lazy loading of the real model
        return d

    def test_returns_three_field_format_on_text(self):
        result = [[
            [[[0, 0]], ("第一行文字", 0.99)],
            [[[0, 1]], ("  ", 0.5)],  # whitespace stripped out
            [[[0, 2]], ("第二行内容", 0.97)],
        ]]
        d = self._describer_with(_FakeOcr(result=result))
        with patch.object(d, "_extract_keywords", return_value="关键词1, 关键词2"):
            out = d.describe_image(Path("img.png"))
        self.assertIn("OCR: 第一行文字\n第二行内容", out)
        self.assertIn("RAG_SUMMARY: 第一行文字 第二行内容", out)
        self.assertIn("KEY_TERMS: 关键词1, 关键词2", out)

    def test_ocr_exception_returns_skip(self):
        d = self._describer_with(_FakeOcr(raise_exc=True))
        self.assertEqual(d.describe_image(Path("img.png")), "SKIP_IMAGE")

    def test_no_text_returns_skip(self):
        d = self._describer_with(_FakeOcr(result=[[]]))
        self.assertEqual(d.describe_image(Path("img.png")), "SKIP_IMAGE")

    def test_none_result_returns_skip(self):
        d = self._describer_with(_FakeOcr(result=None))
        self.assertEqual(d.describe_image(Path("img.png")), "SKIP_IMAGE")

    def test_lazy_ocr_property_constructs_paddleocr(self):
        fake_module = types.ModuleType("paddleocr")
        constructed = {}

        class FakePaddleOCR:
            def __init__(self, lang=None, use_gpu=None):
                constructed["lang"] = lang
                constructed["use_gpu"] = use_gpu

        fake_module.PaddleOCR = FakePaddleOCR
        with patch.dict(sys.modules, {"paddleocr": fake_module}):
            d = PaddleOcrImageDescriber(lang="en", use_gpu=True)
            instance = d.ocr
            # second access returns the cached instance
            self.assertIs(instance, d.ocr)
        self.assertEqual(constructed, {"lang": "en", "use_gpu": True})

    def test_extract_keywords_uses_jieba(self):
        d = PaddleOcrImageDescriber()
        fake_jieba = types.ModuleType("jieba")
        fake_analyse = types.ModuleType("jieba.analyse")
        fake_analyse.textrank = lambda text, topK=10: ["alpha", "beta"]
        fake_jieba.analyse = fake_analyse
        with patch.dict(sys.modules, {"jieba": fake_jieba, "jieba.analyse": fake_analyse}):
            out = d._extract_keywords("some chinese text 内容")
        self.assertEqual(out, "alpha, beta")

    def test_extract_keywords_returns_empty_on_failure(self):
        d = PaddleOcrImageDescriber()
        fake_analyse = types.ModuleType("jieba.analyse")

        def boom(text, topK=10):
            raise RuntimeError("textrank failed")

        fake_analyse.textrank = boom
        fake_jieba = types.ModuleType("jieba")
        fake_jieba.analyse = fake_analyse
        with patch.dict(sys.modules, {"jieba": fake_jieba, "jieba.analyse": fake_analyse}):
            self.assertEqual(d._extract_keywords("text"), "")


class TestCreateImageDescriber(unittest.TestCase):
    def setUp(self):
        self._orig_engine = config.IMAGE_ANALYSIS_ENGINE

    def tearDown(self):
        config.IMAGE_ANALYSIS_ENGINE = self._orig_engine

    def test_paddleocr_engine_returns_ocr_describer(self):
        config.IMAGE_ANALYSIS_ENGINE = "paddleocr"
        describer = create_image_describer()
        self.assertTrue(callable(describer))
        # Bound method of a PaddleOcrImageDescriber
        self.assertIsInstance(describer.__self__, PaddleOcrImageDescriber)

    def test_vlm_engine_returns_vlm_describer(self):
        config.IMAGE_ANALYSIS_ENGINE = "vlm"
        describer = create_image_describer()
        self.assertIsInstance(describer.__self__, LocalVLMImageDescriber)

    def test_none_engine_returns_none(self):
        config.IMAGE_ANALYSIS_ENGINE = "none"
        self.assertIsNone(create_image_describer())

    def test_unknown_engine_raises(self):
        config.IMAGE_ANALYSIS_ENGINE = "bogus"
        with self.assertRaises(ValueError):
            create_image_describer()

    def test_resolve_default_falls_back_to_vlm_when_none(self):
        config.IMAGE_ANALYSIS_ENGINE = "none"
        describer = _resolve_default_describer()
        self.assertIsInstance(describer.__self__, LocalVLMImageDescriber)

    def test_resolve_default_uses_configured_engine(self):
        config.IMAGE_ANALYSIS_ENGINE = "paddleocr"
        describer = _resolve_default_describer()
        self.assertIsInstance(describer.__self__, PaddleOcrImageDescriber)


class TestDefaultSizingHelpers(unittest.TestCase):
    def setUp(self):
        self._orig_engine = config.IMAGE_ANALYSIS_ENGINE

    def tearDown(self):
        config.IMAGE_ANALYSIS_ENGINE = self._orig_engine

    def test_ocr_engine_uses_ocr_thresholds(self):
        config.IMAGE_ANALYSIS_ENGINE = "paddleocr"
        self.assertEqual(_default_max_per_doc(), config.OCR_IMAGE_MAX_PER_DOC)
        self.assertEqual(_default_min_width(), config.OCR_IMAGE_MIN_WIDTH)
        self.assertEqual(_default_min_height(), config.OCR_IMAGE_MIN_HEIGHT)
        self.assertEqual(_default_worker_count(), config.OCR_IMAGE_ANALYSIS_WORKERS)

    def test_vlm_engine_uses_vlm_thresholds(self):
        config.IMAGE_ANALYSIS_ENGINE = "vlm"
        self.assertEqual(_default_max_per_doc(), config.VLM_IMAGE_MAX_PER_DOC)
        self.assertEqual(_default_min_width(), config.VLM_IMAGE_MIN_WIDTH)
        self.assertEqual(_default_min_height(), config.VLM_IMAGE_MIN_HEIGHT)
        self.assertEqual(_default_worker_count(), config.VLM_IMAGE_ANALYSIS_WORKERS)


if __name__ == "__main__":
    unittest.main()