import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT_DIR))

from PIL import Image

from image_describer import (
    enhance_markdown_image_references,
    extract_image_context,
    resolve_image_path,
)
import config


class TestImageDescriber(unittest.TestCase):
    def test_enhances_local_markdown_image_with_vlm_description(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            image_dir.mkdir()
            image_path = image_dir / "figure.png"
            Image.new("RGB", (120, 80), color="white").save(image_path)

            markdown = f"# Optical Fiber\n\nLED laser signal\n\n![]({image_path})\n\nDetector output"

            enhanced = enhance_markdown_image_references(
                markdown,
                describe_image=lambda path, context: (
                    f"OCR: text from {path.name}\n"
                    f"RAG_SUMMARY: optical conversion diagram; context={context}\n"
                    f"KEY_TERMS: LED, laser, detector"
                ),
            )

            self.assertIn("<!-- image-analysis:start", enhanced)
            self.assertIn("image-analysis:end -->", enhanced)
            self.assertIn("text from figure.png", enhanced)
            self.assertIn("optical conversion diagram", enhanced)
            self.assertIn("Optical Fiber", enhanced)
            self.assertNotIn("### Image OCR", enhanced)

    def test_skips_tiny_images_without_calling_vlm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "icon.png"
            Image.new("RGB", (20, 20), color="white").save(image_path)
            markdown = f"![]({image_path})"

            enhanced = enhance_markdown_image_references(
                markdown,
                describe_image=lambda path, context: "should not be called",
            )

            self.assertEqual(enhanced, markdown)

    def test_skips_when_vlm_returns_skip_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "decorative.png"
            Image.new("RGB", (120, 80), color="white").save(image_path)
            markdown = f"![]({image_path})"

            enhanced = enhance_markdown_image_references(
                markdown,
                describe_image=lambda path, context: "SKIP_IMAGE",
            )

            self.assertEqual(enhanced, markdown)

    def test_resolves_image_relative_to_image_root_by_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_root = Path(temp_dir) / "doc"
            image_root.mkdir()
            image_path = image_root / "paper-0001-00.png"
            image_path.write_bytes(b"fake-png")

            resolved = resolve_image_path("paper-0001-00.png", image_root=image_root)

            self.assertEqual(resolved, image_path.resolve())

    def test_resolves_image_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_root = Path(temp_dir) / "doc with spaces"
            image_root.mkdir()
            image_path = image_root / "figure one.png"
            image_path.write_bytes(b"fake-png")

            resolved = resolve_image_path("figure one.png", image_root=image_root)

            self.assertEqual(resolved, image_path.resolve())

    def test_leaves_remote_images_unchanged(self):
        markdown = "![x](https://example.com/image.png)"
        enhanced = enhance_markdown_image_references(
            markdown,
            describe_image=lambda path, context: "should not be called",
        )

        self.assertEqual(enhanced, markdown)

    def test_leaves_image_unchanged_when_description_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "figure.png"
            image_path.write_bytes(b"fake-png")
            markdown = f"![]({image_path})"

            def fail_description(path):
                raise RuntimeError("vlm unavailable")

            with patch("builtins.print"):
                enhanced = enhance_markdown_image_references(
                    markdown,
                    describe_image=lambda path, context: fail_description(path),
                )

            self.assertEqual(enhanced, markdown)

    def test_extracts_nearby_context_without_image_markdown(self):
        markdown = "# Title\n\nRelevant before\n\n![x](image.png)\n\nRelevant after"
        start = markdown.index("![x]")
        end = start + len("![x](image.png)")

        context = extract_image_context(markdown, start, end)

        self.assertIn("Relevant before", context)
        self.assertIn("Relevant after", context)
        self.assertNotIn("![x]", context)

    def test_parallel_image_analysis_preserves_markdown_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_workers = config.VLM_IMAGE_ANALYSIS_WORKERS
            try:
                config.VLM_IMAGE_ANALYSIS_WORKERS = 2
                image_one = Path(temp_dir) / "one.png"
                image_two = Path(temp_dir) / "two.png"
                Image.new("RGB", (120, 80), color="white").save(image_one)
                Image.new("RGB", (120, 80), color="white").save(image_two)
                markdown = f"A\n![]({image_one})\nB\n![]({image_two})\nC"

                def describe(path, context):
                    if path.name == "one.png":
                        time.sleep(0.05)
                    return (
                        f"OCR: {path.name}\n"
                        f"RAG_SUMMARY: useful analysis for {path.name}\n"
                        f"KEY_TERMS: {path.stem}"
                    )

                enhanced = enhance_markdown_image_references(
                    markdown,
                    describe_image=describe,
                )

                self.assertLess(enhanced.index("OCR: one.png"), enhanced.index("OCR: two.png"))
                self.assertIn("A", enhanced)
                self.assertIn("B", enhanced)
                self.assertIn("C", enhanced)
            finally:
                config.VLM_IMAGE_ANALYSIS_WORKERS = old_workers


if __name__ == "__main__":
    unittest.main()
