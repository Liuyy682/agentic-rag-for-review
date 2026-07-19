import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


from PIL import Image

from ingestion import image_extractor
from ingestion.image_extractor import (
    extract_images_from_pptx,
    extract_images_from_docx,
    _is_pptx_picture,
    _save_image_blob,
)


class TestPptxImportFallback(unittest.TestCase):
    def test_missing_python_pptx_logs_and_returns(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "images"
            real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

            def fake_import(name, *args, **kwargs):
                if name == "pptx" or name.startswith("pptx."):
                    raise ImportError("no pptx")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                extract_images_from_pptx(Path(d) / "x.pptx", out_dir)
            # Returned before creating the output directory
            self.assertFalse(out_dir.exists())

    def test_open_failure_logs_and_returns(self):
        with tempfile.TemporaryDirectory() as d:
            pptx_path = Path(d) / "broken.pptx"
            pptx_path.write_bytes(b"not a real pptx")
            out_dir = Path(d) / "images"
            # python-pptx raises when opening invalid file -> warning + return
            extract_images_from_pptx(pptx_path, out_dir)
            self.assertEqual(list(out_dir.iterdir()), [])


class TestIsPptxPictureImportError(unittest.TestCase):
    def test_returns_false_when_pptx_enum_missing(self):
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("pptx"):
                raise ImportError("no pptx enum")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            self.assertFalse(_is_pptx_picture(object()))


class TestPptxShapeWithoutBlob(unittest.TestCase):
    def test_shape_attribute_error_is_skipped(self):
        """A picture shape whose .image raises AttributeError is skipped."""
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "images"
            out_dir.mkdir(parents=True)

            class FakeImage:
                @property
                def blob(self):
                    raise AttributeError("linked picture has no blob")

            class FakeShape:
                name = "Picture 1"
                image = FakeImage()

            class FakeSlide:
                shapes = [FakeShape()]

            class FakePresentation:
                slides = [FakeSlide()]

            with patch("ingestion.image_extractor._is_pptx_picture", return_value=True), \
                 patch("pptx.Presentation", return_value=FakePresentation()):
                extract_images_from_pptx(Path(d) / "x.pptx", out_dir)

            # Nothing written because blob access raised AttributeError
            self.assertEqual(list(out_dir.iterdir()), [])


class TestSaveImageBlob(unittest.TestCase):
    def test_known_format_written_raw(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.png"
            _save_image_blob(b"raw-png-bytes", "image/png", out)
            self.assertEqual(out.read_bytes(), b"raw-png-bytes")

    def test_unknown_format_decoded_and_saved_as_jpeg(self):
        with tempfile.TemporaryDirectory() as d:
            # Build a real RGBA PNG blob, claim an unusual content type
            buf = io.BytesIO()
            Image.new("RGBA", (40, 30), color=(255, 0, 0, 128)).save(buf, "PNG")
            blob = buf.getvalue()
            out = Path(d) / "out.jpg"
            _save_image_blob(blob, "image/tiff", out)
            # Pillow should have re-encoded it as a valid JPEG
            with Image.open(out) as img:
                self.assertEqual(img.format, "JPEG")
                self.assertEqual(img.mode, "RGB")

    def test_undecodable_blob_saved_raw(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.jpg"
            _save_image_blob(b"garbage-not-an-image", "application/octet-stream", out)
            self.assertEqual(out.read_bytes(), b"garbage-not-an-image")


class TestPptxConvertsNonJpegBlob(unittest.TestCase):
    def test_picture_with_tiff_content_type_is_converted(self):
        with tempfile.TemporaryDirectory() as d:
            out_dir = Path(d) / "images"
            out_dir.mkdir(parents=True)

            buf = io.BytesIO()
            Image.new("RGB", (50, 40), color="green").save(buf, "PNG")
            blob = buf.getvalue()

            class FakeImage:
                @property
                def blob(self):
                    return blob

                content_type = "image/tiff"

            class FakeShape:
                name = "Pic@2"
                image = FakeImage()

            class FakeSlide:
                shapes = [FakeShape()]

            class FakePresentation:
                slides = [FakeSlide()]

            with patch("ingestion.image_extractor._is_pptx_picture", return_value=True), \
                 patch("pptx.Presentation", return_value=FakePresentation()):
                extract_images_from_pptx(Path(d) / "x.pptx", out_dir)

            # filename = re.sub(r"\W", "", "Pic@2") + ".jpg" => "Pic2.jpg"
            out_file = out_dir / "Pic2.jpg"
            self.assertTrue(out_file.exists())
            with Image.open(out_file) as img:
                self.assertEqual(img.format, "JPEG")


class TestDocxNonImageEntriesSkipped(unittest.TestCase):
    def test_non_image_extensions_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            docx_path = Path(d) / "test.docx"
            out_dir = Path(d) / "images"

            buf = io.BytesIO()
            Image.new("RGB", (60, 40), color="blue").save(buf, "PNG")
            img_data = buf.getvalue()

            with zipfile.ZipFile(docx_path, "w") as zf:
                zf.writestr("word/media/image1.png", img_data)
                zf.writestr("word/media/notes.xml", "<xml/>")  # skipped: not an image ext
                zf.writestr("word/media/data.bin", b"binary")  # skipped

            hash_map = extract_images_from_docx(docx_path, out_dir)

            self.assertIn("image1.png", hash_map.values())
            self.assertNotIn("notes.xml", hash_map.values())
            self.assertFalse((out_dir / "notes.xml").exists())
            self.assertFalse((out_dir / "data.bin").exists())

    def test_bad_zip_logs_and_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.docx"
            bad.write_bytes(b"this is not a zip archive")
            out_dir = Path(d) / "images"
            hash_map = extract_images_from_docx(bad, out_dir)
            self.assertEqual(hash_map, {})


if __name__ == "__main__":
    unittest.main()
