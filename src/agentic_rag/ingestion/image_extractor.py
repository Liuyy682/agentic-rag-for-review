"""Extract embedded images from PPTX and DOCX documents.

PDF image extraction is handled by pymupdf4llm directly in conversion.py.
"""

import hashlib
import logging
import re
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Match markdown image references like ![alt](url)
IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def extract_images_from_pptx(pptx_path: Path, output_dir: Path) -> None:
    """Extract embedded images from a PPTX file.

    Images are saved with the same naming convention that MarkItDown uses:
    ``re.sub(r"\\W", "", shape.name) + ".jpg"``.  This ensures the
    ``![alt](filename)`` references that MarkItDown writes into the markdown
    will resolve to actual files on disk.

    Parameters
    ----------
    pptx_path : Path
        Path to the .pptx file.
    output_dir : Path
        Directory to save extracted images into (created if needed).
    """
    try:
        import pptx  # type: ignore[import-untyped]
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        logger.warning("python-pptx is not installed; cannot extract images from PPTX.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        presentation = pptx.Presentation(str(pptx_path))
    except Exception as exc:
        logger.warning("Failed to open PPTX %s: %s", pptx_path.name, exc)
        return

    for slide_num, slide in enumerate(presentation.slides, start=1):
        for shape in slide.shapes:
            if not _is_pptx_picture(shape):
                continue

            try:
                image_blob = shape.image.blob
                content_type = shape.image.content_type or ""
            except AttributeError:
                logger.debug(
                    "PPTX shape %r on slide %d has no image blob (may be linked or EMF/WMF).",
                    shape.name,
                    slide_num,
                )
                continue

            # Match MarkItDown's naming: re.sub(r"\W", "", shape.name) + ".jpg"
            filename = re.sub(r"\W", "", shape.name) + ".jpg"
            out_path = output_dir / filename

            # Convert to JPEG if the blob is in another format
            _save_image_blob(image_blob, content_type, out_path)


def _is_pptx_picture(shape) -> bool:
    """Return True if *shape* is a picture shape with usable image data."""
    try:
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        return False
    return shape.shape_type == MSO_SHAPE_TYPE.PICTURE


def _save_image_blob(blob: bytes, content_type: str, out_path: Path) -> None:
    """Save *blob* to *out_path*, converting to JPEG via Pillow if needed."""
    from PIL import Image

    # Common formats that Pillow can save directly — keep original
    if content_type in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        out_path.write_bytes(blob)
        return

    # Try to decode and save as JPEG (handles EMF/WMF/TIFF that Pillow supports)
    try:
        from io import BytesIO

        img = Image.open(BytesIO(blob))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(out_path, "JPEG")
    except Exception:
        logger.debug("Cannot decode image blob for %s; saving raw.", out_path.name)
        out_path.write_bytes(blob)


# ---------------------------------------------------------------------------
# DOCX helpers
# ---------------------------------------------------------------------------

# MIME types and extensions for images that may appear inside DOCX archives
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg"}


def extract_images_from_docx(docx_path: Path, output_dir: Path) -> dict[str, str]:
    """Extract images from a DOCX file (ZIP archive).

    Images live under ``word/media/`` inside the archive.  Each extracted file
    is saved to *output_dir*.

    Returns
    -------
    dict[str, str]
        Mapping from ``{md5_hex: filename}`` for downstream data-URI matching,
        though currently unused because mammoth strips the base64 payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_map: dict[str, str] = {}

    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            for entry in zf.namelist():
                if not entry.startswith("word/media/"):
                    continue
                name = Path(entry).name
                ext = Path(name).suffix.lower()
                if ext not in _IMAGE_EXTENSIONS:
                    continue

                data = zf.read(entry)
                out_path = output_dir / name
                out_path.write_bytes(data)
                md5 = hashlib.md5(data).hexdigest()
                hash_map[md5] = name
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Failed to open DOCX as ZIP %s: %s", docx_path.name, exc)

    return hash_map


def clean_docx_broken_image_refs(markdown_text: str, image_dir: Path) -> str:
    """Replace broken ``![alt](data:image/...;base64...)`` references with
    links to actually-extracted image files, and append references for any
    additional extracted images not already mentioned.

    MarkItDown's DOCX converter strips the base64 payload from data URIs
    (leaving only ``data:image/png;base64...``), so these placeholders are
    useless for retrieval.  We replace them with file references to images
    we extracted from the DOCX archive.
    """
    # Collect extracted image files
    extracted_files: list[str] = []
    if image_dir.exists():
        for f in sorted(image_dir.iterdir()):
            if f.suffix.lower() in _IMAGE_EXTENSIONS:
                extracted_files.append(f.name)

    if not extracted_files:
        return markdown_text

    # Find all broken data:image references
    data_uri_positions: list[tuple[int, int]] = []  # (start, end) of the reference
    for m in IMAGE_MARKDOWN_RE.finditer(markdown_text):
        ref = m.group(1).strip()
        if ref.startswith("data:"):
            data_uri_positions.append((m.start(), m.end()))

    result_parts: list[str] = []
    cursor = 0
    file_idx = 0

    for ref_start, ref_end in data_uri_positions:
        # Keep text before this reference
        result_parts.append(markdown_text[cursor:ref_start])
        if file_idx < len(extracted_files):
            filename = extracted_files[file_idx]
            result_parts.append(f"![]({filename})")
            file_idx += 1
        # else: drop the broken reference entirely (don't append anything)
        cursor = ref_end

    # Append any remaining markdown after the last replaced reference
    result_parts.append(markdown_text[cursor:])

    # Append references for any extra extracted images not matched to a placeholder
    if file_idx < len(extracted_files):
        result_parts.append("\n")
        for filename in extracted_files[file_idx:]:
            result_parts.append(f"![]({filename})\n")

    return "".join(result_parts)
