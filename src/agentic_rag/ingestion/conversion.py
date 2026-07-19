import os
import shutil
from agentic_rag import config
from pathlib import Path
import glob
import tiktoken


SUPPORTED_DOCUMENT_EXTENSIONS = {
    extension.lower()
    for extension in getattr(config, "SUPPORTED_DOCUMENT_EXTENSIONS", [".pdf", ".md", ".docx", ".pptx"])
}


def clear_directory_contents(directory: Path) -> None:
    """Delete everything under directory but not the directory itself (safe for Docker volume / bind mount roots)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def is_supported_document(path) -> bool:
    return Path(str(path)).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def _looks_like_uri(value: str) -> bool:
    return value.strip().lower().startswith(("http:", "https:", "file:", "data:"))


def _normalize_markdown(markdown_text: str) -> str:
    return markdown_text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def _convert_with_markitdown(document_path: Path) -> str:
    from markitdown import MarkItDown

    converter = MarkItDown()
    result = converter.convert_local(document_path)
    return result.text_content


def _convert_pdf_with_pymupdf4llm(document_path: Path, image_dir: Path) -> str:
    """Convert a PDF to Markdown using pymupdf4llm, extracting embedded images.

    pymupdf4llm handles text extraction, image extraction, and writes correct
    ``![](path)`` references into the markdown in a single call.  It also emits
    ``--- end of page.page_number=N ---`` page separators that are recognised
    by the cleaning pipeline.
    """
    import pymupdf4llm  # type: ignore[import-untyped]

    return pymupdf4llm.to_markdown(
        str(document_path),
        write_images=True,
        image_path=str(image_dir),
        dpi=config.PDF_IMAGE_DPI,
        image_format=config.PDF_IMAGE_FORMAT,
        page_separators=True,
    )


def convert_document_to_markdown(
    document_path,
    output_dir=None,
    overwrite: bool = False,
    image_output_dir: Path | None = None,
) -> Path:
    document_path = Path(document_path)
    output_dir = Path(output_dir or config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = str(document_path)
    if _looks_like_uri(raw_path):
        raise ValueError(f"Remote documents are not supported: {raw_path}")
    if not document_path.is_file():
        raise ValueError(f"Document is not a local file: {document_path}")
    if not is_supported_document(document_path):
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    md_path = (output_dir / document_path.stem).with_suffix(".md")
    if md_path.exists() and not overwrite:
        return md_path

    # Resolve the per-document image output directory
    _image_output_dir = Path(image_output_dir or config.DOCUMENT_IMAGE_DIR)
    doc_image_dir = _image_output_dir / document_path.stem
    suffix = document_path.suffix.lower()

    # --- Convert to markdown ---
    if suffix == ".md":
        markdown_text = document_path.read_text(encoding="utf-8")
    elif suffix == ".pdf" and getattr(config, "PDF_EXTRACT_IMAGES", True):
        # pymupdf4llm handles text + image extraction in one call
        try:
            markdown_text = _convert_pdf_with_pymupdf4llm(document_path, doc_image_dir)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "pymupdf4llm failed for %s (%s); falling back to MarkItDown.",
                document_path.name, exc,
            )
            markdown_text = _convert_with_markitdown(document_path)
    else:
        markdown_text = _convert_with_markitdown(document_path)

    # --- Extract images for PPTX / DOCX (MarkItDown doesn't save them) ---
    if suffix == ".pptx":
        from ingestion.image_extractor import extract_images_from_pptx
        doc_image_dir.mkdir(parents=True, exist_ok=True)
        extract_images_from_pptx(document_path, doc_image_dir)
    elif suffix == ".docx":
        from ingestion.image_extractor import extract_images_from_docx, clean_docx_broken_image_refs
        doc_image_dir.mkdir(parents=True, exist_ok=True)
        extract_images_from_docx(document_path, doc_image_dir)
        markdown_text = clean_docx_broken_image_refs(markdown_text, doc_image_dir)

    markdown_text = _normalize_markdown(markdown_text)
    if not markdown_text.strip():
        raise ValueError(f"Converted Markdown is empty: {document_path.name}")

    md_path.write_bytes(markdown_text.encode("utf-8"))
    return md_path


def documents_to_markdowns(path_pattern_or_paths, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(path_pattern_or_paths, (str, Path)):
        paths = glob.glob(str(path_pattern_or_paths))
    else:
        paths = path_pattern_or_paths

    markdown_paths = []
    for document_path in map(Path, paths):
        markdown_paths.append(convert_document_to_markdown(document_path, output_dir, overwrite=overwrite))
    return markdown_paths



def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
