import os
import shutil
import config
import pymupdf.layout
import pymupdf4llm
from pathlib import Path
import glob
import tiktoken
from ingestion.image_describer import enhance_markdown_image_references


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

def pdf_to_markdown(pdf_path, output_dir):
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        image_dir = Path(config.DOCUMENT_IMAGE_DIR) / pdf_path.stem
        if config.PDF_EXTRACT_IMAGES:
            image_dir.mkdir(parents=True, exist_ok=True)

        md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=not config.PDF_EXTRACT_IMAGES,
            write_images=config.PDF_EXTRACT_IMAGES,
            image_path=str(image_dir) if config.PDF_EXTRACT_IMAGES else None,
            dpi=config.PDF_IMAGE_DPI,
            image_format=config.PDF_IMAGE_FORMAT,
        )
        if config.VLM_IMAGE_CAPTION_ENABLED:
            md = enhance_markdown_image_references(
                md,
                markdown_dir=Path(output_dir),
                image_root=image_dir,
            )

        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = Path(output_dir) / pdf_path.stem
        Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
    finally:
        doc.close()

def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)

def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
