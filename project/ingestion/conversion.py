import os
import shutil
import config
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


def convert_document_to_markdown(document_path, output_dir=None, overwrite: bool = False) -> Path:
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

    if document_path.suffix.lower() == ".md":
        markdown_text = document_path.read_text(encoding="utf-8")
    else:
        markdown_text = _convert_with_markitdown(document_path)

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
