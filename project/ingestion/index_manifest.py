from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import config


INDEX_SCHEMA_VERSION = 4


def _stable_json_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_index_config() -> dict:
    """Return the settings that affect generated chunks and stored vectors."""
    chunker_config = {
        "headers_to_split_on": config.HEADERS_TO_SPLIT_ON,
        "min_parent_size": config.MIN_PARENT_SIZE,
        "max_parent_size": config.MAX_PARENT_SIZE,
        "child_chunk_size": config.CHILD_CHUNK_SIZE,
        "child_chunk_overlap": config.CHILD_CHUNK_OVERLAP,
    }
    cleaner_config = {
        "markdown_cleaning_enabled": config.MARKDOWN_CLEANING_ENABLED,
        "header_footer_scan_lines": config.HEADER_FOOTER_SCAN_LINES,
        "min_repeat_pages": config.MIN_REPEAT_PAGES,
        "min_repeat_ratio": config.MIN_REPEAT_RATIO,
    }
    embedding_config = {
        "dense_model": config.DENSE_MODEL,
        "dense_embedding_dimension": config.DENSE_EMBEDDING_DIMENSION,
        "dense_query_instruction": config.DENSE_QUERY_INSTRUCTION,
        "dense_normalize_embeddings": config.DENSE_NORMALIZE_EMBEDDINGS,
        "sparse_retrieval_backend": config.SPARSE_RETRIEVAL_BACKEND,
    }
    converter_config = {
        "document_converter": getattr(config, "DOCUMENT_CONVERTER", "markitdown"),
        "supported_document_extensions": getattr(config, "SUPPORTED_DOCUMENT_EXTENSIONS", [".pdf", ".md", ".docx", ".pptx"]),
    }
    return {
        "document_converter": converter_config["document_converter"],
        "dense_model": config.DENSE_MODEL,
        "sparse_retrieval_backend": config.SPARSE_RETRIEVAL_BACKEND,
        "converter_config_hash": _stable_json_hash(converter_config),
        "chunker_config_hash": _stable_json_hash(chunker_config),
        "cleaner_config_hash": _stable_json_hash(cleaner_config),
        "embedding_config_hash": _stable_json_hash(embedding_config),
    }


def page_number_key(page_number: int | None) -> str:
    return str(page_number if page_number is not None else 1)


def normalize_page_number(page_number: int | str | None) -> int:
    if page_number is None:
        return 1
    return int(page_number)


def build_page_hashes(pages) -> dict[str, str]:
    """Hash the cleaned page text because this is the content that enters RAG."""
    return {
        page_number_key(page.page_number): text_hash(page.cleaned_text)
        for page in pages
        if page.cleaned_text.strip()
    }


def add_chunks_to_document(
    document: dict,
    parent_chunks: list,
    child_chunks: list,
    new_page_hashes: dict[str, str],
) -> dict:
    """Attach written chunks to the document manifest record."""
    next_document = dict(document)
    next_document.setdefault("pages", {})
    next_document.setdefault("parent_pages", {})
    parent_ids = []
    child_ids = []

    for page_number, page_hash in new_page_hashes.items():
        next_document["pages"].setdefault(
            page_number,
            {"page_hash": page_hash, "parent_ids": [], "child_ids": []},
        )
        next_document["pages"][page_number]["page_hash"] = page_hash

    for parent_id, parent_doc in parent_chunks:
        parent_ids.append(parent_id)
        pages = _metadata_pages(parent_doc.metadata)
        next_document["parent_pages"][parent_id] = pages
        for page_number in pages:
            page_key = str(page_number)
            if page_key not in next_document["pages"]:
                continue
            page_data = next_document["pages"][page_key]
            if parent_id not in page_data["parent_ids"]:
                page_data["parent_ids"].append(parent_id)

    for child_doc in child_chunks:
        child_id = child_doc.metadata.get("chunk_id")
        if not child_id:
            continue
        child_ids.append(child_id)
        for page_number in _metadata_pages(child_doc.metadata):
            page_key = str(page_number)
            if page_key not in next_document["pages"]:
                continue
            page_data = next_document["pages"][page_key]
            page_child_ids = page_data["child_ids"]
            if child_id not in page_child_ids:
                page_data["child_ids"].append(child_id)

    next_document["parent_ids"] = parent_ids
    next_document["child_ids"] = child_ids
    next_document["parent_count"] = len(parent_ids)
    next_document["child_count"] = len(child_ids)
    return next_document


def build_document_record(
    doc_id: str,
    source_file: str,
    markdown_path: str | Path,
    markdown_text: str,
    page_hashes: dict[str, str],
    parent_chunks: list,
    child_chunks: list,
    original_file: str | None = None,
    source_path: str | Path | None = None,
    raw_file_hash: str | None = None,
    markdown_hash: str | None = None,
    original_extension: str | None = None,
    last_result: dict | None = None,
    created_at: str | None = None,
) -> dict:
    """Create the initial manifest entry after a full document index."""
    markdown_hash = markdown_hash or text_hash(markdown_text)
    document = {
        "doc_id": doc_id,
        "source_file": source_file,
        "original_file": original_file or source_file,
        "original_extension": original_extension or Path(original_file or source_file).suffix.lower(),
        "source_path": str(source_path) if source_path else None,
        "converter": getattr(config, "DOCUMENT_CONVERTER", "markitdown"),
        "markdown_path": str(markdown_path),
        "raw_file_hash": raw_file_hash,
        "markdown_hash": markdown_hash,
        "document_hash": markdown_hash,
        "status": "success",
        "pages": {
            page_number: {"page_hash": page_hash, "parent_ids": [], "child_ids": []}
            for page_number, page_hash in page_hashes.items()
        },
        "parent_pages": {},
        "last_result": last_result or {},
        "created_at": created_at or _utc_now(),
        "updated_at": _utc_now(),
    }
    return add_chunks_to_document(document, parent_chunks, child_chunks, page_hashes)


def _metadata_pages(metadata: dict) -> list[int]:
    pages = metadata.get("page_numbers")
    if pages is None:
        pages = [metadata.get("page_number")]
    if not isinstance(pages, list):
        pages = [pages]
    normalized = [normalize_page_number(page) for page in pages if page is not None]
    return list(dict.fromkeys(normalized))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexManifest:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(config.INDEX_STATE_DIR) / "index_manifest.json"
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return self._empty()
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Index manifest is invalid JSON: {self.path}") from exc

    def _empty(self) -> dict:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "index_config": current_index_config(),
            "documents": {},
        }

    def is_config_compatible(self) -> bool:
        return (
            self.data.get("schema_version") == INDEX_SCHEMA_VERSION
            and self.data.get("index_config") == current_index_config()
        )

    def get_document(self, source_file: str) -> dict | None:
        return self.data.get("documents", {}).get(source_file)

    def list_documents(self) -> list[dict]:
        return list(self.data.get("documents", {}).values())

    def set_document(self, source_file: str, document: dict) -> None:
        self.data.setdefault("documents", {})[source_file] = document

    def remove_document(self, source_file: str) -> dict | None:
        return self.data.setdefault("documents", {}).pop(source_file, None)

    def update_document_result(self, source_file: str, last_result: dict) -> bool:
        document = self.get_document(source_file)
        if not document:
            return False
        document["last_result"] = last_result
        document["updated_at"] = _utc_now()
        return True

    def save(self) -> None:
        """Write through a temp file so interrupted saves do not corrupt the manifest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["schema_version"] = INDEX_SCHEMA_VERSION
        self.data["index_config"] = current_index_config()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)
