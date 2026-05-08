from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config


INDEX_SCHEMA_VERSION = 1


def _stable_json_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current_index_config() -> dict:
    """Return the index-affecting settings that make local page updates safe."""
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
        "sparse_model": config.SPARSE_MODEL,
        "sparse_vector_name": config.SPARSE_VECTOR_NAME,
    }
    return {
        "dense_model": config.DENSE_MODEL,
        "sparse_model": config.SPARSE_MODEL,
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


def changed_pages(old_document: dict, new_page_hashes: dict[str, str]) -> set[int]:
    """Compare old and new page hashes; added and removed pages count as changed."""
    old_pages = old_document.get("pages", {})
    page_numbers = set(old_pages.keys()) | set(new_page_hashes.keys())
    changed: set[int] = set()
    for page_number in page_numbers:
        old_hash = old_pages.get(page_number, {}).get("page_hash")
        if old_hash != new_page_hashes.get(page_number):
            changed.add(normalize_page_number(page_number))
    return changed


def expand_with_neighbors(page_numbers: Iterable[int], available_pages: Iterable[int]) -> set[int]:
    """Include adjacent pages so overlap and cross-page parent chunks are refreshed."""
    available = set(available_pages)
    expanded: set[int] = set()
    for page_number in page_numbers:
        for candidate in (page_number - 1, page_number, page_number + 1):
            if candidate in available:
                expanded.add(candidate)
    return expanded


def stale_parent_ids(document: dict, affected_pages: Iterable[int]) -> set[str]:
    """Find existing parents whose recorded source pages overlap the rebuild window."""
    affected = set(affected_pages)
    stale: set[str] = set()
    for parent_id, page_numbers in document.get("parent_pages", {}).items():
        if affected.intersection(normalize_page_number(page) for page in page_numbers):
            stale.add(parent_id)
    return stale


def close_rebuild_scope(document: dict, seed_pages: Iterable[int], available_pages: Iterable[int]) -> tuple[set[int], set[str]]:
    """Expand rebuild pages until every stale cross-page parent is fully covered."""
    seed = set(seed_pages)
    available = set(available_pages)
    rebuild_pages = seed.intersection(available)
    stale_ids: set[str] = set()

    changed = True
    while changed:
        changed = False
        current_stale = stale_parent_ids(document, rebuild_pages | seed)
        if not current_stale.issubset(stale_ids):
            stale_ids |= current_stale
            changed = True

        for parent_id in current_stale:
            for page_number in document.get("parent_pages", {}).get(parent_id, []):
                normalized = normalize_page_number(page_number)
                if normalized in available and normalized not in rebuild_pages:
                    rebuild_pages.add(normalized)
                    changed = True

    return rebuild_pages, stale_ids


def remove_parent_ids(document: dict, parent_ids: Iterable[str], new_page_hashes: dict[str, str]) -> dict:
    """Drop stale parent/child references while preserving unaffected page records."""
    removed = set(parent_ids)
    next_document = dict(document)
    next_document["pages"] = {}
    for page_number, page_data in document.get("pages", {}).items():
        if page_number not in new_page_hashes:
            continue
        retained_parent_ids = [pid for pid in page_data.get("parent_ids", []) if pid not in removed]
        child_ids = [cid for cid in page_data.get("child_ids", []) if _child_parent_id(cid) not in removed]
        next_document["pages"][page_number] = {
            "page_hash": new_page_hashes[page_number],
            "parent_ids": retained_parent_ids,
            "child_ids": child_ids,
        }
    next_document["parent_pages"] = {
        pid: pages
        for pid, pages in document.get("parent_pages", {}).items()
        if pid not in removed
    }
    return next_document


def _child_parent_id(child_id: str) -> str:
    marker = "_child_"
    if marker not in child_id:
        return ""
    return child_id.rsplit(marker, 1)[0]


def add_chunks_to_document(
    document: dict,
    parent_chunks: list,
    child_chunks: list,
    new_page_hashes: dict[str, str],
) -> dict:
    """Attach newly written chunks to each page they cover in the manifest."""
    next_document = dict(document)
    next_document.setdefault("pages", {})
    next_document.setdefault("parent_pages", {})

    for page_number, page_hash in new_page_hashes.items():
        next_document["pages"].setdefault(
            page_number,
            {"page_hash": page_hash, "parent_ids": [], "child_ids": []},
        )
        next_document["pages"][page_number]["page_hash"] = page_hash

    for parent_id, parent_doc in parent_chunks:
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
        for page_number in _metadata_pages(child_doc.metadata):
            page_key = str(page_number)
            if page_key not in next_document["pages"]:
                continue
            child_ids = next_document["pages"][page_key]["child_ids"]
            if child_id not in child_ids:
                child_ids.append(child_id)

    return next_document


def build_document_record(
    doc_id: str,
    source_file: str,
    markdown_path: str | Path,
    markdown_text: str,
    page_hashes: dict[str, str],
    parent_chunks: list,
    child_chunks: list,
) -> dict:
    """Create the initial manifest entry after a full document index."""
    document = {
        "doc_id": doc_id,
        "source_file": source_file,
        "markdown_path": str(markdown_path),
        "document_hash": text_hash(markdown_text),
        "pages": {
            page_number: {"page_hash": page_hash, "parent_ids": [], "child_ids": []}
            for page_number, page_hash in page_hashes.items()
        },
        "parent_pages": {},
        "updated_at": _utc_now(),
    }
    return add_chunks_to_document(document, parent_chunks, child_chunks, page_hashes)


def refresh_document_record(
    document: dict,
    markdown_text: str,
    new_page_hashes: dict[str, str],
    stale_parent_ids: Iterable[str],
    parent_chunks: list,
    child_chunks: list,
) -> dict:
    """Replace stale chunk references with the chunks created by the local rebuild."""
    next_document = remove_parent_ids(document, stale_parent_ids, new_page_hashes)
    next_document = add_chunks_to_document(next_document, parent_chunks, child_chunks, new_page_hashes)
    next_document["document_hash"] = text_hash(markdown_text)
    next_document["updated_at"] = _utc_now()
    return next_document


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
        self.path = Path(path) if path else Path(config.PARENT_STORE_PATH) / "index_manifest.json"
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

    def set_document(self, source_file: str, document: dict) -> None:
        self.data.setdefault("documents", {})[source_file] = document

    def save(self) -> None:
        """Write through a temp file so interrupted saves do not corrupt the manifest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["schema_version"] = INDEX_SCHEMA_VERSION
        self.data["index_config"] = current_index_config()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)
