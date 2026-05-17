from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestionDocument:
    doc_id: str
    source_file: str
    original_file: str
    source_path: Path
    markdown_path: Path
    original_extension: str
    raw_file_hash: str
    markdown_hash: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionStageResult:
    name: str
    status: str
    elapsed_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DocumentIngestionResult:
    source_path: str
    source_file: str | None = None
    original_file: str | None = None
    raw_file_hash: str | None = None
    status: str = "failed"
    reason: str = ""
    indexed: bool = False
    course_updated: bool = False
    parent_count: int = 0
    child_count: int = 0
    stages: list[IngestionStageResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AddDocumentsResult:
    added: int = 0
    skipped: int = 0
    failed: int = 0
    course_updated: int = 0
    documents: list[DocumentIngestionResult] = field(default_factory=list)


@dataclass
class DeleteDocumentResult:
    success: bool
    source_file: str
    vector_deleted: bool = False
    parent_deleted: bool = False
    manifest_deleted: bool = False
    markdown_deleted: bool = False
    cleaning_outputs_deleted: bool = False
    course_updated: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class DocumentInfo:
    source_file: str
    original_file: str
    original_extension: str
    raw_file_hash: str | None
    markdown_hash: str | None
    parent_count: int
    child_count: int
    courses: list[str]
    updated_at: str | None
    status: str


@dataclass
class DocumentDetail:
    info: DocumentInfo
    markdown_path: str | None
    parent_ids: list[str]
    child_ids: list[str]
    stage_stats: list[dict[str, Any]]
    last_error: str | None
    course_names: list[str]


@dataclass
class KnowledgeBaseStats:
    document_count: int
    parent_count: int
    child_count: int
    course_count: int
