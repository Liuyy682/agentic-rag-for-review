from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path


PAGE_MARKER_RE = re.compile(
    r"^\s*-{3,}\s*end\s+of\s+page\.page_number\s*=\s*(\d+)\s*-{3,}\s*$",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
BULLET_RE = re.compile(r"^\s*[-*+]\s+\S")
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+\S")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class CleanEvent:
    source_file: str
    page_number: int | None
    text: str
    action: str
    reason: str
    confidence: float


@dataclass
class CleanCandidate:
    source_file: str
    page_number: int | None
    text: str
    action: str
    reason: str
    confidence: float


@dataclass
class PageBlock:
    source_file: str
    page_number: int | None
    raw_text: str
    raw_lines: list[str]
    cleaned_text: str = ""
    cleaned_lines: list[str] = field(default_factory=list)
    slide_title: str | None = None
    removed_items: list[CleanEvent] = field(default_factory=list)
    candidates: list[CleanCandidate] = field(default_factory=list)


@dataclass
class CleanedMarkdown:
    source_file: str
    cleaned_text: str
    pages: list[PageBlock]
    events: list[CleanEvent]
    candidates: list[CleanCandidate]


def clean_markdown_text(
    markdown_text: str,
    source_file: str = "",
    scan_lines: int = 3,
    min_repeat_pages: int = 3,
    min_repeat_ratio: float = 0.3,
) -> CleanedMarkdown:
    """Clean PyMuPDF4LLM Markdown conservatively and keep audit metadata."""
    pages = parse_pages(markdown_text, source_file)
    total_pages = len(pages)
    repeated_edge_lines = _detect_repeated_edge_lines(
        pages,
        scan_lines=scan_lines,
        min_repeat_pages=min_repeat_pages,
        min_repeat_ratio=min_repeat_ratio,
    )

    events: list[CleanEvent] = []
    candidates: list[CleanCandidate] = []

    for page in pages:
        page.slide_title = _extract_slide_title(page.raw_lines)
        cleaned_lines = []
        in_code_block = False

        for index, line in enumerate(page.raw_lines):
            stripped = line.strip()

            # Do not apply line-level noise rules inside fenced code blocks.
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
                continue

            # Page numbers are removed only when they match the current page
            # and appear in an edge area, which avoids deleting body numbers.
            if not in_code_block and _is_page_number_line(
                line,
                page.page_number,
                total_pages=total_pages,
                line_index=index,
                line_count=len(page.raw_lines),
                scan_lines=scan_lines,
            ):
                event = CleanEvent(
                    source_file=source_file,
                    page_number=page.page_number,
                    text=stripped,
                    action="removed",
                    reason="page_number",
                    confidence=0.95,
                )
                page.removed_items.append(event)
                events.append(event)
                continue

            normalized = _normalize_edge_line(line)
            is_repeated_noise = normalized in repeated_edge_lines

            # Repeated edge lines are removed only if they are not protected
            # Markdown content such as slide titles, bullets, lists, or tables.
            if (
                not in_code_block
                and is_repeated_noise
                and _is_edge_line(index, len(page.raw_lines), scan_lines)
                and not _is_protected_content_line(line)
                and stripped != page.slide_title
            ):
                event = CleanEvent(
                    source_file=source_file,
                    page_number=page.page_number,
                    text=stripped,
                    action="removed",
                    reason="repeated_header_footer",
                    confidence=repeated_edge_lines[normalized],
                )
                page.removed_items.append(event)
                events.append(event)
                continue

            # Protected repeated lines are logged as kept candidates so later
            # tuning can inspect why they were not deleted.
            if (
                not in_code_block
                and is_repeated_noise
                and _is_edge_line(index, len(page.raw_lines), scan_lines)
            ):
                candidate = CleanCandidate(
                    source_file=source_file,
                    page_number=page.page_number,
                    text=stripped,
                    action="kept",
                    reason="repeated_edge_line_but_protected",
                    confidence=repeated_edge_lines[normalized],
                )
                page.candidates.append(candidate)
                candidates.append(candidate)

            cleaned_lines.append(line)

        page.cleaned_lines = _trim_blank_edges(cleaned_lines)
        page.cleaned_text = "\n".join(page.cleaned_lines)

    cleaned_text = "\n\n".join(page.cleaned_text for page in pages if page.cleaned_text.strip())
    return CleanedMarkdown(
        source_file=source_file,
        cleaned_text=cleaned_text,
        pages=pages,
        events=events,
        candidates=candidates,
    )


def parse_pages(markdown_text: str, source_file: str = "") -> list[PageBlock]:
    """Split Markdown into page blocks using PyMuPDF4LLM page separators."""
    lines = markdown_text.splitlines()
    pages: list[PageBlock] = []
    current_lines: list[str] = []

    for line in lines:
        marker = PAGE_MARKER_RE.match(line)
        if marker:
            page_number = int(marker.group(1))
            pages.append(
                PageBlock(
                    source_file=source_file,
                    page_number=page_number,
                    raw_text="\n".join(current_lines),
                    raw_lines=current_lines,
                )
            )
            current_lines = []
        else:
            current_lines.append(line)

    # Markdown without page markers is still treated as one page so callers can
    # use the same cleaning pipeline for uploaded .md files.
    if current_lines or not pages:
        page_number = pages[-1].page_number + 1 if pages and pages[-1].page_number is not None else None
        pages.append(
            PageBlock(
                source_file=source_file,
                page_number=page_number,
                raw_text="\n".join(current_lines),
                raw_lines=current_lines,
            )
        )

    return pages


def write_cleaning_log(cleaned: CleanedMarkdown, log_path: str | Path) -> None:
    """Write removed and kept cleaning decisions as JSONL audit records."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(event) for event in cleaned.events]
    records.extend(asdict(candidate) for candidate in cleaned.candidates)
    with open(log_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _detect_repeated_edge_lines(
    pages: list[PageBlock],
    scan_lines: int,
    min_repeat_pages: int,
    min_repeat_ratio: float,
) -> dict[str, float]:
    """Find normalized top/bottom lines that repeat across enough pages."""
    normalized_to_pages: dict[str, set[int]] = defaultdict(set)
    page_count = len(pages)

    for page_index, page in enumerate(pages):
        for line_index in _edge_line_indexes(page.raw_lines, scan_lines):
            line = page.raw_lines[line_index]
            normalized = _normalize_edge_line(line)
            if not normalized:
                continue
            # Page number lines are handled by their own stricter rule, not by
            # repeated header/footer detection.
            if _is_page_number_line(
                line,
                page.page_number,
                total_pages=page_count,
                line_index=line_index,
                line_count=len(page.raw_lines),
                scan_lines=scan_lines,
            ):
                continue
            normalized_to_pages[normalized].add(page_index)

    threshold = max(min_repeat_pages, math.ceil(page_count * min_repeat_ratio))
    repeated = {}
    for normalized, page_indexes in normalized_to_pages.items():
        if len(page_indexes) >= threshold:
            repeated[normalized] = min(0.99, len(page_indexes) / max(page_count, 1))

    return repeated


def _edge_line_indexes(lines: list[str], scan_lines: int) -> list[int]:
    """Return non-empty line indexes from the top and bottom scan windows."""
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    indexes = non_empty[:scan_lines] + non_empty[-scan_lines:]
    return list(dict.fromkeys(indexes))


def _normalize_edge_line(line: str) -> str:
    """Normalize an edge line so equivalent headers/footers compare equal."""
    normalized = line.strip()
    normalized = re.sub(r"^\s{0,3}#{1,6}\s*", "", normalized)
    normalized = re.sub(r"[*_`]+", "", normalized)
    normalized = re.sub(r"\b(page|p\.)\s*\d+\b", "page", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d+\s*/\s*\d+\b", "", normalized)
    normalized = re.sub(r"\b\d+\s+of\s+\d+\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b第\s*\d+\s*页\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _is_page_number_line(
    line: str,
    page_number: int | None,
    total_pages: int,
    line_index: int,
    line_count: int,
    scan_lines: int,
) -> bool:
    """Return True when a line is a standalone page number for this page."""
    if page_number is None or not _is_edge_line(line_index, line_count, scan_lines):
        return False

    stripped = re.sub(r"[*_`]", "", line.strip())
    if not stripped or len(stripped) > 30:
        return False

    escaped_page = re.escape(str(page_number))
    total_pattern = r"\d+"
    if total_pages > 1:
        total_pattern = rf"(?:{total_pages}|\d+)"

    patterns = [
        rf"^{escaped_page}$",
        rf"^-+\s*{escaped_page}\s*-+$",
        rf"^(?:page|p\.)\s*{escaped_page}$",
        rf"^第\s*{escaped_page}\s*页$",
        rf"^{escaped_page}\s*/\s*{total_pattern}$",
        rf"^{escaped_page}\s+of\s+{total_pattern}$",
    ]
    return any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns)


def _is_edge_line(line_index: int, line_count: int, scan_lines: int) -> bool:
    """Check whether a line is inside the configured top or bottom window."""
    return line_index < scan_lines or line_index >= max(line_count - scan_lines, 0)


def _is_protected_content_line(line: str) -> bool:
    """Detect Markdown/body structures that should not be removed as noise."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(
        pattern.match(stripped)
        for pattern in (HEADING_RE, BULLET_RE, NUMBERED_LIST_RE, TABLE_RE)
    ) or ("：" in stripped and len(stripped) > 6) or (":" in stripped and len(stripped) > 6)


def _extract_slide_title(lines: list[str]) -> str | None:
    """Extract the best slide title candidate from one page."""
    for line in lines:
        stripped = line.strip()
        if HEADING_RE.match(stripped):
            return re.sub(r"^\s{0,3}#{1,6}\s*", "", stripped).strip()

    for line in lines[:5]:
        stripped = line.strip()
        if stripped and not _is_protected_content_line(stripped):
            return stripped

    return None


def _trim_blank_edges(lines: list[str]) -> list[str]:
    """Remove leading and trailing blank lines while preserving inner spacing."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]
