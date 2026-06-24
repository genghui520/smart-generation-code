from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from ..utils import write_jsonl


DEFAULT_MANUAL_DIR = Path("E:/研二下/FANUC-0i-MF全套说明书")
DEFAULT_MANUAL_FILES = [
    "B-64604CM-2_01.PDF",
    "B-64604CM_01.PDF",
    "B-64610CM_01.PDF",
    "B-64605CM_01.PDF",
]
MAX_SECTION_CHARS = 6000
SECTION_OVERLAP_CHARS = 500


@dataclass(slots=True)
class ManualSection:
    title: str
    level: int
    page_start: int
    page_end: int


def build_fanuc_manual_chunks(
    output_path: Path,
    *,
    manual_dir: Path = DEFAULT_MANUAL_DIR,
    files: list[str] | None = None,
    max_chars: int = MAX_SECTION_CHARS,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    selected_files = files or DEFAULT_MANUAL_FILES

    for filename in selected_files:
        pdf_path = manual_dir / filename
        if not pdf_path.exists():
            continue
        chunks.extend(chunks_from_pdf(pdf_path, max_chars=max_chars))

    write_jsonl(output_path, chunks)
    return chunks


def chunks_from_pdf(pdf_path: Path, *, max_chars: int = MAX_SECTION_CHARS) -> list[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    sections = sections_from_outline(reader)
    if not sections:
        sections = fallback_sections(reader)

    chunks: list[dict[str, Any]] = []
    manual_type = infer_manual_type(pdf_path.name, reader)
    total_pages = len(reader.pages)

    for section in sections:
        page_start = max(1, min(section.page_start, total_pages))
        page_end = max(page_start, min(section.page_end, total_pages))
        text = extract_pages_text(reader, page_start, page_end)
        text = clean_manual_text(text)
        if not text:
            continue

        parts = split_with_overlap(text, max_chars=max_chars, overlap=SECTION_OVERLAP_CHARS)
        for index, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": manual_chunk_id(pdf_path.stem, page_start, section.title, index),
                    "protocol": "focas",
                    "knowledge_type": "manual_source",
                    "source_type": "fanuc_pdf_manual",
                    "source_file": pdf_path.name,
                    "manual_type": manual_type,
                    "section_title": section.title,
                    "section_level": section.level,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": index,
                    "chunk_total": len(parts),
                    "text": part,
                }
            )

    return chunks


def sections_from_outline(reader: PdfReader) -> list[ManualSection]:
    entries: list[tuple[str, int, int]] = []

    def walk(items: list[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue

            title = clean_title(getattr(item, "title", ""))
            if not title:
                continue
            try:
                page_index = reader.get_destination_page_number(item)
            except Exception:
                continue
            entries.append((title, level, page_index + 1))

    try:
        walk(reader.outline, 1)
    except Exception:
        return []

    if not entries:
        return []

    entries = sorted(entries, key=lambda entry: entry[2])
    sections: list[ManualSection] = []
    total_pages = len(reader.pages)

    for index, (title, level, page_start) in enumerate(entries):
        next_start = entries[index + 1][2] if index + 1 < len(entries) else total_pages + 1
        page_end = max(page_start, next_start - 1)
        if page_start <= total_pages:
            sections.append(ManualSection(title, level, page_start, page_end))

    return merge_tiny_sections(sections)


def fallback_sections(reader: PdfReader, *, pages_per_chunk: int = 4) -> list[ManualSection]:
    sections: list[ManualSection] = []
    total_pages = len(reader.pages)

    for page_start in range(1, total_pages + 1, pages_per_chunk):
        page_end = min(total_pages, page_start + pages_per_chunk - 1)
        sections.append(ManualSection(f"pages {page_start}-{page_end}", 1, page_start, page_end))

    return sections


def merge_tiny_sections(sections: list[ManualSection]) -> list[ManualSection]:
    merged: list[ManualSection] = []

    for section in sections:
        if merged and section.page_start == section.page_end and section.level > merged[-1].level:
            parent = merged[-1]
            parent.page_end = max(parent.page_end, section.page_end)
            continue
        merged.append(section)

    return merged


def extract_pages_text(reader: PdfReader, page_start: int, page_end: int) -> str:
    texts: list[str] = []

    for page_number in range(page_start, page_end + 1):
        try:
            text = reader.pages[page_number - 1].extract_text() or ""
        except Exception:
            text = ""
        if text:
            texts.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(texts)


def infer_manual_type(pdf_name: str, reader: PdfReader) -> str:
    title = str((reader.metadata or {}).get("/Title") or "")
    combined = f"{pdf_name} {title}".lower()

    if "parameter" in combined or "参数" in combined:
        return "parameter_manual"
    if "维修" in combined or "maintenance" in combined:
        return "maintenance_manual"
    if "操作" in combined or "operator" in combined:
        return "operation_manual"
    return "fanuc_manual"


def clean_manual_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def split_with_overlap(text: str, *, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + max_chars // 2:
                end = paragraph_break

        part = text[start:end].strip()
        if part:
            parts.append(part)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return parts


def manual_chunk_id(pdf_stem: str, page_start: int, title: str, index: int) -> str:
    title_id = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", title).strip("-")
    title_id = title_id[:48] or "section"
    return f"fanuc-{pdf_stem}-p{page_start:04d}-{title_id}-{index:03d}"
