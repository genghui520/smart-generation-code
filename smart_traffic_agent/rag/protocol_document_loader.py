from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from ..utils import write_jsonl
from .fanuc_manual_loader import split_with_overlap


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".docx", ".txt", ".md"}
DEFAULT_MAX_CHARS = 6000
DEFAULT_OVERLAP_CHARS = 500


@dataclass(slots=True)
class SourceDocument:
    path: Path
    source_type: str
    title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None


def build_protocol_document_chunks(
    output_path: Path,
    *,
    protocol: str,
    inputs: Iterable[Path],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for source_path in iter_source_files(inputs):
        chunks.extend(
            chunks_from_source_file(
                source_path,
                protocol=protocol,
                max_chars=max_chars,
                overlap=overlap,
            )
        )

    write_jsonl(output_path, chunks)
    return chunks


def iter_source_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            for path in sorted(input_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(path)
        elif input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(input_path)
    return files


def chunks_from_source_file(
    source_path: Path,
    *,
    protocol: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    documents = parse_source_file(source_path)
    chunks: list[dict[str, Any]] = []

    for document in documents:
        text = clean_text(document.text)
        if not text:
            continue

        parts = split_with_overlap(text, max_chars=max_chars, overlap=overlap)
        for index, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": source_chunk_id(protocol, source_path, document.title, index),
                    "protocol": protocol,
                    "knowledge_type": "document_source",
                    "source_type": document.source_type,
                    "source_file": source_path.name,
                    "source_path": str(source_path),
                    "section_title": document.title,
                    "page_start": document.page_start,
                    "page_end": document.page_end,
                    "chunk_index": index,
                    "chunk_total": len(parts),
                    "text": part,
                }
            )

    return chunks


def parse_source_file(source_path: Path) -> list[SourceDocument]:
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(source_path)
    if suffix in {".html", ".htm"}:
        return [parse_html(source_path)]
    if suffix == ".docx":
        return [parse_docx(source_path)]
    if suffix in {".txt", ".md"}:
        return [parse_plain_text(source_path)]
    return []


def parse_pdf(source_path: Path) -> list[SourceDocument]:
    reader = PdfReader(str(source_path))
    documents: list[SourceDocument] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        documents.append(
            SourceDocument(
                path=source_path,
                source_type="pdf_document",
                title=f"page {page_index}",
                text=f"[Page {page_index}]\n{text}",
                page_start=page_index,
                page_end=page_index,
            )
        )

    return documents


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag.lower() in {"p", "br", "div", "section", "article", "tr", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "section", "article", "tr", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def parse_html(source_path: Path) -> SourceDocument:
    html = source_path.read_text(encoding="utf-8", errors="ignore")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = clean_text(title_match.group(1)) if title_match else source_path.stem
    parser = _TextHTMLParser()
    parser.feed(html)
    return SourceDocument(
        path=source_path,
        source_type="html_document",
        title=title or source_path.stem,
        text=parser.text(),
    )


def parse_docx(source_path: Path) -> SourceDocument:
    texts: list[str] = []
    with zipfile.ZipFile(source_path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    for paragraph in root.findall(".//w:p", namespace):
        parts = [
            node.text or ""
            for node in paragraph.findall(".//w:t", namespace)
            if node.text
        ]
        if parts:
            texts.append("".join(parts))

    return SourceDocument(
        path=source_path,
        source_type="docx_document",
        title=source_path.stem,
        text="\n".join(texts),
    )


def parse_plain_text(source_path: Path) -> SourceDocument:
    return SourceDocument(
        path=source_path,
        source_type="text_document",
        title=source_path.stem,
        text=source_path.read_text(encoding="utf-8", errors="ignore"),
    )


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_chunk_id(protocol: str, source_path: Path, title: str, index: int) -> str:
    title_id = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", title).strip("-")
    title_id = title_id[:48] or "section"
    file_id = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", source_path.stem).strip("-")
    return f"{protocol}-{file_id}-{title_id}-{index:03d}"
