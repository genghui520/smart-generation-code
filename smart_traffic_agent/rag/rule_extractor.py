from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..utils import ensure_dir, write_jsonl
from .candidate_filter import read_jsonl
from .scenario_taxonomy import RULE_TYPES, SCENARIOS, build_rule_extraction_prompt


DEFAULT_BATCH_DIR = Path("rag_indexes/focas/rule_extraction_batches")
DEFAULT_RESULTS_DIR = Path("rag_indexes/focas/rule_extraction_results")
DEFAULT_RULE_CHUNKS_PATH = Path("rag_indexes/focas/rule_chunks.jsonl")

ALLOWED_RULE_TYPES = set(RULE_TYPES)
ALLOWED_SCENARIOS = {scenario.scenario_id for scenario in SCENARIOS}
LIST_FIELDS = [
    "traffic_value",
    "applicable_environment",
    "nc_program_requirements",
    "operation_sequence",
    "collection_timing",
    "recommended_api_functions",
    "allowed_operations",
    "restricted_operations",
    "abnormal_traffic_allowed",
    "distinguishing_signals",
    "quality_checks",
]


def prepare_rule_extraction_batches(
    candidate_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 8,
    max_chars_per_chunk: int = 4500,
) -> list[dict[str, Any]]:
    """Write prompt batches for model-based rule extraction.

    The model still does the semantic extraction, but this function removes the
    tedious part: it packs candidate chunks into repeatable, traceable prompts.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be greater than zero")

    candidates = read_jsonl(candidate_path)
    ensure_dir(output_dir)

    manifests: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_batched(candidates, batch_size), start=1):
        batch_id = f"batch-{batch_index:04d}"
        prompt_path = output_dir / f"{batch_id}.md"
        result_path = output_dir.parent / "rule_extraction_results" / f"{batch_id}.json"
        manifest = {
            "batch_id": batch_id,
            "prompt_file": str(prompt_path),
            "result_file": str(result_path),
            "chunk_count": len(batch),
            "chunks": [chunk_manifest(row) for row in batch],
        }
        prompt_path.write_text(
            build_batch_prompt(batch, max_chars_per_chunk=max_chars_per_chunk),
            encoding="utf-8",
        )
        manifests.append(manifest)

    write_jsonl(output_dir / "manifest.jsonl", manifests)
    return manifests


def build_batch_prompt(rows: list[dict[str, Any]], *, max_chars_per_chunk: int = 4500) -> str:
    sections = [build_rule_extraction_prompt().strip()]
    sections.append(
        "Now extract rules from the following chunks. Return one JSON object only. "
        "Because this batch contains multiple chunks, each rule must include source_chunk_id. "
        "Do not include source_file, page_start, page_end, or section_title; the merge program adds them."
    )
    sections.append(
        'Batch output format: {"rules":[{"source_chunk_id":"CHUNK_ID from input",'
        '"rule_type":"nc_rule | operation_rule | collection_rule | safety_rule",'
        '"scenario":"one scenario_id from the taxonomy","traffic_value":[],'
        '"applicable_environment":["simulator"],"rule_text":"",'
        '"nc_program_requirements":[],"operation_sequence":[],"collection_timing":[],'
        '"recommended_api_functions":[],"allowed_operations":[],"restricted_operations":[],'
        '"abnormal_traffic_allowed":[],"distinguishing_signals":[],"quality_checks":[]}]}'
    )
    for row in rows:
        sections.append(format_chunk_for_prompt(row, max_chars=max_chars_per_chunk))
    return "\n\n".join(sections).strip() + "\n"


def format_chunk_for_prompt(row: dict[str, Any], *, max_chars: int) -> str:
    text = str(row.get("text", ""))
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"

    page_start = row.get("page_start", "")
    page_end = row.get("page_end", page_start)
    return "\n".join(
        [
            "---",
            f"CHUNK_ID: {row.get('chunk_id', '')}",
            f"SOURCE_FILE: {row.get('source_file', '')}",
            f"PAGES: {page_start}-{page_end}",
            f"SECTION: {row.get('section_title', '')}",
            "TEXT:",
            text,
        ]
    )


def merge_rule_extraction_results(
    candidate_path: Path,
    results_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Merge model JSON outputs into final rule chunks with source metadata."""

    candidates = {str(row.get("chunk_id")): row for row in read_jsonl(candidate_path)}
    rules: list[dict[str, Any]] = []

    for result_file in sorted(results_dir.glob("*.json")):
        payload = read_model_json(result_file)
        for index, rule in enumerate(payload.get("rules", []), start=1):
            if not isinstance(rule, dict):
                continue
            source_chunk_id = str(rule.get("source_chunk_id") or rule.get("chunk_id") or "")
            source = candidates.get(source_chunk_id, {})
            normalized = normalize_rule(rule)
            if not normalized:
                continue
            normalized.update(source_metadata(source_chunk_id, source))
            normalized["rule_id"] = make_rule_id(source_chunk_id, normalized["rule_type"], index)
            normalized["protocol"] = "focas"
            normalized["knowledge_type"] = "traffic_generation_rule"
            rules.append(normalized)

    write_jsonl(output_path, rules)
    return rules


def read_model_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        raw = strip_markdown_fence(raw)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if "rules" not in data:
        raise ValueError(f"{path} does not contain a rules field")
    if not isinstance(data["rules"], list):
        raise ValueError(f"{path} rules field must be a list")
    return data


def strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize_rule(rule: dict[str, Any]) -> dict[str, Any] | None:
    rule_type = str(rule.get("rule_type", "")).strip()
    scenario = str(rule.get("scenario", "")).strip()
    rule_text = str(rule.get("rule_text", "")).strip()

    if rule_type not in ALLOWED_RULE_TYPES:
        return None
    if scenario not in ALLOWED_SCENARIOS:
        scenario = "general_status_collection"
    if not rule_text:
        return None

    normalized: dict[str, Any] = {
        "rule_type": rule_type,
        "scenario": scenario,
        "rule_text": rule_text,
    }
    for field in LIST_FIELDS:
        normalized[field] = normalize_list(rule.get(field))
    return normalized


def normalize_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    return [value]


def source_metadata(source_chunk_id: str, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": source.get("source_file", ""),
        "source_chunk_id": source_chunk_id,
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "section_title": source.get("section_title", ""),
        "source_candidate_score": source.get("candidate_score"),
        "source_candidate_reason": source.get("candidate_reason", []),
    }


def chunk_manifest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id", ""),
        "source_file": row.get("source_file", ""),
        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),
        "section_title": row.get("section_title", ""),
        "candidate_score": row.get("candidate_score"),
    }


def make_rule_id(source_chunk_id: str, rule_type: str, index: int) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", source_chunk_id).strip("-")
    if not base:
        base = "unknown-source"
    return f"focas-rule-{base}-{rule_type}-{index:03d}"


def _batched(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]
