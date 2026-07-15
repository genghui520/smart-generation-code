from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ..llm import LlmClient
from ..utils import ensure_dir, write_jsonl
from .candidate_filter import read_jsonl
from .scenario_taxonomy import ProtocolTaxonomy, build_rule_extraction_prompt, load_taxonomy


DEFAULT_BATCH_DIR = Path("rag_indexes/focas/rule_extraction_batches")
DEFAULT_RESULTS_DIR = Path("rag_indexes/focas/rule_extraction_results")
DEFAULT_RULE_CHUNKS_PATH = Path("rag_indexes/focas/rule_chunks.jsonl")

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
    taxonomy: ProtocolTaxonomy | None = None,
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
            build_batch_prompt(batch, max_chars_per_chunk=max_chars_per_chunk, taxonomy=taxonomy),
            encoding="utf-8",
        )
        manifests.append(manifest)

    write_jsonl(output_dir / "manifest.jsonl", manifests)
    return manifests


def extract_rules_with_llm(
    candidate_path: Path,
    results_dir: Path,
    *,
    llm_client: LlmClient,
    batch_size: int = 4,
    max_chars_per_chunk: int = 4500,
    limit_batches: int | None = None,
    taxonomy: ProtocolTaxonomy | None = None,
) -> list[Path]:
    """Call the configured chat model and save one JSON result per batch."""

    if not llm_client.enabled:
        raise RuntimeError("LLM is not configured")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    candidates = read_jsonl(candidate_path)
    ensure_dir(results_dir)
    written: list[Path] = []

    system_prompt = (
        "You are an expert in CNC industrial protocol traffic generation. "
        "Extract only rules that help generate high-coverage, distinguishable, "
        "high-quality protocol traffic. Return strict JSON only."
    )

    for batch_index, batch in enumerate(_batched(candidates, batch_size), start=1):
        if limit_batches is not None and batch_index > limit_batches:
            break
        batch_id = f"batch-{batch_index:04d}"
        result_path = results_dir / f"{batch_id}.json"
        user_prompt = build_batch_prompt(
            batch,
            max_chars_per_chunk=max_chars_per_chunk,
            taxonomy=taxonomy,
        )
        payload = llm_client.invoke_json(system_prompt, user_prompt)
        validate_rule_payload(payload, result_path)
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(result_path)

    return written


def build_batch_prompt(
    rows: list[dict[str, Any]],
    *,
    max_chars_per_chunk: int = 4500,
    taxonomy: ProtocolTaxonomy | None = None,
) -> str:
    taxonomy = taxonomy or load_taxonomy()
    sections = [build_rule_extraction_prompt(taxonomy).strip()]
    sections.append(
        "Now extract rules from the following chunks. Return one JSON object only. "
        "Because this batch contains multiple chunks, each rule must include source_chunk_id. "
        "Do not include source_file, page_start, page_end, or section_title; the merge program adds them."
    )
    sections.append(
        'Batch output format: {"rules":[{"source_chunk_id":"CHUNK_ID from input",'
        f'"rule_type":"{" | ".join(taxonomy.rule_types)}",'
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
    *,
    taxonomy: ProtocolTaxonomy | None = None,
) -> list[dict[str, Any]]:
    """Merge model JSON outputs into final rule chunks with source metadata."""

    candidate_rows = read_jsonl(candidate_path)
    candidate_resolver = CandidateResolver(candidate_rows)
    taxonomy = taxonomy or load_taxonomy()
    rules: list[dict[str, Any]] = []

    for result_file in sorted(results_dir.glob("*.json")):
        payload = read_model_json(result_file)
        for index, rule in enumerate(payload.get("rules", []), start=1):
            if not isinstance(rule, dict):
                continue
            source_chunk_id = str(rule.get("source_chunk_id") or rule.get("chunk_id") or "")
            source_chunk_id, source = candidate_resolver.resolve(source_chunk_id)
            normalized = normalize_rule(rule, taxonomy=taxonomy)
            if not normalized:
                continue
            normalized.update(source_metadata(source_chunk_id, source))
            normalized["rule_id"] = make_rule_id(
                source_chunk_id,
                normalized["rule_type"],
                index,
                protocol=taxonomy.protocol,
            )
            normalized["protocol"] = taxonomy.protocol
            normalized["knowledge_type"] = "traffic_generation_rule"
            rules.append(normalized)

    write_jsonl(output_path, rules)
    return rules


def merge_rule_extraction_merged_json(
    candidate_path: Path,
    merged_json_path: Path,
    output_path: Path,
    *,
    taxonomy: ProtocolTaxonomy | None = None,
) -> list[dict[str, Any]]:
    """Merge one already-combined model output into final rule chunks."""

    candidate_rows = read_jsonl(candidate_path)
    candidate_resolver = CandidateResolver(candidate_rows)
    taxonomy = taxonomy or load_taxonomy()
    payload = read_model_json(merged_json_path)
    rules: list[dict[str, Any]] = []
    per_source_counts: dict[tuple[str, str], int] = {}

    for rule in payload.get("rules", []):
        if not isinstance(rule, dict):
            continue
        raw_source_chunk_id = str(rule.get("source_chunk_id") or rule.get("chunk_id") or "")
        source_chunk_id, source = candidate_resolver.resolve(raw_source_chunk_id)
        normalized = normalize_rule(rule, taxonomy=taxonomy)
        if not normalized:
            continue
        key = (source_chunk_id, normalized["rule_type"])
        per_source_counts[key] = per_source_counts.get(key, 0) + 1
        normalized.update(source_metadata(source_chunk_id, source))
        if raw_source_chunk_id and raw_source_chunk_id != source_chunk_id:
            normalized["model_source_chunk_id"] = raw_source_chunk_id
            normalized["source_chunk_id_repaired"] = True
        else:
            normalized["source_chunk_id_repaired"] = False
        normalized["rule_id"] = make_rule_id(
            source_chunk_id,
            normalized["rule_type"],
            per_source_counts[key],
            protocol=taxonomy.protocol,
        )
        normalized["protocol"] = taxonomy.protocol
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


def validate_rule_payload(payload: dict[str, Any], path: Path) -> None:
    if "rules" not in payload:
        raise ValueError(f"{path} model output does not contain a rules field")
    if not isinstance(payload["rules"], list):
        raise ValueError(f"{path} model output rules field must be a list")


def strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def normalize_rule(rule: dict[str, Any], *, taxonomy: ProtocolTaxonomy | None = None) -> dict[str, Any] | None:
    taxonomy = taxonomy or load_taxonomy()
    rule_type = str(rule.get("rule_type", "")).strip()
    scenario = str(rule.get("scenario", "")).strip()
    rule_text = str(rule.get("rule_text", "")).strip()

    if rule_type not in taxonomy.allowed_rule_types:
        return None
    if scenario not in taxonomy.allowed_scenarios:
        scenario = taxonomy.fallback_scenario
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


class CandidateResolver:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.by_id = {str(row.get("chunk_id")): row for row in rows}
        self.by_page_suffix: dict[tuple[str, str, str], list[str]] = {}
        self.by_section_suffix: dict[tuple[str, str, str, str], list[str]] = {}
        for row in rows:
            chunk_id = str(row.get("chunk_id", ""))
            key = page_suffix_key(chunk_id)
            if key:
                self.by_page_suffix.setdefault(key, []).append(chunk_id)
            section_key = section_suffix_key(chunk_id)
            if section_key:
                self.by_section_suffix.setdefault(section_key, []).append(chunk_id)

    def resolve(self, source_chunk_id: str) -> tuple[str, dict[str, Any]]:
        if source_chunk_id in self.by_id:
            return source_chunk_id, self.by_id[source_chunk_id]

        repaired = repair_common_source_id(source_chunk_id)
        if repaired in self.by_id:
            return repaired, self.by_id[repaired]

        section_key = section_suffix_key(repaired)
        if section_key:
            matches = self.by_section_suffix.get(section_key, [])
            if len(matches) == 1:
                match = matches[0]
                return match, self.by_id[match]

        key = page_suffix_key(repaired)
        if key:
            matches = self.by_page_suffix.get(key, [])
            if len(matches) == 1:
                match = matches[0]
                return match, self.by_id[match]

        return source_chunk_id, {}


def repair_common_source_id(source_chunk_id: str) -> str:
    if source_chunk_id.startswith("fanucB-"):
        return "fanuc-" + source_chunk_id[len("fanuc") :]
    return source_chunk_id


def section_suffix_key(source_chunk_id: str) -> tuple[str, str, str, str] | None:
    match = re.search(r"(B-\d+[A-Z-]*_\d+)-p(\d+)-(.+)-([0-9]{3})$", source_chunk_id)
    if not match:
        return None
    section_tokens: list[str] = []
    for token in match.group(3).split("-"):
        if token.isdigit():
            section_tokens.append(token)
        else:
            break
    if not section_tokens:
        return None
    return match.group(1), match.group(2), "-".join(section_tokens), match.group(4)


def page_suffix_key(source_chunk_id: str) -> tuple[str, str, str] | None:
    match = re.search(r"(B-\d+[A-Z-]*_\d+)-p(\d+).*-([0-9]{3})$", source_chunk_id)
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def chunk_manifest(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id", ""),
        "source_file": row.get("source_file", ""),
        "page_start": row.get("page_start"),
        "page_end": row.get("page_end"),
        "section_title": row.get("section_title", ""),
        "candidate_score": row.get("candidate_score"),
    }


def make_rule_id(source_chunk_id: str, rule_type: str, index: int, *, protocol: str = "focas") -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "-", source_chunk_id).strip("-")
    if not base:
        base = "unknown-source"
    protocol_id = re.sub(r"[^A-Za-z0-9_-]+", "-", protocol).strip("-") or "protocol"
    return f"{protocol_id}-rule-{base}-{rule_type}-{index:03d}"


def _batched(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(rows), size):
        yield rows[index : index + size]
