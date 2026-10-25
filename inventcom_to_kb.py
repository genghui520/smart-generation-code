#!/usr/bin/env python3
"""Convert Inventcom per-function page extraction into FOCAS knowledge-base
api_prerequisite rules, appended to rule_chunks.jsonl.

Grounding policy: this converter never invents values. It only promotes what
the extractor already captured (verbatim page sections + explicit parameter-bit
requirements) into the structured schema consumed by build_prerequisite_plan.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PAGES = ROOT / "inventcom_output" / "all_api_inventcom_pages.jsonl"
RULE_CHUNKS = ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
OUT = ROOT / "inventcom_output" / "api_prerequisite_rules.jsonl"

RESTART_HINTS = re.compile(
    r"power\s*(?:off|cycle)|restart|re[- ]?power|switch\s+off|重新上电|断电|重启|再次上电",
    re.I,
)
RELATED_POSTS = re.compile(
    r"\nFanuc Focas Library \|", re.I,
)


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text[:limit]


def clean_section(text: str) -> str:
    """Drop the trailing 'Fanuc Focas Library | ... [read more]' related-post
    excerpts the page renders after the primary section content."""
    text = text or ""
    match = RELATED_POSTS.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def normalized_category(category: str) -> str:
    if not category:
        return "api_prerequisite"
    return re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_") or "api_prerequisite"


def build_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    api = record.get("api", "")
    if not api:
        return None
    sections = record.get("sections", {}) or {}
    prerequisites = record.get("prerequisites", []) or []

    cnc_option = clip(clean_section(sections.get("cnc_option", "")), 1600)
    cnc_parameter = clip(clean_section(sections.get("cnc_parameter", "")), 2400)
    cnc_mode = clip(clean_section(sections.get("cnc_mode", "")), 1200)
    available_cnc = clip(clean_section(sections.get("available_cnc", "")), 1200)
    description = clip(clean_section(sections.get("description", "")), 1200)
    return_codes = clip(clean_section(sections.get("return_codes", "")), 1600)

    param_bits = [
        p for p in prerequisites
        if isinstance(p, dict) and p.get("type") == "cnc_parameter_bit_required"
    ]
    meaningful = any(
        x for x in (cnc_option, cnc_parameter, cnc_mode, available_cnc, param_bits)
    )
    if not meaningful:
        return None

    conditions: list[dict[str, Any]] = []
    preparation_operations: list[dict[str, Any]] = []
    verification_operations: list[dict[str, Any]] = []

    if cnc_option:
        conditions.append({
            "type": "software_option",
            "requirement": cnc_option,
            "check_api": "cnc_rdparam",
            "unknown_state": "block_target",
        })
    if cnc_mode:
        conditions.append({
            "type": "cnc_mode",
            "requirement": cnc_mode,
            "unknown_state": "block_target",
        })
    if available_cnc:
        conditions.append({
            "type": "controller_capability",
            "requirement": available_cnc,
            "unknown_state": "block_target",
        })

    for p in param_bits:
        number = int(p.get("number", 0))
        bit = int(p.get("bit", 0))
        value = int(p.get("value", 1))
        series = str(p.get("series_context", "")).strip()
        requirement = f"参数 {number}#{bit} 需按型号文档确认为 {value}"
        if series and series != "unresolved_series":
            requirement += f"（{series}）"
        conditions.append({
            "type": "cnc_parameter",
            "parameter_number": number,
            "bit": bit,
            "expected_value": value,
            "requirement": requirement,
            "check_api": "cnc_rdparam",
            "series_context": series,
            "unknown_state": "do_not_write",
        })
        if all(
            not (op.get("parameter_number") == number and op.get("bit") == bit)
            for op in preparation_operations
        ):
            preparation_operations.append({
            "operation": "read_parameter",
            "api": "cnc_rdparam",
            "parameter_number": number,
            "bit": bit,
        })
        if all(
            not (op.get("check") == f"{number}#{bit} equals {value}")
            for op in verification_operations
        ):
            verification_operations.append({
            "api": "cnc_rdparam",
            "check": f"{number}#{bit} equals {value}",
        })

    restart_required = bool(RESTART_HINTS.search(cnc_parameter))

    parts = [f"FANUC {api} 前置条件（来源 inventcom.net 官方函数页）。"]
    if description:
        parts.append(f"说明：{description}")
    if cnc_option:
        parts.append(f"CNC option：{cnc_option}")
    if cnc_parameter:
        parts.append(f"CNC parameter：{cnc_parameter}")
    if cnc_mode:
        parts.append(f"CNC mode：{cnc_mode}")
    if available_cnc:
        parts.append(f"Available CNC：{available_cnc}")
    if return_codes:
        parts.append(f"Return codes：{return_codes}")
    rule_text = "\n".join(parts)[:6000]

    return {
        "rule_type": "api_prerequisite",
        "rule_id": f"FANUC-{api}-prerequisite",
        "scenario": normalized_category(record.get("category", "")),
        "rule_text": rule_text,
        "applies_to": [api],
        "conditions": conditions,
        "preparation_operations": preparation_operations,
        "verification_operations": verification_operations,
        "restart_required": restart_required,
        "safe_to_write_in_isolated_simulator": False,
        "failure_policy": "按页面原文记录选项/参数/型号/模式证据；不得臆造编号或未证实状态",
        "source_url": record.get("url", ""),
        "source_file": "inventcom.net",
        "page": -1,
        "protocol": "focas",
        "knowledge_type": "api_prerequisite_rule",
    }


def main() -> None:
    records = [json.loads(line) for line in PAGES.read_text(encoding="utf-8").splitlines() if line.strip()]
    rules: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"not_fetched": 0, "no_prerequisite_sections": 0}
    seen_in_batch: set[str] = set()
    for record in records:
        if record.get("status") != "page_fetched_sections_extracted":
            skipped["not_fetched"] += 1
            continue
        rule = build_rule(record)
        if rule is None:
            skipped["no_prerequisite_sections"] += 1
            continue
        if rule["rule_id"] in seen_in_batch:
            skipped["duplicate_manifest_row"] = skipped.get("duplicate_manifest_row", 0) + 1
            continue
        seen_in_batch.add(rule["rule_id"])
        rules.append(rule)

    existing = [
        json.loads(line) for line in RULE_CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    existing_ids = {r.get("rule_id") for r in existing}
    new_rules = [r for r in rules if r["rule_id"] not in existing_ids]
    duplicates = len(rules) - len(new_rules)

    with RULE_CHUNKS.open("a", encoding="utf-8") as file:
        for rule in new_rules:
            file.write(json.dumps(rule, ensure_ascii=False) + "\n")

    with OUT.open("w", encoding="utf-8") as file:
        for rule in new_rules:
            file.write(json.dumps(rule, ensure_ascii=False) + "\n")

    print(json.dumps({
        "pages_processed": len(records),
        "skipped": skipped,
        "rules_built": len(rules),
        "rules_appended": len(new_rules),
        "duplicate_rule_ids_skipped": duplicates,
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
