from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..utils import write_json, write_jsonl
from .scenario_taxonomy import RULE_TYPES, SCENARIOS


DEFAULT_API_CHUNKS_PATH = Path("rag_indexes/focas/chunks.jsonl")
DEFAULT_RULE_CHUNKS_PATH = Path("rag_indexes/focas/rule_chunks.jsonl")
DEFAULT_SCENARIO_KNOWLEDGE_PATH = Path("rag_indexes/focas/scenario_knowledge.json")
DEFAULT_SCENARIO_CHUNKS_PATH = Path("rag_indexes/focas/scenario_chunks.jsonl")


RULE_TYPE_LIMITS = {
    "nc_rule": 5,
    "operation_rule": 5,
    "collection_rule": 5,
    "safety_rule": 5,
}


def build_scenario_knowledge(
    *,
    api_chunks_path: Path = DEFAULT_API_CHUNKS_PATH,
    rule_chunks_path: Path = DEFAULT_RULE_CHUNKS_PATH,
    output_path: Path = DEFAULT_SCENARIO_KNOWLEDGE_PATH,
    scenario_chunks_path: Path = DEFAULT_SCENARIO_CHUNKS_PATH,
) -> dict[str, Any]:
    api_chunks = load_jsonl(api_chunks_path)
    rules = load_jsonl(rule_chunks_path)
    api_index = build_api_index(api_chunks)
    rules_by_scenario = group_rules_by_scenario(rules)

    scenarios = []
    for definition in SCENARIOS:
        scenario_rules = rules_by_scenario.get(definition.scenario_id, [])
        organized_rules = organize_rules(scenario_rules)
        api_functions = collect_api_functions(definition.recommended_api_functions, scenario_rules)
        api_rules = build_api_rules(api_functions, api_index)
        scenarios.append(
            {
                "scenario_id": definition.scenario_id,
                "name": definition.name,
                "goal": definition.goal,
                "traffic_value": definition.traffic_value,
                "typical_nc_program": definition.typical_nc_program,
                "operation_phases": definition.operation_phases,
                "distinguishing_signals": definition.distinguishing_signals,
                "rules": {
                    "nc_rule": organized_rules["nc_rule"],
                    "api_rule": api_rules,
                    "operation_rule": organized_rules["operation_rule"],
                    "collection_rule": organized_rules["collection_rule"],
                    "safety_rule": organized_rules["safety_rule"],
                },
                "statistics": {
                    "source_rule_count": len(scenario_rules),
                    "organized_rule_count": sum(len(items) for items in organized_rules.values()) + len(api_rules),
                    "api_function_count": len(api_rules),
                },
            }
        )

    payload = {
        "protocol": "focas",
        "knowledge_type": "scenario_centered_knowledge",
        "description": "Scenario-centered organization of API knowledge and traffic-generation rules.",
        "rule_types": {
            **RULE_TYPES,
            "api_rule": "定义不同流量场景应调用哪些 FOCAS API，以及 API 的功能、参数和适用对象。",
        },
        "scenarios": scenarios,
        "statistics": {
            "scenario_count": len(scenarios),
            "source_rule_count": len(rules),
            "api_chunk_count": len(api_chunks),
            "scenario_chunk_count": len(scenarios),
        },
    }

    write_json(output_path, payload)
    write_jsonl(scenario_chunks_path, scenario_to_chunks(scenarios))
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_rules_by_scenario(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        scenario = str(rule.get("scenario", "general_status_collection"))
        grouped[scenario].append(rule)
    return grouped


def organize_rules(rules: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    organized = {rule_type: [] for rule_type in RULE_TYPE_LIMITS}
    for rule_type, limit in RULE_TYPE_LIMITS.items():
        candidates = [rule for rule in rules if rule.get("rule_type") == rule_type]
        candidates.sort(key=rule_score, reverse=True)
        organized[rule_type] = [compact_rule(rule) for rule in candidates[:limit]]
    return organized


def rule_score(rule: dict[str, Any]) -> tuple[int, int]:
    filled_fields = 0
    for field in [
        "nc_program_requirements",
        "operation_sequence",
        "collection_timing",
        "recommended_api_functions",
        "allowed_operations",
        "restricted_operations",
        "abnormal_traffic_allowed",
        "distinguishing_signals",
        "quality_checks",
    ]:
        if rule.get(field):
            filled_fields += 1
    return filled_fields, len(str(rule.get("rule_text", "")))


def compact_rule(rule: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "rule_id",
        "rule_type",
        "rule_text",
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
        "source_file",
        "source_chunk_id",
        "page_start",
        "page_end",
        "section_title",
    ]
    return {field: rule.get(field) for field in fields if rule.get(field) not in (None, "", [])}


def build_api_index(api_chunks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    index: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for chunk in api_chunks:
        function = str(chunk.get("function", ""))
        chunk_type = str(chunk.get("chunk_type", ""))
        if function:
            index[function][chunk_type].append(chunk)
    return index


def collect_api_functions(taxonomy_apis: list[str], rules: list[dict[str, Any]]) -> list[str]:
    counter: Counter[str] = Counter()
    for function in taxonomy_apis:
        counter[function] += 3
    for rule in rules:
        for function in rule.get("recommended_api_functions", []):
            if isinstance(function, str) and function.strip():
                counter[function.strip()] += 1
    return [function for function, _ in counter.most_common(8)]


def build_api_rules(
    functions: list[str],
    api_index: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    api_rules = []
    for function in functions:
        chunks_by_type = api_index.get(function, {})
        chunk = first_chunk(chunks_by_type, ["catalog", "overview", "prototype", "arguments"])
        if not chunk:
            api_rules.append({"function": function, "rule_text": f"调用 {function} 支撑该场景的数据采集或操作。"})
            continue
        summary = summarize_api(function, chunks_by_type)
        api_rules.append(
            {
                "function": function,
                "rule_text": summary,
                "category": chunk.get("category", ""),
                "source_url": chunk.get("source_url", ""),
                "source_chunk_ids": [
                    item.get("chunk_id", "")
                    for items in chunks_by_type.values()
                    for item in items[:1]
                    if item.get("chunk_id")
                ][:4],
            }
        )
    return api_rules


def first_chunk(chunks_by_type: dict[str, list[dict[str, Any]]], chunk_types: list[str]) -> dict[str, Any] | None:
    for chunk_type in chunk_types:
        rows = chunks_by_type.get(chunk_type)
        if rows:
            return rows[0]
    for rows in chunks_by_type.values():
        if rows:
            return rows[0]
    return None


def summarize_api(function: str, chunks_by_type: dict[str, list[dict[str, Any]]]) -> str:
    overview = first_chunk(chunks_by_type, ["overview", "catalog"])
    prototype = first_chunk(chunks_by_type, ["prototype"])
    arguments = first_chunk(chunks_by_type, ["arguments"])
    parts = [f"调用 {function} 支撑该场景。"]
    if overview:
        parts.append(clean_preview(str(overview.get("text", "")), function))
    if prototype:
        parts.append(clean_preview(str(prototype.get("text", "")), function))
    if arguments:
        parts.append(clean_preview(str(arguments.get("text", "")), function))
    return " ".join(part for part in parts if part).strip()


def clean_preview(text: str, function: str, max_chars: int = 220) -> str:
    text = " ".join(text.split())
    text = text.replace(f"Function: {function}", "").strip()
    return text[:max_chars].rstrip()


def scenario_to_chunks(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for scenario in scenarios:
        rules = scenario["rules"]
        text_parts = [
            f"Scenario: {scenario['scenario_id']}",
            f"Name: {scenario['name']}",
            f"Goal: {scenario['goal']}",
            "Traffic value: " + "; ".join(scenario.get("traffic_value", [])),
            "Typical NC program: " + "; ".join(scenario.get("typical_nc_program", [])),
            "Distinguishing signals: " + "; ".join(scenario.get("distinguishing_signals", [])),
        ]
        for rule_type in ["nc_rule", "api_rule", "operation_rule", "collection_rule", "safety_rule"]:
            items = rules.get(rule_type, [])
            if items:
                text_parts.append(f"{rule_type}:")
                text_parts.extend(f"- {item.get('rule_text', '')}" for item in items[:3])
        chunks.append(
            {
                "chunk_id": f"focas-scenario-{scenario['scenario_id']}",
                "protocol": "focas",
                "knowledge_type": "scenario_centered_knowledge",
                "source_type": "scenario_organization",
                "scenario": scenario["scenario_id"],
                "rule_types": ["nc_rule", "api_rule", "operation_rule", "collection_rule", "safety_rule"],
                "text": "\n".join(text_parts),
            }
        )
    return chunks


def main() -> None:
    payload = build_scenario_knowledge()
    print(f"Scenario knowledge: {len(payload['scenarios'])}")
    print(f"Output: {DEFAULT_SCENARIO_KNOWLEDGE_PATH}")
    print(f"Chunks: {DEFAULT_SCENARIO_CHUNKS_PATH}")


if __name__ == "__main__":
    main()
