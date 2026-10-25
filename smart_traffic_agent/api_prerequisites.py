"""Knowledge-driven API prerequisite matching and failure classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def manifest_functions_by_category(
    manifest_rows: Iterable[dict[str, Any]], category: str,
) -> list[str]:
    return sorted({
        str(row.get("function", "")).strip()
        for row in manifest_rows
        if str(row.get("category", "")).strip().casefold() == category.casefold()
        and str(row.get("function", "")).strip()
    })


def match_prerequisite_rules(
    functions: Iterable[str],
    rules: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only rules whose documented API scope intersects the targets."""
    selected = {str(name).strip() for name in functions if str(name).strip()}
    matches: list[dict[str, Any]] = []
    for rule in rules:
        applies = {str(name).strip() for name in rule.get("applies_to", [])}
        if applies.intersection(selected):
            item = dict(rule)
            item["matched_functions"] = sorted(applies.intersection(selected))
            matches.append(item)
    return matches


def build_prerequisite_plan(
    functions: Iterable[str],
    rules: Iterable[dict[str, Any]],
    *,
    environment: str = "simulator",
    permissions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a planner-facing, data-driven prerequisite plan.

    The rule supplies parameter numbers, values, APIs and restart requirements;
    this module never invents controller-specific values.
    """
    permissions = permissions or {}
    matched = match_prerequisite_rules(functions, rules)
    conditions: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for rule in matched:
        for condition in rule.get("conditions", []):
            item = dict(condition)
            item["rule_id"] = rule.get("rule_id", "")
            item["matched_functions"] = rule.get("matched_functions", [])
            item["environment"] = environment
            conditions.append(item)
        if rule.get("preparation_operations"):
            operations.append({
                "rule_id": rule.get("rule_id", ""),
                "matched_functions": rule.get("matched_functions", []),
                "operations": rule.get("preparation_operations", []),
                "verification": rule.get("verification_operations", []),
                "restart_required": bool(rule.get("restart_required", False)),
                "safe_to_write": bool(rule.get("safe_to_write_in_isolated_simulator", False))
                and bool(permissions.get("allow_environment_writes", False)),
                "blocked_without_permission": not bool(permissions.get("allow_environment_writes", False)),
            })
    return {
        "enabled": bool(matched),
        "environment": environment,
        "matched_rules": [rule.get("rule_id", "") for rule in matched],
        "matched_functions": sorted({name for rule in matched for name in rule.get("matched_functions", [])}),
        "conditions": conditions,
        "preparation_operations": operations,
        "unknown_conditions_are_blocking": True,
        "source_evidence": [
            {"rule_id": rule.get("rule_id", ""), "source_url": rule.get("source_url", ""),
             "source_file": rule.get("source_file", ""), "page": rule.get("page", -1)}
            for rule in matched
        ],
    }


def classify_prerequisite_failure(
    function: str, status_code: int, error: str = "",
) -> dict[str, Any] | None:
    text = f"{function} {error}".upper()
    if status_code == 6 or "EW_NOOPT" in text:
        return {"kind": "option_or_controller_capability", "retryable": True,
                "requires_evidence": ["software option", "CNC parameter", "controller model", "communication mode"]}
    if status_code in {7, 13} or "EW_MODE" in text or "EW_REJECT" in text:
        return {"kind": "runtime_mode_or_device_state", "retryable": True,
                "requires_evidence": ["CNC mode", "run state", "alarm/emergency state", "required preceding operation"]}
    return None
