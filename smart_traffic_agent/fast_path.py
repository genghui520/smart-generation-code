"""Deterministic planning/code helpers for fully known, read-only APIs.

This module is deliberately small: it reuses the existing ExecutionContract,
SDK index and guarded capture recipes.  It only admits APIs whose signature and
safe read recipe are already known; everything else remains on the Agent path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .execution_contracts import API_SEQUENCE_RULES, MOTION_TARGET_APIS
from .models import PlanStep
from .sdk_type_index import build_sdk_type_index
from .parameter_coverage import input_schema

_WRITE_PREFIXES = ("cnc_wr", "pmc_wr", "cnc_set", "cnc_start", "cnc_stop", "cnc_reset", "cnc_delete")
_READ_PREFIXES = ("cnc_rd", "pmc_rd", "cnc_get", "cnc_alarm", "cnc_stat", "cnc_absolute", "cnc_machine")


def classify_known_api(function: str, *, header_path: Path, manifest_functions: Iterable[str] = ()) -> dict[str, Any]:
    """Classify knowledge without claiming runtime/controller validation."""
    name = str(function).strip()
    reasons: list[str] = []
    if not name or not header_path.is_file():
        return {"status": "unknown", "function": name, "reasons": ["missing SDK header"]}
    # Parse the active configured header directly so a documented symbol is
    # not treated as known when this controller's SDK does not export it.
    sdk_index = build_sdk_type_index(header_path)
    if name not in sdk_index.functions:
        return {"status": "unknown", "function": name, "reasons": ["not declared by active SDK header"]}
    manifest = {str(item).strip() for item in manifest_functions}
    if manifest and name not in manifest:
        return {"status": "unknown", "function": name, "reasons": ["not in local function manifest"]}
    if name in API_SEQUENCE_RULES:
        return {"status": "partially_known", "function": name, "reasons": ["session requires runtime state"]}
    if name.startswith(_WRITE_PREFIXES) or name in MOTION_TARGET_APIS:
        return {"status": "partially_known", "function": name, "reasons": ["state or device-dependent inputs"]}
    if name.startswith(_READ_PREFIXES):
        schema = input_schema(name)
        if not schema.get("documented") or schema.get("dimensions"):
            return {"status": "partially_known", "function": name,
                    "reasons": ["valid input values or coupled parameters require discovery"]}
        reasons.append("SDK declaration and read-only wrapper are deterministic")
        return {"status": "known", "function": name, "reasons": reasons}
    return {"status": "unknown", "function": name, "reasons": ["no verified deterministic recipe"]}


def build_fast_path_steps(target: str) -> list[PlanStep]:
    """Build only infrastructure plus one atomic read target."""
    return [
        PlanStep("FAST_STATUS", "before", "Read CNC status", "cnc_statinfo", protocol_function="cnc_statinfo"),
        PlanStep("FAST_ALARM", "before", "Read CNC alarm state", "cnc_alarm2", protocol_function="cnc_alarm2"),
        PlanStep("FAST_TARGET", "during", f"Read {target}", target, protocol_function=target,
                 repeat=1, interval_seconds=0.2),
        PlanStep("FAST_STATUS_AFTER", "after", "Read CNC status", "cnc_statinfo", protocol_function="cnc_statinfo"),
    ]


def eligible_read_fast_path(targets: Iterable[str], *, header_path: Path, manifest_functions: Iterable[str] = ()) -> tuple[bool, list[dict[str, Any]]]:
    facts = [classify_known_api(name, header_path=header_path, manifest_functions=manifest_functions) for name in targets]
    return len(facts) == 1 and facts[0]["status"] == "known", facts
