"""Deterministic planning/code helpers for fully known, read-only APIs.

This module is deliberately small: it reuses the existing ExecutionContract,
SDK index and guarded capture recipes.  It only admits APIs whose signature and
safe read recipe are already known; everything else remains on the Agent path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .execution_contracts import (
    API_SEQUENCE_RULES,
    API_TRAFFIC_SESSION_RULES,
    MOTION_TARGET_APIS,
    PROGRAM_OPERATION_CHAIN_RULES,
)
from .models import PlanStep
from .sdk_type_index import build_sdk_type_index
from .parameter_coverage import input_schema
from .traffic_graph import canonical_relation_type

_WRITE_PREFIXES = ("cnc_wr", "pmc_wr", "cnc_set", "cnc_start", "cnc_stop", "cnc_reset", "cnc_delete")
_READ_PREFIXES = ("cnc_rd", "pmc_rd", "cnc_get", "cnc_alarm", "cnc_stat", "cnc_absolute", "cnc_machine")
# These APIs expose only controller-selected values (or reuse the currently
# selected path).  Their pointer-looking arguments are outputs, so they do
# not require an LLM to invent an input domain.
_DETERMINISTIC_OUTPUT_APIS = frozenset({"cnc_getpath", "cnc_setpath"})


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
    if name in _DETERMINISTIC_OUTPUT_APIS:
        return {"status": "known", "function": name, "reasons": ["path value is read from the controller and reused without guessed input"]}
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


def _relation_matches(row: dict[str, Any], source: str, target: str, relation_type: str) -> bool:
    relation = row.get("relation", row) if isinstance(row, dict) else {}
    if not isinstance(relation, dict):
        return False
    source_node = row.get("source", {}) if isinstance(row, dict) else {}
    target_node = row.get("target", {}) if isinstance(row, dict) else {}
    source_value = relation.get("source_api") or row.get("source_api") or (
        source_node.get("name", "") if isinstance(source_node, dict) else ""
    )
    target_value = relation.get("target_api") or row.get("target_api") or (
        target_node.get("name", "") if isinstance(target_node, dict) else ""
    )
    return (
        str(source_value).removeprefix("focas:") == source
        and str(target_value).removeprefix("focas:") == target
        and canonical_relation_type(str(relation.get("relation_type", ""))) == canonical_relation_type(relation_type)
        and str(relation.get("confidence_status", "documented"))
        in {"documented", "sdk_validated", "runtime_verified"}
    )


def _trcm_has_relation(trcm_api: dict[str, Any], source: str, target: str, relation_type: str) -> bool:
    """Search the normalized Planner TRCM response without assuming a backend."""
    dependencies = trcm_api.get("dependencies", {}) if isinstance(trcm_api, dict) else {}
    for row in dependencies.get("dependencies", []) if isinstance(dependencies, dict) else ():
        if _relation_matches(row, source, target, relation_type):
            return True
    dependency_plan = trcm_api.get("dependency_plan", {}) if isinstance(trcm_api, dict) else {}
    for row in dependency_plan.get("edges", []) if isinstance(dependency_plan, dict) else ():
        if _relation_matches(row, source, target, relation_type):
            return True
    chain = trcm_api.get("chain", {}) if isinstance(trcm_api, dict) else {}
    for path in chain.get("paths", []) if isinstance(chain, dict) else ():
        if not isinstance(path, dict):
            continue
        nodes = [
            str(item.get("name", "")).removeprefix("focas:")
            for item in path.get("nodes", []) if isinstance(item, dict)
        ]
        for index, row in enumerate(path.get("relations", [])):
            if _relation_matches(row, source, target, relation_type):
                return True
            # Neo4j's variable length query returns relationship properties
            # without endpoints. Recover the adjacent nodes and accept either
            # orientation because the query intentionally explores both ways.
            if index + 1 < len(nodes) and isinstance(row, dict):
                relation_name = str(row.get("relation_type", ""))
                confidence = str(row.get("confidence_status", "documented"))
                if canonical_relation_type(relation_name) == canonical_relation_type(relation_type) and confidence in {"documented", "sdk_validated", "runtime_verified"}:
                    pair = (nodes[index], nodes[index + 1])
                    if pair in {(source, target), (target, source)}:
                        return True
    return False


def eligible_session_fast_path(
    targets: Iterable[str],
    *,
    header_path: Path,
    manifest_functions: Iterable[str] = (),
    trcm_context: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check whether a documented session has a verified deterministic recipe.

    The check is intentionally evidence based.  A session API is not fast-path
    eligible just because its name looks like a transfer function: the active
    SDK, manifest, TRCM session edges, and input-provider rule must all agree.
    """
    selected = [str(item).strip() for item in targets if str(item).strip()]
    if len(selected) != 1:
        return False, {"status": "unknown", "reasons": ["session fast path requires one target"]}
    target = selected[0]
    chain = next(
        (dict(item) for item in PROGRAM_OPERATION_CHAIN_RULES.values()
         if target in tuple(item.get("session_calls", ()))),
        None,
    )
    session = API_TRAFFIC_SESSION_RULES.get(target)
    reasons: list[str] = []
    if chain is None or not isinstance(session, dict):
        return False, {"status": "partially_known", "target": target, "reasons": ["no evidence-backed session metadata"]}
    recipe = str(chain.get("deterministic_capture_recipe", "")).strip()
    if not recipe:
        reasons.append("no verified deterministic capture recipe")
    try:
        sdk_index = build_sdk_type_index(header_path)
        sdk_names = set(sdk_index.functions)
    except (OSError, ValueError, UnicodeError):
        sdk_names = set()
    session_calls = (
        str(session.get("start_api", "")), target, str(session.get("end_api", ""))
    )
    missing_sdk = [name for name in session_calls if name and name not in sdk_names]
    if missing_sdk:
        reasons.append("missing active SDK declarations: " + ", ".join(missing_sdk))
    manifest = {str(item).strip() for item in manifest_functions if str(item).strip()}
    discovery_api = str(chain.get("directory_discovery", {}).get("api", "")).strip()
    missing_manifest = [name for name in session_calls + ((discovery_api,) if discovery_api else ()) if manifest and name not in manifest]
    if missing_manifest:
        reasons.append("missing active Manifest entries: " + ", ".join(missing_manifest))
    api_context = (trcm_context or {}).get("apis", {}).get(target, {})
    api_available = isinstance(api_context, dict) and (
        bool(api_context.get("available"))
        or any(
            isinstance(api_context.get(key), dict) and bool(api_context[key].get("available"))
            for key in ("api_info", "chain", "dependencies", "parameter_rules")
        )
    )
    if not api_available:
        reasons.append("TRCM target facts unavailable")
    else:
        start_api, end_api = session_calls[0], session_calls[-1]
        if not _trcm_has_relation(api_context, start_api, target, "prepares"):
            reasons.append("TRCM missing Start -> Data session edge")
        if not _trcm_has_relation(api_context, target, end_api, "prepares"):
            reasons.append("TRCM missing Data -> End session edge")
        rules = api_context.get("parameter_rules", {})
        parameter_rules = rules.get("rules", []) if isinstance(rules, dict) else []
        has_provider = any(
            isinstance(item, dict)
            and str(item.get("parameter", "")) == "program_number"
            and str(item.get("source_api", "")).removeprefix("focas:") == discovery_api
            for item in parameter_rules
        )
        if discovery_api and not has_provider:
            reasons.append("TRCM missing program-number provider rule")
    facts = {
        "status": "known" if not reasons else "partially_known",
        "target": target,
        "kind": "session",
        "recipe": recipe,
        "session_type": session.get("session_type", ""),
        "session_calls": list(session_calls),
        "directory_discovery": chain.get("directory_discovery", {}),
        "directory_discovery_required": bool(
            chain.get("directory_discovery", {}).get("required", False)
        ) if isinstance(chain.get("directory_discovery", {}), dict) else False,
        "directory_discovery_required_if_missing_inputs": list(
            chain.get("directory_discovery", {}).get("required_if_missing_inputs", ())
        ) if isinstance(chain.get("directory_discovery", {}), dict) else [],
        "parameter_rules": list(chain.get("parameter_rules", ())),
        "reasons": reasons,
    }
    return not reasons, facts


def build_session_fast_path_steps(target: str, facts: dict[str, Any]) -> list[PlanStep]:
    """Build the ordered plan skeleton from the validated session metadata."""
    calls = [str(item) for item in facts.get("session_calls", ()) if str(item).strip()]
    discovery = facts.get("directory_discovery", {})
    steps: list[PlanStep] = []
    discovery_api = str(discovery.get("api", "")).strip() if isinstance(discovery, dict) else ""
    required_inputs = [
        str(item).strip() for item in facts.get(
            "directory_discovery_required_if_missing_inputs",
            discovery.get("required_if_missing_inputs", ()) if isinstance(discovery, dict) else (),
        ) if str(item).strip()
    ]
    verified_inputs = facts.get("verified_inputs", {})
    has_verified_inputs = isinstance(verified_inputs, dict) and all(
        key in verified_inputs and verified_inputs[key] not in (None, "", 0, "0")
        for key in required_inputs
    )
    discovery_required = bool(facts.get("directory_discovery_required", False)) or not has_verified_inputs
    if discovery_api and discovery_required:
        steps.append(PlanStep(
            "SESSION_DISCOVERY", "before", "Discover program inputs", discovery_api,
            protocol_function=discovery_api,
            expected_state="directory read succeeds; select only evidence-backed values",
        ))
    if len(calls) >= 3:
        steps.extend([
            PlanStep("SESSION_START", "before", "Open transfer session", calls[0], protocol_function=calls[0]),
            PlanStep("SESSION_DATA", "during", f"Transfer data through {target}", target,
                     protocol_function=target, repeat=1,
                     expected_state="data is called only after successful Start"),
            PlanStep("SESSION_END", "after", "Close transfer session", calls[-1], protocol_function=calls[-1]),
        ])
    return steps
