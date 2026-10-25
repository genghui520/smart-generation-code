"""Reusable, evidence-scoped capabilities for constructing observable CNC scenes.

This registry describes what a scene *may* make observable.  It does not claim
that a controller supports an event or that a planned operation actually ran;
the Executor must record those facts from API/PCAP evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .models import ApiCallLog, PlanStep


DEFAULT_CAPABILITIES_PATH = Path("rag_indexes/focas/scenario_capabilities.json")

_DEFAULT_INVALIDATION = [
    "controller model or SDK header changes",
    "NCGuide instance/configuration changes",
    "required helper API fails or loses matched traffic",
]

# These outputs are measurements of an executing NC program.  A baseline
# read is still useful, but it cannot be treated as a scene that can produce
# the requested variation without a validated simulator execution path.
_DYNAMIC_STATE_TARGETS = {
    "cnc_rdposition", "cnc_absolute", "cnc_absolute2", "cnc_machine",
    "cnc_relative", "cnc_relative2", "cnc_distance", "cnc_actf", "cnc_acts",
    "cnc_rdspeed", "cnc_rddynamic", "cnc_rddynamic2", "cnc_rdsvmeter",
    "cnc_srvdelay", "cnc_accdecdly",
}

# Variation is split into two independent obligations.  The target API may
# have a documented input domain without requiring a state-changing scene;
# conversely, a state-reading API may need NC execution only when response
# variation is an explicit objective.  Everything else is call/observation
# coverage and must not be forced into artificial diversity.
_CALL_ONLY_APIS = {
    "cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_statinfo2",
    "cnc_alarm", "cnc_alarm2", "cnc_getdtailerr",
}
_DOCUMENTED_INPUT_VARIATION_APIS = {
    "cnc_rdposition", "cnc_absolute", "cnc_absolute2", "cnc_machine",
    "cnc_relative", "cnc_relative2", "cnc_distance", "cnc_skip",
    "cnc_srvdelay", "cnc_accdecdly", "cnc_rdaxisdata",
}


def classify_api_variation(target: str, dependency: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify independent input and response-state coverage obligations."""
    name = str(target).strip().lower()
    dependency = dependency or analyze_output_dependencies(target)
    kind = str(dependency.get("dependency_kind", "unknown"))
    if name in _CALL_ONLY_APIS or kind == "unknown":
        return {
            "variation_class": "call_only",
            "input_diversity": {"applicable": False, "required": False, "reason": "no safe meaningful input variation is established"},
            "response_state_diversity": {"applicable": False, "required": False, "reason": "no evidence-backed state transition is required"},
            "state_change_scene": "not_applicable",
        }
    if name == "cnc_skip":
        return {
            "variation_class": "input_only",
            "input_diversity": {"applicable": True, "required": True, "dimension": "axis", "reason": "documented ALL_AXES and controller-reported 1..m axis selectors"},
            "response_state_diversity": {"applicable": True, "required": False, "reason": "a direct read returns stored data; a new G31/SKIP event is a separate optional scene"},
            "state_change_scene": "optional_event_scene",
        }
    if kind == "runtime_motion_state":
        input_applicable = name in _DOCUMENTED_INPUT_VARIATION_APIS
        return {
            "variation_class": "input_and_response" if input_applicable else "response_state_only",
            "input_diversity": {"applicable": input_applicable, "required": input_applicable, "reason": "vary documented selectors only when the active SDK/controller exposes more than one valid input"},
            "response_state_diversity": {"applicable": True, "required": True, "reason": "returned measurement depends on CNC runtime state when response variation is requested"},
            "state_change_scene": "nc_execution_when_gated",
        }
    if kind in {"business_data", "program_transfer", "controlled_business_data", "input_or_business_data"}:
        return {
            "variation_class": "input_and_response" if kind != "program_transfer" else "response_state_only",
            "input_diversity": {"applicable": True, "required": kind == "input_or_business_data", "reason": "use documented addresses, ranges, directory entries or transfer inputs when available"},
            "response_state_diversity": {"applicable": True, "required": False, "reason": "business-data change requires an explicitly authorized and reversible scene"},
            "state_change_scene": "optional_authorized_business_scene",
        }
    if kind == "runtime_state":
        return {
            "variation_class": "response_observation_only",
            "input_diversity": {"applicable": False, "required": False, "reason": "status API inputs are handles/output buffers"},
            "response_state_diversity": {"applicable": True, "required": False, "reason": "record observed states; do not force a mode or run-state transition"},
            "state_change_scene": "observation_only",
        }
    return {
        "variation_class": "call_only",
        "input_diversity": {"applicable": False, "required": False, "reason": "no reliable variation domain"},
        "response_state_diversity": {"applicable": False, "required": False, "reason": "no reliable response-state dependency"},
        "state_change_scene": "not_applicable",
    }

_FALLBACK_CAPABILITIES: list[dict[str, Any]] = [
    {
        "capability_id": "documented_axis_input_variation",
        "output_types": ["per_axis_return", "aggregate_axis_return"],
        "preconditions": ["official SDK signature", "documented ALL_AXES selector", "controller-reported controlled-axis count"],
        "tools_or_apis": ["cnc_rdaxisname", "target axis API"],
        "execution_limits": ["never guess axis count", "use only controller-reported axes", "retain aggregate read if discovery fails"],
        "observable_results": ["successful aggregate and per-axis input combinations", "returned fields per axis"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "input_parameter_variation",
        "output_types": ["parameterized_return", "range_or_enum_return"],
        "preconditions": ["official SDK signature", "documented safe domain"],
        "tools_or_apis": ["target API", "documented discovery API"],
        "execution_limits": ["never guess values", "failed calls do not count as variation"],
        "observable_results": ["successful input combinations", "returned field changes"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "controlled_business_data",
        "output_types": ["macro_value", "tool_data", "program_content"],
        "preconditions": ["documented valid address", "writable isolated simulator", "successful snapshot/read-back"],
        "tools_or_apis": ["documented read API", "matching documented write API"],
        "execution_limits": ["no CNC system-parameter writes", "no guessed addresses", "explicit permission and rollback plan required"],
        "observable_results": ["before/write/after read-back", "target output change"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "program_directory_business_data",
        "output_types": ["program_directory", "program_number", "program_status"],
        "preconditions": ["read-only CNC connection", "directory API available"],
        "tools_or_apis": ["cnc_rdprogdir3", "cnc_rdprgnum"],
        "execution_limits": ["do not create/delete programs unless explicitly authorized", "select only confirmed idle numbers"],
        "observable_results": ["directory entries", "program count/number changes"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "program_transfer_and_directory_change",
        "output_types": ["program_directory", "program_content", "transfer_status"],
        "preconditions": ["valid complete start/data/end contract", "confirmed idle target program"],
        "tools_or_apis": ["cnc_rdprogdir3", "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"],
        "execution_limits": ["no automatic cleanup", "no guessed O-number", "preserve real call and PCAP logs"],
        "observable_results": ["start/data/end return codes", "directory read-back"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "nc_simulation_execution",
        "output_types": ["run_state", "position", "feed", "spindle", "program_status"],
        "preconditions": ["verified isolated simulator", "NCGuide AUTO mode", "validated NC syntax and motion bounds"],
        "tools_or_apis": ["UploadProgram", "SelectProgram", "StartProgram", "cnc_statinfo", "cnc_rdposition"],
        "execution_limits": ["no real-machine motion", "no automatic or operator Cycle Start until isolation, permission, and motion bounds are verified", "single-block and bounds checks"],
        "observable_results": ["actual run/motion state", "changed returned fields", "matched PCAP transactions"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "external_event_simulation",
        "output_types": ["event_dependent_position", "skip_position", "alarm_or_signal_state"],
        "preconditions": ["verified isolated NCGuide", "validated bounded G31 program", "controller-specific SKIP event behavior proven", "explicit execution permission"],
        "tools_or_apis": ["G31", "SKIP signal", "cnc_skip", "operator Cycle Start after all gates pass"],
        "execution_limits": ["do not write PMC/system parameters", "do not guess signal injection", "plan-only until isolation, permission, and motion bounds are verified", "when gates pass, explicitly prompt operator to press Cycle Start"],
        "observable_results": ["G31 actually started", "SKIP event actually occurred", "cnc_skip returned position changed", "matched request/response"],
        "verified_in_current_simulator": False,
    },
    {
        "capability_id": "device_state_observation",
        "output_types": ["status", "alarm", "mode", "run_state"],
        "preconditions": ["FOCAS handle connected", "read-only status API"],
        "tools_or_apis": ["cnc_statinfo", "cnc_alarm2"],
        "execution_limits": ["observation does not prove a state transition", "retain failed and auxiliary logs"],
        "observable_results": ["before/during/after state snapshots"],
        "verified_in_current_simulator": False,
    },
]


def load_scenario_capabilities(path: Path = DEFAULT_CAPABILITIES_PATH) -> list[dict[str, Any]]:
    """Load the versioned registry, retaining a safe fallback if absent."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rows = _FALLBACK_CAPABILITIES
    else:
        rows = payload.get("capabilities") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = _FALLBACK_CAPABILITIES
    capabilities = []
    for item in rows:
        if not isinstance(item, dict) or not item.get("capability_id"):
            continue
        capability = dict(item)
        capability.setdefault("verification_status", "unverified_in_current_simulator")
        capability.setdefault("invalidation_conditions", list(_DEFAULT_INVALIDATION))
        capability.setdefault("source_scope", "local SDK, FOCAS documentation and runtime evidence required")
        capabilities.append(capability)
    return capabilities


def analyze_output_dependencies(target: str) -> dict[str, Any]:
    """Return conservative dependency facts for Planner scene selection."""
    name = str(target).strip().lower()
    if name == "cnc_skip":
        result = {
            "target": target,
            "output_fields": ["ODBAXIS skip position"],
            "dependency_kind": "input_parameter_variation",
            "dependencies": ["documented aggregate selector or controller-reported axis number"],
            "event_capability": "documented_axis_input_variation",
            "event_verified": None,
            "variation_requires_nc_program": False,
            "variation_requires_cycle_start": False,
            "fallback": "retain aggregate read and report per-axis limitation if axis discovery fails",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if "rdprogdir" in name or "rdprgnum" in name or "search" in name:
        result = {
            "target": target,
            "output_fields": ["program number", "directory entry", "program status"],
            "dependency_kind": "business_data",
            "dependencies": ["resident program directory", "idle target selection"],
            "event_capability": "program_directory_business_data",
            "event_verified": False,
            "fallback": "read existing directory only; do not mutate programs",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if name in {"cnc_dwnstart3", "cnc_download3", "cnc_dwnend3", "cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"}:
        result = {
            "target": target,
            "output_fields": ["transfer return code", "program directory read-back"],
            "dependency_kind": "program_transfer",
            "dependencies": ["cnc_rdprogdir3", "idle legal program number", "matching start/data/end session"],
            "event_capability": "program_transfer_and_directory_change",
            "event_verified": False,
            "fallback": "plan session and directory verification; do not execute without confirmed target and permissions",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if name.startswith("cnc_rdmacro"):
        result = {
            "target": target,
            "output_fields": ["macro value"],
            "dependency_kind": "controlled_business_data",
            "dependencies": ["documented macro number", "read snapshot", "authorized write", "read-back"],
            "event_capability": "controlled_business_data",
            "event_verified": False,
            "fallback": "read existing macro values only; keep static samples",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if name.startswith(("cnc_rdparam", "cnc_rdset", "cnc_rddiag")):
        result = {
            "target": target,
            "output_fields": ["parameter or macro value"],
            "dependency_kind": "input_or_business_data",
            "dependencies": ["documented parameter/macro address", "valid input domain"],
            "event_capability": "input_parameter_variation",
            "event_verified": False,
            "fallback": "retain successful static reads when safe variation is unknown",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if name in _DYNAMIC_STATE_TARGETS:
        result = {
            "target": target,
            "output_fields": ["device state or runtime measurement"],
            "dependency_kind": "runtime_motion_state",
            "dependencies": ["validated NC program", "current CNC mode/run state", "safe motion bounds"],
            "event_capability": "nc_simulation_execution",
            "event_verified": False,
            "fallback": "retain idle baseline; do not execute unverified motion",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    if name in {"cnc_statinfo", "cnc_statinfo2", "cnc_alarm2"}:
        result = {
            "target": target,
            "output_fields": ["device state or runtime measurement"],
            "dependency_kind": "runtime_state",
            "dependencies": ["current CNC mode/run/alarm state"],
            "event_capability": "device_state_observation",
            "event_verified": False,
            "fallback": "before/after read-only snapshots; no forced motion",
        }
        result["variation_requirements"] = classify_api_variation(target, result)
        return result
    result = {
        "target": target,
        "output_fields": [],
        "dependency_kind": "unknown",
        "dependencies": [],
        "event_capability": None,
        "event_verified": False,
        "fallback": "Planner/knowledge retrieval required",
    }
    result["variation_requirements"] = classify_api_variation(target, result)
    return result


def build_cnc_skip_scene_steps() -> list[PlanStep]:
    """Build the safe read-only portion of the cnc_skip validation case."""
    return [
        PlanStep("SKIP_STATUS_BEFORE", "before", "Observe CNC status before any event", "ReadRunStatus",
                 protocol_function="cnc_statinfo"),
        PlanStep("SKIP_READ", "during", "Read current skip position without injecting an event", "cnc_skip",
                 parameters={"axis": -1, "length": "sizeof(ODBAXIS)", "event_execution": "not_attempted"},
                 protocol_function="cnc_skip"),
        PlanStep("SKIP_STATUS_AFTER", "after", "Observe CNC status after the read-only sample", "ReadRunStatus",
                 protocol_function="cnc_statinfo"),
    ]


def build_scene_capability_plan(target: str, capabilities: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    dependencies = analyze_output_dependencies(target)
    variation = dependencies.get("variation_requirements") or classify_api_variation(target, dependencies)
    rows = list(capabilities or load_scenario_capabilities())
    selected = next((row for row in rows if row.get("capability_id") == dependencies.get("event_capability")), None)
    skip_scene = str(target).lower() == "cnc_skip"
    read_only_observation = dependencies["dependency_kind"] in {
        "runtime_state", "business_data", "input_or_business_data", "input_parameter_variation"
    }
    dynamic_motion = dependencies["dependency_kind"] == "runtime_motion_state"
    transfer_scene = dependencies["dependency_kind"] == "program_transfer"
    # Keep the target separate from operations that establish its inputs or
    # lifecycle.  These helpers are evidence-bearing calls, not target
    # coverage, and are listed explicitly for code generation and auditing.
    if transfer_scene:
        version = "4" if str(target).endswith("4") else "3"
        auxiliary_apis = ["cnc_rdprogdir3", f"cnc_dwnstart{version}", f"cnc_dwnend{version}"]
        scene_start_apis = ["cnc_rdprogdir3"]
    elif skip_scene:
        auxiliary_apis = ["cnc_rdaxisname"]
        scene_start_apis = [target]
    elif dynamic_motion:
        auxiliary_apis = ["cnc_statinfo"]
        scene_start_apis = ["cnc_statinfo"]
    else:
        auxiliary_apis = []
        scene_start_apis = [target] if read_only_observation else []
    return {
        "target_api": target,
        "output_dependency_analysis": dependencies,
        "variation_requirements": variation,
        "selected_capability": selected or {},
        "scene_type": "axis_input_variation" if skip_scene else dependencies["dependency_kind"],
        "selection_reason": "vary the documented axis input using the controller-reported axis count; no NC program or SKIP event is required for the read call" if skip_scene else
            "prefer documented input variation and read-only observation before state-changing scenes",
        "auxiliary_apis": auxiliary_apis,
        "before": (["read controlled-axis names and count with cnc_rdaxisname"]
                   if skip_scene else ["verify applicable prerequisites"]),
        "during": (["call cnc_skip with documented aggregate and controller-reported single-axis selectors"]
                   if skip_scene else ["call target at planned observation points"]),
        "after": ["retain real API/PCAP data and report undiscovered axes"] if skip_scene else ["read back and verify actual state"],
        "nc_program_required": False if skip_scene else (False if read_only_observation else (True if dynamic_motion else None)),
        "nc_program_spec": "not required for documented axis-input variation" if skip_scene else "only when a verified scene requires NC execution",
        "target_sampling_phase": ["during"],
        "scene_start_apis": scene_start_apis,
        "operation_apis": scene_start_apis,
        "requires_response_change_for_scene_success": bool(
            variation.get("response_state_diversity", {}).get("required", False)
        ),
        "expected_changed_fields": dependencies["output_fields"],
        "state_verification": ["compare actual returned fields", "confirm event or operation in real logs"],
        "degradation": dependencies["fallback"],
        "execution_policy": (
            "read_only_axis_input_variation"
            if skip_scene else
            "plan_only_until_nc_simulation_capability_is_verified"
            if dynamic_motion else "execute_only_when_preconditions_are_verified"
        ),
        "read_only_baseline_allowed": True,
        "dynamic_event_execution_allowed": False if skip_scene else (False if dynamic_motion else None),
        "actual_execution_required": False if skip_scene else (True if dynamic_motion else None),
        "state_change_validation": {
            "required": bool(variation.get("response_state_diversity", {}).get("required", False)),
            "evidence": ["before/after API logs", "actual event or operation marker", "matched PCAP transaction"] if not skip_scene else ["actual successful input combinations", "matched PCAP transaction"],
            "never_infer_from_plan": True,
        },
        # Planner may refine these from exact API evidence. Runtime policy is
        # fixed: observe mode first, never switch it automatically, and only
        # attempt a parameter write when exact evidence and permission exist.
        "environment_preflight": {
            "mode_requirement": "unknown",
            "mode_observation_api": "cnc_statinfo",
            "mode_mismatch_policy": "pause_for_operator; do_not_switch_mode",
            "parameter_changes": [],
            "parameter_write_policy": "exact_documented_value_and_scope_plus_explicit_permission_plus_readback",
            "on_unverified_requirement": "retain_read_only_samples_and_pause_state_changing_scene",
        },
        "nc_execution_decision": {
            "program_required": False if skip_scene else (False if read_only_observation else (True if dynamic_motion else None)),
            "execution_required": False if skip_scene else (False if read_only_observation else (True if dynamic_motion else None)),
            "cycle_start_required": False if skip_scene else bool(dynamic_motion),
            "cycle_start_allowed": False,
            "requirements_source": "no NC execution for axis-input variation" if skip_scene else "runtime output dependency; execution gates required",
        },
        "operator_interaction": {
            "required_when": ([] if skip_scene else (["execution setup and bounded NC program are verified; ask the operator to press Cycle Start"] if dynamic_motion else [])) + ["required CNC mode differs from observed mode", "parameter write lacks exact evidence or permission", "restart or simulator capability is unverified"],
            "cycle_start_action": "operator_manual_only_after_all_execution_gates_pass" if dynamic_motion else None,
            "resume_only_after": ["operator confirms environment change", "fresh read-only preflight verifies required state"],
            "target_calls_blocked_while_waiting": True,
        },
    }


def assess_scene_execution(scene_plan: dict[str, Any], api_logs: Iterable[ApiCallLog]) -> dict[str, Any]:
    """Separate API, scene, output and wire outcomes using observed records."""
    target = str(scene_plan.get("target_api", "")).strip()
    calls = list(api_logs)
    target_calls = [log for log in calls if log.protocol_function == target]
    successful = [log for log in target_calls if log.status_code == 0 and not log.error]
    matched = [log for log in target_calls if log.traffic_status == "matched"]
    from .execution_contracts import API_TRAFFIC_SESSION_RULES
    from .traffic_transactions import valid_session_segment_evidence
    eligible = [
        log for log in target_calls
        if (valid_session_segment_evidence(log) if target in API_TRAFFIC_SESSION_RULES
            else log.traffic_status == "matched")
    ]
    values = [str(log.response.get("data", "")) for log in successful]
    changed = len(set(values)) > 1
    operation_apis = {str(name) for name in scene_plan.get("operation_apis", [])}
    operations = [log for log in calls if log.protocol_function in operation_apis and log.status_code == 0]
    # A plan line or expected output never proves that a scene started.
    scene_started = bool(operations)
    requires_change = bool(scene_plan.get("requires_response_change_for_scene_success", True))
    scene_succeeded = scene_started and (changed if requires_change else True)
    skip_event_verified = None
    return {
        "target_api": target,
        "target_call_count": len(target_calls),
        "target_called": bool(target_calls),
        "target_call_succeeded": bool(successful),
        "scene_started": scene_started,
        "scene_execution_succeeded": scene_succeeded,
        "response_changed": changed,
        "matched_target_transactions": len(matched),
        "eligible_target_transactions": len(eligible),
        "complete_target_traffic_matched": bool(matched),
        "target_traffic_coverage_valid": bool(eligible),
        "skip_event_verified": skip_event_verified,
        "causal_association_verified": scene_succeeded and bool(eligible),
        "limitation": "",
    }
