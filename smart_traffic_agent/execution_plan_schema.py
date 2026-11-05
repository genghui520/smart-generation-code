"""Compatibility schema and validation for the Planner -> Generator hand-off.

The project historically carried scenario and step information in several
fields.  This module derives one explicit view without replacing those fields
so saved plans and existing agents remain readable.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import re
from typing import Any, Iterable

from .models import ExecutionPlan, PlanStep

_VALID_RELATIONS = {"filters", "prepares"}
_VALID_ROLES = {"target", "auxiliary"}
_STATUS_VALUES = {"satisfied", "to_check", "to_prepare", "unknown"}
_VALID_TRANSFORMS = {
    "construct_nc_program_with_leading_O_number", "render_nc_payload_with_selected_o_number",
    "select_registered_source_and_unused_destination", "select_one_registered_program_for_upload",
    "select_registered_program_number", "select_registered_non_current_program",
    "parse_axes_ascii_and_bound_to_max_axis", "active_axis_count_to_len",
    "active_axis_count_to_axis_domain_and_odbaxis_length", "axis_1_to_ctrl_axis_or_all_axes",
    "map_name_and_suffix_to_one_based_axis_index", "bound_tool_offset_number_to_use_no",
    "expand_0iD_macro_range_indicator", "walk_parameter_range_then_use_sparse_info",
    "select_parameter_within_reported_range", "use_sparse_parameter_info_and_axis_attribute",
    "bound_work_offset_number_to_use_no", "supported_pmc_address_and_data_type",
    "select_path_in_0_to_maxpath_range", "bind_data_window_to_selected_path",
    "exclude_current_and_main_program_from_destructive_selection", "use_sequence_as_execution_block_cursor",
    "read_back_written_parameter_using_snapshot_metadata", "map_address_kind_and_bound_address_range",
    "read_back_written_macro_value", "read_back_written_tool_offset", "read_back_written_work_offset",
    "identity", "available_sram_name",
}
_PREREQUISITE_CATEGORIES = (
    "cnc_parameters", "pmc_signals", "cnc_mode", "cnc_state", "session", "options", "nc_program",
)


def _api_name(step: PlanStep) -> str:
    protocol = str(step.protocol_function or "").strip()
    if protocol and not any(separator in protocol for separator in ("/", "->", ";", "\\")):
        return protocol
    return str(step.api or step.interface_name or protocol or "").strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_cancelled_error_probe(step: PlanStep) -> bool:
    """Identify deliberate nonzero-return probes disabled by the operator."""
    parameters = step.parameters if isinstance(step.parameters, dict) else {}
    if "expected_return_code" in parameters:
        return True
    # Older Planner responses placed the probe marker on an api_calls item
    # instead of the step parameters. Treat that shape identically so it
    # cannot reach CodeGenerationAgent as a normal atomic call.
    for call in getattr(step, "api_calls", []) or ():
        if isinstance(call, dict):
            call_parameters = call.get("parameters", call.get("arguments", {}))
            if isinstance(call_parameters, dict) and "expected_return_code" in call_parameters:
                return True
    coverage = parameters.get("parameter_coverage_goal")
    coverage_text = " ".join(str(value) for value in coverage.values()) if isinstance(coverage, dict) else ""
    text = " ".join((str(step.action or ""), str(step.expected_state or ""), coverage_text)).casefold()
    return bool(
        re.search(r"error[ -]?path\s+probe", text)
        or "error branch" in text
        or "预期错误" in text
        or "错误探针" in text
    )


def _edge_attributes(edge: dict[str, Any]) -> dict[str, Any]:
    relation = _as_dict(edge.get("relation"))
    attributes = relation.get("attributes")
    if not isinstance(attributes, dict):
        attributes = _as_dict(edge.get("attributes"))
    return attributes


def _effective_edges(trcm_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return only evidence-backed edges selected by TRCM's dependency plan."""
    if not isinstance(trcm_context, dict):
        return []
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for facts in (trcm_context.get("apis", {}) or {}).values():
        if not isinstance(facts, dict):
            continue
        dependency_plan = facts.get("dependency_plan", {})
        if not isinstance(dependency_plan, dict):
            continue
        for edge in dependency_plan.get("edges", ()):
            if not isinstance(edge, dict):
                continue
            relation = _as_dict(edge.get("relation"))
            status = str(relation.get("confidence_status", "")).strip()
            relation_type = str(relation.get("relation_type", "")).strip().lower()
            source = str(edge.get("source_api", "")).strip()
            target = str(edge.get("target_api", "")).strip()
            attrs = _edge_attributes(edge)
            path = str(attrs.get("dependency_path", "")).strip()
            key = (source, target, relation_type, path)
            if not source or not target or relation_type not in _VALID_RELATIONS or not path:
                continue
            if status not in {"documented", "sdk_validated", "runtime_verified"}:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "source_api": source,
                "target_api": target,
                "relation": relation_type,
                "dependency_path": path,
                "source_field": str(attrs.get("source_field", "")).strip() or None,
                "filter_mode": str(attrs.get("filter_mode", "")).strip() or None,
                "transform": str(attrs.get("transform", "")).strip() or None,
                "include": list(attrs.get("include", ()) or ()) if isinstance(attrs.get("include", ()), (list, tuple)) else [],
                "exclude": list(attrs.get("exclude", ()) or ()) if isinstance(attrs.get("exclude", ()), (list, tuple)) else [],
                "confidence_status": status,
                "rule_id": relation.get("rule_id", ""),
                "evidence_ids": list(relation.get("evidence_ids", ()) or ()),
            })
    return result


def _contract_map(trcm_context: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(trcm_context, dict):
        return result
    for api_name, facts in (trcm_context.get("apis", {}) or {}).items():
        if not isinstance(facts, dict):
            continue
        contract = facts.get("node_contract") or _as_dict(facts.get("api_info")).get("api")
        if isinstance(contract, dict):
            result[str(api_name)] = contract
    return result


def _path_exists(value: Any, path: str) -> bool:
    current = value
    parts = [part for part in str(path).split(".") if part]
    if not parts:
        return False
    for part in parts:
        if not isinstance(current, dict):
            return False
        if part in current:
            current = current[part]
        elif isinstance(current.get("fields"), dict) and part in current["fields"]:
            current = current["fields"][part]
        else:
            return False
    return True


def _path_value(value: Any, path: str) -> Any:
    current = value
    for part in [part for part in str(path).split(".") if part]:
        if not isinstance(current, dict):
            return None
        if part in current:
            current = current[part]
        elif isinstance(current.get("fields"), dict) and part in current["fields"]:
            current = current["fields"][part]
        else:
            return None
    return current


def _target_candidates(plan: ExecutionPlan) -> list[str]:
    explicit = str(plan.target_api or "").strip()
    if explicit:
        return [explicit]
    permissions = _as_dict(plan.rag_context.get("task_permissions"))
    values: list[str] = []
    for key in ("coverage_current_batch_functions", "required_api_functions", "coverage_target_functions", "explicit_target_functions"):
        raw = permissions.get(key, plan.rag_context.get(key, ()))
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, (list, tuple, set)):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    if values:
        return list(dict.fromkeys(values))
    # For old single-target plans, the last concrete FOCAS operation is the
    # least surprising target fallback; connection helpers are excluded.
    helpers = {"cnc_allclibhndl3", "cnc_freelibhndl"}
    names = [_api_name(step) for step in plan.steps]
    return list(dict.fromkeys(name for name in names if name and name not in helpers))[-1:]


def _session_for(target_api: str, steps: list[PlanStep], contracts: dict[str, Any]) -> dict[str, Any]:
    rules: dict[str, dict[str, Any]] = {}
    try:
        from .execution_contracts import API_TRAFFIC_SESSION_RULES
        rules = API_TRAFFIC_SESSION_RULES
    except Exception:
        rules = {}
    rule = rules.get(target_api)
    if not isinstance(rule, dict):
        return {"required": False, "order": [], "status": "unknown"}
    by_api = {_api_name(step): step for step in steps}
    order: list[dict[str, Any]] = []
    for role, key in (("start", "start_api"), ("target", "segment_api"), ("end", "end_api")):
        api = str(rule.get(key, "")).strip()
        step = by_api.get(api)
        order.append({
            "order": len(order) + 1,
            "owner": role,
            "api": api,
            "step_id": step.step_id if step else None,
            "required": bool(step),
        })
    versions = {item["api"][-1:] for item in order if item["api"] and item["api"][-1:].isdigit()}
    return {
        "required": True,
        "session_type": rule.get("session_type", ""),
        "wire_direction": rule.get("wire_direction", ""),
        "data_repeatable": bool(rule.get("data_repeatable", False)),
        "version": sorted(versions)[0] if len(versions) == 1 else None,
        "order": order,
        "status": "to_check",
    }


def _prerequisites_for(
    target_api: str,
    contracts: dict[str, dict[str, Any]],
    session: dict[str, Any],
    satisfied_apis: Iterable[str] = (),
) -> dict[str, Any]:
    contract = _as_dict(contracts.get(target_api))
    raw = _as_dict(contract.get("prerequisites"))
    result: dict[str, Any] = {}
    status_by_category: dict[str, str] = {}
    satisfied = {str(item).strip() for item in satisfied_apis if str(item).strip()}
    for category in _PREREQUISITE_CATEGORIES:
        values = raw.get(category, [])
        if not isinstance(values, list):
            values = []
        result[category] = deepcopy(values)
        if not values:
            status_by_category[category] = "unknown"
        elif any(str(item.get("read_api", "")).strip() in satisfied for item in values if isinstance(item, dict)):
            status_by_category[category] = "satisfied"
        elif any(isinstance(item, dict) and (item.get("prepare_api") or item.get("write_api")) for item in values):
            status_by_category[category] = "to_prepare"
        else:
            status_by_category[category] = "to_check"
    if session.get("required"):
        result["session"] = deepcopy(result.get("session", []))
        status_by_category["session"] = "to_check"
    result["status_by_category"] = status_by_category
    result["status"] = "to_check" if any(value == "to_check" for value in status_by_category.values()) else "unknown"
    return result


def normalize_execution_plan(plan: ExecutionPlan, trcm_context: dict[str, Any] | None = None) -> ExecutionPlan:
    """Populate the additive structured fields from existing plan knowledge."""
    if isinstance(trcm_context, dict) and trcm_context:
        plan.rag_context.setdefault("traffic_relationships", trcm_context)
    cancelled = [step for step in plan.steps if _is_cancelled_error_probe(step)]
    if cancelled:
        plan.steps = [step for step in plan.steps if step not in cancelled]
        plan.rag_context["cancelled_error_probes"] = [
            {
                "step_id": step.step_id,
                "protocol_function": step.protocol_function or step.api or step.interface_name,
                "reason": "error probes disabled by operator",
            }
            for step in cancelled
        ]
    targets = _target_candidates(plan)
    plan.target_api = str(plan.target_api or (targets[0] if targets else "")).strip()
    plan.scenario = str(plan.scenario or plan.scenario_type or "").strip()
    contracts = _contract_map(trcm_context or plan.rag_context.get("traffic_relationships"))
    edges = _effective_edges(trcm_context or plan.rag_context.get("traffic_relationships"))
    if not edges:
        # A plan may have already been normalized and persisted.
        edges = [item for item in plan.dependencies if isinstance(item, dict)]
    target_set = set(targets or ([plan.target_api] if plan.target_api else []))
    api_to_step: dict[str, PlanStep] = {}
    for step in plan.steps:
        step.api = _api_name(step)
        if not step.api:
            step.api = str(step.interface_name or "").strip()
        step.role = "target" if step.api in target_set or step.interface_name in target_set else "auxiliary"
        api_to_step.setdefault(step.api, step)
    selected_edges: list[dict[str, Any]] = []
    for edge in edges:
        source, target = str(edge.get("source_api", "")), str(edge.get("target_api", ""))
        if source in api_to_step and target in api_to_step:
            selected_edges.append(edge)
    plan.dependencies = selected_edges
    plan.runtime_bindings = []
    for edge in selected_edges:
        if edge.get("relation") != "filters":
            continue
        source_step = api_to_step.get(str(edge.get("source_api", "")))
        target_step = api_to_step.get(str(edge.get("target_api", "")))
        binding = {
            "source_step_id": source_step.step_id if source_step else None,
            "source_api": edge.get("source_api"),
            "source_field": edge.get("source_field"),
            "source_fields": [edge.get("source_field")] if edge.get("source_field") else [],
            "target_step_id": target_step.step_id if target_step else None,
            "target_api": edge.get("target_api"),
            "target_parameter_path": edge.get("dependency_path"),
            "target_parameters": [edge.get("dependency_path")] if edge.get("dependency_path") else [],
            "filter_mode": edge.get("filter_mode"),
            "transform": edge.get("transform"),
            "include": list(edge.get("include", ()) or ()),
            "exclude": list(edge.get("exclude", ()) or ()),
            "runtime_required": True,
        }
        plan.runtime_bindings.append(binding)
        if target_step is not None:
            target_step.parameter_sources.append(binding)
    # The step-level dependency and output declarations make the execution
    # contract self-contained for Generator and Executor.
    for step in plan.steps:
        step.dependencies = [edge for edge in selected_edges if edge.get("target_api") == step.api]
        step.required_inputs = [
            {"path": edge.get("dependency_path"), "source_api": edge.get("source_api"), "status": "to_check"}
            for edge in step.dependencies
        ]
        produced = [
            {"field": edge.get("source_field"), "for": edge.get("target_api"), "save": True}
            for edge in selected_edges
            if edge.get("source_api") == step.api and edge.get("source_field")
        ]
        step.outputs = [{"field": "return_code", "save": True}, *produced]
        has_dependents = any(edge.get("source_api") == step.api for edge in selected_edges)
        step.blocks_dependents = bool(has_dependents or step.role == "auxiliary")
        step.failure_policy = "block_dependents" if step.blocks_dependents else "record_and_continue"
    plan.session = _session_for(plan.target_api, plan.steps, contracts)
    plan.prerequisites = _prerequisites_for(
        plan.target_api,
        contracts,
        plan.session,
        _as_dict(plan.rag_context).get("satisfied_api_names", ()),
    )
    plan.collection = {
        "npcap": {"required": True, "group_by": ["batch_id", "api_name"], "one_pcap_per_api": True},
        "logs": {
            "input": "focas_api_input.csv",
            "output": "focas_api_output.csv",
            "fixed_input": "input_log.csv",
            "fixed_output": "output_log.csv",
            "structured_calls": "api_calls.jsonl",
            "api_logs": "api_logs.jsonl",
        },
        "record_fields": ["invocation_id", "step_id", "api_name", "input_parameters", "return_code", "start_timestamp_ns", "end_timestamp_ns", "transaction_id"],
        "return_codes": True,
    }
    return plan


def execution_plan_view(plan: ExecutionPlan) -> dict[str, Any]:
    """Return the stable JSON-compatible hand-off consumed by Codegen/Executor."""
    return {
        "schema_version": int(plan.schema_version),
        "target_api": plan.target_api,
        "scenario": plan.scenario or plan.scenario_type,
        "dependencies": deepcopy(plan.dependencies),
        "steps": [asdict(step) for step in plan.steps],
        "runtime_bindings": deepcopy(plan.runtime_bindings),
        "prerequisites": deepcopy(plan.prerequisites),
        "session": deepcopy(plan.session),
        "collection": deepcopy(plan.collection),
        # Legacy aliases remain visible to old consumers.
        "plan_id": plan.plan_id,
        "task_id": plan.task_id,
        "scenario_type": plan.scenario_type,
        "scenario_goal": plan.scenario_goal,
    }


def validate_execution_plan(plan: ExecutionPlan) -> dict[str, Any]:
    """Validate structure, graph paths, ABI names and session ordering."""
    errors: list[str] = []
    warnings: list[str] = []
    if not str(plan.target_api).strip():
        errors.append("target_api is required")
    if not str(plan.scenario or plan.scenario_type).strip():
        errors.append("scenario is required")
    if not plan.collection:
        errors.append("collection contract is required")
    ids: set[str] = set()
    api_to_index: dict[str, int] = {}
    for index, step in enumerate(plan.steps):
        if not step.step_id or step.step_id in ids:
            errors.append(f"steps[{index}] has a duplicate or empty step_id")
        ids.add(step.step_id)
        api = _api_name(step)
        if not api:
            errors.append(f"steps[{index}] api is required")
        if step.role not in _VALID_ROLES:
            errors.append(f"steps[{index}] role must be target or auxiliary")
        api_to_index.setdefault(api, index)
    contracts = _contract_map(plan.rag_context.get("traffic_relationships"))
    for index, edge in enumerate(plan.dependencies):
        relation = str(edge.get("relation", "")).lower()
        source, target, path = str(edge.get("source_api", "")), str(edge.get("target_api", "")), str(edge.get("dependency_path", ""))
        if relation not in _VALID_RELATIONS:
            errors.append(f"dependencies[{index}] has unsupported relation {relation!r}")
        if not source or not target or not path:
            errors.append(f"dependencies[{index}] requires source_api, target_api and dependency_path")
        if source in api_to_index and target in api_to_index and api_to_index[source] >= api_to_index[target]:
            errors.append(f"dependency {source}->{target} is out of execution order")
        if relation == "filters":
            if not edge.get("source_field"):
                errors.append(f"filters dependency {source}->{target} requires source_field")
            if edge.get("filter_mode") not in {"include", "exclude"}:
                errors.append(f"filters dependency {source}->{target} requires include/exclude filter_mode")
            transform = str(edge.get("transform") or "").strip()
            if transform and transform not in _VALID_TRANSFORMS:
                errors.append(f"filters dependency {source}->{target} uses unsupported transform {transform!r}")
            target_parameters = _as_dict(contracts.get(target)).get("parameters", {})
            source_parameters = _as_dict(contracts.get(source)).get("parameters", {})
            if target in contracts and not _path_exists(target_parameters, path):
                errors.append(f"dependency_path {path!r} is absent from {target} parameters")
            if source in contracts and edge.get("source_field") and not _path_exists(source_parameters, str(edge["source_field"])):
                errors.append(f"source_field {edge['source_field']!r} is absent from {source} parameters")
            if target in contracts and _path_exists(target_parameters, path):
                target_root = _path_value(target_parameters, path.split(".")[0])
                if isinstance(target_root, dict) and target_root.get("direction") not in {"in", "inout"}:
                    errors.append(f"target parameter {path!r} is not an input/inout parameter")
                target_leaf = _path_value(target_parameters, path)
                if isinstance(target_leaf, dict) and target_leaf.get("type") is None:
                    warnings.append(f"target parameter {target}.{path} has no SDK type metadata")
            if source in contracts and edge.get("source_field") and _path_exists(source_parameters, str(edge["source_field"])):
                source_root = _path_value(source_parameters, str(edge["source_field"]).split(".")[0])
                if isinstance(source_root, dict) and source_root.get("direction") not in {"out", "inout"}:
                    errors.append(f"source field {source}.{edge['source_field']} is not an output/inout parameter")
                source_leaf = _path_value(source_parameters, str(edge["source_field"]))
                if isinstance(source_leaf, dict) and source_leaf.get("type") is None:
                    warnings.append(f"source field {source}.{edge['source_field']} has no SDK type metadata")
            source_leaf = _path_value(source_parameters, str(edge.get("source_field", ""))) if source in contracts else None
            target_leaf = _path_value(target_parameters, path) if target in contracts else None
            source_type = " ".join(str(_as_dict(source_leaf).get("type", "")).replace("const", "").replace("volatile", "").replace("*", " ").split()).lower()
            target_type = " ".join(str(_as_dict(target_leaf).get("type", "")).replace("const", "").replace("volatile", "").replace("*", " ").split()).lower()
            if source_type and target_type and source_type != target_type and not transform:
                errors.append(f"filters dependency {source}.{edge.get('source_field')} -> {target}.{path} has incompatible SDK types without transform")
            if source == "cnc_rdprogdir3" and target in {"cnc_download3", "cnc_download4"} and path == "data":
                if edge.get("filter_mode") != "include" or transform != "construct_nc_program_with_leading_O_number":
                    errors.append("directory program numbers must be rendered into cnc_download data with construct_nc_program_with_leading_O_number")
            if edge.get("filter_mode") == "exclude" and target_type in {"char", "unsigned char"}:
                errors.append(f"filters dependency {source}->{target} cannot exclude values from a byte buffer")
        elif relation == "prepares":
            target_prerequisites = _as_dict(contracts.get(target)).get("prerequisites", {})
            prerequisite_path = path.removeprefix("prerequisites.")
            if target in contracts and not _path_exists(target_prerequisites, prerequisite_path):
                errors.append(f"dependency_path {path!r} is absent from {target} prerequisites")
    session = _as_dict(plan.session)
    order = session.get("order", [])
    if session.get("required"):
        if not isinstance(order, list) or [item.get("owner") for item in order if isinstance(item, dict)] != ["start", "target", "end"]:
            errors.append("session must declare start, target and end in order")
        elif any(not isinstance(item, dict) or not item.get("step_id") for item in order):
            errors.append("session start/target/end must each own a PlanStep")
        versions = {str(item.get("api", ""))[-1:] for item in order if isinstance(item, dict) and str(item.get("api", ""))[-1:].isdigit()}
        if len(versions) > 1:
            errors.append("session mixes API versions")
    for index, binding in enumerate(plan.runtime_bindings):
        for key in ("source_step_id", "source_field", "target_step_id", "target_parameter_path", "filter_mode"):
            if not binding.get(key):
                errors.append(f"runtime_bindings[{index}] missing {key}")
        if binding.get("source_step_id") not in ids or binding.get("target_step_id") not in ids:
            errors.append(f"runtime_bindings[{index}] references an unknown step")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "schema_version": int(plan.schema_version)}
