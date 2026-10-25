"""Shared, evidence-based parameter coverage contract and wire-backed audit.

No Cartesian products or arbitrary writes are synthesized here. The generator
must implement documented, coupled cases; the audit counts real observations.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .execution_contracts import execution_contract, step_functions
from .execution_contracts import API_TRAFFIC_SESSION_RULES
from .focas_contracts import _api_docs, documented_parameter_names, named_prototype
from .function_coverage import is_non_wire_observable
from .scenario_capabilities import analyze_output_dependencies
from .traffic_transactions import valid_session_segment_evidence

KEY = "parameter_coverage_contract"
# These values must be logged, but changing them is not semantic API diversity.
TRANSPORT = {"flibhndl", "handle", "library_handle", "ipaddr", "port", "timeout"}
SHAPE = {"length", "len", "size", "capacity", "data_len", "bufsize", "buffer_size"}
ALIASES = {"cnc_search": {"number": ["prg_no", "program_number"]}}
# The bundled FS0i-D Fwlib32.h defines MAX_AXIS=32 and declares ODBDY/ODBDY2,
# not the planner's ODBDYNAMIC/ODBDYNAMIC2 placeholders. These are the sizes
# observed from the compiled SDK types; only numeric runtime lengths can match.
DYNAMIC_LENGTH_BYTES = {
    "cnc_rddynamic": {"sizeof(ODBDYNAMIC)": "536", "sizeof(ODBDY)": "536"},
    "cnc_rddynamic2": {"sizeof(ODBDYNAMIC2)": "540", "sizeof(ODBDY2)": "540"},
}


def planned_value_key(function: str, name: str, value: Any) -> str:
    key = value_key(value)
    if name == "length":
        return value_key(DYNAMIC_LENGTH_BYTES.get(function, {}).get(key, key))
    return key


def value_key(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    text = str(value).strip()
    try:
        number = Decimal(text)
        if number.is_finite():
            return str(number.normalize())
    except InvalidOperation:
        pass
    return text


def input_schema(function: str) -> dict[str, Any]:
    arguments = str(_api_docs().get(function, {}).get("arguments", ""))
    directions = {name: re.sub(r"\s+", "", direction).lower() for name, direction in re.findall(
        r"(?mi)^-\s+([A-Za-z_]\w*)\s+\(\s*(in\s*/\s*out|in|out)\s*\):", arguments)}
    formal = documented_parameter_names(function)
    inputs = [name for name in formal if directions.get(name) in {"in", "in/out"}
              and name.lower() not in TRANSPORT]
    meaningful = [name for name in inputs if name.lower() not in SHAPE]
    prototype = named_prototype(function)
    pointers = [name for name in inputs if re.search(r"\*\s*" + re.escape(name) + r"\b", prototype)]
    axis_domain_documented = (
        "axis" in {name.lower() for name in inputs}
        and "all_axes" in arguments.lower()
        and re.search(r"\b1\s*,\s*\.\.\s*,\s*m\b", arguments.lower()) is not None
    )
    return {"documented": bool(prototype) and all(name in directions for name in formal),
            "inputs": inputs, "dimensions": meaningful, "pointer_inputs": pointers,
            "axis_domain_documented": axis_domain_documented}


def strategy_values(row: dict) -> list[Any]:
    kind = row.get("kind")
    if kind == "fixed":
        return [row["value"]] if row.get("value") is not None else []
    if kind == "enum":
        return row.get("values", []) if isinstance(row.get("values", []), list) else []
    if kind == "range":
        values = [row[k] for k in ("min", "max") if row.get(k) is not None]
        values += row.get("samples", []) if isinstance(row.get("samples", []), list) else []
        # Never invent an intermediate value: it can be invalid for discrete domains.
        return values
    return []


def build_parameter_contract(plan, targets=()) -> dict:
    functions = execution_contract(plan, targets).targets
    contracts = {}
    scene_variations = {
        str(scene.get("target_api", "")): scene.get("variation_requirements", {})
        for scene in (plan.rag_context.get("scene_plans", []) if plan else [])
        if isinstance(scene, dict) and scene.get("target_api")
    }
    for function in functions:
        schema = input_schema(function)
        variation = scene_variations.get(function)
        if not variation:
            variation = analyze_output_dependencies(function).get("variation_requirements", {})
        strategies, cases, constraints, goals, states = [], [], [], [], []
        for step in plan.steps:
            names = step_functions(step)
            if function not in names:
                continue
            for key, destination in (("parameter_coverage_goal", goals), ("state_coverage", states)):
                item = step.parameters.get(key)
                if isinstance(item, dict) and (len(names) == 1 or item.get("function") == function):
                    destination.append(item)
            for row in step.parameters.get("parameter_generation", []) or []:
                if isinstance(row, dict) and (row.get("function") == function or
                                             len(names) == 1 and not row.get("function")):
                    strategies.append(row)
            for row in step.parameters.get("parameter_constraints", []) or []:
                if isinstance(row, dict) and (row.get("function") == function or
                                             len(names) == 1 and not row.get("function")):
                    constraints.append(row)
            for row in step.parameters.get("parameter_cases", []) or []:
                if isinstance(row, dict) and (row.get("function") == function or
                                             len(names) == 1 and not row.get("function")):
                    params = row.get("parameters", row)
                    if isinstance(params, dict):
                        case = {k: v for k, v in params.items() if k in schema["inputs"]}
                        if case:
                            cases.append(case)
        expected, minimum_values, ranges = {}, {}, {}
        for row in strategies:
            name = str(row.get("name", ""))
            name = next((formal for formal, aliases in ALIASES.get(function, {}).items()
                         if name in aliases), name)
            if name in schema["inputs"]:
                expected.setdefault(name, set()).update(planned_value_key(function, name, v) for v in strategy_values(row))
                if row.get("kind") == "range":
                    low, high = row.get("min"), row.get("max")
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and low < high:
                        has_interior = high - low > 1 if isinstance(low, int) and isinstance(high, int) else True
                        minimum_values[name] = max(minimum_values.get(name, 1), 3 if has_interior else 2)
                        if has_interior:
                            ranges.setdefault(name, []).append((low, high))
        # Invocation validity and diversity are separate. Planner goals may
        # raise the bar, but cannot lower a multi-value domain documented by
        # the API contract itself.
        requested_minimum = [
            goal["minimum_distinct"] for goal in goals
            if isinstance(goal.get("minimum_distinct"), int)
            and not isinstance(goal.get("minimum_distinct"), bool)
            and goal["minimum_distinct"] > 0
        ]
        enforced_minimum = max(requested_minimum, default=1)
        if schema["axis_domain_documented"]:
            enforced_minimum = max(enforced_minimum, 2)
        planned_domain_size = max(
            [enforced_minimum, len(cases), *(
                len(strategy_values(row)) for row in strategies if row.get("kind") == "enum"
            )],
        )
        permissions = plan.rag_context.get("task_permissions", {}) if plan else {}
        quality_budget = (plan.rag_context.get("quality_targets", {}).get("max_configurations")
                          if plan else None)
        requested_budget = (permissions.get("diversity_configuration_budget")
                            or quality_budget or permissions.get("capture_samples"))
        try:
            configuration_budget = max(1, int(requested_budget)) if requested_budget is not None else planned_domain_size
        except (TypeError, ValueError):
            configuration_budget = planned_domain_size
        try:
            sample_budget = int(permissions.get("capture_samples", 0))
        except (TypeError, ValueError):
            sample_budget = 0
        if sample_budget > 0:
            configuration_budget = min(configuration_budget, sample_budget)
        contracts[function] = {**schema,
            "mode": "parameter_variation" if schema["dimensions"] else "state_observation",
            "minimum_distinct": enforced_minimum,
            "planned_configuration_target": planned_domain_size,
            "configuration_budget": configuration_budget,
            "variation_requirements": variation,
            "input_diversity_applicable": bool(variation.get("input_diversity", {}).get("applicable", bool(schema["dimensions"]))),
            "input_diversity_required": bool(variation.get("input_diversity", {}).get("required", False)),
            "response_state_diversity_applicable": bool(variation.get("response_state_diversity", {}).get("applicable", False)),
            "response_state_diversity_required": bool(variation.get("response_state_diversity", {}).get("required", False)),
            "diversity_goal_evidence": [
                {key: goal.get(key) for key in ("minimum_distinct", "coverage_scope", "evidence", "on_insufficient") if key in goal}
                for goal in goals
            ],
            "planner_strategies": strategies, "planner_constraints": constraints,
            "planner_goals": goals, "state_coverage": states,
            "expected_values": {name: sorted(values) for name, values in expected.items()},
            "cases": cases, "minimum_values": minimum_values, "ranges": ranges,
            "policy": "Documented safe inputs only; discover device-dependent values at runtime; "
                      "preserve coupled parameters; verify and restore state changes. "
                      "Invocation validity is distinct from diversity. Require extra combinations only "
                      "when Planner supplies an evidence-backed goal; if fewer valid cases exist, "
                      "report the limitation and retain successful samples."}
    return {"version": 1, "functions": contracts}


def attach_parameter_contract(plan, targets=()) -> dict:
    contract = build_parameter_contract(plan, targets)
    plan.rag_context[KEY] = contract
    return contract


def input_values(log, function: str) -> dict[str, str]:
    raw = log.input_parameters.get("raw")
    if raw:
        values = {k.strip(): value_key(v) for k, v in
                  (field.split("=", 1) for field in str(raw).split(";") if "=" in field)}
    else:
        values = {str(k): value_key(v) for k, v in log.input_parameters.items()}
    for formal, aliases in ALIASES.get(function, {}).items():
        if formal not in values:
            for alias in aliases:
                if alias in values:
                    values[formal] = values[alias]
                    break
    return values


def audit_parameter_coverage(plan, logs) -> dict:
    contract = plan.rag_context.get(KEY, {}) if plan else {}
    reports, errors = {}, []
    for function, rule in contract.get("functions", {}).items():
        calls = [log for log in logs if log.protocol_function == function and
                 not any(token in log.step_id.upper() for token in
                         ("RESTORE", "VERIFY_SELECTION", "VERIFY_RESTORE", "MOTION_EVIDENCE"))]
        inputs = rule["inputs"]
        dimensions = rule["dimensions"]
        observed, successful, matched, eligible, valid_rows = set(), set(), set(), set(), []
        output_values = set()
        missing_inputs = set()
        for log in calls:
            values = input_values(log, function)
            signature = tuple((name, values.get(name, "<missing>")) for name in dimensions)
            observed.add(signature)
            missing = set(inputs) - values.keys()
            # Pointer addresses and symbolic placeholders are not serialized input values.
            invalid = any(re.fullmatch(r"0x[0-9a-fA-F]{6,}|&[A-Za-z_]\w*|<.*>", values.get(n, ""))
                          for n in rule.get("pointer_inputs", []))
            if log.status_code == 0 and not log.error:
                missing_inputs.update(missing)
                if not missing and not invalid:
                    successful.add(signature)
                    if function in API_TRAFFIC_SESSION_RULES:
                        wire_eligible = valid_session_segment_evidence(log)
                    else:
                        wire_eligible = log.traffic_status == "matched"
                    if wire_eligible or is_non_wire_observable(function):
                        eligible.add(signature)
                        if log.traffic_status == "matched" or is_non_wire_observable(function):
                            matched.add(signature)
                        valid_rows.append(values)
                        data = log.response.get("data", "")
                        if data not in ("", None):
                            output_values.add(value_key(data))
        missing_values = {name: sorted({planned_value_key(function, name, value) for value in required}
                                       - {row[name] for row in valid_rows if name in row})
                          for name, required in rule["expected_values"].items()}
        missing_values = {name: values for name, values in missing_values.items() if values}
        sparse_ranges = {name: count for name, count in rule.get("minimum_values", {}).items()
                         if len({row[name] for row in valid_rows if name in row}) < count}
        for name, bounds in rule.get("ranges", {}).items():
            numeric_values = []
            for row in valid_rows:
                try:
                    numeric_values.append(Decimal(row[name]))
                except (KeyError, InvalidOperation):
                    pass
            if any(not any(Decimal(str(low)) < value < Decimal(str(high)) for value in numeric_values)
                   for low, high in bounds):
                sparse_ranges[name] = 3
        missing_cases = [case for case in rule["cases"] if not any(
            all(row.get(name) == planned_value_key(function, name, value) for name, value in case.items()) for row in valid_rows)]
        issues = []
        if not rule["documented"]:
            issues.append("input_direction_documentation_missing")
        if not calls:
            issues.append("target_not_called")
        if len(eligible) < rule["minimum_distinct"] and not is_non_wire_observable(function):
            issues.append(f"eligible_distinct={len(eligible)};required={rule['minimum_distinct']}")
        if missing_inputs:
            issues.append("input_values_not_logged=" + ",".join(sorted(missing_inputs)))
        if missing_values:
            issues.append("planned_values_missing")
        if missing_cases:
            issues.append("planned_cases_missing")
        if sparse_ranges:
            issues.append("range_typical_or_intermediate_values_missing")
        # Local library/handle APIs are judged by an actual successful call;
        # PCAP-backed diversity requirements do not apply to them.
        if is_non_wire_observable(function) and any(
            log.status_code == 0 and not log.error for log in calls
        ):
            issues = []
        report = {"mode": rule["mode"], "call_count": len(calls),
            "distinct_input_combinations": len(observed),
            "successful_distinct_input_combinations": len(successful),
            "matched_distinct_input_combinations": len(matched),
            "eligible_distinct_input_combinations": len(eligible),
            "matched_output_variants": len(output_values),
            "response_state_variation_observed": len(output_values) > 1,
            "response_state_variation_status": (
                "not_applicable"
                if not rule.get("response_state_diversity_applicable", False)
                else "satisfied"
                if len(output_values) > 1
                else "optional_not_observed"
                if not rule.get("response_state_diversity_required", False)
                else "not_observed"
            ),
            "minimum_distinct": rule["minimum_distinct"],
            "parameter_names": dimensions, "missing_values": missing_values,
            "missing_cases": missing_cases, "issues": issues, "passed": not issues,
            "input_diversity_applicable": rule.get("input_diversity_applicable", bool(dimensions)),
            "input_diversity_required": rule.get("input_diversity_required", bool(dimensions)),
            "response_state_diversity_applicable": rule.get("response_state_diversity_applicable", False),
            "response_state_diversity_required": rule.get("response_state_diversity_required", False),
            "variation_class": rule.get("variation_requirements", {}).get("variation_class", rule["mode"])}
        reports[function] = report
        if issues:
            errors.append("PARAMETER_COVERAGE_INCOMPLETE: " + function + ": " + "; ".join(issues))
    return {"version": 1, "functions": reports, "passed": not errors, "errors": errors}
