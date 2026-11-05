"""Evidence-scoped parameter discovery, filtering, and combination generation."""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class ParameterRule:
    rule_id: str
    target_api: str
    parameter: str
    source_api: str = ""
    filter_strategy: str = "identity"
    filter_mode: str = "include"
    require_runtime_discovery: bool = False
    allowed_values: tuple[Any, ...] = ()
    excluded_values: tuple[Any, ...] = ()
    constraints: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    applicability: dict[str, Any] | None = None
    source_fields: tuple[str, ...] = ()
    target_parameters: tuple[str, ...] = ()
    transform: str = ""
    dangerous_operation: bool = False
    auto_execute: bool = True
    verification_requirements: tuple[str, ...] = ()


@dataclass(slots=True)
class ParameterToolResult:
    success: bool
    api_name: str
    parameter_name: str = ""
    candidates: list[Any] = None  # type: ignore[assignment]
    excluded: list[dict[str, Any]] = None  # type: ignore[assignment]
    combinations: list[dict[str, Any]] = None  # type: ignore[assignment]
    actual_combination_count: int = 0
    source: list[dict[str, Any]] = None  # type: ignore[assignment]
    unverified_conditions: list[str] = None  # type: ignore[assignment]
    error: str = ""

    def __post_init__(self) -> None:
        self.candidates = list(self.candidates or [])
        self.excluded = list(self.excluded or [])
        self.combinations = list(self.combinations or [])
        self.source = list(self.source or [])
        self.unverified_conditions = list(self.unverified_conditions or [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_nc_program_number(nc_program: str, selected_program_number: int) -> dict[str, Any]:
    """Verify that the NC payload's leading O-number matches the selection.

    The selected number is a runtime result of directory filtering.  This
    check deliberately does not invent a legal range or protected set; those
    facts must come from the controller directory/options evidence.
    """
    try:
        selected = int(selected_program_number)
    except (TypeError, ValueError):
        return {
            "valid": False,
            "selected_program_number": selected_program_number,
            "payload_program_numbers": [],
            "error": "selected program number is not an integer",
        }
    payload = str(nc_program or "").replace("\r\n", "\n").replace("\r", "\n")
    numbers = [int(item) for item in re.findall(r"(?im)^\s*O\s*([0-9]{1,8})(?=\s|;|$)", payload)]
    if not numbers:
        return {
            "valid": False,
            "selected_program_number": selected,
            "payload_program_numbers": [],
            "error": "NC payload has no program-number line",
        }
    valid = numbers[0] == selected
    return {
        "valid": valid,
        "selected_program_number": selected,
        "payload_program_numbers": numbers,
        "error": "" if valid else "leading NC program number does not match selected program number",
    }


def normalize_rules(raw_rules: Iterable[dict[str, Any]] | None, api_name: str) -> list[ParameterRule]:
    rules: list[ParameterRule] = []
    for index, raw in enumerate(raw_rules or (), 1):
        if not isinstance(raw, dict):
            continue
        relation = raw.get("attributes", raw)
        relation = relation if isinstance(relation, dict) else raw
        # Neo4j filters are stored directly in attributes_json. Older plans
        # may still wrap them in parameter_rules; accept both shapes.
        nested = relation.get("parameter_rules") if isinstance(relation, dict) else None
        if isinstance(nested, list) and nested:
            for item in normalize_rules(nested, api_name):
                rules.append(item)
            continue
        dependency_path = str(relation.get("dependency_path") or "").strip()
        parameter = str(
            relation.get("parameter") or relation.get("parameter_name")
            or dependency_path.rsplit(".", 1)[-1]
        ).strip()
        if not parameter:
            continue
        rules.append(ParameterRule(
            rule_id=str(relation.get("rule_id") or f"graph:{api_name}:{parameter}:{index}"),
            target_api=api_name,
            parameter=parameter,
            source_api=str(relation.get("source_api") or raw.get("source_api") or ""),
            filter_strategy=str(relation.get("filter_strategy") or "identity"),
            filter_mode=str(relation.get("filter_mode") or "include").strip().lower(),
            require_runtime_discovery=bool(relation.get("require_runtime_discovery", False)),
            allowed_values=tuple(relation.get("allowed_values", relation.get("values", ())) or ()),
            excluded_values=tuple(relation.get("excluded_values", relation.get("protected_values", ())) or ()),
            constraints=tuple(str(item) for item in relation.get("constraints", ()) or ()),
            evidence_ids=tuple(str(item) for item in relation.get("evidence_ids", ()) or ()),
            applicability=dict(relation.get("applicability", {}) or {}),
            source_fields=tuple(
                str(item) for item in (
                    relation.get("source_fields", ())
                    or ((relation.get("source_field"),) if relation.get("source_field") else ())
                ) if str(item).strip()
            ),
            target_parameters=tuple(str(item) for item in relation.get("target_parameters", ()) or ()),
            transform=str(relation.get("transform", "")),
            dangerous_operation=bool(relation.get("dangerous_operation", False)),
            auto_execute=bool(relation.get("auto_execute", True)),
            verification_requirements=tuple(str(item) for item in relation.get("verification_requirements", relation.get("constraints", ())) or ()),
        ))
    return rules


def discover_parameter_values(
    api_name: str,
    parameter_name: str,
    observations: Iterable[dict[str, Any]] | None = None,
    *,
    source_api: str = "",
) -> ParameterToolResult:
    """Extract observed values only; absence is reported as incomplete evidence."""
    values: list[Any] = []
    source: list[dict[str, Any]] = []
    parameter = str(parameter_name).strip()
    for row in observations or ():
        if not isinstance(row, dict):
            continue
        function = str(row.get("protocol_function") or row.get("api_name") or "")
        if source_api and function != source_api:
            continue
        payloads = [row.get("input_parameters", {}), row.get("response", {}), row]
        for payload in payloads:
            if not isinstance(payload, dict) or parameter not in payload:
                continue
            value = payload[parameter]
            if value not in values:
                values.append(value)
                source.append({
                    "api_name": function or source_api,
                    "parameter": parameter,
                    "value": value,
                    "evidence_id": str(row.get("invocation_id") or row.get("evidence_id") or ""),
                    "timestamp": row.get("timestamp", ""),
                })
    return ParameterToolResult(
        success=bool(values), api_name=api_name, parameter_name=parameter,
        candidates=values, source=source,
        unverified_conditions=[] if values else ["runtime discovery returned no observed value"],
    )


class ParameterBindingError(ValueError):
    """A deterministic binding failure with a machine-readable reason."""

    def __init__(self, code: str, message: str, *, rule_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.rule_id = rule_id


def select_parameter_values(
    values: Iterable[Any],
    *,
    full_enum_threshold: int = 20,
    middle_sample_count: int = 5,
) -> dict[str, Any]:
    """Choose deterministic executable values after filtering.

    The input is the already-filtered legal set.  Small sets are fully
    enumerated; large sets use rank-based endpoints and middle samples, so
    non-contiguous domains never produce values outside the set.
    """
    threshold = max(1, int(full_enum_threshold))
    middle = max(0, int(middle_sample_count))
    ordered = list(dict.fromkeys(values or ()))
    size = len(ordered)
    if size <= threshold:
        return {"selected": ordered, "strategy": "fixed" if size <= 1 else "full_enum", "domain_size": size, "omitted_count": 0}
    indexes = {0, size - 1}
    if middle:
        for position in range(1, middle + 1):
            indexes.add(round(position * (size - 1) / (middle + 1)))
    selected = [ordered[index] for index in sorted(indexes)]
    return {
        "selected": selected,
        "strategy": "representative_sample",
        "domain_size": size,
        "omitted_count": size - len(selected),
        "selection_basis": "sorted stable rank endpoints and evenly spaced middle indexes",
    }


def _read_path(payload: Any, path: str) -> tuple[bool, Any]:
    parts = [part for part in str(path or "").split(".") if part]

    def read(value: Any, index: int) -> tuple[bool, Any]:
        if index >= len(parts):
            return True, value
        part = parts[index]
        if part.endswith("[]"):
            key = part[:-2]
            if not isinstance(value, dict) or key not in value or not isinstance(value[key], (list, tuple)):
                return False, None
            collected: list[Any] = []
            for item in value[key]:
                found, nested = read(item, index + 1)
                if found:
                    if isinstance(nested, list):
                        collected.extend(nested)
                    else:
                        collected.append(nested)
            return bool(collected) or len(value[key]) == 0, collected
        if not isinstance(value, dict) or part not in value:
            return False, None
        return read(value[part], index + 1)

    return read(payload, 0)


def _transform_runtime_values(name: str, value: Any, *, rule_id: str = "") -> list[Any]:
    """Apply only transforms registered in the execution plan schema."""
    transform = str(name or "").strip()
    if not transform or transform in {"identity", "none"}:
        return list(value) if isinstance(value, (list, tuple, set)) else [value]
    if transform in {
        "active_axis_count_to_axis_domain",
        "active_axis_count_to_axis_domain_and_odbaxis_length",
    }:
        raw = value
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else ""
        try:
            count = int(str(raw).strip(), 10)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError("transform_failed", f"{transform} requires a numeric axis count", rule_id=rule_id) from exc
        if count < 0:
            raise ParameterBindingError("transform_failed", "axis count cannot be negative", rule_id=rule_id)
        return list(range(1, count + 1))
    if transform == "parse_axes_ascii_and_bound_to_max_axis":
        text = str(value).strip()
        try:
            count = int(text, 10)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError("transform_failed", "axis count is not numeric", rule_id=rule_id) from exc
        return list(range(1, max(0, count) + 1))
    raise ParameterBindingError("transform_unimplemented", f"unsupported runtime transform: {transform}", rule_id=rule_id)


def apply_filter_operation(
    domain: Iterable[Any] | None,
    observed: Iterable[Any] | None,
    filter_mode: str,
    *,
    rule_id: str = "",
) -> dict[str, Any]:
    """Apply one evidence-backed include/exclude operation deterministically."""
    if domain is None:
        raise ParameterBindingError("domain_unknown", "target parameter legal domain is unknown", rule_id=rule_id)
    mode = str(filter_mode or "").strip().lower()
    if mode not in {"include", "exclude"}:
        raise ParameterBindingError("filter_mode_invalid", f"unsupported filter_mode: {filter_mode}", rule_id=rule_id)
    original = list(dict.fromkeys(domain))
    values = list(dict.fromkeys(observed or ()))
    if not original:
        # An explicitly known empty domain is valid and distinct from an
        # unavailable domain (None).
        return {"domain": [], "observed": values, "filter_mode": mode, "rule_id": rule_id}
    type_error = next(
        (item for item in values if not any(type(item) is type(candidate) for candidate in original)),
        None,
    )
    if type_error is not None:
        raise ParameterBindingError(
            "type_mismatch",
            f"observed value {type_error!r} is incompatible with target domain type",
            rule_id=rule_id,
        )
    observed_set = set(values)
    result = [item for item in original if (item in observed_set) == (mode == "include")]
    return {"domain": result, "observed": values, "filter_mode": mode, "rule_id": rule_id}


def bind_runtime_parameter_rules(
    target_api: str,
    execution_context: Any,
    rules: Iterable[ParameterRule | dict[str, Any]],
    *,
    parameter_domains: dict[str, Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Resolve all filters for one target from successful current-run calls.

    The function distinguishes failed calls, valid empty outputs, unknown
    domains, and empty results. It never consults historical memory.
    """
    normalized = [
        item if isinstance(item, ParameterRule) else parsed
        for item in rules
        for parsed in ((normalize_rules([item], target_api) if not isinstance(item, ParameterRule) else [item]))
    ]
    values_by_api = getattr(execution_context, "values_by_api", {}) or {}
    statuses_by_api = getattr(execution_context, "statuses_by_api", {}) or {}
    domains = parameter_domains or {}
    grouped: dict[str, list[ParameterRule]] = {}
    for rule in normalized:
        if rule.target_api == target_api:
            grouped.setdefault(rule.parameter, []).append(rule)
    result: dict[str, Any] = {
        "target_api": target_api, "success": True, "complete": True,
        "selected_parameters": {}, "bindings": [], "errors": [],
        "value_source": "successful_execution_context_only",
    }
    for parameter, parameter_rules in grouped.items():
        domain = list(domains[parameter]) if parameter in domains and domains[parameter] is not None else None
        current = list(domain) if domain is not None else None
        for rule in parameter_rules:
            statuses = list(statuses_by_api.get(rule.source_api, ()))
            responses = list(values_by_api.get(rule.source_api, ()))
            successful = [response for index, response in enumerate(responses)
                          if index < len(statuses) and statuses[index] == 0 and isinstance(response, dict)]
            binding: dict[str, Any] = {
                "rule_id": rule.rule_id, "source_api": rule.source_api,
                "source_field": (rule.source_fields[0] if rule.source_fields else ""),
                "target_parameter": parameter, "filter_mode": rule.filter_mode,
                "transform": rule.transform, "original_domain": list(current) if current is not None else None,
            }
            if not successful:
                code = "source_failed" if statuses and any(status != 0 for status in statuses) else "source_output_missing"
                message = f"{code}: no successful current-run output from {rule.source_api}"
                binding.update({"success": False, "error": message})
                result["bindings"].append(binding)
                result["errors"].append({"code": code, "message": message, "rule_id": rule.rule_id})
                result["success"] = result["complete"] = False
                continue
            incomplete_page = next(
                (
                    response for response in successful
                    if response.get("pagination_complete") is False
                    or response.get("partial") is True
                    or response.get("is_complete") is False
                ),
                None,
            )
            if incomplete_page is not None:
                message = f"source API {rule.source_api} returned an incomplete paginated result"
                binding.update({"success": False, "error": message})
                result["bindings"].append(binding)
                result["errors"].append({"code": "pagination_incomplete", "message": message, "rule_id": rule.rule_id})
                result["success"] = result["complete"] = False
                continue
            field = rule.source_fields[0] if rule.source_fields else parameter
            observed: list[Any] = []
            explicit_empty = False
            for response in successful:
                found, raw = _read_path(response, field)
                if not found and rule.source_api and field.startswith(f"{rule.source_api}."):
                    found, raw = _read_path(response, field[len(rule.source_api) + 1:])
                if not found:
                    continue
                if isinstance(raw, (list, tuple, set)):
                    explicit_empty = explicit_empty or len(raw) == 0
                    observed.extend(raw)
                else:
                    observed.append(raw)
            if not observed and explicit_empty:
                observed = []
            elif not observed:
                message = f"source field {rule.source_api}.{field} is missing or invalid"
                binding.update({"success": False, "error": message})
                result["bindings"].append(binding)
                result["errors"].append({"code": "source_field_missing", "message": message, "rule_id": rule.rule_id})
                result["success"] = result["complete"] = False
                continue
            try:
                transformed: list[Any] = []
                for item in observed:
                    transformed.extend(_transform_runtime_values(rule.transform, item, rule_id=rule.rule_id))
                filtered = apply_filter_operation(current, transformed, rule.filter_mode, rule_id=rule.rule_id)
            except ParameterBindingError as exc:
                binding.update({"success": False, "error": str(exc), "observed_values": observed})
                result["bindings"].append(binding)
                result["errors"].append({"code": exc.code, "message": str(exc), "rule_id": rule.rule_id})
                result["success"] = result["complete"] = False
                continue
            current = list(filtered["domain"])
            binding.update({
                "success": True, "observed_values": list(dict.fromkeys(transformed)),
                "final_values": list(current), "source_status": 0,
                "evidence_count": len(successful),
            })
            result["bindings"].append(binding)
        if current is not None and not any(item.get("target_parameter") == parameter and not item.get("success") for item in result["bindings"]):
            result["selected_parameters"][parameter] = current
    result["empty_result"] = any(value == [] for value in result["selected_parameters"].values())
    result["complete"] = result["success"] and len(result["selected_parameters"]) == len(grouped)
    return result


def bind_plan_runtime_parameters(
    plan: Any,
    execution_context: Any,
    rules_by_target: dict[str, Iterable[ParameterRule | dict[str, Any]]] | None = None,
    *,
    parameter_domains_by_target: dict[str, dict[str, Iterable[Any]]] | None = None,
    full_enum_threshold: int = 20,
    middle_sample_count: int = 5,
    update_plan: bool = True,
) -> dict[str, Any]:
    """Apply runtime bindings in place while preserving the original plan."""
    if plan is None:
        return {"success": False, "errors": [{"code": "plan_missing", "message": "ExecutionPlan is required"}]}
    rules_by_target = rules_by_target or {}
    domains_by_target = parameter_domains_by_target or {}
    results: dict[str, Any] = {}
    target_names = set(str(name) for name in rules_by_target) | set(str(name) for name in domains_by_target)
    for target_api in sorted(target_names):
        rules = rules_by_target.get(target_api, ())
        result = resolve_parameter_combinations(
            str(target_api), domains_by_target.get(str(target_api), {}),
            execution_context=execution_context, rules=rules,
            full_enum_threshold=full_enum_threshold,
            middle_sample_count=middle_sample_count,
        )
        results[str(target_api)] = result
        target_step = next(
            (step for step in getattr(plan, "steps", ())
             if str(getattr(step, "api", "") or getattr(step, "protocol_function", "") or getattr(step, "interface_name", "")) == str(target_api)),
            None,
        )
        if target_step is not None and update_plan:
            target_step.parameters.setdefault("bound_parameters", {})
            # Keep the complete legal result for auditability, while handing
            # CodeGenerationAgent only the deterministic execution selection.
            result["legal_parameters"] = dict(result.get("selected_parameters", {}))
            result["parameter_selections"] = {
                name: select_parameter_values(
                    values, full_enum_threshold=full_enum_threshold,
                    middle_sample_count=middle_sample_count,
                )
                for name, values in result.get("selected_parameters", {}).items()
            }
            target_step.parameters["bound_parameters"].update({
                name: choice["selected"] for name, choice in result["parameter_selections"].items()
            })
            target_step.parameters["runtime_binding_results"] = result.get("bindings", [])
        for binding in getattr(plan, "runtime_bindings", ()) if update_plan else ():
            if str(binding.get("target_api", "")) != str(target_api):
                continue
            matching = next((item for item in result.get("bindings", ()) if item.get("rule_id") == binding.get("rule_id")), None)
            if matching:
                binding.update({
                    "status": "bound" if matching.get("success") else "failed",
                    "selected_values": matching.get("final_values", []),
                    "observed_values": matching.get("observed_values", []),
                    "binding_error": matching.get("error", ""),
                })
    context = getattr(plan, "rag_context", {})
    context["runtime_parameter_bindings"] = results
    context["parameter_binding_status"] = {
        target: {"success": data.get("success", False), "complete": data.get("complete", False), "empty_result": data.get("empty_result", False)}
        for target, data in results.items()
    }
    return {"success": all(item.get("success", False) for item in results.values()) if results else True, "results": results}


def resolve_runtime_parameter_bindings(
    target_api: str,
    execution_context: Any,
    rules: Iterable[ParameterRule | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve graph ``filters`` rules from real prerequisite responses.

    This is deliberately deterministic and evidence scoped.  It reads only
    successful values already recorded in ``ExecutionContext``; an empty or
    failed source never produces a guessed target value.  The result is an
    execution artifact that the next generated step/repair can consume.
    """
    normalized = list(rules or ())
    values_by_api = getattr(execution_context, "values_by_api", {}) or {}
    statuses_by_api = getattr(execution_context, "statuses_by_api", {}) or {}
    selected: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for raw in normalized:
        rule = raw if isinstance(raw, ParameterRule) else (normalize_rules([raw], target_api) or [None])[0]
        if rule is None or rule.target_api != target_api:
            continue
        source = rule.source_api
        source_statuses = list(statuses_by_api.get(source, ())) if source else []
        source_values = list(values_by_api.get(source, ())) if source else []
        candidates: list[Any] = []
        evidence: list[dict[str, Any]] = []
        fields = tuple(rule.source_fields or rule.target_parameters or (rule.parameter,))
        for index, response in enumerate(source_values):
            status = source_statuses[index] if index < len(source_statuses) else None
            if status != 0 or not isinstance(response, dict):
                continue
            for field_name in fields:
                value = response
                found = True
                for part in str(field_name).split("."):
                    if not isinstance(value, dict) or part not in value:
                        found = False
                        break
                    value = value[part]
                if not found:
                    continue
                if isinstance(value, list):
                    values = value
                else:
                    values = [value]
                for candidate in values:
                    if candidate not in candidates:
                        candidates.append(candidate)
                        evidence.append({"source_api": source, "field": field_name, "value": candidate})
        context = {"candidates": candidates, "source": evidence}
        filtered = filter_parameter_candidates(target_api, rule.parameter, context, [rule])
        row = filtered.to_dict()
        row.update({"source_api": source, "rule_id": rule.rule_id, "target_parameter": rule.parameter})
        diagnostics.append(row)
        if filtered.success:
            selected[rule.parameter] = filtered.candidates
    return {
        "target_api": target_api,
        "selected_parameters": selected,
        "bindings": diagnostics,
        "complete": bool(diagnostics) and all(bool(row.get("success")) for row in diagnostics),
        "value_source": "successful_execution_context_only",
    }


def filter_parameter_candidates(
    api_name: str,
    parameter_name: str,
    context: dict[str, Any] | None = None,
    rules: Iterable[ParameterRule | dict[str, Any]] | None = None,
) -> ParameterToolResult:
    context = dict(context or {})
    normalized: list[ParameterRule] = []
    for item in rules or ():
        if isinstance(item, ParameterRule):
            normalized.append(item)
            continue
        parsed = normalize_rules([item], api_name)
        if parsed:
            normalized.append(parsed[0])
    applicable = [rule for rule in normalized if rule.parameter == parameter_name]
    raw_candidates = list(context.get("candidates", context.get("observed_values", ())) or ())
    source = list(context.get("source", context.get("evidence", ())) or ())
    excluded: list[dict[str, Any]] = []
    unverified: list[str] = []
    for rule in applicable:
        if rule.require_runtime_discovery and not raw_candidates:
            unverified.append(f"{rule.rule_id}: runtime discovery is required")
        if rule.allowed_values:
            if raw_candidates:
                raw_candidates = [value for value in raw_candidates if value in rule.allowed_values]
            else:
                raw_candidates = list(rule.allowed_values)
        protected = set(rule.excluded_values)
        for value in list(context.get("protected_values", ()) or ()):
            protected.add(value)
        if protected:
            kept = []
            for value in raw_candidates:
                if value in protected:
                    excluded.append({
                        "value": value,
                        "reason": "protected_program" if context.get("operation") == "download_new" else "protected_or_excluded",
                        "rule_id": rule.rule_id,
                    })
                else:
                    kept.append(value)
            raw_candidates = kept
    if context.get("operation") == "upload_existing":
        allowed = set(context.get("registered_values", ()) or ())
        if allowed:
            kept = []
            for value in raw_candidates:
                if value in allowed:
                    kept.append(value)
                else:
                    excluded.append({"value": value, "reason": "not_registered_for_upload"})
            raw_candidates = kept
        elif not raw_candidates:
            unverified.append("registered program discovery is incomplete for upload_existing")
    elif context.get("operation") == "download_new":
        registered = set(context.get("registered_values", ()) or ())
        protected = set(context.get("protected_values", ()) or ())
        kept = []
        for value in raw_candidates:
            if value in registered:
                excluded.append({"value": value, "reason": "already_registered"})
            elif value in protected:
                excluded.append({"value": value, "reason": "protected_program"})
            else:
                kept.append(value)
        raw_candidates = kept
    deduped = list(dict.fromkeys(raw_candidates))
    return ParameterToolResult(
        success=bool(deduped) and not unverified,
        api_name=api_name,
        parameter_name=parameter_name,
        candidates=deduped,
        excluded=excluded,
        source=source,
        unverified_conditions=unverified,
        error="; ".join(unverified),
    )


def generate_parameter_combinations(
    api_name: str,
    parameter_domains: dict[str, Iterable[Any]],
    budget: int,
    *,
    constraints: Iterable[Callable[[dict[str, Any]], bool]] | None = None,
    constraint_descriptions: Iterable[str] | None = None,
) -> ParameterToolResult:
    names = list(parameter_domains)
    domains = [list(dict.fromkeys(parameter_domains[name])) for name in names]
    if any(not domain for domain in domains):
        return ParameterToolResult(False, api_name, combinations=[], error="one or more parameter domains are empty")
    checks = list(constraints or ())
    combinations: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for values in itertools.product(*domains):
        row = dict(zip(names, values))
        if all(check(row) for check in checks):
            combinations.append(row)
        else:
            excluded.append({"values": row, "reason": "parameter_constraint"})
    actual = len(combinations)
    limit = max(1, int(budget))
    return ParameterToolResult(
        success=actual > 0,
        api_name=api_name,
        combinations=combinations[:limit],
        actual_combination_count=actual,
        excluded=excluded,
        unverified_conditions=list(constraint_descriptions or ()),
    )


def resolve_parameter_combinations(
    api_name: str,
    parameter_domains: dict[str, Iterable[Any]],
    *,
    execution_context: Any | None = None,
    rules: Iterable[ParameterRule | dict[str, Any]] = (),
    budget: int | None = None,
    constraints: Iterable[Callable[[dict[str, Any]], bool]] | None = None,
    full_enum_threshold: int = 20,
    middle_sample_count: int = 5,
) -> dict[str, Any]:
    """Run the complete deterministic filter -> select -> combine pipeline.

    ``pending_dependencies`` is deliberately distinct from an empty legal
    result.  A target whose filters depend on an API that has not run yet is
    deferred until the current execution produces evidence.
    """
    normalized = [
        item if isinstance(item, ParameterRule) else parsed
        for item in rules
        for parsed in ((normalize_rules([item], api_name) if not isinstance(item, ParameterRule) else [item]))
    ]
    grouped = {rule.parameter for rule in normalized if rule.target_api == api_name}
    if normalized and execution_context is None:
        return {
            "success": False, "status": "pending_dependencies", "target_api": api_name,
            "selected_parameters": {}, "parameter_selections": {}, "combinations": [],
            "errors": [], "pending_parameters": sorted(grouped),
        }
    if normalized:
        binding = bind_runtime_parameter_rules(
            api_name, execution_context, normalized, parameter_domains=parameter_domains,
        )
        if not binding.get("success"):
            source_missing = all(
                item.get("code") in {"source_output_missing", "source_field_missing"}
                for item in binding.get("errors", ())
            )
            if source_missing and execution_context is not None:
                return {
                    **binding, "status": "pending_dependencies", "pending_parameters": sorted(grouped),
                    "parameter_selections": {}, "combinations": [],
                }
            return {**binding, "status": "binding_failed", "parameter_selections": {}, "combinations": []}
        legal = dict(binding.get("selected_parameters", {}))
    else:
        legal = {str(name): list(dict.fromkeys(values or ())) for name, values in parameter_domains.items()}
        binding = {
            "success": True, "complete": True, "target_api": api_name,
            "selected_parameters": legal, "bindings": [], "errors": [],
        }
    selections = {
        name: select_parameter_values(
            values,
            full_enum_threshold=full_enum_threshold,
            middle_sample_count=middle_sample_count,
        )
        for name, values in legal.items()
    }
    selected_domains = {name: row["selected"] for name, row in selections.items()}
    if selected_domains and any(not values for values in selected_domains.values()):
        return {
            **binding, "status": "empty_result", "parameter_selections": selections,
            "combinations": [], "combination_count": 0,
            "errors": [*binding.get("errors", []), {
                "code": "parameter_combinations_empty",
                "message": "all applicable filters completed but a legal parameter domain is empty",
            }],
        }
    if not selected_domains:
        return {
            **binding, "status": "domain_unknown", "parameter_selections": {},
            "combinations": [], "combination_count": 0,
            "errors": [*binding.get("errors", []), {
                "code": "domain_unknown", "message": "no legal parameter domain is available",
            }],
        }
    combinations = generate_parameter_combinations(
        api_name, selected_domains, budget or max(1, math.prod(len(values) for values in selected_domains.values())),
        constraints=constraints,
    )
    return {
        **binding, "status": "ready" if combinations.success else "empty_result",
        "parameter_selections": selections, "combinations": combinations.combinations,
        "combination_count": combinations.actual_combination_count,
        "combination_error": combinations.error,
    }
