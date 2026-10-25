"""Evidence-scoped parameter discovery, filtering, and combination generation."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class ParameterRule:
    rule_id: str
    target_api: str
    parameter: str
    source_api: str = ""
    filter_strategy: str = "identity"
    require_runtime_discovery: bool = False
    allowed_values: tuple[Any, ...] = ()
    excluded_values: tuple[Any, ...] = ()
    constraints: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    applicability: dict[str, Any] | None = None


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


def normalize_rules(raw_rules: Iterable[dict[str, Any]] | None, api_name: str) -> list[ParameterRule]:
    rules: list[ParameterRule] = []
    for index, raw in enumerate(raw_rules or (), 1):
        if not isinstance(raw, dict):
            continue
        relation = raw.get("attributes", raw)
        relation = relation if isinstance(relation, dict) else raw
        parameter = str(relation.get("parameter") or relation.get("parameter_name") or "").strip()
        if not parameter:
            continue
        rules.append(ParameterRule(
            rule_id=str(relation.get("rule_id") or f"graph:{api_name}:{parameter}:{index}"),
            target_api=api_name,
            parameter=parameter,
            source_api=str(relation.get("source_api") or raw.get("source_api") or ""),
            filter_strategy=str(relation.get("filter_strategy") or "identity"),
            require_runtime_discovery=bool(relation.get("require_runtime_discovery", False)),
            allowed_values=tuple(relation.get("allowed_values", relation.get("values", ())) or ()),
            excluded_values=tuple(relation.get("excluded_values", relation.get("protected_values", ())) or ()),
            constraints=tuple(str(item) for item in relation.get("constraints", ()) or ()),
            evidence_ids=tuple(str(item) for item in relation.get("evidence_ids", ()) or ()),
            applicability=dict(relation.get("applicability", {}) or {}),
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
                    excluded.append({"value": value, "reason": "protected_or_excluded", "rule_id": rule.rule_id})
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
