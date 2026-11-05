"""Deterministic extraction and evidence promotion for TRCM."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .execution_contracts import (
    API_INTERACTION_RULES,
    API_FUNCTION_FAMILY_RULES,
    API_TRAFFIC_SESSION_RULES,
    PROGRAM_OPERATION_CHAIN_RULES,
)
from .models import ExecutionPlan, ExecutionResult
from .cnc_parameter_prerequisites import extract_f0id_parameter_prerequisites
from .parameter_relationship_rules import resolve_rules
from .traffic_graph import (
    ApiNode,
    TrafficGraphError,
    TrafficGraphUnavailable,
    TrafficRelation,
    TrafficRelationshipService,
    relation_from_dict,
    relation_id,
    canonical_relation_type,
)


def _program_chain_context(source: str, target: str) -> dict[str, Any]:
    """Return evidence and parameter metadata for one documented chain edge."""
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        session = tuple(chain.get("session_calls", ()))
        pairs = list(zip(session, session[1:]))
        if (source, target) not in pairs:
            continue
        return {
            "evidence_ids": tuple(str(item) for item in chain.get("evidence_ids", ())),
            "applicability": {
                "scope": "documented_program_operation_chain",
                "chain_id": chain.get("chain_id", ""),
                "session_type": chain.get("session_type", ""),
                "required": True,
            },
            "attributes": {
                "chain_id": chain.get("chain_id", ""),
                "data_repeatable": bool(chain.get("data_repeatable", False)),
                "parameter_rules": list(chain.get("parameter_rules", ())),
                "dependency_path": "prerequisites.session",
            },
        }
    return {}


def _program_chain_guard_context(source: str, target: str) -> dict[str, Any]:
    """Return evidence for a non-adjacent lifecycle guard edge."""
    context = _program_chain_context(source, target)
    if context:
        return context
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        calls = tuple(chain.get("session_calls", ()))
        if source not in calls or target not in calls:
            continue
        return {
            "evidence_ids": tuple(str(item) for item in chain.get("evidence_ids", ())),
            "applicability": {
                "scope": "documented_program_operation_chain",
                "chain_id": chain.get("chain_id", ""),
                "session_type": chain.get("session_type", ""),
                "required": True,
                "lifecycle_guard": True,
            },
            "attributes": {
                "chain_id": chain.get("chain_id", ""),
                "lifecycle_guard": True,
                "data_repeatable": bool(chain.get("data_repeatable", False)),
                "parameter_rules": list(chain.get("parameter_rules", ())),
                "dependency_path": "prerequisites.session",
            },
        }
    return {}


def seed_documented_relations(
    service: TrafficRelationshipService,
    api_names: Iterable[str],
    *,
    api_metadata: dict[str, dict[str, Any]] | None = None,
    allowed_api_names: Iterable[str] | None = None,
    device_scope: str = "",
) -> list[dict[str, Any]]:
    """Migrate only explicit SDK/session rules as documented relations."""
    if not service.available:
        return []
    names = {str(name).strip() for name in api_names if str(name).strip()}
    scope_names = (
        {str(name).strip() for name in allowed_api_names if str(name).strip()}
        if allowed_api_names is not None else None
    )
    # Seed the requested targets plus their explicitly documented session
    # helpers. This lets a target-only Planner query see its complete known
    # chain without importing unrelated API families into the graph.
    expanded_names = set(names)
    for target in names:
        interaction = API_INTERACTION_RULES.get(target, {})
        expanded_names.update(
            str(item).strip()
            for key in ("before", "optional_before", "after", "on_error")
            for item in interaction.get(key, ())
            if str(item).strip() and (scope_names is None or str(item).strip() in scope_names)
        )
        session = API_TRAFFIC_SESSION_RULES.get(target, {})
        expanded_names.update(
            str(session.get(key, "")).strip()
            for key in ("start_api", "segment_api", "end_api")
            if str(session.get(key, "")).strip()
            and (scope_names is None or str(session.get(key, "")).strip() in scope_names)
        )
        discovery = str(session.get("optional_discovery_api", "")).strip()
        if discovery and (scope_names is None or discovery in scope_names):
            expanded_names.add(discovery)
        for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
            session_calls = tuple(str(item).strip() for item in chain.get("session_calls", ()) if str(item).strip())
            if target not in session_calls:
                continue
            discovery = chain.get("directory_discovery", {})
            for helper in (discovery.get("api", ""), discovery.get("provides_input_to", "")):
                helper_name = str(helper).strip()
                if helper_name and (scope_names is None or helper_name in scope_names):
                    expanded_names.add(helper_name)
    # Alarm history has a documented two-level prerequisite. Expand just this
    # chain transitively so selecting history5 also materializes stop -> count.
    if "cnc_rdalmhistry5" in names:
        expanded_names.update({"cnc_stopophis", "cnc_rdalmhisno", "cnc_startophis"} & (scope_names or {"cnc_stopophis", "cnc_rdalmhisno", "cnc_startophis"}))
        discovery = str(session.get("optional_discovery_api", "")).strip()
        if discovery and (scope_names is None or discovery in scope_names):
            expanded_names.add(discovery)
        # Program transfer chains expose an optional directory discovery API
        # separately from the required Start -> Data -> End session.  Expand
        # both endpoints here so a target-only seed can materialize the
        # PROVIDES_INPUT edge without making directory discovery mandatory.
        for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
            session_calls = tuple(str(item).strip() for item in chain.get("session_calls", ()) if str(item).strip())
            if target not in session_calls:
                continue
            discovery = chain.get("directory_discovery", {})
            for helper in (
                discovery.get("api", ""),
                discovery.get("provides_input_to", ""),
            ):
                helper_name = str(helper).strip()
                if helper_name and (scope_names is None or helper_name in scope_names):
                    expanded_names.add(helper_name)
    metadata = api_metadata or {}
    def scoped_relation_id(source: str, target: str, relation_type: str, rule: str) -> str:
        suffix = f"{rule}|scope={device_scope}" if device_scope else rule
        return relation_id(source, target, relation_type, suffix)

    def scoped_applicability(value: dict[str, Any]) -> dict[str, Any]:
        return {**value, **({"device_scope": device_scope} if device_scope else {})}

    for name in sorted(expanded_names):
        row = metadata.get(name, {})
        family = API_FUNCTION_FAMILY_RULES.get(name, {})
        node_metadata = {
            "source": row.get("source_url", ""),
            **({"device_scope": device_scope} if device_scope else {}),
            **({
                "knowledge_family": family.get("family_id", ""),
                "family_role": family.get("role", ""),
                "family_alternatives": list(family.get("alternatives", ())),
                "family_evidence_ids": list(family.get("evidence_ids", ())),
            } if family else {}),
        }
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{name}", name=name, protocol="FOCAS",
            sdk_signature_ref=str(row.get("chunk_id", "")),
            description=str(row.get("description", row.get("text", "")))[:500],
            metadata=node_metadata,
            parameters=dict(row.get("parameters", {}) or {}) if isinstance(row.get("parameters", {}), dict) else {},
            prerequisites=dict(row.get("prerequisites", {}) or {}) if isinstance(row.get("prerequisites", {}), dict) else {},
        ))
    relations: list[dict[str, Any]] = []
    alarm_chain_names = {"cnc_stopophis", "cnc_rdalmhisno", "cnc_rdalmhistry5", "cnc_startophis"}
    for target, dependencies in API_INTERACTION_RULES.items():
        if target not in names and not ("cnc_rdalmhistry5" in names and target in alarm_chain_names):
            continue
        dependency_path = (
            "prerequisites.alarm_history"
            if target in alarm_chain_names
            else "prerequisites.cnc_cexesram"
        )
        for source in dependencies.get("before", ()):
            if source not in expanded_names:
                continue
            context = _program_chain_guard_context(source, target)
            item = TrafficRelation(
                relation_id=scoped_relation_id(source, target, "REQUIRES", f"sdk_contract:{target}"),
                source_api=source, target_api=target, relation_type="prepares",
                rule_id=f"sdk_contract:{target}", confidence_status="documented",
                evidence_ids=tuple(str(value) for value in dependencies.get("evidence_ids", ()) if str(value)),
                applicability=scoped_applicability({"scope": "documented_session_contract"}),
                attributes={
                    "dependency_path": dependency_path,
                    **({"reason": str(dependencies.get("reason", ""))} if dependencies.get("reason") else {}),
                },
            )
            if context:
                item = TrafficRelation(
                    relation_id=item.relation_id, source_api=item.source_api, target_api=item.target_api,
                    relation_type=item.relation_type, rule_id=item.rule_id,
                    confidence_status=item.confidence_status, evidence_ids=context["evidence_ids"],
                    applicability=scoped_applicability(context["applicability"]), attributes=context["attributes"],
                )
            relations.append(service.call("upsert_relation", item))
        # ``after`` entries are ordered lifecycle edges from the target to
        # its cleanup/restore helper.  Persist them as canonical prepares
        # relations so the graph exposes the complete execution sequence.
        for destination in dependencies.get("after", ()):
            if destination not in expanded_names:
                continue
            item = TrafficRelation(
                relation_id=scoped_relation_id(target, destination, "prepares", f"sdk_contract:{target}:after"),
                source_api=target,
                target_api=destination,
                relation_type="prepares",
                rule_id=f"sdk_contract:{target}:after",
                confidence_status="documented",
                evidence_ids=tuple(str(value) for value in dependencies.get("evidence_ids", ()) if str(value)),
                applicability=scoped_applicability({"scope": "documented_session_contract", "phase": "restore"}),
                attributes={
                    "dependency_path": "prerequisites.session",
                    "order": "after",
                    **({"reason": str(dependencies.get("reason", ""))} if dependencies.get("reason") else {}),
                },
            )
            relations.append(service.call("upsert_relation", item))
        for source in dependencies.get("before", ()):
            if source not in expanded_names:
                continue
            context = _program_chain_context(source, target)
            if not context and any(
                source in tuple(chain.get("session_calls", ()))
                and target in tuple(chain.get("session_calls", ()))
                for chain in PROGRAM_OPERATION_CHAIN_RULES.values()
            ):
                # A non-adjacent Start -> End dependency may be useful as a
                # lifecycle guard, but it is not an ordered session edge.
                continue
            item = TrafficRelation(
                relation_id=scoped_relation_id(source, target, "SESSION_SEQUENCE", f"session:{target}"),
                source_api=source, target_api=target, relation_type="prepares",
                rule_id=f"session:{target}", confidence_status="documented",
                evidence_ids=tuple(str(value) for value in dependencies.get("evidence_ids", ()) if str(value)),
                applicability=scoped_applicability({"scope": "matching_api_version"}),
                attributes={
                    "dependency_path": dependency_path,
                    **({"reason": str(dependencies.get("reason", ""))} if dependencies.get("reason") else {}),
                },
            )
            if context:
                item = TrafficRelation(
                    relation_id=item.relation_id, source_api=item.source_api, target_api=item.target_api,
                    relation_type=item.relation_type, rule_id=item.rule_id,
                    confidence_status=item.confidence_status, evidence_ids=context["evidence_ids"],
                    applicability=scoped_applicability(context["applicability"]), attributes=context["attributes"],
                )
            relations.append(service.call("upsert_relation", item))

    # Materialize both adjacent edges even when the caller selected only the
    # data API.  This keeps the complete Start -> Data -> End session visible
    # without making the optional directory read part of the session.
    existing_relation_ids = {
        str(item.get("relation", {}).get("relation_id", ""))
        for item in relations if isinstance(item, dict)
    }
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        session_calls = tuple(str(item) for item in chain.get("session_calls", ()) if str(item).strip())
        if not session_calls or not any(item in names for item in session_calls):
            continue
        for source, target in zip(session_calls, session_calls[1:]):
            if source not in expanded_names or target not in expanded_names:
                continue
            context = _program_chain_context(source, target)
            for relation_type, rule_prefix in (("prepares", "sdk_contract"), ("prepares", "session")):
                rid = scoped_relation_id(source, target, relation_type, f"{rule_prefix}:{target}")
                if rid in existing_relation_ids:
                    continue
                item = TrafficRelation(
                    relation_id=rid,
                    source_api=source,
                    target_api=target,
                    relation_type=relation_type,
                    rule_id=f"{rule_prefix}:{target}",
                    confidence_status="documented",
                    evidence_ids=tuple(context.get("evidence_ids", ())),
                    applicability=scoped_applicability(dict(context.get("applicability", {}))),
                    attributes=dict(context.get("attributes", {})),
                )
                relations.append(service.call("upsert_relation", item))
                existing_relation_ids.add(rid)
    for target, session in API_TRAFFIC_SESSION_RULES.items():
        if target not in names:
            continue
        ordered_pairs = (
            (session.get("start_api"), target),
            (target, session.get("end_api")),
        )
        for source, destination in ordered_pairs:
            if not source or not destination or source == destination:
                continue
            if source not in expanded_names or destination not in expanded_names:
                continue
            chain = next(
                (candidate for candidate in PROGRAM_OPERATION_CHAIN_RULES.values()
                 if target in tuple(candidate.get("session_calls", ()))),
                {},
            )
            item = TrafficRelation(
                relation_id=scoped_relation_id(source, destination, "prepares", f"session:{session.get('session_type', '')}"),
                source_api=source, target_api=destination, relation_type="prepares",
                rule_id=f"session:{session.get('session_type', '')}", confidence_status="documented",
                evidence_ids=tuple(str(item) for item in chain.get("evidence_ids", ())),
                applicability=scoped_applicability({"session_type": session.get("session_type", ""), "required": True}),
                attributes={
                    "data_repeatable": bool(session.get("data_repeatable", False)),
                    "parameter_rules": list(chain.get("parameter_rules", ())),
                    "dependency_path": "prerequisites.session",
                    **({"end_condition": str(session.get("end_condition"))}
                       if session.get("end_condition") else {}),
                },
                # Session metadata is evidence-backed by the SDK chunks even
                # when there is no PROGRAM_OPERATION_CHAIN_RULES entry.
                # Keep the IDs on both edges for Neo4j auditability.
            )
            if session.get("evidence_ids"):
                item = TrafficRelation(
                    relation_id=item.relation_id, source_api=item.source_api, target_api=item.target_api,
                    relation_type=item.relation_type, rule_id=item.rule_id,
                    confidence_status=item.confidence_status,
                    evidence_ids=tuple(str(value) for value in session.get("evidence_ids", ())),
                    applicability=item.applicability, attributes=item.attributes,
                )
            relations.append(service.call("upsert_relation", item))

    # Directory reads are input providers, not mandatory session steps.  The
    # explicit edge lets Planner retrieve filtering rules without turning the
    # discovery call into a hard prerequisite.
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        discovery = chain.get("directory_discovery", {})
        discovery_api = str(discovery.get("api", "")).strip()
        provided_to = str(discovery.get("provides_input_to", "")).strip()
        target_api = str(chain.get("session_calls", ("",))[1]).strip()
        if not discovery_api or not provided_to or target_api not in names:
            continue
        if discovery_api not in expanded_names or provided_to not in expanded_names:
            continue
        relation = TrafficRelation(
            relation_id=scoped_relation_id(discovery_api, provided_to, "filters", f"program_discovery:{chain.get('chain_id', '')}"),
            source_api=discovery_api,
            target_api=provided_to,
            relation_type="filters",
            rule_id=f"program_discovery:{chain.get('chain_id', '')}",
            confidence_status="documented",
            evidence_ids=tuple(str(item) for item in chain.get("evidence_ids", ())),
            applicability=scoped_applicability({
                "scope": "optional_program_directory_discovery",
                "required": bool(discovery.get("required", False)),
                "required_if_missing_inputs": list(discovery.get("required_if_missing_inputs", ())),
                "runtime_selection": discovery.get("runtime_selection", ""),
                "operation": discovery.get("operation", ""),
            }),
            attributes={
                "optional": not bool(discovery.get("required", False)),
                "source_api": discovery_api,
                "selection_rule": discovery.get("selection_rule", ""),
                "forbid_default_values": dict(discovery.get("forbid_default_values", {})),
                "required_if_missing_inputs": list(discovery.get("required_if_missing_inputs", ())),
                "runtime_selection": discovery.get("runtime_selection", ""),
                "parameter_rules": list(chain.get("parameter_rules", ())),
                "chain_id": chain.get("chain_id", ""),
            },
        )
        relations.append(service.call("upsert_relation", relation))
    # Keep only current relations in the graph. Historical evidence remains
    # available in the mining report and backup artifacts.
    active_discovery_ids = {
        str(item.get("relation", {}).get("relation_id", ""))
        for item in relations
        if isinstance(item, dict)
        and str(item.get("relation", {}).get("rule_id", "")).startswith("program_discovery:")
    }
    store = service.store
    if active_discovery_ids and hasattr(store, "_run"):
        try:
            store._run(
                "MATCH ()-[r:TRG_RELATION]->() "
                "WHERE r.rule_id STARTS WITH 'program_discovery:' "
                "AND (coalesce($device_scope, '') = '' OR coalesce(r.applicability_json, '') CONTAINS $device_scope) "
                "AND NOT r.relation_id IN $active_ids "
                "DELETE r",
                device_scope=device_scope,
                active_ids=sorted(active_discovery_ids),
            )
        except (TrafficGraphUnavailable, TrafficGraphError):
            # A stopped optional Neo4j backend must not turn local, evidence
            # based planning into a repair_plan failure. The next run can
            # reconcile stale graph rows once the backend is available.
            pass
    elif active_discovery_ids and hasattr(store, "relations"):
        for relation_key, relation in list(store.relations.items()):
            if not relation.rule_id.startswith("program_discovery:"):
                continue
            if device_scope and relation.applicability.get("device_scope") not in {"", device_scope}:
                continue
            if relation_key in active_discovery_ids:
                continue
            store.relations.pop(relation_key, None)
    # Parameter discovery rules are a separate evidence-backed layer.  They
    # are added here so the existing Planner/Neo4j migration path remains the
    # single idempotent entry point for both session and input relations.
    relations.extend(seed_parameter_discovery_relations(
        service,
        names,
        api_metadata=metadata,
        allowed_api_names=scope_names,
        device_scope=device_scope,
    ))
    relations.extend(seed_cnc_parameter_prerequisite_relations(
        service,
        names,
        api_metadata=metadata,
        allowed_api_names=scope_names,
        device_scope=device_scope,
    ))
    # Legacy relation migration is deliberately not part of seeding.  The
    # standalone TRG audit command must classify each historical edge with
    # evidence before it can enter the canonical filters/prepares vocabulary.
    unique: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for item in relations:
        relation = item.get("relation", {}) if isinstance(item, dict) else {}
        key = (
            str(relation.get("source_api", "")),
            str(relation.get("target_api", "")),
            canonical_relation_type(str(relation.get("relation_type", ""))),
            str(relation.get("applicability", {}).get("device_scope", "")) if isinstance(relation.get("applicability", {}), dict) else "",
        )
        if key in seen_edges:
            continue
        seen_edges.add(key)
        unique.append(item)
    return unique


def seed_cnc_parameter_prerequisite_relations(
    service: TrafficRelationshipService,
    api_names: Iterable[str],
    *,
    api_metadata: dict[str, dict[str, Any]] | None = None,
    allowed_api_names: Iterable[str] | None = None,
    device_scope: str = "",
) -> list[dict[str, Any]]:
    """Persist documented CNC parameter checks and manual-only write paths.

    This is deliberately part of the existing TRCM seed path.  It does not
    create parameter nodes, choose a concrete value, or authorize a write.
    """
    if not service.available:
        return []
    names = {str(name).strip() for name in api_names if str(name).strip()}
    scope_names = (
        {str(name).strip() for name in allowed_api_names if str(name).strip()}
        if allowed_api_names is not None else None
    )
    report = extract_f0id_parameter_prerequisites(
        names,
        allowed_api_names=scope_names,
        device_scope=device_scope or "fanuc:0i-D",
    )
    metadata = api_metadata or {}
    endpoints = {
        str(item.get("source_api", "")).strip()
        for item in report["relations"]
    } | {
        str(item.get("target_api", "")).strip()
        for item in report["relations"]
    }
    for name in sorted(endpoints):
        if not name:
            continue
        row = metadata.get(name, {})
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{name}",
            name=name,
            protocol="FOCAS",
            sdk_signature_ref=str(row.get("chunk_id", "")),
            description=str(row.get("description", row.get("text", "")))[:500],
            metadata={"source": row.get("source_url", ""), **({"device_scope": device_scope} if device_scope else {})},
        ))
    prerequisite_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in report["relations"]:
        target = str(item.get("target_api", "")).strip()
        if target:
            prerequisite_by_target.setdefault(target, []).append({
                "source_api": item.get("source_api", ""),
                "relation_type": canonical_relation_type(str(item.get("relation_type", ""))),
                "rule_id": item.get("rule_id", ""),
                "parameter": dict(item.get("attributes", {}) or {}),
                "confidence_status": item.get("confidence_status", "proposed"),
                "evidence_ids": list(item.get("evidence_ids", ()) or ()),
            })
    for target, prerequisites in prerequisite_by_target.items():
        row = metadata.get(target, {})
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{target}", name=target, protocol="FOCAS",
            sdk_signature_ref=str(row.get("chunk_id", "")),
            description=str(row.get("description", row.get("text", "")))[:500],
            metadata={"source": row.get("source_url", ""), **({"device_scope": device_scope} if device_scope else {})},
            prerequisites={"rules": prerequisites},
        ))
    active_rule_ids = {str(item.get("rule_id", "")) for item in report["relations"]}
    # The extractor is authoritative for this scoped layer. Remove stale
    # previously promoted candidates when a narrower controller qualifier is
    # discovered; evidence remains in the extractor report.
    store = service.store
    if hasattr(store, "_run"):
        try:
            store._run(
                "MATCH ()-[r:TRG_RELATION]->() "
                "WHERE r.rule_id STARTS WITH 'parameter_prerequisite:' "
                "AND coalesce(r.applicability_json, '') CONTAINS $device_scope "
                "AND NOT r.rule_id IN $active_rule_ids "
                "DELETE r",
                device_scope=device_scope or "fanuc:0i-D",
                active_rule_ids=sorted(active_rule_ids),
            )
        except (TrafficGraphUnavailable, TrafficGraphError):
            # Relation cleanup is best effort during offline planning. It is
            # not evidence that the target API or its local rules failed.
            pass
    elif hasattr(store, "relations"):
        for relation_id_key, relation in list(store.relations.items()):
            if not relation.rule_id.startswith("parameter_prerequisite:"):
                continue
            if relation.applicability.get("device_scope") != (device_scope or "fanuc:0i-D"):
                continue
            if relation.rule_id in active_rule_ids:
                continue
            store.relations.pop(relation_id_key, None)
    output: list[dict[str, Any]] = []
    for item in report["relations"]:
        source = str(item.get("source_api", "")).strip()
        target = str(item.get("target_api", "")).strip()
        if not source or not target:
            continue
        scoped_rule = str(item.get("rule_id", ""))
        scope_suffix = f"{scoped_rule}|scope={device_scope}" if device_scope else scoped_rule
        relation = TrafficRelation(
            relation_id=relation_id(source, target, str(item["relation_type"]), scope_suffix),
            source_api=source,
            target_api=target,
            relation_type=canonical_relation_type(str(item["relation_type"])),
            rule_id=scoped_rule,
            confidence_status=str(item.get("confidence_status", "proposed")),
            evidence_ids=tuple(str(value) for value in item.get("evidence_ids", ()) if str(value)),
            applicability=dict(item.get("applicability", {}) or {}),
            attributes={
                **dict(item.get("attributes", {}) or {}),
                **({"dependency_path": "prerequisites.cnc_parameters"}
                   if canonical_relation_type(str(item["relation_type"])) == "prepares" else {}),
            },
        )
        output.append(service.call("upsert_relation", relation))
    return output


def seed_parameter_discovery_relations(
    service: TrafficRelationshipService,
    api_names: Iterable[str],
    *,
    api_metadata: dict[str, dict[str, Any]] | None = None,
    allowed_api_names: Iterable[str] | None = None,
    device_scope: str = "",
) -> list[dict[str, Any]]:
    """Persist only SDK/RAG-backed parameter supply rules.

    Concrete parameter values are deliberately absent.  The returned relation
    attributes tell Generator/Executor how to filter values obtained from the
    source call at runtime.
    """
    if not service.available:
        return []
    requested = {str(name).strip() for name in api_names if str(name).strip()}
    # A selected repeatable data segment also needs the documented discovery
    # binding for its session start.  Resolve that one extra target here so a
    # target-only seed still materializes getsraminfo -> sramgetstart.name.
    rule_targets = set(requested)
    for segment, session in API_TRAFFIC_SESSION_RULES.items():
        if segment not in requested:
            continue
        start = str(session.get("start_api", "")).strip()
        if start:
            rule_targets.add(start)
    report = resolve_rules(
        rule_targets,
        allowed_api_names=allowed_api_names,
        device_scope=device_scope,
    )
    metadata = api_metadata or {}
    scope_suffix = f"|scope={device_scope}" if device_scope else ""
    source_targets = {(str(item.get("source_api", "")), str(item.get("target_api", ""))) for item in report["relations"]}
    for source, target in source_targets:
        for name in (source, target):
            row = metadata.get(name, {})
            service.call("upsert_api", ApiNode(
                api_id=f"focas:{name}", name=name, protocol="FOCAS",
                sdk_signature_ref=str(row.get("chunk_id", "")),
                description=str(row.get("description", row.get("text", "")))[:500],
                metadata={"source": row.get("source_url", ""), **({"device_scope": device_scope} if device_scope else {})},
                parameters=dict(row.get("parameters", {}) or {}) if isinstance(row.get("parameters", {}), dict) else {},
                prerequisites=dict(row.get("prerequisites", {}) or {}) if isinstance(row.get("prerequisites", {}), dict) else {},
            ))
    parameter_by_target: dict[str, list[dict[str, Any]]] = {}
    for item in report["relations"]:
        target = str(item.get("target_api", "")).strip()
        if target:
            parameter_by_target.setdefault(target, []).append({
                "source_api": item.get("source_api", ""),
                "source_fields": list(item.get("source_fields", ()) or ()),
                "target_parameters": list(item.get("target_parameters", ()) or ()),
                "transform": item.get("transform", ""),
                "filter_strategy": item.get("filter_strategy", ""),
                "constraints": list(item.get("verification_requirements", ()) or ()),
                "confidence_status": item.get("confidence_status", "proposed"),
                "evidence_ids": list(item.get("evidence_ids", ()) or ()),
            })
    for target, parameters in parameter_by_target.items():
        row = metadata.get(target, {})
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{target}", name=target, protocol="FOCAS",
            sdk_signature_ref=str(row.get("chunk_id", "")),
            description=str(row.get("description", row.get("text", "")))[:500],
            metadata={"source": row.get("source_url", ""), **({"device_scope": device_scope} if device_scope else {})},
            parameters={"rules": parameters},
        ))
    output: list[dict[str, Any]] = []
    for item in report["relations"]:
        source = str(item["source_api"])
        target = str(item["target_api"])
        rule_id = str(item["rule_id"])
        applicability = dict(item.get("applicability", {}) or {})
        canonical_type = canonical_relation_type(str(item["relation_type"]))
        source_fields = [str(value) for value in (item.get("source_fields", ()) or ()) if str(value).strip()]
        target_parameters = [str(value) for value in (item.get("target_parameters", ()) or ()) if str(value).strip()]
        evidence_ids = tuple(str(evidence) for evidence in item.get("evidence_ids", ()) if str(evidence))

        # A filters edge describes one concrete output-to-input binding.  The
        # old rule records grouped several fields/parameters into one row,
        # which made it impossible for Planner and the audit to verify the
        # dependency path.  Expand those records into explicit canonical
        # edges and retain the grouped rules as attributes for runtime use.
        pairs = [(field, parameter) for field in source_fields for parameter in target_parameters]
        if canonical_type == "filters" and not pairs:
            pairs = [("", "")]
        if canonical_type != "filters":
            pairs = [("", "")]
        for source_field, dependency_path in pairs:
            filter_strategy = str(item.get("filter_strategy", ""))
            filter_mode = "exclude" if any(token in filter_strategy for token in ("unused", "exclude", "not_current")) else "include"
            attrs = {
                "discovery_family": rule_id.split(":", 1)[0],
                "source_fields": source_fields,
                "target_parameters": target_parameters,
                "transform": str(item.get("transform", "")),
                "filter_strategy": filter_strategy,
                "dangerous_operation": bool(item.get("dangerous_operation", False)),
                "auto_execute": bool(item.get("auto_execute", True)),
                "verification_requirements": list(item.get("verification_requirements", ()) or ()),
                "parameter_rules": [{
                    "rule_id": f"{rule_id}:{parameter}",
                    "source_api": source,
                    "source_fields": source_fields,
                    "target_parameters": target_parameters,
                    "parameter": parameter,
                    "transform": str(item.get("transform", "")),
                    "filter_strategy": filter_strategy,
                    "constraints": list(item.get("verification_requirements", ()) or ()),
                    "require_runtime_discovery": True,
                    "applicability": applicability,
                    "dangerous_operation": bool(item.get("dangerous_operation", False)),
                    "auto_execute": bool(item.get("auto_execute", True)),
                } for parameter in target_parameters],
            }
            if canonical_type == "filters":
                attrs.update({
                    "dependency_path": dependency_path,
                    "source_field": source_field,
                    "filter_mode": filter_mode,
                })
            relation = TrafficRelation(
                relation_id=relation_id(
                    source, target, str(item["relation_type"]),
                    f"{rule_id}:{dependency_path}:{source_field}{scope_suffix}",
                ),
                source_api=source,
                target_api=target,
                relation_type=canonical_type,
                rule_id=rule_id,
                confidence_status=str(item.get("confidence_status", "proposed")),
                evidence_ids=evidence_ids,
                applicability=applicability,
                attributes=attrs,
            )
            output.append(service.call("upsert_relation", relation))
    return output


def build_runtime_relation_payload(
    plan: ExecutionPlan,
    result: ExecutionResult,
) -> dict[str, Any]:
    """Build the structured relation observations passed to the graph tool.

    The Agent does not mutate Neo4j while constructing this payload. Unknown
    edges remain hypotheses; only documented edges retain their documented
    status until an explicit verification tool promotes them.
    """
    logs = list(result.api_logs)
    successes = [row for row in logs if row.status_code == 0 and not row.error]
    by_function = {row.protocol_function for row in successes if row.protocol_function}
    observations: list[dict[str, Any]] = []

    # A knowledge-base fallback is only promoted after the exact planned
    # source/target calls succeed in this run.  Mere adjacency in the log is
    # insufficient evidence because planners may include diagnostic helpers.
    planned_relations = [item for item in (getattr(plan, "dependencies", ()) or ()) if isinstance(item, dict)]
    planned_steps = {
        str(step.protocol_function or "").strip()
        for step in plan.steps
        if str(step.protocol_function or "").strip()
    }
    perfect_execution = bool(result.success) and not any(row.error or row.status_code != 0 for row in logs)
    if planned_steps and not planned_steps.issubset(by_function):
        perfect_execution = False
    target_api = str(getattr(plan, "target_api", "") or "").strip()
    if target_api and target_api not in by_function:
        perfect_execution = False

    for item in planned_relations:
        source = str(item.get("source_api", item.get("source", "")) or "").strip()
        target = str(item.get("target_api", item.get("target", "")) or "").strip()
        if not source or not target or source not in by_function or target not in by_function:
            continue
        relation = item.get("relation") if isinstance(item.get("relation"), dict) else item
        relation_type = canonical_relation_type(str(relation.get("relation_type", relation.get("relation", "prepares"))))
        attrs = dict(relation.get("attributes", {}) or {})
        for key in ("dependency_path", "source_field", "filter_mode", "transform"):
            if key in relation and key not in attrs:
                attrs[key] = relation[key]
        if relation_type == "filters" and not all(str(attrs.get(key, "")).strip() for key in ("dependency_path", "source_field", "filter_mode")):
            continue
        if relation_type == "prepares":
            attrs.setdefault("dependency_path", "prerequisites.session")
        rule_id = str(relation.get("rule_id", item.get("rule_id", f"runtime:planned:{plan.plan_id}")))
        rel = TrafficRelation(
            relation_id=relation_id(source, target, relation_type, f"{rule_id}:runtime:{plan.plan_id}"),
            source_api=source,
            target_api=target,
            relation_type=relation_type,
            rule_id=rule_id,
            # Knowledge-base candidates are reported for user review only;
            # even a perfect run must not mutate Neo4j automatically.
            confidence_status="proposed",
            evidence_ids=(plan.plan_id, result.task_id),
            applicability={"task_id": result.task_id, "source": "knowledge_base_fallback"},
            attributes={**attrs, "planned_by_knowledge_base": True, "complete_successful_calls": perfect_execution},
        )
        observations.append({
            "relation": asdict(rel),
            "evidence": [
                {"evidence_id": plan.plan_id, "kind": "knowledge_base_plan"},
                {"evidence_id": result.task_id, "kind": "complete_successful_api_calls"},
            ],
            "verification": {
                "success": perfect_execution,
                "basis": "knowledge_base_plan_and_complete_execution" if perfect_execution else "incomplete_execution",
                "status": "awaiting_user_approval",
            },
        })

    for step_index, first in enumerate(plan.steps):
        first_name = str(first.protocol_function or "").strip()
        if first_name not in by_function:
            continue
        for second in plan.steps[step_index + 1:]:
            second_name = str(second.protocol_function or "").strip()
            if not second_name or second_name not in by_function:
                continue
            # A sequence is only a hypothesis unless it is covered by a known
            # session contract; runtime observations alone do not prove need.
            known = any(
                first_name in tuple(item.get("before", ())) and second_name == target
                for target, item in API_INTERACTION_RULES.items()
            )
            rel = TrafficRelation(
                relation_id=relation_id(first_name, second_name, "PRECEDES", f"runtime:{plan.plan_id}"),
                source_api=first_name, target_api=second_name, relation_type="prepares",
                rule_id=f"runtime:{plan.plan_id}",
                confidence_status="documented" if known else "unconfirmed",
                evidence_ids=(plan.plan_id, result.task_id),
                applicability={
                    "task_id": result.task_id,
                    **({"device_scope": str(plan.rag_context.get("task_permissions", {}).get("device_scope", ""))}
                       if plan.rag_context.get("task_permissions", {}).get("device_scope") else {}),
                },
                attributes={
                    "complete_successful_calls": True,
                    "requires_validation": not known,
                    **({"dependency_path": "prerequisites.session"} if known else {}),
                },
            )
            observations.append({
                "relation": asdict(rel),
                "evidence": [
                    {"evidence_id": plan.plan_id, "kind": "plan_step_order"},
                    {"evidence_id": result.task_id, "kind": "successful_api_calls"},
                ],
                "verification": {
                    "success": bool(known),
                    "basis": "documented_contract" if known else "runtime_observation_only",
                },
                "validation_plan": None if known else {
                    "replay_with_source_omitted": True,
                    "observe_target_success": True,
                    "required_evidence": ["source omission trial", "target return code", "wire correlation"],
                },
            })
    return {
        "schema_version": "trcm.runtime_relation_observations.v1",
        "source": "ExecutionAgent",
        "plan_id": plan.plan_id,
        "task_id": result.task_id,
        "execution_success": bool(result.success),
        "perfect_execution": perfect_execution,
        "observed_successful_apis": sorted(by_function),
        "discovered_relations": [
            {
                "source_api": item["relation"]["source_api"],
                "target_api": item["relation"]["target_api"],
                "relation_type": item["relation"]["relation_type"],
                "rule_id": item["relation"]["rule_id"],
                "attributes": item["relation"]["attributes"],
                "status": "awaiting_user_approval",
            }
            for item in observations
            if item.get("verification", {}).get("success")
            and item.get("relation", {}).get("attributes", {}).get("planned_by_knowledge_base")
        ],
        "relations": observations,
    }


def persist_runtime_relation_payload(
    service: TrafficRelationshipService,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist only confirmed structured relations through the graph boundary."""
    if not service.available:
        return []
    output: list[dict[str, Any]] = []
    for item in payload.get("relations", []) if isinstance(payload.get("relations", []), list) else []:
        if not isinstance(item, dict):
            continue
        relation_payload = item.get("relation", item)
        if not isinstance(relation_payload, dict):
            continue
        try:
            relation = relation_from_dict(relation_payload)
        except (TypeError, ValueError):
            continue
        if relation.confidence_status not in {"documented", "sdk_validated", "runtime_verified"}:
            output.append({
                "ok": False,
                "stored": False,
                "status": "unconfirmed",
                "relation_id": relation.relation_id,
                "source_api": relation.source_api,
                "target_api": relation.target_api,
                "reason": "execution feedback is observational only; unknown dependencies are not written",
            })
            continue
        output.append(service.call("upsert_relation", relation))
    return output


def mine_runtime_relations(
    service: TrafficRelationshipService,
    plan: ExecutionPlan,
    result: ExecutionResult,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers outside ToolRuntime."""
    return persist_runtime_relation_payload(service, build_runtime_relation_payload(plan, result))
