"""Deterministic extraction and evidence promotion for TRCM."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from .execution_contracts import API_INTERACTION_RULES, API_TRAFFIC_SESSION_RULES
from .models import ExecutionPlan, ExecutionResult
from .traffic_graph import ApiNode, TrafficRelation, TrafficRelationshipService, relation_id


def seed_documented_relations(
    service: TrafficRelationshipService,
    api_names: Iterable[str],
    *,
    api_metadata: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Migrate only explicit SDK/session rules as documented relations."""
    if not service.available:
        return []
    names = {str(name).strip() for name in api_names if str(name).strip()}
    # Seed the requested targets plus their explicitly documented session
    # helpers. This lets a target-only Planner query see its complete known
    # chain without importing unrelated API families into the graph.
    expanded_names = set(names)
    for target in names:
        interaction = API_INTERACTION_RULES.get(target, {})
        expanded_names.update(
            str(item).strip()
            for key in ("before", "after", "on_error")
            for item in interaction.get(key, ())
            if str(item).strip()
        )
        session = API_TRAFFIC_SESSION_RULES.get(target, {})
        expanded_names.update(
            str(session.get(key, "")).strip()
            for key in ("start_api", "segment_api", "end_api")
            if str(session.get(key, "")).strip()
        )
    metadata = api_metadata or {}
    for name in sorted(expanded_names):
        row = metadata.get(name, {})
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{name}", name=name, protocol="FOCAS",
            sdk_signature_ref=str(row.get("chunk_id", "")),
            description=str(row.get("description", row.get("text", "")))[:500],
            metadata={"source": row.get("source_url", "")},
        ))
    relations: list[dict[str, Any]] = []
    for target, dependencies in API_INTERACTION_RULES.items():
        if target not in names:
            continue
        for source in dependencies.get("before", ()):
            if source not in expanded_names:
                continue
            item = TrafficRelation(
                relation_id=relation_id(source, target, "REQUIRES", f"sdk_contract:{target}"),
                source_api=source, target_api=target, relation_type="REQUIRES",
                rule_id=f"sdk_contract:{target}", confidence_status="documented",
                applicability={"scope": "documented_session_contract"},
            )
            relations.append(service.call("upsert_relation", item))
        for source in dependencies.get("before", ()):
            if source not in expanded_names:
                continue
            item = TrafficRelation(
                relation_id=relation_id(source, target, "SESSION_SEQUENCE", f"session:{target}"),
                source_api=source, target_api=target, relation_type="SESSION_SEQUENCE",
                rule_id=f"session:{target}", confidence_status="documented",
                applicability={"scope": "matching_api_version"},
            )
            relations.append(service.call("upsert_relation", item))
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
            item = TrafficRelation(
                relation_id=relation_id(source, destination, "PRECEDES", f"session:{session.get('session_type', '')}"),
                source_api=source, target_api=destination, relation_type="PRECEDES",
                rule_id=f"session:{session.get('session_type', '')}", confidence_status="documented",
                applicability={"session_type": session.get("session_type", "")},
            )
            relations.append(service.call("upsert_relation", item))
    return relations


def mine_runtime_relations(
    service: TrafficRelationshipService,
    plan: ExecutionPlan,
    result: ExecutionResult,
) -> list[dict[str, Any]]:
    """Propose or verify relations from complete, successful evidence only."""
    if not service.available:
        return []
    logs = list(result.api_logs)
    successes = [row for row in logs if row.status_code == 0 and not row.error]
    by_function = {row.protocol_function for row in successes if row.protocol_function}
    for function in sorted(by_function):
        # Runtime evidence may arrive before a Planner graph seed (for
        # example, a manually supplied plan). Register only the observed API
        # node; any relationship inferred below remains proposed unless an
        # explicit documented contract proves it.
        service.call("upsert_api", ApiNode(
            api_id=f"focas:{function}", name=function, protocol="FOCAS",
            metadata={"evidence_task_id": result.task_id},
        ))
    output: list[dict[str, Any]] = []
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
                source_api=first_name, target_api=second_name, relation_type="PRECEDES",
                rule_id=f"runtime:{plan.plan_id}",
                confidence_status="documented" if known else "proposed",
                evidence_ids=(plan.plan_id, result.task_id),
                applicability={"task_id": result.task_id},
                attributes={"complete_successful_calls": True, "requires_validation": not known},
            )
            upsert = service.call("upsert_relation", rel)
            if known:
                output.append(upsert)
            else:
                output.append({"proposal": upsert, "validation_plan": {
                    "replay_with_source_omitted": True,
                    "observe_target_success": True,
                    "required_evidence": ["source omission trial", "target return code", "wire correlation"],
                }})
    return output
