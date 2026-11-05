"""Evidence-aware audit and incremental migration for the Neo4j TRG.

This module is intentionally separate from Planner reads.  A graph read never
rewrites historical relations.  Migration is an explicit, repeatable command
which backs up the graph, classifies each old edge against the local SDK/RAG
and execution contracts, and only then writes canonical ``filters`` and
``prepares`` facts.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .api_compatibility import F0ID_SCOPE_ID, f0id_api_names
from .cnc_parameter_prerequisites import extract_f0id_parameter_prerequisites
from .execution_contracts import API_TRAFFIC_SESSION_RULES, PROGRAM_OPERATION_CHAIN_RULES
from .focas_contracts import documented_parameter_names, named_prototype
from .parameter_relationship_rules import resolve_rules
from .traffic_graph import CANONICAL_RELATION_TYPES, LEGACY_RELATION_TYPE_MAP


VALID_STATUSES = {"documented", "sdk_validated", "runtime_verified"}
DOWNLOAD_PROGRAM_TRANSFORM = "construct_nc_program_with_leading_O_number"


def _json(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, type(fallback)) else fallback
        except (TypeError, ValueError):
            return fallback
    return fallback


def _relation(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row.get("relation") or row.get("r") or {})
    value["source_api"] = str(row.get("source_api") or value.get("source_api") or "")
    value["target_api"] = str(row.get("target_api") or value.get("target_api") or "")
    value["attributes"] = _json(value.get("attributes_json"), {})
    value["applicability"] = _json(value.get("applicability_json"), {})
    value["evidence_ids"] = list(value.get("evidence_ids") or [])
    return value


def read_graph(store: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read graph facts without changing Neo4j."""
    if hasattr(store, "_run"):
        nodes = store._run("MATCH (n:API) RETURN n{.*} AS node")
        rows = store._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) "
            "RETURN s.name AS source_api, t.name AS target_api, r{.*} AS relation"
        )
        return [dict(row.get("node") or {}) for row in nodes], [_relation(row) for row in rows]
    nodes = []
    for key, node in getattr(store, "nodes", {}).items():
        if key == getattr(node, "name", None):
            nodes.append({
                "api_id": node.api_id, "name": node.name, "protocol": node.protocol,
                "sdk_signature_ref": node.sdk_signature_ref, "description": node.description,
                "parameters_json": json.dumps(node.parameters, ensure_ascii=False),
                "prerequisites_json": json.dumps(node.prerequisites, ensure_ascii=False),
            })
    relations = []
    for item in getattr(store, "relations", {}).values():
        relations.append({
            "relation_id": item.relation_id, "source_api": item.source_api,
            "target_api": item.target_api, "relation_type": item.relation_type,
            "rule_id": item.rule_id, "confidence_status": item.confidence_status,
            "evidence_ids": list(item.evidence_ids), "attributes": dict(item.attributes),
            "applicability": dict(item.applicability), "version": item.version,
        })
    return nodes, relations


def audit_graph(store: Any) -> dict[str, Any]:
    nodes, relations = read_graph(store)
    type_counts = Counter(str(item.get("relation_type", "")) for item in relations)
    status_counts = Counter(str(item.get("confidence_status", "")) for item in relations)
    missing_evidence = [str(item.get("relation_id", "")) for item in relations if not item.get("evidence_ids")]
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item in relations:
        groups[(str(item.get("source_api", "")), str(item.get("target_api", "")), str(item.get("relation_type", "")))].append(str(item.get("relation_id", "")))
    duplicate_groups = [
        {"source_api": key[0], "target_api": key[1], "relation_type": key[2], "relation_ids": ids}
        for key, ids in groups.items() if len(ids) > 1
    ]
    return {
        "api_nodes": len(nodes),
        "relations": len(relations),
        "relation_type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_count": len(missing_evidence),
        "missing_evidence_relation_ids": missing_evidence,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "empty_node_parameters": sum(not _json(node.get("parameters_json"), {}) for node in nodes),
        "empty_node_prerequisites": sum(not _json(node.get("prerequisites_json"), {}) for node in nodes),
    }


def _path_value(value: Any, path: str) -> Any:
    current = value
    for part in str(path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _parameter_definition(knowledge: dict[str, Any], path: str) -> dict[str, Any] | None:
    """Resolve a top-level parameter or a nested SDK structure field."""
    current: Any = knowledge.get("parameters", {})
    parts = [part for part in str(path or "").split(".") if part]
    for index, part in enumerate(parts):
        if not isinstance(current, dict):
            return None
        if part in current:
            current = current[part]
        elif isinstance(current.get("fields"), dict) and part in current["fields"]:
            current = current["fields"][part]
        else:
            return None
        # SDK struct fields are stored below the parent parameter's ``fields``
        # map.  A field may itself have nested fields in generated indexes.
        if index < len(parts) - 1 and not isinstance(current, dict):
            return None
    return current if isinstance(current, dict) else None


def _base_type(definition: dict[str, Any] | None) -> str:
    if not isinstance(definition, dict):
        return ""
    value = str(definition.get("type", "")).lower()
    value = value.replace("const", "").replace("volatile", "")
    value = value.replace("*", " ")
    return " ".join(value.split())


def _types_compatible(source: str, target: str) -> bool:
    """Compare SDK scalar types conservatively, allowing common aliases."""
    aliases = {
        "short int": "short", "signed short": "short", "unsigned short int": "unsigned short",
        "long int": "long", "signed long": "long", "unsigned long int": "unsigned long",
        "signed int": "int", "unsigned": "unsigned int", "long long int": "long long",
    }
    source = aliases.get(source, source)
    target = aliases.get(target, target)
    if not source or not target:
        return True
    if source == target:
        return True
    # Do not silently bind different-width or different-signedness values.
    # A documented transform is required for those conversions.
    return False


def _filter_semantic_error(
    source: str,
    target: str,
    attrs: dict[str, Any],
    node_knowledge: dict[str, dict[str, Any]],
) -> str | None:
    target_path = str(attrs.get("dependency_path", ""))
    source_path = str(attrs.get("source_field", ""))
    source_def = _parameter_definition(node_knowledge.get(source, {}), source_path)
    target_def = _parameter_definition(node_knowledge.get(target, {}), target_path)
    if not source_def or not target_def:
        return "filters path does not resolve"
    source_root = source_path.split(".", 1)[0]
    target_root = target_path.split(".", 1)[0]
    source_root_def = _parameter_definition(node_knowledge.get(source, {}), source_root)
    target_root_def = _parameter_definition(node_knowledge.get(target, {}), target_root)
    if not isinstance(source_root_def, dict) or source_root_def.get("direction") not in {"out", "inout"}:
        return "source field is not an output/inout value"
    if not isinstance(target_root_def, dict) or target_root_def.get("direction") not in {"in", "inout"}:
        return "target dependency is not an input/inout value"
    source_type = _base_type(source_def)
    target_type = _base_type(target_def)
    transform = str(attrs.get("transform", ""))
    # A directory number is a scalar output. cnc_download4's formal data
    # argument is a char buffer; it must be constructed with O<number>, never
    # treated as a set of scalar values to exclude from the whole buffer.
    if source == "cnc_rdprogdir3" and target in {"cnc_download3", "cnc_download4"} and source_path.endswith(".number") and target_path == "data":
        if attrs.get("filter_mode") != "include":
            return "program number cannot use exclude against the NC data buffer"
        if transform != DOWNLOAD_PROGRAM_TRANSFORM:
            return "program number to NC data requires explicit construction transform"
        return None
    if source_type and target_type and not _types_compatible(source_type, target_type) and not transform:
        return f"incompatible types {source_type!r} -> {target_type!r} without transform"
    if attrs.get("filter_mode") == "exclude" and target_type in {"char", "unsigned char"}:
        return "exclude is not valid for a byte buffer target"
    return None


def _session_rules() -> dict[tuple[str, str], dict[str, Any]]:
    rules: dict[tuple[str, str], dict[str, Any]] = {}
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        calls = tuple(str(item) for item in chain.get("session_calls", ()) if str(item))
        for source, target in zip(calls, calls[1:]):
            rules[(source, target)] = chain
    for target, session in API_TRAFFIC_SESSION_RULES.items():
        start, end = session.get("start_api"), session.get("end_api")
        if start and target:
            rules[(str(start), str(target))] = session
        if target and end:
            rules[(str(target), str(end))] = session
    return rules


def _prepares_semantic_error(source: str, target: str, path: str, knowledge: dict[str, dict[str, Any]]) -> str | None:
    if path.removeprefix("prerequisites.") == "session":
        session = _session_rules().get((source, target))
        return None if session and _session_source_error(source, target, {"session_calls": session.get("session_calls", ()) or (session.get("start_api"), session.get("segment_api", target), session.get("end_api"))}) is None else "prepares edge is not a matching session transition"
    prereq = knowledge.get(target, {}).get("prerequisites", {})
    value = _path_value(prereq, path.removeprefix("prerequisites."))
    if value is None:
        return "prepares dependency path does not resolve"
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if source in {str(entry.get("read_api", "")), str(entry.get("prepare_api", "")), str(entry.get("write_api", ""))}:
            return None
    return "source API is not declared as a provider of the target prerequisite"


def _session_source_error(source: str, target: str, session: dict[str, Any]) -> str | None:
    raw_calls = session.get("session_calls", ())
    if not raw_calls:
        raw_calls = (session.get("start_api"), session.get("segment_api", target), session.get("end_api"))
    calls = [str(item) for item in raw_calls if str(item)]
    if target not in calls or source not in calls:
        return "source/target are outside the documented session"
    source_index, target_index = calls.index(source), calls.index(target)
    if source_index >= target_index:
        return "source does not precede target in the session"
    # A target data call requires its matching start; an end call can be
    # prepared by the start or by a successful data call.  This prevents
    # arbitrary API-order edges from becoming executable prerequisites.
    if target_index == 1 and source_index != 0:
        return "session data target requires the matching start API"
    return None


def validate_canonical_edges(store: Any) -> dict[str, Any]:
    """Validate dependency paths against the persisted node JSON."""
    nodes, relations = read_graph(store)
    by_name = {
        str(node.get("name", "")): {
            "parameters": _json(node.get("parameters_json"), {}) or node.get("parameters", {}),
            "prerequisites": _json(node.get("prerequisites_json"), {}) or node.get("prerequisites", {}),
        }
        for node in nodes
    }
    invalid: list[dict[str, Any]] = []
    effective = Counter()
    for item in relations:
        if str(item.get("confidence_status")) not in VALID_STATUSES:
            continue
        kind = str(item.get("relation_type", ""))
        if kind not in CANONICAL_RELATION_TYPES:
            invalid.append({"relation_id": item.get("relation_id"), "reason": "non-canonical relation type"})
            continue
        attrs = item.get("attributes") or {}
        source = str(item.get("source_api", ""))
        target = str(item.get("target_api", ""))
        if kind == "filters":
            target_path = str(attrs.get("dependency_path", ""))
            source_path = str(attrs.get("source_field", ""))
            target_value = _path_value(by_name.get(target, {}).get("parameters", {}), target_path)
            source_parts = source_path.split(".", 1)
            source_value = _path_value(by_name.get(source, {}).get("parameters", {}), source_path)
            if source_value is None and len(source_parts) == 2:
                source_value = _path_value(by_name.get(source, {}).get("parameters", {}).get(source_parts[0], {}).get("fields", {}), source_parts[1])
            source_def = _path_value(by_name.get(source, {}).get("parameters", {}), source_parts[0]) if source_parts else None
            if target_value is None or source_value is None or not isinstance(source_def, dict) or source_def.get("direction") not in {"out", "inout"}:
                invalid.append({"relation_id": item.get("relation_id"), "reason": "filters path does not resolve", "dependency_path": target_path, "source_field": source_path})
                continue
            semantic_error = _filter_semantic_error(source, target, attrs, by_name)
            if semantic_error:
                invalid.append({"relation_id": item.get("relation_id"), "reason": semantic_error, "dependency_path": target_path, "source_field": source_path})
                continue
        else:
            dependency_path = str(attrs.get("dependency_path", ""))
            if _path_value(by_name.get(target, {}).get("prerequisites", {}), dependency_path.removeprefix("prerequisites.")) is None and _path_value(by_name.get(target, {}).get("prerequisites", {}), dependency_path) is None:
                invalid.append({"relation_id": item.get("relation_id"), "reason": "prepares path does not resolve", "dependency_path": dependency_path})
                continue
            semantic_error = _prepares_semantic_error(source, target, dependency_path, by_name)
            if semantic_error:
                invalid.append({"relation_id": item.get("relation_id"), "reason": semantic_error, "dependency_path": dependency_path})
                continue
        effective[kind] += 1
    return {"effective_relation_counts": dict(effective), "invalid_effective_edge_count": len(invalid), "invalid_effective_edges": invalid}


def audit_confirmed_relations(store: Any) -> dict[str, Any]:
    """Audit only confirmed filters/prepares with evidence and semantics.

    Status labels are treated as hints. An edge is retained only when its
    fields resolve against the API node contract, its direction is valid, its
    semantic rule matches the documented session/parameter contract, and it
    carries evidence IDs. Equivalent duplicate facts are reduced to the
    strongest evidence-bearing record.
    """
    nodes, relations = read_graph(store)
    by_name = {
        str(node.get("name", "")): {
            "parameters": _json(node.get("parameters_json"), {}) or node.get("parameters", {}),
            "prerequisites": _json(node.get("prerequisites_json"), {}) or node.get("prerequisites", {}),
        }
        for node in nodes
    }
    confirmed: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    unconfirmed: list[dict[str, Any]] = []
    for item in relations:
        status = str(item.get("confidence_status", ""))
        if status not in VALID_STATUSES:
            unconfirmed.append({"relation_id": item.get("relation_id", ""), "status": status or "unknown"})
            continue
        kind = str(item.get("relation_type", ""))
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        source = str(item.get("source_api", ""))
        target = str(item.get("target_api", ""))
        reason = ""
        if kind not in CANONICAL_RELATION_TYPES:
            reason = "relation type is not filters or prepares"
        elif not item.get("evidence_ids"):
            reason = "confirmed relation has no evidence_ids"
        elif kind == "filters":
            for key in ("dependency_path", "source_field", "filter_mode"):
                if not str(attrs.get(key, "")).strip():
                    reason = f"filters relation is missing {key}"
                    break
            if not reason:
                reason = _filter_semantic_error(source, target, attrs, by_name) or ""
        elif kind == "prepares":
            if not str(attrs.get("dependency_path", "")).strip():
                reason = "prepares relation is missing dependency_path"
            else:
                reason = _prepares_semantic_error(source, target, str(attrs.get("dependency_path", "")), by_name) or ""
        if reason:
            invalid.append({"relation_id": item.get("relation_id", ""), "reason": reason, "source_api": source, "target_api": target, "relation_type": kind})
        else:
            confirmed.append(item)

    status_rank = {"documented": 1, "sdk_validated": 2, "runtime_verified": 3}
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in confirmed:
        attrs = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        groups[(
            str(item.get("source_api", "")), str(item.get("target_api", "")), str(item.get("relation_type", "")),
            str(attrs.get("dependency_path", "")), str(attrs.get("source_field", "")),
            str(attrs.get("filter_mode", "")), str(attrs.get("transform", "")),
        )].append(item)
    retained: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key, rows in groups.items():
        rows.sort(key=lambda item: (
            status_rank.get(str(item.get("confidence_status", "")), 0),
            len(item.get("evidence_ids") or []),
            str(item.get("relation_id", "")),
        ), reverse=True)
        retained.append(rows[0])
        for duplicate in rows[1:]:
            duplicates.append({"relation_id": duplicate.get("relation_id", ""), "kept_relation_id": rows[0].get("relation_id", ""), "semantic_key": list(key)})
    counts = Counter(str(item.get("relation_type", "")) for item in retained)
    return {
        "api_nodes": len(nodes),
        "total_relations": len(relations),
        "confirmed_input_count": len(confirmed),
        "unconfirmed_count": len(unconfirmed),
        "unconfirmed_relations": unconfirmed,
        "invalid_confirmed_count": len(invalid),
        "invalid_confirmed_relations": invalid,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "retained_relation_ids": sorted(str(item.get("relation_id", "")) for item in retained),
        "retained_relation_count": len(retained),
        "retained_relation_counts": dict(sorted(counts.items())),
        "retained_relations": retained,
    }


def cleanup_confirmed_graph(store: Any, *, backup_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    """Backup and reduce Neo4j to semantically audited confirmed relations."""
    nodes, relations = read_graph(store)
    before = audit_graph(store)
    backup = write_backup(backup_dir, nodes, relations, before)
    audit = audit_confirmed_relations(store)
    delete_ids = {
        str(item.get("relation_id", "")) for item in audit["invalid_confirmed_relations"]
    }
    delete_ids.update(str(item.get("relation_id", "")) for item in audit["duplicates"])
    delete_ids.update(str(item.get("relation_id", "")) for item in audit["unconfirmed_relations"])
    deleted = 0
    if not dry_run and delete_ids:
        if hasattr(store, "_run"):
            rows = store._run(
                "MATCH ()-[r:TRG_RELATION]->() WHERE r.relation_id IN $relation_ids DELETE r RETURN count(r) AS deleted",
                relation_ids=sorted(delete_ids),
            )
            deleted = int(rows[0].get("deleted", 0)) if rows else 0
        elif hasattr(store, "relations"):
            for relation_id in list(store.relations):
                if relation_id in delete_ids:
                    store.relations.pop(relation_id, None)
                    deleted += 1
    after = audit_graph(store) if not dry_run else before
    after_confirmed = audit_confirmed_relations(store) if not dry_run else audit
    return {
        "backup": str(backup),
        "dry_run": dry_run,
        "before": before,
        "after": after,
        "confirmed_audit_before_cleanup": audit,
        "confirmed_audit_after_cleanup": after_confirmed,
        "proposed_before": sum(1 for item in relations if str(item.get("confidence_status")) == "proposed"),
        "unconfirmed_before": audit["unconfirmed_count"],
        "deleted_relations": deleted,
        "deleted_unconfirmed": audit["unconfirmed_count"],
        "deleted_invalid_confirmed": len(audit["invalid_confirmed_relations"]),
        "deleted_duplicates": len(audit["duplicates"]),
        "retained_filters": after_confirmed["retained_relation_counts"].get("filters", 0),
        "retained_prepares": after_confirmed["retained_relation_counts"].get("prepares", 0),
        "retained_relation_ids": after_confirmed["retained_relation_ids"],
    }


def write_backup(directory: Path, nodes: list[dict[str, Any]], relations: list[dict[str, Any]], audit: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trg_backup.json"
    payload = {
        "schema_version": "trg_backup.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
        "nodes": nodes,
        "relations": relations,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _parameter_definitions(api_name: str, rules_by_target: dict[str, list[dict[str, Any]]], sdk: Any) -> dict[str, Any]:
    prototype = named_prototype(api_name) or str(getattr(sdk, "functions", {}).get(api_name, ""))
    args_text = prototype.partition("(")[2].rsplit(")", 1)[0] if "(" in prototype else ""
    docs: dict[str, str] = {}
    # Direction is taken from the indexed API arguments, never inferred from
    # a parameter name or pointer spelling alone.
    try:
        from .focas_contracts import _api_docs  # local, evidence-backed index
        arguments = str(_api_docs().get(api_name, {}).get("arguments", ""))
        import re
        for match in re.finditer(r"(?m)^-\s+([A-Za-z_]\w*)\s+\((in\s*/\s*out|in|out)\):", arguments):
            direction = match.group(2).replace(" ", "")
            docs[match.group(1)] = "inout" if direction == "in/out" else direction
    except Exception:
        pass
    output: dict[str, Any] = {}
    for raw in args_text.split(",") if args_text else []:
        raw = " ".join(raw.split())
        if not raw or raw == "void":
            continue
        import re
        match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", raw)
        if not match:
            continue
        name = match.group(1)
        c_type = raw[:match.start(1)].strip()
        direction = docs.get(name, "in")
        source_rules = rules_by_target.get(api_name, [])
        contract_rules = [
            item for chain in PROGRAM_OPERATION_CHAIN_RULES.values()
            for item in chain.get("parameter_rules", ())
            if str(item.get("api", "")) == api_name or str(item.get("source_api", "")) == api_name
        ]
        runtime = any(name in (rule.get("target_parameters") or []) for rule in source_rules)
        valid_values: Any = None
        constraints: list[str] = []
        for rule in source_rules:
            if name not in (rule.get("target_parameters") or []):
                continue
            constraints.extend(str(item) for item in rule.get("verification_requirements", ()) if str(item))
            # Existing contract rules may provide a documented enumeration.
        for contract in contract_rules:
            contract_parameter = str(contract.get("parameter", ""))
            # Program number is encoded in the NC payload for the v3/v4
            # download data call; it is not a formal C argument named
            # ``program_number``.
            if contract_parameter == "program_number" and name == "data" and api_name in {"cnc_download3", "cnc_download4"}:
                contract_parameter = name
            if contract_parameter == name and contract.get("allowed_values") is not None:
                valid_values = list(contract["allowed_values"])
                constraints.extend(str(item) for item in contract.get("constraints", ()) if str(item))
            elif contract_parameter == name:
                constraints.extend(str(item) for item in contract.get("constraints", ()) if str(item))
        if direction in {"out", "inout"}:
            value_source = "runtime"
        elif runtime:
            value_source = "runtime"
        else:
            value_source = "generated"
        if isinstance(valid_values, list) and len(valid_values) == 1:
            strategy = "fixed"
        elif isinstance(valid_values, list):
            strategy = "enumeration"
        elif runtime:
            strategy = "runtime_filter"
        else:
            strategy = "range"
        item: dict[str, Any] = {
            "type": c_type or "unknown",
            "direction": direction,
            "valid_values": valid_values,
            "constraints": list(dict.fromkeys(constraints)),
            "value_source": value_source,
            "variation_strategy": strategy,
        }
        type_name = c_type.replace("const", "").replace("*", " ").split()[-1].upper() if c_type else ""
        struct = getattr(sdk, "structs", {}).get(type_name) or getattr(sdk, "unions", {}).get(type_name)
        if struct and isinstance(struct.get("fields"), dict):
            item["fields"] = struct["fields"]
        output[name] = item
    return output


def _node_knowledge(nodes: list[dict[str, Any]], relations: list[dict[str, Any]], *, rules_by_target: dict[str, list[dict[str, Any]]], prereq_by_target: dict[str, list[dict[str, Any]]], sdk: Any) -> dict[str, dict[str, Any]]:
    session_by_api: dict[str, dict[str, Any]] = {}
    for target, session in API_TRAFFIC_SESSION_RULES.items():
        for api in (target, session.get("start_api"), session.get("segment_api"), session.get("end_api")):
            if api:
                session_by_api[str(api)] = {
                    "session_type": str(session.get("session_type", "")),
                    "start_api": str(session.get("start_api", "")),
                    "end_api": str(session.get("end_api", "")),
                    "required_state": "active",
                }
    program_by_api: dict[str, dict[str, Any]] = {}
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        calls = tuple(str(item) for item in chain.get("session_calls", ()) if str(item))
        for api in calls:
            program_by_api[api] = {
                "requirement": "exists" if "upload" in str(chain.get("chain_id", "")) else "not_exists",
                "program_number": None,
            }
    result: dict[str, dict[str, Any]] = {}
    for node in nodes:
        name = str(node.get("name", ""))
        params = _parameter_definitions(name, rules_by_target, sdk)
        prereq = {
            "cnc_parameters": [dict(item) for item in prereq_by_target.get(name, [])],
            "pmc_signals": [],
            "cnc_mode": [],
            "cnc_state": [],
            "nc_program": [program_by_api[name]] if name in program_by_api else [],
            "session": [session_by_api[name]] if name in session_by_api else [],
            "options": [],
            "knowledge_status": "known" if (name in session_by_api or name in program_by_api or prereq_by_target.get(name)) else "unknown",
        }
        result[name] = {"parameters": params, "prerequisites": prereq}
    return result


def _output_paths(node_knowledge: dict[str, Any], field: str) -> list[str]:
    paths: list[str] = []
    for name, definition in node_knowledge.get("parameters", {}).items():
        if definition.get("direction") not in {"out", "inout"}:
            continue
        if name == field:
            paths.append(name)
        fields = definition.get("fields", {})
        if isinstance(fields, dict) and field in fields:
            paths.append(f"{name}.{field}")
    return paths


def _classify_relations(relations: list[dict[str, Any]], node_knowledge: dict[str, dict[str, Any]], all_names: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rules = resolve_rules(all_names, allowed_api_names=all_names, device_scope="")
    parameter_index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in rules.get("relations", []):
        parameter_index[(str(item.get("source_api")), str(item.get("target_api")))].append(item)
    session_index: dict[tuple[str, str], dict[str, Any]] = {}
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        calls = tuple(str(item) for item in chain.get("session_calls", ()) if str(item))
        for source, target in zip(calls, calls[1:]):
            session_index[(source, target)] = chain
    for target, session in API_TRAFFIC_SESSION_RULES.items():
        for source, destination in ((session.get("start_api"), target), (target, session.get("end_api"))):
            if source and destination:
                session_index[(str(source), str(destination))] = session
    prereq_report = extract_f0id_parameter_prerequisites(all_names, allowed_api_names=all_names, device_scope=F0ID_SCOPE_ID)
    prereq_index: set[tuple[str, str]] = {(str(item.get("source_api")), str(item.get("target_api"))) for item in prereq_report.get("relations", [])}
    classified: list[dict[str, Any]] = []
    counts = Counter()
    for item in relations:
        source, target = str(item.get("source_api", "")), str(item.get("target_api", ""))
        old = str(item.get("relation_type", ""))
        key = (source, target)
        status = str(item.get("confidence_status", "proposed"))
        evidence = list(item.get("evidence_ids") or [])
        attrs: dict[str, Any] = {}
        semantic = ("filters" if old in {"PROVIDES_INPUT", "READS_AFTER_WRITE"} else "prepares" if old in LEGACY_RELATION_TYPE_MAP else str(old).lower())
        valid = False
        reason = "no matching evidence-backed rule"
        if key in parameter_index and old in {"PROVIDES_INPUT", "filters"}:
            for rule in parameter_index[key]:
                target_params = [str(value) for value in rule.get("target_parameters", ())]
                source_fields = [str(value) for value in rule.get("source_fields", ())]
                target_defs = node_knowledge.get(target, {}).get("parameters", {})
                source_defs = node_knowledge.get(source, {})
                if not target_params or not source_fields or any(param not in target_defs for param in target_params):
                    continue
                mapped = []
                for field in source_fields:
                    mapped.extend(_output_paths(source_defs, field))
                if not mapped:
                    continue
                semantic = "filters"
                attrs = {
                    "dependency_path": target_params[0],
                    "filter_mode": "exclude" if any(token in str(rule.get("filter_strategy", "")) for token in ("unused", "exclude", "not_current")) else "include",
                    "source_field": mapped[0],
                    "transform": str(rule.get("transform", "") or ""),
                }
                semantic_error = _filter_semantic_error(source, target, attrs, node_knowledge)
                valid = semantic_error is None
                status = str(item.get("confidence_status")) if str(item.get("confidence_status")) in VALID_STATUSES else str(rule.get("confidence_status", "sdk_validated"))
                evidence = list(dict.fromkeys(evidence + list(rule.get("evidence_ids", ()))))
                reason = "matched parameter discovery rule and SDK/RAG fields" if valid else semantic_error or "filter semantic mismatch"
                break
        if not valid and key in session_index and old in {"REQUIRES", "PRECEDES", "SESSION_SEQUENCE", "prepares"}:
            semantic = "prepares"
            attrs = {"dependency_path": "prerequisites.session"}
            chain = session_index[key]
            valid = bool(node_knowledge.get(target, {}).get("prerequisites", {}).get("session"))
            semantic_error = _session_source_error(source, target, chain)
            if semantic_error:
                valid = False
                reason = semantic_error
            else:
                reason = "matched documented Start/Data/End session contract" if valid else "session target has no populated prerequisite"
            evidence = list(dict.fromkeys(evidence + list(chain.get("evidence_ids", ()))))
            if valid:
                status = status if status in VALID_STATUSES else "documented"
        if not valid and key in prereq_index and old in {"REQUIRES", "prepares"}:
            semantic = "prepares"
            if node_knowledge.get(target, {}).get("prerequisites", {}).get("cnc_parameters"):
                attrs = {"dependency_path": "prerequisites.cnc_parameters"}
                semantic_error = _prepares_semantic_error(source, target, "prerequisites.cnc_parameters", node_knowledge)
                valid = semantic_error is None
                status = status if status in VALID_STATUSES else "sdk_validated"
                reason = "matched CNC parameter prerequisite rule" if valid else semantic_error or "prepares semantic mismatch"
        if not valid:
            semantic = "filters" if semantic == "filters" else "prepares"
            status = "proposed"
            attrs = {}
            counts["proposed"] += 1
        else:
            counts[semantic] += 1
        # Canonical edge attributes are deliberately narrow.  The old
        # attributes_json is preserved in the backup/report, while Planner
        # sees only the validated dependency path fields requested by TRCM.
        attrs_json = dict(attrs)
        classified.append({**item, "relation_type": semantic, "confidence_status": status, "evidence_ids": evidence, "attributes": attrs_json})
    # Keep one active record for each semantic fact.  Historical duplicates
    # remain in the graph as proposed candidates and therefore cannot be
    # scheduled by Planner.
    semantic_groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in classified:
        attrs = item.get("attributes", {})
        semantic_groups[(
            str(item.get("source_api", "")), str(item.get("target_api", "")),
            str(item.get("relation_type", "")), str(attrs.get("dependency_path", "")),
            str(attrs.get("source_field", "")), str(attrs.get("filter_mode", "")),
        )].append(item)
    for group in semantic_groups.values():
        active = [item for item in group if item.get("confidence_status") in VALID_STATUSES]
        for duplicate in active[1:]:
            duplicate["confidence_status"] = "proposed"
            duplicate["attributes"] = {}
            counts["proposed"] = counts.get("proposed", 0) + 1
    return classified, dict(counts)


def _materialize_verified_rules(
    classified: list[dict[str, Any]],
    node_knowledge: dict[str, dict[str, Any]],
    all_names: set[str],
) -> list[dict[str, Any]]:
    """Create missing canonical records from explicit evidence-backed rules.

    This is additive and deterministic.  It does not promote an ambiguous
    historical edge; it only fills a missing canonical fact whose SDK/RAG
    rule has a resolvable source field and target parameter.
    """
    resolved = resolve_rules(all_names, allowed_api_names=all_names, device_scope="")
    existing: set[tuple[str, str, str, str, str, str]] = set()
    for item in classified:
        if item.get("confidence_status") not in VALID_STATUSES:
            continue
        attrs = item.get("attributes", {})
        existing.add((
            str(item.get("source_api", "")), str(item.get("target_api", "")),
            str(item.get("relation_type", "")), str(attrs.get("dependency_path", "")),
            str(attrs.get("source_field", "")), str(attrs.get("filter_mode", "")),
        ))
    additions: list[dict[str, Any]] = []
    for rule in resolved.get("relations", []):
        if str(rule.get("relation_type", "")).upper() != "PROVIDES_INPUT":
            continue
        source, target = str(rule.get("source_api", "")), str(rule.get("target_api", ""))
        source_defs = node_knowledge.get(source, {})
        target_defs = node_knowledge.get(target, {}).get("parameters", {})
        fields: list[str] = []
        for field in rule.get("source_fields", ()):
            fields.extend(_output_paths(source_defs, str(field)))
        if not fields:
            continue
        for parameter in (str(item) for item in rule.get("target_parameters", ())):
            if parameter not in target_defs:
                continue
            filter_mode = "exclude" if any(token in str(rule.get("filter_strategy", "")) for token in ("unused", "exclude", "not_current")) else "include"
            attrs = {
                "dependency_path": parameter,
                "filter_mode": filter_mode,
                "source_field": fields[0],
                "transform": str(rule.get("transform", "") or ""),
            }
            semantic_error = _filter_semantic_error(source, target, attrs, node_knowledge)
            if semantic_error:
                continue
            key = (source, target, "filters", parameter, fields[0], filter_mode)
            if key in existing:
                continue
            rid = "rel:trg_audit_filters:" + ":".join((source, target, str(rule.get("rule_id", "")), parameter))
            additions.append({
                "relation_id": rid, "source_api": source, "target_api": target,
                "relation_type": "filters", "rule_id": str(rule.get("rule_id", "")),
                "confidence_status": "sdk_validated", "evidence_ids": list(rule.get("evidence_ids", ()) or ()),
                "attributes": attrs, "applicability": dict(rule.get("applicability", {}) or {}), "version": 1,
            })
            existing.add(key)
    # Program transfer contracts also describe v4 and upload parameters that
    # are not repeated in the generic parameter-rule table.
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        discovery = chain.get("directory_discovery", {})
        source = str(discovery.get("api", "")).strip()
        if not source:
            continue
        for contract in chain.get("parameter_rules", ()):
            # Constants such as Start.type and local buffer lengths are not
            # supplied by a program-directory read. Only an explicit
            # source_api establishes a directory-to-target mapping.
            if not str(contract.get("source_api", "")).strip():
                continue
            target = str(contract.get("api", "")).strip()
            parameter = str(contract.get("parameter", "")).strip()
            if not target or not parameter or target not in all_names or source not in all_names:
                continue
            if parameter == "directory_type":
                continue
            if target in {"cnc_download3", "cnc_download4"} and parameter == "program_number":
                parameter = "data"
            target_defs = node_knowledge.get(target, {}).get("parameters", {})
            source_fields = _output_paths(node_knowledge.get(source, {}), "number")
            if parameter not in target_defs or not source_fields:
                continue
            # Directory output supplies an occupied-program set. For a
            # download data buffer the selected free number must be rendered
            # into the NC payload; excluding a scalar number from raw bytes is
            # not a valid binding.
            if target in {"cnc_download3", "cnc_download4"} and parameter == "data":
                filter_mode = "include"
                transform = DOWNLOAD_PROGRAM_TRANSFORM
            else:
                filter_mode = "exclude" if "unused" in str(contract.get("filter_strategy", "")) else "include"
                transform = str(contract.get("transform", "") or "")
            key = (source, target, "filters", parameter, source_fields[0], filter_mode)
            if key in existing:
                continue
            additions.append({
                "relation_id": "rel:trg_audit_contract_filters:" + ":".join((source, target, parameter, str(chain.get("chain_id", "")))),
                "source_api": source, "target_api": target, "relation_type": "filters",
                "rule_id": "program_contract:" + str(chain.get("chain_id", "")),
                "confidence_status": "documented", "evidence_ids": list(chain.get("evidence_ids", ()) or ()),
                "attributes": {"dependency_path": parameter, "filter_mode": filter_mode, "source_field": source_fields[0], "transform": transform},
                "applicability": {"required": False, "required_if_missing_inputs": ["program_number"], "runtime_selection": "must_execute_when_no_verified_program_number"} if target in {"cnc_download3", "cnc_download4"} and parameter == "data" else {}, "version": 1,
            })
            existing.add(key)
    # Session contracts are the only source allowed to create a prepares edge
    # without an old row.  Directory discovery is intentionally absent here.
    for chain in PROGRAM_OPERATION_CHAIN_RULES.values():
        calls = tuple(str(item) for item in chain.get("session_calls", ()) if str(item))
        for source, target in zip(calls, calls[1:]):
            key = (source, target, "prepares", "prerequisites.session", "", "")
            if key in existing or not node_knowledge.get(target, {}).get("prerequisites", {}).get("session"):
                continue
            additions.append({
                "relation_id": "rel:trg_audit_prepares:" + ":".join((source, target, str(chain.get("chain_id", "")))),
                "source_api": source, "target_api": target, "relation_type": "prepares",
                "rule_id": "session:" + str(chain.get("chain_id", "")), "confidence_status": "documented",
                "evidence_ids": list(chain.get("evidence_ids", ()) or ()),
                "attributes": {"dependency_path": "prerequisites.session"}, "applicability": {}, "version": 1,
            })
            existing.add(key)
    return additions


def migrate_graph(store: Any, *, backup_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    """Compatibility entry point for the strict confirmed-only migration."""
    return cleanup_confirmed_graph(store, backup_dir=backup_dir, dry_run=dry_run)
