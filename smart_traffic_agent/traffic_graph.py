"""Traffic Relationship Graph (TRG) storage and evidence-aware queries.

The production store is Neo4j. An explicit in-memory store is provided for
offline planning tests and migration previews; it is never treated as device
evidence unless the caller supplies that evidence.
"""
from __future__ import annotations

import os
import re
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


# ``filters`` and ``prepares`` are the only relationship semantics exposed by
# the current TRCM model.  The uppercase names are accepted at the boundary so
# old JSON plans and old Neo4j rows can be read during migration, but all new
# structured relations should use the two canonical names.
CANONICAL_RELATION_TYPES = {"filters", "prepares"}
RELATION_TYPES = CANONICAL_RELATION_TYPES
LEGACY_RELATION_TYPES = {"REQUIRES", "PRECEDES", "PROVIDES_INPUT", "READS_AFTER_WRITE", "SESSION_SEQUENCE"}
LEGACY_RELATION_TYPE_MAP = {
    "PROVIDES_INPUT": "filters",
    "READS_AFTER_WRITE": "filters",
    "REQUIRES": "prepares",
    "PRECEDES": "prepares",
    "SESSION_SEQUENCE": "prepares",
}
RELATION_STATUSES = {"proposed", "unconfirmed", "documented", "sdk_validated", "runtime_verified", "rejected", "superseded"}
CONFIRMED_RELATION_STATUSES = {"documented", "sdk_validated", "runtime_verified"}


def _relation_write_rejection(relation: "TrafficRelation", reason: str) -> dict[str, Any]:
    """Return a stable non-persistence result for an unconfirmed relation."""
    return {
        "ok": False,
        "stored": False,
        "status": "unconfirmed",
        "relation_id": relation.relation_id,
        "source_api": relation.source_api,
        "target_api": relation.target_api,
        "relation_type": canonical_relation_type(relation.relation_type),
        "reason": reason,
    }


def _confirmed_relation_error(relation: "TrafficRelation") -> str | None:
    """Validate the minimum evidence contract before a relation is persisted."""
    if relation.confidence_status not in CONFIRMED_RELATION_STATUSES:
        return "only documented, sdk_validated, or runtime_verified relations may be persisted"
    if canonical_relation_type(relation.relation_type) not in CANONICAL_RELATION_TYPES:
        return "only filters or prepares relations may be persisted"
    if not relation.evidence_ids:
        return "confirmed relations require at least one evidence_id"
    attrs = relation.attributes or {}
    if relation.relation_type == "filters":
        missing = [key for key in ("dependency_path", "source_field", "filter_mode") if not str(attrs.get(key, "")).strip()]
        if missing:
            return "filters relation is missing: " + ", ".join(missing)
    elif relation.relation_type == "prepares" and not str(attrs.get("dependency_path", "")).strip():
        return "prepares relation requires dependency_path"
    return None


class CanonicalRelationType(str):
    """Canonical string that remains comparable to a legacy alias in Python.

    Persisted JSON and Neo4j values are the lowercase canonical value.  The
    equality shim keeps existing plugins/tests that compare an in-memory
    relation to ``SESSION_SEQUENCE`` or ``PROVIDES_INPUT`` working while the
    migration is rolled out.
    """

    def __new__(cls, value: str, aliases: Iterable[str] = ()):
        obj = str.__new__(cls, value)
        obj._aliases = frozenset(str(alias) for alias in aliases)
        return obj

    def __eq__(self, other: object) -> bool:
        return str.__eq__(self, other) or str(other) in getattr(self, "_aliases", frozenset())

    def __hash__(self) -> int:
        return str.__hash__(self)


def canonical_relation_type(value: str) -> str:
    """Return the two-type TRCM vocabulary while accepting legacy aliases."""
    raw = str(value or "").strip()
    if raw in CANONICAL_RELATION_TYPES:
        aliases = (
            ("PROVIDES_INPUT", "READS_AFTER_WRITE") if raw == "filters"
            else ("REQUIRES", "PRECEDES", "SESSION_SEQUENCE")
        )
        return CanonicalRelationType(raw, aliases)
    upper = raw.upper()
    if upper in LEGACY_RELATION_TYPE_MAP:
        canonical = LEGACY_RELATION_TYPE_MAP[upper]
        return CanonicalRelationType(canonical, (upper,))
    return raw.lower()


def _scope_allowed_names(device_scope: str, allowed_api_names: Iterable[str] | None) -> set[str] | None:
    """Resolve a device scope to an explicit endpoint allow-list."""
    if allowed_api_names is None and device_scope == "fanuc:0i-D":
        try:
            from .api_compatibility import f0id_api_names
            allowed_api_names = f0id_api_names()
        except Exception:
            # A missing compatibility index must make the scoped query empty,
            # never silently broaden it to the 704-function global graph.
            return set()
    if allowed_api_names is None:
        return None
    return {str(item).removeprefix("focas:") for item in allowed_api_names if str(item).strip()}


class TrafficGraphError(RuntimeError):
    pass


class TrafficGraphUnavailable(TrafficGraphError):
    pass


def _decode_relation_properties(relation: dict[str, Any] | None) -> dict[str, Any]:
    """Expose JSON-backed Neo4j maps using the same shape as the memory store."""
    item = dict(relation or {})
    for encoded, decoded in (("applicability_json", "applicability"), ("attributes_json", "attributes")):
        raw = item.get(encoded)
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                value = {}
            if isinstance(value, dict):
                item[decoded] = value
    return item


def _decode_node_properties(node: dict[str, Any] | None) -> dict[str, Any]:
    item = dict(node or {})
    for encoded, decoded in (("metadata_json", "metadata"), ("parameters_json", "parameters"), ("prerequisites_json", "prerequisites")):
        raw = item.get(encoded)
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                value = {}
            if isinstance(value, dict):
                item[decoded] = value
    return item


def _relation_dict(relation: TrafficRelation) -> dict[str, Any]:
    value = asdict(relation)
    value["relation_type"] = canonical_relation_type(str(value.get("relation_type", "")))
    return value


def _parameter_rules_from_relations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten confirmed filters metadata for deterministic parameter tools."""
    output: list[dict[str, Any]] = []
    for row in rows:
        relation = _decode_relation_properties(row.get("relation") if isinstance(row, dict) else None)
        source_api = str(row.get("source_api", "")) if isinstance(row, dict) else ""
        attributes = relation.get("attributes", {})
        raw_rules = attributes.get("parameter_rules", []) if isinstance(attributes, dict) else []
        if isinstance(attributes, dict) and (
            attributes.get("dependency_path") or attributes.get("source_field")
        ):
            raw_rules = [attributes, *raw_rules] if isinstance(raw_rules, list) else [attributes]
        for raw in raw_rules if isinstance(raw_rules, list) else []:
            if not isinstance(raw, dict):
                continue
            rule = dict(raw)
            rule.setdefault("source_api", source_api)
            # A relation may omit the optional rule_id while still having a
            # stable relation_id.  Keep a deterministic identifier for the
            # runtime binding evidence instead of emitting an empty id.
            rule.setdefault("rule_id", relation.get("rule_id") or relation.get("relation_id", ""))
            rule.setdefault("confidence_status", relation.get("confidence_status", ""))
            rule.setdefault("evidence_ids", relation.get("evidence_ids", []))
            rule.setdefault("applicability", relation.get("applicability", {}))
            if "dependency_path" in rule and "parameter" not in rule:
                rule["parameter"] = str(rule.get("dependency_path", "")).rsplit(".", 1)[-1]
            if "source_field" in rule and "source_fields" not in rule:
                rule["source_fields"] = [rule.get("source_field")]
            rule.setdefault("filter_mode", "include")
            output.append(rule)
    return output


def _dedupe_relation_objects(relations: Iterable[TrafficRelation], device_scope: str = "") -> list[TrafficRelation]:
    """Collapse generic and scoped copies while preferring exact scope facts."""
    selected: dict[tuple[str, str, str, str], TrafficRelation] = {}
    for relation in relations:
        key = (
            relation.source_api.removeprefix("focas:"),
            relation.target_api.removeprefix("focas:"),
            relation.relation_type,
            relation.rule_id,
        )
        current = selected.get(key)
        current_scope = str(current.applicability.get("device_scope", "")) if current else ""
        relation_scope = str(relation.applicability.get("device_scope", ""))
        if current is None or (device_scope and relation_scope == device_scope and current_scope != device_scope):
            selected[key] = relation
    return list(selected.values())


def _dedupe_relation_rows(rows: Iterable[dict[str, Any]], device_scope: str = "") -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        relation = _decode_relation_properties(row.get("relation") if isinstance(row, dict) else None)
        key = (
            str(row.get("source_api") or row.get("source", {}).get("name", "") or relation.get("source_api", "")),
            str(row.get("target_api") or row.get("target", {}).get("name", "") or relation.get("target_api", "")),
            str(relation.get("relation_type", "")),
            str(relation.get("rule_id", "")),
        )
        current = selected.get(key)
        current_scope = str((current or {}).get("relation", {}).get("applicability", {}).get("device_scope", ""))
        relation_scope = str(relation.get("applicability", {}).get("device_scope", ""))
        if current is None or (device_scope and relation_scope == device_scope and current_scope != device_scope):
            selected[key] = row
    return list(selected.values())


def _dedupe_graph_paths(rows: Iterable[dict[str, Any]], device_scope: str = "") -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        nodes = tuple(str(item.get("name", "")) for item in row.get("nodes", []) if isinstance(item, dict))
        relations = row.get("relations", [])
        relation_key = tuple(
            (str(item.get("relation_type", "")), str(item.get("rule_id", "")))
            for item in relations if isinstance(item, dict)
        )
        key = (nodes, relation_key)
        current = selected.get(key)
        current_score = sum(
            str(item.get("applicability", {}).get("device_scope", "")) == device_scope
            for item in (current or {}).get("relations", []) if isinstance(item, dict)
        )
        row_score = sum(
            str(item.get("applicability", {}).get("device_scope", "")) == device_scope
            for item in relations if isinstance(item, dict)
        )
        if current is None or row_score > current_score:
            selected[key] = row
    return list(selected.values())


@dataclass(frozen=True, slots=True)
class ApiNode:
    api_id: str
    name: str
    protocol: str = "FOCAS"
    sdk_signature_ref: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Parameters and prerequisites live on the API node.  They are JSON maps
    # in Neo4j because nested property maps are not supported by all Neo4j
    # Community versions used by the desktop installer.
    parameters: dict[str, Any] = field(default_factory=dict)
    prerequisites: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrafficRelation:
    relation_id: str
    source_api: str
    target_api: str
    relation_type: str
    rule_id: str = ""
    confidence_status: str = "proposed"
    evidence_ids: tuple[str, ...] = ()
    applicability: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def normalized(self) -> "TrafficRelation":
        relation_type = canonical_relation_type(self.relation_type)
        status = str(self.confidence_status).lower()
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"Unsupported traffic relation type: {relation_type}")
        if status not in RELATION_STATUSES:
            raise ValueError(f"Unsupported traffic relation status: {status}")
        if not self.source_api or not self.target_api:
            raise ValueError("Traffic relations require source_api and target_api")
        return TrafficRelation(
            relation_id=str(self.relation_id),
            source_api=str(self.source_api),
            target_api=str(self.target_api),
            relation_type=relation_type,
            rule_id=str(self.rule_id),
            confidence_status=status,
            evidence_ids=tuple(str(item) for item in self.evidence_ids),
            applicability=dict(self.applicability),
            attributes=dict(self.attributes),
            version=max(1, int(self.version)),
        )


class TrafficGraphStore(Protocol):
    def upsert_api(self, node: ApiNode) -> dict[str, Any]: ...
    def upsert_relation(self, relation: TrafficRelation) -> dict[str, Any]: ...
    def get_api_info(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_traffic_chain(self, api_name: str, max_depth: int = 3, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_api_dependencies(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_parameter_rules(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_parameter_prerequisites(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_dependency_plan(self, api_name: str, max_depth: int = 8, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None, satisfied: Iterable[str] | None = None) -> dict[str, Any]: ...
    def get_graph_overview(self, *, query: str = "", device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]: ...
    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]: ...
    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]: ...
    def record_relation_failure(self, relation_id: str, evidence: list[dict[str, Any]], *, reason: str = "", replacement_relation_id: str = "") -> dict[str, Any]: ...
    def migrate_legacy_relations(self) -> dict[str, Any]: ...


class InMemoryTrafficGraphStore:
    """Deterministic store used by offline simulations and unit tests."""

    def __init__(self) -> None:
        self.nodes: dict[str, ApiNode] = {}
        self.relations: dict[str, TrafficRelation] = {}

    def upsert_api(self, node: ApiNode) -> dict[str, Any]:
        previous = self.nodes.get(node.name) or self.nodes.get(node.api_id)
        if previous is not None:
            node = ApiNode(
                api_id=node.api_id or previous.api_id,
                name=node.name or previous.name,
                protocol=node.protocol or previous.protocol,
                sdk_signature_ref=node.sdk_signature_ref or previous.sdk_signature_ref,
                description=node.description or previous.description,
                metadata={**previous.metadata, **node.metadata},
                parameters={**previous.parameters, **node.parameters},
                prerequisites={**previous.prerequisites, **node.prerequisites},
            )
        self.nodes[node.name] = node
        self.nodes[node.api_id] = node
        return {"ok": True, "api": asdict(node), "created": previous is None}

    def upsert_relation(self, relation: TrafficRelation) -> dict[str, Any]:
        item = relation.normalized()
        rejection = _confirmed_relation_error(item)
        if rejection:
            return _relation_write_rejection(item, rejection)
        previous = self.relations.get(item.relation_id)
        if previous is not None and item.version < previous.version:
            raise ValueError(f"Relation version regressed: {item.relation_id}")
        self.relations[item.relation_id] = item
        return {"ok": True, "relation": _relation_dict(item), "created": previous is None}

    @staticmethod
    def _scope_names(allowed_api_names: Iterable[str] | None) -> set[str] | None:
        if allowed_api_names is None:
            return None
        return {str(item).removeprefix("focas:") for item in allowed_api_names if str(item).strip()}

    def _relation_in_scope(self, relation: TrafficRelation, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> bool:
        names = _scope_allowed_names(device_scope, allowed_api_names)
        source = relation.source_api.removeprefix("focas:")
        target = relation.target_api.removeprefix("focas:")
        if names is not None and (source not in names or target not in names):
            return False
        relation_scope = str(relation.applicability.get("device_scope", ""))
        return not device_scope or not relation_scope or relation_scope == device_scope

    def get_api_info(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        node = self.nodes.get(api_name) or self.nodes.get(f"focas:{api_name}")
        names = _scope_allowed_names(device_scope, allowed_api_names)
        if names is not None and api_name.removeprefix("focas:") not in names:
            node = None
        return {"available": node is not None, "api": asdict(node) if node else None, "backend": "memory"}

    def _relations_for(self, api_name: str) -> list[TrafficRelation]:
        name = api_name.removeprefix("focas:")
        return [
            row for row in self.relations.values()
            if row.source_api.removeprefix("focas:") == name
            or row.target_api.removeprefix("focas:") == name
        ]

    def get_traffic_chain(self, api_name: str, max_depth: int = 3, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        scope_names = _scope_allowed_names(device_scope, allowed_api_names)
        api_known = name in self.nodes and (scope_names is None or name in scope_names)
        if scope_names is not None and name not in scope_names:
            return {
                "available": True, "api_name": name, "device_scope": device_scope,
                "scope_api_count": len(scope_names), "knowledge_status": "out_of_scope",
                "nodes": [], "relations": [], "backend": "memory",
            }
        frontier = {name}
        visited = {name}
        selected: dict[str, TrafficRelation] = {}
        for _ in range(max(1, int(max_depth))):
            next_frontier: set[str] = set()
            for relation in self.relations.values():
                if not self._relation_in_scope(relation, device_scope, allowed_api_names):
                    continue
                if relation.confidence_status not in CONFIRMED_RELATION_STATUSES:
                    continue
                source = relation.source_api.removeprefix("focas:")
                target = relation.target_api.removeprefix("focas:")
                if source in frontier or target in frontier:
                    selected[relation.relation_id] = relation
                    next_frontier.update({source, target} - visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        deduped = _dedupe_relation_objects(selected.values(), device_scope)
        return {
            "available": True,
            "api_name": name,
            "device_scope": device_scope,
            "scope_api_count": len(scope_names or ()),
            "knowledge_status": "known" if api_known else "unknown",
            "nodes": [asdict(self.nodes[node]) for node in sorted(visited) if node in self.nodes],
            "relations": [_relation_dict(item) for item in deduped],
            "backend": "memory",
        }

    def get_api_dependencies(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        rows = [
            {
                "source": asdict(self.nodes.get(relation.source_api) or self.nodes.get(f"focas:{relation.source_api}", ApiNode("", relation.source_api))),
                "relation": _relation_dict(relation),
                "target": asdict(self.nodes.get(relation.target_api) or self.nodes.get(f"focas:{relation.target_api}", ApiNode("", relation.target_api))),
            }
            for relation in self.relations.values()
            if relation.target_api.removeprefix("focas:") == name
            and self._relation_in_scope(relation, device_scope, allowed_api_names)
            and canonical_relation_type(relation.relation_type) in CANONICAL_RELATION_TYPES
            and relation.confidence_status in CONFIRMED_RELATION_STATUSES
        ]
        rows = _dedupe_relation_rows(rows, device_scope)
        return {"available": True, "api_name": name, "device_scope": device_scope, "knowledge_status": "known" if api_known else "unknown", "dependencies": rows, "backend": "memory"}

    def get_parameter_rules(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        rows = [
            {"relation": _relation_dict(relation), "source_api": relation.source_api}
            for relation in self.relations.values()
            if relation.target_api.removeprefix("focas:") == name
            and self._relation_in_scope(relation, device_scope, allowed_api_names)
            and canonical_relation_type(relation.relation_type) == "filters"
            and relation.confidence_status in CONFIRMED_RELATION_STATUSES
        ]
        rows = _dedupe_relation_rows(rows, device_scope)
        rules = _parameter_rules_from_relations(rows)
        return {"available": True, "api_name": name, "device_scope": device_scope, "knowledge_status": "known" if api_known and rules else "unknown", "rules": rules, "backend": "memory"}

    def get_parameter_prerequisites(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        rows: list[dict[str, Any]] = []
        for relation in self.relations.values():
            if relation.target_api.removeprefix("focas:") != name:
                continue
            if canonical_relation_type(relation.relation_type) != "prepares" or relation.confidence_status not in CONFIRMED_RELATION_STATUSES or not self._relation_in_scope(relation, device_scope, allowed_api_names):
                continue
            attributes = dict(relation.attributes)
            if "parameter_number" not in attributes:
                continue
            rows.append({
                "source_api": relation.source_api,
                "target_api": relation.target_api,
                "relation": _relation_dict(relation),
                "parameter": attributes,
            })
        return {
            "available": True,
            "api_name": name,
            "device_scope": device_scope,
            "knowledge_status": "known" if api_known and rows else "unknown",
            "prerequisites": rows,
            "backend": "memory",
        }

    def get_dependency_plan(self, api_name: str, max_depth: int = 8, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None, satisfied: Iterable[str] | None = None) -> dict[str, Any]:
        """Recursively collect incoming filters/prepares edges in execution order.

        The graph stores only API nodes.  ``satisfied`` contains API names whose
        output/condition is already present in the current ExecutionContext;
        those nodes remain visible in ``satisfied_nodes`` but are omitted from
        ``required_calls``.
        """
        target = str(api_name).removeprefix("focas:")
        scope_names = _scope_allowed_names(device_scope, allowed_api_names)
        if scope_names is not None and target not in scope_names:
            return {"available": True, "backend": "memory", "api_name": target, "knowledge_status": "out_of_scope", "required_calls": [], "edges": [], "satisfied_nodes": []}
        already = {str(item).removeprefix("focas:") for item in (satisfied or ())}
        visited: set[str] = set()
        ordered: list[str] = []
        optional_ordered: list[str] = []
        edges: list[dict[str, Any]] = []
        cycles: list[list[str]] = []

        def walk(node: str, path: tuple[str, ...], depth: int) -> None:
            if depth > max(1, min(int(max_depth), 16)):
                return
            incoming = [r for r in self.relations.values()
                        if r.target_api.removeprefix("focas:") == node
                        and self._relation_in_scope(r, device_scope, allowed_api_names)
                        and canonical_relation_type(r.relation_type) in CANONICAL_RELATION_TYPES
                        and r.confidence_status in {"documented", "sdk_validated", "runtime_verified"}]
            for relation in incoming:
                source = relation.source_api.removeprefix("focas:")
                if source in path:
                    cycles.append([*path, source])
                    continue
                edges.append({"source_api": source, "target_api": node, "relation": _relation_dict(relation)})
                if source in already:
                    visited.add(source)
                elif source not in visited:
                    walk(source, (*path, source), depth + 1)
                    visited.add(source)
                    optional = bool(relation.attributes.get("optional", False)) or relation.applicability.get("required") is False
                    (optional_ordered if optional else ordered).append(source)
        walk(target, (target,), 0)
        return {
            "available": True,
            "backend": "memory",
            "api_name": target,
            "knowledge_status": "known" if edges else "unknown",
            "required_calls": ordered,
            "optional_calls": optional_ordered,
            "target_call": target,
            "satisfied_nodes": sorted(already & set(visited)),
            "edges": _dedupe_relation_rows(edges, device_scope),
            "cycles": cycles,
        }

    def get_graph_overview(self, *, query: str = "", device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        scope_names = _scope_allowed_names(device_scope, allowed_api_names)
        nodes_by_name = {
            node.name: node for key, node in self.nodes.items() if key == node.name
        }
        if scope_names is not None:
            nodes_by_name = {name: node for name, node in nodes_by_name.items() if name in scope_names}
        relations = [
            relation for relation in self.relations.values()
            if self._relation_in_scope(relation, device_scope, allowed_api_names)
            and relation.confidence_status in CONFIRMED_RELATION_STATUSES
            and relation.source_api.removeprefix("focas:") in nodes_by_name
            and relation.target_api.removeprefix("focas:") in nodes_by_name
        ]
        # Generic and device-scoped copies describe the same graph fact for
        # an operator. Keep the more specific scoped copy in overview output;
        # the underlying relation records remain available to Planner tools.
        relations = _dedupe_relation_objects(relations, device_scope)
        needle = str(query or "").strip().casefold()
        if needle:
            matched = {name for name in nodes_by_name if needle in name.casefold()}
            visible = set(matched)
            for relation in relations:
                source = relation.source_api.removeprefix("focas:")
                target = relation.target_api.removeprefix("focas:")
                if source in matched or target in matched:
                    visible.update((source, target))
            nodes_by_name = {name: node for name, node in nodes_by_name.items() if name in visible}
            relations = [
                relation for relation in relations
                if relation.source_api.removeprefix("focas:") in visible
                and relation.target_api.removeprefix("focas:") in visible
            ]
        return {
            "available": True,
            "backend": "memory",
            "device_scope": device_scope,
            "query": str(query or ""),
            "scope_api_count": len(scope_names or nodes_by_name),
            "matched_api_names": sorted(nodes_by_name),
            "nodes": [asdict(nodes_by_name[name]) for name in sorted(nodes_by_name)],
            "relations": [_relation_dict(item) for item in relations],
            "node_count": len(nodes_by_name),
            "relation_count": len(relations),
        }

    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]:
        selected = [self.relations[item] for item in relation_ids if item in self.relations] if relation_ids else list(self.relations.values())
        unconfirmed = [
            item.relation_id for item in selected
            if _confirmed_relation_error(item) is not None
        ]
        return {
            "valid": not unconfirmed,
            "api_name": api_name.removeprefix("focas:"),
            "validated_relation_ids": [item.relation_id for item in selected if item.relation_id not in unconfirmed],
            "unconfirmed_relation_ids": unconfirmed,
        }

    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        relation = self.relations.get(relation_id)
        if relation is None:
            return {"ok": False, "stored": False, "status": "unconfirmed", "error": f"Unknown relation: {relation_id}"}
        if relation.confidence_status not in CONFIRMED_RELATION_STATUSES:
            return {"ok": False, "stored": False, "status": "unconfirmed", "error": "unknown relations must be audited before persistence"}
        evidence_ids = [str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id")]
        if not evidence_ids:
            return {"ok": False, "stored": False, "status": "unconfirmed", "relation": _relation_dict(relation), "error": "Verification requires execution evidence."}
        updated = TrafficRelation(
            relation_id=relation.relation_id,
            source_api=relation.source_api,
            target_api=relation.target_api,
            relation_type=relation.relation_type,
            rule_id=relation.rule_id,
            confidence_status="runtime_verified",
            evidence_ids=tuple(dict.fromkeys((*relation.evidence_ids, *evidence_ids))),
            applicability=relation.applicability,
            attributes=relation.attributes,
            version=relation.version + 1,
        )
        self.relations[relation_id] = updated
        return {"ok": True, "stored": True, "status": "runtime_verified", "relation": _relation_dict(updated), "evidence_count": len(evidence_ids)}

    def record_relation_failure(self, relation_id: str, evidence: list[dict[str, Any]], *, reason: str = "", replacement_relation_id: str = "") -> dict[str, Any]:
        relation = self.relations.get(relation_id)
        if relation is None:
            return {"ok": False, "error": f"Unknown relation: {relation_id}"}
        observation = {
            "reason": str(reason),
            "evidence_ids": [str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id")],
            **({"replacement_relation_id": replacement_relation_id} if replacement_relation_id else {}),
        }
        history = list(relation.attributes.get("prune_history", [])) if isinstance(relation.attributes, dict) else []
        history.append(observation)
        status = relation.confidence_status
        if status not in {"documented", "sdk_validated", "runtime_verified"}:
            status = "superseded" if replacement_relation_id else "rejected"
        updated = TrafficRelation(
            relation_id=relation.relation_id,
            source_api=relation.source_api,
            target_api=relation.target_api,
            relation_type=relation.relation_type,
            rule_id=relation.rule_id,
            confidence_status=status,
            evidence_ids=tuple(dict.fromkeys((*relation.evidence_ids, *observation["evidence_ids"]))),
            applicability=relation.applicability,
            attributes={**relation.attributes, "prune_history": history},
            version=relation.version + 1,
        )
        if status in {"rejected", "superseded"}:
            self.relations.pop(relation_id, None)
            return {
                "ok": True,
                "deleted": True,
                "relation": _relation_dict(updated),
                "preserved_verified_status": False,
            }
        self.relations[relation_id] = updated
        return {"ok": True, "relation": _relation_dict(updated), "preserved_verified_status": True}

    def migrate_legacy_relations(self) -> dict[str, Any]:
        """Normalize only already confirmed, structurally complete relations.

        Historical aliases are accepted at the input boundary, but this
        compatibility method must never turn a proposed or incomplete row
        into a persisted canonical edge. Unsafe rows are removed and reported
        for the evidence-aware audit command.
        """
        changed = 0
        deduplicated = 0
        removed = 0
        canonical_keys: dict[tuple[str, str, str, str], str] = {}
        for relation_id_key, relation in list(self.relations.items()):
            try:
                canonical = canonical_relation_type(relation.relation_type)
                normalized = TrafficRelation(
                    relation_id=relation.relation_id, source_api=relation.source_api,
                    target_api=relation.target_api, relation_type=canonical,
                    rule_id=relation.rule_id, confidence_status=relation.confidence_status,
                    evidence_ids=relation.evidence_ids, applicability=relation.applicability,
                    attributes=relation.attributes, version=relation.version,
                ).normalized()
            except (TypeError, ValueError):
                self.relations.pop(relation_id_key, None)
                removed += 1
                continue
            cleaned_attributes = {
                key: value for key, value in relation.attributes.items()
                if str(key).lower() not in {"provides", "verifies", "execution_order", "ordered_calls"}
            }
            normalized = TrafficRelation(
                relation_id=normalized.relation_id, source_api=normalized.source_api,
                target_api=normalized.target_api, relation_type=normalized.relation_type,
                rule_id=normalized.rule_id, confidence_status=normalized.confidence_status,
                evidence_ids=normalized.evidence_ids, applicability=normalized.applicability,
                attributes=cleaned_attributes,
                version=normalized.version + (0 if canonical == relation.relation_type and cleaned_attributes == relation.attributes else 1),
            )
            rejection = _confirmed_relation_error(normalized)
            if rejection:
                self.relations.pop(relation_id_key, None)
                removed += 1
                continue
            key = (
                normalized.source_api.removeprefix("focas:"),
                normalized.target_api.removeprefix("focas:"),
                str(canonical),
                str(normalized.applicability.get("device_scope", "")),
            )
            previous_id = canonical_keys.get(key)
            if previous_id is not None:
                previous = self.relations[previous_id]
                merged_attrs = dict(previous.attributes)
                for attr_name in ("parameter_rules",):
                    values = [*(merged_attrs.get(attr_name, []) or []), *(normalized.attributes.get(attr_name, []) or [])]
                    if values:
                        merged_attrs[attr_name] = list({json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item): item for item in values}.values())
                self.relations[previous_id] = TrafficRelation(
                    relation_id=previous.relation_id, source_api=previous.source_api, target_api=previous.target_api,
                    relation_type=canonical, rule_id=previous.rule_id or normalized.rule_id,
                    confidence_status=previous.confidence_status,
                    evidence_ids=tuple(dict.fromkeys((*previous.evidence_ids, *normalized.evidence_ids))),
                    applicability={**normalized.applicability, **previous.applicability}, attributes=merged_attrs,
                    version=max(previous.version, normalized.version) + 1,
                )
                self.relations.pop(relation_id_key, None)
                deduplicated += 1
                continue
            canonical_keys[key] = relation_id_key
            self.relations[relation_id_key] = normalized
            if canonical != relation.relation_type or cleaned_attributes != relation.attributes:
                changed += 1
        return {"ok": True, "changed": changed, "deduplicated": deduplicated, "removed_unconfirmed": removed, "relation_types": sorted(CANONICAL_RELATION_TYPES)}


class Neo4jTrafficGraphStore:
    """Neo4j-backed TRG store with explicit connection failures."""

    def __init__(self, uri: str, user: str, password: str, *, database: str = "neo4j", driver: Any | None = None, query_timeout_s: float = 10.0, max_dependency_depth: int = 8) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = driver
        self._schema_ready = False
        self.query_timeout_s = max(0.1, float(query_timeout_s))
        self.max_dependency_depth = max(1, min(int(max_dependency_depth), 16))

    @property
    def driver(self) -> Any:
        if self._driver is None:
            try:
                from neo4j import GraphDatabase  # type: ignore
            except Exception as exc:
                raise TrafficGraphUnavailable("Neo4j Python Driver is not installed.") from exc
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                self._driver.verify_connectivity()
            except Exception as exc:
                self._driver = None
                raise TrafficGraphUnavailable(f"Neo4j connection failed for {self.uri}: {exc}") from exc
        self.ensure_schema()
        return self._driver

    def ensure_schema(self) -> None:
        if self._schema_ready or self._driver is None:
            return
        try:
            with self._driver.session(database=self.database) as session:
                session.run("CREATE CONSTRAINT api_id_unique IF NOT EXISTS FOR (n:API) REQUIRE n.api_id IS UNIQUE")
                session.run("CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR (r:TRG_RELATION) REQUIRE r.relation_id IS UNIQUE")
            self._schema_ready = True
        except Exception as exc:
            raise TrafficGraphUnavailable(f"Neo4j schema initialization failed: {exc}") from exc

    def _run(self, query: str, *, timeout_s: float | None = None, **params: Any) -> list[dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                timeout = self.query_timeout_s if timeout_s is None else max(0.1, float(timeout_s))
                return [record.data() for record in session.run(query, timeout=timeout, **params)]
        except TrafficGraphUnavailable:
            raise
        except Exception as exc:
            raise TrafficGraphError(f"Neo4j query failed: {exc}") from exc

    def upsert_api(self, node: ApiNode) -> dict[str, Any]:
        rows = self._run(
            "MERGE (n:API {api_id:$api_id}) SET n.name=$name, n.protocol=$protocol, "
            "n.sdk_signature_ref=CASE WHEN $sdk_signature_ref='' THEN coalesce(n.sdk_signature_ref,'') ELSE $sdk_signature_ref END, "
            "n.description=CASE WHEN $description='' THEN coalesce(n.description,'') ELSE $description END, "
            "n.metadata_json=CASE WHEN $metadata_json='{}' THEN coalesce(n.metadata_json,'{}') ELSE $metadata_json END, "
            "n.parameters_json=CASE WHEN $parameters_json='{}' THEN coalesce(n.parameters_json,'{}') ELSE $parameters_json END, "
            "n.prerequisites_json=CASE WHEN $prerequisites_json='{}' THEN coalesce(n.prerequisites_json,'{}') ELSE $prerequisites_json END "
            "RETURN n.api_id AS api_id",
            api_id=node.api_id,
            name=node.name,
            protocol=node.protocol,
            sdk_signature_ref=node.sdk_signature_ref,
            description=node.description,
            metadata_json=json.dumps(node.metadata, ensure_ascii=False, sort_keys=True),
            parameters_json=json.dumps(node.parameters, ensure_ascii=False, sort_keys=True),
            prerequisites_json=json.dumps(node.prerequisites, ensure_ascii=False, sort_keys=True),
        )
        return {"ok": True, "api": asdict(node), "created": bool(rows)}

    def upsert_relation(self, relation: TrafficRelation) -> dict[str, Any]:
        item = relation.normalized()
        rejection = _confirmed_relation_error(item)
        if rejection:
            return _relation_write_rejection(item, rejection)
        query = (
            "MATCH (s:API {name:$source_api}), (t:API {name:$target_api}) "
            "MERGE (s)-[r:TRG_RELATION {relation_id:$relation_id}]->(t) "
            "SET r.relation_type=$relation_type, r.rule_id=$rule_id, r.confidence_status=$confidence_status, "
            "r.evidence_ids=$evidence_ids, r.applicability_json=$applicability_json, "
            "r.attributes_json=$attributes_json, r.version=$version "
            "RETURN r.relation_id AS relation_id"
        )
        rows = self._run(
            query,
            relation_id=item.relation_id,
            source_api=item.source_api,
            target_api=item.target_api,
            relation_type=item.relation_type,
            rule_id=item.rule_id,
            confidence_status=item.confidence_status,
            evidence_ids=list(item.evidence_ids),
            applicability_json=json.dumps(item.applicability, ensure_ascii=False, sort_keys=True),
            attributes_json=json.dumps(item.attributes, ensure_ascii=False, sort_keys=True),
            version=item.version,
        )
        if not rows:
            raise TrafficGraphError("Cannot create relation before both API nodes exist.")
        return {"ok": True, "relation": _relation_dict(item), "created": True}

    def get_api_info(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        scope_clause = " AND n.name IN $allowed_names" if scoped else ""
        rows = self._run(f"MATCH (n:API) WHERE (n.name=$name OR n.api_id=$api_id){scope_clause} RETURN n", name=api_name, api_id=f"focas:{api_name}", allowed_names=names)
        return {"available": bool(rows), "api": _decode_node_properties(rows[0].get("n")) if rows else None, "backend": "neo4j"}

    def get_traffic_chain(self, api_name: str, max_depth: int = 3, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        try:
            depth = max(1, min(int(max_depth), self.max_dependency_depth))
        except (TypeError, ValueError):
            depth = min(3, self.max_dependency_depth)
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        root = api_name.removeprefix("focas:")
        if scoped and root not in names:
            return {"available": True, "api_name": root, "device_scope": device_scope, "scope_api_count": len(names), "knowledge_status": "out_of_scope", "paths": [], "backend": "neo4j"}
        frontier = {root}
        visited = {root}
        paths: list[dict[str, Any]] = []
        for _ in range(depth):
            if not frontier:
                break
            endpoint_clause = " AND source.name IN $allowed_names AND target.name IN $allowed_names" if scoped else ""
            scope_clause = " AND (coalesce(r.applicability_json, '') = '' OR r.applicability_json CONTAINS $device_scope)" if device_scope else ""
            rows = self._run(
                "MATCH (source:API)-[r:TRG_RELATION]-(target:API) "
                "WHERE (source.name IN $frontier OR target.name IN $frontier) "
                "AND r.relation_type IN ['filters','prepares'] "
                "AND r.confidence_status IN ['documented','sdk_validated','runtime_verified']"
                f"{endpoint_clause}{scope_clause} "
                "RETURN source{.*} AS source, target{.*} AS target, r{.*} AS relation",
                frontier=sorted(frontier), allowed_names=names, device_scope=device_scope,
            )
            next_frontier: set[str] = set()
            seen_edges: set[tuple[str, str, str]] = set()
            for row in rows:
                source = _decode_node_properties(row.get("source"))
                target = _decode_node_properties(row.get("target"))
                relation = _decode_relation_properties(row.get("relation"))
                relation["relation_type"] = canonical_relation_type(relation.get("relation_type", ""))
                source_name = str(source.get("name", ""))
                target_name = str(target.get("name", ""))
                edge_key = (source_name, target_name, str(relation.get("relation_id", "")))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                for endpoint in (source_name, target_name):
                    if endpoint not in visited:
                        next_frontier.add(endpoint)
                paths.append({"nodes": [source, target], "relations": [relation]})
            visited.update(next_frontier)
            frontier = next_frontier
        return {"available": True, "api_name": root, "device_scope": device_scope, "scope_api_count": len(names), "knowledge_status": "known" if paths else "unknown", "paths": _dedupe_graph_paths(paths, device_scope), "backend": "neo4j", "max_depth": depth, "cycle_detection": True}

    def get_api_dependencies(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        endpoint_clause = (" AND ((source.name IN $allowed_names AND target.name IN $allowed_names) "
                           "OR coalesce(r.applicability_json, '') CONTAINS $device_scope)" if scoped else "")
        scope_clause = " AND (coalesce(r.applicability_json, '') = '' OR r.applicability_json CONTAINS $device_scope)" if device_scope else ""
        rows = self._run(
            "MATCH (source:API)-[r:TRG_RELATION]->(target:API) "
            f"WHERE target.name=$name AND r.relation_type IN ['filters','prepares'] AND r.confidence_status IN ['documented','sdk_validated','runtime_verified']{endpoint_clause}{scope_clause} "
            "RETURN source{.*} AS source, r{.*} AS relation, target{.*} AS target",
            name=api_name, allowed_names=names, device_scope=device_scope,
        )
        for row in rows:
            row["relation"] = _decode_relation_properties(row.get("relation"))
            if isinstance(row.get("relation"), dict):
                row["relation"]["relation_type"] = canonical_relation_type(row["relation"].get("relation_type", ""))
        rows = _dedupe_relation_rows(rows, device_scope)
        return {"available": True, "api_name": api_name, "device_scope": device_scope, "knowledge_status": "known" if rows else "unknown", "dependencies": rows, "backend": "neo4j"}

    def get_parameter_rules(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        endpoint_clause = (" AND ((source.name IN $allowed_names AND target.name IN $allowed_names) "
                           "OR coalesce(r.applicability_json, '') CONTAINS $device_scope)" if scoped else "")
        scope_clause = " AND (coalesce(r.applicability_json, '') = '' OR r.applicability_json CONTAINS $device_scope)" if device_scope else ""
        rows = self._run(
            "MATCH (source:API)-[r:TRG_RELATION]->(target:API) "
            f"WHERE target.name=$name AND r.relation_type = 'filters' AND r.confidence_status IN ['documented','sdk_validated','runtime_verified']{endpoint_clause}{scope_clause} "
            "RETURN r{.*} AS relation, source.name AS source_api",
            name=api_name, allowed_names=names, device_scope=device_scope,
        )
        rows = _dedupe_relation_rows(rows, device_scope)
        for row in rows:
            if isinstance(row.get("relation"), dict):
                row["relation"]["relation_type"] = "filters"
        rules = _parameter_rules_from_relations(rows)
        return {"available": True, "api_name": api_name, "device_scope": device_scope, "knowledge_status": "known" if rules else "unknown", "rules": rules, "backend": "neo4j"}

    def get_parameter_prerequisites(self, api_name: str, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        endpoint_clause = (" AND ((source.name IN $allowed_names AND target.name IN $allowed_names) "
                           "OR coalesce(r.applicability_json, '') CONTAINS $device_scope)" if scoped else "")
        scope_clause = " AND (coalesce(r.applicability_json, '') = '' OR r.applicability_json CONTAINS $device_scope)" if device_scope else ""
        rows = self._run(
            "MATCH (source:API)-[r:TRG_RELATION]->(target:API) "
            f"WHERE target.name=$name AND r.relation_type = 'prepares' AND r.confidence_status IN ['documented','sdk_validated','runtime_verified']{endpoint_clause}{scope_clause} "
            "RETURN source.name AS source_api, target.name AS target_api, r{.*} AS relation",
            name=api_name, allowed_names=names, device_scope=device_scope,
        )
        for row in rows:
            row["relation"] = _decode_relation_properties(row.get("relation"))
            if isinstance(row.get("relation"), dict):
                row["relation"]["relation_type"] = "prepares"
        rows = [
            row for row in rows
            if isinstance(row.get("relation"), dict)
            and isinstance(row["relation"].get("attributes"), dict)
            and "parameter_number" in row["relation"]["attributes"]
        ]
        for row in rows:
            row["parameter"] = dict(row["relation"].get("attributes", {}))
        return {
            "available": True,
            "api_name": api_name,
            "device_scope": device_scope,
            "knowledge_status": "known" if rows else "unknown",
            "prerequisites": _dedupe_relation_rows(rows, device_scope),
            "backend": "neo4j",
        }

    def get_dependency_plan(self, api_name: str, max_depth: int = 8, *, device_scope: str = "", allowed_api_names: Iterable[str] | None = None, satisfied: Iterable[str] | None = None) -> dict[str, Any]:
        """Return confirmed incoming dependencies using bounded BFS queries.

        Neo4j variable-length path matching enumerates every path before a
        post-filter can reject it.  The graph is therefore traversed one
        frontier at a time, with confirmed relation predicates in each query,
        a hard depth cap, cycle detection, and a driver timeout.
        """
        try:
            depth = max(1, min(int(max_depth), self.max_dependency_depth))
        except (TypeError, ValueError):
            depth = self.max_dependency_depth
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        target_name = api_name.removeprefix("focas:")
        if scoped and target_name not in names:
            return {"available": True, "backend": "neo4j", "api_name": target_name, "knowledge_status": "out_of_scope", "required_calls": [], "edges": [], "satisfied_nodes": [], "cycles": []}
        already = {str(item).removeprefix("focas:") for item in (satisfied or ())}
        edge_rows: list[dict[str, Any]] = []
        ordered: list[str] = []
        optional_ordered: list[str] = []
        encountered: set[str] = set()
        frontier = {target_name}
        visited = {target_name}
        cycles: list[list[str]] = []
        for level in range(depth):
            if not frontier:
                break
            endpoint_clause = (" AND ((source.name IN $allowed_names AND target.name IN $allowed_names) "
                               "OR coalesce(r.applicability_json, '') CONTAINS $device_scope)" if scoped else "")
            scope_clause = " AND (coalesce(r.applicability_json, '') = '' OR r.applicability_json CONTAINS $device_scope)" if device_scope else ""
            rows = self._run(
                "MATCH (source:API)-[r:TRG_RELATION]->(target:API) "
                "WHERE target.name IN $targets "
                "AND r.relation_type IN ['filters','prepares'] "
                "AND r.confidence_status IN ['documented','sdk_validated','runtime_verified']"
                f"{endpoint_clause}{scope_clause} "
                "RETURN source.name AS source_api, target.name AS target_api, r{.*} AS relation",
                targets=sorted(frontier), allowed_names=names, device_scope=device_scope,
            )
            next_frontier: set[str] = set()
            for row in rows:
                source = str(row.get("source_api", "")).removeprefix("focas:")
                target = str(row.get("target_api", "")).removeprefix("focas:")
                relation = _decode_relation_properties(row.get("relation"))
                relation["relation_type"] = canonical_relation_type(relation.get("relation_type", ""))
                edge_key = (source, target, str(relation.get("relation_id", "")))
                if any(item.get("_edge_key") == edge_key for item in edge_rows):
                    continue
                if source in visited:
                    cycles.append([target, source])
                else:
                    next_frontier.add(source)
                encountered.add(source)
                edge_rows.append({"source_api": source, "target_api": target, "relation": relation, "_edge_key": edge_key})
                optional = bool(relation.get("attributes", {}).get("optional", False)) or relation.get("applicability", {}).get("required") is False
                if source not in already:
                    (optional_ordered if optional else ordered).append(source)
            visited.update(next_frontier)
            frontier = next_frontier
        ordered = list(dict.fromkeys(reversed(ordered)))
        optional_ordered = list(dict.fromkeys(reversed(optional_ordered)))
        for row in edge_rows:
            row.pop("_edge_key", None)
        return {"available": True, "backend": "neo4j", "api_name": target_name, "knowledge_status": "known" if edge_rows else "unknown", "required_calls": ordered, "optional_calls": optional_ordered, "target_call": target_name, "satisfied_nodes": sorted(set(already) & encountered), "edges": _dedupe_relation_rows(edge_rows, device_scope), "cycles": cycles}

    def get_graph_overview(self, *, query: str = "", device_scope: str = "", allowed_api_names: Iterable[str] | None = None) -> dict[str, Any]:
        names = sorted(_scope_allowed_names(device_scope, allowed_api_names) or ())
        scoped = allowed_api_names is not None or bool(device_scope)
        node_scope = " WHERE n.name IN $allowed_names" if scoped else ""
        node_rows = self._run(
            f"MATCH (n:API){node_scope} RETURN n{{.*}} AS node",
            allowed_names=names,
        )
        relation_query = "MATCH (s:API)-[r:TRG_RELATION]->(t:API) "
        relation_query += (
            "WHERE ((s.name IN $allowed_names AND t.name IN $allowed_names) "
            "OR coalesce(r.applicability_json, '') CONTAINS $device_scope) AND "
            if scoped else "WHERE "
        )
        relation_query += (
            "r.relation_type IN ['filters','prepares'] AND "
            "r.confidence_status IN ['documented','sdk_validated','runtime_verified'] "
            "RETURN s.name AS source_api, t.name AS target_api, r{.*} AS relation"
        )
        relation_rows = self._run(
            relation_query,
            allowed_names=names,
            device_scope=device_scope,
        )
        # A seeded graph can contain a generic fact and an equivalent
        # FANUC 0i-D scoped fact. They are distinct knowledge records but
        # should render as one overview edge.
        relation_rows = _dedupe_relation_rows(relation_rows, device_scope)
        nodes = [_decode_node_properties(row.get("node")) for row in node_rows if isinstance(row.get("node"), dict)]
        # A confirmed, device-scoped dependency may point to a documented
        # helper that is absent from the compatibility allow-list. Keep that
        # endpoint visible so the graph does not show a dangling edge.
        visible_names = {str(node.get("name", "")) for node in nodes}
        relation_endpoint_names = {
            str(row.get(key, "")) for row in relation_rows for key in ("source_api", "target_api")
            if str(row.get(key, ""))
        }
        missing_names = sorted(relation_endpoint_names - visible_names)
        if missing_names:
            extra_rows = self._run(
                "MATCH (n:API) WHERE n.name IN $missing_names RETURN n{.*} AS node",
                missing_names=missing_names,
            )
            nodes.extend(
                _decode_node_properties(row.get("node"))
                for row in extra_rows if isinstance(row.get("node"), dict)
            )
        relations: list[dict[str, Any]] = []
        for row in relation_rows:
            relation = _decode_relation_properties(row.get("relation"))
            relation["relation_type"] = canonical_relation_type(relation.get("relation_type", ""))
            relation["source_api"] = row.get("source_api")
            relation["target_api"] = row.get("target_api")
            relations.append(relation)
        needle = str(query or "").strip().casefold()
        if needle:
            matched = {str(node.get("name", "")) for node in nodes if needle in str(node.get("name", "")).casefold()}
            visible = set(matched)
            for relation in relations:
                source = str(relation.get("source_api", ""))
                target = str(relation.get("target_api", ""))
                if source in matched or target in matched:
                    visible.update((source, target))
            nodes = [node for node in nodes if str(node.get("name", "")) in visible]
            relations = [
                relation for relation in relations
                if str(relation.get("source_api", "")) in visible
                and str(relation.get("target_api", "")) in visible
            ]
        return {
            "available": True,
            "backend": "neo4j",
            "device_scope": device_scope,
            "query": str(query or ""),
            "scope_api_count": len(names) if scoped else len(node_rows),
            "matched_api_names": sorted(str(node.get("name", "")) for node in nodes if node.get("name")),
            "nodes": nodes,
            "relations": relations,
            "node_count": len(nodes),
            "relation_count": len(relations),
        }

    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]:
        ids = list(relation_ids)
        rows = self._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) WHERE (s.name=$name OR t.name=$name) "
            "AND ($ids = [] OR r.relation_id IN $ids) "
            "RETURN s.name AS source_api, t.name AS target_api, r{.*} AS relation",
            name=api_name, ids=ids,
        )
        valid: list[str] = []
        unknown: list[str] = []
        for row in rows:
            raw = _decode_relation_properties(row.get("relation"))
            relation = TrafficRelation(
                relation_id=str(raw.get("relation_id", "")),
                source_api=str(row.get("source_api", "")),
                target_api=str(row.get("target_api", "")),
                relation_type=str(raw.get("relation_type", "")),
                rule_id=str(raw.get("rule_id", "")),
                confidence_status=str(raw.get("confidence_status", "")),
                evidence_ids=tuple(raw.get("evidence_ids", ()) or ()),
                applicability=dict(raw.get("applicability", {}) or {}),
                attributes=dict(raw.get("attributes", {}) or {}),
                version=int(raw.get("version", 1) or 1),
            )
            if _confirmed_relation_error(relation) is None:
                valid.append(relation.relation_id)
            else:
                unknown.append(relation.relation_id)
        return {"valid": not unknown, "api_name": api_name, "validated_relation_ids": valid, "unconfirmed_relation_ids": unknown}

    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        evidence_ids = [str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id")]
        if not evidence_ids:
            return {"ok": False, "stored": False, "status": "unconfirmed", "error": "Verification requires execution evidence."}
        current = self._run(
            "MATCH (s:API)-[r:TRG_RELATION {relation_id:$relation_id}]->(t:API) "
            "RETURN s.name AS source_api, t.name AS target_api, r{.*} AS relation",
            relation_id=relation_id,
        )
        if not current:
            return {"ok": False, "stored": False, "status": "unconfirmed", "error": f"Unknown relation: {relation_id}"}
        existing = _decode_relation_properties(current[0].get("relation"))
        if existing.get("confidence_status") not in CONFIRMED_RELATION_STATUSES:
            return {"ok": False, "stored": False, "status": "unconfirmed", "error": "unknown relations must be audited before persistence"}
        rows = self._run(
            "MATCH ()-[r:TRG_RELATION {relation_id:$relation_id}]->() "
            "SET r.confidence_status='runtime_verified', r.evidence_ids=$evidence_ids, r.version=coalesce(r.version,0)+1 "
            "RETURN r{.*} AS relation",
            relation_id=relation_id,
            evidence_ids=list(dict.fromkeys([*(existing.get("evidence_ids", []) or []), *evidence_ids])),
        )
        return {"ok": bool(rows), "stored": bool(rows), "status": "runtime_verified" if rows else "unconfirmed", "relation": rows[0].get("relation") if rows else None, "evidence_count": len(evidence_ids)}

    def record_relation_failure(self, relation_id: str, evidence: list[dict[str, Any]], *, reason: str = "", replacement_relation_id: str = "") -> dict[str, Any]:
        current = self._run(
            "MATCH ()-[r:TRG_RELATION {relation_id:$relation_id}]->() RETURN r{.*} AS relation",
            relation_id=relation_id,
        )
        if not current:
            return {"ok": False, "error": f"Unknown relation: {relation_id}"}
        relation = current[0].get("relation", {})
        try:
            attributes = json.loads(relation.get("attributes_json", "{}"))
        except (TypeError, ValueError):
            attributes = {}
        history = list(attributes.get("prune_history", [])) if isinstance(attributes, dict) else []
        observation = {
            "reason": str(reason),
            "evidence_ids": [str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id")],
            **({"replacement_relation_id": replacement_relation_id} if replacement_relation_id else {}),
        }
        history.append(observation)
        attributes = {**attributes, "prune_history": history}
        status = str(relation.get("confidence_status", "proposed"))
        if status not in {"documented", "sdk_validated", "runtime_verified"}:
            status = "superseded" if replacement_relation_id else "rejected"
        if status in {"rejected", "superseded"}:
            rows = self._run(
                "MATCH ()-[r:TRG_RELATION {relation_id:$relation_id}]->() "
                "DELETE r RETURN count(r) AS deleted",
                relation_id=relation_id,
            )
            deleted = int(rows[0].get("deleted", 0)) if rows else 0
            return {
                "ok": bool(deleted),
                "deleted": bool(deleted),
                "relation": {
                    **relation,
                    "confidence_status": status,
                    "attributes_json": json.dumps(attributes, ensure_ascii=False, sort_keys=True),
                    "evidence_ids": list(dict.fromkeys([*(relation.get("evidence_ids", []) or []), *observation["evidence_ids"]])),
                },
                "preserved_verified_status": False,
            }
        rows = self._run(
            "MATCH ()-[r:TRG_RELATION {relation_id:$relation_id}]->() "
            "SET r.confidence_status=$status, r.attributes_json=$attributes_json, "
            "r.evidence_ids=$evidence_ids, r.version=coalesce(r.version,0)+1 "
            "RETURN r{.*} AS relation",
            relation_id=relation_id,
            status=status,
            attributes_json=json.dumps(attributes, ensure_ascii=False, sort_keys=True),
            evidence_ids=list(dict.fromkeys([*(relation.get("evidence_ids", []) or []), *observation["evidence_ids"]])),
        )
        return {"ok": bool(rows), "relation": rows[0].get("relation") if rows else None, "preserved_verified_status": status in {"documented", "sdk_validated", "runtime_verified"}}

    def migrate_legacy_relations(self) -> dict[str, Any]:
        """Refuse unsafe bulk conversion; use ``scripts/audit_trg_neo4j.py``."""
        return {
            "ok": False,
            "blocked": True,
            "reason": "Legacy relations require evidence-aware per-edge audit; run scripts/audit_trg_neo4j.py.",
            "relation_types": sorted(CANONICAL_RELATION_TYPES),
        }
    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._schema_ready = False


class TrafficRelationshipService:
    """Stable service boundary used by Agents and tools."""

    def __init__(self, store: TrafficGraphStore | None = None) -> None:
        self.store = store
        self.last_error = ""

    @classmethod
    def from_environment(cls) -> "TrafficRelationshipService":
        """Build the graph service from process configuration or local install metadata.

        The desktop launcher does not necessarily inherit shell environment
        variables.  The Neo4j installer writes a small credentials file next
        to the configured D: drive installation, so use it as a local
        configuration source when no explicit URI is present.  Only the
        four expected keys are accepted; the password is never returned in
        diagnostics or persisted by this service.
        """
        values: dict[str, str] = {}
        credentials_file = os.getenv("SMP_TRG_NEO4J_CREDENTIALS_FILE", "").strip()
        candidate_paths = [Path(credentials_file)] if credentials_file else []
        if not candidate_paths:
            candidate_paths.append(Path(r"D:\Neo4j\neo4j.credentials"))
        for path in candidate_paths:
            try:
                if not path.is_file():
                    continue
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip().lower()
                    if key in {"uri", "user", "password", "database"}:
                        values[key] = value.strip()
                break
            except (OSError, UnicodeError):
                # An unreadable optional file should not make offline planning
                # fail; explicit environment variables still take precedence.
                continue
        def configured(name: str, fallback: str) -> str:
            value = os.getenv(name)
            return value.strip() if value and value.strip() else fallback

        uri = configured("SMP_TRG_NEO4J_URI", values.get("uri", "")).strip()
        if not uri:
            return cls(None)
        try:
            query_timeout_s = float(configured("SMP_TRG_NEO4J_QUERY_TIMEOUT_S", "10"))
        except (TypeError, ValueError):
            query_timeout_s = 10.0
        try:
            max_dependency_depth = int(configured("SMP_TRG_NEO4J_MAX_DEPENDENCY_DEPTH", "8"))
        except (TypeError, ValueError):
            max_dependency_depth = 8
        return cls(Neo4jTrafficGraphStore(
            uri,
            configured("SMP_TRG_NEO4J_USER", values.get("user", "neo4j")),
            configured("SMP_TRG_NEO4J_PASSWORD", values.get("password", "")),
            database=configured("SMP_TRG_NEO4J_DATABASE", values.get("database", "neo4j")),
            query_timeout_s=query_timeout_s,
            max_dependency_depth=max_dependency_depth,
        ))

    @property
    def available(self) -> bool:
        return self.store is not None

    def ensure_local_server_started(self, *, timeout_s: float = 20.0) -> dict[str, Any]:
        """Start a configured local Neo4j server and wait for Bolt readiness.

        This is deliberately limited to localhost and the installed server
        layout. A remote URI is only checked by the normal driver connection;
        the application never attempts to start a remote service.
        """
        if not isinstance(self.store, Neo4jTrafficGraphStore):
            return self.status
        uri = str(self.store.uri).lower()
        if not any(host in uri for host in ("localhost", "127.0.0.1", "::1")):
            return {**self.status, "status": "remote", "error": "remote Neo4j server is not auto-started"}
        try:
            self.store.driver
            self.last_error = ""
            return {**self.status, "status": "connected", "available": True, "error": ""}
        except (TrafficGraphUnavailable, TrafficGraphError):
            pass
        server_home = os.getenv("SMP_TRG_NEO4J_HOME", "").strip() or r"D:\Neo4j\server"
        batch = Path(server_home) / "bin" / "neo4j.bat"
        if not batch.is_file():
            self.last_error = f"Neo4j server launcher not found: {batch}"
            return {**self.status, "status": "error", "error": self.last_error}
        # This installation is a server distribution without a registered
        # Windows service, so ``start`` cannot work. Run ``console`` detached
        # and keep the server process independent of the desktop UI.
        try:
            launch_env = dict(os.environ)
            if not launch_env.get("JAVA_HOME"):
                runtime_root = Path(server_home).parent / "runtime"
                jdk_candidates = sorted(runtime_root.glob("jdk-*")) if runtime_root.is_dir() else []
                if jdk_candidates:
                    launch_env["JAVA_HOME"] = str(jdk_candidates[-1])
            if os.name == "nt":
                flags = (
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                )
                subprocess.Popen(
                    ["cmd.exe", "/d", "/c", str(batch), "console"],
                    cwd=str(Path(server_home)), env=launch_env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, close_fds=True,
                    creationflags=flags,
                )
            else:
                subprocess.Popen(
                    [str(batch), "console"], cwd=str(Path(server_home)), env=launch_env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, close_fds=True,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.last_error = f"Neo4j 自动启动失败: {exc}"
            return {**self.status, "status": "error", "error": self.last_error}
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        while time.monotonic() < deadline:
            try:
                # Accessing the driver performs verify_connectivity and schema
                # initialization through the existing connection path.
                self.store.driver
                self.last_error = ""
                return {**self.status, "status": "connected", "available": True, "error": ""}
            except (TrafficGraphUnavailable, TrafficGraphError):
                time.sleep(0.5)
        self.last_error = f"Neo4j 自动启动后仍无法连接：{self.store.uri}"
        return {**self.status, "status": "error", "error": self.last_error}

    @property
    def status(self) -> dict[str, Any]:
        """Return connection/configuration state without exposing credentials."""
        if self.store is None:
            return {
                "status": "unconfigured",
                "available": False,
                "backend": "none",
                "error": self.last_error or "TRG backend is not configured",
            }
        backend = "neo4j" if isinstance(self.store, Neo4jTrafficGraphStore) else "memory"
        if self.last_error:
            return {"status": "error", "available": False, "backend": backend, "error": self.last_error}
        return {"status": "configured", "available": True, "backend": backend, "error": ""}

    def unavailable(self, operation: str) -> dict[str, Any]:
        return {
            "available": False,
            "backend": "none",
            "operation": operation,
            "knowledge_status": "unknown",
            "error": self.last_error or "TRG backend is not configured; no relation is treated as verified.",
        }

    def call(self, method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.store is None:
            return self.unavailable(method)
        try:
            result = getattr(self.store, method)(*args, **kwargs)
            if isinstance(result, dict) and result.get("available") is False:
                self.last_error = str(result.get("error", "TRG backend unavailable"))
            else:
                self.last_error = ""
            return result
        except (TrafficGraphUnavailable, TrafficGraphError) as exc:
            # A configured but stopped Neo4j instance is an infrastructure
            # condition.  Return an explicit unavailable result so Planner can
            # use its documented local facts without routing into repair_plan.
            self.last_error = str(exc)
            return {"available": False, "backend": "neo4j", "knowledge_status": "unknown", "error": self.last_error}


def relation_id(source_api: str, target_api: str, relation_type: str, rule_id: str = "") -> str:
    raw = "|".join((source_api, target_api, relation_type.upper(), rule_id))
    return "rel:" + re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)


def relation_from_dict(payload: dict[str, Any]) -> TrafficRelation:
    return TrafficRelation(
        relation_id=str(payload.get("relation_id") or relation_id(
            str(payload.get("source_api", "")), str(payload.get("target_api", "")), str(payload.get("relation_type", "")), str(payload.get("rule_id", ""))
        )),
        source_api=str(payload.get("source_api", "")),
        target_api=str(payload.get("target_api", "")),
        relation_type=str(payload.get("relation_type", "")),
        rule_id=str(payload.get("rule_id", "")),
        confidence_status=str(payload.get("confidence_status", "proposed")),
        evidence_ids=tuple(payload.get("evidence_ids", ()) or ()),
        applicability=dict(payload.get("applicability", {}) or {}),
        attributes=dict(payload.get("attributes", {}) or {}),
        version=int(payload.get("version", 1)),
    ).normalized()


def build_default_relationship_service() -> TrafficRelationshipService:
    return TrafficRelationshipService.from_environment()
