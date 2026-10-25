"""Traffic Relationship Graph (TRG) storage and evidence-aware queries.

The production store is Neo4j. An explicit in-memory store is provided for
offline planning tests and migration previews; it is never treated as device
evidence unless the caller supplies that evidence.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol


RELATION_TYPES = {
    "REQUIRES",
    "PRECEDES",
    "PROVIDES_INPUT",
    "READS_AFTER_WRITE",
    "SESSION_SEQUENCE",
}
RELATION_STATUSES = {"proposed", "documented", "runtime_verified", "rejected", "superseded"}


class TrafficGraphError(RuntimeError):
    pass


class TrafficGraphUnavailable(TrafficGraphError):
    pass


@dataclass(frozen=True, slots=True)
class ApiNode:
    api_id: str
    name: str
    protocol: str = "FOCAS"
    sdk_signature_ref: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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
        relation_type = str(self.relation_type).upper()
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
    def get_api_info(self, api_name: str) -> dict[str, Any]: ...
    def get_traffic_chain(self, api_name: str, max_depth: int = 3) -> dict[str, Any]: ...
    def get_api_dependencies(self, api_name: str) -> dict[str, Any]: ...
    def get_parameter_rules(self, api_name: str) -> dict[str, Any]: ...
    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]: ...
    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]: ...


class InMemoryTrafficGraphStore:
    """Deterministic store used by offline simulations and unit tests."""

    def __init__(self) -> None:
        self.nodes: dict[str, ApiNode] = {}
        self.relations: dict[str, TrafficRelation] = {}

    def upsert_api(self, node: ApiNode) -> dict[str, Any]:
        self.nodes[node.name] = node
        self.nodes[node.api_id] = node
        return {"ok": True, "api": asdict(node), "created": True}

    def upsert_relation(self, relation: TrafficRelation) -> dict[str, Any]:
        item = relation.normalized()
        previous = self.relations.get(item.relation_id)
        if previous is not None and item.version < previous.version:
            raise ValueError(f"Relation version regressed: {item.relation_id}")
        self.relations[item.relation_id] = item
        return {"ok": True, "relation": asdict(item), "created": previous is None}

    def get_api_info(self, api_name: str) -> dict[str, Any]:
        node = self.nodes.get(api_name) or self.nodes.get(f"focas:{api_name}")
        return {"available": node is not None, "api": asdict(node) if node else None, "backend": "memory"}

    def _relations_for(self, api_name: str) -> list[TrafficRelation]:
        name = api_name.removeprefix("focas:")
        return [
            row for row in self.relations.values()
            if row.source_api.removeprefix("focas:") == name
            or row.target_api.removeprefix("focas:") == name
        ]

    def get_traffic_chain(self, api_name: str, max_depth: int = 3) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        frontier = {name}
        visited = {name}
        selected: dict[str, TrafficRelation] = {}
        for _ in range(max(1, int(max_depth))):
            next_frontier: set[str] = set()
            for relation in self.relations.values():
                source = relation.source_api.removeprefix("focas:")
                target = relation.target_api.removeprefix("focas:")
                if source in frontier or target in frontier:
                    selected[relation.relation_id] = relation
                    next_frontier.update({source, target} - visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return {
            "available": True,
            "api_name": name,
            "knowledge_status": "known" if api_known else "unknown",
            "nodes": [asdict(self.nodes[node]) for node in sorted(visited) if node in self.nodes],
            "relations": [asdict(item) for item in selected.values()],
            "backend": "memory",
        }

    def get_api_dependencies(self, api_name: str) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        rows = [
            relation for relation in self.relations.values()
            if relation.target_api.removeprefix("focas:") == name
            and relation.relation_type in {"REQUIRES", "SESSION_SEQUENCE", "PROVIDES_INPUT"}
        ]
        return {"available": True, "api_name": name, "knowledge_status": "known" if api_known else "unknown", "dependencies": [asdict(item) for item in rows], "backend": "memory"}

    def get_parameter_rules(self, api_name: str) -> dict[str, Any]:
        name = api_name.removeprefix("focas:")
        api_known = name in self.nodes
        rows = [
            relation for relation in self.relations.values()
            if relation.target_api.removeprefix("focas:") == name
            and relation.relation_type == "PROVIDES_INPUT"
        ]
        return {"available": True, "api_name": name, "knowledge_status": "known" if api_known else "unknown", "rules": [asdict(item) for item in rows], "backend": "memory"}

    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]:
        selected = [self.relations[item] for item in relation_ids if item in self.relations] if relation_ids else list(self.relations.values())
        proposed = [item.relation_id for item in selected if item.confidence_status == "proposed"]
        rejected = [item.relation_id for item in selected if item.confidence_status in {"rejected", "superseded"}]
        return {
            "valid": not proposed and not rejected,
            "api_name": api_name.removeprefix("focas:"),
            "validated_relation_ids": [item.relation_id for item in selected if item.confidence_status in {"documented", "runtime_verified"}],
            "proposed_relation_ids": proposed,
            "rejected_relation_ids": rejected,
        }

    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        relation = self.relations.get(relation_id)
        if relation is None:
            return {"ok": False, "error": f"Unknown relation: {relation_id}"}
        if not evidence:
            return {"ok": False, "relation": asdict(relation), "error": "Verification requires execution evidence."}
        updated = TrafficRelation(
            relation_id=relation.relation_id,
            source_api=relation.source_api,
            target_api=relation.target_api,
            relation_type=relation.relation_type,
            rule_id=relation.rule_id,
            confidence_status="runtime_verified",
            evidence_ids=tuple(dict.fromkeys((*relation.evidence_ids, *(str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id"))))),
            applicability=relation.applicability,
            attributes=relation.attributes,
            version=relation.version + 1,
        )
        self.relations[relation_id] = updated
        return {"ok": True, "relation": asdict(updated), "evidence_count": len(evidence)}


class Neo4jTrafficGraphStore:
    """Neo4j-backed TRG store with explicit connection failures."""

    def __init__(self, uri: str, user: str, password: str, *, database: str = "neo4j", driver: Any | None = None) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = driver
        self._schema_ready = False

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

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self.driver.session(database=self.database) as session:
                return [record.data() for record in session.run(query, **params)]
        except TrafficGraphUnavailable:
            raise
        except Exception as exc:
            raise TrafficGraphError(f"Neo4j query failed: {exc}") from exc

    def upsert_api(self, node: ApiNode) -> dict[str, Any]:
        rows = self._run(
            "MERGE (n:API {api_id:$api_id}) SET n.name=$name, n.protocol=$protocol, "
            "n.sdk_signature_ref=$sdk_signature_ref, n.description=$description, n.metadata=$metadata "
            "RETURN n.api_id AS api_id",
            **asdict(node),
        )
        return {"ok": True, "api": asdict(node), "created": bool(rows)}

    def upsert_relation(self, relation: TrafficRelation) -> dict[str, Any]:
        item = relation.normalized()
        query = (
            "MATCH (s:API {name:$source_api}), (t:API {name:$target_api}) "
            "MERGE (s)-[r:TRG_RELATION {relation_id:$relation_id}]->(t) "
            "SET r.relation_type=$relation_type, r.rule_id=$rule_id, r.confidence_status=$confidence_status, "
            "r.evidence_ids=$evidence_ids, r.applicability=$applicability, r.attributes=$attributes, r.version=$version "
            "RETURN r.relation_id AS relation_id"
        )
        rows = self._run(query, **asdict(item))
        if not rows:
            raise TrafficGraphError("Cannot create relation before both API nodes exist.")
        return {"ok": True, "relation": asdict(item), "created": True}

    def get_api_info(self, api_name: str) -> dict[str, Any]:
        rows = self._run("MATCH (n:API) WHERE n.name=$name OR n.api_id=$api_id RETURN n", name=api_name, api_id=f"focas:{api_name}")
        return {"available": bool(rows), "api": rows[0].get("n") if rows else None, "backend": "neo4j"}

    def get_traffic_chain(self, api_name: str, max_depth: int = 3) -> dict[str, Any]:
        try:
            depth = max(1, min(int(max_depth), 8))
        except (TypeError, ValueError):
            depth = 3
        rows = self._run(
            f"MATCH p=(n:API)-[rels:TRG_RELATION*1..{depth}]-(target:API) "
            "WHERE n.name=$name OR target.name=$name "
            "RETURN [node IN nodes(p) | node{.*}] AS nodes, "
            "[rel IN relationships(p) | rel{.*}] AS relations",
            name=api_name,
        )
        return {"available": True, "api_name": api_name, "knowledge_status": "known" if rows else "unknown", "paths": rows, "backend": "neo4j"}

    def get_api_dependencies(self, api_name: str) -> dict[str, Any]:
        rows = self._run(
            "MATCH (source:API)-[r:TRG_RELATION]->(target:API) "
            "WHERE target.name=$name AND r.confidence_status IN ['documented','runtime_verified'] "
            "RETURN source{.*} AS source, r{.*} AS relation, target{.*} AS target",
            name=api_name,
        )
        return {"available": True, "api_name": api_name, "knowledge_status": "known" if rows else "unknown", "dependencies": rows, "backend": "neo4j"}

    def get_parameter_rules(self, api_name: str) -> dict[str, Any]:
        rows = self._run(
            "MATCH (source:API)-[r:TRG_RELATION {relation_type:'PROVIDES_INPUT'}]->(target:API) "
            "WHERE target.name=$name AND r.confidence_status IN ['documented','runtime_verified'] "
            "RETURN r{.*} AS rule, source.name AS source_api",
            name=api_name,
        )
        return {"available": True, "api_name": api_name, "knowledge_status": "known" if rows else "unknown", "rules": rows, "backend": "neo4j"}

    def validate_traffic_chain(self, api_name: str, relation_ids: Iterable[str] = ()) -> dict[str, Any]:
        ids = list(relation_ids)
        rows = self._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) WHERE (s.name=$name OR t.name=$name) "
            "AND ($ids = [] OR r.relation_id IN $ids) RETURN r.relation_id AS relation_id, r.confidence_status AS status",
            name=api_name, ids=ids,
        )
        proposed = [row["relation_id"] for row in rows if row.get("status") == "proposed"]
        valid = [row["relation_id"] for row in rows if row.get("status") in {"documented", "runtime_verified"}]
        return {"valid": not proposed, "api_name": api_name, "validated_relation_ids": valid, "proposed_relation_ids": proposed}

    def verify_relation(self, relation_id: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence:
            return {"ok": False, "error": "Verification requires execution evidence."}
        rows = self._run(
            "MATCH ()-[r:TRG_RELATION {relation_id:$relation_id}]->() "
            "SET r.confidence_status='runtime_verified', r.evidence_ids=$evidence_ids, r.version=coalesce(r.version,0)+1 "
            "RETURN r{.*} AS relation",
            relation_id=relation_id,
            evidence_ids=[str(item.get("evidence_id", "")) for item in evidence if item.get("evidence_id")],
        )
        return {"ok": bool(rows), "relation": rows[0].get("relation") if rows else None, "evidence_count": len(evidence)}

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._schema_ready = False


class TrafficRelationshipService:
    """Stable service boundary used by Agents and tools."""

    def __init__(self, store: TrafficGraphStore | None = None) -> None:
        self.store = store

    @classmethod
    def from_environment(cls) -> "TrafficRelationshipService":
        uri = os.getenv("SMP_TRG_NEO4J_URI", "").strip()
        if not uri:
            return cls(None)
        return cls(Neo4jTrafficGraphStore(
            uri,
            os.getenv("SMP_TRG_NEO4J_USER", "neo4j"),
            os.getenv("SMP_TRG_NEO4J_PASSWORD", ""),
            database=os.getenv("SMP_TRG_NEO4J_DATABASE", "neo4j"),
        ))

    @property
    def available(self) -> bool:
        return self.store is not None

    def unavailable(self, operation: str) -> dict[str, Any]:
        return {
            "available": False,
            "backend": "none",
            "operation": operation,
            "knowledge_status": "unknown",
            "error": "TRG backend is not configured; no relation is treated as verified.",
        }

    def call(self, method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.store is None:
            return self.unavailable(method)
        try:
            return getattr(self.store, method)(*args, **kwargs)
        except TrafficGraphUnavailable as exc:
            return {"available": False, "backend": "neo4j", "knowledge_status": "unknown", "error": str(exc)}


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
