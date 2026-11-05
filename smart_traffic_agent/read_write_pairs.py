"""Evidence-backed read/write API pair discovery and graph seeding.

The naming convention is useful for inventorying FOCAS APIs, but it is not
enough to authorize a write.  Persisted pair edges therefore describe an
optional write-then-read verification workflow and remain manual-only.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .focas_contracts import _api_docs
from .traffic_graph import ApiNode, TrafficRelation, TrafficRelationshipService, relation_id


@dataclass(frozen=True, slots=True)
class ReadWritePair:
    read_api: str
    write_api: str
    category: str
    read_description: str
    write_description: str
    evidence_ids: tuple[str, ...]

    @property
    def rule_id(self) -> str:
        return f"readback_pair:{self.write_api}->{self.read_api}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"rule_id": self.rule_id}


def _manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(row.get("function", "")).strip()
            if name:
                rows.setdefault(name, row)
    return rows


def _doc_text(function_name: str, kind: str) -> str:
    return str(_api_docs().get(function_name, {}).get(kind, "")).strip()


def _description(row: dict[str, Any], function_name: str) -> str:
    value = str(row.get("description", "")).strip()
    if value:
        return value
    overview = _doc_text(function_name, "overview")
    purpose = overview.split("Purpose:", 1)[-1].strip() if "Purpose:" in overview else overview
    return " ".join(purpose.split())[:300]


def discover_read_write_pairs(
    manifest_path: Path = Path("rag_indexes/focas/function_manifest.jsonl"),
) -> tuple[list[ReadWritePair], list[dict[str, str]]]:
    """Return documented ``rd...``/``wr...`` pairs and excluded session pairs."""
    rows = _manifest_rows(manifest_path)
    pairs: list[ReadWritePair] = []
    excluded: list[dict[str, str]] = []
    names = set(rows)
    for read_api in sorted(names):
        match = re.match(r"^(?P<prefix>(?:cnc|pmc|ds)_)rd(?P<suffix>.+)$", read_api)
        if not match:
            continue
        write_api = match.group("prefix") + "wr" + match.group("suffix")
        if write_api not in names:
            continue
        read_description = _description(rows[read_api], read_api)
        write_description = _description(rows[write_api], write_api)
        combined = f"{read_description} {write_description}".casefold()
        # PMC parameter start/end calls are upload/download session markers;
        # they are not readback APIs for the data written by the paired call.
        if (
            "upload" in combined
            and "download" in combined
            and ("start" in combined or "end" in combined)
        ):
            excluded.append({
                "read_api": read_api,
                "write_api": write_api,
                "reason": "upload/download session marker, not data readback",
            })
            continue
        category = str(rows[read_api].get("category", "")).strip() or "uncategorized"
        evidence_ids = tuple(dict.fromkeys((
            f"manifest:{read_api}",
            f"manifest:{write_api}",
            f"focas-{read_api}-overview",
            f"focas-{write_api}-overview",
            f"focas-{read_api}-arguments",
            f"focas-{write_api}-arguments",
        )))
        pairs.append(ReadWritePair(
            read_api=read_api,
            write_api=write_api,
            category=category,
            read_description=read_description,
            write_description=write_description,
            evidence_ids=evidence_ids,
        ))
    return pairs, excluded


def _node_rows(store: Any, names: Iterable[str]) -> dict[str, dict[str, Any]]:
    requested = sorted(set(str(name) for name in names if str(name).strip()))
    if hasattr(store, "_run"):
        rows = store._run(
            "MATCH (n:API) WHERE n.name IN $names "
            "RETURN n.name AS name, n.api_id AS api_id, n.protocol AS protocol, "
            "n.sdk_signature_ref AS sdk_signature_ref, n.description AS description, "
            "n.metadata_json AS metadata_json, n.parameters_json AS parameters_json, "
            "n.prerequisites_json AS prerequisites_json",
            names=requested,
        )
        return {str(row.get("name", "")): dict(row) for row in rows}
    output: dict[str, dict[str, Any]] = {}
    for name in requested:
        node = getattr(store, "nodes", {}).get(name)
        if node is not None:
            output[name] = {
                "name": node.name,
                "api_id": node.api_id,
                "protocol": node.protocol,
                "sdk_signature_ref": node.sdk_signature_ref,
                "description": node.description,
                "metadata": node.metadata,
                "parameters": node.parameters,
                "prerequisites": node.prerequisites,
            }
    return output


def _decode_json(value: Any, fallback: Any) -> Any:
    if isinstance(value, type(fallback)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback
    return fallback


def _readback_prerequisites(node: dict[str, Any]) -> dict[str, Any]:
    raw = node.get("prerequisites", node.get("prerequisites_json", {}))
    value = _decode_json(raw, {})
    return dict(value) if isinstance(value, dict) else {}


def _write_prerequisites(store: Any, api_name: str, prerequisites: dict[str, Any]) -> bool:
    encoded = json.dumps(prerequisites, ensure_ascii=False, sort_keys=True)
    if hasattr(store, "_run"):
        rows = store._run(
            "MATCH (n:API {name:$name}) SET n.prerequisites_json=$value "
            "RETURN n.name AS name",
            name=api_name,
            value=encoded,
        )
        return bool(rows)
    node = getattr(store, "nodes", {}).get(api_name)
    if node is None:
        return False
    store.nodes[api_name] = ApiNode(
        api_id=node.api_id,
        name=node.name,
        protocol=node.protocol,
        sdk_signature_ref=node.sdk_signature_ref,
        description=node.description,
        metadata=dict(node.metadata),
        parameters=dict(node.parameters),
        prerequisites=prerequisites,
    )
    store.nodes[node.api_id] = store.nodes[api_name]
    return True


def seed_read_write_pair_relations(
    service: TrafficRelationshipService,
    pairs: Iterable[ReadWritePair],
    *,
    device_scope: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Seed optional, manual-only write -> readback ``prepares`` edges."""
    if not service.available or service.store is None:
        return {"stored": [], "already_present": [], "unconfirmed": [{"status": "unconfirmed", "reason": "Neo4j unavailable"}]}
    store = service.store
    pair_list = list(pairs)
    nodes = _node_rows(store, [name for pair in pair_list for name in (pair.read_api, pair.write_api)])
    output: dict[str, list[dict[str, Any]]] = {"stored": [], "already_present": [], "unconfirmed": []}
    for pair in pair_list:
        missing = [name for name in (pair.read_api, pair.write_api) if name not in nodes]
        if missing:
            output["unconfirmed"].append({**pair.as_dict(), "status": "unconfirmed", "reason": f"missing API nodes: {', '.join(missing)}"})
            continue
        relation_id_value = relation_id(pair.write_api, pair.read_api, "prepares", pair.rule_id)
        existing = []
        if hasattr(store, "_run"):
            existing = store._run(
                "MATCH (s:API)-[r:TRG_RELATION {relation_id:$relation_id}]->(t:API) "
                "RETURN r{.*} AS relation",
                relation_id=relation_id_value,
            )
        else:
            item = getattr(store, "relations", {}).get(relation_id_value)
            existing = [{"relation": {"confidence_status": item.confidence_status}}] if item else []
        if existing:
            output["already_present"].append({**pair.as_dict(), "status": "already_present", "relation_id": relation_id_value})
            continue
        applicability = {
            "scope": "documented_rd_wr_api_pair",
            "required": False,
            "verification": "write_then_readback",
            **({"device_scope": device_scope} if device_scope else {}),
        }
        attributes = {
            "dependency_path": "prerequisites.readback",
            "pair_role": "write_then_readback",
            "read_api": pair.read_api,
            "write_api": pair.write_api,
            "verification": "compare the documented data fields after a successful write",
            "dangerous_operation": True,
            "auto_execute": False,
            "requires_read_snapshot": True,
            "requires_readback": True,
        }
        relation = TrafficRelation(
            relation_id=relation_id_value,
            source_api=pair.write_api,
            target_api=pair.read_api,
            relation_type="prepares",
            rule_id=pair.rule_id,
            confidence_status="sdk_validated",
            evidence_ids=pair.evidence_ids,
            applicability=applicability,
            attributes=attributes,
        )
        readback_entry = {
            "write_api": pair.write_api,
            "read_api": pair.read_api,
            "relation_type": "prepares",
            "rule_id": pair.rule_id,
            "verification": "compare the documented data fields after a successful write",
            "required": False,
            "dangerous_operation": True,
            "auto_execute": False,
            "evidence_ids": list(pair.evidence_ids),
        }
        prerequisites = _readback_prerequisites(nodes[pair.read_api])
        entries = prerequisites.setdefault("readback", [])
        if not isinstance(entries, list):
            entries = prerequisites["readback"] = []
        if not any(isinstance(item, dict) and item.get("rule_id") == pair.rule_id for item in entries):
            entries.append(readback_entry)
        if dry_run:
            output["stored"].append({**pair.as_dict(), "status": "would_store", "relation_id": relation_id_value})
            continue
        result = service.call("upsert_relation", relation)
        if not result.get("ok") or not result.get("stored", True):
            output["unconfirmed"].append({**pair.as_dict(), "status": "unconfirmed", "reason": result.get("reason", "relation rejected")})
            continue
        if not _write_prerequisites(store, pair.read_api, prerequisites):
            output["unconfirmed"].append({**pair.as_dict(), "status": "unconfirmed", "reason": "readback prerequisite could not be persisted"})
            continue
        output["stored"].append({**pair.as_dict(), "status": "stored", "relation_id": relation_id_value})
    output["counts"] = {key: len(value) for key, value in output.items() if isinstance(value, list)}
    return output
