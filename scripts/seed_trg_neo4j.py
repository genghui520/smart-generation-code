"""Seed the Neo4j TRG from the FOCAS API manifest.

The script is intentionally data-driven and safe to rerun. Credentials are
read from SMP_TRG_NEO4J_* environment variables; no password is accepted on
the command line or written to the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow ``python scripts/seed_trg_neo4j.py`` from the repository root.  When
# Python executes a file directly it puts ``scripts`` (rather than the
# repository root) on sys.path, which otherwise makes the local package
# imports fail before the seed can connect to Neo4j.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart_traffic_agent.api_prerequisites import load_manifest_rows
from smart_traffic_agent.api_compatibility import F0ID_MODEL, F0ID_SCOPE_ID, f0id_api_names
from smart_traffic_agent.traffic_graph import ApiNode, TrafficRelation, TrafficRelationshipService, relation_id
from smart_traffic_agent.traffic_relationship_mining import seed_documented_relations
from smart_traffic_agent.parameter_relationship_rules import resolve_rules
from smart_traffic_agent.cnc_parameter_prerequisites import extract_f0id_parameter_prerequisites


def _manifest_index(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("function", "")).strip()
        if not name:
            continue
        # Keep the first manifest row for stable incremental updates.
        metadata.setdefault(name, {
            "manifest_id": str(row.get("manifest_id", "")),
            "category": str(row.get("category", "")),
            "function_family": str(row.get("function_family", "")),
            "coverage_status": str(row.get("coverage_status", "")),
            "source_layout": str(row.get("source_layout", "")),
            "source_workbook": str(row.get("source_workbook", "")),
            "source_url": str(row.get("source_url", "")),
        })
        metadata[name].setdefault("description", str(row.get("description", "")))
    return sorted(metadata), metadata


def _stats(service: TrafficRelationshipService, *, allowed_api_names: set[str] | None = None) -> dict[str, int]:
    store = service.store
    if store is None or not hasattr(store, "_run"):
        return {"api_nodes": 0, "relations": 0}
    if allowed_api_names is None:
        nodes = store._run("MATCH (n:API) RETURN count(n) AS count")
        relations = store._run("MATCH ()-[r:TRG_RELATION]->() RETURN count(r) AS count")
        effective_relations = store._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) "
            "RETURN count(DISTINCT [s.name,t.name,r.relation_type,r.rule_id]) AS count"
        )
        candidates = store._run("MATCH ()-[r:TRG_RELATION {confidence_status:'proposed'}]->() RETURN count(r) AS count")
        isolated = store._run("MATCH (n:API) WHERE NOT (n)-[:TRG_RELATION]-() RETURN count(n) AS count")
    else:
        names = sorted(allowed_api_names)
        nodes = store._run("MATCH (n:API) WHERE n.name IN $names RETURN count(n) AS count", names=names)
        relations = store._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) WHERE s.name IN $names AND t.name IN $names RETURN count(r) AS count",
            names=names,
        )
        effective_relations = store._run(
            "MATCH (s:API)-[r:TRG_RELATION]->(t:API) WHERE s.name IN $names AND t.name IN $names "
            "RETURN count(DISTINCT [s.name,t.name,r.relation_type,r.rule_id]) AS count",
            names=names,
        )
        candidates = store._run(
            "MATCH (s:API)-[r:TRG_RELATION {confidence_status:'proposed'}]->(t:API) WHERE s.name IN $names AND t.name IN $names RETURN count(r) AS count",
            names=names,
        )
        isolated = store._run(
            "MATCH (n:API) WHERE n.name IN $names AND NOT (n)-[:TRG_RELATION]-() RETURN count(n) AS count",
            names=names,
        )
    return {
        "api_nodes": int(nodes[0]["count"]) if nodes else 0,
        "relations": int(relations[0]["count"]) if relations else 0,
        "effective_relations": int(effective_relations[0]["count"]) if effective_relations else 0,
        "candidate_relations": int(candidates[0]["count"]) if candidates else 0,
        "isolated_api_nodes": int(isolated[0]["count"]) if isolated else 0,
    }


RDNGRP_PREREQUISITES: dict[str, Any] = {
    "tool_registration": [{
        "requirement": "tool_life_management_enabled",
        "condition": "Tool Life Management option is enabled",
        "prepare_api": "cnc_instlifedt",
        "dangerous_operation": True,
        "auto_execute": False,
        "requires_isolated_simulator_write_permission": True,
        "values_source": "runtime_readback_or_explicit_graph_rag_evidence",
    }, {
        "requirement": "at_least_one_registered_tool_group",
        "verify_apis": ["cnc_rdngrp", "cnc_rdntool", "cnc_rdgrpid"],
        "success_condition": "cnc_rdngrp.data >= 1 and a tool/group readback succeeds",
    }],
    "variation": [{
        "requirement": "tool_life_variation",
        "states": "distinct registered tool groups or registration states",
        "success_condition": "at least two successful registration/output states with distinct cnc_rdngrp.data",
        "evidence_required": ["registration_success_or_verified_existing_group", "target_read_success"],
    }],
    "nc_program": [{
        "requirement": "tool_life_variation_running",
        "requirements": [
            "NC program must reference a verified registered tool number or tool group with a T code",
            "NC program must be uploaded, selected, and verified before Cycle Start",
            "Cycle Start is allowed only in an isolated NCGuide simulator with explicit permission",
            "cnc_statinfo.run must prove the program is executing before dynamic cnc_rdngrp samples count",
            "only successful registration, NC execution, and target reads count as valid diversity traffic",
        ],
        "registration_api": "cnc_instlifedt",
        "verification_apis": ["cnc_rdngrp", "cnc_rdntool", "cnc_rdgrpid"],
        "run_gate_api": "cnc_statinfo",
        "run_gate_field": "run",
        "variation_samples": ["registered_group_state_A", "registered_group_state_B"],
    }],
}


def _rdngrp_relation() -> TrafficRelation:
    return TrafficRelation(
        relation_id=relation_id("cnc_instlifedt", "cnc_rdngrp", "prepares", "FANUC-tool-life-registration-for-rdngrp"),
        source_api="cnc_instlifedt",
        target_api="cnc_rdngrp",
        relation_type="prepares",
        rule_id="FANUC-tool-life-registration-for-rdngrp",
        confidence_status="documented",
        evidence_ids=("focas-cnc_instlifedt-overview", "focas-cnc_rdngrp-overview"),
        applicability={"scope": "FANUC 0i-D", "operation": "tool_life_management"},
        attributes={
            "dependency_path": "prerequisites.tool_registration",
            "source_field": "cnc_instlifedt.success",
            "verification_apis": ["cnc_rdngrp", "cnc_rdntool", "cnc_rdgrpid"],
            "dangerous_operation": True,
            "auto_execute": False,
            "requires_isolated_simulator_write_permission": True,
            "verify_after_prepare": True,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("rag_indexes/focas/function_manifest.jsonl"),
    )
    parser.add_argument("--scope", choices=("all", "f0id"), default="all")
    args = parser.parse_args()
    rows = load_manifest_rows(args.manifest)
    names, metadata = _manifest_index(rows)
    scope_names = set(f0id_api_names(rows)) if args.scope == "f0id" else None
    if not names:
        raise SystemExit(f"Manifest contains no API functions: {args.manifest}")

    service = TrafficRelationshipService.from_environment()
    if not service.available:
        raise SystemExit(
            "Neo4j is not configured. Set SMP_TRG_NEO4J_URI, "
            "SMP_TRG_NEO4J_USER and SMP_TRG_NEO4J_PASSWORD."
        )
    try:
        for name in names:
            row = metadata[name]
            prerequisites = {}
            if name == "cnc_rdopmsg":
                prerequisites = {
                    "nc_program": [{
                        "requirement": "running",
                        "requirements": [
                            "NC program must generate an external operator message before cnc_rdopmsg",
                            "NC program must be uploaded, selected, and actually executed to the message block",
                            "read cnc_rdopmsg after execution and record datano/type/char_num/data",
                        ],
                        "message_source": "NC macro/operator-message event",
                        "variation_samples": ["MESSAGE_A", "MESSAGE_B"],
                    }],
                }
            if name == "cnc_rdngrp":
                prerequisites = json.loads(json.dumps(RDNGRP_PREREQUISITES))
            service.call(
                "upsert_api",
                ApiNode(
                    api_id=f"focas:{name}",
                    name=name,
                    protocol="FOCAS",
                    sdk_signature_ref=str(row.get("manifest_id", "")),
                    description=str(row.get("description", ""))[:500],
                    metadata=row,
                    prerequisites=prerequisites,
                ),
            )

        # Target-only seeds still need the helper nodes and the documented
        # registration edge so Planner can obtain a complete dependency plan.
        if "cnc_rdngrp" in names:
            for helper in ("cnc_instlifedt", "cnc_rdntool", "cnc_rdgrpid", "cnc_statinfo"):
                row = metadata.get(helper, {})
                service.call(
                    "upsert_api",
                    ApiNode(
                        api_id=f"focas:{helper}", name=helper, protocol="FOCAS",
                        sdk_signature_ref=str(row.get("manifest_id", "")),
                        description=str(row.get("description", ""))[:500], metadata=row,
                    ),
                )
            service.call("upsert_relation", _rdngrp_relation())

        # This function is the existing authoritative migration path for
        # documented SDK interaction and transfer-session rules.
        relation_results = seed_documented_relations(
            service,
            scope_names if scope_names is not None else names,
            api_metadata=metadata,
            allowed_api_names=scope_names,
            device_scope=F0ID_SCOPE_ID if scope_names is not None else "",
        )
        parameter_report = resolve_rules(
            scope_names if scope_names is not None else names,
            allowed_api_names=scope_names,
            device_scope=F0ID_SCOPE_ID if scope_names is not None else "",
        )
        prerequisite_report = extract_f0id_parameter_prerequisites(
            scope_names if scope_names is not None else names,
            allowed_api_names=scope_names,
            device_scope=F0ID_SCOPE_ID if scope_names is not None else "",
        )
        global_stats = _stats(service)
        scoped_stats = _stats(service, allowed_api_names=scope_names)
        print(json.dumps({
            "manifest": str(args.manifest),
            "manifest_rows": len(rows),
            "unique_api_names": len(names),
            "scope": args.scope,
            "device_model": F0ID_MODEL if scope_names is not None else "",
            "scoped_api_names": len(scope_names) if scope_names is not None else len(names),
            "documented_relation_upserts": len(relation_results),
            "parameter_discovery_relation_upserts": len(parameter_report["relations"]),
            "parameter_discovery_scanned_api_count": parameter_report["scanned_api_count"],
            "parameter_discovery_analyzed_api_count": parameter_report["analyzed_api_count"],
            "parameter_discovery_candidate_count": parameter_report["candidate_count"],
            "parameter_discovery_relation_types": {
                relation_type: sum(1 for item in parameter_report["relations"] if item["relation_type"] == relation_type)
                for relation_type in sorted({item["relation_type"] for item in parameter_report["relations"]})
            },
            "cnc_parameter_prerequisite_scanned_api_count": prerequisite_report["scanned_api_count"],
            "cnc_parameter_prerequisite_analyzed_api_count": prerequisite_report["analyzed_api_count"],
            "cnc_parameter_prerequisite_condition_count": sum(
                1 for item in prerequisite_report["relations"]
                if item.get("relation_type") == "REQUIRES"
            ),
            "cnc_parameter_prerequisite_relation_count": len(prerequisite_report["relations"]),
            "cnc_parameter_prerequisite_candidate_count": prerequisite_report["candidate_count"],
            "cnc_parameter_prerequisite_relation_types": {
                relation_type: sum(1 for item in prerequisite_report["relations"] if item.get("relation_type") == relation_type)
                for relation_type in sorted({str(item.get("relation_type", "")) for item in prerequisite_report["relations"]})
            },
            "cnc_parameter_8132_relations": sum(
                1 for item in prerequisite_report["relations"]
                if int(item.get("attributes", {}).get("parameter_number", -1)) == 8132
            ),
            "global_api_nodes": global_stats["api_nodes"],
            "global_relations": global_stats["relations"],
            "scoped_relations": scoped_stats["relations"],
            "scoped_effective_relations": scoped_stats["effective_relations"],
            "scoped_candidate_relations": scoped_stats["candidate_relations"],
            "scoped_isolated_api_nodes": scoped_stats["isolated_api_nodes"],
        }, ensure_ascii=False, sort_keys=True))
    finally:
        store = service.store
        if store is not None and hasattr(store, "close"):
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
