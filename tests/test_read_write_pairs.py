import json
from pathlib import Path

from smart_traffic_agent.read_write_pairs import discover_read_write_pairs, seed_read_write_pair_relations
from smart_traffic_agent.traffic_graph import ApiNode, InMemoryTrafficGraphStore, TrafficRelationshipService


def test_discover_read_write_pairs_excludes_parameter_transfer_markers():
    pairs, excluded = discover_read_write_pairs(Path("rag_indexes/focas/function_manifest.jsonl"))
    assert len(pairs) == 108
    assert {item["read_api"] for item in excluded} == {"pmc_rdprmstart", "pmc_rdprmend"}
    assert all(item.read_api.startswith(("cnc_", "pmc_", "ds_")) for item in pairs)


def test_seed_read_write_pair_is_optional_manual_only_prepares_edge():
    pairs, _ = discover_read_write_pairs(Path("rag_indexes/focas/function_manifest.jsonl"))
    pair = next(item for item in pairs if item.read_api == "cnc_rdmacro")
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode(api_id=f"focas:{pair.read_api}", name=pair.read_api))
    store.upsert_api(ApiNode(api_id=f"focas:{pair.write_api}", name=pair.write_api))
    result = seed_read_write_pair_relations(TrafficRelationshipService(store), [pair])
    assert result["counts"] == {"stored": 1, "already_present": 0, "unconfirmed": 0}
    relation = next(iter(store.relations.values()))
    assert relation.relation_type == "prepares"
    assert relation.confidence_status == "sdk_validated"
    assert relation.attributes["dependency_path"] == "prerequisites.readback"
    assert relation.attributes["auto_execute"] is False
    node = store.nodes[pair.read_api]
    assert node.prerequisites["readback"][0]["write_api"] == pair.write_api
