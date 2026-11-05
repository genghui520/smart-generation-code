import json
from pathlib import Path

from smart_traffic_agent.trg_audit import audit_graph, migrate_graph, validate_canonical_edges
from smart_traffic_agent.traffic_graph import ApiNode, InMemoryTrafficGraphStore, TrafficRelation


def test_audit_reports_legacy_types_missing_evidence_and_duplicates():
    store = InMemoryTrafficGraphStore()
    for name in ("a", "b"):
        store.upsert_api(ApiNode(f"focas:{name}", name))
    # The in-memory store normalizes aliases, so duplicate canonical facts are
    # intentional here; the audit still reports the duplicate semantic group.
    store.upsert_relation(TrafficRelation("r1", "a", "b", "filters"))
    store.upsert_relation(TrafficRelation("r2", "a", "b", "filters"))
    report = audit_graph(store)
    assert report["relations"] == 2
    assert report["missing_evidence_count"] == 2
    assert report["duplicate_group_count"] == 1


def test_migration_populates_structured_nodes_and_keeps_uncertain_edges_proposed():
    tmp_path = Path("runs/test_trg_audit_case")
    store = InMemoryTrafficGraphStore()
    for name in ("cnc_rdprogdir3", "cnc_copyprog", "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"):
        store.upsert_api(ApiNode(f"focas:{name}", name))
    store.upsert_relation(TrafficRelation(
        "directory", "cnc_rdprogdir3", "cnc_copyprog", "PROVIDES_INPUT",
        confidence_status="documented", evidence_ids=("old-record",),
    ))
    store.upsert_relation(TrafficRelation(
        "session", "cnc_dwnstart3", "cnc_download3", "SESSION_SEQUENCE",
        confidence_status="documented", evidence_ids=("old-record",),
    ))
    store.upsert_relation(TrafficRelation(
        "unknown", "cnc_rdprogdir3", "cnc_copyprog", "SESSION_SEQUENCE",
        confidence_status="documented", evidence_ids=(),
    ))
    report = migrate_graph(store, backup_dir=tmp_path, dry_run=False)
    assert Path(report["backup"]).is_file()
    target = store.nodes["cnc_download3"]
    assert set(target.prerequisites) == {
        "cnc_parameters", "pmc_signals", "cnc_mode", "cnc_state",
        "nc_program", "session", "options", "knowledge_status",
    }
    directory = store.relations["directory"]
    assert directory.confidence_status in {"documented", "sdk_validated"}
    assert directory.relation_type == "filters"
    assert directory.attributes["dependency_path"] in {"src_no", "dst_no"}
    session = store.relations["session"]
    assert session.relation_type == "prepares"
    assert session.attributes["dependency_path"] == "prerequisites.session"
    uncertain = store.relations["unknown"]
    assert uncertain.confidence_status == "proposed"
    assert validate_canonical_edges(store)["invalid_effective_edge_count"] == 0
    assert json.loads(Path(report["backup"]).read_text(encoding="utf-8"))["relations"]


def test_migration_is_idempotent_for_node_and_relation_counts():
    tmp_path = Path("runs/test_trg_audit_idempotent")
    store = InMemoryTrafficGraphStore()
    for name in ("cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"):
        store.upsert_api(ApiNode(f"focas:{name}", name))
    store.upsert_relation(TrafficRelation("session", "cnc_dwnstart3", "cnc_download3", "SESSION_SEQUENCE", confidence_status="documented"))
    first = migrate_graph(store, backup_dir=tmp_path / "one", dry_run=False)
    before = (len(store.nodes), len(store.relations), first["after"]["relations"])
    second = migrate_graph(store, backup_dir=tmp_path / "two", dry_run=False)
    assert (len(store.nodes), len(store.relations), second["after"]["relations"]) == before


def test_download_directory_number_cannot_exclude_raw_nc_data_buffer():
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode(
        "focas:cnc_rdprogdir3", "cnc_rdprogdir3",
        parameters={"buf": {"direction": "out", "type": "PRGDIR3", "fields": {
            "number": {"direction": "out", "type": "short"},
        }}},
    ))
    store.upsert_api(ApiNode(
        "focas:cnc_download4", "cnc_download4",
        parameters={"data": {"direction": "in", "type": "char *"}},
    ))
    store.upsert_relation(TrafficRelation(
        "bad-download-filter", "cnc_rdprogdir3", "cnc_download4", "filters",
        confidence_status="documented", evidence_ids=("test:bad",),
        attributes={"dependency_path": "data", "source_field": "buf.number", "filter_mode": "exclude"},
    ))
    report = migrate_graph(store, backup_dir=Path("runs/test_trg_download_semantics"), dry_run=False)
    bad = store.relations["bad-download-filter"]
    assert bad.confidence_status == "proposed"
    assert bad.attributes == {}
    good = [item for item in store.relations.values() if item.target_api == "cnc_download4" and item.relation_id.startswith("rel:trg_audit_contract_filters:")]
    assert good
    assert good[0].attributes["filter_mode"] == "include"
    assert good[0].attributes["source_field"] == "buf.number"
    assert good[0].attributes["transform"] == "construct_nc_program_with_leading_O_number"
    assert report["edge_validation"]["invalid_effective_edge_count"] == 0
