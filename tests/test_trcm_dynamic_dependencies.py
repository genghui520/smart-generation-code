from smart_traffic_agent.traffic_graph import (
    ApiNode,
    InMemoryTrafficGraphStore,
    TrafficGraphError,
    TrafficRelation,
    TrafficRelationshipService,
)
from smart_traffic_agent.models import ApiCallLog
from smart_traffic_agent.agents.executor import build_execution_context
from smart_traffic_agent.parameter_rules import resolve_runtime_parameter_bindings


def test_recursive_filters_and_prepares_plan_omits_satisfied_nodes():
    store = InMemoryTrafficGraphStore()
    for name in ("discover", "prepare", "target"):
        store.upsert_api(ApiNode(f"focas:{name}", name))
    store.upsert_relation(TrafficRelation("r1", "discover", "prepare", "filters", confidence_status="documented"))
    store.upsert_relation(TrafficRelation("r2", "prepare", "target", "prepares", confidence_status="sdk_validated"))
    service = TrafficRelationshipService(store)

    plan = service.call("get_dependency_plan", "target")
    assert plan["required_calls"] == ["discover", "prepare"]
    assert {row["relation"]["relation_type"] for row in plan["edges"]} == {"filters", "prepares"}

    satisfied = service.call("get_dependency_plan", "target", satisfied=("discover",))
    assert satisfied["required_calls"] == ["prepare"]
    assert satisfied["satisfied_nodes"] == ["discover"]


def test_node_contract_is_preserved_as_structured_data():
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode(
        "focas:target", "target",
        parameters={"rules": [{"name": "number", "source_api": "discover"}]},
        prerequisites={"mode": "MEM", "signals": ["ready"]},
    ))
    node = TrafficRelationshipService(store).call("get_api_info", "target")["api"]
    assert node["parameters"]["rules"][0]["source_api"] == "discover"
    assert node["prerequisites"]["signals"] == ["ready"]


def test_legacy_relation_is_normalized_to_canonical_type():
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode("focas:a", "a"))
    store.upsert_api(ApiNode("focas:b", "b"))
    store.upsert_relation(TrafficRelation("legacy", "a", "b", "PROVIDES_INPUT", confidence_status="documented"))
    relation = next(iter(store.relations.values()))
    assert str(relation.relation_type) == "filters"
    assert TrafficRelationshipService(store).call("migrate_legacy_relations")["changed"] == 0


def test_proposed_dependency_is_candidate_but_never_scheduled():
    store = InMemoryTrafficGraphStore()
    for name in ("source", "target"):
        store.upsert_api(ApiNode(f"focas:{name}", name))
    store.upsert_relation(TrafficRelation("candidate", "source", "target", "prepares", confidence_status="proposed"))
    plan = store.get_dependency_plan("target")
    assert plan["required_calls"] == []
    assert plan["candidate_edges"][0]["source_api"] == "source"


def test_runtime_filters_use_only_successful_prerequisite_output():
    logs = [ApiCallLog(
        timestamp="t", task_id="task", step_id="s1", phase="Before",
        interface_name="ReadDirectory", protocol_function="cnc_rdprogdir3",
        input_parameters={}, status_code=0, response={"program_number": 123},
        invocation_id="i1",
    )]
    context = build_execution_context(logs)
    rules = [{
        "rule_id": "directory-program",
        "target_api": "cnc_upstart3",
        "parameter": "s_number",
        "source_api": "cnc_rdprogdir3",
        "source_fields": ["program_number"],
        "target_parameters": ["s_number"],
        "require_runtime_discovery": True,
    }]
    binding = resolve_runtime_parameter_bindings("cnc_upstart3", context, rules)
    assert binding["selected_parameters"] == {"s_number": [123]}


def test_runtime_filters_ignore_failed_prerequisite_response():
    logs = [
        ApiCallLog(timestamp="t", task_id="task", step_id="s1", phase="Before",
                   interface_name="ReadDirectory", protocol_function="cnc_rdprogdir3",
                   input_parameters={}, status_code=5, response={"program_number": 0}, invocation_id="bad"),
        ApiCallLog(timestamp="t", task_id="task", step_id="s2", phase="Before",
                   interface_name="ReadDirectory", protocol_function="cnc_rdprogdir3",
                   input_parameters={}, status_code=0, response={"program_number": 123}, invocation_id="good"),
    ]
    context = build_execution_context(logs)
    binding = resolve_runtime_parameter_bindings("cnc_upstart3", context, [{
        "target_api": "cnc_upstart3", "parameter": "s_number", "source_api": "cnc_rdprogdir3",
        "source_fields": ["program_number"], "require_runtime_discovery": True,
    }])
    assert binding["selected_parameters"] == {"s_number": [123]}


def test_configured_graph_error_is_reported_as_unavailable_without_repair_exception():
    class BrokenStore:
        def get_api_info(self, *_args, **_kwargs):
            raise TrafficGraphError("connection refused")

    result = TrafficRelationshipService(BrokenStore()).call("get_api_info", "cnc_statinfo")
    assert result["available"] is False
    assert result["backend"] == "neo4j"
    assert "connection refused" in result["error"]
