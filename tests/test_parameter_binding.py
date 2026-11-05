from __future__ import annotations

from types import SimpleNamespace

from smart_traffic_agent.models import ExecutionContext, ExecutionPlan, NcProgramSpec, ApiCallLog, PlanStep
from smart_traffic_agent.parameter_rules import (
    apply_filter_operation,
    bind_plan_runtime_parameters,
    bind_runtime_parameter_rules,
    select_parameter_values,
    generate_parameter_combinations,
    resolve_parameter_combinations,
)
from smart_traffic_agent.traffic_graph import ApiNode, InMemoryTrafficGraphStore, TrafficRelation


def context_for(api: str, response: dict, status: int = 0) -> ExecutionContext:
    context = ExecutionContext()
    context.record(ApiCallLog(
        timestamp="t", task_id="task", step_id="S1", phase="Before",
        interface_name=api, protocol_function=api, input_parameters={},
        status_code=status, response=response, invocation_id="inv-1",
    ))
    return context


def context_for_many(rows: list[tuple[str, dict, int]]) -> ExecutionContext:
    context = ExecutionContext()
    for index, (api, response, status) in enumerate(rows, 1):
        context.record(ApiCallLog(
            timestamp="t", task_id="task", step_id=f"S{index}", phase="Before",
            interface_name=api, protocol_function=api, input_parameters={},
            status_code=status, response=response, invocation_id=f"inv-{index}",
        ))
    return context


def rule(mode: str, *, source_field: str = "values", transform: str = "", source_api: str = "source") -> dict:
    return {
        "rule_id": f"r-{mode}-{source_field}-{source_api}", "source_api": source_api,
        "dependency_path": "number", "source_field": source_field,
        "filter_mode": mode, "transform": transform,
    }


def test_include_intersects_domain():
    assert apply_filter_operation(range(1, 101), [1, 2, 3], "include")["domain"] == [1, 2, 3]


def test_parameter_selection_fixed_and_full_enum():
    assert select_parameter_values([5])["selected"] == [5]
    selected = select_parameter_values([1, 2, 3, 4, 5], full_enum_threshold=20)
    assert selected["strategy"] == "full_enum"
    assert selected["selected"] == [1, 2, 3, 4, 5]


def test_parameter_selection_samples_ranks_from_non_contiguous_domain():
    domain = list(range(1, 1001, 2))
    selected = select_parameter_values(domain, full_enum_threshold=20, middle_sample_count=5)
    assert selected["strategy"] == "representative_sample"
    assert selected["selected"][0] == domain[0]
    assert selected["selected"][-1] == domain[-1]
    assert set(selected["selected"]).issubset(set(domain))


def test_read_and_write_use_identical_selection_policy():
    for api_kind in ("read", "write"):
        selected = select_parameter_values(range(1, 101), full_enum_threshold=20, middle_sample_count=5)
        assert selected["strategy"] == "representative_sample", api_kind
        assert len(selected["selected"]) == 7
        assert selected["selected"][0] == 1
        assert selected["selected"][-1] == 100


def test_combination_generation_is_unique_and_constraint_aware():
    result = generate_parameter_combinations(
        "target", {"number": [1, 2], "axis": [1, 2]}, budget=10,
        constraints=(lambda row: row["number"] != 2 or row["axis"] == 1,),
    )
    assert result.combinations == [
        {"number": 1, "axis": 1}, {"number": 1, "axis": 2}, {"number": 2, "axis": 1}
    ]


def test_exclude_subtracts_domain():
    assert apply_filter_operation(range(1, 101), [1, 2, 3], "exclude")["domain"] == list(range(4, 101))


def test_unified_parameter_pipeline_defers_until_runtime_dependency_exists():
    pending = resolve_parameter_combinations(
        "cnc_accdecdly", {"axis": [1, 2, 3]},
        rules=[rule("include", source_field="sysinfo.axes", source_api="cnc_sysinfo")],
    )
    assert pending["status"] == "pending_dependencies"
    assert pending["combinations"] == []


def test_unified_parameter_pipeline_returns_filtered_combinations_after_upstream():
    ready = resolve_parameter_combinations(
        "target", {"number": list(range(1, 6))},
        execution_context=context_for("source", {"values": [1, 2, 3]}),
        rules=[rule("include")],
    )
    assert ready["status"] == "ready"
    assert ready["parameter_selections"]["number"]["selected"] == [1, 2, 3]
    assert ready["combinations"] == [{"number": 1}, {"number": 2}, {"number": 3}]


def test_multiple_filters_accumulate_in_order():
    context = context_for_many([("include_source", {"values": [1, 2, 3, 4, 5]}, 0), ("exclude_source", {"values": [2, 4]}, 0)])
    result = bind_runtime_parameter_rules(
        "target", context,
        [rule("include", source_api="include_source"), rule("exclude", source_api="exclude_source")],
        parameter_domains={"number": list(range(1, 11))},
    )
    assert result["selected_parameters"]["number"] == [1, 3, 5]


def test_axis_count_transform_is_deterministic():
    context = context_for("source", {"sysinfo": {"axes": "03"}})
    result = bind_runtime_parameter_rules(
        "target", context,
        [rule("include", source_field="source.sysinfo.axes", transform="active_axis_count_to_axis_domain")],
        parameter_domains={"number": [1, 2, 3, 4]},
    )
    assert result["success"] is True
    assert result["selected_parameters"]["number"] == [1, 2, 3]


def test_binding_distinguishes_failure_from_valid_empty_result():
    failed = bind_runtime_parameter_rules(
        "target", context_for("source", {"values": []}, status=5),
        [rule("exclude")], parameter_domains={"number": [1, 2, 3]},
    )
    assert failed["success"] is False
    assert failed["errors"][0]["code"] == "source_failed"

    empty = bind_runtime_parameter_rules(
        "target", context_for("source", {"values": []}),
        [rule("include")], parameter_domains={"number": [1, 2, 3]},
    )
    assert empty["success"] is True
    assert empty["empty_result"] is True
    assert empty["selected_parameters"]["number"] == []

    prefixed_empty = bind_runtime_parameter_rules(
        "target", context_for("source", {"values": []}),
        [rule("exclude", source_field="source.values")],
        parameter_domains={"number": [1, 2, 3]},
    )
    assert prefixed_empty["success"] is True
    assert prefixed_empty["selected_parameters"]["number"] == [1, 2, 3]


def test_plan_backfill_preserves_steps_and_records_evidence():
    step = PlanStep("T", "during", "target", "target", protocol_function="target", api="target")
    plan = ExecutionPlan(
        "p", "task", "scenario", "goal", "env", "test", [],
        NcProgramSpec("O1"), [step], [],
        runtime_bindings=[{
            "rule_id": "r-include-values-source", "target_api": "target", "target_parameter_path": "number",
            "source_step_id": "S", "source_field": "values", "filter_mode": "include",
        }],
    )
    result = bind_plan_runtime_parameters(
        plan, context_for("source", {"values": [2, 3]}),
        {"target": [rule("include")]},
        parameter_domains_by_target={"target": {"number": [1, 2, 3, 4]}},
    )
    assert result["success"] is True
    assert step.parameters["bound_parameters"]["number"] == [2, 3]
    assert plan.steps[0] is step
    assert plan.runtime_bindings[0]["status"] == "bound"


def test_plan_backfill_selects_representative_values_after_filtering():
    step = PlanStep("T", "during", "target", "target", protocol_function="target", api="target")
    plan = ExecutionPlan("p2", "task", "scenario", "goal", "env", "test", [], NcProgramSpec("O1"), [step], [])
    result = bind_plan_runtime_parameters(
        plan, context_for("source", {"values": list(range(1, 1001))}),
        {"target": [rule("include")]},
        parameter_domains_by_target={"target": {"number": list(range(1, 1001))}},
    )
    assert result["success"] is True
    assert step.parameters["bound_parameters"]["number"][0] == 1
    assert step.parameters["bound_parameters"]["number"][-1] == 1000
    assert len(step.parameters["bound_parameters"]["number"]) == 7
    assert len(result["results"]["target"]["legal_parameters"]["number"]) == 1000


def test_runtime_binding_can_be_stored_without_mutating_static_plan():
    step = PlanStep("T", "during", "target", "target", {}, protocol_function="target", api="target")
    plan = ExecutionPlan("p3", "task", "scenario", "goal", "env", "test", [], NcProgramSpec("O1"), [step], [])
    bind_plan_runtime_parameters(
        plan, context_for("source", {"values": [2, 3]}),
        {"target": [rule("include")]},
        parameter_domains_by_target={"target": {"number": [1, 2, 3, 4]}},
        update_plan=False,
    )
    assert "bound_parameters" not in step.parameters
    assert plan.rag_context["runtime_parameter_bindings"]["target"]["combinations"] == [
        {"number": 2}, {"number": 3}
    ]


def test_filters_relation_reads_direct_attributes_json_shape():
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode("source", "cnc_sysinfo"))
    store.upsert_api(ApiNode("target", "cnc_wrpm_item"))
    stored = store.upsert_relation(TrafficRelation(
        relation_id="rel-direct", source_api="cnc_sysinfo", target_api="cnc_wrpm_item",
        relation_type="filters", confidence_status="documented", evidence_ids=("ev-1",),
        attributes={
            "source_field": "sysinfo.axes",
            "dependency_path": "axis",
            "filter_mode": "include",
            "transform": "active_axis_count_to_axis_domain",
        },
    ))
    assert stored["ok"] is True
    result = store.get_parameter_rules("cnc_wrpm_item")
    assert len(result["rules"]) == 1
    parsed = result["rules"][0]
    assert parsed["source_api"] == "cnc_sysinfo"
    assert parsed["rule_id"] == "rel-direct"
    assert parsed["parameter"] == "axis"
    assert parsed["source_fields"] == ["sysinfo.axes"]
    assert parsed["filter_mode"] == "include"
