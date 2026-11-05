from __future__ import annotations

from smart_traffic_agent.execution_plan_schema import (
    execution_plan_view,
    normalize_execution_plan,
    validate_execution_plan,
)
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep


def make_plan(target: str, steps: list[PlanStep]) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=f"plan-{target}",
        task_id=f"task-{target}",
        scenario_type="api_capture",
        scenario_goal=f"capture {target}",
        target_environment="simulator",
        nc_program_type="not_required",
        nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name=""),
        steps=steps,
        expected_outputs=["api_logs"],
        target_api=target,
    )


def test_no_dependency_plan_has_explicit_target_scenario_and_collection_contract():
    plan = make_plan(
        "cnc_statinfo",
        [PlanStep("S001", "during", "read status", "cnc_statinfo", {}, protocol_function="cnc_statinfo")],
    )
    normalize_execution_plan(plan, {})
    view = execution_plan_view(plan)
    result = validate_execution_plan(plan)

    assert result["valid"]
    assert view["target_api"] == "cnc_statinfo"
    assert view["scenario"] == "api_capture"
    assert view["steps"][0]["role"] == "target"
    assert view["collection"]["npcap"]["group_by"] == ["batch_id", "api_name"]


def test_normalization_removes_error_probe_steps_when_operator_disables_them():
    plan = make_plan(
        "cnc_rdpm_item",
        [
            PlanStep("S001", "during", "read valid item", "cnc_rdpm_item", {}, protocol_function="cnc_rdpm_item"),
            PlanStep(
                "S009", "during",
                "Documented error-path probe",
                "cnc_rdpm_item",
                {"parameter_coverage_goal": {"coverage_scope": "single documented error branch"}},
                protocol_function="cnc_rdpm_item",
                expected_state="EW_LENGTH",
            ),
        ],
    )
    normalize_execution_plan(plan, {})
    assert [step.step_id for step in plan.steps] == ["S001"]
    assert plan.rag_context["cancelled_error_probes"][0]["step_id"] == "S009"


def test_normalization_removes_error_probe_marker_nested_in_api_call():
    plan = make_plan(
        "cnc_rdpm_item",
        [PlanStep("S010", "during", "probe", "cnc_rdpm_item", {},
                  protocol_function="cnc_rdpm_item",
                  api_calls=[{"call_id": "C1", "protocol_function": "cnc_rdpm_item",
                              "parameters": {"expected_return_code": 5}}])],
    )
    normalize_execution_plan(plan, {})
    assert plan.steps == []
    assert plan.rag_context["cancelled_error_probes"][0]["step_id"] == "S010"


def test_filters_dependency_exposes_runtime_source_and_target_binding():
    plan = make_plan(
        "cnc_setpath",
        [
            PlanStep("S001", "before", "discover path", "cnc_getpath", {}, protocol_function="cnc_getpath"),
            PlanStep("S002", "during", "set path", "cnc_setpath", {}, protocol_function="cnc_setpath"),
        ],
    )
    context = {
        "apis": {
            "cnc_getpath": {
                "node_contract": {"parameters": {"maxpath_no": {"direction": "out"}}, "prerequisites": {}}
            },
            "cnc_setpath": {
                "node_contract": {"parameters": {"path_no": {"direction": "in"}}, "prerequisites": {}}
            },
            "target": {},
            "dependency_plan": {},
        }
    }
    context["apis"]["cnc_setpath"]["dependency_plan"] = {
        "edges": [{
            "source_api": "cnc_getpath",
            "target_api": "cnc_setpath",
            "relation": {
                "relation_type": "filters",
                "confidence_status": "sdk_validated",
                "rule_id": "path:get->set",
                "evidence_ids": ["sdk:path"],
                "attributes": {
                    "dependency_path": "path_no",
                    "source_field": "maxpath_no",
                    "filter_mode": "include",
                },
            },
        }]
    }
    normalize_execution_plan(plan, context)
    result = validate_execution_plan(plan)

    assert result["valid"]
    assert plan.dependencies[0]["relation"] == "filters"
    assert plan.runtime_bindings[0]["source_step_id"] == "S001"
    assert plan.runtime_bindings[0]["source_field"] == "maxpath_no"
    assert plan.runtime_bindings[0]["target_parameter_path"] == "path_no"
    assert plan.runtime_bindings[0]["filter_mode"] == "include"


def test_prepares_multilevel_session_requires_start_target_end_in_one_version():
    plan = make_plan(
        "cnc_download4",
        [
            PlanStep("S001", "before", "start download", "cnc_dwnstart4", {}, protocol_function="cnc_dwnstart4"),
            PlanStep("S002", "during", "download data", "cnc_download4", {}, protocol_function="cnc_download4"),
            PlanStep("S003", "after", "end download", "cnc_dwnend4", {}, protocol_function="cnc_dwnend4"),
        ],
    )
    context = {
        "apis": {
            name: {"node_contract": {"parameters": {}, "prerequisites": {"session": [{}]}}}
            for name in ("cnc_dwnstart4", "cnc_download4", "cnc_dwnend4")
        }
    }
    context["apis"]["cnc_download4"]["dependency_plan"] = {
        "edges": [
            {"source_api": "cnc_dwnstart4", "target_api": "cnc_download4", "relation": {"relation_type": "prepares", "confidence_status": "documented", "attributes": {"dependency_path": "prerequisites.session"}}},
            {"source_api": "cnc_download4", "target_api": "cnc_dwnend4", "relation": {"relation_type": "prepares", "confidence_status": "documented", "attributes": {"dependency_path": "prerequisites.session"}}},
        ]
    }
    normalize_execution_plan(plan, context)
    result = validate_execution_plan(plan)

    assert result["valid"]
    assert [item["owner"] for item in plan.session["order"]] == ["start", "target", "end"]
    assert plan.session["version"] == "4"
    assert [step.role for step in plan.steps] == ["auxiliary", "target", "auxiliary"]
    assert all(step.failure_policy == "block_dependents" for step in plan.steps)


def test_download_filter_binding_requires_program_payload_transform():
    plan = make_plan(
        "cnc_download4",
        [
            PlanStep("S001", "before", "discover directory", "cnc_rdprogdir3", {}, protocol_function="cnc_rdprogdir3"),
            PlanStep("S002", "before", "start download", "cnc_dwnstart4", {}, protocol_function="cnc_dwnstart4"),
            PlanStep("S003", "during", "download data", "cnc_download4", {}, protocol_function="cnc_download4"),
            PlanStep("S004", "after", "end download", "cnc_dwnend4", {}, protocol_function="cnc_dwnend4"),
        ],
    )
    context = {"apis": {
        "cnc_rdprogdir3": {"node_contract": {"parameters": {"buf": {"direction": "out", "fields": {"number": {"direction": "out", "type": "short"}}}}, "prerequisites": {}}},
        "cnc_dwnstart4": {"node_contract": {"parameters": {}, "prerequisites": {"session": [{}]}}},
        "cnc_download4": {"node_contract": {"parameters": {"data": {"direction": "in", "type": "char *"}}, "prerequisites": {"session": [{}]}}},
        "cnc_dwnend4": {"node_contract": {"parameters": {}, "prerequisites": {"session": [{}]}}},
        "dependency_plan": {},
    }}
    context["apis"]["cnc_download4"]["dependency_plan"] = {"edges": [
        {"source_api": "cnc_rdprogdir3", "target_api": "cnc_download4", "relation": {"relation_type": "filters", "confidence_status": "documented", "attributes": {"dependency_path": "data", "source_field": "buf.number", "filter_mode": "include", "transform": "construct_nc_program_with_leading_O_number"}}},
        {"source_api": "cnc_dwnstart4", "target_api": "cnc_download4", "relation": {"relation_type": "prepares", "confidence_status": "documented", "attributes": {"dependency_path": "prerequisites.session"}}},
        {"source_api": "cnc_download4", "target_api": "cnc_dwnend4", "relation": {"relation_type": "prepares", "confidence_status": "documented", "attributes": {"dependency_path": "prerequisites.session"}}},
    ]}
    normalize_execution_plan(plan, context)
    result = validate_execution_plan(plan)
    assert result["valid"], result
    binding = next(item for item in plan.runtime_bindings if item["target_parameter_path"] == "data")
    assert binding["source_field"] == "buf.number"
    assert binding["transform"] == "construct_nc_program_with_leading_O_number"


def test_download_filter_binding_rejects_exclude_without_payload_transform():
    plan = make_plan(
        "cnc_download4",
        [PlanStep("S001", "before", "discover", "cnc_rdprogdir3", {}, protocol_function="cnc_rdprogdir3"), PlanStep("S002", "during", "download", "cnc_download4", {}, protocol_function="cnc_download4")],
    )
    plan.rag_context["traffic_relationships"] = {"apis": {
        "cnc_rdprogdir3": {"node_contract": {"parameters": {"buf": {"direction": "out", "fields": {"number": {"direction": "out", "type": "short"}}}}, "prerequisites": {}}},
        "cnc_download4": {"node_contract": {"parameters": {"data": {"direction": "in", "type": "char *"}}, "prerequisites": {}}},
    }}
    plan.dependencies = [{"source_api": "cnc_rdprogdir3", "target_api": "cnc_download4", "relation": "filters", "dependency_path": "data", "source_field": "buf.number", "filter_mode": "exclude"}]
    plan.runtime_bindings = [{"source_step_id": "S001", "source_field": "buf.number", "target_step_id": "S002", "target_parameter_path": "data", "filter_mode": "exclude"}]
    result = validate_execution_plan(plan)
    assert not result["valid"]
    assert any("directory program numbers" in error for error in result["errors"])
