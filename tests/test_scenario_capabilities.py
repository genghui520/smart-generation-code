from pathlib import Path

from smart_traffic_agent.models import ApiCallLog
from smart_traffic_agent.scenario_capabilities import (
    analyze_output_dependencies,
    assess_scene_execution,
    build_cnc_skip_scene_steps,
    build_scene_capability_plan,
    build_variation_contract,
    classify_api_variation,
    load_scenario_capabilities,
)


def _log(function: str, data: str, *, status: int = 0, traffic: str = "matched") -> ApiCallLog:
    return ApiCallLog(
        timestamp="2026-10-24T00:00:00Z", task_id="scene-test", step_id="S1",
        phase="during", interface_name=function, protocol_function=function,
        input_parameters={}, status_code=status, response={"data": data},
        traffic_status=traffic,
    )


def test_registry_declares_unverified_scope_and_limits() -> None:
    rows = load_scenario_capabilities()
    ids = {row["capability_id"] for row in rows}
    assert {"input_parameter_variation", "program_directory_business_data",
            "nc_simulation_execution", "external_event_simulation",
            "device_state_observation"}.issubset(ids)
    event = next(row for row in rows if row["capability_id"] == "external_event_simulation")
    assert event["verified_in_current_simulator"] is False
    assert "do not write PMC/system parameters" in event["execution_limits"]


def test_cnc_skip_uses_documented_axis_variation_without_event_execution() -> None:
    analysis = analyze_output_dependencies("cnc_skip")
    plan = build_scene_capability_plan("cnc_skip")
    assert analysis["dependency_kind"] == "input_parameter_variation"
    assert analysis["variation_requires_nc_program"] is False
    assert analysis["variation_requires_cycle_start"] is False
    assert plan["nc_program_required"] is False
    assert plan["actual_execution_required"] is False
    assert plan["execution_policy"] == "read_only_axis_input_variation"
    assert plan["dynamic_event_execution_allowed"] is False
    assert plan["auxiliary_apis"] == ["cnc_rdaxisname"]
    assert plan["nc_execution_decision"] == {
        "program_required": False,
        "execution_required": False,
        "cycle_start_required": False,
        "cycle_start_allowed": False,
        "requirements_source": "no NC execution for axis-input variation",
    }
    assert [step.protocol_function for step in build_cnc_skip_scene_steps()] == [
        "cnc_statinfo", "cnc_skip", "cnc_statinfo"
    ]


def test_dependency_classes_choose_low_cost_scene_without_forcing_nc() -> None:
    macro = analyze_output_dependencies("cnc_rdmacro")
    directory = analyze_output_dependencies("cnc_rdprogdir3")
    transfer = analyze_output_dependencies("cnc_download3")
    runtime = analyze_output_dependencies("cnc_statinfo")
    assert macro["event_capability"] == "controlled_business_data"
    assert directory["event_capability"] == "program_directory_business_data"
    assert transfer["event_capability"] == "program_transfer_and_directory_change"
    assert runtime["event_capability"] == "device_state_observation"
    assert build_scene_capability_plan("cnc_rdprogdir3")["nc_program_required"] is False


def test_variation_requirements_split_input_and_response_state_obligations() -> None:
    skip = classify_api_variation("cnc_skip")
    assert skip["variation_class"] == "input_only"
    assert skip["input_diversity"] == {
        "applicable": True,
        "required": True,
        "dimension": "axis",
        "reason": "documented ALL_AXES and controller-reported 1..m axis selectors",
    }
    assert skip["response_state_diversity"]["required"] is False

    servo = classify_api_variation("cnc_srvdelay")
    assert servo["variation_class"] == "input_and_response"
    assert servo["input_diversity"]["required"] is True
    assert servo["response_state_diversity"]["required"] is True

    status = classify_api_variation("cnc_statinfo")
    assert status["variation_class"] == "call_only"
    assert status["input_diversity"]["required"] is False
    assert status["response_state_diversity"]["required"] is False


def test_variation_contract_explains_input_response_and_state_evidence() -> None:
    contract = build_variation_contract(
        ["pmc_rdpmcrng", "cnc_statinfo", "cnc_accdecdly"],
        task_description="采集输入多样性和输出变化",
    )
    pmc = contract["targets"]["pmc_rdpmcrng"]
    assert pmc["input_diversity"]["required"] is True
    assert "documented API inputs" in pmc["input_diversity"]["source"]
    assert "input signatures" in pmc["input_diversity"]["success_metric"]
    status = contract["targets"]["cnc_statinfo"]
    assert status["input_diversity"]["required"] is False
    assert status["state_diversity"]["acquisition"]
    acc = contract["targets"]["cnc_accdecdly"]
    assert acc["input_diversity"]["required"] is True
    assert acc["response_diversity"]["required"] is False
    assert acc["state_diversity"]["required"] is False


def test_write_diversity_is_visible_but_blocked_without_permission() -> None:
    contract = build_variation_contract(
        ["pmc_wrpmcrng"], task_description="需要输入多样性采集", permissions={}
    )
    row = contract["targets"]["pmc_wrpmcrng"]
    assert row["input_diversity"]["required"] is True
    assert row["execution_gate"] == "blocked_until_explicit_permission"
    assert "allow_environment_writes" in row["input_diversity"]["blocked_reason"]


def test_dynamic_state_and_transfer_plans_expose_required_scene_helpers() -> None:
    position = build_scene_capability_plan("cnc_rdposition")
    assert position["scene_type"] == "runtime_motion_state"
    assert position["nc_program_required"] is True
    assert position["auxiliary_apis"] == ["cnc_statinfo"]
    assert position["execution_policy"] == "plan_only_until_nc_simulation_capability_is_verified"
    assert position["nc_execution_decision"]["cycle_start_required"] is True
    assert position["operator_interaction"]["cycle_start_action"] == "operator_manual_only_after_all_execution_gates_pass"

    transfer = build_scene_capability_plan("cnc_download3")
    assert transfer["auxiliary_apis"] == ["cnc_rdprogdir3", "cnc_dwnstart3", "cnc_dwnend3"]
    assert transfer["required_auxiliary_apis"] == ["cnc_dwnstart3", "cnc_dwnend3"]
    assert transfer["optional_auxiliary_apis"] == ["cnc_rdprogdir3"]
    assert transfer["session_chain"] == ["cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"]
    assert transfer["data_repeatable"] is True
    assert transfer["nc_program_required"] is None

    transfer4 = build_scene_capability_plan("cnc_download4")
    assert transfer4["selected_capability"]["transfer_version"] == "4"
    assert transfer4["selected_capability"]["tools_or_apis"] == [
        "cnc_rdprogdir3", "cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"
    ]
    assert transfer4["selected_capability"]["session_chain"] == [
        "cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"
    ]

    upload = build_scene_capability_plan("cnc_upload3")
    assert upload["transfer_direction"] == "upload_existing"
    assert upload["required_auxiliary_apis"] == ["cnc_upstart3", "cnc_upend3"]
    assert upload["optional_auxiliary_apis"] == ["cnc_rdprogdir3"]
    assert upload["session_chain"] == ["cnc_upstart3", "cnc_upload3", "cnc_upend3"]


def test_scene_plan_carries_mode_parameter_and_operator_gates() -> None:
    plan = build_scene_capability_plan("cnc_wrparam")
    preflight = plan["environment_preflight"]
    assert preflight["mode_observation_api"] == "cnc_statinfo"
    assert preflight["mode_mismatch_policy"] == "pause_for_operator; do_not_switch_mode"
    assert preflight["parameter_changes"] == []
    assert plan["operator_interaction"]["target_calls_blocked_while_waiting"] is True
    assert plan["nc_execution_decision"]["cycle_start_allowed"] is False


def test_scene_evidence_counts_skip_read_without_inventing_event_requirement() -> None:
    plan = build_scene_capability_plan("cnc_skip")
    evidence = assess_scene_execution(plan, [_log("cnc_skip", "type=0;data0=12")])
    assert evidence["target_called"] is True
    assert evidence["target_call_succeeded"] is True
    assert evidence["skip_event_verified"] is None
    assert evidence["scene_execution_succeeded"] is True
    assert evidence["response_changed"] is False
