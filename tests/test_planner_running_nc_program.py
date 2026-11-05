from smart_traffic_agent.agents.planner import (
    apply_running_nc_program_rule,
    build_accdecdly_running_steps,
    extract_nc_program_requirement,
    extract_nc_program_requirements,
)
from smart_traffic_agent.nc_program_rules import axis_domain_from_count
from smart_traffic_agent.execution_contracts import motion_required_for_targets
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec


def test_running_requirement_is_read_from_existing_prerequisites_shape():
    prerequisites = {
        "nc_program": [{"program_number": None, "requirement": "running"}],
    }
    assert extract_nc_program_requirement(prerequisites) == "running"
    assert extract_nc_program_requirement({"nc_program": [{"requirement": "exists"}]}) == "exists"
    assert extract_nc_program_requirement({"nc_program": [{"requirement": "not_exists"}]}) == "not_exists"
    assert extract_nc_program_requirement({}) is None


def test_nc_program_requirements_are_read_from_existing_prerequisites_shape():
    prerequisites = {
        "nc_program": [{
            "requirement": "running",
            "requirements": ["NC 程序必须包含加减速变化阶段"],
        }],
    }
    assert extract_nc_program_requirements(prerequisites) == ["NC 程序必须包含加减速变化阶段"]


def test_spindle_measurements_request_running_scene_for_variation():
    assert motion_required_for_targets(["cnc_rdspmeter"])
    assert motion_required_for_targets(["cnc_rdspload"])
    assert motion_required_for_targets(["cnc_rdspeed"])


def test_accdecdly_running_steps_use_sysinfo_axis_binding_and_order():
    assert axis_domain_from_count(3) == [1, 2, 3]
    steps = build_accdecdly_running_steps()
    interfaces = [step.interface_name for step in steps]
    assert interfaces[:6] == [
        "UploadProgram", "SelectProgram", "ReadProgramNumber",
        "ReadRunStatus", "ReadSystemInfo", "ReadAccelerationDecelerationDelay",
    ]
    assert "StartProgram" in interfaces
    assert interfaces.index("StartProgram") < interfaces.index("ReadAccelerationDecelerationDelay", 6)
    assert interfaces[-1] == "ReadAccelerationDecelerationDelay"
    assert all(step.protocol_function != "cnc_rdaxisname" for step in steps)
    samples = [step for step in steps if step.protocol_function == "cnc_accdecdly"]
    assert len(samples) == 3
    assert samples[0].phase == "before"
    assert samples[1].phase == "during"
    assert samples[2].phase == "after"
    assert samples[1].repeat >= 2
    assert samples[1].parameters["axis_source"] == "cnc_sysinfo.sysinfo.axes"
    assert samples[1].parameters["axis_domain"] == "1..N"
    assert samples[1].parameters["length_rule"] == "axis=-1: 4 + 4 * MAX_AXIS; axis=1..N: 8"


def test_running_rule_replaces_generic_motion_steps_for_accdecdly():
    plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="coordinate_motion",
        scenario_goal="capture", target_environment="ncguide-generated-cpp",
        nc_program_type="coordinate_motion", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name=""), steps=[], expected_outputs=[],
    )
    context = {"apis": {"cnc_accdecdly": {
        "api_info": {"api": {"prerequisites": {
            "nc_program": [{"program_number": None, "requirement": "running"}],
        }}},
    }}}
    assert apply_running_nc_program_rule(plan, context, ["cnc_accdecdly"])
    assert plan.target_api == "cnc_accdecdly"
    assert [step.protocol_function for step in plan.steps].count("cnc_sysinfo") == 1
    assert [step.protocol_function for step in plan.steps].count("cnc_accdecdly") == 3
    assert plan.rag_context["running_nc_program"]["axis_source"] == "cnc_sysinfo.axes"
    assert plan.rag_context["running_nc_program"]["axis_count_source"] == "runtime_output"


def test_running_rule_passes_graph_nc_requirements_to_plan():
    plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="coordinate_motion",
        scenario_goal="capture", target_environment="ncguide-generated-cpp",
        nc_program_type="coordinate_motion", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name=""), steps=[], expected_outputs=[],
    )
    context = {"apis": {"cnc_accdecdly": {
        "api_info": {"api": {"prerequisites": {
            "nc_program": [{
                "requirement": "running",
                "requirements": ["NC 程序必须包含加减速变化阶段"],
            }],
        }}},
    }}}
    assert apply_running_nc_program_rule(plan, context, ["cnc_accdecdly"])
    assert "NC 程序必须包含加减速变化阶段" in plan.nc_program_requirements
    assert "NC 程序必须包含加减速变化阶段" in plan.nc_program_spec.constraints


def test_fixed_runner_binds_accdecdly_to_sysinfo_without_axis_name_discovery():
    from smart_traffic_agent.agents.code_generator import render_fixed_ncguide_cpp_script
    from smart_traffic_agent.models import TaskRequest, WorkflowState

    state = WorkflowState(request=TaskRequest(
        description="capture cnc_accdecdly while running",
        task_id="accdecdly-runner",
        permissions={"required_api_functions": ["cnc_sysinfo", "cnc_accdecdly"]},
    ))
    state.plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="coordinate_motion",
        scenario_goal="capture", target_environment="ncguide-generated-cpp",
        nc_program_type="required_for_running_state", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name=""),
        steps=build_accdecdly_running_steps(), expected_outputs=[],
    )
    cpp = render_fixed_ncguide_cpp_script(state, state.plan.steps, "O1234\nG90 G54\nM30\n")
    assert "cnc_sysinfo_fn" in cpp
    assert "ParseSysinfoAxes" in cpp
    assert "axis_count_source=cnc_sysinfo.sysinfo.axes" in cpp
    assert "cnc_accdecdly_fn(handle, selectedAxis, length, &axis)" in cpp
    assert "NO_RESPONSE_VARIATION" in cpp
    assert "accdecdly_variation_observed" in cpp
    assert "runAxisApi(\"cnc_accdecdly\"" not in cpp
