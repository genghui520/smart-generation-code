from smart_traffic_agent.deterministic_cpp_assembler import (
    DeterministicAssemblyError,
    _validate_recipe_against_sdk,
    assemble_core_cpp,
)
from types import SimpleNamespace


def test_assembler_uses_structured_cases_and_official_call_recipe():
    core = assemble_core_cpp(
        targets=["cnc_rdmacro"],
        planned_functions=["cnc_rdmacro"],
        steps=[],
        parameter_contract={"functions": {"cnc_rdmacro": {
            "dimensions": ["number", "length"],
            "cases": [{"number": 100, "length": 8}, {"number": 101, "length": 8}],
        }}},
    )
    assert core.count("api.cnc_rdmacro(") == 2
    assert "api.function(" not in core
    assert "ODBM" in core
    assert "macro_number = static_cast<short>(100)" in core
    assert "macro_number = static_cast<short>(101)" in core


def test_assembler_reports_missing_deterministic_recipe():
    try:
        assemble_core_cpp(
            targets=["cnc_rdopmsg"],
            planned_functions=["cnc_rdopmsg"],
            steps=[],
            parameter_contract={"functions": {"cnc_rdopmsg": {"dimensions": [], "cases": []}}},
        )
    except DeterministicAssemblyError as exc:
        assert "DETERMINISTIC_ASSEMBLY_UNSUPPORTED" in str(exc)
    else:
        raise AssertionError("unsupported API must not fall back to model-authored C++")


def test_assembler_reads_parameter_cases_from_execution_plan_steps():
    from smart_traffic_agent.models import PlanStep

    core = assemble_core_cpp(
        targets=["cnc_rdmacro"],
        planned_functions=["cnc_rdmacro"],
        steps=[PlanStep(
            "S001", "during", "macro", "ReadMacro",
            {"parameter_cases": [{"number": 200, "length": 8}, {"number": 201, "length": 8}]},
            protocol_function="cnc_rdmacro",
        )],
        parameter_contract={"functions": {
            "cnc_rdmacro": {"dimensions": ["number", "length"], "cases": []},
        }},
    )
    assert core.count("api.cnc_rdmacro(") == 2
    assert "macro_number = static_cast<short>(200)" in core
    assert "macro_number = static_cast<short>(201)" in core


def test_assembler_preserves_execution_plan_order_for_multiple_targets():
    from smart_traffic_agent.models import PlanStep

    core = assemble_core_cpp(
        # The contract may be sorted, but the plan step order is authoritative.
        targets=["cnc_getpath", "cnc_rdcncid"],
        planned_functions=["cnc_getpath", "cnc_rdcncid"],
        steps=[
            PlanStep("S001", "during", "cncid", "ReadCncId", {}, protocol_function="cnc_rdcncid"),
            PlanStep("S002", "during", "path", "ReadPath", {}, protocol_function="cnc_getpath"),
        ],
        parameter_contract={"functions": {
            "cnc_getpath": {"dimensions": [], "cases": []},
            "cnc_rdcncid": {"dimensions": [], "cases": []},
        }},
    )
    assert core.index('"cnc_rdcncid"') < core.index('"cnc_getpath"')


def test_sdk_type_check_only_rejects_unknown_declaration_types():
    sdk = SimpleNamespace(structs={"ODBM": {}}, unions={}, functions={"cnc_rdmacro": ""})
    _validate_recipe_against_sdk(
        'ODBM value{}; ret = EW_OK; api.cnc_rdmacro(handle, 1, 8, &value);', sdk
    )
    try:
        _validate_recipe_against_sdk('ODBPARAINFO value{}; ret = EW_OK;', sdk)
    except DeterministicAssemblyError as exc:
        assert "ODBPARAINFO" in str(exc)
    else:
        raise AssertionError("unknown SDK declaration must be blocked")


def test_assembler_has_sdk_backed_wrrelpos_recipe():
    from smart_traffic_agent.models import PlanStep

    core = assemble_core_cpp(
        targets=["cnc_wrrelpos"],
        planned_functions=["cnc_wrrelpos"],
        steps=[PlanStep(
            "S001", "during", "wrrelpos", "PresetRelative", {
                "parameter_cases": [{"type": 1, "data0": 12}],
            }, protocol_function="cnc_wrrelpos",
        )],
        parameter_contract={"functions": {
            "cnc_wrrelpos": {"dimensions": ["type", "data0"], "cases": []},
        }},
    )
    assert "IDBWRR wrrelpos_value" in core
    assert "api.cnc_wrrelpos(handle, wrrelpos_length, &wrrelpos_value)" in core
    assert "api.function(" not in core


def test_sdk_cpp_generator_converts_json_plan_without_model_authored_cpp():
    from smart_traffic_agent.sdk_cpp_generator import generate_core_cpp_from_plan

    core = generate_core_cpp_from_plan({
        "targets": ["cnc_wrrelpos"],
        "planned_functions": ["cnc_wrrelpos"],
        "steps": [{
            "protocol_function": "cnc_wrrelpos",
            "parameters": {"parameter_cases": [{"type": 1, "data0": 12}]},
        }],
        "parameter_contract": {"functions": {
            "cnc_wrrelpos": {"dimensions": ["type", "data0"], "cases": []},
        }},
    })
    assert "api.cnc_wrrelpos(handle, wrrelpos_length, &wrrelpos_value)" in core
    assert "IDBWRR" in core
    assert "api.function(" not in core
