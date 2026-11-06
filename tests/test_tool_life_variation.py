from pathlib import Path

from smart_traffic_agent.agents.code_generator import (
    validate_nc_program_requirements,
    validate_tool_life_runtime_evidence,
)
from smart_traffic_agent.fast_path import eligible_read_fast_path
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec
from smart_traffic_agent.nc_program_rules import (
    apply_tool_life_variation_rule,
    build_tool_life_variation_steps,
)
from smart_traffic_agent.traffic_graph import ApiNode, InMemoryTrafficGraphStore
from scripts.seed_trg_neo4j import RDNGRP_PREREQUISITES, _rdngrp_relation


SDK_HEADER = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD\Fwlib32.h")


def test_rdngrp_contract_and_registration_relation_are_graph_facts():
    store = InMemoryTrafficGraphStore()
    store.upsert_api(ApiNode("focas:cnc_rdngrp", "cnc_rdngrp", prerequisites=RDNGRP_PREREQUISITES))
    store.upsert_api(ApiNode("focas:cnc_instlifedt", "cnc_instlifedt"))
    stored = store.upsert_relation(_rdngrp_relation())
    assert stored["ok"]
    assert store.nodes["cnc_rdngrp"].prerequisites["nc_program"][0]["run_gate_api"] == "cnc_statinfo"
    deps = store.get_api_dependencies("cnc_rdngrp")
    assert any(row["relation"]["source_api"] == "cnc_instlifedt" for row in deps["dependencies"])


def test_rdngrp_contract_disables_ordinary_read_fast_path():
    eligible, facts = eligible_read_fast_path(
        ["cnc_rdngrp"], header_path=SDK_HEADER, prerequisites=RDNGRP_PREREQUISITES,
    )
    assert not eligible
    assert "stateful path" in " ".join(facts[0]["reasons"])


def test_tool_life_plan_contains_registration_nc_lifecycle_and_run_gate():
    plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="tool_management", scenario_goal="",
        target_environment="simulator", nc_program_type="", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(""), steps=[], expected_outputs=[],
    )
    context = {"apis": {"cnc_rdngrp": {"api_info": {"api": {"prerequisites": RDNGRP_PREREQUISITES}}}}}
    assert apply_tool_life_variation_rule(plan, context, ["cnc_rdngrp"])
    functions = [step.protocol_function for step in plan.steps]
    assert functions.index("cnc_instlifedt") < functions.index("cnc_rdngrp", functions.index("cnc_instlifedt"))
    assert "ncguide_ui_cycle_start" in functions
    assert functions.count("cnc_statinfo") >= 3
    assert plan.rag_context["tool_life_variation"]["status_gate"] == "cnc_statinfo.run"


def test_nc_program_requires_verified_tool_t_code():
    context = {
        "nc_target_api": "cnc_rdngrp",
        "nc_program_requirements": ["tool_life_variation"],
        "tool_life_variation": {"target_api": "cnc_rdngrp"},
        "verified_tool_numbers": [101],
    }
    assert validate_nc_program_requirements("O1234\nT0101\nM30\n", context) == []
    assert any("UNREGISTERED" in item for item in validate_nc_program_requirements("O1234\nT0202\nM30\n", context))


def test_dynamic_rdngrp_samples_need_registration_run_and_two_values():
    def log(api, response, status=0):
        return {"protocol_function": api, "response": response, "status_code": status, "error": None}

    base = [
        log("cnc_instlifedt", {}),
        log("cnc_rdgrpid", {"data": 1}),
        log("cnc_statinfo", {"run": 3}),
        log("cnc_rdngrp", {"data": 1}),
        log("cnc_rdngrp", {"data": 2}),
    ]
    assert validate_tool_life_runtime_evidence(base) == []
    missing_run = [item for item in base if item["protocol_function"] != "cnc_statinfo"]
    assert any("NC_RUN_GATE_MISSING" in item for item in validate_tool_life_runtime_evidence(missing_run))
