from smart_traffic_agent.focas_capture_runtime import deterministic_read_core, validate_capture_fragment, wrap_capture_fragment
from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.parameter_coverage import attach_parameter_contract, audit_parameter_coverage
from smart_traffic_agent.pmc_parameter_profiles import PMC_RANGE_PROFILES, pmc_profile_inputs, validate_all_pmc_profiles


def _plan(function: str = "pmc_rdpmcrng") -> ExecutionPlan:
    plan = ExecutionPlan(
        "p", "t", "capture", "PMC read", "simulator", "none", [], NcProgramSpec(""),
        [PlanStep("PMC", "during", "read PMC ranges", function, protocol_function=function)],
        [],
        rag_context={"function_coverage_manifest": {
            "selected_batch": [{"function": function, "coverage_role": "target"}]
        }},
    )
    attach_parameter_contract(plan)
    return plan


def test_requested_profiles_have_sdk_lengths_and_union_members():
    assert validate_all_pmc_profiles() == {profile["id"]: [] for profile in PMC_RANGE_PROFILES}
    assert [(p["length"], p["union"]) for p in PMC_RANGE_PROFILES] == [
        (10, "idata"), (18, "cdata"), (12, "fdata"), (24, "dfdata")
    ]


def test_read_recipe_expands_all_four_structured_ranges_without_writes():
    fragment = deterministic_read_core({"pmc_rdpmcrng"})
    assert fragment is not None
    assert fragment.count('"pmc_rdpmcrng"') == 4
    for index in range(1, 5):
        assert f"TARGET_PMC_RDPMCRNG_{index:02d}" in fragment
    assert "pmc_wrpmcrng" not in fragment
    assert "buf.type_a=" in fragment
    assert "buf.u.cdata[9]" in fragment
    assert "reinterpret_cast<float*>(buf->u.cdata)[0]" in fragment
    assert "reinterpret_cast<double*>(buf->u.cdata)[1]" in fragment
    assert validate_capture_fragment(wrap_capture_fragment(fragment)) == []


def test_four_profiles_satisfy_input_diversity_audit():
    plan = _plan()
    logs = []
    for index, profile in enumerate(PMC_RANGE_PROFILES, 1):
        raw = ";".join(f"{key}={value}" for key, value in pmc_profile_inputs(profile).items())
        logs.append(ApiCallLog(
            "t", "t", f"PMC-{index}", "during", "pmc_rdpmcrng", {"raw": raw}, 0,
            {"data": f"buf.type_a={profile['type_a']};buf.type_d={profile['type_d']}"},
            protocol_function="pmc_rdpmcrng", traffic_status="matched",
        ))
    row = audit_parameter_coverage(plan, logs)["functions"]["pmc_rdpmcrng"]
    assert row["distinct_input_combinations"] == 4
    assert row["eligible_distinct_input_combinations"] == 4
    assert row["input_diversity_required"] is True
    assert row["passed"] is True


def test_write_contract_uses_all_four_profiles_like_read_contract():
    plan = _plan("pmc_wrpmcrng")
    contract = plan.rag_context["parameter_coverage_contract"]["functions"]["pmc_wrpmcrng"]
    assert len(contract["cases"]) == 4
    assert contract["minimum_distinct"] == 4
    assert [case["length"] for case in contract["cases"]] == [10, 18, 12, 24]
    assert [case["buf"]["type_d"] for case in contract["cases"]] == [1, 0, 4, 5]
