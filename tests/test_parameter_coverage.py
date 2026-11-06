from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, NcProgramSpec, PlanStep, ExecutionResult, TaskRequest, WorkflowState
from smart_traffic_agent.parameter_coverage import attach_parameter_contract, audit_parameter_coverage, input_schema
from smart_traffic_agent.agents.router import valid_next_stages


def plan_for(function="cnc_rdparam", strategies=None, cases=None, minimum_distinct=None):
    parameters = {"parameter_generation": strategies or [], "parameter_cases": cases or []}
    if minimum_distinct is not None:
        parameters["parameter_coverage_goal"] = {
            "minimum_distinct": minimum_distinct,
            "coverage_scope": "test evidence-backed domain",
            "evidence": "explicit test planner contract",
        }
    step = PlanStep("sample", "during", "capture", function, protocol_function=function,
                    parameters=parameters)
    return ExecutionPlan(plan_id="p", task_id="t", scenario_type="parameter_read", scenario_goal="capture",
                         target_environment="ncguide-generated-cpp", nc_program_type="generated",
                         nc_program_requirements=[], nc_program_spec=NcProgramSpec(""), steps=[step], expected_outputs=[],
                         rag_context={"function_coverage_manifest": {"enabled": True,
                            "selected_batch": [{"function": function, "coverage_role": "target"}]}})


def call(raw="number=1;axis=0;length=8", function="cnc_rdparam", code=0, matched=True, step="SAMPLE"):
    return ApiCallLog(timestamp="2026-10-15T00:00:00Z", task_id="t", step_id=step, phase="during",
                      interface_name=function, protocol_function=function, input_parameters={"raw": raw},
                      status_code=code, error="EW_DATA" if code else None, response={"data": "value=1"},
                      traffic_status="matched" if matched else "unmatched")


def audit(plan, logs):
    attach_parameter_contract(plan)
    return audit_parameter_coverage(plan, logs)


class ParameterCoverageTests(unittest.TestCase):
    def test_rdposition_spaced_in_out_direction(self):
        schema = input_schema("cnc_rdposition")
        self.assertTrue(schema["documented"])
        self.assertIn("data_num", schema["inputs"])
        self.assertIn("data_num", schema["pointer_inputs"])
        self.assertNotIn("position", schema["inputs"])

    def test_planner_discovery_and_goal_survive_handoff(self):
        from smart_traffic_agent.agents.planner import plan_steps_from_llm_rows
        p = plan_for("cnc_search", minimum_distinct=2)
        p.steps = plan_steps_from_llm_rows([{
            "step_id": "S1", "phase": "during", "interface_name": "cnc_search", "protocol_function": "cnc_search",
            "parameter_generation": [{"name": "number", "kind": "discovered", "evidence": "program directory",
                "discovery": {"api": "cnc_rdprogdir3"}, "selection_rule": "distinct resident programs",
                "constraints": ["idle"], "dependencies": ["directory"], "on_insufficient": "report"}],
            "parameter_coverage_goal": {"minimum_distinct": 3, "restore_strategy": "restore original"},
            "state_coverage": {"states": ["idle"]}}])
        contract = attach_parameter_contract(p)["functions"]["cnc_search"]
        self.assertEqual(contract["minimum_distinct"], 3)
        self.assertEqual(contract["planner_strategies"][0]["discovery"], {"api": "cnc_rdprogdir3"})
        self.assertEqual(contract["state_coverage"], [{"states": ["idle"]}])
        self.assertFalse(audit_parameter_coverage(p, [call("number=1", "cnc_search"), call("number=2", "cnc_search")])["passed"])

    def test_repeated_calls_and_handles_are_one_combination(self):
        report = audit(plan_for(minimum_distinct=2), [call(f"FlibHndl={i};number=1;axis=0;length=8") for i in range(30)])
        row = report["functions"]["cnc_rdparam"]
        self.assertEqual(row["call_count"], 30)
        self.assertEqual(row["matched_distinct_input_combinations"], 1)
        self.assertFalse(report["passed"])

    def test_failed_and_unmatched_variants_do_not_count(self):
        report = audit(plan_for(minimum_distinct=2), [call(), call("number=2;axis=0;length=8", code=5),
                                  call("number=3;axis=0;length=8", matched=False)])
        self.assertEqual(report["functions"]["cnc_rdparam"]["successful_distinct_input_combinations"], 2)
        self.assertEqual(report["functions"]["cnc_rdparam"]["matched_distinct_input_combinations"], 1)
        self.assertFalse(report["passed"])

    def test_shape_variations_do_not_count(self):
        self.assertFalse(audit(plan_for(minimum_distinct=2), [call(), call("number=1;axis=0;length=16")])["passed"])

    def test_single_matched_call_satisfies_callable_coverage_without_diversity_goal(self):
        report = audit(plan_for("cnc_statinfo"), [call("FlibHndl=1", "cnc_statinfo")])
        row = report["functions"]["cnc_statinfo"]
        self.assertTrue(report["passed"])
        self.assertEqual(row["minimum_distinct"], 1)
        self.assertEqual(row["matched_distinct_input_combinations"], 1)
        self.assertEqual(row["matched_output_variants"], 1)

    def test_planner_cannot_lower_documented_axis_domain_to_one_input(self):
        p = plan_for("cnc_skip", minimum_distinct=1)
        contract = attach_parameter_contract(p)["functions"]["cnc_skip"]
        self.assertTrue(contract["axis_domain_documented"])
        self.assertEqual(contract["minimum_distinct"], 2)
        report = audit_parameter_coverage(p, [
            call("axis=-1;length=132", "cnc_skip"),
            call("axis=-1;length=132", "cnc_skip"),
        ])
        self.assertFalse(report["passed"])
        self.assertEqual(report["functions"]["cnc_skip"]["matched_distinct_input_combinations"], 1)

    def test_contract_reports_separate_input_and_response_requirements(self):
        p = plan_for("cnc_skip", minimum_distinct=2)
        contract = attach_parameter_contract(p)["functions"]["cnc_skip"]
        self.assertEqual(contract["variation_requirements"]["variation_class"], "input_only")
        self.assertTrue(contract["input_diversity_required"])
        self.assertFalse(contract["response_state_diversity_required"])
        report = audit_parameter_coverage(p, [call("axis=-1;length=132", "cnc_skip")])
        row = report["functions"]["cnc_skip"]
        self.assertEqual(row["response_state_variation_status"], "optional_not_observed")

    def test_successful_distinct_parameters_pass(self):
        self.assertTrue(audit(plan_for(), [call(), call("number=2;axis=0;length=8")])["passed"])

    def test_enum_requires_every_planned_value(self):
        p = plan_for(strategies=[{"name": "number", "kind": "enum", "values": [1, 2, 3]}])
        report = audit(p, [call(), call("number=2;axis=0;length=8")])
        self.assertFalse(report["passed"])
        self.assertEqual(report["functions"]["cnc_rdparam"]["missing_values"], {"number": ["3"]})

    def test_range_requires_intermediate_sample(self):
        p = plan_for(strategies=[{"name": "number", "kind": "range", "min": 1, "max": 9}])
        logs = [call(), call("number=9;axis=0;length=8")]
        self.assertFalse(audit(p, logs)["passed"])
        self.assertFalse(audit(p, logs + [call("number=10;axis=0;length=8")])["passed"])
        self.assertTrue(audit(p, logs + [call("number=5;axis=0;length=8")])["passed"])

    def test_coupled_cases_are_not_marginal_coverage(self):
        p = plan_for(cases=[{"number": 1, "axis": 0, "length": 8}, {"number": 2, "axis": 1, "length": 12}])
        self.assertFalse(audit(p, [call(), call("number=2;axis=0;length=12")])["passed"])
        self.assertTrue(audit(p, [call(), call("number=2;axis=1;length=12")])["passed"])

    def test_dynamic_length_alias_requires_actual_sdk_byte_count(self):
        for function, placeholder, length in (
            ("cnc_rddynamic", "ODBDYNAMIC", 536),
            ("cnc_rddynamic2", "ODBDYNAMIC2", 540),
        ):
            with self.subTest(function=function):
                case = {"axis": -1, "length": f"sizeof({placeholder})"}
                p = plan_for(function, strategies=[
                    {"name": "length", "kind": "fixed", "value": case["length"]}
                ], cases=[case])
                valid = [
                    call(f"FlibHndl=1;axis=-1;length={length}", function),
                    call(f"FlibHndl=1;axis=1;length={length}", function),
                ]
                self.assertTrue(audit(p, valid)["passed"])
                wrong = [
                    call(f"FlibHndl=1;axis=-1;length={length - 4}", function),
                    call(f"FlibHndl=1;axis=1;length={length - 4}", function),
                ]
                row = audit(p, wrong)["functions"][function]
                self.assertFalse(row["passed"])
                self.assertIn("planned_values_missing", row["issues"])
                self.assertIn("planned_cases_missing", row["issues"])

    def test_restore_cannot_create_diversity(self):
        p = plan_for("cnc_search", minimum_distinct=2)
        self.assertFalse(audit(p, [call("prg_no=1", "cnc_search"),
                                  call("prg_no=2", "cnc_search", step="RESTORE")])["passed"])
        self.assertTrue(audit(p, [call("prg_no=1", "cnc_search"), call("prg_no=2", "cnc_search")])["passed"])

    def test_output_only_buffers_are_not_input_dimensions(self):
        self.assertEqual(input_schema("cnc_actf")["inputs"], [])
        report = audit(plan_for("cnc_actf"), [call("FlibHndl=1", "cnc_actf")])
        self.assertTrue(report["passed"])
        self.assertFalse(report["functions"]["cnc_actf"]["input_diversity_applicable"])

    def test_unknown_documentation_is_unverified(self):
        self.assertFalse(audit(plan_for("cnc_unknown"), [call("foo=1", "cnc_unknown")])["passed"])

    def test_missing_inputs_cannot_count(self):
        self.assertFalse(audit(plan_for(), [call("number=1"), call("number=2")])["passed"])

    def test_numeric_formatting_does_not_create_diversity(self):
        self.assertFalse(audit(plan_for(minimum_distinct=2), [call(), call("number=1.0;axis=0;length=8")])["passed"])

    def test_parameter_audit_failure_does_not_repair_successful_execution(self):
        p = plan_for(minimum_distinct=2); report = audit(p, [call()])
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        state.result = ExecutionResult("t", True, [call()], [], Path("."), offline_audit_status="completed")
        self.assertFalse(report["passed"])
        self.assertEqual(valid_next_stages(state), {"complete"})

    def test_incomplete_diversity_is_not_marked_covered(self):
        from smart_traffic_agent.function_coverage import merge_workflow_function_coverage_state, has_captured_traffic
        p = plan_for(minimum_distinct=2); audit(p, [call()])
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        state.result = ExecutionResult("t", False, [call()], [], Path("."))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "coverage.json"
            merge_workflow_function_coverage_state(state, path)
            import json
            record = json.loads(path.read_text(encoding="utf-8"))["functions"]["cnc_rdparam"]
            self.assertEqual(record["status"], "covered_success")
            self.assertTrue(has_captured_traffic(record))

    def test_variable_motion_target_does_not_bypass_generator(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_absolute")
        p.steps[0].parameters["parameter_cases"] = [{"axis": -1, "length": 132}]
        p.steps.append(PlanStep("start", "before", "start", "StartProgram"))
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.return_value = {"api_fragments": [{
            "step_id": "sample",
            "apis": ["cnc_absolute"],
            "calls": [{
                "api_name": "cnc_absolute",
                "call_expr": "api.cnc_absolute(handle, static_cast<short>(-1), static_cast<short>(132), &__out_absolute)",
                "parameter_values": {"axis": -1, "length": 132},
            }],
        }]}
        fragment = generate_capture_fragment(state, p.steps, "", client)
        client.invoke_json.assert_called_once()
        self.assertIn("api.cnc_absolute", fragment)

    def test_repair_accepts_complete_callback_and_preserves_log_contract(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_search")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock()
        client.enabled = True
        with self.assertRaisesRegex(ValueError, "API_FRAGMENT_GENERATION_FAILED"):
            generate_capture_fragment(state, p.steps, "", client,
                                      previous="old fragment", diagnostics="repair logging")
        client.invoke_json.assert_called()

    def test_api_without_deterministic_recipe_uses_one_plan_bundle_call(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_rdopmsg")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.return_value = {"api_fragments": [{
            "step_id": "sample",
            "apis": ["cnc_rdopmsg"],
            "calls": [{
                "api_name": "cnc_rdopmsg",
                "call_expr": "api.cnc_rdopmsg(handle, 0, 256, &__out_opmsg)",
                "parameter_values": {"type": 0, "length": 256},
            }],
        }]}
        fragment = generate_capture_fragment(state, p.steps, "", client)
        self.assertIn("api.cnc_rdopmsg", fragment)
        self.assertEqual(client.invoke_json.call_count, 1)

    def test_batch_generation_passes_struct_values_to_generic_sdk_assembler(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_wrrelpos")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.return_value = {"api_fragments": [{
            "step_id": "sample",
            "apis": ["cnc_wrrelpos"],
            "calls": [{
                "api_name": "cnc_wrrelpos",
                "call_expr": "api.cnc_wrrelpos(handle, 8, &__out_wrrelpos)",
                "parameter_values": {"length": 8},
                "struct_values": {"wrrelpos": {"datano": 0, "type": 1, "data": [12]}},
            }],
        }]}
        fragment = generate_capture_fragment(state, p.steps, "", client)
        self.assertIn("sample_wrrelpos.type = static_cast<short>(1);", fragment)
        self.assertIn("sample_wrrelpos.data[0] = static_cast<long>(12);", fragment)

    def test_one_step_can_emit_multiple_parameter_combinations(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_statinfo")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.return_value = {"api_fragments": [{
            "step_id": "sample",
            "apis": ["cnc_statinfo"],
            "calls": [
                {"api_name": "cnc_statinfo", "call_expr": "api.cnc_statinfo(handle, &__out_statinfo)", "parameter_values": {}},
                {"api_name": "cnc_statinfo", "call_expr": "api.cnc_statinfo(handle, &__out_statinfo)", "parameter_values": {}},
            ],
        }]}
        fragment = generate_capture_fragment(state, p.steps, "", client)
        assert fragment.count('"cnc_statinfo"') == 2
        assert "sample_C002_statinfo" in fragment
        assert client.invoke_json.call_count == 1

    def test_batch_generation_rejects_legacy_capture_call_before_compile(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_statinfo")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.return_value = {"api_fragments": [{
            "step_id": "sample",
            "apis": ["cnc_statinfo"],
            "core_cpp": 'ret = ctx.call("cnc_statinfo", [&]() { return api.cnc_statinfo(handle, &status); }, {{"FlibHndl", std::to_string(handle)}}, {{"run", "0"}});',
        }]}
        with self.assertRaisesRegex(ValueError, "CODEGEN_MODEL_OUTPUT_CONTRACT"):
            generate_capture_fragment(state, p.steps, "", client)
        self.assertEqual(client.invoke_json.call_count, 1)

    def test_repair_regenerates_only_named_api_fragment(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_statinfo")
        p.steps.append(PlanStep("S2", "during", "read message", "cnc_rdopmsg", {}, protocol_function="cnc_rdopmsg"))
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.side_effect = [
            {"api_fragments": [
                {"step_id": "sample", "apis": ["cnc_statinfo"], "calls": [{"api_name": "cnc_statinfo", "call_expr": 'api.cnc_statinfo(handle, &__out_statinfo)', "parameter_values": {}}]},
                {"step_id": "S2", "apis": ["cnc_rdopmsg"], "calls": [{"api_name": "cnc_rdopmsg", "call_expr": 'api.cnc_rdopmsg(handle, 0, 256, &__out_opmsg)', "parameter_values": {"type": 0, "length": 256}}]},
            ]},
            {"api_fragments": [
                {"step_id": "S2", "apis": ["cnc_rdopmsg"], "calls": [{"api_name": "cnc_rdopmsg", "call_expr": 'api.cnc_rdopmsg(handle, 0, 256, &__out_opmsg)', "parameter_values": {"type": 0, "length": 256}}]},
            ]},
        ]
        first = generate_capture_fragment(state, p.steps, "", client)
        second = generate_capture_fragment(state, p.steps, "", client, previous=first, diagnostics="BLOCKING: cnc_rdopmsg call")
        self.assertIn('"sample"', second)
        self.assertIn('"S2"', second)
        self.assertEqual(client.invoke_json.call_count, 2)

    def test_failed_fragment_keeps_valid_cached_fragments_for_code_repair(self):
        from smart_traffic_agent.agents.code_generator import generate_capture_fragment
        p = plan_for("cnc_statinfo")
        p.steps.append(PlanStep("S2", "during", "read message", "cnc_rdopmsg", {}, protocol_function="cnc_rdopmsg"))
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=p)
        client = Mock(); client.enabled = True
        client.invoke_json.side_effect = [
            {"api_fragments": [
                {"step_id": "sample", "apis": ["cnc_statinfo"], "calls": [{"api_name": "cnc_statinfo", "call_expr": 'api.cnc_statinfo(handle, &__out_statinfo)', "parameter_values": {}}]},
                {"step_id": "S2", "apis": ["cnc_rdopmsg"], "core_cpp":
                    'ret = ctx.call("cnc_rdopmsg", [&]() { return api.cnc_rdopmsg(handle, &msg); }, {{""}}, {{""}});'},
            ]},
            {"api_fragments": [
                {"step_id": "S2", "apis": ["cnc_rdopmsg"], "calls": [{"api_name": "cnc_rdopmsg", "call_expr": 'api.cnc_rdopmsg(handle, 0, 128, &__out_opmsg)', "parameter_values": {"type": 0, "length": 128}}]},
            ]},
        ]
        with self.assertRaisesRegex(ValueError, "CODEGEN_MODEL_OUTPUT_CONTRACT"):
            generate_capture_fragment(state, p.steps, "", client)
        self.assertIn("sample", state.plan.rag_context["codegen_api_fragments"])
        self.assertNotIn("S2", state.plan.rag_context["codegen_api_fragments"])
        state.errors = ["CodeGenerationAgent failed: CAPTURE_CALL_SIGNATURE_MISMATCH api=cnc_rdopmsg"]
        state.repair_history = [{"repair_stage": "repair_code"}]
        repaired = generate_capture_fragment(state, p.steps, "", client)
        self.assertIn('"sample"', repaired)
        self.assertIn('"S2"', repaired)
        self.assertIn("std::to_string(128)", repaired)
        self.assertEqual(client.invoke_json.call_count, 2)


if __name__ == "__main__":
    unittest.main()
