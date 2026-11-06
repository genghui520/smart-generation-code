from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smart_traffic_agent.agents.code_generator import (
    CodeGenerationAgent,
    generate_nc_program,
    blocking_codegen_diagnostics,
    hard_blocking_codegen_diagnostics,
    normalize_focas_header_include,
    normalize_nc_download_payload,
    normalize_llm_nc_program,
    normalize_official_odbpos_axis_access,
    normalize_plan_nc_program_contract,
    official_focas_abi_context,
    preflight_compile_generated_cpp,
    validate_real_packet_capture_usage,
    validate_cycle_start_ready_states,
    repair_cpp_api_script_after_compile_error,
    retrieve_codegen_knowledge_context,
    select_codegen_executable_steps,
    steps_for_prompt,
    validate_generated,
    validate_focas_loader_type_safety,
    validate_focas_position_abi,
    validate_focas_argument_semantics,
    validate_official_output_usage,
    validate_poselem_scalar_access,
    validate_all_focas_symbols_against_header,
    extract_real_focas_symbol_references,
    build_scenario_fragment,
    cpp_step_block,
    normalize_capture_core,
    _validate_api_fragment_call,
    validate_capture_context_call_contract,
    validate_scenario_fragment,
    validate_semantic_api_bindings,
    _ensure_parameter_combinations,
    should_skip_nc_program_generation,
    should_defer_nc_program_generation,
)
from smart_traffic_agent.agent_tools import (
    CompileCppOutput,
    RenderFocasCppTool,
    ValidateFocasCppTool,
    WriteGeneratedArtifactTool,
)
from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep, TaskRequest, WorkflowState
from smart_traffic_agent.agent_roles import ApiContract, CodeSpec


class FakeCodegenLlm:
    enabled = True

    def __init__(self, repair_cpp: str = "int main(){return 0;}") -> None:
        self.prompts: list[str] = []
        self.repair_cpp = repair_cpp

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.prompts.append(user_prompt)
        if "MSVC preflight compiler diagnostics" in user_prompt or "Contract violation:" in user_prompt:
            return {"cpp_code": self.repair_cpp}
        return {"ok": True, "diagnostics": []}


def hard_pcap_valid_cpp(extra: str = "") -> str:
    return (
        "#include <Fwlib32.h>\n"
        "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW "
        "auto fn1 = reinterpret_cast<decltype(&::cnc_allclibhndl3)>(GetProcAddress(dll, \"cnc_allclibhndl3\")); "
        "auto fn2 = reinterpret_cast<decltype(&::cnc_freelibhndl)>(GetProcAddress(dll, \"cnc_freelibhndl\")); "
        "class PacketSniffer { public: "
        "bool Start(){ pcap_open_live; pcap_dump_open; return true; } "
        "void CapturePackets(){ pcap_dispatch; pcap_dump; pcap_dump_flush; } "
        "void Stop(){} }; "
        "int main(){ const char* envDevice = getenv(\"PCAP_NETWORK_DEVICE\"); std::string networkDevice; std::string pcapFile; "
        "PacketSniffer sniffer(networkDevice, pcapFile); "
        "if (!sniffer.Start()) { return 1; } "
        "sniffer.CapturePackets(); sniffer.Stop(); const char* status=\"pcap_capture_enabled=true\"; "
        f"{extra} return 0; }}"
    )


class CodeGenerationAgentTests(unittest.TestCase):
    def test_running_nc_fast_path_still_generates_motion_program(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "coordinate_motion", "goal", "env", "test", [],
            NcProgramSpec(program_name=""), [], [],
            rag_context={"fast_path": {"enabled": True, "kind": "running_nc_program"}},
        )
        self.assertFalse(should_skip_nc_program_generation(plan))

    def test_read_only_fast_path_skips_nc_program(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "read", "goal", "env", "test", [],
            NcProgramSpec(program_name=""), [], [],
            rag_context={"fast_path": {"enabled": True, "kind": "read_only_api"}},
        )
        self.assertTrue(should_skip_nc_program_generation(plan))

    def test_deferred_running_nc_plan_keeps_program_payload(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "coordinate_motion", "goal", "env",
            "required_for_running_state", [], NcProgramSpec(program_name=""), [], [],
        )
        self.assertFalse(should_defer_nc_program_generation(plan, True))

    def test_deferred_read_only_motion_plan_can_skip_program(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "coordinate_motion", "goal", "env",
            "coordinate_motion", [], NcProgramSpec(program_name=""), [], [],
        )
        self.assertTrue(should_defer_nc_program_generation(plan, True))

    def test_alarm_history_requires_controller_specific_ncguide_trigger(self) -> None:
        spec = NcProgramSpec(program_name="O9123")
        fake_llm = FakeCodegenLlm()
        fake_llm.invoke_json = lambda *_args, **_kwargs: {
            "nc_program": "\nO9123\nG90 G54\nG04 P1\nM30\n%",
            "notes": ["controller-specific alarm rule: PS0010; evidence=manual:test"],
            "alarm_trigger": {
                "alarm_code": "PS0010",
                "condition": "controller-specific documented NC condition",
                "evidence": "manual:test",
            },
        }
        program, diagnostics = generate_nc_program(
            "collect cnc_rdalmhistry5",
            "alarm_query",
            spec,
            {"alarm_history_trigger": {
                "required": True,
                "alarm_code": "PS0010",
                "trigger": "G999",
                "scope": "NCGuide simulator only",
            }},
            fake_llm,
        )
        self.assertNotIn("G999", program)
        self.assertIn("M30", program)
        self.assertTrue(program.startswith("\nO9123"))

    def test_pending_runtime_dependency_does_not_raise_empty_combinations(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "scenario", "goal", "env", "test", [], NcProgramSpec("O1"),
            [PlanStep("S1", "before", "read source", "cnc_sysinfo", {}, protocol_function="cnc_sysinfo"),
             PlanStep("S2", "during", "read target", "cnc_accdecdly", {}, protocol_function="cnc_accdecdly")],
            [], runtime_bindings=[{
                "target_api": "cnc_accdecdly", "target_parameter_path": "axis",
                "source_api": "cnc_sysinfo", "filter_mode": "include",
            }],
        )
        plan.rag_context["parameter_coverage_contract"] = {
            "functions": {"cnc_accdecdly": {"dimensions": ["axis"], "cases": [], "planner_strategies": []}}
        }
        plan.rag_context["execution_plan_validation"] = {"valid": True}
        _ensure_parameter_combinations(plan, None, {"cnc_accdecdly"})

    def test_runtime_axis_domain_declared_on_step_is_not_empty(self) -> None:
        plan = ExecutionPlan(
            "p", "task", "scenario", "goal", "env", "test", [], NcProgramSpec("O1"),
            [PlanStep(
                "S1", "during", "read target", "cnc_accdecdly",
                {"axis_source": "cnc_sysinfo.sysinfo.axes", "axis_domain": "1..N"},
                protocol_function="cnc_accdecdly",
            )],
            [],
        )
        plan.rag_context["parameter_coverage_contract"] = {
            "functions": {"cnc_accdecdly": {"dimensions": ["axis"], "cases": [], "planner_strategies": []}}
        }
        plan.rag_context["execution_plan_validation"] = {"valid": True}
        _ensure_parameter_combinations(plan, None, {"cnc_accdecdly"})

    def test_start_program_uses_current_run_ui_environment_without_overwrite(self) -> None:
        source = "\n".join(
            cpp_step_block(
                PlanStep(
                    "S001",
                    "during",
                    "start",
                    "StartProgram",
                    {
                        "window_title": "OLD WINDOW",
                        "button_text": "OLD BUTTON",
                        "click_mode": "screen",
                        "cycle_start_x": 989,
                        "cycle_start_y": 914,
                    },
                )
            )
        )

        self.assertIn('TriggerNcGuideCycleStart("OLD WINDOW", data)', source)
        self.assertNotIn('SetEnvironmentVariableA("NCGUIDE_', source)
        self.assertNotIn('"989"', source)
        self.assertNotIn('"914"', source)

    def test_empty_button_text_forces_calibrated_coordinate_path(self) -> None:
        from smart_traffic_agent.agents.code_generator import render_cpp_api_script_v2

        runtime = render_cpp_api_script_v2(
            "coordinate_motion",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
            "O1234\nM30\n",
        )

        self.assertIn('EnvString("NCGUIDE_START_BUTTON_TEXT", "")', runtime)
        self.assertNotIn('EnvString("NCGUIDE_START_BUTTON_TEXT", "Run")', runtime)

    def test_tool_data_access_validator_requires_real_official_call(self) -> None:
        step = PlanStep(
            "TM-1", "during", "read tool data", "ToolDataAccess",
            protocol_function="cnc_rdtool",
        )
        missing = validate_semantic_api_bindings(
            "// cnc_rdtool(handle, 1, &count, data);", [step]
        )
        self.assertTrue(any("Missing required semantic operation: tool_data_access" in item for item in missing))

        valid = validate_semantic_api_bindings(
            "IODBTLMNG data{}; short count = 1; ret = api.cnc_rdtool(handle, 1, &count, &data);",
            [step],
        )
        self.assertEqual(valid, [])

    def test_normalize_focas_header_include_strips_generated_controller_path(self) -> None:
        variants = [
            '#include "Fwlib/0iD/Fwlib32.h"',
            "#include <Fwlib/0iD/Fwlib32.h>",
            '#include "Fwlib\\\\0iD\\\\Fwlib32.h"',
            '# include "C:/Lib/FOCAS2 Library/Fwlib/0iD/Fwlib32.h"',
        ]

        for include_line in variants:
            with self.subTest(include_line=include_line):
                normalized = normalize_focas_header_include(
                    f"#include <windows.h>\n{include_line}\nint main(){{return 0;}}\n"
                )

                self.assertIn("#include <Fwlib32.h>", normalized)
                self.assertNotIn("0iD/Fwlib32.h", normalized)
                self.assertNotIn("0iD\\\\Fwlib32.h", normalized)

    def test_normalize_nc_download_payload_removes_leading_percent_line(self) -> None:
        source = r'''std::string program = "%\n"
            "O3114\n"
            "G90 G54\n"
            "M30\n"
            "%\n";'''

        normalized = normalize_nc_download_payload(source)

        self.assertNotIn(r'"%\n"', normalized)
        self.assertIn(r'"\n"', normalized)
        self.assertIn(r'"%"', normalized)

    def test_validate_official_output_accepts_dynamic_rdposition_wrapper(self) -> None:
        source = "ODBPOS pos{}; ret = api.cnc_rdposition_fn(handle, 0, &axis_count, &pos);"
        code_spec = type(
            "CodeSpecStub",
            (),
            {"api_contracts": [type("ContractStub", (), {"tool_name": "cnc_rdposition"})()]},
        )()

        self.assertEqual(validate_official_output_usage(source, code_spec), [])

    def test_strict_header_gate_rejects_unknown_focas_symbol(self) -> None:
        source = '#include <Fwlib32.h>\nint main(){ cnc_not_in_fwlib32(handle); }'
        diagnostics = validate_all_focas_symbols_against_header(source)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("cnc_not_in_fwlib32", diagnostics[0])

    def test_strict_header_gate_accepts_declared_focas_symbol(self) -> None:
        source = '#include <Fwlib32.h>\nint main(){ cnc_statinfo(handle, &status); }'
        self.assertEqual(validate_all_focas_symbols_against_header(source), [])

    def test_strict_header_gate_ignores_local_function_pointer_aliases(self) -> None:
        source = r'''
#include <Fwlib32.h>
using cnc_actf_t = decltype(&::cnc_actf);
typedef decltype(&::cnc_statinfo) cnc_statinfo_t;
auto cnc_actf_fn = reinterpret_cast<cnc_actf_t>(GetProcAddress(dll, "cnc_actf"));
auto cnc_statinfo_fn = reinterpret_cast<cnc_statinfo_t>(GetProcAddress(dll, "cnc_statinfo"));
short cnc_local_helper(unsigned short handle) { return cnc_actf_fn(handle, nullptr); }
int main() { return cnc_statinfo_fn(handle, nullptr); }
'''

        self.assertEqual(
            extract_real_focas_symbol_references(source),
            {"cnc_actf", "cnc_statinfo"},
        )
        self.assertEqual(validate_all_focas_symbols_against_header(source), [])

    def test_strict_header_gate_still_rejects_unknown_real_references(self) -> None:
        dynamic = 'GetProcAddress(dll, "cnc_not_in_fwlib32")'
        direct = "int main(){ cnc_also_not_in_fwlib32(handle); }"

        self.assertTrue(validate_all_focas_symbols_against_header(dynamic))
        self.assertTrue(validate_all_focas_symbols_against_header(direct))

    def test_official_focas_abi_context_extracts_selected_prototypes_and_types(self) -> None:
        context = official_focas_abi_context(["cnc_absolute", "cnc_actf", "cnc_rdprogdir"])

        self.assertIn("cnc_absolute", context)
        self.assertIn("cnc_actf", context)
        self.assertIn("cnc_rdprogdir", context)
        self.assertIn("typedef struct odbaxis", context.lower())
        self.assertIn("long    data[MAX_AXIS]", context)
        self.assertIn("typedef struct odbact", context.lower())
        self.assertIn("typedef struct prgdir", context.lower())

    def test_normalize_llm_nc_program_forces_planned_program_name(self) -> None:
        nc_program = normalize_llm_nc_program(
            "O9999\nG90 G54\nG01 X10.0 F20\n",
            "O2468",
        )

        self.assertTrue(nc_program.startswith("O2468\n"))
        self.assertIn("G90 G54", nc_program)
        self.assertTrue(nc_program.rstrip().endswith("M30"))

    def test_normalize_llm_nc_program_rejects_unsafe_blocks(self) -> None:
        nc_program = normalize_llm_nc_program(
            "O1234\nG90 G54\nG28 X0\nM30\n",
            "O1234",
        )

        self.assertEqual(nc_program, "")

    def test_normalize_llm_nc_program_removes_standalone_generator_comment(self) -> None:
        nc_program = normalize_llm_nc_program(
            "O1234\n(NON-MOTION BATCH: PROGRAM PAYLOAD ONLY)\nG01 X1 F100\nM30\n",
            "O2468",
        )
        self.assertNotIn("NON-MOTION BATCH", nc_program)
        self.assertIn("G01 X1 F100", nc_program)

    def test_milling_program_omits_unneeded_g97_spindle_mode(self) -> None:
        from smart_traffic_agent.agents.code_generator import validate_nc_controller_modes

        nc_program = normalize_llm_nc_program(
            "O1234\nG90 G17 G54\nG97 S800 M03\nG01 X10 Y2 F100\nM30\n",
            "O1234",
        )
        self.assertNotIn("G97", nc_program)
        self.assertIn("S800 M03", nc_program)
        self.assertEqual(
            normalize_llm_nc_program("O1234\nG17\nG96 S100 M03\nM30\n", "O1234"), "",
        )
        self.assertNotIn("G97", normalize_llm_nc_program("O1234\nG97 S800 M03\nM30\n", "O1234"))
        self.assertIn("G97", normalize_llm_nc_program("O1234\nG96 S100\nG97 S800 M03\nM30\n", "O1234"))
        self.assertEqual(validate_nc_controller_modes("\nO1234\nG17\nG97 S800 M03\nM30\n%"),
                         ["NC program must omit unneeded G97; use S with M03/M04 directly"])

    def test_codegen_allocates_program_number_when_planner_leaves_it_blank(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-codegen-number",
            task_id="codegen-number",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name=""),
            steps=[
                PlanStep("S001", "before", "upload", "UploadProgram", {}),
                PlanStep("S002", "before", "select", "SelectProgram", {}),
            ],
            expected_outputs=[],
        )

        diagnostics = normalize_plan_nc_program_contract(plan)

        self.assertRegex(plan.nc_program_spec.program_name, r"^O[1-7]\d{3}$")
        self.assertEqual(
            [step.parameters["program_name"] for step in plan.steps],
            [plan.nc_program_spec.program_name, plan.nc_program_spec.program_name],
        )
        self.assertEqual(
            plan.rag_context["code_generation_program_allocation"]["reason"],
            "allocated_by_code_generation",
        )
        self.assertTrue(any("CodeGenerationAgent allocated" in item for item in diagnostics))

    def test_normalize_plan_program_contract_updates_all_consumers(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-codegen001",
            task_id="codegen001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O8946"),
            steps=[
                PlanStep("S001", "before", "upload", "UploadProgram", {"program_name": "O8946"}),
                PlanStep("S002", "before", "select", "SelectProgram", {"program_name": "O8946"}),
            ],
            expected_outputs=[],
            code_spec=object(),
        )

        diagnostics = normalize_plan_nc_program_contract(plan)

        self.assertEqual(plan.nc_program_spec.program_name, "O6646")
        self.assertEqual([step.parameters["program_name"] for step in plan.steps], ["O6646", "O6646"])
        self.assertIsNone(plan.code_spec)
        self.assertEqual(
            plan.rag_context["nc_program_name_normalization"],
            {
                "requested": "O8946",
                "effective": "O6646",
                "reason": "protected_or_invalid_program_number",
            },
        )
        self.assertIn("O8946 to O6646", diagnostics[0])

    def test_normalize_odbpos_axis_access_uses_one_object_per_axis(self) -> None:
        source = (
            "void dump(const ODBPOS& pos) { "
            "use(pos.abs.data[0], pos.mach.data[1], pos.dist.data[2]); } "
            "int main() { ODBPOS pos[3]{}; dump(pos); }"
        )

        normalized = normalize_official_odbpos_axis_access(source)

        self.assertIn("const ODBPOS* pos", normalized)
        self.assertIn("pos[0].abs.data", normalized)
        self.assertIn("pos[1].mach.data", normalized)
        self.assertIn("pos[2].dist.data", normalized)
        self.assertIn("dump(pos)", normalized)

    def test_validate_focas_loader_rejects_one_fixed_type_for_all_symbols(self) -> None:
        source = (
            "auto load = [&](auto& fn, const char* name) { "
            "using Fn = decltype(&::cnc_allclibhndl3); "
            "fn = reinterpret_cast<Fn>(GetProcAddress(module, name)); }; "
            "load(api.cnc_allclibhndl3, \"cnc_allclibhndl3\"); "
            "load(api.cnc_actf, \"cnc_actf\");"
        )

        diagnostics = validate_focas_loader_type_safety(source)

        self.assertEqual(len(diagnostics), 1)
        self.assertIn("fixed type", diagnostics[0])

    def test_validate_focas_loader_accepts_destination_derived_type(self) -> None:
        source = (
            "auto load = [&](auto& fn, const char* name) { "
            "using Fn = std::remove_reference_t<decltype(fn)>; "
            "fn = reinterpret_cast<Fn>(GetProcAddress(module, name)); }; "
            "load(api.cnc_allclibhndl3, \"cnc_allclibhndl3\"); "
            "load(api.cnc_actf, \"cnc_actf\");"
        )

        self.assertEqual(validate_focas_loader_type_safety(source), [])

    def test_code_only_template_quotes_nc_payload_and_inserts_core(self) -> None:
        class CoreLlm:
            enabled = True

            def invoke_json(self, system_prompt: str, user_prompt: str) -> dict:
                return {"api_fragments": [{
                    "step_id": "S001",
                    "apis": ["cnc_actf"],
                    "calls": [{
                        "api_name": "cnc_actf",
                        "call_expr": "api.cnc_actf(handle, &__out_actualfeed)",
                        "parameter_values": {},
                    }],
                }]}

        from smart_traffic_agent.agents.code_generator import render_code_only_cpp_artifacts

        state = WorkflowState(
            request=TaskRequest(
                description="read actual feed",
                task_id="template001",
                permissions={"code_only_evaluation": True, "representative_scenario": True},
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-template001",
            task_id="template001",
            scenario_type="coordinate_motion",
            scenario_goal="read feed",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "during", "feed", "ReadFeedSpeed", {}, protocol_function="cnc_actf")],
            expected_outputs=[],
        )

        source, fragment = render_code_only_cpp_artifacts(state, state.plan.steps, "O1234\nM30\n", CoreLlm())

        self.assertIn("#include <chrono>", source)
        self.assertIn("#include <thread>", source)
        self.assertIn("SCENARIO_CORE_BEGIN", fragment)
        # Code-only rendering uses the same one-request API-fragment contract
        # as production capture when an LLM is configured.
        self.assertIn("api.cnc_actf(handle, &S001_actualfeed)", fragment)

    def test_scenario_fragment_rejects_main_and_loader_ownership_leaks(self) -> None:
        fragment = build_scenario_fragment(
            'int main(){ return 0; }\nLoadLibraryW(L"Fwlib32.dll");'
        )

        diagnostics = validate_scenario_fragment(fragment)

        self.assertTrue(any("main()" in item for item in diagnostics))
        self.assertTrue(any("LoadLibraryW" in item for item in diagnostics))

    def test_scenario_fragment_accepts_minimal_business_function(self) -> None:
        fragment = build_scenario_fragment(
            'ret = api.cnc_actf(handle, &actual_feed); ok = check_rc(ret, "cnc_actf") && ok;'
        )

        self.assertEqual(validate_scenario_fragment(fragment), [])

    def test_capture_call_contract_rejects_legacy_four_argument_form(self) -> None:
        fragment = '''
ret = ctx.call("cnc_statinfo",
    [&]() { return api.cnc_statinfo(handle, &status); },
    {{"FlibHndl", std::to_string(handle)}},
    {{"run", std::to_string(status.run)}});
'''
        diagnostics = validate_capture_context_call_contract(fragment)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("CAPTURE_CALL_SIGNATURE_MISMATCH", diagnostics[0])
        self.assertIn("api=cnc_statinfo", diagnostics[0])
        self.assertIn("expected=6", diagnostics[0])
        self.assertIn("actual=4", diagnostics[0])
        self.assertIn("line=2", diagnostics[0])

    def test_capture_call_contract_accepts_nested_six_argument_form(self) -> None:
        fragment = '''
ret = ctx.call("S002", "cnc_statinfo", "FlibHndl=" + std::to_string(handle),
    api.cnc_statinfo,
    [&]() -> short { return api.cnc_statinfo(handle, &status); },
    [&](short) -> std::string {
        return "run=" + std::to_string(status.run) +
               ";data=" + std::to_string(status.data[0]);
    });
'''
        self.assertEqual(validate_capture_context_call_contract(fragment), [])

    def test_capture_call_contract_ignores_comments_and_string_literals(self) -> None:
        fragment = r'''
// ctx.call("cnc_fake", one, two, three);
const char* text = "ctx.call(\\\"cnc_fake\\\", one, two, three)";
ret = ctx.call("S003", "cnc_statinfo", "note=(a,b)", api.cnc_statinfo,
    [&]() -> short { return api.cnc_statinfo(handle, &status); },
    [&](short) -> std::string { return "run=" + std::to_string(status.run); });
'''
        self.assertEqual(validate_capture_context_call_contract(fragment), [])

    def test_capture_call_contract_rejects_crossed_api_names(self) -> None:
        fragment = (
            'ret = ctx.call("S004", "cnc_statinfo", "", api.cnc_alarm2, '
            '[&]() -> short { return api.cnc_alarm2(handle, &alarm); }, '
            '[&](short) -> std::string { return ""; });'
        )
        diagnostics = validate_capture_context_call_contract(fragment)
        self.assertTrue(any("CAPTURE_CALL_API_NAME_MISMATCH" in item for item in diagnostics))

    def test_api_fragment_rejects_extra_main_before_render(self) -> None:
        fragment = (
            'int main() { return 0; }\n'
            'ret = ctx.call("S005", "cnc_statinfo", "", api.cnc_statinfo, '
            '[&]() -> short { return api.cnc_statinfo(handle, &status); }, '
            '[&](short) -> std::string { return ""; });'
        )
        with self.assertRaisesRegex(ValueError, "API_FRAGMENT_STATEMENTS_ONLY"):
            _validate_api_fragment_call(fragment, ["cnc_statinfo"])

    def test_validate_official_output_usage_accepts_api_alias_for_rdposition(self) -> None:
        code_spec = CodeSpec(
            scenario="coordinate_motion",
            api_contracts=[
                ApiContract(
                    call_id="test-call",
                    phase="during",
                    operation_id="test-operation",
                    tool_name="cnc_rdposition",
                    prototype="short cnc_rdposition(unsigned short, short, short*, ODBPOS*)",
                    parameter_types=["unsigned short", "short", "short*", "ODBPOS*"],
                    output_fields=["abs", "mach", "rel", "dist"],
                )
            ]
        )
        source = "ODBPOS positions[3]{}; api.rdposition(handle, -1, &count, positions);"

        self.assertEqual(validate_official_output_usage(source, code_spec), [])

    def test_normalize_nc_download_payload_handles_array_declaration(self) -> None:
        source = (
            'const char program[] = "O1234\\n" "M30\\n" "%\\n"; '
            'api.download3(handle, &length, program);'
        )

        normalized = normalize_nc_download_payload(source)

        self.assertIn('"\\n"\n        "O1234', normalized)
        self.assertIn('"M30\\n" "%"', normalized)

    def test_validate_poselem_scalar_access_rejects_subscripted_data(self) -> None:
        source = (
            "static void dumpPosEl(const char* label, const POSELM& p) { "
            "use(p.data[0], p.data[1], p.data[2]); }"
        )

        diagnostics = validate_poselem_scalar_access(source)

        self.assertEqual(len(diagnostics), 1)
        self.assertIn("POSELM.data is a scalar long", diagnostics[0])
        self.assertIn("p.data[0]", diagnostics[0])

    def test_validate_poselem_scalar_access_accepts_odbpos_axis_array(self) -> None:
        source = (
            "ODBPOS positions[3]{}; "
            "use(positions[0].abs.data, positions[1].mach.data, "
            "positions[2].rel.data, positions[2].dist.data);"
        )

        self.assertEqual(validate_poselem_scalar_access(source), [])

    def test_validate_generated_blocks_poselem_array_misuse(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "struct POSELM { long data; }; "
                "void dump(const POSELM& p) { use(p.data[0]); }"
            ),
            "O1234\nM30\n",
        )

        blocking = [item for item in diagnostics if "POSELM.data is a scalar long" in item]
        self.assertEqual(len(blocking), 1)

    def test_select_codegen_executable_steps_keeps_api_steps_for_llm(self) -> None:
        selected, skipped = select_codegen_executable_steps(
            [
                PlanStep("S001", "before", "read status", "ReadRunStatus", {}),
                PlanStep("S002", "during", "write parameter", "WriteParameter", {}),
            ]
        )

        self.assertEqual([step.interface_name for step in selected], ["ReadRunStatus", "WriteParameter"])
        self.assertEqual(skipped, [])

    def test_select_codegen_executable_steps_skips_analysis_steps(self) -> None:
        selected, skipped = select_codegen_executable_steps(
            [
                PlanStep("S001", "during", "read status", "ReadRunStatus", {}),
                PlanStep("S_FINAL", "after", "evaluate traffic quality", "EvaluateTrafficVariation", {}),
            ]
        )

        self.assertEqual([step.interface_name for step in selected], ["ReadRunStatus"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("analysis/evaluation", skipped[0])

    def test_select_codegen_executable_steps_skips_npcap_lifecycle(self) -> None:
        selected, skipped = select_codegen_executable_steps(
            [
                PlanStep(
                    "S001", "before", "Start Npcap capture", "Npcap loopback capture",
                    protocol_function="N/A (capture preflight)",
                ),
                PlanStep("S002", "during", "read CNC status", "cnc_statinfo", protocol_function="cnc_statinfo"),
            ]
        )

        self.assertEqual([step.step_id for step in selected], ["S002"])
        self.assertEqual(len(skipped), 1)
        self.assertIn("deterministic runner", skipped[0])

    def test_preflight_compile_keeps_success_before_cleanup(self) -> None:
        class FakeCompileTool:
            name = "compile_generated_cpp"
            description = "test compiler"

            def invoke(self, tool_input):
                tool_input.executable_path.write_text("", encoding="utf-8")
                return CompileCppOutput(0, "api_script.cpp\n", "", tool_input.executable_path)

            def input_summary(self, tool_input):
                return {"source_path": str(tool_input.source_path)}

            def output_summary(self, tool_output):
                return {"return_code": tool_output.return_code}

            def output_succeeded(self, tool_output):
                return tool_output.success

            def output_error(self, tool_output):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "api_script.cpp"
            source_path.write_text("int main(){return 0;}", encoding="utf-8")

            with patch("smart_traffic_agent.agent_tools.CompileGeneratedCppTool", return_value=FakeCompileTool()):
                error = preflight_compile_generated_cpp(source_path, work_dir)

            self.assertEqual(error, "")
            self.assertFalse((work_dir / "preflight_api_script.exe").exists())

    def test_preflight_compile_records_compile_tool_call(self) -> None:
        from smart_traffic_agent.tool_runtime import ToolRegistry, ToolRuntime

        class RecordingCompileTool:
            name = "compile_generated_cpp"
            description = "test compiler"

            def invoke(self, tool_input):
                tool_input.executable_path.write_text("", encoding="utf-8")
                return CompileCppOutput(0, "", "", tool_input.executable_path)

            def input_summary(self, tool_input):
                return {"source_path": str(tool_input.source_path)}

            def output_summary(self, tool_output):
                return {"return_code": tool_output.return_code}

            def output_succeeded(self, tool_output):
                return tool_output.success

            def output_error(self, tool_output):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "api_script.cpp"
            source_path.write_text("int main(){return 0;}", encoding="utf-8")
            registry = ToolRegistry()
            registry.register(RecordingCompileTool())
            runtime = ToolRuntime(registry, history=[])

            self.assertEqual(preflight_compile_generated_cpp(source_path, work_dir, tool_runtime=runtime), "")
            self.assertEqual([record.tool_name for record in runtime.history], ["compile_generated_cpp"])

    def test_compile_error_repair_prompt_returns_corrected_cpp(self) -> None:
        llm = FakeCodegenLlm()
        state = WorkflowState(request=TaskRequest(description="generate FOCAS traffic", task_id="codegen001"))
        state.plan = ExecutionPlan(
            plan_id="plan-codegen001",
            task_id="codegen001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "before", "connect", "OpenConnection", {}, protocol_function="cnc_allclibhndl3")],
            expected_outputs=[],
        )

        repaired = repair_cpp_api_script_after_compile_error(
            state,
            state.plan.steps,
            "O1234\nM30\n",
            "bad cpp",
            "api_script.cpp(10): error C2362",
            llm,
        )

        self.assertEqual(repaired, "int main(){return 0;}")
        self.assertIn("error C2362", llm.prompts[-1])

    def test_compile_repair_revalidates_preserved_helper_abi(self) -> None:
        llm = FakeCodegenLlm(repair_cpp=(
            "#include <Fwlib32.h>\n"
            "void dump(const ODBST& status) { use(status.aut, status.run); }\n"
            "int main(){return 0;}"
        ))
        state = WorkflowState(
            request=TaskRequest(
                description="generate FOCAS traffic",
                task_id="codegen001",
                permissions={"code_only_evaluation": True},
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-codegen001",
            task_id="codegen001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[],
            expected_outputs=[],
        )
        previous = (
            "#include <Fwlib32.h>\n"
            "void dump(const ODBST& status) { use(status.manual, status.warning); }\n"
            "int main(){return 1;}"
        )

        repaired = repair_cpp_api_script_after_compile_error(
            state,
            [],
            "O1234\nM30\n",
            previous,
            "error C2039: manual is not a member of odbst",
            llm,
            code_only=True,
        )

        self.assertNotIn("status.manual", repaired)
        self.assertNotIn("status.warning", repaired)
        self.assertIn("int main(){return 0;}", repaired)

    def test_capture_core_deduplicates_harness_owned_state(self) -> None:
        complete = ("short run_generated_scenario(FocasApi& api, unsigned short handle, CaptureContext& ctx)\n"
                    "{\n    short ret = EW_OK;\n    bool ok = true;\n"
                    "short ret = EW_OK;\nbool ok = true;\nreturn ret;\n}\n")
        cleaned = normalize_capture_core(complete)
        self.assertEqual(cleaned.count("short ret = EW_OK;"), 1)
        self.assertEqual(cleaned.count("bool ok = true;"), 1)
        body = normalize_capture_core("short ret = EW_OK;\nbool ok = true;\nret = EW_OK;\n")
        self.assertEqual(body, "ret = EW_OK;\n")

    def test_codegen_repairs_fragment_after_preflight_compile_failure(self) -> None:
        repaired_fragment = "short run_generated_scenario(FocasApi& api, unsigned short handle){ return EW_OK; }\n"
        llm = FakeCodegenLlm()
        state = WorkflowState(
            request=TaskRequest(
                description="generate FOCAS traffic",
                task_id="codegen001",
                target_environment="ncguide-generated-cpp",
                permissions={"code_only_evaluation": True},
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-codegen001",
            task_id="codegen001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "before", "connect", "OpenConnection", {}, protocol_function="cnc_allclibhndl3")],
            expected_outputs=[],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("smart_traffic_agent.agents.code_generator.generate_nc_program", return_value=("O1234\nM30\n", [])),
                patch(
                    "smart_traffic_agent.agents.code_generator.generate_cpp_api_script",
                    side_effect=AssertionError("legacy full-source generator was called"),
                ),
                patch(
                    "smart_traffic_agent.agents.code_generator.render_code_only_cpp_artifacts",
                    return_value=('#include "scenario_fragment.inc"\nint main(){return 0;}\n', repaired_fragment),
                ),
                patch(
                    "smart_traffic_agent.agents.code_generator.CodeGenerationAgent._validate_focas_cpp",
                    return_value=type("ValidationOutput", (), {"errors": [], "warnings": []})(),
                ),
                patch(
                    "smart_traffic_agent.agents.code_generator.repair_scenario_fragment_after_compile_error",
                    return_value=repaired_fragment,
                ) as repair_mock,
                patch(
                    "smart_traffic_agent.agents.code_generator.preflight_compile_generated_cpp",
                    side_effect=["api_script.cpp(10): error C2362", ""],
                ) as compile_mock,
            ):
                with self.assertRaisesRegex(ValueError, r"Generated C\+\+ failed MSVC preflight compilation"):
                    CodeGenerationAgent(llm_client=llm).run(state, Path(temp_dir))
                repair_mock.assert_not_called()

        # Deterministic C++ failures are reported with compiler evidence; the
        # removed model-authored full-source repair path must not run.
        self.assertEqual(compile_mock.call_count, 1)

    def test_only_delete_permission_diagnostics_hard_block_codegen(self) -> None:
        diagnostics = [
            "BLOCKING: Single Block execution must count and drive every effective NC segment.",
            "BLOCKING: generated C++ uses cnc_delall without explicit allow_delete_all_programs permission.",
        ]

        self.assertEqual(
            hard_blocking_codegen_diagnostics(diagnostics),
            ["BLOCKING: generated C++ uses cnc_delall without explicit allow_delete_all_programs permission."],
        )

    def test_hard_blocking_codegen_includes_pcap_and_fake_coverage_guards(self) -> None:
        diagnostics = [
            "BLOCKING: generated C++ creates fake/minimal pcap marker files instead of attempting real packet capture.",
            "BLOCKING: generated C++ must not emit fake manifest coverage rows without real API calls; found ManifestCoverage.",
        ]

        self.assertEqual(hard_blocking_codegen_diagnostics(diagnostics), diagnostics)

    def test_validate_generated_rejects_minimal_pcap_marker_writer(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "void WriteMinimalPcap(const string& path){} "
                "void CaptureMarker(){}"
            ),
            "O1234\nM30\n",
            [],
            require_real_pcap_capture=True,
        )

        self.assertTrue(any("fake/minimal pcap" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_real_capture_comments_do_not_trigger_fake_capture_diagnostic(self) -> None:
        source = """
        // minimal pcap dynamic declarations
        /* pcap marker is not a runtime marker */
        void Capture() {
            auto handle = pcap_open_live(device, 65536, 1, 1000, errbuf);
            auto dumper = pcap_dump_open(handle, file.c_str());
            pcap_dispatch(handle, 1, callback, nullptr);
            pcap_dump(dumper, header, packet);
            pcap_dump_flush(dumper);
        }
        PacketSniffer sniffer(networkDevice, pcapFile);
        auto envDevice = getenv("PCAP_NETWORK_DEVICE");
        output << "pcap_capture_enabled=true";
        """
        self.assertEqual(validate_real_packet_capture_usage(source), [])

    def test_real_capture_must_abort_when_start_fails(self) -> None:
        source = """
        void Run() {
            PacketSniffer sniffer(networkDevice, pcapFile);
            bool pcapReady = sniffer.Start();
            cnc_statinfo(handle, &status);
        }
        pcap_open_live pcap_dump_open pcap_dispatch pcap_dump pcap_dump_flush
        getenv("PCAP_NETWORK_DEVICE")
        output << "pcap_capture_enabled=true";
        """
        diagnostics = validate_real_packet_capture_usage(source)
        self.assertTrue(any("abort before any FOCAS call" in item for item in diagnostics))

    def test_real_capture_rejects_continue_without_pcap(self) -> None:
        source = '''
        PacketSniffer sniffer(networkDevice, pcapFile);
        pcap_open_live pcap_dump_open pcap_dispatch pcap_dump pcap_dump_flush
        getenv("PCAP_NETWORK_DEVICE");
        output << "pcap_capture_started=true";
        output << "PCAP_UNAVAILABLE_CONTINUING";
        '''
        self.assertTrue(any("continues after Npcap failed" in issue for issue in validate_real_packet_capture_usage(source)))

    def test_single_block_ready_gate_accepts_run_one_stationary(self) -> None:
        bad = "bool ready = st.run == 0 && st.motion == 0;"
        good = "bool ready = (st.run == 0 || st.run == 1) && st.motion == 0;"
        self.assertTrue(validate_cycle_start_ready_states(bad))
        self.assertEqual(validate_cycle_start_ready_states(good), [])

    def test_validate_generated_requires_packetsniffer_real_capture_api(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "class PacketCapture{}; int main(){ PacketCapture capture; }"
            ),
            "O1234\nM30\n",
            [],
            require_real_pcap_capture=True,
        )

        blocking = blocking_codegen_diagnostics(diagnostics)
        self.assertTrue(any("PacketSniffer sniffer" in item for item in blocking))
        self.assertTrue(any("pcap_open_live" in item for item in blocking))

    def test_validate_generated_rejects_aggregate_api_manifest_coverage_rows(self) -> None:
        diagnostics = validate_generated(
            (
                hard_pcap_valid_cpp(
                    'LogCall(inputCsv, outputCsv, idx, "S016", "after", "LifecycleManifestCoverage", '
                    '"cnc_dwnstart;cnc_download;cnc_cdownload", "bounded_lifecycle_artifact=true", '
                    '0, "coverage_role=target;best_effort_lifecycle_segment_recorded");'
                )
            ),
            "O1234\nM30\n",
            [],
        )

        blocking = blocking_codegen_diagnostics(diagnostics)
        self.assertTrue(any("aggregate API literal" in item for item in blocking))
        self.assertTrue(any("fake manifest coverage" in item for item in blocking))

    def test_validate_generated_accepts_packetsniffer_capture_shape(self) -> None:
        diagnostics = validate_generated(
            hard_pcap_valid_cpp(),
            "O1234\nM30\n",
            [],
            require_official_focas_header=True,
            require_real_pcap_capture=True,
        )

        self.assertFalse(any("fake/minimal pcap" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_blocks_skip_markers_for_supported_steps(self) -> None:
        diagnostics = validate_generated(
            (
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_statinfo SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "read status", "ReadRunStatus", {})],
        )

        self.assertTrue(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_requires_lifecycle_implementation_for_planned_steps(self) -> None:
        diagnostics = validate_generated(
            "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl cnc_statinfo",
            "O1234\nM30\n",
            [
                PlanStep("S001", "before", "upload", "UploadProgram", {}),
                PlanStep("S002", "before", "select", "SelectProgram", {}),
                PlanStep("S003", "before", "read program", "ReadProgramNumber", {}),
                PlanStep("S004", "during", "start", "StartProgram", {}),
            ],
        )

        blocking = blocking_codegen_diagnostics(diagnostics)
        self.assertTrue(any("cnc_dwnstart3" in item for item in blocking))
        self.assertTrue(any("cnc_search" in item for item in blocking))
        self.assertTrue(any("cnc_rdprgnum" in item for item in blocking))
        self.assertTrue(any("SetCursorPos" in item for item in blocking))
        self.assertTrue(any("cycle_start_ready_gate" in item for item in blocking))

    def test_validate_generated_requires_cycle_start_ready_gate(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_statinfo SetCursorPos mouse_event"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertTrue(any("cycle_start_ready_gate" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_accepts_cycle_start_ready_gate_marker(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "NCGUIDE_CYCLE_START_X NCGUIDE_CYCLE_START_Y NCGUIDE_CLICK_MODE NCGUIDE_WINDOW_TITLE "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "cnc_rdposition pos[i].dist distance_to_go program_completion_gate "
                "completed timeout waited_ms last_run last_motion run==0 run==1 motion==0 "
                "RunUploadedProgramToCompletion effective_nc_segment_count "
                "expected_nc_segment_count cycle_start_click_count M30 "
                "struct POSELM { long data; short dec; short unit; short disp; char name; char suff; }; "
                "struct ODBPOS_MOCK { POSELM abs; POSELM mach; POSELM rel; POSELM dist; };"
            ),
            "O1234\nM30\n",
            [
                PlanStep("S001", "during", "start", "StartProgram", {}),
                PlanStep("S002", "during", "read position", "ReadPosition", {}),
            ],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_non_official_rdposition_buffer(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "short (__stdcall *cnc_rdposition)(unsigned short, short, short*, void*); "
                "struct POS_ITEM { long data; short dec; }; struct ODBPOS_MOCK { POS_ITEM pos[32]; };"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "position", "ReadPosition", {}, protocol_function="cnc_rdposition")],
            require_official_focas_header=False,
        )

        self.assertTrue(any("ODBPOS abs/mach/rel/dist fields" in item for item in diagnostics))

    def test_validate_generated_requires_program_completion_gate(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "NCGUIDE_CYCLE_START_X NCGUIDE_CYCLE_START_Y NCGUIDE_CLICK_MODE NCGUIDE_WINDOW_TITLE "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "cnc_rdposition pos[i].dist distance_to_go"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertTrue(any("program_completion_gate" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_distance_to_go_for_position_sampling(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_rdposition"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "read position", "ReadPosition", {})],
        )

        self.assertTrue(any("remaining movement" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_accepts_cnc_distance_for_remaining_motion(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "NCGUIDE_CYCLE_START_X NCGUIDE_CYCLE_START_Y NCGUIDE_CLICK_MODE NCGUIDE_WINDOW_TITLE "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "program_completion_gate completed timeout waited_ms last_run last_motion run==0 run==1 motion==0 "
                "cnc_distance distance_to_go RunUploadedProgramToCompletion effective_nc_segment_count "
                "expected_nc_segment_count cycle_start_click_count M30"
            ),
            "O1234\nM30\n",
            [
                PlanStep("S001", "during", "start", "StartProgram", {}),
                PlanStep("S002", "during", "read remaining distance", "ReadDistanceToGo", {}),
            ],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_requires_whole_program_single_block_loop(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "program_completion_gate completed timeout waited_ms last_run last_motion run==0 motion==0 "
                "cnc_distance distance_to_go"
            ),
            "O1234\nG90 G54\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertTrue(
            any("every effective NC segment" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_accepts_semantic_single_block_loop_without_fixed_helper_name(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_statinfo SetCursorPos mouse_event "
                "cycle_start_ready_gate program_completion_gate completed timeout waited_ms "
                "last_run last_motion run==0 cnc_distance distance_to_go M30 "
                "expected_nc_segment_count cycle_start_click_count "
                "int CountSegments(const string& payload); "
                "int expectedSegments=CountSegments(payload); int clickCount=0; "
                "for(int segment=0; segment<expectedSegments; ++segment){ "
                "TriggerNcGuideCycleStart(); clickCount++; }"
            ),
            "O1234\nG90 G54\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertFalse(
            any("every effective NC segment" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_accepts_prefix_increment_click_count(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_statinfo cnc_start "
                "cycle_start_ready_gate program_completion_gate completed timeout waited_ms "
                "last_run last_motion motion == 0 cnc_distance distance_to_go M30 "
                "expected_nc_segment_count cycle_start_click_count "
                "int CountEffectiveNcSegments(const string& payload); "
                "int expectedSegments=CountEffectiveNcSegments(payload); int clickCount=0; "
                "for(int seg=1; seg<=expectedSegments; ++seg){ "
                "WaitUntilCycleStartReady(); short ret=cnc_start_fn(handle); if(ret==0) ++clickCount; }"
            ),
            "O1234\nG90 G54\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertFalse(
            any("every effective NC segment" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_accepts_bounded_completion_polling_helper(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_statinfo SetCursorPos mouse_event "
                "cycle_start_ready_gate program_completion_gate completed timeout waited_ms "
                "last_run last_motion cnc_distance distance_to_go M30 "
                "expected_nc_segment_count cycle_start_click_count RunUploadedProgramToCompletion "
                "bool completed=false; while(true){ cnc_statinfo; completed=IsIdleLike(status); "
                "if(completed) break; steady_clock timeout_ms sleep_for; }"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertFalse(
            any("must poll cnc_statinfo" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_requires_program_identity_gate(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_dwnstart3 cnc_download3 cnc_dwnend3 cnc_search cnc_rdprgnum "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "program_completion_gate completed timeout waited_ms last_run last_motion run==0 motion==0 "
                "cnc_distance distance_to_go RunUploadedProgramToCompletion effective_nc_segment_count "
                "expected_nc_segment_count cycle_start_click_count M30"
            ),
            "O1234\nG90 G54\nM30\n",
            [
                PlanStep("S001", "before", "upload", "UploadProgram", {}),
                PlanStep("S002", "before", "select", "SelectProgram", {}),
                PlanStep("S003", "before", "verify", "ReadProgramNumber", {}),
                PlanStep("S004", "during", "start", "StartProgram", {}),
                PlanStep("S005", "during", "distance", "ReadDistanceToGo", {}),
            ],
        )

        self.assertTrue(any("PROGRAM_NOT_VERIFIED" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_rejects_leading_percent_download_payload(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 cnc_delete "
                'static const char* payload = "%\\n" "O1234\\n" "M30\\n" "%\\n";'
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertTrue(any("must begin with LF" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_non_destructive_availability_flow(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 cnc_delete"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertTrue(
            any("selected_program_number" in item or "program directory" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_accepts_runtime_choose_unused_program_flow(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_rdprogdir3 target_program_exists program_number_available "
                "expected_program_number exact_match preferred_program_number selected_program_number "
                "collision_strategy=choose_unused"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_unsafe_single_program_delete(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_rdprogdir3 cnc_delete target_program_exists program_number_available "
                "expected_program_number preferred_program_number selected_program_number"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertTrue(any("exact confirmed single-program collision" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_accepts_exact_single_program_delete_flow(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_rdprogdir3 cnc_delete target_program_exists program_number_available "
                "expected_program_number exact_match preferred_program_number selected_program_number "
                "collision_strategy=delete_exact_conflict delete_called=true delete_target_program=O1234 "
                "delete_ret PROGRAM_REPLACEMENT_FAILED return 3"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_unapproved_delete_all(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 cnc_delall"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertTrue(any("without explicit" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_accepts_explicitly_authorized_delete_all(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_delall delete_all_authorized=true collision_strategy=delete_all_authorized "
                "PROGRAM_REPLACEMENT_FAILED return 3"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
            allow_delete_all_programs=True,
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_rdprogdir_family_abi_mismatch(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "short (__stdcall *cnc_rdprogdir)(unsigned short, short, short*, short*, void*);"
            ),
            "O1234\nM30\n",
            [
                PlanStep(
                    "S001",
                    "before",
                    "check program",
                    "CheckTargetProgramExists",
                    {},
                    protocol_function="cnc_rdprogdir",
                )
            ],
        )

        self.assertTrue(any("requires 6 arguments" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_accepts_exact_rdprogdir_prototype(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "short (__stdcall *cnc_rdprogdir)(unsigned short, short, short, short, unsigned short, void*);"
            ),
            "O1234\nM30\n",
            [
                PlanStep(
                    "S001",
                    "before",
                    "check program",
                    "CheckTargetProgramExists",
                    {},
                    protocol_function="cnc_rdprogdir",
                )
            ],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_odbaxis_length_pointer(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "short (__stdcall *cnc_absolute)(unsigned short, short, short*, void*);"
            ),
            "O1234\nM30\n",
            [
                PlanStep(
                    "S001",
                    "during",
                    "absolute position",
                    "ReadPosition",
                    {},
                    protocol_function="cnc_absolute",
                )
            ],
        )

        self.assertTrue(any("length value, not a pointer" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_rejects_long_only_variation_helper_for_status(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_statinfo "
                "bool HasVariation(const std::vector<long>& values); "
                "bool changed = HasVariation(metrics.run);"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "status", "ReadRunStatus", {})],
        )

        self.assertTrue(any("type-generic template" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_program_completion_wait_logic(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "program_completion_gate cnc_rdposition pos[i].dist distance_to_go"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "start", "StartProgram", {})],
        )

        self.assertTrue(any("program_completion_gate must poll" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_blocks_windows_minmax_macro_collision(self) -> None:
        diagnostics = validate_generated(
            (
                "#include <windows.h>\n"
                "#include <algorithm>\n"
                "int main(){ return std::min(1, 2); }\n"
                "LoadLibraryW GetProcAddress cnc_allclibhndl3"
            ),
            "O1234\nM30\n",
            [],
        )

        self.assertTrue(any("NOMINMAX" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_demo_style_focas_dll_loading(self) -> None:
        diagnostics = validate_generated(
            (
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
                "int main(){ return 0; }"
            ),
            "O1234\nM30\n",
            [],
        )

        self.assertTrue(any("FOCAS_DLL_DIR" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_official_focas_header_when_enabled(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl"
            ),
            "O1234\nM30\n",
            [],
            require_official_focas_header=True,
        )

        self.assertTrue(any("official" in item.lower() and "Fwlib32.h" in item for item in diagnostics))

    def test_validate_generated_accepts_official_decltype_for_dynamic_focas_api(self) -> None:
        diagnostics = validate_generated(
            (
                "#include <Fwlib32.h>\n"
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW "
                "auto fn = reinterpret_cast<decltype(&::cnc_allclibhndl3)>("
                "GetProcAddress(dll, \"cnc_allclibhndl3\")); cnc_freelibhndl"
            ),
            "O1234\nM30\n",
            [],
            require_official_focas_header=True,
        )

        self.assertFalse(any("official Fwlib32.h declaration" in item for item in diagnostics))

    def test_validate_generated_rejects_handwritten_dynamic_focas_prototype(self) -> None:
        diagnostics = validate_generated(
            (
                "#include <Fwlib32.h>\n"
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW "
                "using api_t = short (__stdcall *)(const char*, unsigned short, long, unsigned short*); "
                "auto fn = reinterpret_cast<api_t>(GetProcAddress(dll, \"cnc_allclibhndl3\")); "
                "cnc_freelibhndl"
            ),
            "O1234\nM30\n",
            [],
            require_official_focas_header=True,
        )

        self.assertTrue(any("official Fwlib32.h declaration" in item for item in diagnostics))

    def test_validate_generated_checks_resolve_helper_against_official_declaration(self) -> None:
        diagnostics = validate_generated(
            (
                "#include <Fwlib32.h>\n"
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "using api_t = short (__stdcall *)(unsigned short, void*); "
                "Resolve(dll, \"cnc_statinfo\", fn); cnc_allclibhndl3 cnc_freelibhndl"
            ),
            "O1234\nM30\n",
            [],
            require_official_focas_header=True,
        )

        self.assertTrue(any("cnc_statinfo" in item and "official Fwlib32.h declaration" in item for item in diagnostics))

    def test_codegen_agent_can_retrieve_direct_api_knowledge(self) -> None:
        state = WorkflowState(request=TaskRequest(description="generate coordinate traffic", task_id="codegen001"))
        state.plan = ExecutionPlan(
            plan_id="plan-codegen001",
            task_id="codegen001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="simulator",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "during", "read position", "ReadPosition", {}, protocol_function="cnc_rdposition")],
            expected_outputs=[],
        )

        rows = retrieve_codegen_knowledge_context(sample_knowledge(), state, state.plan.steps)

        self.assertTrue(rows)
        self.assertTrue(any("ReadPosition" in str(row) or "coordinate" in str(row) for row in rows))

    def test_steps_for_prompt_includes_parameter_generation_strategy(self) -> None:
        rows = steps_for_prompt(
            [
                PlanStep(
                    "S001",
                    "during",
                    "read position",
                    "ReadPosition",
                    {
                        "parameter_generation": [
                            {"name": "type", "kind": "enum", "values": [-1, 0, 1]},
                            {"name": "axis_count", "kind": "range", "min": 1, "max": 8, "samples": [1, 4, 8]},
                        ]
                    },
                    protocol_function="cnc_rdposition",
                )
            ]
        )

        strategy = rows[0]["parameters"]["parameter_generation"]

        self.assertEqual(strategy[0]["kind"], "enum")
        self.assertEqual(strategy[1]["kind"], "range")

    def test_code_generation_fast_path_uses_render_tool_before_compile(self) -> None:
        from smart_traffic_agent.agent_tools import RenderFocasCppTool, WriteGeneratedArtifactTool
        from smart_traffic_agent.tool_runtime import ToolRegistry, ToolRuntime

        state = WorkflowState(
            request=TaskRequest(
                description="read CNC status",
                task_id="render-flow001",
                target_environment="ncguide-generated-cpp",
                permissions={"code_only_evaluation": True},
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-render-flow001",
            task_id="render-flow001",
            scenario_type="coordinate_motion",
            scenario_goal="read status",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "during", "read status", "ReadRunStatus", {}, protocol_function="cnc_statinfo")],
            expected_outputs=[],
            code_spec=CodeSpec(scenario="coordinate_motion", api_contracts=[]),
        )
        harness = '#include "scenario_fragment.inc"\nint main(){return 0;}\n'
        fragment = "short run_generated_scenario(FocasApi& api, unsigned short handle){ return EW_OK; }\n"
        registry = ToolRegistry()
        registry.register(WriteGeneratedArtifactTool())
        registry.register(RenderFocasCppTool())
        registry.register(ValidateFocasCppTool())
        runtime = ToolRuntime(registry, history=[])
        agent = CodeGenerationAgent(llm_client=FakeCodegenLlm())
        agent.llm_client.enabled = False
        agent.tool_runtime = runtime

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("smart_traffic_agent.agents.code_generator.generate_nc_program", return_value=("O1234\nM30\n", [])),
                patch(
                    "smart_traffic_agent.agents.code_generator.generate_cpp_api_script",
                    side_effect=AssertionError("legacy full-source generator was called"),
                ),
                patch("smart_traffic_agent.agents.code_generator.render_code_only_cpp_artifacts", return_value=(harness, fragment)),
                patch("smart_traffic_agent.agents.code_generator.validate_generated", return_value=[]),
                patch("smart_traffic_agent.agents.code_generator.preflight_compile_generated_cpp", return_value=""),
            ):
                result = agent.run(state, Path(temp_dir))

            generated = Path(temp_dir) / "generated"
            self.assertEqual(result.stage, "execution")
            self.assertEqual((generated / "api_script.cpp").read_text(encoding="utf-8"), harness)
            self.assertTrue((generated / "scenario_fragment.inc").exists())
            render_records = [item for item in runtime.history if item.tool_name == "render_focas_cpp"]
            self.assertEqual(len(render_records), 1)
            self.assertTrue(render_records[0].success)


    def test_validate_focas_argument_semantics_rejects_swapped_axis_length(self) -> None:
        source = "api.cnc_relative(handle, relLen, static_cast<short>(-1), &relative);"
        diagnostics = validate_focas_argument_semantics(source)
        self.assertTrue(any("arguments are reversed" in item for item in diagnostics))

    def test_validate_focas_argument_semantics_accepts_documented_order(self) -> None:
        source = "api.cnc_relative(handle, axis, relLen, &relative);"
        self.assertEqual(validate_focas_argument_semantics(source), [])

    def test_validate_focas_argument_semantics_rejects_missing_axis_and_odbpos(self) -> None:
        source = "ODBPOS snap{}; api.cnc_relative(handle, relLen, &snap);"
        diagnostics = validate_focas_argument_semantics(source)
        self.assertTrue(any("requires (handle, axis, length" in item for item in diagnostics))
        self.assertTrue(any("must use the SDK ODBAXIS" in item for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
