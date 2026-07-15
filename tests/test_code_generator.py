from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smart_traffic_agent.agents.code_generator import (
    CodeGenerationAgent,
    blocking_codegen_diagnostics,
    hard_blocking_codegen_diagnostics,
    normalize_llm_nc_program,
    official_focas_abi_context,
    preflight_compile_generated_cpp,
    repair_cpp_api_script_after_compile_error,
    retrieve_codegen_knowledge_context,
    select_codegen_executable_steps,
    steps_for_prompt,
    validate_generated,
)
from smart_traffic_agent.agent_tools import CompileCppOutput
from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep, TaskRequest, WorkflowState


class FakeCodegenLlm:
    enabled = True

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.prompts.append(user_prompt)
        if "MSVC preflight compiler diagnostics" in user_prompt:
            return {"cpp_code": "int main(){return 0;}"}
        return {"ok": True, "diagnostics": []}


class CodeGenerationAgentTests(unittest.TestCase):
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

    def test_preflight_compile_keeps_success_before_cleanup(self) -> None:
        class FakeCompileTool:
            def invoke(self, tool_input):
                tool_input.executable_path.write_text("", encoding="utf-8")
                return CompileCppOutput(0, "api_script.cpp\n", "", tool_input.executable_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            source_path = work_dir / "api_script.cpp"
            source_path.write_text("int main(){return 0;}", encoding="utf-8")

            with patch("smart_traffic_agent.agent_tools.CompileGeneratedCppTool", return_value=FakeCompileTool()):
                error = preflight_compile_generated_cpp(source_path, work_dir)

            self.assertEqual(error, "")
            self.assertFalse((work_dir / "preflight_api_script.exe").exists())

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

    def test_codegen_repairs_once_after_preflight_compile_failure(self) -> None:
        llm = FakeCodegenLlm()
        state = WorkflowState(
            request=TaskRequest(
                description="generate FOCAS traffic",
                task_id="codegen001",
                target_environment="ncguide-generated-cpp",
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
                patch("smart_traffic_agent.agents.code_generator.generate_cpp_api_script", return_value=("bad cpp", [])),
                patch(
                    "smart_traffic_agent.agents.code_generator.preflight_compile_generated_cpp",
                    side_effect=["api_script.cpp(10): error C2362", ""],
                ) as compile_mock,
            ):
                result = CodeGenerationAgent(llm_client=llm).run(state, Path(temp_dir))

        self.assertEqual(result.stage, "execution")
        self.assertEqual(result.artifacts.api_script, "int main(){return 0;}")
        self.assertEqual(compile_mock.call_count, 2)

    def test_only_delete_permission_diagnostics_hard_block_codegen(self) -> None:
        diagnostics = [
            "BLOCKING: Single Block execution must count and drive every effective NC segment.",
            "BLOCKING: generated C++ uses cnc_delall without explicit allow_delete_all_programs permission.",
        ]

        self.assertEqual(
            hard_blocking_codegen_diagnostics(diagnostics),
            ["BLOCKING: generated C++ uses cnc_delall without explicit allow_delete_all_programs permission."],
        )

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
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "cnc_rdposition pos[i].dist distance_to_go program_completion_gate "
                "completed timeout waited_ms last_run last_motion run==0 motion==0 "
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

    def test_validate_generated_rejects_reduced_rdposition_buffer(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl "
                "short (__stdcall *cnc_rdposition)(unsigned short, short, short*, void*); "
                "struct POS_ITEM { long data; short dec; }; struct ODBPOS_MOCK { POS_ITEM pos[32]; };"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "during", "position", "ReadPosition", {}, protocol_function="cnc_rdposition")],
        )

        self.assertTrue(any("overwrite stack memory" in item for item in blocking_codegen_diagnostics(diagnostics)))

    def test_validate_generated_requires_program_completion_gate(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW "
                "LoadLibraryW GetProcAddress cnc_allclibhndl3 cnc_freelibhndl "
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
                "cnc_statinfo SetCursorPos mouse_event cycle_start_ready_gate "
                "program_completion_gate completed timeout waited_ms last_run last_motion run==0 motion==0 "
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
            any("program_number_available" in item for item in blocking_codegen_diagnostics(diagnostics))
        )

    def test_validate_generated_accepts_non_destructive_availability_flow(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_rdprogdir3 target_program_exists program_number_available "
                "expected_program_number exact_match TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertFalse(blocking_codegen_diagnostics(diagnostics))

    def test_validate_generated_rejects_default_single_program_delete(self) -> None:
        diagnostics = validate_generated(
            (
                "FOCAS_DLL_DIR GetEnvironmentVariableW SetDllDirectoryW LoadLibraryW GetProcAddress "
                "cnc_allclibhndl3 cnc_freelibhndl cnc_dwnstart3 cnc_download3 cnc_dwnend3 "
                "cnc_rdprogdir3 cnc_delete target_program_exists program_number_available "
                "expected_program_number exact_match TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED"
            ),
            "O1234\nM30\n",
            [PlanStep("S001", "before", "upload", "UploadProgram", {})],
        )

        self.assertTrue(any("preserve existing programs" in item for item in blocking_codegen_diagnostics(diagnostics)))

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
                "cnc_delall delete_all_authorized=true PROGRAM_REPLACEMENT_FAILED return 3"
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


if __name__ == "__main__":
    unittest.main()
