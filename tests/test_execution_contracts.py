"""Regression checks for capability-based production generation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from smart_traffic_agent.execution_contracts import (
    analyze_api_interactions, dependency_closure, execution_contract,
    api_sequence_rules, normalize_lifecycle_steps, program_operation_chain,
    validate_api_lifecycle_fragment,
)
from smart_traffic_agent.focas_capture_runtime import (
    guarded_core, motion_read_core, render_capture_harness,
    validate_production_source, wrap_capture_fragment,
    _validate_write_snapshot_restore,
)
from smart_traffic_agent.integrations.ncguide import default_focas_header_dir
from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.models import TaskRequest, WorkflowState
from smart_traffic_agent.workflow import requires_dynamic_quality_gate


ROOT = Path(__file__).resolve().parents[1]
NC = "%\nO6701\nG90 G54\nG00 X0 Y0 Z5\nG01 X16 F240\nM30\n%\n"


def plan(*steps: PlanStep, targets: tuple[str, ...] = ()) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="coordinate_motion",
        scenario_goal="capture", target_environment="ncguide-generated-cpp",
        nc_program_type="generated", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec("O6701"), steps=list(steps), expected_outputs=[],
        rag_context={"function_coverage_manifest": {
            "enabled": True,
            "selected_batch": [{"function": name, "coverage_role": "target"} for name in targets],
        }},
    )


def step(name: str, phase: str = "during", interface: str = "ReadValue") -> PlanStep:
    return PlanStep(name, phase, "capture", interface, protocol_function=name)


class ExecutionContractTests(unittest.TestCase):
    def test_rdparar_contract_includes_live_parameter_discovery(self):
        contract = execution_contract(
            plan(step("cnc_rdparar"), targets=("cnc_rdparar",))
        )
        self.assertIn("cnc_sysinfo", contract.functions)
        self.assertIn("cnc_rdparanum", contract.functions)
        self.assertIn("cnc_rdparainfo", contract.functions)

    def test_read_target_is_not_forced_into_motion_by_scenario_label(self):
        contract = execution_contract(plan(step("cnc_rdparam"), targets=("cnc_rdparam",)))
        self.assertFalse(contract.motion)
        self.assertEqual(contract.targets, ("cnc_rdparam",))
        self.assertNotIn("cnc_download3", contract.functions)

    def test_declared_plan_target_overrides_stale_coverage_manifest(self):
        candidate = plan(step("cnc_wrrelpos"), targets=("cnc_tofs_rnge",))
        candidate.target_api = "cnc_wrrelpos"
        contract = execution_contract(candidate)
        self.assertEqual(contract.targets, ("cnc_wrrelpos",))

    def test_motion_comes_from_operation_and_does_not_replace_target(self):
        contract = execution_contract(plan(
            step("cnc_actf"), step("", interface="StartProgram"), targets=("cnc_actf",)))
        self.assertTrue(contract.motion)
        self.assertEqual(contract.targets, ("cnc_actf",))
        self.assertIn("cnc_rdseqnum", contract.helpers)
        self.assertIn("cnc_download3", contract.helpers)

    def test_verify_end_has_actual_session_and_resident_read_dependencies(self):
        contract = execution_contract(plan(step("cnc_vrfend"), targets=("cnc_vrfend",)))
        self.assertFalse(contract.motion)
        for name in ("cnc_vrfstart", "cnc_verify", "cnc_upstart3", "cnc_upload3", "cnc_upend3"):
            self.assertIn(name, contract.helpers)

    def test_download4_uses_version4_session_dependencies(self):
        contract = execution_contract(plan(step("cnc_download4"), targets=("cnc_download4",)))
        self.assertTrue({"cnc_dwnstart4", "cnc_dwnend4"} <= set(contract.helpers))
        self.assertNotIn("cnc_dwnstart3", contract.functions)
        self.assertIn("cnc_dwnstart4", dependency_closure(["cnc_dwnend4"]))
        self.assertIn("version-4", analyze_api_interactions(["cnc_download4"])["cnc_download4"]["reason"])

    def test_download_versions_can_have_independent_sessions(self):
        contract = execution_contract(plan(
            step("cnc_download3"), step("cnc_download4"),
            targets=("cnc_download3", "cnc_download4"),
        ))
        self.assertIn("cnc_dwnstart3", contract.functions)
        self.assertIn("cnc_dwnstart4", contract.functions)
        self.assertEqual(len(contract.targets), 2)

    def test_lifecycle_validator_accepts_two_non_crossed_versions(self):
        fragment = r'''short a = ctx.call("S", "cnc_dwnstart3", "", api.cnc_dwnstart3, [&]{ return api.cnc_dwnstart3(handle, 0); }, [](short){ return ""; });
        if (a != EW_OK) return a;
        auto b = ctx.call("D", "cnc_download3", "", api.cnc_download3, [&]{ return api.cnc_download3(handle, &len, data); }, [](short){ return ""; });
        auto c = ctx.call("E", "cnc_dwnend3", "", api.cnc_dwnend3, [&]{ return api.cnc_dwnend3(handle); }, [](short){ return ""; });
        short d = ctx.call("S", "cnc_dwnstart4", "", api.cnc_dwnstart4, [&]{ return api.cnc_dwnstart4(handle, 0, nullptr); }, [](short){ return ""; });
        if (d != EW_OK) return d;
        auto e = ctx.call("D", "cnc_download4", "", api.cnc_download4, [&]{ return api.cnc_download4(handle, &len, data); }, [](short){ return ""; });
        auto f = ctx.call("E", "cnc_dwnend4", "", api.cnc_dwnend4, [&]{ return api.cnc_dwnend4(handle); }, [](short){ return ""; });'''
        self.assertEqual(validate_api_lifecycle_fragment(fragment), [])

    def test_lifecycle_validator_rejects_mixed_download_versions(self):
        fragment = r'''auto transfer = ctx.call("DL", "cnc_download4", "", api.cnc_download4,
            [&]{ return api.cnc_download4(handle, &len, data); }, [](short){ return ""; });
        auto start = ctx.call("S", "cnc_dwnstart3", "", api.cnc_dwnstart3,
            [&]{ return api.cnc_dwnstart3(handle, 0); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_dwnend3", "", api.cnc_dwnend3,
            [&]{ return api.cnc_dwnend3(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("cannot be paired" in error for error in errors))
        self.assertTrue(any("cnc_dwnstart4" in error for error in errors))

    def test_lifecycle_validator_requires_download_start_return_gate(self):
        fragment = r'''short sr = ctx.call("S", "cnc_dwnstart4", "", api.cnc_dwnstart4,
            [&]{ return api.cnc_dwnstart4(handle, 0); }, [](short){ return ""; });
        auto transfer = ctx.call("DL", "cnc_download4", "", api.cnc_download4,
            [&]{ return api.cnc_download4(handle, &len, data); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_dwnend4", "", api.cnc_dwnend4,
            [&]{ return api.cnc_dwnend4(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("skipped unless" in error for error in errors))

    def test_download4_validator_rejects_non_nc_download_type(self):
        fragment = r'''short sr = ctx.call("S", "cnc_dwnstart4", "", api.cnc_dwnstart4,
            [&]{ return api.cnc_dwnstart4(handle, 99, nullptr); }, [](short){ return ""; });
        if (sr != EW_OK) return sr;
        auto transfer = ctx.call("DL", "cnc_download4", "", api.cnc_download4,
            [&]{ return api.cnc_download4(handle, &len, data); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_dwnend4", "", api.cnc_dwnend4,
            [&]{ return api.cnc_dwnend4(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("type=0" in error for error in errors))

    def test_download4_validator_accepts_zero_inside_nested_short_cast(self):
        fragment = r'''short sr = ctx.call("S", "cnc_dwnstart4", "", api.cnc_dwnstart4,
            [&]{ return api.cnc_dwnstart4(handle, static_cast<short>(0), nullptr); }, [](short){ return ""; });
        if (sr != EW_OK) return sr;
        auto transfer = ctx.call("DL", "cnc_download4", "", api.cnc_download4,
            [&]{ return api.cnc_download4(handle, &len, data); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_dwnend4", "", api.cnc_dwnend4,
            [&]{ return api.cnc_dwnend4(handle); }, [](short){ return ""; });'''
        self.assertEqual(validate_api_lifecycle_fragment(fragment), [])

    def test_program_transfer_sequences_expose_upload_and_download_chains(self):
        rows = api_sequence_rules(("cnc_upload3", "cnc_download3"))
        self.assertIn({"cnc_upstart3", "cnc_upload3", "cnc_upend3"},
                      [set(row["calls"]) for row in rows])
        self.assertIn({"cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"},
                      [set(row["calls"]) for row in rows])
        upload_interaction = analyze_api_interactions(["cnc_upload3"])["cnc_upload3"]
        self.assertEqual(upload_interaction["before"], ["cnc_upstart3"])
        self.assertEqual(upload_interaction["optional_before"], ["cnc_rdprogdir3"])
        self.assertIsNone(program_operation_chain("cnc_rdprogdir3"))

    def test_upload_lifecycle_requires_start_and_end(self):
        missing_start = r'''auto data = ctx.call("D", "cnc_upload3", "", api.cnc_upload3,
            [&]{ return api.cnc_upload3(handle, &len, buffer); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_upend3", "", api.cnc_upend3,
            [&]{ return api.cnc_upend3(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(missing_start)
        self.assertTrue(any("requires matching cnc_upstart3" in error for error in errors))

        valid = r'''short start_rc = ctx.call("S", "cnc_upstart3", "", api.cnc_upstart3,
            [&]{ return api.cnc_upstart3(handle, 0, program, program); }, [](short){ return ""; });
        if (start_rc != EW_OK) return start_rc;
        auto data = ctx.call("D", "cnc_upload3", "", api.cnc_upload3,
            [&]{ return api.cnc_upload3(handle, &len, buffer); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_upend3", "", api.cnc_upend3,
            [&]{ return api.cnc_upend3(handle); }, [](short){ return ""; });'''
        self.assertEqual(validate_api_lifecycle_fragment(valid), [])

    def test_upload_lifecycle_rejects_zero_program_numbers(self):
        fragment = r'''short start_rc = ctx.call("S", "cnc_upstart3", "", api.cnc_upstart3,
            [&]{ return api.cnc_upstart3(handle, 0, 0, 0); }, [](short){ return ""; });
        if (start_rc != EW_OK) return start_rc;
        auto data = ctx.call("D", "cnc_upload3", "", api.cnc_upload3,
            [&]{ return api.cnc_upload3(handle, &len, buffer); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_upend3", "", api.cnc_upend3,
            [&]{ return api.cnc_upend3(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("s_number" in error and "literal zero" in error for error in errors))
        self.assertTrue(any("e_number" in error and "literal zero" in error for error in errors))

    def test_upload_lifecycle_rejects_zero_initialized_selection(self):
        fragment = r'''const long selected = 0;
        short start_rc = ctx.call("S", "cnc_upstart3", "", api.cnc_upstart3,
            [&]{ return api.cnc_upstart3(handle, 0, selected, selected); }, [](short){ return ""; });
        if (start_rc != EW_OK) return start_rc;
        auto data = ctx.call("D", "cnc_upload3", "", api.cnc_upload3,
            [&]{ return api.cnc_upload3(handle, &len, buffer); }, [](short){ return ""; });
        auto end = ctx.call("E", "cnc_upend3", "", api.cnc_upend3,
            [&]{ return api.cnc_upend3(handle); }, [](short){ return ""; });'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("initialized to zero" in error for error in errors))

    def test_lifecycle_validator_rejects_dnc_calls_outside_success_gate(self):
        fragment = r'''short start_rc = ctx.call("S", "cnc_dncstart2", "", api.cnc_dncstart2,
            [&]{ return api.cnc_dncstart2(handle, filename); }, [](short){ return ""; });
        if (start_rc == EW_OK) { api.cnc_dnc2(handle, &len, data); api.cnc_dncend2(handle, DNC_NORMAL); }
        else { api.cnc_dnc2(handle, &len, data); }'''
        errors = validate_api_lifecycle_fragment(fragment)
        self.assertTrue(any("only be called after" in error for error in errors))

    def test_lifecycle_validator_accepts_dnc_calls_only_after_success(self):
        fragment = r'''short start_rc = ctx.call("S", "cnc_dncstart2", "", api.cnc_dncstart2,
            [&]{ return api.cnc_dncstart2(handle, filename); }, [](short){ return ""; });
        if (start_rc == EW_OK) { api.cnc_dnc2(handle, &len, data); api.cnc_dncend2(handle, DNC_NORMAL); }'''
        self.assertEqual(validate_api_lifecycle_fragment(fragment), [])

    def test_rename_program_requires_discovery_and_restoration_context(self):
        contract = execution_contract(plan(step("cnc_renameprog"), targets=("cnc_renameprog",)))
        self.assertEqual(contract.targets, ("cnc_renameprog",))
        for name in ("cnc_rdprogdir3", "cnc_rdprgnum", "cnc_getdtailerr"):
            self.assertIn(name, contract.helpers)
        interactions = analyze_api_interactions(["cnc_renameprog"])
        self.assertEqual(interactions["cnc_renameprog"]["before"], ["cnc_rdprogdir3", "cnc_rdprgnum"])
        self.assertIn("rename it back", interactions["cnc_renameprog"]["reason"])
        self.assertIn("cnc_renameprog", dependency_closure(["cnc_renameprog"]))

    def test_quality_gate_follows_execution_capability_not_scenario_label(self):
        static_plan = plan(step("cnc_rdparam"), targets=("cnc_rdparam",))
        static_plan.rag_context["execution_contract"] = execution_contract(static_plan).to_dict()
        motion_plan = plan(step("cnc_actf"), step("", interface="StartProgram"), targets=("cnc_actf",))
        motion_plan.rag_context["execution_contract"] = execution_contract(motion_plan).to_dict()
        request = TaskRequest("capture", "t")
        self.assertFalse(requires_dynamic_quality_gate(WorkflowState(request=request, plan=static_plan)))
        self.assertTrue(requires_dynamic_quality_gate(WorkflowState(request=request, plan=motion_plan)))

    def test_lifecycle_setup_follows_connection_without_duplicate_wrappers(self):
        steps = [
            step("cnc_download3", "before"),
            step("cnc_allclibhndl3", "before", "AllocateHandle"),
            step("", "before", "UploadProgram"),
            step("cnc_dwnstart3", "before"),
            step("cnc_dwnend3", "before"),
            step("cnc_freelibhndl", "after", "FreeHandle"),
        ]
        normalize_lifecycle_steps(steps)
        self.assertEqual(steps[0].interface_name, "AllocateHandle")
        self.assertEqual(steps[-1].interface_name, "FreeHandle")
        self.assertNotIn("UploadProgram", [item.interface_name for item in steps])

    def test_rendered_motion_and_verification_shells_validate_and_compile(self):
        header_file = default_focas_header_dir() / "Fwlib32.h"
        if not header_file.is_file():
            self.skipTest("FOCAS SDK header not installed")
        header = header_file.read_text(encoding="latin-1")
        cases = (
            ("actf-motion", execution_contract(plan(step("cnc_actf"), step("", interface="StartProgram"), targets=("cnc_actf",))), wrap_capture_fragment(motion_read_core({"cnc_actf"})), NC),
            ("acts-numeric-step", execution_contract(plan(step("cnc_acts"), step("", interface="StartProgram"), targets=("cnc_acts",))),
             wrap_capture_fragment('ODBACT acts{}; ret = ctx.call(1, "cnc_acts", "FlibHndl=" + std::to_string(handle), api.cnc_acts, [&]{return api.cnc_acts(handle, &acts);}, [&](short){return std::to_string(acts.data);});'), NC),
            ("statinfo-static", execution_contract(plan(step("cnc_statinfo"), targets=("cnc_statinfo",))), wrap_capture_fragment(guarded_core({"cnc_statinfo"}, 2, 0)), ""),
            ("verify-end", execution_contract(plan(step("cnc_vrfend"), targets=("cnc_vrfend",))), wrap_capture_fragment(guarded_core({"cnc_vrfend"}, 2, 0)), ""),
        )
        from smart_traffic_agent.agents.code_generator import preflight_compile_generated_cpp
        scratch = ROOT / ".test_tmp"
        scratch.mkdir(exist_ok=True)
        for label, contract, fragment, nc in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(dir=scratch) as folder:
                source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program=nc)
                self.assertEqual(validate_production_source(source, fragment, header, contract, nc), [])
                path = Path(folder)
                (path / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
                cpp = path / "api_script.cpp"
                cpp.write_text(source, encoding="utf-8")
                self.assertEqual(preflight_compile_generated_cpp(cpp, path), "")

    def test_pmc_panel_gate_rejects_off_lock_and_active_low_feed_hold(self):
        import subprocess
        from smart_traffic_agent.agent_tools.cpp_execution import CompileGeneratedCppTool, CompileCppInput
        driver = (ROOT / "smart_traffic_agent/templates/ncguide_motion_driver.cpp").read_text(encoding="utf-8")
        predicate = driver[driver.index("static bool runtime_panel_ready("):driver.index("static short check_runtime_panel(")]
        # Exercise production C++ with the observed off/on snapshots and every
        # combination of the three prerequisites, including active-low *SP.
        source = predicate + r"""
int main() {
    if (runtime_panel_ready(59, 0, 120)) return 1;
    if (!runtime_panel_ready(59, 0, 122)) return 2;
    for (int sbk = 0; sbk < 2; ++sbk)
        for (int lock = 0; lock < 2; ++lock)
            for (int released = 0; released < 2; ++released)
                if (runtime_panel_ready(released ? 32 : 0, lock ? 2 : 0, sbk ? 2 : 0)
                    != (sbk && !lock && released)) return 3;
    return 0;
}
"""
        scratch = ROOT / ".test_tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            path = Path(folder)
            cpp, exe = path / "panel.cpp", path / "panel.exe"
            cpp.write_text(source, encoding="utf-8")
            result = CompileGeneratedCppTool().invoke(CompileCppInput(cpp, exe, path))
            self.assertTrue(result.success, result.stdout + result.stderr)
            self.assertEqual(subprocess.run([str(exe)], capture_output=True, timeout=10).returncode, 0)

    def test_forged_or_missing_target_cannot_pass_production_validation(self):
        header_file = default_focas_header_dir() / "Fwlib32.h"
        if not header_file.is_file():
            self.skipTest("FOCAS SDK header not installed")
        header = header_file.read_text(encoding="latin-1")
        contract = execution_contract(plan(step("cnc_actf"), targets=("cnc_actf",)))
        fragment = wrap_capture_fragment("    // No actual target invocation.")
        source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program=NC)
        errors = validate_production_source(source + "\n// framework_owned", fragment, header, contract, NC)
        self.assertTrue(any("differs from the framework" in item for item in errors))
        self.assertTrue(any("no logged invocation" in item for item in errors))

    def test_unplanned_reset_and_alarm_clear_are_rejected(self):
        header_file = default_focas_header_dir() / "Fwlib32.h"
        if not header_file.is_file():
            self.skipTest("FOCAS SDK header not installed")
        header = header_file.read_text(encoding="latin-1")
        contract = execution_contract(plan(step("cnc_statinfo"), targets=("cnc_statinfo",)))
        self.assertEqual(contract.planned_functions, ("cnc_statinfo",))
        fragment = wrap_capture_fragment(
            'ctx.call("TARGET", "cnc_statinfo", "", api.cnc_statinfo, '
            '[&]{ return api.cnc_statinfo(handle, nullptr); }, [](short){ return ""; });\n'
            'ctx.call("RECOVERY", "cnc_reset", "", api.cnc_reset, '
            '[&]{ return api.cnc_reset(handle); }, [](short){ return ""; });'
        )
        source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program="")
        errors = validate_production_source(source, fragment, header, contract, "")
        self.assertTrue(any("unplanned API call cnc_reset" in item for item in errors))

    def test_unresolved_call_placeholder_is_blocked_for_multiple_targets(self):
        header_file = default_focas_header_dir() / "Fwlib32.h"
        if not header_file.is_file():
            self.skipTest("FOCAS SDK header not installed")
        header = header_file.read_text(encoding="latin-1")
        for target in ("cnc_wrpm_item", "cnc_sysinfo_ex"):
            with self.subTest(target=target):
                contract = execution_contract(plan(step(target), targets=(target,)))
                fragment = wrap_capture_fragment(
                    f'ctx.call("TARGET", "{target}", "", api.function, '
                    f'[&]{{ return api.function(handle); }}, [](short){{ return ""; }});'
                )
                source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program="")
                errors = validate_production_source(source, fragment, header, contract, "")
                self.assertTrue(any("unresolved production call placeholder" in item for item in errors))
                self.assertTrue(any("no logged invocation paired" in item for item in errors))

    def test_target_name_in_log_without_real_call_is_blocked(self):
        header_file = default_focas_header_dir() / "Fwlib32.h"
        if not header_file.is_file():
            self.skipTest("FOCAS SDK header not installed")
        header = header_file.read_text(encoding="latin-1")
        contract = execution_contract(plan(step("cnc_sysinfo_ex"), targets=("cnc_sysinfo_ex",)))
        fragment = wrap_capture_fragment(
            'ctx.call("TARGET_cnc_sysinfo_ex", "cnc_sysinfo_ex", "", '
            'api.cnc_statinfo, [&]{ return api.cnc_statinfo(handle, nullptr); }, [](short){ return ""; });'
        )
        source = render_capture_harness(contract.functions, header, fragment, contract=contract, nc_program="")
        errors = validate_production_source(source, fragment, header, contract, "")
        self.assertTrue(any("no logged invocation paired" in item for item in errors))

    def test_write_requires_real_snapshot_and_restore_value(self):
        fragment = (
            'api.cnc_rdparam(handle, 1, 0, &before); '
            'api.cnc_wrparam(handle, 1, 0, &changed); '
            'api.cnc_rdparam(handle, 1, 0, &after);'
        )
        errors = _validate_write_snapshot_restore(fragment, ("cnc_wrparam",))
        self.assertTrue(any("second call" in item for item in errors))

        zero_restore = (
            'api.cnc_rdparam(handle, 1, 0, &before); '
            'api.cnc_wrparam(handle, 1, 0, &changed); '
            'api.cnc_rdparam(handle, 1, 0, &after); '
            'ODBPARA restore{}; restore.data = 0; '
            'api.cnc_wrparam(handle, 1, 0, &restore); '
            'api.cnc_rdparam(handle, 1, 0, &verified);'
        )
        zero_errors = _validate_write_snapshot_restore(zero_restore, ("cnc_wrparam",))
        self.assertTrue(any("zero-valued restore" in item for item in zero_errors))

    def test_wrrelpos_uses_documented_relative_read_api(self):
        interaction = analyze_api_interactions(["cnc_wrrelpos"])["cnc_wrrelpos"]
        self.assertEqual(interaction["before"], [])
        self.assertEqual(interaction["after"], [])
        fragment = (
            "api.cnc_wrrelpos(handle, sizeof(IDBWRR), &changed);"
        )
        errors = _validate_write_snapshot_restore(fragment, ("cnc_wrrelpos",))
        self.assertFalse(errors)

    def test_write_without_read_pair_is_safely_blocked(self):
        errors = _validate_write_snapshot_restore(
            'api.cnc_wrpm_item(handle, 1, 2, 1, &item);',
            ("cnc_wrpm_item",),
        )
        self.assertTrue(any("requires a real cnc_rdpm_item" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
