from pathlib import Path
from dataclasses import replace
import csv
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from smart_traffic_agent import focas_capture_runtime
from smart_traffic_agent.focas_capture_runtime import (
    guarded_core, skip_read_core, download4_core, deterministic_read_core,
    wrap_capture_fragment, validate_capture_fragment, render_capture_harness,
    verify4_core, dncstart2_core,
)
from smart_traffic_agent.agent_tools.cpp_execution import discover_capture_pcap, read_cpp_csv_logs
from smart_traffic_agent.agents.router import valid_next_stages
from smart_traffic_agent.agents.executor import (
    required_traffic_errors,
    assess_plan_step_execution,
    build_api_capture_summary,
    collection_status_fields,
    write_api_traffic_classification,
)
from smart_traffic_agent.models import ApiCallLog, ExecutionResult, TaskRequest, WorkflowState, ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.function_coverage import build_function_coverage_metrics
from smart_traffic_agent.focas_contracts import named_contract_context


class ProductionCaptureTests(unittest.TestCase):
    def test_deterministic_read_recipe_uses_capture_context_for_logs(self):
        fragment = wrap_capture_fragment(deterministic_read_core({"cnc_rdcncid"}))
        self.assertIn('ctx.call("TARGET_CNC_RDCNCID", "cnc_rdcncid"', fragment)
        self.assertIn('cncid=" + std::to_string(cncid)', fragment)
        self.assertNotIn("focas_api_input.csv", fragment)
        self.assertNotIn("focas_api_output.csv", fragment)

    def test_output_only_path_recipe_does_not_treat_outputs_as_input_dimensions(self):
        fragment = wrap_capture_fragment(deterministic_read_core({"cnc_getpath"}))
        self.assertIn("path_no=out;maxpath_no=out", fragment)
        self.assertIn("api.cnc_getpath(handle, &path_no, &maxpath_no)", fragment)

    def test_dncstart2_recipe_is_read_only_and_does_not_require_remote(self):
        fragment = wrap_capture_fragment(dncstart2_core())
        self.assertIn('ctx.call("TARGET_CNC_DNCSTART2", "cnc_dncstart2"', fragment)
        self.assertIn('api.cnc_dncstart2(handle, dnc_filename)', fragment)
        self.assertIn('filename=" + std::string(dnc_filename)', fragment)
        self.assertNotIn("cnc_dnc2", fragment)
        self.assertNotIn("cnc_dncend2", fragment)
        self.assertIn("ret == EW_REJECT || ret == EW_BUSY || ret == EW_NOOPT || ret == EW_PARAM", fragment)
        self.assertIn("return EW_OK;", fragment)

    def test_setpath_recipe_discovers_path_before_setting_it(self):
        fragment = wrap_capture_fragment(deterministic_read_core({"cnc_setpath"}))
        self.assertLess(fragment.index('"cnc_getpath"'), fragment.index('"cnc_setpath"'))
        self.assertIn("api.cnc_setpath(handle, current_path)", fragment)

    def test_parameterized_recipe_requires_explicit_values(self):
        self.assertIsNone(deterministic_read_core({"cnc_rdmacro"}))
        fragment = deterministic_read_core(
            {"cnc_rdmacro"}, parameter_values={"cnc_rdmacro": {"number": 500, "length": 10}}
        )
        self.assertIsNotNone(fragment)
        self.assertIn("number=", fragment)
        self.assertIn("api.cnc_rdmacro", fragment)

    def test_rdparam_recipe_requires_documented_length(self):
        self.assertIsNone(deterministic_read_core(
            {"cnc_rdparam"}, parameter_values={"cnc_rdparam": {"number": 1005, "axis": -1}}
        ))
        fragment = deterministic_read_core(
            {"cnc_rdparam"},
            parameter_values={"cnc_rdparam": {"number": 1005, "axis": -1, "length": 132}},
        )
        self.assertIn("parameter_length", fragment)
        self.assertIn("132", fragment)

    def test_rdparar_recipe_discovers_domain_and_derives_byte_length(self):
        fragment = wrap_capture_fragment(deterministic_read_core(
            {"cnc_rdparar"},
            planned_functions={"cnc_sysinfo", "cnc_rdparanum", "cnc_rdparainfo", "cnc_rdparar"},
        ))
        self.assertLess(fragment.index('"cnc_sysinfo"'), fragment.index('"cnc_rdparar"'))
        self.assertLess(fragment.index('"cnc_rdparanum"'), fragment.index('"cnc_rdparar"'))
        self.assertLess(fragment.index('"cnc_rdparainfo"'), fragment.index('"cnc_rdparar"'))
        self.assertIn("api.cnc_rdparar(handle, &call_s_number, selected_axis", fragment)
        self.assertIn("4 + value_bytes * axis_count", fragment)
        self.assertIn("length_rule=4+value_bytes*axis_count", fragment)
        self.assertNotIn("length=8", fragment)
        self.assertNotIn("1002", fragment)

    def test_rdparar_recipe_requires_discovery_steps(self):
        self.assertIsNone(deterministic_read_core(
            {"cnc_rdparar"}, planned_functions={"cnc_rdparar"}
        ))

    def test_wrrelpos_recipe_uses_idbwrr_and_documented_argument_order(self):
        fragment = wrap_capture_fragment(deterministic_read_core(
            {"cnc_wrrelpos"},
            parameter_values={"cnc_wrrelpos": {"type": 1, "data0": 12}},
        ))
        self.assertIn("IDBWRR wrrelpos_value", fragment)
        self.assertIn("wrrelpos_length = static_cast<short>(8)", fragment)
        self.assertIn(
            "api.cnc_wrrelpos(handle, wrrelpos_length, &wrrelpos_value)",
            fragment,
        )
        self.assertIn('ctx.call("TARGET_CNC_WRRELPOS", "cnc_wrrelpos"', fragment)
        self.assertNotIn("api.function(", fragment)

    def test_wrrelpos_all_axes_requires_complete_data_and_derives_length(self):
        values = {"type": -1, "data": list(range(32))}
        fragment = deterministic_read_core(
            {"cnc_wrrelpos"}, parameter_values={"cnc_wrrelpos": values}
        )
        self.assertIsNotNone(fragment)
        self.assertIn("wrrelpos_length = static_cast<short>(132)", fragment)
        self.assertIn("wrrelpos_value.data[31]", fragment)
        self.assertIsNone(deterministic_read_core(
            {"cnc_wrrelpos"},
            parameter_values={"cnc_wrrelpos": {"type": -1, "data": [0]}},
        ))

    def test_wrrelpos_rejects_inconsistent_or_missing_sdk_values(self):
        self.assertIsNone(deterministic_read_core(
            {"cnc_wrrelpos"},
            parameter_values={"cnc_wrrelpos": {"type": 1, "length": 132, "data0": 0}},
        ))
        self.assertIsNone(deterministic_read_core(
            {"cnc_wrrelpos"}, parameter_values={"cnc_wrrelpos": {"data0": 0}}
        ))

    def test_renderer_sets_manual_mode_gates_from_target_apis(self):
        with patch.object(focas_capture_runtime._legacy, "render_capture_harness", return_value="remote=__REQUIRE_REMOTE__;reads=__ALLOW_ALARM_READS__"):
            dnc_source = render_capture_harness(
                ["cnc_dnc2"], contract=SimpleNamespace(targets=("cnc_dnc2",))
            )
            dnc_query_source = render_capture_harness(
                ["cnc_dncstart2"], contract=SimpleNamespace(targets=("cnc_dncstart2",))
            )
            read_source = render_capture_harness(
                ["cnc_statinfo"], contract=SimpleNamespace(targets=("cnc_statinfo",))
            )

        self.assertEqual(dnc_source, "remote=true;reads=false")
        self.assertEqual(dnc_query_source, "remote=false;reads=false")
        self.assertEqual(read_source, "remote=false;reads=true")

    def test_capture_template_emits_manual_action_events_and_logs_status_polls(self):
        source = Path(focas_capture_runtime.TEMPLATE).read_text(encoding="utf-8")
        self.assertIn("operator_action_required", source)
        self.assertIn("operator_action_verified", source)
        self.assertIn("MANUAL_ACTION_TIMEOUT", source)
        self.assertIn("OPERATOR_STATE_POLL", source)
        self.assertIn("wait_for_cnc_mode", source)
        self.assertIn("OPERATOR_MODE_POLL", source)
        self.assertIn("MODE_REQUIREMENT_VERIFIED", source)
        self.assertIn("__REQUIRE_REMOTE__", source)
        self.assertIn("__ALLOW_ALARM_READS__", source)

    def test_complete_callback_is_not_wrapped_twice(self):
        fragment = wrap_capture_fragment('return EW_OK;')
        self.assertEqual(wrap_capture_fragment(fragment), fragment)
        self.assertEqual(validate_capture_fragment(wrap_capture_fragment(fragment)), [])

    def test_nested_or_legacy_callback_is_rejected_before_wrapping(self):
        fragment = wrap_capture_fragment('return EW_OK;')
        nested = fragment.replace('return EW_OK;', fragment)
        for invalid in (nested, fragment.replace(', CaptureContext& ctx', ''),
                        'void helper() {}\n' + fragment, fragment + 'void helper() {}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wrap_capture_fragment(invalid)

    def test_callback_mentions_in_comments_and_strings_are_not_definitions(self):
        core = '// short run_generated_scenario() {}\nstd::string note = "run_generated_scenario()";'
        self.assertTrue(wrap_capture_fragment(core).startswith('short run_generated_scenario('))

    def test_log_fields_require_semicolons_but_array_commas_are_allowed(self):
        for payload in ('FlibHndl=32769,number=3329',
                        'FlibHndl=32769;number=1,expected_return_code=5',
                        'err_no=2,err_dtno=0'):
            fragment = wrap_capture_fragment('std::string inputs = "' + payload + '";')
            self.assertTrue(any('semicolon' in error for error in validate_capture_fragment(fragment)))
        fragment = wrap_capture_fragment('std::string inputs = "FlibHndl=32769;number=3329;buf=[1,2,3]";')
        self.assertEqual(validate_capture_fragment(fragment), [])

    def test_guarded_recipes_use_logged_typed_calls(self):
        for api in ("cnc_statinfo", "cnc_search", "cnc_rdpdf_line", "cnc_wrmdipntr"):
            with self.subTest(api=api):
                fragment = wrap_capture_fragment(guarded_core({api}, 3, 5))
                self.assertEqual(validate_capture_fragment(fragment), [])

    def test_search_diversity_and_restoration_with_mock_controller(self):
        import subprocess
        from smart_traffic_agent.agent_tools.cpp_execution import CompileGeneratedCppTool, CompileCppInput
        source = r"""
#define NOMINMAX
#include <Windows.h>
#include "Fwlib32.h"
#include <algorithm>
#include <string>
#include <vector>
static std::vector<long> resident{100,200,300};
static std::vector<short> calls;
static long selected=200;
static short fail_on=0;
static bool mismatch=false;
static int reads=0;
static short status(unsigned short, ODBST* s) { *s={}; s->aut=1; return 0; }
static short directory(unsigned short,short,long*,short* n,PRGDIR3* rows) {
    *n=static_cast<short>(resident.size());
    for(int i=0;i<*n;++i) rows[i].number=resident[i];
    return 0;
}
static short number(unsigned short, ODBPRO* p) {
    ++reads; p->mdata=(mismatch && reads==2) ? 999 : selected; return 0;
}
static short search(unsigned short, short n) { calls.push_back(n); selected=n; return n==fail_on ? 5 : 0; }
static short detail(unsigned short, ODBERR*) { return 0; }
struct FocasApi {
    decltype(&status) cnc_statinfo=status;
    decltype(&directory) cnc_rdprogdir3=directory;
    decltype(&number) cnc_rdprgnum=number;
    decltype(&search) cnc_search=search;
    decltype(&detail) cnc_getdtailerr=detail;
};
struct CaptureContext {
    template<class F,class Call,class Output>
    short call(const char*,const char*,const std::string&,F,Call invoke,Output output) {
        short result=invoke(); output(result); return result;
    }
    void block(const char*,const std::string&) {}
    void host(const char*,short,const char*,const std::string&) {}
};
""" + wrap_capture_fragment(guarded_core({"cnc_search"}, 30, 0)) + r"""
int main() {
    FocasApi api; CaptureContext ctx;
    if(run_generated_scenario(api,1,ctx)!=0 || calls!=std::vector<short>({100,200,300,200}) || selected!=200) return 1;
    calls.clear(); reads=0; fail_on=300;
    if(run_generated_scenario(api,1,ctx)==0 || calls.back()!=200 || selected!=200) return 2;
    calls.clear(); reads=0; fail_on=0; mismatch=true;
    if(run_generated_scenario(api,1,ctx)==0 || calls!=std::vector<short>({100,200}) || selected!=200) return 3;
    calls.clear(); reads=0; mismatch=false; resident={100};
    if(run_generated_scenario(api,1,ctx)==0 || !calls.empty()) return 4;
    return 0;
}
"""
        root = Path(__file__).resolve().parents[1] / ".test_tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path = Path(folder)
            cpp, exe = path / "search_test.cpp", path / "search_test.exe"
            cpp.write_text(source, encoding="utf-8")
            result = CompileGeneratedCppTool().invoke(CompileCppInput(cpp, exe, path))
            self.assertTrue(result.success, result.stdout + result.stderr)
            self.assertEqual(subprocess.run([str(exe)], capture_output=True, timeout=10).returncode, 0)

    def test_direct_call_cannot_hide_in_comment(self):
        fragment = wrap_capture_fragment('// ctx.call(\nret=api.cnc_statinfo(handle, nullptr);\n// )')
        self.assertTrue(any("bypasses" in x for x in validate_capture_fragment(fragment)))

    def test_selected_session_targets_remain_owned_by_harness(self):
        for api in ("cnc_allclibhndl3", "cnc_freelibhndl"):
            fragment = wrap_capture_fragment(guarded_core({api}))
            self.assertEqual(validate_capture_fragment(fragment), [])
            self.assertNotIn("api." + api, fragment)

    def test_comments_do_not_change_contract(self):
        fragment = wrap_capture_fragment(guarded_core({"cnc_statinfo"}))
        self.assertEqual(validate_capture_fragment(fragment + "\n// main() LoadLibraryW() memcpy()"), [])

    def test_null_pdf_name_rejected(self):
        fragment = wrap_capture_fragment('ret=ctx.call("S", "cnc_rdpdf_line", "", api.cnc_rdpdf_line, [&]{return api.cnc_rdpdf_line(handle, nullptr, 0, text, &n, &size);}, [](short){return "";});')
        self.assertTrue(any("explicit verified" in x for x in validate_capture_fragment(fragment)))

    def test_nonstandard_pcap_name_and_basename_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            pcap = data / "focas_traffic_batch.pcap"
            pcap.write_bytes(b"x" * 25)
            self.assertEqual(discover_capture_pcap(data, {})[0], pcap)
            self.assertEqual(discover_capture_pcap(data, {"1": {"data": "pcap_file=focas_traffic_batch.pcap"}})[0], pcap.resolve())

    def test_ambiguous_and_escape_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for name in ("a.pcap", "b.pcap"):
                (data / name).write_bytes(b"x" * 25)
            self.assertTrue(discover_capture_pcap(data, {})[1])
            self.assertTrue(discover_capture_pcap(data, {"1": {"data": "pcap_file=../outside.pcap"}})[1])

    def test_crashed_call_retains_input_without_fabricated_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            (data / "focas_api_input.csv").write_text("index,timestamp,step_id,interface_name,protocol_function,parameters\n1,t,S,ReadProgramLine,cnc_rdpdf_line,prog_name=null\n")
            (data / "focas_api_output.csv").write_text("index,timestamp,step_id,api_name,return_code,return_text,data\n")
            logs = read_cpp_csv_logs("crash", Path(tmp)).api_logs
            incomplete = [log for log in logs if log.protocol_function == "cnc_rdpdf_line"]
            self.assertEqual(len(incomplete), 1)
            self.assertEqual(incomplete[0].traffic_status, "incomplete_call")
            self.assertFalse(incomplete[0].response["output_valid"])

    def test_runtime_per_api_mapping_survives_alternating_unmatched_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution = Path(tmp)
            data = execution / "data"
            by_api = data / "by_api"
            by_api.mkdir(parents=True)
            (by_api / ".runtime_per_api_capture").write_text(
                "mode=runtime_per_api_dumper;batch_id=batch-alternating\n", encoding="utf-8"
            )
            pcap_header = b"\xd4\xc3\xb2\xa1" + b"\x00" * 20 + b"packet"
            (data / "focas_capture.pcap").write_bytes(pcap_header)
            (by_api / "batch-alternating_cnc_statinfo.pcap").write_bytes(pcap_header)
            (by_api / "batch-alternating_cnc_getpath.pcap").write_bytes(pcap_header)

            input_path = data / "focas_api_input.csv"
            output_path = data / "focas_api_output.csv"
            input_fields = [
                "index", "timestamp", "step_id", "interface_name",
                "protocol_function", "parameters", "api_parameter_count",
                "invocation_id", "transaction_id", "start_timestamp_ns",
                "device_context",
            ]
            output_fields = [
                "index", "timestamp", "step_id", "api_name", "return_code",
                "return_text", "data", "api_parameter_count", "invocation_id",
                "transaction_id", "end_timestamp_ns",
            ]
            calls = [
                ("cnc_statinfo", "stat-1", "mode=1"),
                ("cnc_getpath", "path-1", "requested=1"),
                ("cnc_statinfo", "stat-2", "mode=2"),
                ("cnc_getpath", "path-2", "requested=2"),
            ]
            with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=input_fields)
                writer.writeheader()
                for index, (api, invocation, parameters) in enumerate(calls, 1):
                    writer.writerow({
                        "index": index,
                        "timestamp": "2026-10-26T10:25:15.000000Z",
                        "step_id": f"S{index:03d}",
                        "interface_name": api,
                        "protocol_function": api,
                        "parameters": parameters,
                        "api_parameter_count": 1,
                        "invocation_id": invocation,
                        "transaction_id": f"tx-{invocation}",
                        "start_timestamp_ns": index * 1_000_000_000,
                        "device_context": "pcap_capture_started=true",
                    })
            with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=output_fields)
                writer.writeheader()
                for index, (api, invocation, _parameters) in enumerate(calls, 1):
                    writer.writerow({
                        "index": index,
                        "timestamp": "2026-10-26T10:25:15.001000Z",
                        "step_id": f"S{index:03d}",
                        "api_name": api,
                        "return_code": 0,
                        "return_text": "EW_OK",
                        "data": "pcap_capture_started=true;value=1",
                        "api_parameter_count": 1,
                        "invocation_id": invocation,
                        "transaction_id": f"tx-{invocation}",
                        "end_timestamp_ns": index * 1_000_000_000 + 1,
                    })

            logs = read_cpp_csv_logs("batch-alternating", execution).api_logs
            api_logs = [log for log in logs if log.protocol_function in {"cnc_statinfo", "cnc_getpath"}]
            self.assertEqual(len(api_logs), 4)
            self.assertEqual(
                [str(log.response.get("api_pcap_file")).replace("\\", "/") for log in api_logs],
                [
                    "by_api/batch-alternating_cnc_statinfo.pcap",
                    "by_api/batch-alternating_cnc_getpath.pcap",
                    "by_api/batch-alternating_cnc_statinfo.pcap",
                    "by_api/batch-alternating_cnc_getpath.pcap",
                ],
            )
            self.assertTrue(all(log.traffic_status == "no_corresponding_traffic" for log in api_logs))
            summary = build_api_capture_summary(api_logs)
            self.assertEqual(summary["cnc_statinfo"]["call_count"], 2)
            self.assertEqual(summary["cnc_getpath"]["call_count"], 2)
            self.assertEqual(summary["cnc_statinfo"]["success_count"], 2)
            self.assertEqual(summary["cnc_getpath"]["distinct_parameter_configurations"], 2)
            self.assertEqual(summary["cnc_statinfo"]["traffic_matched_count"], 0)

    def test_mode_error_is_terminal_failure_without_repair(self):
        state = WorkflowState(request=TaskRequest(description="capture", task_id="mode"))
        log = ApiCallLog(timestamp="t", task_id="mode", step_id="S", phase="during", interface_name="cnc_rdmdipntr", input_parameters={}, status_code=12, response={}, protocol_function="cnc_rdmdipntr", error="EW_MODE")
        state.result = ExecutionResult(task_id="mode", success=False, api_logs=[log], capture_events=[], output_dir=Path("."))
        self.assertEqual(valid_next_stages(state), {"complete"})
        self.assertFalse(state.result.success)

    def test_error_is_expected_only_when_declared_in_plan(self):
        step = PlanStep(step_id="S", phase="during", action="read", interface_name="cnc_search", parameters={}, protocol_function="cnc_search")
        plan = ExecutionPlan(plan_id="p", task_id="t", scenario_type="program_management", scenario_goal="capture", target_environment="ncguide-generated-cpp", nc_program_type="test", nc_program_requirements=[], nc_program_spec=NcProgramSpec(program_name=""), steps=[step], expected_outputs=[], rag_context={"function_coverage_manifest": {"enabled": True, "selected_batch": [{"function": "cnc_search", "coverage_role": "target"}]}})
        log = ApiCallLog(timestamp="t", task_id="t", step_id="S", phase="during", interface_name="cnc_search", input_parameters={"raw": "prg_no=7777;expected_return_code=5"}, status_code=5, response={}, protocol_function="cnc_search", error="EW_DATA", traffic_status="matched")
        result = ExecutionResult(task_id="t", success=False, api_logs=[log], capture_events=[], output_dir=Path("."))
        self.assertEqual(build_function_coverage_metrics(plan, result)["expected_error_target_function_count"], 0)
        step.parameters["expected_return_code"] = 5
        self.assertEqual(build_function_coverage_metrics(plan, result)["expected_error_target_function_count"], 1)
        log.status_code = 12
        self.assertEqual(build_function_coverage_metrics(plan, result)["expected_error_target_function_count"], 0)

    def test_target_invocation_presence_is_independent_of_offline_traffic_match(self):
        state = WorkflowState(request=TaskRequest(description="capture", task_id="mode", permissions={"required_api_functions": ["cnc_search"]}))
        failed = ApiCallLog(timestamp="t", task_id="mode", step_id="S", phase="during", interface_name="cnc_search", input_parameters={}, status_code=12, response={}, protocol_function="cnc_search", error="EW_MODE", traffic_status="matched")
        self.assertEqual(required_traffic_errors(state, [failed]), [])
        unmatched_success = replace(failed, status_code=0, error="", traffic_status="no_corresponding_traffic")
        self.assertEqual(required_traffic_errors(state, [unmatched_success]), [])
        matched_success = replace(unmatched_success, traffic_status="matched")
        self.assertEqual(required_traffic_errors(state, [matched_success]), [])

    def test_full_contract_keeps_prerequisites_and_null_meaning(self):
        docs = named_contract_context(["cnc_rdpdf_line", "cnc_wrmdipntr"], max_chars=30000, full=True)
        self.assertIn("previous progaram name", docs)
        self.assertIn("MDI", docs)
        self.assertIn("Required options", docs)

    def test_skip_read_core_sweeps_documented_axes_without_event_injection(self):
        fragment = wrap_capture_fragment(skip_read_core())
        self.assertEqual(validate_capture_fragment(fragment), [])
        self.assertIn('"cnc_skip"', fragment)
        self.assertIn('"cnc_rdaxisname"', fragment)
        self.assertIn("axis=-1", fragment)
        self.assertIn('selected_axis <= axis_count', fragment)
        self.assertIn("skip_all_length", fragment)
        self.assertIn("skip_axis_length", fragment)
        self.assertIn("skip_signal_required=false", fragment)

    def test_download4_core_uses_type_zero_and_guards_start_data_end(self):
        fragment = wrap_capture_fragment(download4_core())
        self.assertEqual(validate_capture_fragment(fragment), [])
        self.assertIn('"cnc_dwnstart4"', fragment)
        self.assertIn('type=0;dir_name=NULL;destination=current_folder', fragment)
        self.assertIn('if (ret != EW_OK)', fragment)
        self.assertIn('"cnc_download4"', fragment)
        self.assertIn('DOWNLOAD_DATA_" + std::to_string(chunk_index + 1)', fragment)
        self.assertIn('"cnc_dwnend4"', fragment)
        self.assertNotIn('type=99', fragment)

    def test_verify4_core_uses_current_folder_and_output_buffer(self):
        fragment = wrap_capture_fragment(verify4_core())
        self.assertEqual(validate_capture_fragment(fragment), [])
        self.assertIn('"cnc_vrfstart4"', fragment)
        self.assertIn('dir_name=NULL;destination=current_folder', fragment)
        self.assertIn('api.cnc_vrfstart4(handle, nullptr)', fragment)
        self.assertIn('"cnc_verify4"', fragment)
        self.assertIn('data=output_buffer', fragment)
        self.assertIn('api.cnc_verify4(handle, &length, data)', fragment)
        self.assertIn('"cnc_vrfend4"', fragment)
        self.assertNotIn('O4764', fragment)

    def test_plan_step_evidence_separates_target_session_from_unexecuted_cleanup(self):
        plan = ExecutionPlan(
            "plan", "task", "program_transfer", "download", "ncguide-generated-cpp", "",
            [], NcProgramSpec("O1"), [
                PlanStep("S1", "before", "start", "start", protocol_function="cnc_dwnstart4"),
                PlanStep("S0", "before", "discover", "discover", protocol_function="cnc_rdprogdir3"),
                PlanStep("S2", "during", "data", "data", repeat=30, protocol_function="cnc_download4"),
                PlanStep("S3", "after", "detail", "detail", protocol_function="cnc_getdtailerr"),
                PlanStep("S4", "after", "delete", "delete", protocol_function="cnc_delete"),
                PlanStep("S6", "after", "verify", "verify", protocol_function="cnc_rdprogdir3"),
                PlanStep("S5", "after", "end", "end", protocol_function="cnc_dwnend4"),
            ], [], rag_context={
                "scene_plan": {
                    "target_api": "cnc_download4",
                    "session_chain": ["cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"],
                }
            },
        )
        logs = [
            ApiCallLog("t", "task", "START", "During", "start", {}, 0, {}, protocol_function="cnc_dwnstart4"),
            ApiCallLog("t", "task", "DISCOVER", "During", "discover", {}, 0, {}, protocol_function="cnc_rdprogdir3"),
            ApiCallLog("t", "task", "DATA", "During", "data", {}, 0, {}, protocol_function="cnc_download4"),
            ApiCallLog("t", "task", "END", "During", "end", {}, 0, {}, protocol_function="cnc_dwnend4"),
        ]
        evidence = assess_plan_step_execution(plan, logs)
        self.assertTrue(evidence["target_session_complete"])
        self.assertFalse(evidence["plan_execution_complete"])
        self.assertIn("cnc_delete", evidence["missing_protocol_functions"])
        self.assertIn("S4", evidence["authorization_blocked_steps"])
        self.assertIn("S3", evidence["conditional_skipped_steps"])
        verify_step = next(row for row in evidence["steps"] if row["step_id"] == "S6")
        self.assertEqual(verify_step["status"], "not_executed")

    def test_repeated_step_ids_are_assigned_by_explicit_prefix(self):
        plan = ExecutionPlan(
            "plan", "task", "read", "read", "ncguide-generated-cpp", "", [],
            NcProgramSpec("O1"), [
                PlanStep("S005", "during", "sample", "getpath", repeat=30, protocol_function="cnc_getpath"),
                PlanStep("S006", "after", "verify", "getpath", repeat=1, protocol_function="cnc_getpath"),
                PlanStep("S001", "before", "capture", "npcap", protocol_function="n/a (host-side capture)"),
            ], [], rag_context={"scene_plan": {"target_api": "cnc_getpath"}},
        )
        logs = [
            ApiCallLog("t", "task", f"S005-{index}", "During", "getpath", {}, 0, {}, protocol_function="cnc_getpath")
            for index in range(1, 31)
        ] + [
            ApiCallLog("t", "task", "S006", "After", "getpath", {}, 0, {}, protocol_function="cnc_getpath")
        ]
        evidence = assess_plan_step_execution(plan, logs)
        by_id = {row["step_id"]: row for row in evidence["steps"]}
        self.assertEqual(by_id["S005"]["observed_count"], 30)
        self.assertEqual(by_id["S006"]["observed_count"], 1)
        self.assertEqual(by_id["S001"]["status"], "not_applicable")
        self.assertNotIn("S001", evidence["missing_step_ids"])

    def test_status_fields_keep_successful_api_separate_from_partial_capture(self):
        state = WorkflowState(TaskRequest(
            description="read path", task_id="task", permissions={"required_api_functions": ["cnc_getpath"]}
        ))
        logs = [ApiCallLog(
            "t", "task", "S005-1", "During", "getpath", {}, 0, {"data": "path_no=1"},
            protocol_function="cnc_getpath", invocation_id="inv-1", traffic_status="no_corresponding_traffic"
        )]
        fields = collection_status_fields(state, logs, True, "partial")
        self.assertEqual(fields["workflow_status"], "completed")
        self.assertEqual(fields["api_execution_status"], "success")
        self.assertEqual(fields["capture_status"], "not_required")
        self.assertIn("句柄本地路径选择", fields["capture_reason"])

    def test_api_traffic_classification_writes_separate_manifests(self):
        logs = [
            ApiCallLog("t", "task", "S1", "During", "statinfo", {}, 0,
                       {"api_pcap_file": "by_api/batch_cnc_statinfo.pcap"},
                       protocol_function="cnc_statinfo", invocation_id="wire-1",
                       transaction_id="tx-1", traffic_status="matched"),
            ApiCallLog("t", "task", "S2", "During", "setpath", {}, 0,
                       {"api_pcap_file": "by_api/batch_cnc_setpath.pcap"},
                       protocol_function="cnc_setpath", invocation_id="local-1",
                       traffic_status="no_corresponding_traffic"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = write_api_traffic_classification(Path(folder), logs)
            with_traffic = Path(result["with_traffic_file"]).read_text(encoding="utf-8")
            without_traffic = Path(result["without_traffic_file"]).read_text(encoding="utf-8")
        self.assertIn('"cnc_statinfo"', with_traffic)
        self.assertNotIn('"cnc_setpath"', with_traffic)
        self.assertIn('"cnc_setpath"', without_traffic)
        self.assertIn('"local_success"', without_traffic)


if __name__ == "__main__":
    unittest.main()
