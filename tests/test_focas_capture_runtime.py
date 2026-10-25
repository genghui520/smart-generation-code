from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from smart_traffic_agent import focas_capture_runtime
from smart_traffic_agent.focas_capture_runtime import guarded_core, skip_read_core, download4_core, wrap_capture_fragment, validate_capture_fragment, render_capture_harness
from smart_traffic_agent.agent_tools.cpp_execution import discover_capture_pcap, read_cpp_csv_logs
from smart_traffic_agent.agents.router import valid_next_stages
from smart_traffic_agent.agents.executor import required_traffic_errors
from smart_traffic_agent.models import ApiCallLog, ExecutionResult, TaskRequest, WorkflowState, ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.function_coverage import build_function_coverage_metrics
from smart_traffic_agent.focas_contracts import named_contract_context


class ProductionCaptureTests(unittest.TestCase):
    def test_renderer_sets_manual_mode_gates_from_target_apis(self):
        with patch.object(focas_capture_runtime._legacy, "render_capture_harness", return_value="remote=__REQUIRE_REMOTE__;reads=__ALLOW_ALARM_READS__"):
            dnc_source = render_capture_harness(
                ["cnc_dnc2"], contract=SimpleNamespace(targets=("cnc_dnc2",))
            )
            read_source = render_capture_harness(
                ["cnc_statinfo"], contract=SimpleNamespace(targets=("cnc_statinfo",))
            )

        self.assertEqual(dnc_source, "remote=true;reads=false")
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
        self.assertIn("sizeof(ODBAXIS)", fragment)
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


if __name__ == "__main__":
    unittest.main()
