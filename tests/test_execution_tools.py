from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent.agent_tools import (
    CollectExecutionArtifactsOutput,
    CollectExecutionArtifactsTool,
    CompileCppOutput,
    CompileGeneratedCppTool,
    RunExecutableOutput,
    RunGeneratedExecutableTool,
)
from smart_traffic_agent.agents.executor import ExecutionAgent
from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    GeneratedArtifacts,
    NcProgramSpec,
    TaskRequest,
    WorkflowState,
    utc_now,
)
from smart_traffic_agent.agent_tools.cpp_execution import (
    env_path_list,
    pcap_sdk_include_paths,
    pcap_sdk_lib_paths,
    read_cpp_csv_logs,
    safe_csv_filename,
    validate_pcap_artifacts,
)


class RecordingCompileTool(CompileGeneratedCppTool):
    def __init__(self, calls: list[str], return_code: int = 0) -> None:
        self.calls = calls
        self.return_code = return_code

    def invoke(self, tool_input):
        self.calls.append(self.name)
        if self.return_code == 0:
            tool_input.executable_path.write_bytes(b"test executable")
        return CompileCppOutput(
            return_code=self.return_code,
            stdout="compile ok" if self.return_code == 0 else "",
            stderr="" if self.return_code == 0 else "compile failed",
            executable_path=tool_input.executable_path,
            command=["cl.exe"],
        )


class RecordingRunTool(RunGeneratedExecutableTool):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def invoke(self, tool_input):
        self.calls.append(self.name)
        return RunExecutableOutput(0, "run ok", "", False, tool_input.timeout_seconds)


class RecordingCollectTool(CollectExecutionArtifactsTool):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def invoke(self, tool_input):
        self.calls.append(self.name)
        log = ApiCallLog(
            timestamp=utc_now(),
            task_id=tool_input.task_id,
            step_id="S001",
            phase="during",
            interface_name="ReadRunStatus",
            input_parameters={},
            status_code=0,
            response={"data": "run=1"},
            protocol_function="cnc_statinfo",
            semantic_label="status_query",
        )
        return CollectExecutionArtifactsOutput([log], [])


class ExecutionToolOrchestrationTests(unittest.TestCase):
    def make_state(self, root: Path) -> WorkflowState:
        source = root / "generated" / "api_script.cpp"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("int main(){return 0;}", encoding="utf-8")
        state = WorkflowState(
            request=TaskRequest(
                description="test generated execution tools",
                task_id="tool-test",
                target_environment="ncguide-generated-cpp",
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-tool-test",
            task_id="tool-test",
            scenario_type="test",
            scenario_goal="test tools",
            target_environment="ncguide-generated-cpp",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[],
            expected_outputs=[],
        )
        state.artifacts = GeneratedArtifacts(
            api_script=source.read_text(encoding="utf-8"),
            nc_program="O1234\nM30\n",
            api_script_path=source,
        )
        return state

    def test_execution_agent_invokes_compile_run_collect_tools_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[str] = []
            agent = ExecutionAgent(
                compile_tool=RecordingCompileTool(calls),
                run_tool=RecordingRunTool(calls),
                collect_tool=RecordingCollectTool(calls),
            )

            state = agent.run(self.make_state(root), root / "run")

            self.assertEqual(
                calls,
                ["compile_generated_cpp", "run_generated_executable", "collect_execution_artifacts"],
            )
            self.assertTrue(state.result.success)
            self.assertEqual([call.tool_name for call in state.result.tool_calls], calls)
            self.assertTrue((root / "run" / "execution" / "tool_calls.jsonl").exists())

    def test_execution_agent_stops_when_compile_tool_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[str] = []
            agent = ExecutionAgent(
                compile_tool=RecordingCompileTool(calls, return_code=2),
                run_tool=RecordingRunTool(calls),
                collect_tool=RecordingCollectTool(calls),
            )

            state = agent.run(self.make_state(root), root / "run")

            self.assertEqual(calls, ["compile_generated_cpp"])
            self.assertFalse(state.result.success)
            self.assertIn("compilation failed", state.result.errors[0])
            self.assertEqual(len(state.result.tool_calls), 1)


class CppToolchainPathTests(unittest.TestCase):
    def test_env_path_list_parses_visual_studio_style_semicolon_paths(self) -> None:
        with patch.dict(
            "os.environ",
            {"WINPCAP_SDK_INCLUDE": r"C:\Lib\WinPcapSDK\Include;D:\Npcap SDK\Include"},
            clear=False,
        ):
            self.assertEqual(
                env_path_list("WINPCAP_SDK_INCLUDE"),
                [Path(r"C:\Lib\WinPcapSDK\Include"), Path(r"D:\Npcap SDK\Include")],
            )

    def test_pcap_sdk_paths_accept_configured_include_and_x86_lib_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "WinPcapSDK"
            include_dir = root / "Include"
            lib_dir = root / "Lib" / "x86"
            include_dir.mkdir(parents=True)
            lib_dir.mkdir(parents=True)
            with patch.dict("os.environ", {"WINPCAP_SDK_DIR": str(root)}, clear=False):
                self.assertIn(include_dir, pcap_sdk_include_paths())
                self.assertIn(lib_dir, pcap_sdk_lib_paths())

    def test_pcap_validation_rejects_header_only_marker_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "capture_before.pcap").write_bytes(b"\x00" * 24)
            (data_dir / "capture_after.pcap").write_bytes(b"\x00" * 24)

            errors = validate_pcap_artifacts(data_dir)

        self.assertTrue(any("24 bytes or smaller" in item for item in errors))

    def test_pcap_validation_accepts_packet_records_beyond_global_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "capture.pcap").write_bytes(b"\x00" * 25)

            errors = validate_pcap_artifacts(data_dir)

        self.assertEqual(errors, [])

    def test_pcap_validation_rejects_csv_disabled_capture_even_with_nonempty_pcap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "capture.pcap").write_bytes(b"\x00" * 25)
            output = data_dir / "focas_api_output.csv"
            output.write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data\n"
                "1,now,S001,cnc_allclibhndl3,0,EW_OK,pcap_capture_enabled=false\n",
                encoding="utf-8",
            )

            errors = validate_pcap_artifacts(data_dir, output_path=output)

        self.assertTrue(any("disabled" in item for item in errors))

    def test_pcap_validation_rejects_stale_pcap_for_current_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            pcap = data_dir / "capture.pcap"
            pcap.write_bytes(b"\x00" * 25)
            output = data_dir / "focas_api_output.csv"
            output.write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data\n"
                "1,now,S001,cnc_allclibhndl3,0,EW_OK,pcap_capture_enabled=true\n",
                encoding="utf-8",
            )
            old_time = output.stat().st_mtime - 600
            pcap.touch()
            import os

            os.utime(pcap, (old_time, old_time))

            errors = validate_pcap_artifacts(data_dir, output_path=output)

        self.assertTrue(any("stale" in item for item in errors))

    def test_collect_writes_combined_api_call_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            execution_dir = Path(tmp)
            data_dir = execution_dir / "data"
            data_dir.mkdir()
            (data_dir / "focas_capture.pcap").write_bytes(b"\x00" * 25)
            input_csv = data_dir / "focas_api_input.csv"
            output_csv = data_dir / "focas_api_output.csv"
            input_csv.write_text(
                "index,timestamp,step_id,phase,interface_name,protocol_function,parameters,api_parameter_count\n"
                "1,now,S001,during,FOCAS,cnc_allclibhndl3,host=127.0.0.1;port=8193,4\n",
                encoding="utf-8",
            )
            output_csv.write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n"
                "1,now,S001,cnc_allclibhndl3,0,EW_OK,handle=32769;pcap_capture_enabled=true,4\n",
                encoding="utf-8",
            )

            read_cpp_csv_logs("task", execution_dir)

            combined = data_dir / "focas_api_calls.csv"
            self.assertTrue(combined.exists())
            text = combined.read_text(encoding="utf-8-sig")
            self.assertIn("input_parameters", text)
            self.assertIn("output_data", text)
            self.assertIn("host=127.0.0.1;port=8193", text)
            self.assertIn("handle=32769", text)
            self.assertIn(",4,", text)

    def test_collect_writes_per_api_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            execution_dir = Path(tmp)
            data_dir = execution_dir / "data"
            data_dir.mkdir()
            (data_dir / "focas_capture.pcap").write_bytes(b"\x00" * 25)
            (data_dir / "focas_api_input.csv").write_text(
                "index,timestamp,step_id,phase,interface_name,protocol_function,parameters,api_parameter_count\n"
                "1,2026-07-16T10:00:00,S001,during,ReadFeedSpeed,cnc_actf,handle=1,2\n"
                "2,2026-07-16T10:00:00,S002,during,ReadRunStatus,cnc_statinfo,handle=1,2\n",
                encoding="utf-8",
            )
            (data_dir / "focas_api_output.csv").write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n"
                "1,2026-07-16T10:00:00,S001,cnc_actf,0,EW_OK,actual_feed=120;pcap_capture_enabled=true,2\n"
                "2,2026-07-16T10:00:00,S002,cnc_statinfo,0,EW_OK,run=3;motion=1,2\n",
                encoding="utf-8",
            )

            read_cpp_csv_logs("task", execution_dir)

            actf_csv = data_dir / "by_api" / "20260716_cnc_actf.csv"
            stat_csv = data_dir / "by_api" / "20260716_cnc_statinfo.csv"
            descriptive_input_csv = data_dir / "20260716_input.csv"
            descriptive_output_csv = data_dir / "20260716_output.csv"
            self.assertTrue(actf_csv.exists())
            self.assertTrue(stat_csv.exists())
            self.assertTrue(descriptive_input_csv.exists())
            self.assertTrue(descriptive_output_csv.exists())
            actf_text = actf_csv.read_text(encoding="utf-8-sig")
            self.assertIn("handle=1", actf_text)
            self.assertIn("actual_feed=120", actf_text)
            self.assertIn(",2,", actf_text)

    def test_collect_removes_stale_per_api_csvs_from_previous_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            execution_dir = Path(tmp)
            data_dir = execution_dir / "data"
            by_api_dir = data_dir / "by_api"
            data_dir.mkdir()
            by_api_dir.mkdir()
            stale = by_api_dir / "ncguide_ui_cycle_start.csv"
            stale.write_text("stale previous repair attempt\n", encoding="utf-8")
            (data_dir / "focas_capture.pcap").write_bytes(b"\x00" * 25)
            (data_dir / "focas_api_input.csv").write_text(
                "index,timestamp,step_id,phase,interface_name,protocol_function,parameters,api_parameter_count\n"
                "1,now,S001,during,ReadRunStatus,cnc_statinfo,handle=1,2\n",
                encoding="utf-8",
            )
            (data_dir / "focas_api_output.csv").write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n"
                "1,now,S001,cnc_statinfo,0,EW_OK,run=1;motion=0;pcap_capture_enabled=true,2\n",
                encoding="utf-8",
            )

            read_cpp_csv_logs("task", execution_dir)

            self.assertFalse(stale.exists())
            self.assertTrue(list(by_api_dir.glob("*_cnc_statinfo.csv")))

    def test_safe_csv_filename_truncates_long_aggregate_api_names(self) -> None:
        name = "cnc_exaxisname -> cnc_exaxisname2 -> cnc_rdspdlname -> cnc_rdmdiprgstat -> cnc_rdmdipntr"

        safe = safe_csv_filename(name, max_length=64)

        self.assertLessEqual(len(safe), 64)
        self.assertNotIn(" ", safe)
        self.assertNotIn(">", safe)
        self.assertRegex(safe, r"_[0-9a-f]{10}$")


if __name__ == "__main__":
    unittest.main()
