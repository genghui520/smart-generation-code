from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
