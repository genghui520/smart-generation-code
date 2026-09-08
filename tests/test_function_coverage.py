from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.function_coverage import (
    load_terminal_covered_functions,
    merge_workflow_function_coverage_state,
)
from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    ExecutionResult,
    NcProgramSpec,
    TaskRequest,
    WorkflowState,
)


class FunctionCoverageStateTests(unittest.TestCase):
    def test_merge_state_tracks_target_updates_and_support_observations_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "coverage_state.json"
            workflow_state = WorkflowState(
                request=TaskRequest(description="cover functions", task_id="coverage-state-001")
            )
            workflow_state.plan = ExecutionPlan(
                plan_id="plan-coverage-state-001",
                task_id="coverage-state-001",
                scenario_type="coordinate_motion",
                scenario_goal="cover coordinate batch",
                target_environment="ncguide-generated-cpp",
                nc_program_type="coverage",
                nc_program_requirements=[],
                nc_program_spec=NcProgramSpec(program_name="O1234"),
                steps=[],
                expected_outputs=[],
                rag_context={
                    "function_coverage_manifest": {
                        "enabled": True,
                        "scenario_batch_id": "programmed_coordinate_motion",
                        "source_manifest": "manifest.jsonl",
                        "target_function_count": 704,
                        "remaining_function_count": 704,
                        "selected_batch": [
                            {"function": "cnc_absolute", "coverage_role": "target"},
                            {"function": "cnc_machine", "coverage_role": "target"},
                            {"function": "cnc_statinfo", "coverage_role": "support"},
                        ],
                    }
                },
            )
            workflow_state.result = ExecutionResult(
                task_id="coverage-state-001",
                success=True,
                api_logs=[
                    api_log("coverage-state-001", "S001", "cnc_absolute", 0),
                    api_log("coverage-state-001", "S002", "cnc_machine", 6),
                    api_log("coverage-state-001", "S003", "cnc_statinfo", 0),
                ],
                capture_events=[],
                output_dir=Path(tmp),
            )

            summary = merge_workflow_function_coverage_state(workflow_state, state_path)
            terminal = load_terminal_covered_functions(state_path)

            self.assertEqual(summary["terminal_covered_function_count"], 2)
            self.assertIn("cnc_absolute", terminal)
            self.assertIn("cnc_machine", terminal)
            self.assertNotIn("cnc_statinfo", terminal)
            self.assertEqual(summary["support_observed_function_count"], 1)


def api_log(task_id: str, step_id: str, function: str, status_code: int) -> ApiCallLog:
    return ApiCallLog(
        timestamp="2026-07-15T00:00:00+00:00",
        task_id=task_id,
        step_id=step_id,
        phase="during",
        interface_name=function,
        input_parameters={},
        status_code=status_code,
        response={},
        protocol_function=function,
    )


if __name__ == "__main__":
    unittest.main()
