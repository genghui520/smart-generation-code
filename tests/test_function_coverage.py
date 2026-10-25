from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.function_coverage import (
    load_resolved_functions,
    load_terminal_covered_functions,
    merge_workflow_function_coverage_state,
    summarize_function_coverage_state,
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
    def test_full_api_target_is_not_terminal_without_matched_pcap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_state.json"
            state = WorkflowState(request=TaskRequest(
                description="full API", task_id="coverage-pcap",
                permissions={"require_planned_target_traffic": True},
            ))
            state.plan = ExecutionPlan(
                plan_id="coverage-pcap", task_id="coverage-pcap", scenario_type="program_management",
                scenario_goal="capture", target_environment="ncguide-generated-cpp",
                nc_program_type="test", nc_program_requirements=[],
                nc_program_spec=NcProgramSpec(program_name="O1234"), steps=[], expected_outputs=[],
                rag_context={"function_coverage_manifest": {
                    "enabled": True, "selected_batch": [{"function": "cnc_statinfo", "coverage_role": "target"}],
                }},
            )
            log = api_log("coverage-pcap", "S001", "cnc_statinfo", 0)
            state.result = ExecutionResult(
                task_id="coverage-pcap", success=True, api_logs=[log],
                capture_events=[], output_dir=Path(tmp),
            )
            merge_workflow_function_coverage_state(state, path)
            self.assertNotIn("cnc_statinfo", load_terminal_covered_functions(path))
            # An unmatched success plus a matched error is not captured success.
            state.result.api_logs.append(api_log("coverage-pcap", "S002", "cnc_statinfo", 12, traffic_status="matched"))
            merge_workflow_function_coverage_state(state, path)
            self.assertNotIn("cnc_statinfo", load_terminal_covered_functions(path))
            log.traffic_status = "matched"
            merge_workflow_function_coverage_state(state, path)
            self.assertIn("cnc_statinfo", load_terminal_covered_functions(path))

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
                    api_log("coverage-state-001", "S001", "cnc_absolute", 0, traffic_status="matched"),
                    api_log("coverage-state-001", "S002", "cnc_machine", 6, traffic_status="matched"),
                    api_log("coverage-state-001", "S003", "cnc_statinfo", 0, traffic_status="matched"),
                ],
                capture_events=[],
                output_dir=Path(tmp),
            )

            summary = merge_workflow_function_coverage_state(workflow_state, state_path)
            terminal = load_terminal_covered_functions(state_path)

            self.assertEqual(summary["terminal_covered_function_count"], 2)
            self.assertEqual(summary["captured_function_count"], 2)
            self.assertEqual(summary["resolved_function_count"], 3)
            self.assertIn("cnc_absolute", terminal)
            self.assertNotIn("cnc_machine", terminal)
            self.assertIn("cnc_machine", load_resolved_functions(state_path))
            self.assertEqual(summary["status_counts"].get("option_unsupported"), 1)
            self.assertIn("cnc_statinfo", terminal)
            self.assertEqual(summary["support_observed_function_count"], 1)

    def test_unavailable_is_resolved_but_never_counted_as_captured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_state.json"
            from smart_traffic_agent.utils import write_json
            write_json(path, {"functions": {
                "cnc_missing": {"status": "unsupported_by_header", "terminal": True},
                "cnc_good": {"status": "covered_success", "terminal": True, "traffic_matched": True},
                "cnc_legacy": {"status": "covered_success", "terminal": True},
                "cnc_pending": {"status": "attempted_no_traffic", "terminal": False},
            }, "runs": []})
            from smart_traffic_agent.function_coverage import load_function_coverage_state
            summary = summarize_function_coverage_state(load_function_coverage_state(path))
            self.assertEqual(summary["captured_function_count"], 1)
            self.assertEqual(summary["resolved_function_count"], 2)
            self.assertEqual(summary["unavailable_function_count"], 1)
            self.assertEqual(summary["status_counts"]["legacy_unverified"], 1)
            self.assertEqual(load_terminal_covered_functions(path), {"cnc_good"})
            self.assertEqual(load_resolved_functions(path), {"cnc_good", "cnc_missing"})


def api_log(task_id: str, step_id: str, function: str, status_code: int,
            *, traffic_status: str = "unavailable") -> ApiCallLog:
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
        traffic_status=traffic_status,
    )


if __name__ == "__main__":
    unittest.main()
