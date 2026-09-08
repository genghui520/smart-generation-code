from __future__ import annotations

import unittest
from unittest.mock import Mock

from smart_traffic_agent.agents.router import (
    RouterAgent,
    retrieve_router_knowledge_context,
    summarize_api_log_quality_evidence,
    summarize_quality_for_router,
    valid_next_stages,
)
from smart_traffic_agent.knowledge import sample_knowledge
from pathlib import Path

from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    ExecutionResult,
    GeneratedArtifacts,
    NcProgramSpec,
    QualityAssessment,
    TaskRequest,
    WorkflowState,
)


def state_with_plan_without_artifacts() -> WorkflowState:
    state = WorkflowState(request=TaskRequest(description="generate traffic", task_id="router001"))
    state.plan = ExecutionPlan(
        plan_id="plan-router001",
        task_id="router001",
        scenario_type="coordinate_motion",
        scenario_goal="test",
        target_environment="simulator",
        nc_program_type="test",
        nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name="O1234"),
        steps=[],
        expected_outputs=[],
    )
    return state


class RouterAgentTests(unittest.TestCase):
    def test_router_deterministically_completes_after_quality_gate_passes(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=True,
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )
        state.quality_assessment = QualityAssessment(
            passed=True,
            metrics={
                "changed_output_parameter_count": 6,
                "feed_sample_count": 20,
                "feed_unique_count": 3,
                "position_sample_count": 20,
                "position_unique_count": 5,
                "run_active_count": 10,
                "motion_active_count": 10,
                "program_completed": True,
            },
        )
        llm = Mock()
        llm.enabled = True
        router = RouterAgent(llm_client=llm)

        self.assertEqual(router.route(state), "complete")
        llm.invoke_json.assert_not_called()
        self.assertEqual(router.last_route_source, "deterministic_quality_gate")

    def test_valid_next_stage_requires_planning_when_plan_missing(self) -> None:
        state = WorkflowState(request=TaskRequest(description="generate traffic", task_id="router001"))

        self.assertEqual(valid_next_stages(state), {"planning"})

    def test_valid_next_stage_requires_code_generation_when_artifacts_missing(self) -> None:
        self.assertEqual(valid_next_stages(state_with_plan_without_artifacts()), {"code_generation"})

    def test_failed_quality_allows_repair_stages(self) -> None:
        state = state_with_plan_without_artifacts()
        state.quality_assessment = QualityAssessment(
            passed=False,
            metrics={},
            issues=["feed did not vary"],
            recommendations=["replan slower feed and denser sampling"],
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan", "repair_code", "repair_execution"})

    def test_successful_execution_still_allows_router_quality_judgment(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=True,
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )
        state.quality_assessment = QualityAssessment(
            passed=True,
            metrics={"feed_sample_count": 3, "feed_unique_count": 1},
            recommendations=["RouterAgent must evaluate metrics."],
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan", "repair_code", "repair_execution"})

        summary = summarize_quality_for_router(state)

        self.assertEqual(summary["feed_sample_count"], 3)
        self.assertIn("feed_unique_count", summary)

    def test_successful_execution_optional_lifecycle_skip_does_not_force_repair(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=True,
            api_logs=[
                ApiCallLog(
                    timestamp="2026-07-17T00:00:00Z",
                    task_id="router001",
                    step_id="S003_PLANNED_BEFORE_API_cnc_delete_or_equivalent_exact_program_delete",
                    phase="before",
                    interface_name="cnc_delete_or_equivalent_exact_program_delete",
                    protocol_function="cnc_delete_or_equivalent_exact_program_delete",
                    input_parameters={},
                    status_code=1,
                    response={
                        "return_text": "SKIPPED_UNSUPPORTED_BY_FIXED_ADAPTER",
                        "data": "",
                    },
                    error="SKIPPED_UNSUPPORTED_BY_FIXED_ADAPTER",
                )
            ],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan", "repair_code", "repair_execution"})

    def test_failed_result_with_sufficient_output_variation_cannot_complete(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["cycle_start_ready_gate timeout; click suppressed"],
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )
        state.quality_assessment = QualityAssessment(
            passed=True,
            metrics={
                "changed_output_parameter_count": 6,
                "feed_sample_count": 20,
                "feed_unique_count": 3,
                "position_sample_count": 20,
                "position_unique_count": 5,
                "run_active_count": 10,
                "motion_active_count": 10,
                "program_completed": True,
            },
            recommendations=["RouterAgent must evaluate metrics."],
        )

        self.assertNotIn("complete", valid_next_stages(state))

    def test_router_can_retrieve_error_knowledge(self) -> None:
        state = state_with_plan_without_artifacts()
        state.errors.append("FOCAS return code 5 during UploadProgram")

        rows = retrieve_router_knowledge_context(sample_knowledge(), state)

        self.assertTrue(rows)

    def test_generated_cpp_stack_overflow_requires_code_repair(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["C++ API script execution failed with exit code 3221225725."],
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_code"})

    def test_failed_program_delete_requires_code_repair(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["FOCAS_RET_-1"],
            api_logs=[
                ApiCallLog(
                    timestamp="2026-07-14T00:00:00Z",
                    task_id="router001",
                    step_id="S003",
                    phase="before",
                    interface_name="DeleteProgram",
                    protocol_function="cnc_delete",
                    input_parameters={},
                    status_code=-1,
                    response={"data": "target_program_deleted=false"},
                    error="FOCAS_RET_-1",
                )
            ],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_code"})

    def test_selected_target_program_requires_replan(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["TARGET_PROGRAM_SELECTED_REPLAN_REQUIRED"],
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan"})

    def test_existing_target_program_requires_replan(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O1234\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED"],
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan"})

    def test_upload_end_program_number_conflict_requires_replan(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O3586\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["FOCAS_RET_5", "PROGRAM_NOT_VERIFIED"],
            api_logs=[
                ApiCallLog(
                    timestamp="2026-07-14T00:00:00Z",
                    task_id="router001",
                    step_id="S005",
                    phase="before",
                    interface_name="EndProgramUpload",
                    protocol_function="cnc_dwnend3",
                    input_parameters={"program_name": "O3586"},
                    status_code=5,
                    response={"data": "upload_end_success=false;PROGRAM_NOT_VERIFIED"},
                    error="FOCAS_RET_5",
                )
            ],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertEqual(valid_next_stages(state), {"repair_plan"})

    def test_router_requires_success_even_when_raw_api_log_evidence_is_good(self) -> None:
        state = state_with_plan_without_artifacts()
        state.artifacts = GeneratedArtifacts(api_script="int main(){}", nc_program="O3266\nM30\n")
        state.result = ExecutionResult(
            task_id="router001",
            success=False,
            errors=["FOCAS_RETURN_2"],
            api_logs=[
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="S002",
                    phase="before",
                    interface_name="ReadProgramDirectoryForExistence",
                    protocol_function="cnc_rdprogdir",
                    input_parameters={},
                    status_code=2,
                    response={"data": "target_program_exists=false;program_number_available=true"},
                    error="FOCAS_RETURN_2",
                ),
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="S007",
                    phase="before",
                    interface_name="ReadProgramNumber",
                    protocol_function="cnc_rdprgnum",
                    input_parameters={},
                    status_code=0,
                    response={"data": "current_program=3266;main_program=3266;program_verified=true"},
                ),
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="S015",
                    phase="during",
                    interface_name="cnc_absolute",
                    protocol_function="cnc_absolute",
                    input_parameters={},
                    status_code=0,
                    response={"data": "position_axis1=168;position_axis2=0;position_axis3=5000"},
                ),
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="S016",
                    phase="during",
                    interface_name="ReadFeedSpeed",
                    protocol_function="cnc_actf",
                    input_parameters={},
                    status_code=0,
                    response={"data": "actual_feed=210;feed=210"},
                ),
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="PROGRAM_COMPLETION_GATE",
                    phase="after",
                    interface_name="program_completion_gate",
                    protocol_function="program_completion_gate",
                    input_parameters={},
                    status_code=0,
                    response={"data": "program_completion_gate=true;completed=true;timeout=false"},
                ),
                ApiCallLog(
                    timestamp="t",
                    task_id="router001",
                    step_id="S023",
                    phase="after",
                    interface_name="EvaluateTrafficQuality",
                    protocol_function="local_quality_evaluation",
                    input_parameters={},
                    status_code=0,
                    response={"data": "quality_pass=true;position_variation=true;feed_variation=true"},
                ),
            ],
            capture_events=[],
            output_dir=Path("."),
        )
        state.quality_assessment = QualityAssessment(
            passed=True,
            metrics={
                "feed_sample_count": 134,
                "feed_unique_count": 8,
                "position_sample_count": 0,
                "position_unique_count": 0,
                "program_completed": True,
            },
        )

        self.assertNotIn("complete", valid_next_stages(state))

        evidence = summarize_api_log_quality_evidence(state)
        self.assertTrue(evidence["program_verified"])
        self.assertTrue(evidence["program_completed"])
        self.assertTrue(evidence["local_quality_pass"])
        self.assertTrue(evidence["position_rows"])


if __name__ == "__main__":
    unittest.main()
