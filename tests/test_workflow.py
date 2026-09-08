from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    ExecutionResult,
    NcProgramSpec,
    PlanStep,
    QualityAssessment,
    TaskRequest,
    WorkflowState,
)
from smart_traffic_agent.workflow import TrafficGenerationWorkflow, workflow_success, workflow_summary


class WorkflowTests(unittest.TestCase):
    def test_workflow_success_skips_dynamic_quality_gate_when_disabled(self) -> None:
        state = WorkflowState(
            request=TaskRequest(
                description="generate coordinate traffic",
                task_id="test-quality-disabled",
                target_environment="simulator",
                quality_gate_enabled=False,
            )
        )
        state.plan = ExecutionPlan(
            plan_id="plan-quality-disabled",
            task_id="test-quality-disabled",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="simulator",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[PlanStep("S001", "during", "start", "StartProgram", {})],
            expected_outputs=[],
        )
        state.result = ExecutionResult(
            task_id="test-quality-disabled",
            success=True,
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )

        self.assertTrue(workflow_success(state))

    def test_workflow_requires_configured_llm_in_agent_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(
                description="generate coordinate change traffic for CNC simulator",
                task_id="test001",
            )
            workflow = TrafficGenerationWorkflow(sample_knowledge())

            with self.assertRaisesRegex(RuntimeError, "requires an LLM decision"):
                workflow.run(request, Path(tmp))

    def test_workflow_success_ignores_quality_metrics_for_failed_execution(self) -> None:
        state = WorkflowState(request=TaskRequest(description="generate coordinate traffic", task_id="test002"))
        state.result = ExecutionResult(
            task_id="test002",
            success=False,
            errors=["nonfatal optional API warning"],
            api_logs=[],
            capture_events=[],
            output_dir=Path("."),
        )
        state.quality_assessment = QualityAssessment(
            passed=True,
            metrics={
                "changed_output_parameter_count": 4,
                "feed_sample_count": 10,
                "feed_unique_count": 2,
                "position_sample_count": 10,
                "position_unique_count": 2,
                "run_active_count": 3,
                "motion_active_count": 3,
                "program_completed": True,
            },
        )

        self.assertFalse(workflow_success(state))

    def test_workflow_uses_repair_aware_recursion_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(description="generate coordinate traffic", task_id="test003")
            workflow = TrafficGenerationWorkflow(sample_knowledge())
            workflow.graph = Mock()
            workflow.graph.invoke.return_value = {"workflow": WorkflowState(request=request)}

            workflow.run(request, Path(tmp))

            config = workflow.graph.invoke.call_args.kwargs["config"]
            self.assertGreaterEqual(config["recursion_limit"], 128)

    def test_workflow_exposes_explicit_agent_graph_contract(self) -> None:
        workflow = TrafficGenerationWorkflow(sample_knowledge())

        graph = workflow.agent_graph.describe()
        nodes = {item["name"]: item["agent"] for item in graph["nodes"]}

        self.assertEqual(nodes["router"], "RouterRole")
        self.assertEqual(nodes["planning"], "PlanningRole")
        self.assertEqual(nodes["code_generation"], "CodeEngineerRole")
        self.assertEqual(nodes["execution"], "ExecutionRole")
        self.assertNotIn("repair_plan", nodes)
        self.assertNotIn("repair_code", nodes)
        self.assertNotIn("repair_execution", nodes)
        self.assertTrue(graph["conditional_edges"])

    def test_workflow_summary_includes_function_coverage_metrics(self) -> None:
        state = WorkflowState(
            request=TaskRequest(description="cover all functions", task_id="coverage001")
        )
        state.plan = ExecutionPlan(
            plan_id="plan-coverage001",
            task_id="coverage001",
            scenario_type="comprehensive_focas_traffic",
            scenario_goal="cover manifest batch",
            target_environment="ncguide-generated-cpp",
            nc_program_type="coverage",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[],
            expected_outputs=[],
            rag_context={
                "function_coverage_manifest": {
                    "enabled": True,
                    "source_manifest": "rag_indexes/focas/function_manifest.jsonl",
                    "scenario_batch_id": "programmed_coordinate_motion",
                    "main_state_driver": "upload_select_run_single_block_nc_program",
                    "quality_target": "program lifecycle evidence plus position/feed/status variation",
                    "target_function_count": 704,
                    "remaining_function_count": 704,
                    "selected_batch": [
                        {"function": "cnc_rdnodenum", "coverage_role": "target", "segment_id": "01_system"},
                        {"function": "cnc_rdnodeinfo", "coverage_role": "target", "segment_id": "01_system"},
                        {"function": "pmc_rdpmcrng", "coverage_role": "target", "segment_id": "02_pmc"},
                        {"function": "cnc_statinfo", "coverage_role": "support", "segment_id": "01_system"},
                    ],
                }
            },
        )
        state.result = ExecutionResult(
            task_id="coverage001",
            success=True,
            api_logs=[
                ApiCallLog(
                    timestamp="2026-07-15T00:00:00+00:00",
                    task_id="coverage001",
                    step_id="S001",
                    phase="during",
                    interface_name="ReadNodeNumber",
                    input_parameters={},
                    status_code=0,
                    response={},
                    protocol_function="cnc_rdnodenum",
                ),
                ApiCallLog(
                    timestamp="2026-07-15T00:00:01+00:00",
                    task_id="coverage001",
                    step_id="S002",
                    phase="during",
                    interface_name="ReadPmcRange",
                    input_parameters={},
                    status_code=6,
                    response={},
                    protocol_function="pmc_rdpmcrng",
                ),
                ApiCallLog(
                    timestamp="2026-07-15T00:00:02+00:00",
                    task_id="coverage001",
                    step_id="S003",
                    phase="during",
                    interface_name="ReadRunStatus",
                    input_parameters={},
                    status_code=0,
                    response={},
                    protocol_function="cnc_statinfo",
                ),
            ],
            capture_events=[],
            output_dir=Path("."),
        )

        coverage = workflow_summary(state)["function_coverage"]

        self.assertEqual(coverage["batch_function_count"], 3)
        self.assertEqual(coverage["support_batch_function_count"], 1)
        self.assertEqual(coverage["attempted_batch_function_count"], 2)
        self.assertEqual(coverage["attempted_support_function_count"], 1)
        self.assertEqual(coverage["successful_batch_functions"], ["cnc_rdnodenum"])
        self.assertEqual(coverage["successful_support_functions"], ["cnc_statinfo"])
        self.assertEqual(coverage["expected_error_batch_functions"], ["pmc_rdpmcrng"])
        self.assertEqual(coverage["missing_batch_functions"], ["cnc_rdnodeinfo"])
        self.assertEqual(len(coverage["segment_metrics"]), 2)
        self.assertEqual(coverage["segment_metrics"][0]["segment_id"], "01_system")


if __name__ == "__main__":
    unittest.main()
