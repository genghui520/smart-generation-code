from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import ExecutionResult, QualityAssessment, TaskRequest, WorkflowState
from smart_traffic_agent.workflow import TrafficGenerationWorkflow, workflow_success


class WorkflowTests(unittest.TestCase):
    def test_workflow_requires_configured_llm_in_agent_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(
                description="generate coordinate change traffic for CNC simulator",
                task_id="test001",
            )
            workflow = TrafficGenerationWorkflow(sample_knowledge())

            with self.assertRaisesRegex(RuntimeError, "requires an LLM decision"):
                workflow.run(request, Path(tmp))

    def test_workflow_success_accepts_sufficient_output_variation(self) -> None:
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

        self.assertTrue(workflow_success(state))

    def test_workflow_uses_repair_aware_recursion_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(description="generate coordinate traffic", task_id="test003")
            workflow = TrafficGenerationWorkflow(sample_knowledge())
            workflow.graph = Mock()
            workflow.graph.invoke.return_value = {"workflow": WorkflowState(request=request)}

            workflow.run(request, Path(tmp))

            config = workflow.graph.invoke.call_args.kwargs["config"]
            self.assertGreaterEqual(config["recursion_limit"], 128)


if __name__ == "__main__":
    unittest.main()
