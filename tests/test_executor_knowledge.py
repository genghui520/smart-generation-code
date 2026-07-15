from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.agents.executor import attach_execution_error_knowledge, read_cpp_csv_logs
from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    ExecutionResult,
    NcProgramSpec,
    TaskRequest,
    WorkflowState,
)


class ExecutorKnowledgeTests(unittest.TestCase):
    def test_execution_error_knowledge_is_attached_to_plan(self) -> None:
        state = WorkflowState(request=TaskRequest(description="generate coordinate traffic", task_id="exec001"))
        state.plan = ExecutionPlan(
            plan_id="plan-exec001",
            task_id="exec001",
            scenario_type="coordinate_motion",
            scenario_goal="test",
            target_environment="simulator",
            nc_program_type="test",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[],
            expected_outputs=[],
        )
        state.result = ExecutionResult(
            task_id="exec001",
            success=False,
            api_logs=[
                ApiCallLog(
                    timestamp="t",
                    task_id="exec001",
                    step_id="S001",
                    phase="before",
                    interface_name="ReadPosition",
                    input_parameters={},
                    status_code=5,
                    response={"data": "FOCAS return code 5"},
                    protocol_function="cnc_rdposition",
                    error="FOCAS return code 5",
                )
            ],
            capture_events=[],
            output_dir=Path(tempfile.gettempdir()),
            errors=["FOCAS return code 5"],
        )

        attach_execution_error_knowledge(state, sample_knowledge())

        self.assertIn("execution_error_knowledge", state.plan.rag_context)
        self.assertTrue(state.plan.rag_context["execution_error_knowledge"])

    def test_read_cpp_csv_logs_uses_csv_timestamp(self) -> None:
        state = WorkflowState(request=TaskRequest(description="generate coordinate traffic", task_id="exec001"))
        with tempfile.TemporaryDirectory() as tmp:
            execution_dir = Path(tmp)
            data_dir = execution_dir / "data"
            data_dir.mkdir()
            (data_dir / "focas_api_input.csv").write_text(
                "index,timestamp,step_id,phase,interface_name,protocol_function,parameters\n"
                '1,"2026-07-10T10:00:00.123","S001","during","ReadFeedSpeed","cnc_actf","sample=1"\n',
                encoding="utf-8",
            )
            (data_dir / "focas_api_output.csv").write_text(
                "index,timestamp,step_id,api_name,return_code,return_text,data\n"
                '1,"2026-07-10T10:00:00.123","S001","cnc_actf",0,"EW_OK","feed=120"\n',
                encoding="utf-8",
            )

            api_logs, capture_events = read_cpp_csv_logs(state, execution_dir)

        self.assertEqual(api_logs[0].timestamp, "2026-07-10T10:00:00.123")
        self.assertEqual(capture_events[0].timestamp, "2026-07-10T10:00:00.123")


if __name__ == "__main__":
    unittest.main()
