from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.knowledge import sample_knowledge
from smart_traffic_agent.models import TaskRequest
from smart_traffic_agent.workflow import TrafficGenerationWorkflow


class WorkflowTests(unittest.TestCase):
    def test_coordinate_workflow_runs_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = TaskRequest(
                description="generate coordinate change traffic for CNC simulator",
                task_id="test001",
            )
            state = TrafficGenerationWorkflow(sample_knowledge()).run(request, Path(tmp))

            self.assertEqual(state.stage, "complete")
            self.assertIsNotNone(state.plan)
            self.assertEqual(state.plan.scenario_type, "coordinate_motion")
            self.assertIsNotNone(state.result)
            self.assertTrue(state.result.success)
            self.assertGreater(len(state.mapping), 0)
            self.assertTrue((Path(tmp) / "generated" / "api_script.py").exists())
            self.assertTrue((Path(tmp) / "execution" / "mapping.json").exists())


if __name__ == "__main__":
    unittest.main()

