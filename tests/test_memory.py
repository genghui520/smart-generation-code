from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.memory import LongTermMemoryStore, MemoryRecord


class LongTermMemoryStoreTests(unittest.TestCase):
    def test_appends_and_searches_repair_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "repair_memory.jsonl")
            store.append(
                MemoryRecord(
                    memory_id="mem001",
                    timestamp="2026-07-08T00:00:00+00:00",
                    task_description="coordinate motion traffic",
                    task_id="task001",
                    target_environment="ncguide-generated-cpp",
                    scenario_type="coordinate_motion",
                    final_success=True,
                    repair_attempts=1,
                    repair_history=[
                        {
                            "repair_stage": "repair_execution",
                            "errors": ["Cycle Start click failed"],
                            "action": "retry ExecutionAgent",
                        }
                    ],
                    notes=["repair_execution fixed Cycle Start click"],
                )
            )

            results = store.search("NCGuide Cycle Start execution click")

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["memory_id"], "mem001")
            self.assertGreater(results[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
