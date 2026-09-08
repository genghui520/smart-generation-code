from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent.memory import LongTermMemoryStore, MemoryRecord


class LongTermMemoryStoreTests(unittest.TestCase):
    def test_appends_and_searches_repair_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("smart_traffic_agent.memory._BGEEmbedding", side_effect=RuntimeError("disabled in fallback test")):
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

    def test_embedding_search_is_used_when_vector_store_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "repair_memory.jsonl"
            store = LongTermMemoryStore(memory_path)
            store._vector_sync_attempted = True
            store.append(
                MemoryRecord(
                    memory_id="mem-embedding",
                    timestamp="2026-07-08T00:00:00+00:00",
                    task_description="coordinate motion generated code",
                    task_id="task-embedding",
                    target_environment="ncguide-generated-cpp",
                    scenario_type="coordinate_motion",
                    final_success=False,
                    repair_attempts=1,
                    repair_history=[{"repair_stage": "repair_code", "errors": ["ODBPOS ABI mismatch"]}],
                )
            )

            class FakeCollection:
                def query(self, **kwargs):
                    return {"ids": [["mem-embedding"]], "distances": [[0.1]], "metadatas": [[{"memory_id": "mem-embedding"}]]}

            store._vector_collection = FakeCollection()
            store._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
            store._vector_sync_attempted = True
            results = store.search("coordinate position ABI")
            store.close()

            self.assertEqual(results[0]["memory_id"], "mem-embedding")
            self.assertEqual(results[0]["retrieval_method"], "embedding")


if __name__ == "__main__":
    unittest.main()
