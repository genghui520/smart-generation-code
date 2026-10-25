from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent.memory import LongTermMemoryStore, MemoryRecord
from smart_traffic_agent.models import TaskRequest, WorkflowState


class LongTermMemoryStoreTests(unittest.TestCase):
    def test_search_includes_curated_error_experience_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("smart_traffic_agent.memory._BGEEmbedding", side_effect=RuntimeError("disabled in fallback test")):
                store = LongTermMemoryStore(Path(tmp) / "repair_memory.jsonl")
                store.experience_path.write_text(
                    json.dumps(
                        {
                            "experience_id": "download4-pairing",
                            "title": "cnc_download4 requires version 4 lifecycle",
                            "status": "verified",
                            "trigger": "EW_DATA during cnc_download4 transfer",
                            "root_cause": "Version 3 start/end functions were paired with version 4 transfer.",
                            "evidence": ["Prior run log showed the mismatched API sequence."],
                            "fix": "Use cnc_dwnstart4, cnc_download4, cnc_dwnend4.",
                            "prevention": "Validate lifecycle API family before execution.",
                            "related_apis": ["cnc_download4", "cnc_dwnstart4", "cnc_dwnend4"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                results = store.search("EW_DATA cnc_download4 version 4 lifecycle", top_k=3)

            self.assertEqual(results[0]["experience_id"], "download4-pairing")
            self.assertEqual(results[0]["record_type"], "curated_error_experience")
            self.assertIn("cnc_dwnstart4", results[0]["fix"])

    def test_curated_experiences_are_added_to_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LongTermMemoryStore(Path(tmp) / "repair_memory.jsonl")
            store.experience_path.write_text(
                json.dumps({"experience_id": "axis-length", "title": "EW_LENGTH axis count"}) + "\n",
                encoding="utf-8",
            )

            class FakeCollection:
                added_ids = []

                def get(self, **kwargs):
                    return {"ids": []}

                def add(self, **kwargs):
                    self.added_ids.extend(kwargs["ids"])

            collection = FakeCollection()
            store._vector_collection = collection
            store._embed_fn = lambda texts: [[1.0, 0.0] for _ in texts]
            store._vector_sync_attempted = True
            store._sync_vector_store()

            self.assertEqual(collection.added_ids, ["experience:axis-length"])

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

    def test_error_episode_is_persisted_immediately_and_updated_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("smart_traffic_agent.memory._BGEEmbedding", side_effect=RuntimeError("disabled in fallback test")):
                store = LongTermMemoryStore(Path(tmp) / "repair_memory.jsonl")
                state = WorkflowState(
                    request=TaskRequest(description="采集轴位置 API", task_id="task-error"),
                    stage="execution",
                    errors=["EW_LENGTH from cnc_absolute"],
                )

                memory_id = store.remember_error(state)
                initial = store.read_all()
                self.assertEqual(len(initial), 1)
                self.assertEqual(initial[0]["status"], "pending")
                self.assertEqual(initial[0]["errors"], ["EW_LENGTH from cnc_absolute"])

                state.repair_attempts = 1
                state.repair_history.append({"repair_stage": "repair_code", "action": "use axis count as length"})
                state.errors = ["EW_LENGTH from cnc_absolute", "axis parameter coverage incomplete"]
                store.remember_error(state)

                updated = store.read_all()
                self.assertEqual(len(updated), 1)
                self.assertEqual(updated[0]["memory_id"], memory_id)
                self.assertEqual(updated[0]["repair_attempts"], 1)
                self.assertIn("axis parameter coverage incomplete", updated[0]["errors"])
                matches = store.search("cnc_absolute EW_LENGTH axis parameter", top_k=3)

            self.assertEqual(matches[0]["memory_id"], memory_id)
            self.assertEqual(matches[0]["record_type"], "workflow_error_episode")


if __name__ == "__main__":
    unittest.main()
