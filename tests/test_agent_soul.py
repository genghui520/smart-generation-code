from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smart_traffic_agent.agent_soul import AgentSoulStore, normalize_agent_name
from smart_traffic_agent.memory import LongTermMemoryStore


class AgentSoulTests(unittest.TestCase):
    def test_loads_only_requested_agent_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            router = root / "router"
            router.mkdir()
            (router / "core.md").write_text("route from evidence", encoding="utf-8")
            (router / "skills.yaml").write_text("tools:\n  - search_knowledge\n", encoding="utf-8")

            context = AgentSoulStore(root).prompt_context("RouterRole", task_context="compile error")

            self.assertIn("route from evidence", context)
            self.assertIn("search_knowledge", context)
            self.assertNotIn("planner", context.lower())

    def test_aliases_are_stable(self) -> None:
        self.assertEqual(normalize_agent_name("CodeGenerationAgent"), "code_generator")
        with self.assertRaises(ValueError):
            normalize_agent_name("unknown-agent")

    def test_verified_experience_is_not_downgraded_by_later_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "smart_traffic_agent.memory._BGEEmbedding", side_effect=RuntimeError("disabled")
        ):
            store = LongTermMemoryStore(Path(tmp) / "memory.jsonl")
            store.submit_agent_experience("PlannerRole", {"experience_id": "exp-1", "fact": "verified"}, verified=True)
            record = store.submit_agent_experience("PlannerRole", {"experience_id": "exp-1", "fact": "new hypothesis"}, verified=False)
            self.assertEqual(record["status"], "runtime_verified")

    def test_workflow_updates_each_agent_soul_and_prompt_reloads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("router", "planner", "code_generator", "executor"):
                folder = root / name
                folder.mkdir()
                (folder / "core.md").write_text(f"{name} core", encoding="utf-8")
                (folder / "skills.yaml").write_text("version: 1\n", encoding="utf-8")
            log = SimpleNamespace(protocol_function="cnc_statinfo", status_code=0, traffic_status="matched")
            result = SimpleNamespace(errors=[], api_logs=[log], collection_status="collected", capture_events=[{}])
            request = SimpleNamespace(
                task_id="soul-update-task",
                target_environment="ncguide-generated-cpp",
                permissions={"coverage_current_batch_functions": ["cnc_statinfo"]},
            )
            plan = SimpleNamespace(scenario_type="read", rag_context={"scene_plan": {"target_api": "cnc_statinfo"}})
            state = SimpleNamespace(
                request=request, plan=plan, result=result, errors=[], repair_attempts=0,
                repair_history=[], artifacts=object(), stage="complete",
            )
            store = AgentSoulStore(root)
            records = store.learn_from_workflow(state, workflow_success=True)
            self.assertEqual(len(records), 4)
            self.assertTrue(all(item["status"] == "runtime_verified" for item in records))
            context = store.prompt_context("PlannerRole", task_context="cnc_statinfo")
            self.assertIn("Dynamic Agent Evolution", context)
            self.assertIn("soul-update-task", context)
            self.assertNotIn('"learning_scope": "actual calls, return codes, PCAP and terminal evidence"', context)
            self.assertTrue((root / "planner" / "evolution.jsonl").is_file())

    def test_agent_memory_is_scoped_and_unscoped_legacy_rows_are_not_injected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "smart_traffic_agent.memory._BGEEmbedding", side_effect=RuntimeError("disabled")
        ):
            store = LongTermMemoryStore(Path(tmp) / "memory.jsonl")
            store.submit_agent_experience("planner", {"experience_id": "planner-only", "fact": "planning fact"})
            store.submit_agent_experience("executor", {"experience_id": "executor-only", "fact": "execution fact"})
            store.submit_agent_experience("", {"experience_id": "shared-legacy", "fact": "legacy shared fact"})
            planner = store.search_agent_experiences("planner", "fact", top_k=10)
            self.assertEqual({row["experience_id"] for row in planner}, {"planner-only"})
            shared = store.search_agent_experiences("planner", "fact", top_k=10, include_shared=True)
            self.assertEqual({row["experience_id"] for row in shared}, {"planner-only", "shared-legacy"})


if __name__ == "__main__":
    unittest.main()
