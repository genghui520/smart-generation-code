from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
