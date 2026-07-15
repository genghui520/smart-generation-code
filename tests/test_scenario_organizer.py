from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.rag.scenario_organizer import build_scenario_knowledge
from smart_traffic_agent.utils import write_jsonl


class ScenarioOrganizerTests(unittest.TestCase):
    def test_builds_scene_centered_rule_organization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api_chunks = root / "api.jsonl"
            rule_chunks = root / "rules.jsonl"
            output = root / "scenario_knowledge.json"
            scenario_chunks = root / "scenario_chunks.jsonl"
            write_jsonl(
                api_chunks,
                [
                    {
                        "chunk_id": "api-cnc-statinfo",
                        "function": "cnc_statinfo",
                        "chunk_type": "overview",
                        "category": "status",
                        "text": "Function: cnc_statinfo Read CNC status information.",
                    }
                ],
            )
            write_jsonl(
                rule_chunks,
                [
                    sample_rule("nc_rule", "Use a short NC program."),
                    sample_rule("operation_rule", "Start after selecting the program."),
                    sample_rule("collection_rule", "Read status before and after execution."),
                    sample_rule("safety_rule", "Use simulator-only execution for risky actions."),
                ],
            )

            payload = build_scenario_knowledge(
                api_chunks_path=api_chunks,
                rule_chunks_path=rule_chunks,
                output_path=output,
                scenario_chunks_path=scenario_chunks,
            )

            by_id = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}
            coordinate = by_id["coordinate_motion"]
            self.assertIn("api_rule", coordinate["rules"])
            self.assertEqual(len(coordinate["rules"]["nc_rule"]), 1)
            self.assertEqual(len(coordinate["rules"]["operation_rule"]), 1)
            self.assertEqual(len(coordinate["rules"]["collection_rule"]), 1)
            self.assertEqual(len(coordinate["rules"]["safety_rule"]), 1)
            self.assertTrue(coordinate["rules"]["api_rule"])
            self.assertTrue(output.exists())
            self.assertTrue(scenario_chunks.exists())


def sample_rule(rule_type: str, text: str) -> dict[str, object]:
    return {
        "rule_id": f"rule-{rule_type}",
        "rule_type": rule_type,
        "scenario": "coordinate_motion",
        "rule_text": text,
        "recommended_api_functions": ["cnc_statinfo"],
        "source_file": "manual.pdf",
        "source_chunk_id": "chunk-1",
    }


if __name__ == "__main__":
    unittest.main()
