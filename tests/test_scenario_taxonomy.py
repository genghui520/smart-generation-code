from __future__ import annotations

import unittest

from smart_traffic_agent.rag.scenario_taxonomy import RULE_TYPES, SCENARIOS


class ScenarioTaxonomyTests(unittest.TestCase):
    def test_scenario_ids_are_unique(self) -> None:
        scenario_ids = [scenario.scenario_id for scenario in SCENARIOS]
        self.assertEqual(len(scenario_ids), len(set(scenario_ids)))

    def test_four_rule_types(self) -> None:
        self.assertEqual(
            set(RULE_TYPES),
            {"nc_rule", "operation_rule", "collection_rule", "safety_rule"},
        )


if __name__ == "__main__":
    unittest.main()
