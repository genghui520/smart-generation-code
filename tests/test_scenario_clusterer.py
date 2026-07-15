from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.rag.scenario_clusterer import (
    build_scenario_21_clusters,
    cluster_vectors,
    rule_chunk_to_unit,
    scenario_21_cluster_payload,
    vectorize_units,
)
from smart_traffic_agent.rag.scenario_taxonomy import default_taxonomy


class ScenarioClustererTests(unittest.TestCase):
    def test_default_taxonomy_auto_selects_cluster_count(self) -> None:
        payload = scenario_21_cluster_payload(default_taxonomy(), min_clusters=3, max_clusters=8)

        self.assertGreaterEqual(payload["statistics"]["scenario_cluster_count"], 3)
        self.assertLessEqual(payload["statistics"]["scenario_cluster_count"], 8)
        self.assertEqual(payload["statistics"]["unique_cluster_count"], payload["statistics"]["scenario_cluster_count"])
        self.assertEqual(len(payload["clusters"]), payload["statistics"]["scenario_cluster_count"])
        self.assertEqual(payload["method"], "feature_vector_kmeans_clustering")
        self.assertEqual(payload["cluster_count_selection"]["mode"], "auto")

    def test_writes_cluster_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scenario_clusters.json"

            payload = build_scenario_21_clusters(out, taxonomy=default_taxonomy(), min_clusters=3, max_clusters=8)

            self.assertTrue(out.exists())
            self.assertGreaterEqual(payload["statistics"]["scenario_cluster_count"], 3)

    def test_similar_units_fall_into_same_natural_cluster(self) -> None:
        units = [
            rule_chunk_to_unit(
                {
                    "rule_id": "sample-spindle-speed-1",
                    "rule_type": "nc_rule",
                    "rule_text": "Use M03 with multiple S value commands, then M05; collect cnc_acts.",
                    "recommended_api_functions": ["cnc_acts"],
                    "distinguishing_signals": ["spindle_speed_change"],
                }
            ),
            rule_chunk_to_unit(
                {
                    "rule_id": "sample-spindle-speed-2",
                    "rule_type": "collection_rule",
                    "rule_text": "Collect spindle speed change with cnc_acts while S value changes.",
                    "recommended_api_functions": ["cnc_acts"],
                    "distinguishing_signals": ["spindle_speed_change"],
                }
            ),
            rule_chunk_to_unit(
                {
                    "rule_id": "sample-alarm",
                    "rule_type": "collection_rule",
                    "rule_text": "Query current alarm state and alarm message with cnc_alarm2.",
                    "recommended_api_functions": ["cnc_alarm2"],
                    "distinguishing_signals": ["alarm_bits_change"],
                }
            ),
        ]

        labels, _ = cluster_vectors(vectorize_units(units), cluster_count=2)

        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])


if __name__ == "__main__":
    unittest.main()
