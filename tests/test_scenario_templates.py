from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.rag.scenario_templates import build_final_scenario_templates


class ScenarioTemplateTests(unittest.TestCase):
    def test_builds_templates_from_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            review = tmp_path / "review.csv"
            clusters = tmp_path / "clusters.json"
            out = tmp_path / "templates.json"
            with review.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "cluster_id",
                        "member_count",
                        "dominant_trigger",
                        "dominant_objects",
                        "dominant_apis",
                        "dominant_nc_or_operation",
                        "dominant_signals",
                        "dominant_rule_types",
                        "suggested_scene_name",
                        "suggested_action",
                        "review_note",
                        "representative_units",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "cluster_id": "cluster_01_axis_motion",
                        "member_count": "12",
                        "dominant_trigger": "motion_trigger",
                        "dominant_objects": "axis; feed",
                        "dominant_apis": "cnc_rdposition; cnc_actf",
                        "dominant_nc_or_operation": "G01; G00",
                        "dominant_signals": "axis_change; feed_change",
                        "dominant_rule_types": "nc_rule; collection_rule",
                        "suggested_scene_name": "coordinate_feed_motion",
                        "suggested_action": "keep_or_merge_by_semantics",
                        "review_note": "",
                        "representative_units": "sample",
                    }
                )
            clusters.write_text(json.dumps({"clusters": []}), encoding="utf-8")

            payload = build_final_scenario_templates(review_csv_path=review, clusters_path=clusters, output_path=out)

            self.assertTrue(out.exists())
            self.assertEqual(payload["template_count"], 1)
            template = payload["templates"][0]
            self.assertEqual(template["scenario_name"], "coordinate_feed_motion")
            self.assertTrue(template["operation_template"])
            self.assertIn("collection_template", template)
            self.assertIn("safety_template", template)


if __name__ == "__main__":
    unittest.main()
