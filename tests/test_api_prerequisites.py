from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.api_prerequisites import (
    build_prerequisite_plan,
    classify_prerequisite_failure,
    load_manifest_rows,
    manifest_functions_by_category,
)


class ApiPrerequisiteTests(unittest.TestCase):
    def test_manifest_category_drives_tool_life_scope(self) -> None:
        rows = load_manifest_rows(Path("rag_indexes/focas/function_manifest.jsonl"))
        functions = manifest_functions_by_category(rows, "tool life management data")
        self.assertIn("cnc_rdgrpid", functions)
        self.assertIn("cnc_rdgrpinfo4", functions)
        self.assertNotIn("cnc_statinfo", functions)

    def test_rule_builds_data_driven_read_write_verify_plan(self) -> None:
        rule = {
            "rule_id": "tool-life",
            "applies_to": ["cnc_rdgrpid"],
            "conditions": [{"type": "cnc_parameter", "parameter_number": 8132, "bit": 0, "expected_value": 1}],
            "preparation_operations": [{"operation": "read_parameter", "api": "cnc_rdparam"}],
            "verification_operations": [{"api": "cnc_rdparam", "check": "bit=1"}],
            "restart_required": True,
            "safe_to_write_in_isolated_simulator": True,
            "source_url": "doc",
        }
        plan = build_prerequisite_plan(["cnc_rdgrpid"], [rule], environment="ncguide-generated-cpp", permissions={})
        self.assertEqual(plan["matched_functions"], ["cnc_rdgrpid"])
        self.assertTrue(plan["preparation_operations"][0]["blocked_without_permission"])
        self.assertTrue(plan["preparation_operations"][0]["restart_required"])

    def test_ew_noopt_is_classified_without_assuming_parameter_disabled(self) -> None:
        failure = classify_prerequisite_failure("cnc_rdgrpid", 6, "EW_NOOPT")
        self.assertEqual(failure["kind"], "option_or_controller_capability")
        self.assertIn("software option", failure["requires_evidence"])


if __name__ == "__main__":
    unittest.main()
