from __future__ import annotations

import unittest

from smart_traffic_agent.parameter_rules import (
    ParameterRule,
    filter_parameter_candidates,
    generate_parameter_combinations,
)


class ParameterRuleTests(unittest.TestCase):
    def test_upload_and_download_filtering_remain_distinct(self) -> None:
        rule = ParameterRule(
            "programs", "cnc_upstart3", "program_number",
            allowed_values=(100, 101, 102), excluded_values=(101,)
        )
        upload = filter_parameter_candidates(
            "cnc_upstart3", "program_number",
            {"candidates": [100, 101, 103], "registered_values": [100, 101]},
            [rule],
        )
        download = filter_parameter_candidates(
            "cnc_dwnstart3", "program_number",
            {"candidates": [100, 101, 102], "registered_values": [100], "protected_values": [102], "operation": "download_new"},
            [ParameterRule("programs", "cnc_dwnstart3", "program_number")],
        )
        self.assertEqual(upload.candidates, [100])
        self.assertEqual(download.candidates, [101])
        self.assertTrue(any(item["reason"] == "already_registered" for item in download.excluded))

    def test_combinations_are_bounded_and_unique(self) -> None:
        result = generate_parameter_combinations(
            "cnc_rdmacro", {"number": [100, 101], "length": [1, 2]}, budget=3
        )
        self.assertEqual(result.actual_combination_count, 4)
        self.assertEqual(len(result.combinations), 3)
        self.assertEqual(len({tuple(sorted(item.items())) for item in result.combinations}), 3)


if __name__ == "__main__":
    unittest.main()
