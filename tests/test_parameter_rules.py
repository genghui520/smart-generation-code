from __future__ import annotations

import unittest

from smart_traffic_agent.parameter_rules import (
    ParameterRule,
    filter_parameter_candidates,
    generate_parameter_combinations,
    validate_nc_program_number,
)
from smart_traffic_agent.focas_contracts import validate_cnc_settimer_value


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

    def test_download_conflict_filter_excludes_registered_and_protected_numbers(self) -> None:
        result = filter_parameter_candidates(
            "cnc_download3", "program_number",
            {
                "candidates": [100, 101, 102, 103],
                "registered_values": [100, 101],
                "protected_values": [102],
                "operation": "download_new",
            },
            [ParameterRule("download-program", "cnc_download3", "program_number")],
        )
        self.assertEqual(result.candidates, [103])
        self.assertEqual(
            {item["reason"] for item in result.excluded},
            {"already_registered", "protected_program"},
        )

    def test_download_payload_number_must_match_selected_number(self) -> None:
        self.assertTrue(validate_nc_program_number("\nO103\nM30\n%", 103)["valid"])
        mismatch = validate_nc_program_number("\nO104\nM30\n%", 103)
        self.assertFalse(mismatch["valid"])
        self.assertEqual(mismatch["payload_program_numbers"], [104])

    def test_cnc_settimer_uses_api_contract_not_graph_filter(self) -> None:
        valid_date = validate_cnc_settimer_value(
            {"type": 0, "data.date.year": 2026, "data.date.month": 10, "data.date.date": 30}
        )
        self.assertTrue(valid_date["valid"])
        self.assertFalse(validate_cnc_settimer_value(
            {"type": 0, "data.date.year": 1996, "data.date.month": 12, "data.date.date": 31}
        )["valid"])
        self.assertFalse(validate_cnc_settimer_value(
            {"type": 0, "data.date.year": 2026, "data.date.month": 2, "data.date.date": 29}
        )["valid"])
        self.assertTrue(validate_cnc_settimer_value(
            {"type": 1, "data.time.hour": 23, "data.time.minute": 59, "data.time.second": 59}
        )["valid"])
        self.assertFalse(validate_cnc_settimer_value(
            {"type": 1, "data.time.hour": 24, "data.time.minute": 0, "data.time.second": 0}
        )["valid"])

    def test_cnc_settimer_accepts_nested_buf_shape(self) -> None:
        result = validate_cnc_settimer_value({
            "buf": {"type": 0, "data": {"date": {"year": 2028, "month": 2, "date": 29}}}
        })
        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
