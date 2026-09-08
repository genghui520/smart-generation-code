from __future__ import annotations

import unittest

from smart_traffic_agent.models import PlanStep
from smart_traffic_agent.tool_protocol import tool_calls_from_step, validate_tool_calls


class ToolProtocolTests(unittest.TestCase):
    def test_sequence_is_preserved_as_atomic_calls(self) -> None:
        step = PlanStep(
            "UPLOAD-001",
            "before",
            "upload program",
            "UploadProgram",
            protocol_function="cnc_dwnstart3",
            api_calls=[
                {"call_id": "a", "protocol_function": "cnc_dwnstart3"},
                {"call_id": "b", "protocol_function": "cnc_download3"},
                {"call_id": "c", "protocol_function": "cnc_dwnend3"},
            ],
        )

        calls = tool_calls_from_step(step)

        self.assertEqual([call.tool_name for call in calls], [
            "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"
        ])
        self.assertEqual(validate_tool_calls(calls), [])

    def test_composite_tool_name_is_normalized_to_atomic_calls(self) -> None:
        step = PlanStep("BAD", "during", "bad call", "Unknown", protocol_function="cnc_a+cnc_b")

        calls = tool_calls_from_step(step)

        self.assertEqual([call.tool_name for call in calls], ["cnc_a", "cnc_b"])
        self.assertEqual(validate_tool_calls(calls), [])

    def test_atomic_host_tool_is_accepted_without_registry_lookup(self) -> None:
        step = PlanStep("UI-001", "during", "start cycle", "CycleStart", protocol_function="cycle_start")

        self.assertEqual(validate_tool_calls(tool_calls_from_step(step)), [])

    def test_host_sequence_is_normalized_to_ordered_atomic_tools(self) -> None:
        step = PlanStep(
            "SEQ-001", "after", "record payload facts", "HostSequence",
            protocol_function="sequence: derive_payload -> log_segments -> log_clicks",
        )

        calls = tool_calls_from_step(step)

        self.assertEqual([call.tool_name for call in calls], [
            "derive_payload", "log_segments", "log_clicks"
        ])
        self.assertEqual(validate_tool_calls(calls), [])

    def test_legacy_composite_api_entry_is_normalized(self) -> None:
        step = PlanStep("LEGACY-001", "during", "read state", "ReadState", protocol_function="cnc_statinfo + cnc_absolute")

        calls = tool_calls_from_step(step)

        self.assertEqual([call.tool_name for call in calls], ["cnc_statinfo", "cnc_absolute"])
        self.assertEqual(validate_tool_calls(calls), [])


if __name__ == "__main__":
    unittest.main()
