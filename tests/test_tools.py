from __future__ import annotations

import unittest

from smart_traffic_agent.agents.executor import SUPPORTED_READONLY_INTERFACES, semantic_label
from smart_traffic_agent.agents.planner import INTERFACE_TO_FOCAS
from smart_traffic_agent.tools import (
    TOOL_REGISTRY,
    focas_function_for,
    semantic_label_for,
    supported_ncguide_readonly_interfaces,
)


class ToolRegistryTests(unittest.TestCase):
    def test_registry_contains_core_readonly_focas_tools(self) -> None:
        for name in [
            "ReadRunStatus",
            "ReadPosition",
            "ReadDistanceToGo",
            "ReadFeedSpeed",
            "ReadSpindleSpeed",
            "ReadAlarm",
            "ReadProgramNumber",
        ]:
            self.assertIn(name, TOOL_REGISTRY)
            self.assertTrue(TOOL_REGISTRY[name].supported_by_cpp_codegen)
            self.assertTrue(TOOL_REGISTRY[name].supported_by_ncguide_readonly)

    def test_registry_contains_verified_program_lifecycle_codegen_tools(self) -> None:
        self.assertTrue(TOOL_REGISTRY["UploadProgram"].supported_by_cpp_codegen)
        self.assertEqual(TOOL_REGISTRY["UploadProgram"].focas_function, "cnc_dwnstart3/cnc_download3/cnc_dwnend3")
        self.assertEqual(TOOL_REGISTRY["ReadProgramDirectory"].focas_function, "cnc_rdprogdir3")
        self.assertEqual(TOOL_REGISTRY["DeleteProgram"].focas_function, "cnc_delete")
        self.assertTrue(TOOL_REGISTRY["ReadProgramDirectory"].supported_by_cpp_codegen)
        self.assertTrue(TOOL_REGISTRY["DeleteProgram"].supported_by_cpp_codegen)
        self.assertTrue(TOOL_REGISTRY["SelectProgram"].supported_by_cpp_codegen)
        self.assertEqual(TOOL_REGISTRY["SelectProgram"].focas_function, "cnc_search")
        self.assertTrue(TOOL_REGISTRY["StartProgram"].supported_by_cpp_codegen)
        self.assertEqual(TOOL_REGISTRY["StartProgram"].focas_function, "ncguide_ui_cycle_start")
        self.assertFalse(TOOL_REGISTRY["StopProgram"].supported_by_cpp_codegen)

    def test_agents_use_registry_as_metadata_not_codegen_gate(self) -> None:
        self.assertEqual(SUPPORTED_READONLY_INTERFACES, supported_ncguide_readonly_interfaces())
        self.assertEqual(INTERFACE_TO_FOCAS["ReadPosition"], focas_function_for("ReadPosition"))
        self.assertEqual(INTERFACE_TO_FOCAS["ReadDistanceToGo"], "cnc_distance")
        self.assertEqual(semantic_label("ReadPosition"), semantic_label_for("ReadPosition"))


if __name__ == "__main__":
    unittest.main()
