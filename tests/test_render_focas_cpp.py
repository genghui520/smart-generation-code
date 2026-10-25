from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.agent_tools.base import invoke_tool
from smart_traffic_agent.agent_tools.render_focas_cpp import (
    RenderFocasCppInput,
    RenderFocasCppTool,
)


HARNESS = '#include "scenario_fragment.inc"\nint main(){return 0;}\n'
FRAGMENT = 'short run_generated_scenario(FocasApi& api, unsigned short handle){ return EW_OK; }\n'


class RenderFocasCppToolTests(unittest.TestCase):
    def test_normal_assembly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            fragment = root / "scenario_fragment.inc"
            output = root / "api_script.cpp"
            harness.write_text(HARNESS, encoding="utf-8")
            fragment.write_text(FRAGMENT, encoding="utf-8")

            result = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, fragment, output))

            self.assertTrue(result.success)
            self.assertEqual(output.read_text(encoding="utf-8"), HARNESS)

    def test_harness_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fragment = root / "scenario_fragment.inc"
            fragment.write_text(FRAGMENT, encoding="utf-8")
            result = RenderFocasCppTool().invoke(RenderFocasCppInput(root / "missing.cpp", fragment, root / "out.cpp"))
            self.assertFalse(result.success)
            self.assertIn("Harness file does not exist", result.error or "")

    def test_fragment_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            harness.write_text(HARNESS, encoding="utf-8")
            result = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, root / "missing.inc", root / "out.cpp"))
            self.assertFalse(result.success)
            self.assertIn("Scenario fragment file does not exist", result.error or "")

    def test_fragment_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            fragment = root / "scenario_fragment.inc"
            harness.write_text(HARNESS, encoding="utf-8")
            fragment.write_text("\n", encoding="utf-8")
            result = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, fragment, root / "out.cpp"))
            self.assertFalse(result.success)
            self.assertIn("Scenario fragment is empty", result.error or "")

    def test_harness_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            fragment = root / "scenario_fragment.inc"
            output = root / "api_script.cpp"
            harness.write_text(HARNESS, encoding="utf-8")
            fragment.write_text(FRAGMENT, encoding="utf-8")
            before = hashlib.sha256(harness.read_bytes()).hexdigest()

            result = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, fragment, output))

            self.assertTrue(result.success)
            self.assertEqual(before, hashlib.sha256(harness.read_bytes()).hexdigest())
            self.assertEqual(result.harness_sha256, before)

    def test_render_does_not_read_old_api_script(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            fragment = root / "scenario_fragment.inc"
            output = root / "api_script.cpp"
            harness.write_text(HARNESS, encoding="utf-8")
            fragment.write_text(FRAGMENT, encoding="utf-8")
            output.write_text("OLD_INVALID_CODE\n", encoding="utf-8")

            result = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, fragment, output))

            self.assertTrue(result.success)
            self.assertNotIn("OLD_INVALID_CODE", output.read_text(encoding="utf-8"))
            self.assertEqual(output.read_text(encoding="utf-8"), HARNESS)

    def test_invoke_tool_records_render_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            harness = root / "focas_harness.cpp"
            fragment = root / "scenario_fragment.inc"
            harness.write_text(HARNESS, encoding="utf-8")
            fragment.write_text(FRAGMENT, encoding="utf-8")

            output, record = invoke_tool(
                RenderFocasCppTool(),
                RenderFocasCppInput(harness, fragment, root / "api_script.cpp"),
                agent_name="CodeEngineerRole",
            )

            self.assertEqual(record.tool_name, "render_focas_cpp")
            self.assertTrue(record.success)
            self.assertGreaterEqual(record.duration_ms, 0)
            self.assertEqual(record.input_summary["harness_path"], str(harness))
            self.assertTrue(output.success)
            self.assertEqual(record.output_summary["success"], True)


if __name__ == "__main__":
    unittest.main()
