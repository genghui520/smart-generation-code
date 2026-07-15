from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent.integrations.ncguide import (
    DEFAULT_FOCAS_HEADER_DIR,
    DEFAULT_FOCAS_PORTS,
    DEFAULT_NCGUIDE_DIR,
    FocasBridgeClient,
    FocasCppBridgeClient,
    NCGUIDE_FS0I_D_DIR,
    default_focas_header_dir,
    default_focas_runtime_dir,
    inspect_dll,
    pe_machine_bits,
    probe_ncguide,
)
from smart_traffic_agent.integrations.focas_bridge import BridgeConfig, run_once
from smart_traffic_agent.agents.executor import executable_steps, make_execution_client
from smart_traffic_agent.models import PlanStep


class NcGuideProbeTests(unittest.TestCase):
    def test_default_ncguide_profile_is_fs0i_d(self) -> None:
        self.assertEqual(DEFAULT_NCGUIDE_DIR, NCGUIDE_FS0I_D_DIR)
        self.assertEqual(DEFAULT_FOCAS_PORTS[0], 8193)

    def test_pe_machine_bits(self) -> None:
        self.assertEqual(pe_machine_bits("x86"), 32)
        self.assertEqual(pe_machine_bits("x64"), 64)
        self.assertIsNone(pe_machine_bits("unknown"))

    def test_inspect_missing_dll(self) -> None:
        info = inspect_dll(Path("missing.dll"))

        self.assertFalse(info["exists"])
        self.assertIsNone(info["bits"])

    def test_probe_missing_install_dir_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_ncguide"
            result = probe_ncguide(
                install_dir=missing,
                host="127.0.0.1",
                ports=[1],
                timeout_seconds=0.01,
            )

            self.assertFalse(result.install_dir_exists)
            self.assertIn(1, result.ports)
            self.assertFalse(result.can_load_focas_dll_in_current_python)
            self.assertTrue(result.notes)

    def test_bridge_client_reports_missing_python(self) -> None:
        client = FocasBridgeClient(python_exe=Path("Z:/missing/python.exe"))

        result = client.run_bridge("read_run_status")

        self.assertEqual(result["status_code"], 400)
        self.assertIn("not found", result["error"])

    def test_cpp_bridge_client_reports_missing_exe(self) -> None:
        client = FocasCppBridgeClient(bridge_exe=Path("Z:/missing/focas_bridge.exe"))

        result = client.run_bridge("read_run_status")

        self.assertEqual(result["status_code"], 400)
        self.assertIn("not found", result["error"])

    def test_bridge_requires_32_bit_python(self) -> None:
        result = run_once("probe", BridgeConfig(install_dir=Path("missing")))

        if result["status_code"] != 0:
            self.assertIn("32-bit Python", result["error"])

    def test_readonly_bridge_keeps_only_supported_steps(self) -> None:
        skipped: list[str] = []
        steps = [
            PlanStep("S001", "before", "upload", "UploadProgram", {}),
            PlanStep("S002", "before", "read status", "ReadRunStatus", {}),
            PlanStep("S003", "during", "read position", "ReadPosition", {}),
            PlanStep("S004", "during", "read feed", "ReadFeedSpeed", {}),
            PlanStep("S005", "during", "read spindle", "ReadSpindleSpeed", {}),
            PlanStep("S006", "after", "read alarm", "ReadAlarm", {}),
        ]

        selected = executable_steps(steps, "ncguide-bridge-readonly", skipped)

        self.assertEqual(
            [step.interface_name for step in selected],
            ["ReadRunStatus", "ReadPosition", "ReadFeedSpeed", "ReadSpindleSpeed", "ReadAlarm"],
        )
        self.assertEqual(len(skipped), 1)

    def test_default_focas_runtime_dir_falls_back_to_ncguide_install(self) -> None:
        with patch("smart_traffic_agent.integrations.ncguide.PROJECT_ROOT", Path("Z:/missing/project")), patch(
            "smart_traffic_agent.integrations.ncguide.Path.cwd", return_value=Path("Z:/missing/cwd")
        ), patch("smart_traffic_agent.integrations.ncguide.DEFAULT_FOCAS_RUNTIME_DIR", Path("Z:/missing/focas")):
            self.assertEqual(default_focas_runtime_dir(), DEFAULT_NCGUIDE_DIR)

    def test_default_focas_header_dir_uses_0id_sdk(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_focas_header_dir(), DEFAULT_FOCAS_HEADER_DIR)
            self.assertEqual(default_focas_header_dir().name, "0iD")

    def test_focas_header_dir_can_be_overridden(self) -> None:
        with patch.dict("os.environ", {"FOCAS_HEADER_DIR": "Z:/custom/0iD"}):
            self.assertEqual(default_focas_header_dir(), Path("Z:/custom/0iD"))

    def test_execution_client_uses_focas_dll_dir_override(self) -> None:
        with patch.dict("os.environ", {"FOCAS_DLL_DIR": "Z:/custom/focas", "FOCAS_CPP_BRIDGE_EXE": "Z:/missing.exe"}):
            client = make_execution_client("ncguide-bridge-readonly")

        self.assertIsInstance(client, FocasBridgeClient)
        self.assertEqual(client.install_dir, Path("Z:/custom/focas"))


if __name__ == "__main__":
    unittest.main()
