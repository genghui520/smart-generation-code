from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smart_traffic_agent import desktop_backend as backend
from smart_traffic_agent.desktop_backend import capture_unavailable_after_batch
from smart_traffic_agent.models import WorkflowState


class BatchFailureClassificationTests(unittest.TestCase):
    def test_unmatched_target_does_not_invalidate_real_pcap(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            folder = Path(temp)
            (folder / "raw.pcap").write_bytes(b"X" * 64)
            self.assertFalse(capture_unavailable_after_batch(
                ["目标 API cnc_rdtofs 没有匹配到完整的 PCAP 请求/响应"], folder
            ))

    def test_missing_or_explicitly_failed_capture_still_stops(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            folder = Path(temp)
            self.assertTrue(capture_unavailable_after_batch(["target failed"], folder))
            (folder / "raw.pcap").write_bytes(b"X" * 64)
            self.assertTrue(capture_unavailable_after_batch(["PCAP_CAPTURE_FAILED"], folder))

    def test_failure_before_plan_defers_current_api_and_reaches_next(self) -> None:
        attempted = []

        def failed_run(request, folder):
            attempted.extend(request.permissions["coverage_current_batch_functions"])
            folder.mkdir(parents=True)
            return WorkflowState(request=request, errors=["generation failed before plan"])

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            folder = Path(temp)
            header = folder / "sdk"
            header.mkdir()
            (header / "Fwlib32.h").write_text("short cnc_statinfo(void); short cnc_actf(void);", encoding="ascii")
            config = backend.defaults()
            config.update(selected=[], selected_apis=["cnc_statinfo", "cnc_actf"], header=str(header))
            with patch("smart_traffic_agent.workflow.TrafficGenerationWorkflow") as workflow, \
                    patch.object(backend, "make_llm", return_value=None), \
                    patch.object(backend, "simulator_is_idle_after_failure", side_effect=AssertionError("pre-execution failure must not probe controller")):
                workflow.return_value.run.side_effect = failed_run
                with self.assertRaisesRegex(RuntimeError, "未判定 API"):
                    backend.run_full_api_capture(config, folder, None, lambda *_: None)
            self.assertEqual(set(attempted), {"cnc_statinfo", "cnc_actf"})
            self.assertEqual(len(attempted), 2)


if __name__ == "__main__":
    unittest.main()
