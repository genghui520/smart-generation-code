from __future__ import annotations

import json
from pathlib import Path

import pyLSV2

from smart_traffic_agent import desktop_backend
from smart_traffic_agent.heidenhain_catalog import HEIDENHAIN_READ_ONLY_API_NAMES
from smart_traffic_agent.heidenhain_catalog import heidenhain_tnc640_catalog
from smart_traffic_agent.integrations.heidenhain_lsv2 import HeidenhainLsv2Client
from smart_traffic_agent.integrations.heidenhain_lsv2 import HeidenhainLsv2Error


class _FakeLsv2:
    def __init__(self, *_args, **_kwargs):
        self.last_error = ""
        self.versions = {"control": "TNC640"}
        self.parameters = {"lsv2_version": 5}

    def connect(self):
        return None

    def disconnect(self):
        return None

    def program_status(self): return "idle"
    def program_stack(self): return {"stack": []}
    def directory_info(self, path): return {"path": path}
    def directory_content(self): return []
    def drive_info(self): return []
    def file_info(self, path): return {"path": path}
    def get_file_list(self, *_args, **_kwargs): return []
    def read_plc_memory(self, *_args, **_kwargs): return [False]
    def read_plc_address(self, *_args, **_kwargs): return False
    def read_data_path(self, *_args, **_kwargs): return 0
    def axes_location(self): return {"X": 0.0}
    def override_state(self): return {"feed": 100}
    def spindle_tool_status(self): return {"tool": 1}
    def get_remote_datetime(self): return "2026-11-04T00:00:00"
    def get_error_messages(self): return []
    def read_scope_signals(self): return [{"name": "X"}]
    def grab_screen_dump(self, _path): return True
    def get_machine_parameter(self, *_args, **_kwargs): return "1"


def test_collects_all_tnc640_read_only_operations(monkeypatch, tmp_path):
    monkeypatch.setattr(pyLSV2, "LSV2", _FakeLsv2)
    result = HeidenhainLsv2Client("127.0.0.1").collect_read_only(
        sorted(HEIDENHAIN_READ_ONLY_API_NAMES), output_dir=tmp_path, samples=1,
    )
    assert set(result["successful_operations"]) == set(HEIDENHAIN_READ_ONLY_API_NAMES)
    assert len(result["records"]) == len(HEIDENHAIN_READ_ONLY_API_NAMES)
    assert all(row["protocol"] == "LSV2" and row["status_code"] == 0 for row in result["records"])


def test_catalog_exposes_official_lsv2_library_symbols():
    rows = {row["function"]: row for row in heidenhain_tnc640_catalog()}
    assert rows["versions"]["library_function"] == "LSV2ReceiveVersions"
    assert rows["directory_info"]["library_function"] == "LSV2ReceiveDirInfo"
    assert rows["directory_info"]["library_category"] == "File system"


def test_write_or_control_operations_are_rejected_before_connect(monkeypatch):
    class _ShouldNotConnect(_FakeLsv2):
        def connect(self):
            raise AssertionError("write/control selection must be rejected before connecting")

    monkeypatch.setattr(pyLSV2, "LSV2", _ShouldNotConnect)
    try:
        HeidenhainLsv2Client("127.0.0.1").collect_read_only(["set_machine_parameter"])
    except HeidenhainLsv2Error as exc:
        assert "只允许只读 API" in str(exc)
    else:
        raise AssertionError("expected write/control API rejection")


def test_lsv2_capture_writes_structured_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(pyLSV2, "LSV2", _FakeLsv2)
    config = dict(
        protocol="lsv2", host="127.0.0.1", port="19000", timeout="5",
        samples="1", interval="0", output=str(tmp_path), capture_mode="single_api",
        selected_apis=["directory_info"],
    )
    events = []
    assert desktop_backend.run_lsv2_capture(
        config, tmp_path / "run", lambda name, data: events.append((name, data)),
    ) == 0
    log_path = tmp_path / "run" / "batch_001" / "execution" / "api_logs.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["protocol_function"] == "directory_info"
    assert records[0]["input_parameters"]["remote_directory"] == "TNC:"
    summary = json.loads((tmp_path / "run" / "batch_001" / "summary.json").read_text(encoding="utf-8"))
    assert summary["collection_status"] == "local_success"
    assert any(name == "success" for name, _data in events)


def test_lsv2_validation_skips_focas_and_llm_requirements(tmp_path):
    config = dict(
        desktop_backend.defaults(), protocol="lsv2", host="127.0.0.1", port="19000",
        output=str(tmp_path), capture_mode="single_api", selected_apis=["directory_info"],
        base_url="", model="", api_key="",
    )
    assert desktop_backend.validate(config, capture=True) == []
