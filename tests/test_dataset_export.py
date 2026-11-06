from __future__ import annotations

import csv
import json
import struct
import tempfile
from pathlib import Path

from smart_traffic_agent.dataset_export import export_legacy_dataset


def _tcp_frame(payload: bytes, sequence: int, *, response: bool = False, port: int = 8193) -> bytes:
    ethernet = b"\x00" * 12 + b"\x08\x00"
    ip = bytearray(20)
    ip[0] = 0x45
    ip[2:4] = struct.pack("!H", 40 + len(payload))
    ip[9] = 6
    ip[12:16] = b"\x7f\x00\x00\x01"
    ip[16:20] = b"\x7f\x00\x00\x01"
    tcp = bytearray(20)
    source, destination = ((port, 50000) if response else (50000, port))
    tcp[0:2] = struct.pack("!H", source)
    tcp[2:4] = struct.pack("!H", destination)
    tcp[4:8] = struct.pack("!I", sequence)
    tcp[12] = 0x50
    return ethernet + bytes(ip) + bytes(tcp) + payload


def _pcap(path: Path) -> None:
    global_header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    records = []
    for timestamp_ns, frame in (
        (1_100_000_000, _tcp_frame(b"request-frame", 10)),
        (1_200_000_000, _tcp_frame(b"response-frame", 20, response=True)),
    ):
        seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
        micros = nanos // 1_000
        records.append(struct.pack("<IIII", seconds, micros, len(frame), len(frame)) + frame)
    path.write_bytes(global_header + b"".join(records))


def _invocation(identifier: str, transaction: str, *, api: str = "cnc_absolute") -> dict:
    return {
        "invocation_id": identifier,
        "transaction_id": transaction,
        "api": api,
        "input_parameters": "axis=1;length=8",
        "api_parameter_count": 2,
        "output_data": "position=13440;decimal=3",
        "return_code": 0,
        "phase": "During",
        "log_timestamp": "2026-09-11T12:00:00.100000",
        "start_timestamp_ns": 1_000_000_000,
        "end_timestamp_ns": 1_300_000_000,
        "device_context": {"target": "NCGuide"},
    }


def _transaction(invocation: str, identifier: str) -> dict:
    return {
        "invocation_id": invocation,
        "transaction_id": identifier,
        "request_response_associated": True,
        "request_response_order_valid": True,
        "tcp_reassembled": True,
        "request_payload_hex": b"REQ".hex(),
        "response_payload_hex": b"RES".hex(),
    }


def test_export_writes_legacy_layout_and_auditable_mapping() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        pcap = execution / "capture.pcap"
        _pcap(pcap)

        root = export_legacy_dataset(
            execution,
            [_invocation("inv-1", "tx-1")],
            [_transaction("inv-1", "tx-1")],
            pcap,
        )

        stem = "20260911_focas_absolute"
        assert (root / f"{stem}_input.csv").exists()
        assert (root / f"{stem}_output.csv").exists()
        assert (root / f"{stem}.pcap").stat().st_size > 24
        document = json.loads((root / "dataset" / f"{stem}.txt").read_text(encoding="utf-8"))
        assert document["csv_data"][0]["axis"] == "1"
        assert document["packet_data"][0][0][0] == "Request:"
        assert document["mapping_data"][0]["invocation_id"] == "inv-1"
        assert document["mapping_data"][0]["transaction_id"] == "tx-1"
        training = json.loads((root / "training_dataset.jsonl").read_text(encoding="utf-8"))
        assert training["request_hex"] == b"REQ".hex()
        assert training["response_hex"] == b"RES".hex()
        assert training["label_str"] == "absolute|"

        with (root / f"{stem}_input.csv").open(encoding="utf-8-sig", newline="") as source:
            input_row = next(csv.DictReader(source))
        with (root / f"{stem}_output.csv").open(encoding="utf-8-sig", newline="") as source:
            output_row = next(csv.DictReader(source))
        assert input_row["api_name"] == "cnc_absolute"
        assert input_row["parameters"] == "axis=1;length=8"
        assert "api_parameter_count" not in input_row
        assert output_row["data"] == "position=13440;decimal=3"
        assert input_row["index"] == output_row["index"] == "1"
        assert input_row["invocation_id"] == output_row["invocation_id"] == "inv-1"
        assert input_row["transaction_id"] == output_row["transaction_id"] == "tx-1"


def test_multi_connection_export_keeps_each_stream_separate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        transaction = _transaction("inv-1", "tx-1")
        transaction.update(request_payload_hex=None, response_payload_hex=None, tcp_streams=[
            {"stream_id": "socket-1", "request_payload_hex": b"Q1".hex(), "response_payload_hex": b"A1".hex()},
            {"stream_id": "socket-2", "request_payload_hex": b"Q2".hex(), "response_payload_hex": b"A2".hex()},
        ])
        root = export_legacy_dataset(Path(temporary), [_invocation("inv-1", "tx-1")], [transaction], None)
        training = json.loads((root / "training_dataset.jsonl").read_text(encoding="utf-8"))
        assert training["request_hex"] is None
        assert len(training["tcp_streams"]) == 2
        document = json.loads((root / "dataset/20260911_focas_absolute.txt").read_text(encoding="utf-8"))
        assert len(document["packet_data"][0]) == 4
        assert document["packet_data"][0][0][0] == "Request [socket-1]:"


def test_unmatched_invocation_has_paired_logs_but_no_fake_pcap() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        root = export_legacy_dataset(
            execution,
            [_invocation("inv-missing", "tx-candidate", api="cnc_rdtool")],
            [{
                "invocation_id": "inv-missing",
                "transaction_id": None,
                "request_response_associated": False,
                "request_response_order_valid": False,
                "tcp_reassembled": True,
                "request_payload_hex": None,
                "response_payload_hex": None,
            }],
            None,
        )

        stem = "20260911_focas_rdtool"
        assert len((root / f"{stem}_input.csv").read_text(encoding="utf-8-sig").splitlines()) == 2
        assert len((root / f"{stem}_output.csv").read_text(encoding="utf-8-sig").splitlines()) == 2
        assert not (root / f"{stem}.pcap").exists()
        with (root / f"{stem}_input.csv").open(encoding="utf-8-sig", newline="") as source:
            input_row = next(csv.DictReader(source))
        assert input_row["traffic_status"] == "no_corresponding_traffic"
        assert input_row["transaction_id"] == ""
        assert (root / "training_dataset.jsonl").read_text(encoding="utf-8") == ""
        document = json.loads((root / "dataset" / f"{stem}.txt").read_text(encoding="utf-8"))
        assert document == {"csv_data": [], "packet_data": [], "mapping_data": []}
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["raw_invocation_count"] == 1
        assert manifest["valid_transaction_count"] == 0


def test_same_api_calls_share_files_and_other_api_stays_separate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        pcap = execution / "capture.pcap"
        header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
        records = []
        for timestamp_ns, frame in (
            (1_100_000_000, _tcp_frame(b"actf-1-request", 10)),
            (1_150_000_000, _tcp_frame(b"unrelated-traffic", 11, port=1234)),
            (1_200_000_000, _tcp_frame(b"actf-1-response", 20, response=True)),
            (2_100_000_000, _tcp_frame(b"stat-request", 30)),
            (2_200_000_000, _tcp_frame(b"stat-response", 40, response=True)),
            (3_100_000_000, _tcp_frame(b"actf-2-request", 50)),
            (3_200_000_000, _tcp_frame(b"actf-2-response", 60, response=True)),
        ):
            seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
            records.append(struct.pack("<IIII", seconds, nanos // 1_000, len(frame), len(frame)) + frame)
        pcap.write_bytes(header + b"".join(records))
        calls = []
        transactions = []
        for identifier, api, second in (
            ("actf-1", "cnc_actf", 1),
            ("stat-1", "cnc_statinfo", 2),
            ("actf-2", "cnc_actf", 3),
        ):
            call = _invocation(identifier, f"tx-{identifier}", api=api)
            call["start_timestamp_ns"] = second * 1_000_000_000
            call["end_timestamp_ns"] = second * 1_000_000_000 + 300_000_000
            calls.append(call)
            transactions.append(_transaction(identifier, f"tx-{identifier}"))

        root = export_legacy_dataset(execution, calls, transactions, pcap)
        actf = "20260911_focas_actf"
        stat = "20260911_focas_statinfo"
        with (root / f"{actf}_input.csv").open(encoding="utf-8-sig", newline="") as source:
            actf_inputs = list(csv.DictReader(source))
        with (root / f"{actf}_output.csv").open(encoding="utf-8-sig", newline="") as source:
            actf_outputs = list(csv.DictReader(source))
        assert len(actf_inputs) == len(actf_outputs) == 2
        assert [row["invocation_id"] for row in actf_inputs] == [row["invocation_id"] for row in actf_outputs]
        assert [row["index"] for row in actf_inputs] == [row["index"] for row in actf_outputs] == ["1", "2"]
        assert b"actf-1-request" in (root / f"{actf}.pcap").read_bytes()
        assert b"actf-2-response" in (root / f"{actf}.pcap").read_bytes()
        assert b"stat-request" not in (root / f"{actf}.pcap").read_bytes()
        assert b"unrelated-traffic" not in (root / f"{actf}.pcap").read_bytes()
        assert b"stat-request" in (root / f"{stat}.pcap").read_bytes()


def test_reexport_removes_stale_api_files_from_previous_attempt() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        pcap = execution / "capture.pcap"
        _pcap(pcap)
        root = export_legacy_dataset(
            execution, [_invocation("old", "tx-old", api="cnc_actf")],
            [_transaction("old", "tx-old")], pcap,
        )
        old_stem = "20260911_focas_actf"
        assert (root / f"{old_stem}.pcap").exists()
        export_legacy_dataset(
            execution, [_invocation("new", "tx-new", api="cnc_statinfo")],
            [_transaction("new", "tx-new")], pcap,
        )
        assert not (root / f"{old_stem}.pcap").exists()
        assert not (root / f"{old_stem}_input.csv").exists()
        assert not (root / f"{old_stem}_output.csv").exists()


def test_runtime_per_api_pcaps_are_preferred_and_include_unmatched_calls() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        root = execution / "data" / "by_api"
        root.mkdir(parents=True)
        (root / ".runtime_per_api_capture").write_text(
            "mode=runtime_per_api_dumper;batch_id=batch_001\n", encoding="utf-8"
        )
        _pcap(root / "batch_001_cnc_actf.pcap")
        _pcap(root / "batch_001_cnc_statinfo.pcap")
        calls = [
            _invocation("actf-1", "tx-actf", api="cnc_actf"),
            _invocation("actf-2", "tx-actf-2", api="cnc_actf"),
            _invocation("stat-1", "tx-stat", api="cnc_statinfo"),
        ]
        transactions = [
            _transaction("actf-1", "tx-actf"),
            {"invocation_id": "actf-2", "transaction_id": None, "request_response_associated": False,
             "request_response_order_valid": False, "tcp_reassembled": True},
            _transaction("stat-1", "tx-stat"),
        ]
        export_legacy_dataset(execution, calls, transactions, None, output_root=root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pcap_mode"] == "runtime_per_api_dumper"
        assert manifest["pcap_files"] == ["batch_001_cnc_actf.pcap", "batch_001_cnc_statinfo.pcap"]
        assert (root / "batch_001_cnc_actf.pcap").exists()
        with (root / "20260911_focas_actf_input.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 2
        assert rows[1]["traffic_status"] == "no_corresponding_traffic"


def test_runtime_marker_does_not_fallback_to_time_window_pcaps_when_dumper_missing() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        root = execution / "data" / "by_api"
        root.mkdir(parents=True)
        (root / ".runtime_per_api_capture").write_text(
            "mode=runtime_per_api_dumper;batch_id=batch_missing\n", encoding="utf-8"
        )
        batch_pcap = execution / "data" / "focas_capture.pcap"
        _pcap(batch_pcap)
        export_legacy_dataset(
            execution,
            [_invocation("missing-1", "tx-missing", api="cnc_statinfo")],
            [_transaction("missing-1", "tx-missing")],
            batch_pcap,
            output_root=root,
        )
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pcap_mode"] == "runtime_per_api_dumper"
        assert manifest["pcap_files"] == []
        assert not list(root.glob("*.pcap"))
