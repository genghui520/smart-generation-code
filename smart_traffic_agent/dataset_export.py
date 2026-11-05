"""Export captured FOCAS calls in a per-API dataset layout.

The structured audit logs remain the source of truth.  This module creates a
consumer-friendly compatibility view without weakening the invocation-to-
transaction checks used by Dataset Quality evaluation.
"""
from __future__ import annotations

import base64
import csv
import json
import re
import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .traffic_transactions import parse_tcp_payload_frame


INPUT_CSV_FIELDS = [
    "index", "timestamp", "invocation_id", "transaction_id", "traffic_status",
    "api_name", "api_parameter_count", "parameters",
]
OUTPUT_CSV_FIELDS = [
    "index", "timestamp", "invocation_id", "transaction_id", "traffic_status",
    "api_name", "return_code", "data",
]


def export_legacy_dataset(
    execution_dir: Path,
    invocation_rows: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    pcap_path: Path | None,
    *,
    output_root: Path | None = None,
    focas_port: int = 8193,
) -> Path:
    """Write one PCAP and paired input/output CSV logs per FOCAS API.

    CSV logs contain every invocation, including calls without mapped traffic.
    PCAP, dataset TXT and training JSONL contain only calls with a unique,
    ordered, TCP-reassembled request/response transaction.
    """

    output_root = output_root or execution_dir / "dataset_export"
    dataset_dir = output_root / "dataset"
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    remove_previous_export_files(output_root, dataset_dir)

    transaction_by_invocation = {
        str(row.get("invocation_id")): row
        for row in transactions
        if row.get("invocation_id")
    }
    inputs_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outputs_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_by_stem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for invocation in invocation_rows:
        api = str(invocation.get("api") or "unknown_api")
        timestamp = str(invocation.get("log_timestamp") or "")
        stem = _stem_for_api(api, timestamp)
        transaction = transaction_by_invocation.get(str(invocation.get("invocation_id") or ""))
        valid = _is_valid_transaction(transaction)
        index = len(inputs_by_stem[stem]) + 1
        common = {
            "index": index,
            "timestamp": _display_timestamp(invocation.get("log_timestamp")),
            "invocation_id": str(invocation.get("invocation_id") or ""),
            "transaction_id": str(transaction.get("transaction_id") or "") if valid else "",
            "traffic_status": "matched" if valid else str((transaction or {}).get("traffic_status") or "no_corresponding_traffic"),
            "api_name": api,
        }
        inputs_by_stem[stem].append({
            **common,
            "api_parameter_count": invocation.get("api_parameter_count", 0),
            "parameters": invocation.get("input_parameters", ""),
        })
        outputs_by_stem[stem].append({
            **common,
            "return_code": invocation.get("return_code"),
            "data": invocation.get("output_data", ""),
        })
        if valid:
            matched_by_stem[stem].append((invocation, transaction))

    for stem, rows in inputs_by_stem.items():
        for old_name in (f"{stem}.csv", f"{stem}_CLEANED.csv"):
            try:
                (output_root / old_name).unlink(missing_ok=True)
            except OSError:
                pass
        _write_csv(output_root / f"{stem}_input.csv", rows, INPUT_CSV_FIELDS)
        _write_csv(output_root / f"{stem}_output.csv", outputs_by_stem[stem], OUTPUT_CSV_FIELDS)
        _write_dataset_txt(dataset_dir / f"{stem}.txt", matched_by_stem.get(stem, []))

    runtime_marker = output_root / ".runtime_per_api_capture"
    runtime_pcap_paths = _discover_runtime_per_api_pcaps(output_root)
    if runtime_marker.is_file():
        # A runtime marker is an explicit statement that the harness attempted
        # per-API capture.  Do not silently replace missing API dumpers with a
        # time-window split of the batch PCAP; that would hide capture failure.
        pcap_stems = [path.stem for path in runtime_pcap_paths]
        pcap_files = [path.name for path in runtime_pcap_paths]
        pcap_mode = "runtime_per_api_dumper"
    else:
        pcap_stems = _write_per_api_pcaps(output_root, matched_by_stem, pcap_path, focas_port=focas_port)
        pcap_files = [f"{stem}.pcap" for stem in pcap_stems]
        pcap_mode = "batch_time_window_fallback"
    _write_labels_and_training(output_root, matched_by_stem)
    _write_manifest(
        output_root, inputs_by_stem, matched_by_stem, pcap_stems,
        pcap_files=pcap_files, pcap_mode=pcap_mode,
        source_of_truth=("../../api_logs.jsonl and ../../protocol_transactions.jsonl"
                         if output_root == execution_dir / "data" / "by_api"
                         else "../api_logs.jsonl and ../protocol_transactions.jsonl"),
    )
    return output_root


def _discover_runtime_per_api_pcaps(output_root: Path) -> list[Path]:
    """Return API PCAPs written during execution, when the marker is present.

    A marker prevents an old compatibility export from being mistaken for a
    runtime capture. Runtime files contain all calls for an API, including
    calls whose transaction audit later remains unmatched.
    """
    marker = output_root / ".runtime_per_api_capture"
    if not marker.is_file():
        return []
    return sorted(
        path for path in output_root.glob("*.pcap")
        if path.is_file() and path.stat().st_size >= 24
    )


def _operation_name(api: str) -> str:
    return re.sub(r"^(?:cnc|pmc)_", "", api, flags=re.IGNORECASE)


def _stem_for_api(api: str, timestamp: str) -> str:
    family = "pmc" if api.lower().startswith("pmc_") else "focas"
    return f"{_date_prefix(timestamp)}_{family}_{_safe_name(_operation_name(api))}"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown_api"


def _date_prefix(value: str) -> str:
    candidate = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).strftime("%Y%m%d")
    except ValueError:
        match = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", candidate)
        if match:
            return "".join(match.groups())
        return datetime.now().strftime("%Y%m%d")


def _display_timestamp(value: Any) -> str:
    candidate = str(value or "").strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate).strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    except ValueError:
        return str(value or "")


def _key_values(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, part in enumerate(str(value or "").split(";")):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, item = part.split("=", 1)
        else:
            key, item = f"value{index}", part
        pairs.append((key.strip(), item.strip()))
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _transaction_streams(transaction: dict[str, Any]) -> list[dict[str, Any]]:
    streams = transaction.get("tcp_streams")
    if isinstance(streams, list) and streams:
        return streams
    # Existing single-connection records remain readable.
    return [{"request_payload_hex": transaction.get("request_payload_hex"),
             "response_payload_hex": transaction.get("response_payload_hex")}]


def _is_valid_transaction(transaction: dict[str, Any] | None) -> bool:
    return bool(
        transaction
        and transaction.get("transaction_id")
        and transaction.get("request_response_associated") is True
        and transaction.get("request_response_order_valid") is True
        and transaction.get("tcp_reassembled") is True
        and all(stream.get("request_payload_hex") and stream.get("response_payload_hex")
                for stream in _transaction_streams(transaction))
    )


def _write_dataset_txt(
    path: Path,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    csv_data: list[dict[str, Any]] = []
    packet_data: list[list[list[str]]] = []
    mapping_data: list[dict[str, Any]] = []
    for invocation, transaction in pairs:
        operation = _operation_name(str(invocation.get("api") or ""))
        item: dict[str, Any] = {
            "time_ip": _display_timestamp(invocation.get("log_timestamp")),
            "operation": operation,
            "arg_num": str(invocation.get("api_parameter_count", 0)),
        }
        item.update(dict(_key_values(invocation.get("input_parameters"))))
        csv_data.append(item)
        stream_messages = []
        streams = _transaction_streams(transaction)
        for stream in streams:
            request = bytes.fromhex(str(stream["request_payload_hex"]))
            response = bytes.fromhex(str(stream["response_payload_hex"]))
            suffix = f" [{stream['stream_id']}]" if len(streams) > 1 else ""
            stream_messages.extend([
                [f"Request{suffix}:", base64.b64encode(request).decode("ascii")],
                [f"Response{suffix}:", base64.b64encode(response).decode("ascii")],
            ])
        packet_data.append(stream_messages)
        mapping_data.append(
            {
                "invocation_id": invocation.get("invocation_id"),
                "transaction_id": transaction.get("transaction_id"),
                "phase": invocation.get("phase"),
                "return_code": invocation.get("return_code"),
                "start_timestamp_ns": invocation.get("start_timestamp_ns"),
                "end_timestamp_ns": invocation.get("end_timestamp_ns"),
                "device_context": invocation.get("device_context", {}),
                "tcp_streams": streams,
            }
        )
    document = {
        "csv_data": csv_data,
        "packet_data": packet_data,
        "mapping_data": mapping_data,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=4), encoding="utf-8")


def _write_labels_and_training(
    output_root: Path,
    matched_by_stem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
) -> None:
    operations = sorted(
        {
            _operation_name(str(invocation.get("api") or ""))
            for pairs in matched_by_stem.values()
            for invocation, _transaction in pairs
        }
    )
    labels = {f"{operation}|": index for index, operation in enumerate(operations)}
    (output_root / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "training_dataset.jsonl").open("w", encoding="utf-8", newline="\n") as output:
        sample_index = 0
        for stem in sorted(matched_by_stem):
            for invocation, transaction in matched_by_stem[stem]:
                operation = _operation_name(str(invocation.get("api") or ""))
                label = f"{operation}|"
                record = {
                    "func": stem,
                    "index": sample_index,
                    "label_str": label,
                    "label_id": labels[label],
                    "request_hex": transaction.get("request_payload_hex"),
                    "response_hex": transaction.get("response_payload_hex"),
                    "tcp_streams": _transaction_streams(transaction),
                    "invocation_id": invocation.get("invocation_id"),
                    "transaction_id": transaction.get("transaction_id"),
                    "phase": invocation.get("phase"),
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                sample_index += 1


def _read_pcap_records(path: Path) -> tuple[bytes, list[tuple[int, bytes]]]:
    raw = path.read_bytes()
    if len(raw) < 24:
        return b"", []
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1),
        b"\xa1\xb2\x3c\x4d": (">", 1),
    }
    if raw[:4] not in formats:
        return b"", []
    endian, multiplier = formats[raw[:4]]
    records: list[tuple[int, bytes]] = []
    offset = 24
    while offset + 16 <= len(raw):
        header_offset = offset
        seconds, fraction, captured_length, _wire_length = struct.unpack(
            endian + "IIII", raw[offset : offset + 16]
        )
        offset += 16
        if offset + captured_length > len(raw):
            break
        offset += captured_length
        records.append((seconds * 1_000_000_000 + fraction * multiplier, raw[header_offset:offset]))
    return raw[:24], records


def _write_per_api_pcaps(
    output_root: Path,
    matched_by_stem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    pcap_path: Path | None,
    *,
    focas_port: int,
) -> list[str]:
    header, records = _read_pcap_records(pcap_path) if pcap_path and pcap_path.exists() else (b"", [])
    written: list[str] = []
    if not header:
        return written
    endian = "<" if header[:4] in {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"} else ">"
    link_type = struct.unpack(endian + "I", header[20:24])[0]
    focas_records = [
        (timestamp_ns, record)
        for timestamp_ns, record in records
        if (
            (packet := parse_tcp_payload_frame(record[16:], timestamp_ns, link_type)) is not None
            and (packet.source_port == focas_port or packet.destination_port == focas_port)
        )
    ]
    for stem, pairs in matched_by_stem.items():
        selected = bytearray(header)
        intervals = [
            (int(invocation.get("start_timestamp_ns") or 0), int(invocation.get("end_timestamp_ns") or 0))
            for invocation, _transaction in pairs
        ]
        for timestamp_ns, record in focas_records:
            if any(start > 0 and start <= timestamp_ns <= end for start, end in intervals):
                selected.extend(record)
        if len(selected) > len(header):
            (output_root / f"{stem}.pcap").write_bytes(bytes(selected))
            written.append(stem)
    return written


def remove_previous_export_files(output_root: Path, dataset_dir: Path) -> None:
    """Remove only files listed by the previous export manifest."""
    manifest_path = output_root / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    preserve_runtime_pcaps = (output_root / ".runtime_per_api_capture").is_file()
    for name in manifest.get("generated_files", []):
        if not isinstance(name, str) or Path(name).name != name:
            continue
        path = output_root / name
        if path.suffix not in {".csv", ".pcap"}:
            continue
        if preserve_runtime_pcaps and path.suffix == ".pcap":
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    for stem in manifest.get("api_stems", []):
        if not isinstance(stem, str) or _safe_name(stem) != stem:
            continue
        try:
            (dataset_dir / f"{stem}.txt").unlink(missing_ok=True)
        except OSError:
            pass
    for name in ("manifest.json", "labels.json", "training_dataset.jsonl"):
        try:
            (output_root / name).unlink(missing_ok=True)
        except OSError:
            pass


def _write_manifest(
    output_root: Path,
    rows_by_stem: dict[str, list[dict[str, Any]]],
    matched_by_stem: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    pcap_stems: list[str],
    *,
    pcap_files: list[str] | None = None,
    pcap_mode: str = "batch_time_window_fallback",
    source_of_truth: str,
) -> None:
    stems = sorted(rows_by_stem)
    pcap_files = list(pcap_files or [f"{stem}.pcap" for stem in pcap_stems])
    manifest = {
        "format": "per-api-pcap-and-paired-logs-v2",
        "source_of_truth": source_of_truth,
        "raw_invocation_count": sum(len(rows) for rows in rows_by_stem.values()),
        "valid_transaction_count": sum(len(rows) for rows in matched_by_stem.values()),
        "api_file_count": len(rows_by_stem),
        "cleaning_rule": "unique ordered TCP-reassembled request-response association",
        "pcap_mode": pcap_mode,
        "pcap_files": pcap_files,
        "api_stems": stems,
        "generated_files": [
            *(f"{stem}_input.csv" for stem in stems),
            *(f"{stem}_output.csv" for stem in stems),
            *pcap_files,
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
