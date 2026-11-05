"""Build auditable API-invocation/FOCAS-transaction mappings from classic PCAP.

The mapper uses runtime nanosecond call boundaries and TCP sequence numbers.  It
never treats an individual TCP packet as an application message: payload bytes
are de-duplicated and reassembled per direction before a transaction is emitted.
"""
from __future__ import annotations

import json
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .execution_contracts import API_TRAFFIC_SESSION_RULES, TRAFFIC_SESSION_TYPES


FOCAS_API = re.compile(r"^(?:cnc|pmc)_[A-Za-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class TcpPayload:
    timestamp_ns: int
    source_ip: str
    source_port: int
    destination_ip: str
    destination_port: int
    sequence: int
    payload: bytes


def _ipv4_text(raw: bytes) -> str:
    return ".".join(str(value) for value in raw)


def _ip_offset(frame: bytes, link_type: int) -> int | None:
    if link_type == 1:  # Ethernet
        if len(frame) < 14:
            return None
        ether_type = struct.unpack("!H", frame[12:14])[0]
        offset = 14
        while ether_type in {0x8100, 0x88A8} and len(frame) >= offset + 4:
            ether_type = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
            offset += 4
        return offset if ether_type == 0x0800 else None
    if link_type in {0, 108}:  # BSD/Windows loopback headers
        return 4 if len(frame) >= 4 and (frame[4] >> 4) == 4 else None
    if link_type == 101:  # raw IPv4
        return 0 if frame and (frame[0] >> 4) == 4 else None
    if link_type == 113:  # Linux cooked v1
        return 16 if len(frame) >= 16 and struct.unpack("!H", frame[14:16])[0] == 0x0800 else None
    if link_type == 276:  # Linux cooked v2
        return 20 if len(frame) >= 20 and struct.unpack("!H", frame[0:2])[0] == 0x0800 else None
    return None


def parse_tcp_payload_frame(frame: bytes, timestamp_ns: int, link_type: int) -> TcpPayload | None:
    ip = _ip_offset(frame, link_type)
    if ip is None or len(frame) < ip + 20 or (frame[ip] >> 4) != 4:
        return None
    ip_header_length = (frame[ip] & 0x0F) * 4
    if ip_header_length < 20 or len(frame) < ip + ip_header_length or frame[ip + 9] != 6:
        return None
    total_length = struct.unpack("!H", frame[ip + 2 : ip + 4])[0]
    tcp = ip + ip_header_length
    if len(frame) < tcp + 20:
        return None
    tcp_header_length = (frame[tcp + 12] >> 4) * 4
    if tcp_header_length < 20 or len(frame) < tcp + tcp_header_length:
        return None
    payload_end = min(len(frame), ip + total_length)
    payload = frame[tcp + tcp_header_length : payload_end]
    if not payload:
        return None
    return TcpPayload(
        timestamp_ns=timestamp_ns,
        source_ip=_ipv4_text(frame[ip + 12 : ip + 16]),
        source_port=struct.unpack("!H", frame[tcp : tcp + 2])[0],
        destination_ip=_ipv4_text(frame[ip + 16 : ip + 20]),
        destination_port=struct.unpack("!H", frame[tcp + 2 : tcp + 4])[0],
        sequence=struct.unpack("!I", frame[tcp + 4 : tcp + 8])[0],
        payload=payload,
    )


def read_classic_pcap_tcp_payloads(path: Path) -> list[TcpPayload]:
    """Read payload-bearing IPv4/TCP records from a classic pcap file."""

    raw = path.read_bytes()
    if len(raw) < 24:
        return []
    magic = raw[:4]
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1),
        b"\xa1\xb2\x3c\x4d": (">", 1),
    }
    if magic not in formats:
        return []
    endian, fraction_multiplier = formats[magic]
    link_type = struct.unpack(endian + "I", raw[20:24])[0]
    offset = 24
    payloads: list[TcpPayload] = []
    while offset + 16 <= len(raw):
        seconds, fraction, captured_length, _wire_length = struct.unpack(
            endian + "IIII", raw[offset : offset + 16]
        )
        offset += 16
        frame = raw[offset : offset + captured_length]
        if len(frame) != captured_length:
            break
        offset += captured_length
        parsed = parse_tcp_payload_frame(
            frame,
            seconds * 1_000_000_000 + fraction * fraction_multiplier,
            link_type,
        )
        if parsed is not None:
            payloads.append(parsed)
    return payloads


def _reassemble_direction(packets: Iterable[TcpPayload]) -> bytes:
    """Reassemble one directional byte stream and remove retransmitted bytes."""

    ordered = sorted(packets, key=lambda packet: (packet.sequence, packet.timestamp_ns))
    if not ordered:
        return b""
    output = bytearray()
    next_sequence = ordered[0].sequence
    for packet in ordered:
        start = packet.sequence
        payload = packet.payload
        if start < next_sequence:
            already_seen = next_sequence - start
            if already_seen >= len(payload):
                continue
            payload = payload[already_seen:]
            start = next_sequence
        if start > next_sequence:
            # Do not invent bytes across a capture gap. A gapped stream is not
            # valid evidence and is rejected by returning no application data.
            return b""
        output.extend(payload)
        next_sequence += len(payload)
    return bytes(output)


def _phase(value: Any) -> str:
    lowered = str(value or "during").strip().lower()
    return {"before": "Before", "after": "After"}.get(lowered, "During")


def build_protocol_transactions(
    invocation_rows: list[dict[str, Any]],
    tcp_payloads: list[TcpPayload],
    *,
    focas_port: int = 8193,
    capture_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Map payloads to non-overlapping API call intervals.

    Every FOCAS invocation produces one audit record. A call with no complete
    request/response is explicit and retains no valid transaction association.
    """

    rows = sorted(invocation_rows, key=lambda row: int(row.get("start_timestamp_ns") or 0))
    invocation_id_counts = Counter(str(row.get("invocation_id") or "") for row in rows)
    transaction_id_counts = Counter(str(row.get("transaction_id") or "") for row in rows)
    packet_owners: dict[int, list[int]] = {}
    endpoint_packets = [
        packet for packet in tcp_payloads
        if packet.source_port == focas_port or packet.destination_port == focas_port
    ]
    for packet_index, packet in enumerate(tcp_payloads):
        if packet.source_port != focas_port and packet.destination_port != focas_port:
            continue
        packet_owners[packet_index] = [
            row_index
            for row_index, row in enumerate(rows)
            if int(row.get("start_timestamp_ns") or 0) > 0
            and int(row.get("start_timestamp_ns") or 0) <= packet.timestamp_ns
            <= int(row.get("end_timestamp_ns") or 0)
        ]
    records: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for invocation_order, row in enumerate(rows):
        invocation_id = str(row.get("invocation_id") or "")
        candidate_transaction_id = str(row.get("transaction_id") or "")
        api = str(row.get("api") or row.get("api_name") or "")
        start_ns = int(row.get("start_timestamp_ns") or 0)
        end_ns = int(row.get("end_timestamp_ns") or 0)
        candidate_indexes = [
            index
            for index, owners in packet_owners.items()
            if owners == [invocation_order]
        ]
        endpoint_packet_count = len(endpoint_packets)
        window_packet_count = sum(
            1
            for packet in endpoint_packets
            if start_ns > 0 and start_ns <= packet.timestamp_ns <= end_ns
        )
        before_packets = [packet.timestamp_ns for packet in endpoint_packets if start_ns > 0 and packet.timestamp_ns < start_ns]
        after_packets = [packet.timestamp_ns for packet in endpoint_packets if end_ns > 0 and packet.timestamp_ns > end_ns]
        nearest_before_ns = max(before_packets) if before_packets else None
        nearest_after_ns = min(after_packets) if after_packets else None
        packets = [tcp_payloads[index] for index in candidate_indexes]
        # Each socket has independent sequence spaces. Connection allocation
        # can exchange data on multiple sockets during one library invocation.
        connections: dict[tuple[str, int, str, int], list[TcpPayload]] = {}
        for packet in packets:
            key = ((packet.source_ip, packet.source_port, packet.destination_ip, packet.destination_port)
                   if packet.destination_port == focas_port else
                   (packet.destination_ip, packet.destination_port, packet.source_ip, packet.source_port))
            connections.setdefault(key, []).append(packet)
        streams: list[dict[str, Any]] = []
        for key, stream_packets in sorted(connections.items(), key=lambda item: min(p.timestamp_ns for p in item[1])):
            requests = [packet for packet in stream_packets if packet.destination_port == focas_port]
            responses = [packet for packet in stream_packets if packet.source_port == focas_port]
            request = _reassemble_direction(requests)
            response = _reassemble_direction(responses)
            ordered = bool(requests and responses and min(p.timestamp_ns for p in requests) < min(p.timestamp_ns for p in responses))
            streams.append({
                "stream_id": f"{key[0]}:{key[1]}->{key[2]}:{key[3]}",
                "client_ip": key[0], "client_port": key[1], "server_ip": key[2], "server_port": key[3],
                "request_payload_hex": request.hex() or None,
                "response_payload_hex": response.hex() or None,
                "request_timestamp_ns": min((p.timestamp_ns for p in requests), default=None),
                "response_timestamp_ns": min((p.timestamp_ns for p in responses), default=None),
                "request_response_order_valid": ordered,
            })
        correct_order = bool(streams) and all(stream["request_response_order_valid"] for stream in streams)
        associated = bool(
            streams and all(stream["request_payload_hex"] and stream["response_payload_hex"] for stream in streams) and correct_order
            and invocation_id and candidate_transaction_id
            and invocation_id_counts[invocation_id] == 1
            and transaction_id_counts[candidate_transaction_id] == 1
        )
        messages: list[dict[str, Any]] = []
        for stream in streams:
            for direction in ("request", "response"):
                if stream[f"{direction}_payload_hex"]:
                    messages.append({
                        "direction": direction, "stream_id": stream["stream_id"],
                        "timestamp_ns": stream[f"{direction}_timestamp_ns"],
                        "payload_hex": stream[f"{direction}_payload_hex"],
                    })
        messages.sort(key=lambda message: message["timestamp_ns"])
        for message_order, message in enumerate(messages):
            message["message_order"] = message_order
        transaction_id = candidate_transaction_id if associated else None
        if associated:
            association_reason = "matched_request_response"
        elif any(len(owners) > 1 and invocation_order in owners for owners in packet_owners.values()):
            association_reason = "overlapping_invocation_windows"
        elif not window_packet_count:
            association_reason = (
                "no_endpoint_packets_in_invocation_window"
                if endpoint_packet_count
                else "no_endpoint_packets_in_capture"
            )
        elif not streams:
            association_reason = "exclusive_packet_ownership_failed"
        elif not correct_order or not all(
            stream["request_payload_hex"] and stream["response_payload_hex"] for stream in streams
        ):
            association_reason = "incomplete_request_response_exchange"
        elif not invocation_id or not candidate_transaction_id:
            association_reason = "missing_invocation_or_candidate_transaction_id"
        else:
            association_reason = "request_response_association_failed"
        record = {
            "transaction_id": transaction_id,
            "candidate_transaction_id": candidate_transaction_id or None,
            "invocation_id": invocation_id or None,
            "api": api,
            "decoded_parameters": row.get("input_parameters", {}),
            "decoded_return_code": row.get("return_code"),
            "call_association_verified": bool(
                invocation_id and candidate_transaction_id
                and invocation_id_counts[invocation_id] == 1
                and transaction_id_counts[candidate_transaction_id] == 1
            ),
            "phase": _phase(row.get("phase")),
            "device_context": row.get("device_context", {}),
            "invocation_order": invocation_order,
            "log_timestamp": row.get("log_timestamp"),
            "invocation_start_timestamp_ns": start_ns or None,
            "invocation_end_timestamp_ns": end_ns or None,
            "request_response_associated": associated,
            "request_response_order_valid": correct_order,
            "tcp_reassembled": True,
            "contains_transport_headers": False,
            "application_messages": messages if associated else [],
            # Preserve the legacy scalar only for a single TCP stream. Never
            # concatenate payloads from unrelated connections into fake bytes.
            "request_payload_hex": streams[0]["request_payload_hex"] if associated and len(streams) == 1 else None,
            "response_payload_hex": streams[0]["response_payload_hex"] if associated and len(streams) == 1 else None,
            "tcp_streams": streams,
            "wire_evidence_diagnostic": {
                "focas_port": focas_port,
                "endpoint_packet_count": endpoint_packet_count,
                "invocation_window_packet_count": window_packet_count,
                "exclusive_packet_count": len(packets),
                "association_reason": association_reason,
                "nearest_endpoint_packet_before_ns": nearest_before_ns,
                "nearest_endpoint_packet_after_ns": nearest_after_ns,
                "window_start_gap_ns": (start_ns - nearest_before_ns) if nearest_before_ns is not None else None,
                "window_end_gap_ns": (nearest_after_ns - end_ns) if nearest_after_ns is not None else None,
                **(capture_metadata or {}).get(api, {}),
            },
            "annotation": row.get("annotation"),
            "traffic_status": "matched" if associated else (
                "ambiguous_overlapping_invocation_windows"
                if any(len(owners) > 1 and invocation_order in owners for owners in packet_owners.values())
                else "no_corresponding_traffic"
            ),
            "matching_method": "exclusive_invocation_time_window_and_tcp_connection",
            "traffic_observation_complete": True,
        }
        records.append(record)
        if invocation_id:
            mapping[invocation_id] = record
    annotate_session_transactions(records)
    return records, mapping


def annotate_session_transactions(records: list[dict[str, Any]]) -> None:
    """Attach evidence for documented data calls that complete in a session."""
    if any(rule.get("session_type") not in TRAFFIC_SESSION_TYPES
           for rule in API_TRAFFIC_SESSION_RULES.values()):
        raise ValueError("Unknown FOCAS traffic session type in API metadata")

    invocation_counts = Counter(str(row.get("invocation_id") or "") for row in records)
    transaction_counts = Counter(str(row.get("candidate_transaction_id") or row.get("transaction_id") or "")
                                 for row in records)
    for row in records:
        invocation_id = str(row.get("invocation_id") or "")
        transaction_id = str(row.get("candidate_transaction_id") or row.get("transaction_id") or "")
        row["call_association_verified"] = bool(
            invocation_id and transaction_id and invocation_counts[invocation_id] == 1
            and transaction_counts[transaction_id] == 1
        )

    ordered = sorted(records, key=lambda row: int(row.get("invocation_order", -1)))
    for rule in API_TRAFFIC_SESSION_RULES.values():
        segment_api = rule["segment_api"]
        starts = [row for row in ordered if row.get("api") == rule["start_api"]]
        ends = [row for row in ordered if row.get("api") == rule["end_api"]]
        segments = [row for row in ordered if row.get("api") == segment_api]
        if not segments:
            continue

        for segment in segments:
            order = int(segment.get("invocation_order", -1))
            prior_starts = [row for row in starts if int(row.get("invocation_order", -1)) < order]
            start = prior_starts[-1] if prior_starts else None
            next_start_order = min((int(row.get("invocation_order", -1)) for row in starts
                                    if int(row.get("invocation_order", -1)) > order), default=None)
            following_ends = [row for row in ends
                              if int(row.get("invocation_order", -1)) > order
                              and (next_start_order is None or int(row.get("invocation_order", -1)) < next_start_order)]
            end = following_ends[0] if following_ends else None
            session_id = (
                f"{start.get('invocation_id')}->{end.get('invocation_id')}"
                if start and end else
                f"{start.get('invocation_id')}->incomplete" if start else "unpaired"
            )
            direction = rule["wire_direction"]
            wire_verified = _segment_wire_verified(segment, direction)
            start_result = _return_code(start)
            end_result = _return_code(end)
            same_handle = bool(start and end and _session_handle(start)
                               and _session_handle(start) == _session_handle(segment) == _session_handle(end))
            start_valid = bool(start and start_result == 0 and _has_matched_exchange(start)
                               and start.get("call_association_verified"))
            end_valid = bool(end and end_result == 0 and _has_matched_exchange(end)
                             and end.get("call_association_verified"))
            segment_valid = bool(_return_code(segment) == 0 and wire_verified
                                 and segment.get("call_association_verified"))
            peer_segments = [row for row in segments
                             if start and int(row.get("invocation_order", -1)) > int(start.get("invocation_order", -1))
                             and (end is None or int(row.get("invocation_order", -1)) < int(end.get("invocation_order", -1)))
                             and (next_start_order is None or int(row.get("invocation_order", -1)) < next_start_order)]
            all_segments_valid = bool(peer_segments) and all(
                _return_code(row) == 0 and _segment_wire_verified(row, direction)
                and row.get("call_association_verified") for row in peer_segments
            )
            completed = bool(start_valid and end_valid and same_handle and all_segments_valid)
            completion_status = "completed_successfully" if completed else (
                "failed" if start and start_result not in (None, 0)
                or end and end_result not in (None, 0)
                or any(_return_code(row) not in (None, 0) for row in peer_segments)
                else "incomplete"
            )

            # Keep the transport classification literal: a request-only segment
            # remains request-only and is never represented as a fake exchange.
            if direction == "request" and _has_request_only_stream(segment):
                segment["traffic_status"] = "request_only_segment"
                segment["matching_method"] = "exclusive_invocation_window_with_session_evidence"
            analysis = {
                "kind": "session_segment",
                "api_name": segment_api,
                "call_id": segment.get("invocation_id"),
                "session_type": rule["session_type"],
                "wire_direction": direction,
                "session_id": session_id,
                "session_start_call_id": start.get("invocation_id") if start else None,
                "session_end_call_id": end.get("invocation_id") if end else None,
                "wire_segment_verified": wire_verified,
                "wire_evidence_source": "pcap_tcp_reassembly" if wire_verified else None,
                "wire_stream_count": sum(bool(stream.get("request_payload_hex") or stream.get("response_payload_hex"))
                                         for stream in (segment.get("tcp_streams") or [])) if wire_verified else 0,
                "session_start_result": start_result,
                "session_end_result": end_result,
                "session_completion_status": completion_status,
                "api_return_code": _return_code(segment),
                "call_association_verified": bool(segment.get("call_association_verified")),
                "request_captured": _has_request_only_stream(segment) or bool(segment.get("request_response_associated")),
                "response_expected_in_this_invocation": direction != "request",
                "lifecycle_start_api": rule["start_api"],
                "lifecycle_end_api": rule["end_api"],
            }
            segment["session_traffic"] = analysis
            segment["traffic_analysis"] = analysis


def _return_code(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    try:
        return int(row.get("decoded_return_code"))
    except (TypeError, ValueError):
        return None


def _has_matched_exchange(row: dict[str, Any]) -> bool:
    return bool(row.get("traffic_status") == "matched"
                and row.get("request_response_associated") is True
                and row.get("request_response_order_valid") is True)


def _segment_wire_verified(row: dict[str, Any], direction: str) -> bool:
    streams = row.get("tcp_streams") or []
    if not streams or not row.get("call_association_verified"):
        return False
    has_request = any(stream.get("request_payload_hex") for stream in streams)
    has_response = any(stream.get("response_payload_hex") for stream in streams)
    if direction == "request":
        return bool(has_request and not has_response and not row.get("request_response_associated"))
    if direction == "response":
        return bool(_has_matched_exchange(row) and has_response)
    if direction == "both":
        return bool(_has_matched_exchange(row) and has_request and has_response)
    return False


def _session_handle(row: dict[str, Any]) -> str:
    value = row.get("decoded_parameters", {})
    if isinstance(value, dict):
        value = value.get("raw", value)
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).strip().lower() in {"flibhndl", "handle"}:
                    return str(item).strip()
    fields = dict(part.split("=", 1) for part in str(value).split(";") if "=" in part)
    return next((str(item).strip() for key, item in fields.items()
                 if key.strip().lower() in {"flibhndl", "handle"}), "")


def valid_session_segment_evidence(row: Any) -> bool:
    """Validate a data-call's per-call, wire, association, and session evidence."""
    if isinstance(row, dict):
        api = str(row.get("protocol_function") or row.get("api") or row.get("api_name") or "")
        call_id = str(row.get("invocation_id") or row.get("call_id") or "")
        traffic_status = str(row.get("traffic_status") or "")
        return_code = row.get("status_code", row.get("decoded_return_code", row.get("api_return_code")))
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        session = row.get("session_traffic") or response.get("session_traffic") or response.get("traffic_analysis") or row.get("traffic_analysis")
    else:
        api = str(getattr(row, "protocol_function", "") or "")
        call_id = str(getattr(row, "invocation_id", "") or "")
        traffic_status = str(getattr(row, "traffic_status", "") or "")
        return_code = getattr(row, "status_code", None)
        response = getattr(row, "response", {})
        session = (response.get("session_traffic") or response.get("traffic_analysis")
                   if isinstance(response, dict) else {})
    if not isinstance(session, dict) or session.get("kind") != "session_segment":
        return False
    try:
        code = int(return_code)
        api_code = int(session.get("api_return_code"))
        start_code = int(session.get("session_start_result"))
        end_code = int(session.get("session_end_result"))
    except (TypeError, ValueError):
        return False
    direction = session.get("wire_direction")
    traffic_matches_direction = (
        direction == "request" and traffic_status == "request_only_segment"
        or direction in {"response", "both"} and traffic_status == "matched"
    )
    return bool(
        api and call_id and session.get("session_id") not in (None, "", "unpaired")
        and session.get("api_name") == api and session.get("call_id") == call_id
        and session.get("session_start_call_id") and session.get("session_end_call_id")
        and session.get("session_id") == f"{session.get('session_start_call_id')}->{session.get('session_end_call_id')}"
        and session.get("call_association_verified") is True
        and session.get("wire_segment_verified") is True
        and session.get("wire_evidence_source") == "pcap_tcp_reassembly"
        and int(session.get("wire_stream_count") or 0) > 0
        and session.get("session_completion_status") == "completed_successfully"
        and traffic_matches_direction and code == api_code == start_code == end_code == 0
    )


def _has_request_only_stream(record: dict[str, Any]) -> bool:
    streams = record.get("tcp_streams") or []
    return bool(
        streams
        and any(stream.get("request_payload_hex") and not stream.get("response_payload_hex")
                for stream in streams)
        and not record.get("request_response_associated")
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
