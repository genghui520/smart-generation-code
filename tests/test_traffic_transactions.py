from __future__ import annotations

import json
import csv
import struct
import tempfile
from pathlib import Path

from smart_traffic_agent.agent_roles import CodeSpec
from smart_traffic_agent.agent_tools.cpp_execution import (
    CompileCppInput,
    CompileGeneratedCppTool,
    find_msvc_x86_toolchain,
    read_cpp_csv_logs,
)
from smart_traffic_agent.agent_tools.render_focas_cpp import RenderFocasCppInput, RenderFocasCppTool
from smart_traffic_agent.agent_tools.validate_focas_cpp import ValidateFocasCppInput, ValidateFocasCppTool
from smart_traffic_agent.agents.code_generator import (
    build_scenario_fragment,
    render_fixed_ncguide_cpp_script,
)
from smart_traffic_agent.models import (
    ExecutionPlan,
    NcProgramSpec,
    PlanStep,
    TaskRequest,
    WorkflowState,
)
from smart_traffic_agent.traffic_transactions import TcpPayload, build_protocol_transactions


def invocation(number: int, start: int, end: int, phase: str = "During") -> dict:
    return {
        "invocation_id": f"inv-{number}",
        "transaction_id": f"tx-{number}",
        "api": "cnc_statinfo",
        "input_parameters": {"sample": number},
        "return_code": 0,
        "phase": phase,
        "start_timestamp_ns": start,
        "end_timestamp_ns": end,
        "device_context": {"target": "NCGuide", "phase": phase},
        "annotation": "status_query",
    }


def packet(timestamp: int, sequence: int, payload: bytes, *, response: bool = False, client_port: int = 50000) -> TcpPayload:
    return TcpPayload(
        timestamp_ns=timestamp,
        source_ip="127.0.0.1",
        source_port=8193 if response else client_port,
        destination_ip="127.0.0.1",
        destination_port=client_port if response else 8193,
        sequence=sequence,
        payload=payload,
    )


def ethernet_tcp_frame(payload: bytes, sequence: int, *, response: bool = False) -> bytes:
    ethernet = b"\x00" * 12 + b"\x08\x00"
    total_length = 20 + 20 + len(payload)
    ip = bytearray(20)
    ip[0] = 0x45
    ip[2:4] = struct.pack("!H", total_length)
    ip[8] = 64
    ip[9] = 6
    ip[12:16] = b"\x7f\x00\x00\x01"
    ip[16:20] = b"\x7f\x00\x00\x01"
    tcp = bytearray(20)
    source_port, destination_port = ((8193, 50000) if response else (50000, 8193))
    tcp[0:2] = struct.pack("!H", source_port)
    tcp[2:4] = struct.pack("!H", destination_port)
    tcp[4:8] = struct.pack("!I", sequence)
    tcp[12] = 0x50
    return ethernet + bytes(ip) + bytes(tcp) + payload


def write_pcap(path: Path, frames: list[tuple[int, bytes]]) -> None:
    data = bytearray(b"\xd4\xc3\xb2\xa1")
    data.extend(struct.pack("<HHIIII", 2, 4, 0, 0, 65535, 1))
    for timestamp_ns, frame in frames:
        seconds, remainder = divmod(timestamp_ns, 1_000_000_000)
        data.extend(struct.pack("<IIII", seconds, remainder // 1_000, len(frame), len(frame)))
        data.extend(frame)
    path.write_bytes(bytes(data))


def test_every_invocation_and_valid_transaction_id_is_unique() -> None:
    rows = [invocation(1, 100, 199), invocation(2, 200, 299)]
    packets = [
        packet(110, 10, b"Q1"), packet(120, 20, b"A1", response=True),
        packet(210, 30, b"Q2"), packet(220, 40, b"A2", response=True),
    ]

    transactions, _mapping = build_protocol_transactions(rows, packets)

    assert len({row["invocation_id"] for row in transactions}) == 2
    assert len({row["transaction_id"] for row in transactions}) == 2


def test_invocation_transaction_mapping_is_bidirectional() -> None:
    rows = [invocation(1, 100, 199)]
    transactions, mapping = build_protocol_transactions(
        rows, [packet(110, 10, b"Q"), packet(120, 20, b"A", response=True)]
    )

    assert mapping["inv-1"]["transaction_id"] == "tx-1"
    assert transactions[0]["invocation_id"] == "inv-1"


def test_before_during_after_are_preserved() -> None:
    rows = [
        invocation(1, 100, 199, "before"),
        invocation(2, 200, 299, "During"),
        invocation(3, 300, 399, "AFTER"),
    ]

    transactions, _mapping = build_protocol_transactions(rows, [])

    assert [row["phase"] for row in transactions] == ["Before", "During", "After"]


def test_tcp_segments_are_reassembled_and_request_precedes_response() -> None:
    rows = [invocation(1, 100, 199)]
    packets = [
        packet(130, 12, b"CD"),
        packet(110, 10, b"AB"),
        packet(115, 10, b"AB"),  # retransmission
        packet(150, 20, b"OK", response=True),
    ]

    transactions, _mapping = build_protocol_transactions(rows, packets)
    messages = transactions[0]["application_messages"]

    assert [message["direction"] for message in messages] == ["request", "response"]
    assert [message["message_order"] for message in messages] == [0, 1]
    assert messages[0]["payload_hex"] == b"ABCD".hex()


def test_consecutive_calls_do_not_share_or_cross_bind_payloads() -> None:
    rows = [invocation(1, 100, 199), invocation(2, 200, 299)]
    packets = [
        packet(110, 10, b"REQ1"), packet(120, 20, b"RES1", response=True),
        packet(210, 30, b"REQ2"), packet(220, 40, b"RES2", response=True),
    ]

    transactions, _mapping = build_protocol_transactions(rows, packets)

    assert transactions[0]["request_payload_hex"] == b"REQ1".hex()
    assert transactions[1]["request_payload_hex"] == b"REQ2".hex()
    assert transactions[0]["response_payload_hex"] == b"RES1".hex()
    assert transactions[1]["response_payload_hex"] == b"RES2".hex()


def test_overlapping_invocation_windows_are_rejected_as_ambiguous() -> None:
    rows = [invocation(1, 100, 250), invocation(2, 200, 299)]
    packets = [packet(210, 10, b"Q"), packet(220, 20, b"A", response=True)]

    transactions, _mapping = build_protocol_transactions(rows, packets)

    assert all(row["transaction_id"] is None for row in transactions)
    assert all(row["traffic_status"] == "ambiguous_overlapping_invocation_windows" for row in transactions)


def test_missing_response_is_explicit_and_not_bound() -> None:
    transactions, mapping = build_protocol_transactions(
        [invocation(1, 100, 199)], [packet(110, 10, b"REQUEST_ONLY")]
    )

    assert transactions[0]["transaction_id"] is None
    assert transactions[0]["traffic_status"] == "no_corresponding_traffic"
    assert transactions[0]["request_response_associated"] is False
    assert mapping["inv-1"]["application_messages"] == []
    assert mapping["inv-1"]["wire_evidence_diagnostic"]["association_reason"] == "incomplete_request_response_exchange"


def test_no_wire_packets_in_call_window_is_explicitly_diagnosed() -> None:
    rows = [invocation(1, 100, 199)]
    packets = [
        packet(300, 10, b"Q"),
        packet(310, 20, b"A", response=True),
    ]
    records, _ = build_protocol_transactions(rows, packets)
    diagnostic = records[0]["wire_evidence_diagnostic"]
    assert records[0]["traffic_status"] == "no_corresponding_traffic"
    assert diagnostic["endpoint_packet_count"] == 2
    assert diagnostic["invocation_window_packet_count"] == 0
    assert diagnostic["association_reason"] == "no_endpoint_packets_in_invocation_window"
    assert diagnostic["nearest_endpoint_packet_before_ns"] is None
    assert diagnostic["nearest_endpoint_packet_after_ns"] == 300
    assert diagnostic["window_end_gap_ns"] == 101


def test_capture_metadata_is_preserved_without_turning_a_file_into_match() -> None:
    row = invocation(1, 100, 199)
    row["api"] = "cnc_setpath"
    records, _ = build_protocol_transactions(
        [row], [],
        capture_metadata={
            "cnc_setpath": {
                "api_pcap_file": "by_api/batch_cnc_setpath.pcap",
                "api_pcap_payload_count": 0,
                "capture_source": "top_level_plus_runtime_per_api",
            }
        },
    )
    diagnostic = records[0]["wire_evidence_diagnostic"]
    assert records[0]["traffic_status"] == "no_corresponding_traffic"
    assert diagnostic["api_pcap_file"].endswith("cnc_setpath.pcap")
    assert diagnostic["api_pcap_payload_count"] == 0
    assert diagnostic["capture_source"] == "top_level_plus_runtime_per_api"


def test_segmented_download_request_is_explained_by_lifecycle_end() -> None:
    start = invocation(1, 100, 109)
    start["api"] = "cnc_dwnstart3"
    chunk = invocation(2, 110, 119)
    chunk["api"] = "cnc_download3"
    end = invocation(3, 120, 139)
    end["api"] = "cnc_dwnend3"
    packets = [
        packet(105, 10, b"START"), packet(106, 20, b"OK", response=True),
        packet(115, 30, b"CHUNK"),
        packet(125, 40, b"END"), packet(126, 50, b"OK", response=True),
    ]
    records, _mapping = build_protocol_transactions([start, chunk, end], packets)
    row = next(item for item in records if item["api"] == "cnc_download3")
    assert row["traffic_status"] == "request_only_segment"
    assert row["traffic_analysis"]["lifecycle_end_api"] == "cnc_dwnend3"
    assert row["traffic_analysis"]["response_expected_in_this_invocation"] is False


def test_request_and_response_on_different_connections_never_match() -> None:
    records, _ = build_protocol_transactions([invocation(1, 100, 199)], [
        packet(110, 10, b"Q", client_port=50001),
        packet(120, 20, b"A", response=True, client_port=50002),
    ])
    assert records[0]["request_response_associated"] is False
    assert records[0]["transaction_id"] is None


def test_multi_connection_invocation_preserves_independent_sequence_spaces() -> None:
    records, _ = build_protocol_transactions([invocation(1, 100, 199)], [
        packet(110, 1000, b"Q1", client_port=50001),
        packet(120, 2000, b"A1", response=True, client_port=50001),
        packet(130, 9000, b"Q2", client_port=50002),
        packet(140, 4000, b"A2", response=True, client_port=50002),
    ])
    row = records[0]
    assert row["request_response_associated"] is True
    assert row["transaction_id"] == "tx-1"
    assert row["request_payload_hex"] is None  # No invented combined TCP stream.
    assert [stream["request_payload_hex"] for stream in row["tcp_streams"]] == [b"Q1".hex(), b"Q2".hex()]
    assert [stream["response_payload_hex"] for stream in row["tcp_streams"]] == [b"A1".hex(), b"A2".hex()]
    assert len(row["application_messages"]) == 4


def runtime_capture_source() -> str:
    state = WorkflowState(
        request=TaskRequest(
            description="collect coordinate traffic",
            task_id="dataset-quality-capture",
            target_environment="ncguide-generated-cpp",
        )
    )
    step = PlanStep(
        "S001", "during", "read status", "ReadRunStatus", {}, protocol_function="cnc_statinfo"
    )
    state.plan = ExecutionPlan(
        plan_id="plan-dataset-quality-capture",
        task_id="dataset-quality-capture",
        scenario_type="coordinate_motion",
        scenario_goal="collect traffic",
        target_environment="ncguide-generated-cpp",
        nc_program_type="test",
        nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name="O1234"),
        steps=[step],
        expected_outputs=[],
        code_spec=CodeSpec(scenario="coordinate_motion", api_contracts=[]),
    )
    return render_fixed_ncguide_cpp_script(state, [step], "O1234\nG90 G54\nM30\n")


def test_runtime_capture_source_generates_ids_timing_phase_and_context() -> None:
    source = runtime_capture_source()

    assert "InvocationId(idx)" in source
    assert "TransactionId(idx)" in source
    assert "start_timestamp_ns" in source
    assert "end_timestamp_ns" in source
    assert "device_context" in source
    assert 'gCurrentPhase = "After"' in source
    assert "cnc_delall" not in source


def test_runtime_capture_source_compiles_offline_when_msvc_is_available() -> None:
    if find_msvc_x86_toolchain() is None:
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "capture_runner.cpp"
        executable = root / "capture_runner.exe"
        source.write_text(runtime_capture_source(), encoding="utf-8")

        result = CompileGeneratedCppTool().invoke(CompileCppInput(source, executable, root))

        assert result.success, result.stderr or result.stdout


def test_coordinate_runtime_render_validate_compile_smoke() -> None:
    if find_msvc_x86_toolchain() is None:
        return
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        harness = root / "focas_harness.cpp"
        fragment = root / "scenario_fragment.inc"
        output = root / "api_script.cpp"
        executable = root / "api_script.exe"
        source = runtime_capture_source()
        harness.write_text(source, encoding="utf-8")
        fragment.write_text(
            build_scenario_fragment("    // deterministic coordinate-motion capture\n"),
            encoding="utf-8",
        )

        rendered = RenderFocasCppTool().invoke(RenderFocasCppInput(harness, fragment, output))
        assert rendered.success, rendered.error
        validation = ValidateFocasCppTool().invoke(
            ValidateFocasCppInput(
                cpp_path=output,
                fragment_path=fragment,
                nc_program="O1234\nG90 G54\nM30\n",
                executable_steps=(
                    PlanStep("UPLOAD", "before", action="upload motion program", interface_name="UploadProgram",
                             protocol_function="cnc_dwnstart3/cnc_download3/cnc_dwnend3"),
                    PlanStep("START", "during", action="start planned motion", interface_name="StartProgram",
                             protocol_function="cnc_statinfo"),
                    PlanStep(
                        "S001",
                        "during",
                        action="read status",
                        interface_name="ReadRunStatus",
                        protocol_function="cnc_statinfo",
                    ),
                ),
                allow_delete_all_programs=False,
                require_official_focas_header=True,
                require_real_pcap_capture=True,
                code_spec=CodeSpec(scenario="coordinate_motion", api_contracts=[]),
            )
        )
        assert validation.success, validation.errors
        compiled = CompileGeneratedCppTool().invoke(CompileCppInput(output, executable, root))
        assert compiled.success, compiled.stderr or compiled.stdout


def test_collector_writes_linked_protocol_transaction_sidecar() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        execution = Path(temporary)
        data = execution / "data"
        data.mkdir()
        (data / "focas_api_input.csv").write_text(
            "index,timestamp,step_id,phase,interface_name,protocol_function,parameters,api_parameter_count,invocation_id,transaction_id,start_timestamp_ns,device_context\n"
            "1,1970-01-01T00:00:01.000000000,S1,Before,FOCAS,cnc_statinfo,type=-1,2,inv-1,tx-1,1000000000,target=NCGuide;run_state=idle\n",
            encoding="utf-8",
        )
        (data / "focas_api_output.csv").write_text(
            "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count,invocation_id,transaction_id,end_timestamp_ns\n"
            "1,1970-01-01T00:00:01.300000000,S1,cnc_statinfo,0,EW_OK,run=0;pcap_capture_enabled=true,2,inv-1,tx-1,1300000000\n",
            encoding="utf-8",
        )
        write_pcap(
            data / "focas_capture.pcap",
            [
                (1_100_000_000, ethernet_tcp_frame(b"REQ", 10)),
                (1_200_000_000, ethernet_tcp_frame(b"RES", 20, response=True)),
            ],
        )

        collected = read_cpp_csv_logs("task", execution)
        sidecar = [
            json.loads(line)
            for line in (execution / "protocol_transactions.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        assert collected.api_logs[0].invocation_id == "inv-1"
        assert collected.api_logs[0].transaction_id == "tx-1"
        assert collected.api_logs[0].phase == "Before"
        assert sidecar[0]["request_payload_hex"] == b"REQ".hex()
        assert sidecar[0]["response_payload_hex"] == b"RES".hex()
        stem = "19700101_focas_statinfo"
        by_api = data / "by_api"
        with (by_api / f"{stem}_input.csv").open(encoding="utf-8-sig", newline="") as file:
            input_row = next(csv.DictReader(file))
        with (by_api / f"{stem}_output.csv").open(encoding="utf-8-sig", newline="") as file:
            output_row = next(csv.DictReader(file))
        assert input_row["parameters"] == "type=-1"
        assert output_row["data"] == "run=0"
        assert input_row["invocation_id"] == output_row["invocation_id"] == "inv-1"
        assert input_row["transaction_id"] == output_row["transaction_id"] == "tx-1"
        assert b"REQ" in (by_api / f"{stem}.pcap").read_bytes()
        assert b"RES" in (by_api / f"{stem}.pcap").read_bytes()
