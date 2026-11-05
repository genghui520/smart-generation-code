from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smart_traffic_agent.agents.router import valid_next_stages
from smart_traffic_agent.desktop_backend import refresh_overall_collection_csv
from smart_traffic_agent.execution_contracts import API_TRAFFIC_SESSION_RULES
from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, ExecutionResult, NcProgramSpec, PlanStep, TaskRequest, WorkflowState
from smart_traffic_agent.parameter_coverage import attach_parameter_contract, audit_parameter_coverage
from smart_traffic_agent.function_coverage import (
    build_function_coverage_metrics,
    load_function_coverage_state,
    merge_workflow_function_coverage_state,
    summarize_function_coverage_state,
)
from smart_traffic_agent.traffic_transactions import (
    TcpPayload, annotate_session_transactions, build_protocol_transactions,
    valid_session_segment_evidence, write_jsonl,
)
from smart_traffic_agent.offline_audit import reaudit_saved_execution
from smart_traffic_agent.agents.executor import classify_collection_status


def _plan(function: str, minimum: int = 3) -> ExecutionPlan:
    step = PlanStep(
        "sample", "during", "capture", function, protocol_function=function,
        parameters={"parameter_coverage_goal": {"minimum_distinct": minimum}},
    )
    plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="capture", scenario_goal="capture",
        target_environment="NCGuide", nc_program_type="none", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(""), steps=[step], expected_outputs=[],
        rag_context={"function_coverage_manifest": {"enabled": True,
            "selected_batch": [{"function": function, "coverage_role": "target"}]}},
    )
    attach_parameter_contract(plan)
    return plan


def _session_log(function: str, value: str, *, direction: str = "request", end_rc: int = 0,
                 api_rc: int = 0, wire: bool = True, call_id: str | None = None,
                 session_id: str | None = None, traffic_status: str | None = None) -> ApiCallLog:
    call_id = call_id or f"call-{value}"
    rule = API_TRAFFIC_SESSION_RULES[function]
    start_id, end_id = "start-call", "end-call"
    session_id = session_id or f"{start_id}->{end_id}"
    analysis = {
        "kind": "session_segment", "api_name": function, "call_id": call_id,
        "session_type": rule["session_type"], "wire_direction": direction,
        "session_id": session_id, "session_start_call_id": start_id,
        "session_end_call_id": end_id, "wire_segment_verified": wire,
        "wire_evidence_source": "pcap_tcp_reassembly" if wire else None,
        "wire_stream_count": 1 if wire else 0,
        "session_start_result": 0, "session_end_result": end_rc,
        "session_completion_status": "completed_successfully" if end_rc == 0 else "failed",
        "api_return_code": api_rc, "call_association_verified": True,
    }
    return ApiCallLog(
        timestamp="t", task_id="t", step_id="S", phase="during", interface_name=function,
        protocol_function=function,
        input_parameters={"raw": f"FlibHndl=32769;length=64;data={value}"},
        status_code=api_rc, response={"data": "ack", "session_traffic": analysis},
        invocation_id=call_id, error="EW_DATA" if api_rc else None,
        traffic_status=traffic_status or ("request_only_segment" if direction == "request" else "matched"),
    )


def _invocation(call_id: str, api: str, start: int, end: int, code: int = 0) -> dict:
    return {
        "invocation_id": call_id, "transaction_id": f"tx-{call_id}", "api": api,
        "input_parameters": "FlibHndl=32769;type=0" if "start" in api else "FlibHndl=32769;length=8;data=payload",
        "return_code": code, "start_timestamp_ns": start, "end_timestamp_ns": end,
    }


def _packet(timestamp: int, payload: bytes, *, response: bool = False) -> TcpPayload:
    return TcpPayload(
        timestamp_ns=timestamp, source_ip="127.0.0.1", source_port=8193 if response else 50000,
        destination_ip="127.0.0.1", destination_port=50000 if response else 8193,
        sequence=timestamp * 10, payload=payload,
    )


def test_transaction_mapper_requires_pcap_call_and_successful_session_evidence():
    rows = [
        _invocation("start", "cnc_dwnstart3", 100, 109),
        _invocation("chunk", "cnc_download3", 110, 119),
        _invocation("end", "cnc_dwnend3", 120, 129),
    ]
    packets = [
        _packet(102, b"START"), _packet(103, b"OK", response=True),
        _packet(112, b"CHUNK"),
        _packet(122, b"END"), _packet(123, b"OK", response=True),
    ]
    records, _ = build_protocol_transactions(rows, packets)
    segment = next(row for row in records if row["api"] == "cnc_download3")
    assert segment["traffic_status"] == "request_only_segment"
    assert valid_session_segment_evidence(segment)

    for row in records:
        row.pop("call_association_verified", None)
    annotate_session_transactions(records)
    assert valid_session_segment_evidence(next(row for row in records if row["api"] == "cnc_download3"))

    duplicated = [dict(row) for row in records]
    duplicated[1]["candidate_transaction_id"] = duplicated[0]["candidate_transaction_id"]
    annotate_session_transactions(duplicated)
    assert not valid_session_segment_evidence(next(row for row in duplicated if row["api"] == "cnc_download3"))

    failed_rows = [dict(row) for row in rows]
    failed_rows[-1]["return_code"] = 5
    failed, _ = build_protocol_transactions(failed_rows, packets)
    failed_segment = next(row for row in failed if row["api"] == "cnc_download3")
    assert failed_segment["session_traffic"]["session_completion_status"] == "failed"
    assert not valid_session_segment_evidence(failed_segment)

    no_wire, _ = build_protocol_transactions(rows, [])
    no_wire_segment = next(row for row in no_wire if row["api"] == "cnc_download3")
    assert not valid_session_segment_evidence(no_wire_segment)


def test_ordinary_request_response_coverage_is_unchanged():
    plan = _plan("cnc_rdparam", minimum=2)
    rows = [
        ApiCallLog("t", "t", "S", "during", "cnc_rdparam", {"raw": f"FlibHndl=1;number={n};axis=0;length=8"},
                   0, {"data": "ok"}, protocol_function="cnc_rdparam", invocation_id=f"ordinary-{n}", traffic_status="matched")
        for n in (1, 2)
    ]
    report = audit_parameter_coverage(plan, rows)["functions"]["cnc_rdparam"]
    assert report["matched_distinct_input_combinations"] == 2
    assert report["eligible_distinct_input_combinations"] == 2


def test_request_segment_with_successful_session_is_eligible():
    log = _session_log("cnc_download3", "A")
    assert valid_session_segment_evidence(log)
    row = audit_parameter_coverage(_plan("cnc_download3"), [log])["functions"]["cnc_download3"]
    assert row["eligible_distinct_input_combinations"] == 1
    assert row["matched_distinct_input_combinations"] == 0
    state = WorkflowState(request=TaskRequest(
        "capture", "t", permissions={"required_api_functions": ["cnc_download3"]}
    ))
    assert classify_collection_status(state, [log], True)[0] == "collected"


def test_response_segment_with_successful_session_is_eligible():
    log = _session_log("cnc_upload3", "A", direction="response")
    assert valid_session_segment_evidence(log)
    row = audit_parameter_coverage(_plan("cnc_upload3"), [log])["functions"]["cnc_upload3"]
    assert row["eligible_distinct_input_combinations"] == 1
    assert row["matched_distinct_input_combinations"] == 1


def test_failed_session_end_invalidates_segment_coverage():
    log = _session_log("cnc_download3", "A", end_rc=5)
    assert not valid_session_segment_evidence(log)
    assert audit_parameter_coverage(_plan("cnc_download3"), [log])["functions"]["cnc_download3"]["eligible_distinct_input_combinations"] == 0


def test_failed_segment_call_is_not_eligible():
    log = _session_log("cnc_download3", "A", api_rc=5)
    assert not valid_session_segment_evidence(log)


def test_segment_without_pcap_evidence_is_not_eligible():
    log = _session_log("cnc_download3", "A", wire=False)
    assert not valid_session_segment_evidence(log)


@pytest.mark.parametrize("field,value", [("call_id", "other-call"), ("session_id", "unrelated-session")])
def test_bad_call_or_session_association_is_not_eligible(field, value):
    log = _session_log("cnc_download3", "A")
    log.response["session_traffic"][field] = value
    assert not valid_session_segment_evidence(log)


def test_duplicate_segment_inputs_are_deduplicated():
    plan = _plan("cnc_download3", minimum=2)
    logs = [_session_log("cnc_download3", "same", call_id=f"call-{i}") for i in range(4)]
    row = audit_parameter_coverage(plan, logs)["functions"]["cnc_download3"]
    assert row["eligible_distinct_input_combinations"] == 1
    assert not row["passed"]


def test_four_wire_valid_inputs_meet_required_three_without_fake_responses():
    plan = _plan("cnc_download3", minimum=3)
    logs = [_session_log("cnc_download3", f"chunk-{i}", call_id=f"call-{i}") for i in range(4)]
    row = audit_parameter_coverage(plan, logs)["functions"]["cnc_download3"]
    assert row["eligible_distinct_input_combinations"] == 4
    assert row["matched_distinct_input_combinations"] == 0
    assert row["minimum_distinct"] == 3
    assert row["passed"]


def test_global_function_metrics_keep_session_evidence_after_call_aggregation():
    plan = _plan("cnc_download3", minimum=1)
    log = _session_log("cnc_download3", "A")
    result = ExecutionResult("t", True, [log], [], Path("."))
    metrics = build_function_coverage_metrics(plan, result)
    assert metrics["traffic_successful_target_functions"] == ["cnc_download3"]


def test_global_state_merge_marks_valid_session_target_as_covered(tmp_path):
    plan = _plan("cnc_download3", minimum=1)
    log = _session_log("cnc_download3", "A")
    state = WorkflowState(
        request=TaskRequest("capture", "t"),
        plan=plan,
        result=ExecutionResult("t", True, [log], [], tmp_path),
    )
    path = tmp_path / "coverage.json"

    merge_workflow_function_coverage_state(state, path)

    record = load_function_coverage_state(path)["functions"]["cnc_download3"]
    assert record["status"] == "covered_success"
    assert record["terminal"] is True
    assert record["traffic_matched"] is True
    assert "parameter_coverage_passed" not in record


def test_local_library_api_resolves_without_network_traffic(tmp_path):
    plan = _plan("cnc_settimeout", minimum=1)
    log = ApiCallLog("t", "t", "S", "during", "cnc_settimeout", {"raw": "timeout=10"},
                     0, {}, protocol_function="cnc_settimeout", invocation_id="local-1",
                     traffic_status="no_corresponding_traffic")
    state = WorkflowState(
        request=TaskRequest("capture", "t"), plan=plan,
        result=ExecutionResult("t", True, [log], [], tmp_path),
    )
    status, evidence = classify_collection_status(state, [log], True)
    assert status == "local_success"
    assert evidence["wire_evidence_status"] == "not_required"
    assert evidence["target_apis_without_wire_evidence"] == []

    path = tmp_path / "coverage.json"
    merge_workflow_function_coverage_state(state, path)
    record = load_function_coverage_state(path)["functions"]["cnc_settimeout"]
    assert record["status"] == "local_success"
    assert record["terminal"] is True
    assert record.get("traffic_matched") is not True
    summary = summarize_function_coverage_state(load_function_coverage_state(path))
    assert summary["local_success_function_count"] == 1
    assert summary["unavailable_function_count"] == 0

    mixed_state = WorkflowState(request=TaskRequest(
        "capture", "mixed", permissions={"required_api_functions": ["cnc_settimeout", "cnc_rdparam"]}
    ))
    wire_log = ApiCallLog("t", "mixed", "S2", "during", "cnc_rdparam", {}, 0, {},
                          protocol_function="cnc_rdparam", traffic_status="matched")
    mixed_status, _ = classify_collection_status(mixed_state, [log, wire_log], True)
    assert mixed_status == "collected"


def test_path_selection_apis_are_local_handle_operations(tmp_path):
    for function, raw in (("cnc_getpath", "FlibHndl=1"), ("cnc_setpath", "FlibHndl=1;path_no=0")):
        plan = _plan(function, minimum=1)
        log = ApiCallLog("t", "t", "S", "during", function, {"raw": raw}, 0, {},
                         protocol_function=function, invocation_id=f"{function}-1",
                         traffic_status="no_corresponding_traffic")
        state = WorkflowState(request=TaskRequest("capture", "t"), plan=plan,
                              result=ExecutionResult("t", True, [log], [], tmp_path))
        status, evidence = classify_collection_status(state, [log], True)
        assert status == "local_success"
        assert evidence["wire_evidence_status"] == "not_required"


def test_zero_wire_target_is_partial_not_online_execution_failure():
    state = WorkflowState(request=TaskRequest(
        "capture", "t", permissions={"required_api_functions": ["cnc_rdparam"]}
    ))
    log = ApiCallLog("t", "t", "S", "during", "cnc_rdparam", {}, 0, {},
                     protocol_function="cnc_rdparam", invocation_id="read-1",
                     traffic_status="no_corresponding_traffic")
    status, evidence = classify_collection_status(state, [log], True)
    assert status == "partial"
    assert evidence["wire_evidence_status"] == "insufficient"
    assert evidence["target_apis_without_wire_evidence"] == ["cnc_rdparam"]


def test_offline_audit_can_be_repeated_without_modifying_raw_artifacts(tmp_path):
    execution = tmp_path / "execution"
    (execution / "data").mkdir(parents=True)
    log = _session_log("cnc_download3", "chunk-1")
    raw_logs = execution / "api_logs.jsonl"
    write_jsonl(raw_logs, [asdict(log)])
    raw_pcap = execution / "data" / "focas_capture.pcap"
    raw_pcap.write_bytes(b"original-pcap-bytes")
    (execution / "protocol_transactions.jsonl").write_text("", encoding="utf-8")
    logs_before = raw_logs.read_bytes()
    pcap_before = raw_pcap.read_bytes()

    first = reaudit_saved_execution(execution, _plan("cnc_download3", minimum=1))
    second = reaudit_saved_execution(execution, _plan("cnc_download3", minimum=1))

    assert first["request_only_segments"] == second["request_only_segments"] == 1
    assert first["session_verified_calls"] == second["session_verified_calls"] == 1
    assert raw_logs.read_bytes() == logs_before
    assert raw_pcap.read_bytes() == pcap_before


def test_offline_audit_failure_does_not_change_online_completion():
    plan = _plan("cnc_rdparam")
    state = WorkflowState(request=TaskRequest("capture", "t"), plan=plan)
    state.result = ExecutionResult("t", True, [], [], Path("."),
                                   offline_audit_status="failed")
    assert valid_next_stages(state) == {"complete"}


def test_cp936_csv_is_backed_up_then_incrementally_merged_idempotently(tmp_path):
    ledger = tmp_path / "ledger.csv"
    original_text = "function,collected,status,evidence_runs,latest_status\r\ncnc_rdparam,,, ,\r\n"
    original = original_text.encode("cp936")
    ledger.write_bytes(original)
    execution = tmp_path / "run-001" / "batch_001" / "execution"
    (execution / "data").mkdir(parents=True)
    (execution / "data" / "capture.pcap").write_bytes(b"x" * 32)
    log = {"protocol_function": "cnc_rdparam", "status_code": 0,
           "traffic_status": "matched", "invocation_id": "ordinary-1"}
    (execution / "api_logs.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")
    (execution / "protocol_transactions.jsonl").write_text("", encoding="utf-8")

    refresh_overall_collection_csv(ledger)
    backups = list(tmp_path.glob("ledger.csv.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    with ledger.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["evidence_runs"] == "run-001"
    once = ledger.read_bytes()
    refresh_overall_collection_csv(ledger)
    with ledger.open(encoding="utf-8-sig", newline="") as stream:
        row_again = next(csv.DictReader(stream))
    assert row_again["evidence_runs"] == "run-001"
    assert ledger.read_bytes() == once


def test_global_csv_records_local_success_without_pcap(tmp_path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text("function,collected,status,evidence_runs,latest_status\ncnc_settimeout,,,,\n", encoding="utf-8")
    execution = tmp_path / "local-run" / "batch_001" / "execution"
    execution.mkdir(parents=True)
    log = {"protocol_function": "cnc_settimeout", "status_code": 0,
           "traffic_status": "no_corresponding_traffic", "invocation_id": "local-run-1"}
    (execution / "api_logs.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")

    progress = refresh_overall_collection_csv(ledger)

    with ledger.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["collected"] == "✓"
    assert row["latest_status"] == "local_success"
    assert row["evidence_runs"] == "local-run"
    assert progress["local_success_count"] == 1
    assert progress["traffic_observed_count"] == 0


def test_invalid_csv_bytes_are_rejected_without_modifying_original(tmp_path):
    ledger = tmp_path / "bad.csv"
    original = b"function,collected\r\ncnc_rdparam,\x81\r\n"
    ledger.write_bytes(original)
    with pytest.raises(UnicodeError):
        refresh_overall_collection_csv(ledger)
    assert ledger.read_bytes() == original
    assert not list(tmp_path.glob("bad.csv.bak.*"))
