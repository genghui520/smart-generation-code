from __future__ import annotations

import json

from smart_traffic_agent.diversity_audit import build_diversity_audit, load_recent_diversity_feedback
from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.parameter_coverage import attach_parameter_contract, audit_parameter_coverage


def _plan(function: str, *, minimum: int = 1, response_varies: bool = False):
    step = PlanStep(
        "sample", "during", "capture", function, protocol_function=function,
        parameters={"parameter_coverage_goal": {
            "minimum_distinct": minimum,
            "coverage_scope": "documented input domain",
            "evidence": "unit test source",
        }},
    )
    plan = ExecutionPlan(
        plan_id="p", task_id="t", scenario_type="capture", scenario_goal="capture",
        target_environment="NCGuide", nc_program_type="none", nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(""), steps=[step], expected_outputs=[],
        rag_context={"function_coverage_manifest": {"enabled": True,
            "selected_batch": [{"function": function, "coverage_role": "target"}] }},
    )
    attach_parameter_contract(plan)
    rule = plan.rag_context["parameter_coverage_contract"]["functions"][function]
    rule["response_state_diversity_applicable"] = response_varies
    return plan


def _log(function: str, *, number: int, output: str, request: str, response: str,
         invocation: str) -> ApiCallLog:
    return ApiCallLog(
        timestamp="t", task_id="t", step_id="S001", phase="during", interface_name=function,
        protocol_function=function,
        input_parameters={"raw": f"FlibHndl=1;number={number};axis=0;length=8"},
        status_code=0, response={
            "data": output,
            "traffic_analysis": {"tcp_streams": [{
                "request_payload_hex": request.encode().hex(),
                "response_payload_hex": response.encode().hex(),
            }]},
        }, invocation_id=invocation, traffic_status="matched",
    )


def test_ordinary_api_counts_distinct_valid_inputs_and_wire_payloads():
    plan = _plan("cnc_rdparam", minimum=2, response_varies=True)
    logs = [
        _log("cnc_rdparam", number=1, output="value=10", request="req1", response="res1", invocation="a"),
        _log("cnc_rdparam", number=2, output="value=20", request="req2", response="res2", invocation="b"),
    ]
    parameter_report = audit_parameter_coverage(plan, logs)
    report = build_diversity_audit(plan, logs, parameter_report)
    row = report["functions"]["cnc_rdparam"]
    assert row["input_diversity"]["eligible_distinct_configurations"] == 2
    assert row["response_diversity"]["distinct_effective_responses"] == 2
    assert row["application_traffic"]["distinct_request_payloads"] == 2
    assert row["application_traffic"]["distinct_response_payloads"] == 2


def test_planned_configuration_target_is_clamped_by_capture_budget():
    plan = _plan("cnc_rdparam", minimum=1)
    plan.steps[0].parameters["parameter_generation"] = [{
        "name": "number", "kind": "enum", "values": [1, 2, 3, 4],
    }]
    plan.rag_context["quality_targets"] = {"max_configurations": 40}
    plan.rag_context["task_permissions"] = {"capture_samples": 3}
    row = attach_parameter_contract(plan)["functions"]["cnc_rdparam"]
    assert row["planned_configuration_target"] == 4
    assert row["configuration_budget"] == 3


def test_saved_plan_rebuilds_legacy_parameter_contract(tmp_path):
    from dataclasses import asdict
    from smart_traffic_agent.offline_audit import load_saved_plan
    from smart_traffic_agent.utils import write_json

    plan = _plan("cnc_skip", minimum=1)
    plan.rag_context["parameter_coverage_contract"]["functions"]["cnc_skip"][
        "variation_requirements"
    ] = {"variation_class": "legacy_wrong_classification"}
    path = tmp_path / "plan.json"
    write_json(path, asdict(plan))
    loaded = load_saved_plan(path)
    rule = loaded.rag_context["parameter_coverage_contract"]["functions"]["cnc_skip"]
    assert rule["variation_requirements"]["variation_class"] == "input_only"
    assert rule["input_diversity_required"] is True


def test_request_only_session_segments_count_only_with_complete_session_proof():
    from smart_traffic_agent.execution_contracts import API_TRAFFIC_SESSION_RULES

    plan = _plan("cnc_download3", minimum=1)
    logs = []
    for index, data in enumerate(("program-a", "program-b"), 1):
        call_id = f"call-{index}"
        session = {
            "kind": "session_segment", "api_name": "cnc_download3", "call_id": call_id,
            "session_type": API_TRAFFIC_SESSION_RULES["cnc_download3"]["session_type"],
            "wire_direction": "request", "session_id": "start->end",
            "session_start_call_id": "start", "session_end_call_id": "end",
            "wire_segment_verified": True, "wire_evidence_source": "pcap_tcp_reassembly",
            "wire_stream_count": 1, "session_start_result": 0, "session_end_result": 0,
            "session_completion_status": "completed_successfully", "api_return_code": 0,
            "call_association_verified": True,
        }
        log = ApiCallLog(
            timestamp="t", task_id="t", step_id="S001", phase="during",
            interface_name="cnc_download3", protocol_function="cnc_download3",
            input_parameters={"raw": f"FlibHndl=1;length={len(data)};data={data}"},
            status_code=0, response={"data": "ack", "session_traffic": session},
            invocation_id=call_id, traffic_status="request_only_segment",
        )
        log.response["traffic_analysis"] = {"tcp_streams": [{
            "request_payload_hex": data.encode().hex(), "response_payload_hex": None,
        }]}
        logs.append(log)
    report = build_diversity_audit(plan, logs, audit_parameter_coverage(plan, logs))
    row = report["functions"]["cnc_download3"]
    assert row["sessions"]["successful_complete_sessions"] == 1
    assert row["sessions"]["request_only_segment_payload_variants"] == 2
    assert row["application_traffic"]["distinct_request_payloads"] == 2


def test_response_and_mixed_sessions_keep_separate_segment_variation_counts():
    from smart_traffic_agent.execution_contracts import API_TRAFFIC_SESSION_RULES

    plan = _plan("cnc_download3", minimum=1)
    logs = []
    for index, direction in enumerate(("response", "both"), 1):
        call_id = f"seg-{index}"
        session = {
            "kind": "session_segment", "api_name": "cnc_download3", "call_id": call_id,
            "session_type": API_TRAFFIC_SESSION_RULES["cnc_download3"]["session_type"],
            "wire_direction": direction, "session_id": f"start-{index}->end-{index}",
            "session_start_call_id": f"start-{index}", "session_end_call_id": f"end-{index}",
            "wire_segment_verified": True, "wire_evidence_source": "pcap_tcp_reassembly",
            "wire_stream_count": 1, "session_start_result": 0, "session_end_result": 0,
            "session_completion_status": "completed_successfully", "api_return_code": 0,
            "call_association_verified": True,
        }
        log = ApiCallLog(
            timestamp="t", task_id="t", step_id="S001", phase="during",
            interface_name="cnc_download3", protocol_function="cnc_download3",
            input_parameters={"raw": f"FlibHndl=1;length=4;data=case-{index}"},
            status_code=0, response={"data": "ack", "session_traffic": session,
                "traffic_analysis": {"tcp_streams": [{
                    "request_payload_hex": f"req-{index}".encode().hex() if direction == "both" else None,
                    "response_payload_hex": f"res-{index}".encode().hex(),
                }]}},
            invocation_id=call_id, traffic_status="matched",
        )
        logs.append(log)
    report = build_diversity_audit(plan, logs, audit_parameter_coverage(plan, logs))
    row = report["functions"]["cnc_download3"]
    assert row["sessions"]["response_only_segment_payload_variants"] == 1
    assert row["sessions"]["mixed_session_request_payload_variants"] == 1
    assert row["sessions"]["mixed_session_response_payload_variants"] == 1
    assert row["sessions"]["successful_complete_sessions"] == 2


def test_history_feedback_is_recent_and_excludes_current_run(tmp_path):
    old = tmp_path / "20261020" / "batch_001" / "execution"
    current = tmp_path / "20261024" / "batch_001" / "execution"
    for execution, config in ((old, {"number": "1"}), (current, {"number": "9"})):
        execution.mkdir(parents=True)
        (execution / "offline_audit.json").write_text(json.dumps({"diversity": {"functions": {
            "cnc_rdparam": {
                "input_diversity": {"observed_input_configurations": [config]},
                "application_traffic": {"request_payload_sha256s": ["abc"],
                                         "response_payload_sha256s": ["def"]},
                "incremental_recommendation": {"next_action": "try_unseen_documented_input_configuration"},
            },
        }}}), encoding="utf-8")
    feedback = load_recent_diversity_feedback(
        tmp_path, ["cnc_rdparam"], exclude_run=current.parent.parent,
    )
    assert feedback["cnc_rdparam"]["observed_input_configurations"] == [{"number": "1"}]
    assert feedback["cnc_rdparam"]["request_payload_sha256s"] == ["abc"]

