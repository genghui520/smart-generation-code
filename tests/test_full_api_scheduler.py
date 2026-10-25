from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from smart_traffic_agent import desktop_backend as backend
from smart_traffic_agent.function_coverage import load_function_coverage_state
from smart_traffic_agent.models import ApiCallLog, ExecutionResult, WorkflowState
from smart_traffic_agent.utils import write_json
from smart_traffic_agent.workflow import TrafficGenerationWorkflow


class RecordingWorkflow:
    actions: dict[str, dict] = {}
    calls: list[str] = []
    requests: list[dict] = []
    constructor_kwargs: list[dict] = []

    def __init__(self, _knowledge_base, **kwargs):
        self.coverage_path = kwargs["coverage_state_path"]
        self.constructor_kwargs.append(kwargs)

    def run(self, request, folder):
        target = request.permissions["coverage_current_batch_functions"][0]
        self.calls.append(target)
        self.requests.append({
            "target": target,
            "coverage_current_batch_functions": request.permissions["coverage_current_batch_functions"],
        })
        folder.mkdir(parents=True, exist_ok=True)
        action = self.actions.get(target, {"kind": "success"})
        if action.get("kind") == "raise":
            raise RuntimeError(action.get("error", "environment preflight failed"))

        state = WorkflowState(
            request=request,
            repair_attempts=int(action.get("repair_attempts", 0)),
            repair_history=list(action.get("repair_history", [])),
            errors=list(action.get("errors", [])),
        )
        if action.get("kind") == "failure":
            return state

        success = action.get("kind") != "partial"
        traffic_status = "matched" if success else "unmatched"
        log = ApiCallLog(
            timestamp="test", task_id=request.task_id, step_id="S001", phase="during",
            interface_name="TestCall", protocol_function=target,
            input_parameters={"value": 1}, status_code=0 if success else 9,
            response={"data": "ok" if success else "no matched traffic"},
            invocation_id=f"{request.task_id}-call", traffic_status=traffic_status,
        )
        state.result = ExecutionResult(
            task_id=request.task_id, success=True, api_logs=[log], capture_events=[],
            output_dir=folder, collection_status="collected" if success else "partial",
        )
        (folder / "focas_capture.pcap").write_bytes(b"pcap evidence")
        execution = folder / "execution"
        execution.mkdir(exist_ok=True)
        (execution / "api_logs.jsonl").write_text(
            json.dumps({
                "protocol_function": target,
                "status_code": log.status_code,
                "error": None,
                "traffic_status": traffic_status,
                "invocation_id": log.invocation_id,
            }) + "\n",
            encoding="utf-8",
        )
        support = list(action.get("support_functions", []))
        if support:
            rows = []
            for name in support:
                rows.append({
                    "protocol_function": name, "status_code": 0, "error": None,
                    "traffic_status": "matched", "invocation_id": f"support-{name}",
                })
            with (execution / "api_logs.jsonl").open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
        write_json(folder / "summary.json", {
            "success": success,
            "function_coverage": {"support_functions": support},
        })

        coverage = load_function_coverage_state(self.coverage_path)
        record = {
            "function": target,
            "coverage_role": "target",
            "terminal": success,
            "status": "covered_success" if success else "attempted_no_traffic",
            "traffic_matched": success,
        }
        coverage.setdefault("functions", {})[target] = record
        write_json(self.coverage_path, coverage)
        return state


def setup_run(tmp_path, monkeypatch, names):
    project = tmp_path / "project"
    manifest_path = project / "rag_indexes" / "focas" / "function_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    rows = [
        {"function": name, "category": "test", "description": name,
         "symbol_candidates": [name]}
        for name in names
    ]
    manifest_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sdk = project / "sdk"
    sdk.mkdir()
    (sdk / "Fwlib32.h").write_text("\n".join(f"short {name}(void);" for name in names), encoding="ascii")
    out = project / "runs" / "current"
    out.mkdir(parents=True)
    config = backend.defaults()
    config.update(
        capture_mode="all_apis", selected=[], selected_apis=list(names),
        header=str(sdk), output=str(out.parent), resume_from="",
    )
    monkeypatch.setattr(backend, "ROOT", project)
    monkeypatch.setattr(backend, "function_manifest", lambda: rows)
    monkeypatch.setattr(backend, "selected_api_names", lambda _config: list(names))
    monkeypatch.setattr(backend, "require_capture_adapter", lambda _config: None)
    monkeypatch.setattr(backend, "make_llm", lambda _config: None)
    monkeypatch.setattr(backend, "manual_operator_action_timed_out", lambda _folder: False)
    monkeypatch.setattr(backend, "simulator_is_idle_after_failure", lambda _config: True)
    monkeypatch.setattr(backend, "write_selected_api_coverage_progress", lambda *a, **k: {})
    monkeypatch.setattr(
        backend, "render_full_api_prompt",
        lambda _config, batch_targets=None: f"capture targets: {','.join(batch_targets or [])}",
    )
    monkeypatch.setattr(
        "smart_traffic_agent.diversity_audit.load_recent_diversity_feedback",
        lambda *_a, **_k: [],
    )
    RecordingWorkflow.actions = {}
    RecordingWorkflow.calls = []
    RecordingWorkflow.requests = []
    RecordingWorkflow.constructor_kwargs = []
    events = []
    return project, out, config, events


def run_with_recording_workflow(config, out, monkeypatch, events):
    monkeypatch.setattr("smart_traffic_agent.workflow.TrafficGenerationWorkflow", RecordingWorkflow)
    return backend.run_full_api_capture(config, out, None, lambda name, data: events.append((name, data)))


def test_single_target_dispatch_repairs_are_per_workflow_and_failures_move_on(tmp_path, monkeypatch):
    names = ["cnc_test_a", "cnc_test_b", "cnc_test_c", "cnc_test_d"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    RecordingWorkflow.actions = {
        "cnc_test_a": {"kind": "failure", "errors": ["historical target should not run"]},
        "cnc_test_b": {
            "kind": "success", "repair_attempts": 1,
            "repair_history": [{"repair_stage": "repair_code", "attempt": 1, "code_version": "v2"}],
        },
        "cnc_test_c": {"kind": "failure", "errors": ["repair budget exhausted"], "repair_attempts": 2},
        "cnc_test_d": {"kind": "success", "repair_attempts": 0},
    }
    state_path = out / "full_api_coverage_state.json"
    write_json(state_path, {"schema_version": 1, "functions": {
        "cnc_test_a": {"function": "cnc_test_a", "status": "covered_success", "traffic_matched": True},
    }, "runs": []})

    run_with_recording_workflow(config, out, monkeypatch, events)

    assert Counter(RecordingWorkflow.calls) == {"cnc_test_b": 1, "cnc_test_c": 1, "cnc_test_d": 1}
    assert backend.API_CAPTURE_BATCH_SIZE == 1
    assert all(len(call["coverage_current_batch_functions"]) == 1 for call in RecordingWorkflow.requests)
    assert all(call["coverage_batch_size"] == 1 for call in RecordingWorkflow.constructor_kwargs)
    assert all("max_repair_attempts" not in call for call in RecordingWorkflow.constructor_kwargs)
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["targets"]["cnc_test_a"]["status"] == "history_skipped"
    assert scheduler["targets"]["cnc_test_b"]["repair_attempts"] == 1
    assert scheduler["targets"]["cnc_test_b"]["repair_history"][0]["repair_stage"] == "repair_code"
    assert scheduler["targets"]["cnc_test_c"]["status"] == "failed"
    assert scheduler["targets"]["cnc_test_d"]["repair_attempts"] == 0
    assert scheduler["summary"]["duplicate_target_attempt_count"] == 0
    assert scheduler["summary"]["historical_skip"] == 1
    assert scheduler["summary"]["failure"] == 1
    assert scheduler["summary"]["success"] == 2


def test_two_consecutive_failures_are_recorded_without_repeating_targets(tmp_path, monkeypatch):
    names = ["cnc_test_a", "cnc_test_b"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    RecordingWorkflow.actions = {
        name: {"kind": "failure", "errors": ["compile failed"]} for name in names
    }

    assert run_with_recording_workflow(config, out, monkeypatch, events) == 0

    assert RecordingWorkflow.calls == names
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["summary"]["failure"] == 2
    assert scheduler["summary"]["unprocessed"] == 0
    assert scheduler["summary"]["target_execution_counts"] == {name: 1 for name in names}
    assert any(event == "coverage_summary" for event, _ in events)

    run_with_recording_workflow(config, out, monkeypatch, events)
    assert RecordingWorkflow.calls == names
    resumed = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert resumed["summary"]["duplicate_target_attempt_count"] == 0


def test_partial_history_is_retried_but_successful_local_history_is_skipped(tmp_path, monkeypatch):
    names = ["cnc_test_partial", "cnc_settimeout"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    write_json(out.parent / "prior" / "full_api_coverage_state.json", {
        "schema_version": 1,
        "functions": {
            "cnc_test_partial": {"function": "cnc_test_partial", "status": "attempted_no_traffic"},
            "cnc_settimeout": {"function": "cnc_settimeout", "status": "local_success"},
        },
        "runs": [],
    })
    RecordingWorkflow.actions = {"cnc_test_partial": {"kind": "success"}}

    run_with_recording_workflow(config, out, monkeypatch, events)

    assert RecordingWorkflow.calls == ["cnc_test_partial"]
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["targets"]["cnc_settimeout"]["status"] == "history_skipped"
    assert scheduler["targets"]["cnc_settimeout"]["attempt_count"] == 0


def test_verified_auxiliary_traffic_can_satisfy_target_without_counting_a_bare_call(tmp_path, monkeypatch):
    names = ["cnc_test_target", "cnc_test_support"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    prior_batch = out.parent / "prior" / "batch_001"
    execution = prior_batch / "execution"
    execution.mkdir(parents=True)
    write_json(prior_batch / "summary.json", {
        "function_coverage": {"support_functions": ["cnc_test_support"]},
    })
    (execution / "api_logs.jsonl").write_text(json.dumps({
        "protocol_function": "cnc_test_support", "status_code": 0,
        "traffic_status": "matched", "invocation_id": "wire-1",
    }) + "\n", encoding="utf-8")
    RecordingWorkflow.actions = {"cnc_test_target": {"kind": "success"}}

    run_with_recording_workflow(config, out, monkeypatch, events)

    assert RecordingWorkflow.calls == ["cnc_test_target"]
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["targets"]["cnc_test_support"]["status"] == "history_skipped"
    assert scheduler["targets"]["cnc_test_support"]["skip_source"] == "verified_auxiliary_traffic"
    assert scheduler["summary"]["target_execution_counts"]["cnc_test_support"] == 0


def test_resume_skips_completed_and_interrupted_attempts_and_continues_pending(tmp_path, monkeypatch):
    names = ["cnc_test_a", "cnc_test_b", "cnc_test_c"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    write_json(out / backend.FULL_API_SCHEDULER_FILE, {
        "schema_version": backend.FULL_API_SCHEDULER_SCHEMA,
        "selected_api_functions": names,
        "attempted_this_run": ["cnc_test_a", "cnc_test_b"],
        "targets": {
            "cnc_test_a": {"status": "success", "attempt_count": 1},
            "cnc_test_b": {"status": "running", "attempt_count": 1},
            "cnc_test_c": {"status": "pending", "attempt_count": 0},
        },
        "auxiliary_coverage": {},
    })
    RecordingWorkflow.actions = {"cnc_test_c": {"kind": "success"}}

    run_with_recording_workflow(config, out, monkeypatch, events)

    assert RecordingWorkflow.calls == ["cnc_test_c"]
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["targets"]["cnc_test_a"]["status"] == "success"
    assert scheduler["targets"]["cnc_test_b"]["status"] == "failed"
    assert scheduler["targets"]["cnc_test_b"]["errors"] == ["INTERRUPTED_DURING_WORKFLOW"]
    assert scheduler["targets"]["cnc_test_c"]["attempt_count"] == 1


def test_offline_diversity_partial_does_not_retry_and_prerequisite_is_blocked(tmp_path, monkeypatch):
    names = ["cnc_test_partial", "cnc_test_blocked", "cnc_test_next"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    RecordingWorkflow.actions = {
        "cnc_test_partial": {"kind": "partial", "repair_attempts": 0},
        "cnc_test_blocked": {"kind": "failure", "errors": ["API_PREREQUISITE_BLOCKED: option unavailable"]},
        "cnc_test_next": {"kind": "success"},
    }

    run_with_recording_workflow(config, out, monkeypatch, events)

    assert Counter(RecordingWorkflow.calls) == Counter(names)
    scheduler = json.loads((out / backend.FULL_API_SCHEDULER_FILE).read_text(encoding="utf-8"))
    assert scheduler["targets"]["cnc_test_partial"]["status"] == "partial"
    assert scheduler["targets"]["cnc_test_partial"]["repair_attempts"] == 0
    assert scheduler["targets"]["cnc_test_blocked"]["status"] == "blocked"
    assert scheduler["targets"]["cnc_test_next"]["status"] == "success"


def test_manual_single_api_keeps_default_repair_budget(tmp_path, monkeypatch):
    names = ["cnc_test_manual"]
    _, out, config, events = setup_run(tmp_path, monkeypatch, names)
    config.update(capture_mode="single_api")
    RecordingWorkflow.actions = {"cnc_test_manual": {"kind": "failure", "errors": ["compile failed"]}}

    with pytest.raises(RuntimeError, match="单 API 采集失败"):
        run_with_recording_workflow(config, out, monkeypatch, events)

    assert RecordingWorkflow.calls == names
    assert "max_repair_attempts" not in RecordingWorkflow.constructor_kwargs[0]
    assert TrafficGenerationWorkflow.max_repair_attempts == 3
    assert TrafficGenerationWorkflow.__init__.__defaults__[-1] == 3
