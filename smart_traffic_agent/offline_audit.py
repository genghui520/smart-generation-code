"""Offline traffic and parameter audit helpers for saved FOCAS executions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ApiCallLog, ExecutionPlan
from .models import NcProgramSpec, PlanStep
from .parameter_coverage import audit_parameter_coverage
from .diversity_audit import build_diversity_audit, load_recent_diversity_feedback
from .traffic_transactions import annotate_session_transactions, valid_session_segment_evidence
from .utils import write_json, write_jsonl


def build_offline_audit(
    plan: ExecutionPlan | None,
    api_logs: list[ApiCallLog],
    scene_evidence: dict[str, Any] | None = None,
    historical_diversity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize traffic quality without changing online collection outcome."""
    audit: dict[str, Any] = {
        "status": "completed",
        "matched_calls": sum(log.traffic_status == "matched" for log in api_logs),
        "unmatched_calls": sum(log.traffic_status not in {"matched", "request_only_segment"} for log in api_logs),
        "request_only_segments": sum(log.traffic_status == "request_only_segment" for log in api_logs),
        "session_verified_calls": sum(valid_session_segment_evidence(log) for log in api_logs),
        "parameter_coverage": {},
        "eligible_parameter_combinations": {},
        "response_variation": {},
        "semantic_alignment": (scene_evidence or {}).get("causal_association_verified"),
    }
    if plan is None:
        audit["status"] = "not_available"
        audit["error"] = "execution plan is unavailable"
        return audit
    try:
        parameter_report = audit_parameter_coverage(plan, api_logs)
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        return audit
    audit["parameter_coverage"] = parameter_report
    audit["diversity"] = build_diversity_audit(
        plan, api_logs, parameter_report, historical_diversity,
    )
    for function, row in parameter_report.get("functions", {}).items():
        audit["eligible_parameter_combinations"][function] = row.get(
            "eligible_distinct_input_combinations", row.get("matched_distinct_input_combinations", 0)
        )
        audit["response_variation"][function] = {
            "observed": row.get("response_state_variation_observed"),
            "status": row.get("response_state_variation_status"),
            "distinct_outputs": row.get("matched_output_variants"),
        }
    return audit


def reaudit_saved_execution(
    execution_dir: Path,
    plan: ExecutionPlan | None = None,
) -> dict[str, Any]:
    """Rebuild derived transaction annotations and coverage from saved files only."""
    api_path = execution_dir / "api_logs.jsonl"
    transaction_path = execution_dir / "protocol_transactions.jsonl"
    if plan is None:
        plan = load_saved_plan(execution_dir.parent / "plan.json")
    logs = [
        ApiCallLog(**{key: value for key, value in row.items() if key in ApiCallLog.__dataclass_fields__})
        for row in _read_jsonl(api_path)
    ]
    transactions = _read_jsonl(transaction_path)
    annotate_session_transactions(transactions)
    by_call = {
        str(row.get("invocation_id")): row
        for row in transactions
        if row.get("invocation_id")
    }
    for log in logs:
        transaction = by_call.get(log.invocation_id)
        if not transaction:
            continue
        log.transaction_id = transaction.get("transaction_id")
        log.traffic_status = str(transaction.get("traffic_status", "no_corresponding_traffic"))
        for key in ("traffic_analysis", "session_traffic"):
            if transaction.get(key):
                log.response[key] = transaction[key]
        if transaction.get("tcp_streams"):
            log.response["traffic_analysis"] = {
                key: transaction[key]
                for key in ("request_payload_hex", "response_payload_hex", "tcp_streams")
                if transaction.get(key) is not None
            }
    write_jsonl(execution_dir / "api_logs.reaudited.jsonl", logs)
    write_jsonl(execution_dir / "protocol_transactions.reaudited.jsonl", transactions)
    scene_path = execution_dir / "scene_execution_evidence.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.is_file() else None
    targets = list((plan.rag_context.get("parameter_coverage_contract", {}).get("functions", {}))
                   if plan else [])
    if not targets and plan:
        targets = list(plan.rag_context.get("explicit_target_functions", []))
    history = load_recent_diversity_feedback(
        execution_dir.parent.parent.parent, targets,
        exclude_run=execution_dir.parent.parent,
    )
    audit = build_offline_audit(plan, logs, scene, history)
    audit["source_api_logs"] = api_path.name
    audit["source_transactions"] = transaction_path.name
    audit["source_pcap_files"] = sorted(path.name for path in (execution_dir / "data").glob("*.pcap"))
    write_json(execution_dir / "offline_audit.json", audit)
    return audit


def load_saved_plan(path: Path) -> ExecutionPlan | None:
    """Load a serialized execution plan for an offline replay, without CNC access."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    allowed = set(ExecutionPlan.__dataclass_fields__)
    payload = {key: value for key, value in payload.items() if key in allowed}
    raw_spec = payload.get("nc_program_spec")
    if isinstance(raw_spec, dict):
        payload["nc_program_spec"] = NcProgramSpec(**{
            key: value for key, value in raw_spec.items() if key in NcProgramSpec.__dataclass_fields__
        })
    payload["steps"] = [
        PlanStep(**{key: value for key, value in row.items() if key in PlanStep.__dataclass_fields__})
        for row in payload.get("steps", []) if isinstance(row, dict)
    ]
    try:
        plan = ExecutionPlan(**payload)
    except (TypeError, ValueError):
        return None
    from .parameter_coverage import attach_parameter_contract
    attach_parameter_contract(plan)
    return plan


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
