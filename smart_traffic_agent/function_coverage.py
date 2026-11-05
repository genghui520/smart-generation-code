from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import ApiCallLog, ExecutionPlan, ExecutionResult, WorkflowState, utc_now
from .focas_contracts import resolve_api_return_code


DEFAULT_FUNCTION_COVERAGE_STATE = Path("rag_indexes/focas/function_coverage_state.json")
CAPTURED_TRAFFIC_STATUSES = {
    "covered_success",
    "covered_expected_error",
}
RESOLVED_FUNCTION_STATUSES = CAPTURED_TRAFFIC_STATUSES | {
    "local_success",
    "unsupported_by_header",
    "export_missing",
    "option_unsupported",
    "blocked_permission_required",
}
UNAVAILABLE_FUNCTION_STATUSES = RESOLVED_FUNCTION_STATUSES - CAPTURED_TRAFFIC_STATUSES - {"local_success"}

# Local library/handle configuration calls do not necessarily create a CNC
# Ethernet request/response. A successful call is still valid API coverage.
NON_WIRE_OBSERVABLE_APIS = frozenset({
    "cnc_settimeout",
    "cnc_getlibopt",
    "cnc_setlibopt",
    "cnc_setdefnode",
    # Multi-path selection is maintained by the FOCAS client handle.  The
    # FANUC reference describes these as selecting/returning the path used by
    # subsequent Data Window calls; on the verified 0i-D NCGuide run they
    # complete locally and emit no CNC request/response of their own.
    "cnc_getpath",
    "cnc_setpath",
})


def is_non_wire_observable(function: str) -> bool:
    return str(function or "").strip() in NON_WIRE_OBSERVABLE_APIS


def has_captured_traffic(record: dict[str, Any]) -> bool:
    # Collection coverage requires wire evidence, while diversity quality is
    # retained independently in the batch's offline_audit.json.
    return record.get("status") in CAPTURED_TRAFFIC_STATUSES and record.get("traffic_matched") is True


def has_successful_coverage(record: dict[str, Any]) -> bool:
    """Whether persisted evidence satisfies the independent API success rule."""
    return has_captured_traffic(record) or record.get("status") == "local_success"


def is_resolved(record: dict[str, Any]) -> bool:
    return (has_successful_coverage(record)
            or record.get("status") in UNAVAILABLE_FUNCTION_STATUSES)


def build_function_coverage_metrics(
    plan: ExecutionPlan | None,
    result: ExecutionResult | None,
) -> dict[str, Any]:
    if plan is None:
        return {}
    manifest_context = plan.rag_context.get("function_coverage_manifest", {})
    if not isinstance(manifest_context, dict) or not manifest_context.get("enabled"):
        return {}

    selected_rows = [
        row
        for row in manifest_context.get("selected_batch", []) or []
        if isinstance(row, dict) and str(row.get("function", "")).strip()
    ]
    target_functions = {
        str(row.get("function", "")).strip()
        for row in selected_rows
        if str(row.get("coverage_role", "target")) == "target"
    }
    support_functions = {
        str(row.get("function", "")).strip()
        for row in selected_rows
        if str(row.get("coverage_role", "target")) == "support"
    }
    if not target_functions:
        target_functions = {str(row.get("function", "")).strip() for row in selected_rows}
    from .parameter_coverage import audit_parameter_coverage
    try:
        parameter_report = audit_parameter_coverage(plan, result.api_logs if result else [])
    except Exception as exc:
        parameter_report = {
            "version": 1, "functions": {}, "passed": False,
            "errors": [f"PARAMETER_COVERAGE_AUDIT_ERROR: {type(exc).__name__}: {exc}"],
        }
    attempted_by_function = collect_attempted_functions(result.api_logs if result is not None else [])
    attempted_functions = set(attempted_by_function)
    successful_functions = {
        function
        for function, rows in attempted_by_function.items()
        if any(int(row.get("status_code", 9999)) == 0 and not row.get("error") for row in rows)
    }
    expected_error_functions = {
        function
        for function, rows in attempted_by_function.items()
        if function not in successful_functions and any(is_planned_expected_error(plan, function, row) for row in rows)
    }
    matched_success_functions = {
        function for function, rows in attempted_by_function.items()
        if any(((_has_wire_evidence(row) or is_non_wire_observable(function))
                and row["status_code"] == 0
                and not row.get("error")) for row in rows)
    }
    matched_expected_error_functions = {
        function for function, rows in attempted_by_function.items()
        if any(row["traffic_status"] == "matched" and is_planned_expected_error(plan, function, row)
               for row in rows)
    }
    option_unsupported_functions = {
        function for function, rows in attempted_by_function.items()
        if function not in matched_success_functions and any(row["status_code"] == 6 and row["traffic_status"] == "matched" for row in rows)
    }
    attempted_batch_functions = target_functions.intersection(attempted_functions)
    attempted_support_functions = support_functions.intersection(attempted_functions)
    successful_batch_functions = target_functions.intersection(successful_functions)
    successful_support_functions = support_functions.intersection(successful_functions)
    expected_error_batch_functions = target_functions.intersection(expected_error_functions)
    expected_error_support_functions = support_functions.intersection(expected_error_functions)
    batch_count = len(target_functions)
    segment_metrics = build_segment_metrics(selected_rows, attempted_functions, successful_functions, expected_error_functions)
    return {
        "parameter_coverage": parameter_report,
        "parameter_coverage_incomplete_functions": [name for name, row in parameter_report["functions"].items() if not row["passed"]],
        "source_manifest": manifest_context.get("source_manifest"),
        "device_scope": manifest_context.get("device_scope") or plan.rag_context.get("task_permissions", {}).get("device_scope", ""),
        "device_model": manifest_context.get("device_model") or plan.rag_context.get("task_permissions", {}).get("device_model", ""),
        "scenario_batch_id": manifest_context.get("scenario_batch_id"),
        "main_state_driver": manifest_context.get("main_state_driver"),
        "quality_target": manifest_context.get("quality_target"),
        "target_function_count": manifest_context.get("target_function_count", batch_count),
        "remaining_function_count_before_run": manifest_context.get("remaining_function_count"),
        "batch_function_count": batch_count,
        "target_batch_function_count": batch_count,
        "support_batch_function_count": len(support_functions),
        "attempted_function_count": len(attempted_functions),
        "attempted_batch_function_count": len(attempted_batch_functions),
        "attempted_target_function_count": len(attempted_batch_functions),
        "attempted_support_function_count": len(attempted_support_functions),
        "successful_function_count": len(successful_functions),
        "successful_batch_function_count": len(successful_batch_functions),
        "successful_target_function_count": len(successful_batch_functions),
        "successful_support_function_count": len(successful_support_functions),
        "expected_error_function_count": len(expected_error_functions),
        "expected_error_batch_function_count": len(expected_error_batch_functions),
        "expected_error_target_function_count": len(expected_error_batch_functions),
        "expected_error_support_function_count": len(expected_error_support_functions),
        "batch_coverage_percent": round(100.0 * len(attempted_batch_functions) / batch_count, 2)
        if batch_count
        else 0.0,
        "attempted_batch_functions": sorted(attempted_batch_functions),
        "attempted_target_functions": sorted(attempted_batch_functions),
        "attempted_support_functions": sorted(attempted_support_functions),
        "successful_batch_functions": sorted(successful_batch_functions),
        "successful_target_functions": sorted(successful_batch_functions),
        "traffic_successful_target_functions": sorted(target_functions & matched_success_functions),
        "traffic_expected_error_target_functions": sorted(target_functions & matched_expected_error_functions),
        "successful_support_functions": sorted(successful_support_functions),
        "expected_error_batch_functions": sorted(expected_error_batch_functions),
        "expected_error_target_functions": sorted(expected_error_batch_functions),
        "expected_error_support_functions": sorted(expected_error_support_functions),
        "unexpected_error_target_functions": sorted(attempted_batch_functions - successful_functions - expected_error_functions),
        "option_unsupported_target_functions": sorted(target_functions & option_unsupported_functions),
        "support_functions": sorted(support_functions),
        "segment_metrics": segment_metrics,
        "missing_batch_functions": sorted(target_functions - attempted_batch_functions),
        "missing_target_functions": sorted(target_functions - attempted_batch_functions),
    }


def build_segment_metrics(
    selected_rows: list[dict[str, Any]],
    attempted_functions: set[str],
    successful_functions: set[str],
    expected_error_functions: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        function = str(row.get("function", "")).strip()
        if not function:
            continue
        segment_id = str(row.get("segment_id") or row.get("scenario_batch_id") or "unsegmented")
        item = grouped.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "scenario_batch_id": row.get("scenario_batch_id"),
                "target_functions": set(),
                "support_functions": set(),
            },
        )
        if str(row.get("coverage_role", "target")) == "support":
            item["support_functions"].add(function)
        else:
            item["target_functions"].add(function)
    metrics: list[dict[str, Any]] = []
    for item in grouped.values():
        targets = item["target_functions"]
        support = item["support_functions"]
        attempted_targets = targets.intersection(attempted_functions)
        attempted_support = support.intersection(attempted_functions)
        metrics.append(
            {
                "segment_id": item["segment_id"],
                "scenario_batch_id": item.get("scenario_batch_id"),
                "target_function_count": len(targets),
                "support_function_count": len(support),
                "attempted_target_function_count": len(attempted_targets),
                "successful_target_function_count": len(targets.intersection(successful_functions)),
                "expected_error_target_function_count": len(targets.intersection(expected_error_functions)),
                "attempted_support_function_count": len(attempted_support),
                "missing_target_functions": sorted(targets - attempted_targets),
            }
        )
    return sorted(metrics, key=lambda row: str(row.get("segment_id", "")))


def load_function_coverage_state(path: Path = DEFAULT_FUNCTION_COVERAGE_STATE) -> dict[str, Any]:
    if not path.exists():
        return empty_function_coverage_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_function_coverage_state()
    if not isinstance(payload, dict):
        return empty_function_coverage_state()
    payload.setdefault("schema_version", 1)
    payload.setdefault("functions", {})
    payload.setdefault("runs", [])
    return payload


def empty_function_coverage_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "functions": {},
        "runs": [],
    }


def load_terminal_covered_functions(path: Path = DEFAULT_FUNCTION_COVERAGE_STATE) -> set[str]:
    """Return only functions with captured request/response traffic."""
    state = load_function_coverage_state(path)
    functions = state.get("functions", {})
    if not isinstance(functions, dict):
        return set()
    return {
        str(function) for function, record in functions.items()
        if isinstance(record, dict) and has_captured_traffic(record)
    }


def load_resolved_functions(path: Path = DEFAULT_FUNCTION_COVERAGE_STATE) -> set[str]:
    """Return captured plus proven unavailable functions for batch selection."""
    state = load_function_coverage_state(path)
    functions = state.get("functions", {})
    if not isinstance(functions, dict):
        return set()
    return {
        str(function) for function, record in functions.items()
        if isinstance(record, dict) and is_resolved(record)
    }


def merge_workflow_function_coverage_state(
    workflow_state: WorkflowState,
    path: Path = DEFAULT_FUNCTION_COVERAGE_STATE,
) -> dict[str, Any]:
    metrics = build_function_coverage_metrics(workflow_state.plan, workflow_state.result)
    if not metrics:
        return {}
    state = load_function_coverage_state(path)
    state["updated_at"] = utc_now()
    functions = state.setdefault("functions", {})
    if not isinstance(functions, dict):
        functions = {}
        state["functions"] = functions

    now = utc_now()
    task_id = workflow_state.request.task_id
    scenario_batch_id = metrics.get("scenario_batch_id")
    target_updates: list[dict[str, Any]] = []
    support_observations: list[dict[str, Any]] = []
    attempted_by_function = collect_attempted_functions(
        workflow_state.result.api_logs if workflow_state.result else []
    )
    # Coverage is about observed wire traffic, not merely a logged API call.
    matched_functions = {
        log.protocol_function for log in (workflow_state.result.api_logs if workflow_state.result else [])
        if _has_wire_evidence({
            "function": log.protocol_function,
            "protocol_function": log.protocol_function,
            "invocation_id": log.invocation_id,
            "status_code": log.status_code,
            "traffic_status": log.traffic_status,
            "response": log.response,
        })
    }

    for function in metrics.get("attempted_target_functions", []):
        if function not in matched_functions:
            local_ok = is_non_wire_observable(function) and any(
                log.protocol_function == function and log.status_code == 0 and not log.error
                for log in (workflow_state.result.api_logs if workflow_state.result else [])
            )
            status = "local_success" if local_ok else "attempted_no_traffic"
        elif function in set(metrics.get("traffic_successful_target_functions", [])):
            status = "covered_success"
        elif function in set(metrics.get("option_unsupported_target_functions", [])):
            status = "option_unsupported"
        elif function in set(metrics.get("traffic_expected_error_target_functions", [])):
            status = "covered_expected_error"
        else:
            status = "attempted_nonterminal"
        record = function_record(functions, function)
        if record.get("status") in CAPTURED_TRAFFIC_STATUSES and not has_captured_traffic(record):
            record["status"] = "unplanned"
            record["terminal"] = False
        record.pop("parameter_coverage", None)
        record.pop("parameter_coverage_passed", None)
        record["status"] = strongest_status(str(record.get("status", "")), status)
        if status in CAPTURED_TRAFFIC_STATUSES:
            record["traffic_matched"] = True
            record["traffic_matched_at"] = now
        elif status == "local_success":
            record["local_observation"] = True
            record["local_observed_at"] = now
        if status == "option_unsupported":
            record["unavailable_evidence"] = "EW_NOOPT returned with matched request/response traffic"
        record["terminal"] = is_resolved(record)
        record["coverage_role"] = "target"
        if metrics.get("device_scope"):
            record["device_scope"] = metrics.get("device_scope")
        if metrics.get("device_model"):
            record["device_model"] = metrics.get("device_model")
        record["last_seen_at"] = now
        record["last_task_id"] = task_id
        record["last_scenario_batch_id"] = scenario_batch_id
        record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
        if status in {"covered_success", "local_success"}:
            record["success_count"] = int(record.get("success_count", 0)) + 1
        elif status == "covered_expected_error":
            record["expected_error_count"] = int(record.get("expected_error_count", 0)) + 1
        target_updates.append({"function": function, "status": record["status"], "terminal": record["terminal"]})

    for function in metrics.get("missing_target_functions", []):
        record = function_record(functions, function)
        if not is_resolved(record):
            record["status"] = "not_attempted"
            record["terminal"] = False
            for log in workflow_state.result.api_logs if workflow_state.result else []:
                if log.protocol_function != "host_api_prerequisite":
                    continue
                data = str(log.response.get("data", ""))
                fields = dict(item.split("=", 1) for item in data.split(";") if "=" in item)
                if fields.get("function") == function:
                    reason = fields.get("reason", "")
                    record["status"] = "export_missing" if reason == "export_missing" else "blocked_prerequisite"
                    record["prerequisite_reason"] = reason
                    record["terminal"] = is_resolved(record)
                    target_updates.append({"function": function, "status": record["status"], "terminal": record["terminal"]})
                    break
        record["coverage_role"] = "target"
        record["last_task_id"] = task_id
        record["last_scenario_batch_id"] = scenario_batch_id

    for function in metrics.get("attempted_support_functions", []):
        record = function_record(functions, function)
        record["support_observation_count"] = int(record.get("support_observation_count", 0)) + 1
        record["last_support_seen_at"] = now
        record["last_support_task_id"] = task_id
        record["last_support_scenario_batch_id"] = scenario_batch_id
        support_rows = attempted_by_function.get(function, [])
        local_success = is_non_wire_observable(function) and any(
            _successful_return_code(row) for row in support_rows
        )
        traffic_success = any(
            _successful_return_code(row)
            and _has_wire_evidence(row)
            for row in support_rows
        )
        if local_success or traffic_success:
            verified_status = "local_success" if local_success else "covered_success"
            record["status"] = strongest_status(str(record.get("status", "")), verified_status)
            record["terminal"] = True
            record["support_coverage_verified"] = True
            record["support_success_count"] = int(record.get("support_success_count", 0)) + 1
            if local_success:
                record["local_observation"] = True
                record["local_observed_at"] = now
            else:
                record["traffic_matched"] = True
                record["traffic_matched_at"] = now
        elif not record.get("status"):
            record["status"] = "support_observed"
            record["terminal"] = False
        if record.get("coverage_role") != "target":
            record["coverage_role"] = "support"
        if metrics.get("device_scope"):
            record["device_scope"] = metrics.get("device_scope")
        if metrics.get("device_model"):
            record["device_model"] = metrics.get("device_model")
        if not record.get("status"):
            record["status"] = "support_observed"
            record["terminal"] = False
        support_observations.append({"function": function, "status": record.get("status", "support_observed")})

    run_record = {
        "timestamp": now,
        "task_id": task_id,
        "scenario_batch_id": scenario_batch_id,
        "device_scope": metrics.get("device_scope", ""),
        "device_model": metrics.get("device_model", ""),
        "target_batch_function_count": metrics.get("target_batch_function_count"),
        "attempted_target_function_count": metrics.get("attempted_target_function_count"),
        "successful_target_function_count": metrics.get("successful_target_function_count"),
        "expected_error_target_function_count": metrics.get("expected_error_target_function_count"),
        "missing_target_function_count": len(metrics.get("missing_target_functions", [])),
        "attempted_support_function_count": metrics.get("attempted_support_function_count"),
    }
    runs = state.setdefault("runs", [])
    if isinstance(runs, list):
        runs.append(run_record)
        del runs[:-200]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize_function_coverage_state(state)
    summary.update(
        {
            "state_path": str(path),
            "target_updates": target_updates,
            "support_observations": support_observations,
            "run_record": run_record,
        }
    )
    return summary


def _has_wire_evidence(row: dict[str, Any]) -> bool:
    """Return true for ordinary exchanges or fully validated session members."""
    from .execution_contracts import API_TRAFFIC_SESSION_RULES
    from .traffic_transactions import valid_session_segment_evidence

    function = str(row.get("function") or row.get("protocol_function") or row.get("api") or "")
    if function in API_TRAFFIC_SESSION_RULES:
        return valid_session_segment_evidence(row)
    return row.get("traffic_status") == "matched"


def _successful_return_code(row: dict[str, Any]) -> bool:
    try:
        return int(row.get("status_code", 9999)) == 0 and not row.get("error")
    except (TypeError, ValueError):
        return False


def summarize_function_coverage_state(state: dict[str, Any]) -> dict[str, Any]:
    functions = state.get("functions", {})
    if not isinstance(functions, dict):
        functions = {}
    status_counts: dict[str, int] = {}
    captured_count = 0
    local_success_count = 0
    resolved_count = 0
    unavailable_count = 0
    support_observed_count = 0
    for record in functions.values():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", "unknown"))
        if status in CAPTURED_TRAFFIC_STATUSES and not has_captured_traffic(record):
            status = "legacy_unverified"
        status_counts[status] = status_counts.get(status, 0) + 1
        if has_captured_traffic(record):
            captured_count += 1
        if status == "local_success":
            local_success_count += 1
        if status in UNAVAILABLE_FUNCTION_STATUSES:
            unavailable_count += 1
        if is_resolved(record):
            resolved_count += 1
        if int(record.get("support_observation_count", 0)) > 0:
            support_observed_count += 1
    return {
        "schema_version": state.get("schema_version", 1),
        "tracked_function_count": len(functions),
        "captured_function_count": captured_count,
        "local_success_function_count": local_success_count,
        "resolved_function_count": resolved_count,
        "unavailable_function_count": unavailable_count,
        # Backward-compatible name, now strictly traffic-backed.
        "terminal_covered_function_count": captured_count,
        "support_observed_function_count": support_observed_count,
        "status_counts": dict(sorted(status_counts.items())),
        "run_count": len(state.get("runs", [])) if isinstance(state.get("runs"), list) else 0,
        "updated_at": state.get("updated_at"),
    }


def function_record(functions: dict[str, Any], function: str) -> dict[str, Any]:
    record = functions.setdefault(function, {"function": function, "status": "unplanned", "terminal": False})
    if not isinstance(record, dict):
        record = {"function": function, "status": "unplanned", "terminal": False}
        functions[function] = record
    return record


def strongest_status(existing: str, new: str) -> str:
    priority = {
        "covered_success": 100,
        "local_success": 95,
        "covered_expected_error": 90,
        "option_unsupported": 85,
        "export_missing": 80,
        "unsupported_by_header": 80,
        "blocked_permission_required": 70,
        "blocked_prerequisite": 45,
        "attempted_nonterminal": 40,
        "attempted_no_traffic": 35,
        "not_attempted": 10,
        "support_observed": 5,
        "unplanned": 0,
        "": 0,
    }
    return new if priority.get(new, 0) >= priority.get(existing, 0) else existing


def merge_persisted_coverage_state(source_path: Path, target_path: Path) -> dict[str, Any]:
    """Promote verified records from an output-local state into global history."""
    source = load_function_coverage_state(source_path)
    target = load_function_coverage_state(target_path)
    functions = target.setdefault("functions", {})
    changed: list[str] = []
    for name, candidate in (source.get("functions", {}) or {}).items():
        if not isinstance(candidate, dict):
            continue
        current = functions.get(name)
        if not isinstance(current, dict):
            current = {"function": name, "status": "unplanned", "terminal": False}
        candidate_status = str(candidate.get("status", ""))
        if candidate_status in {"covered_success", "covered_expected_error", "local_success",
                                "option_unsupported", "export_missing", "unsupported_by_header",
                                "blocked_permission_required"}:
            merged = dict(current)
            merged.update(candidate)
            merged["status"] = strongest_status(str(current.get("status", "")), candidate_status)
            merged["terminal"] = is_resolved(merged)
            if merged != current:
                functions[name] = merged
                changed.append(str(name))
    target["updated_at"] = utc_now()
    runs = target.setdefault("runs", [])
    for run in source.get("runs", []) or []:
        if isinstance(run, dict) and run not in runs:
            runs.append(run)
    if isinstance(runs, list):
        del runs[:-200]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source": str(source_path), "target": str(target_path), "changed": sorted(changed)}


def sync_coverage_state_from_progress_csv(csv_path: Path, state_path: Path) -> dict[str, Any]:
    """Use the verified desktop progress ledger to refresh global history."""
    import csv

    if not csv_path.is_file():
        return {"changed": [], "missing": True}
    state = load_function_coverage_state(state_path)
    functions = state.setdefault("functions", {})
    changed: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = str(row.get("function", "")).strip()
            if not name or str(row.get("collected", "")).strip() != "✓":
                continue
            current = functions.get(name)
            if not isinstance(current, dict):
                current = {"function": name, "status": "unplanned", "terminal": False}
            latest = str(row.get("latest_status", "")).strip()
            status = "local_success" if latest == "local_success" else "covered_success"
            merged = dict(current)
            merged.update({
                "function": name,
                "status": strongest_status(str(current.get("status", "")), status),
                "terminal": True,
                "traffic_matched": status != "local_success",
                "coverage_role": "target",
                "progress_ledger": str(csv_path),
                "evidence_runs": str(row.get("evidence_runs", "")),
                "last_progress_run_id": str(row.get("latest_run_id", "")),
            })
            if merged != current:
                functions[name] = merged
                changed.append(name)
    state["updated_at"] = utc_now()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"changed": sorted(changed), "missing": False, "source": str(csv_path)}


def collect_attempted_functions(api_logs: list[ApiCallLog]) -> dict[str, list[dict[str, Any]]]:
    attempted: dict[str, list[dict[str, Any]]] = {}
    for index, log in enumerate(api_logs):
        for function in split_protocol_functions(log.protocol_function or str(log.response.get("function", ""))):
            attempted.setdefault(function, []).append(
                {
                    "function": function,
                    "protocol_function": function,
                    "log_index": index,
                    "step_id": log.step_id,
                    "interface_name": log.interface_name,
                    "status_code": log.status_code,
                    "error": log.error,
                    "input_parameters": log.input_parameters,
                    "traffic_status": log.traffic_status,
                    "invocation_id": log.invocation_id,
                    "response": log.response,
                }
            )
    return attempted


def is_planned_expected_error(plan: ExecutionPlan, function: str, row: dict[str, Any]) -> bool:
    """An error is expected only when the plan explicitly requests that code."""
    code = int(row.get("status_code", 0))
    if code == 0 or not resolve_api_return_code(function, code)["documented_for_api"]:
        return False
    raw = str(row.get("input_parameters", {}).get("raw", ""))
    annotated = re.search(r"(?:^|;)expected_return_code=(-?\d+)(?:;|$)", raw)
    if not annotated or int(annotated[1]) != code:
        return False
    for step in plan.steps:
        if step.step_id != row.get("step_id") or function not in split_protocol_functions(step.protocol_function or ""):
            continue
        expected = step.parameters.get("expected_return_code")
        if expected is not None and str(expected) == str(code):
            return True
        # Planner-generated negative probes may serialize the expected FOCAS
        # symbol in the structured expected state/coverage text instead of a
        # scalar expected_return_code. Require the observed symbol to match
        # that explicit plan text; an arbitrary nonzero return remains fatal.
        expected_symbols = set(
            re.findall(
                r"\bEW_[A-Z0-9_]+\b",
                " ".join(
                    (
                        str(step.expected_state or ""),
                        str(step.action or ""),
                        str((step.parameters.get("parameter_coverage_goal") or {}).get("coverage_scope", "")),
                    )
                ),
            )
        )
        observed_symbols = set(
            re.findall(
                r"\bEW_[A-Z0-9_]+\b",
                " ".join(
                    (
                        str(row.get("error") or ""),
                        str((row.get("response") or {}).get("return_text", "")),
                    )
                ),
            )
        )
        if expected_symbols & observed_symbols:
            return True
    return False


def split_protocol_functions(value: str) -> list[str]:
    names: list[str] = []
    for token in str(value or "").replace("+", "/").replace(",", "/").split("/"):
        name = token.strip()
        if name:
            names.append(name)
    return names
