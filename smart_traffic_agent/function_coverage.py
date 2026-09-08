from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ApiCallLog, ExecutionPlan, ExecutionResult, WorkflowState, utc_now


DEFAULT_FUNCTION_COVERAGE_STATE = Path("rag_indexes/focas/function_coverage_state.json")
TERMINAL_COVERAGE_STATUSES = {
    "covered_success",
    "covered_expected_error",
    "unsupported_by_header",
    "export_missing",
    "option_unsupported",
    "blocked_permission_required",
}


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
        if function not in successful_functions and rows
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
        "source_manifest": manifest_context.get("source_manifest"),
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
        "successful_support_functions": sorted(successful_support_functions),
        "expected_error_batch_functions": sorted(expected_error_batch_functions),
        "expected_error_target_functions": sorted(expected_error_batch_functions),
        "expected_error_support_functions": sorted(expected_error_support_functions),
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
    state = load_function_coverage_state(path)
    functions = state.get("functions", {})
    if not isinstance(functions, dict):
        return set()
    covered: set[str] = set()
    for function, record in functions.items():
        if not isinstance(record, dict):
            continue
        if record.get("terminal") or record.get("status") in TERMINAL_COVERAGE_STATUSES:
            covered.add(str(function))
    return covered


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

    for function in metrics.get("attempted_target_functions", []):
        status = (
            "covered_success"
            if function in set(metrics.get("successful_target_functions", []))
            else "covered_expected_error"
            if function in set(metrics.get("expected_error_target_functions", []))
            else "attempted_nonterminal"
        )
        record = function_record(functions, function)
        record["status"] = strongest_status(str(record.get("status", "")), status)
        record["terminal"] = record["status"] in TERMINAL_COVERAGE_STATUSES
        record["coverage_role"] = "target"
        record["last_seen_at"] = now
        record["last_task_id"] = task_id
        record["last_scenario_batch_id"] = scenario_batch_id
        record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
        if status == "covered_success":
            record["success_count"] = int(record.get("success_count", 0)) + 1
        elif status == "covered_expected_error":
            record["expected_error_count"] = int(record.get("expected_error_count", 0)) + 1
        target_updates.append({"function": function, "status": record["status"], "terminal": record["terminal"]})

    for function in metrics.get("missing_target_functions", []):
        record = function_record(functions, function)
        if str(record.get("status", "")) not in TERMINAL_COVERAGE_STATUSES:
            record["status"] = "not_attempted"
            record["terminal"] = False
        record["coverage_role"] = "target"
        record["last_task_id"] = task_id
        record["last_scenario_batch_id"] = scenario_batch_id

    for function in metrics.get("attempted_support_functions", []):
        record = function_record(functions, function)
        record["support_observation_count"] = int(record.get("support_observation_count", 0)) + 1
        record["last_support_seen_at"] = now
        record["last_support_task_id"] = task_id
        record["last_support_scenario_batch_id"] = scenario_batch_id
        if not record.get("status"):
            record["status"] = "support_observed"
            record["terminal"] = False
        support_observations.append({"function": function, "status": record.get("status", "support_observed")})

    run_record = {
        "timestamp": now,
        "task_id": task_id,
        "scenario_batch_id": scenario_batch_id,
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


def summarize_function_coverage_state(state: dict[str, Any]) -> dict[str, Any]:
    functions = state.get("functions", {})
    if not isinstance(functions, dict):
        functions = {}
    status_counts: dict[str, int] = {}
    terminal_count = 0
    support_observed_count = 0
    for record in functions.values():
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if record.get("terminal") or status in TERMINAL_COVERAGE_STATUSES:
            terminal_count += 1
        if int(record.get("support_observation_count", 0)) > 0:
            support_observed_count += 1
    return {
        "schema_version": state.get("schema_version", 1),
        "tracked_function_count": len(functions),
        "terminal_covered_function_count": terminal_count,
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
        "covered_expected_error": 90,
        "option_unsupported": 85,
        "export_missing": 80,
        "unsupported_by_header": 80,
        "blocked_permission_required": 70,
        "attempted_nonterminal": 40,
        "not_attempted": 10,
        "support_observed": 5,
        "unplanned": 0,
        "": 0,
    }
    return new if priority.get(new, 0) >= priority.get(existing, 0) else existing


def collect_attempted_functions(api_logs: list[ApiCallLog]) -> dict[str, list[dict[str, Any]]]:
    attempted: dict[str, list[dict[str, Any]]] = {}
    for index, log in enumerate(api_logs):
        for function in split_protocol_functions(log.protocol_function or str(log.response.get("function", ""))):
            attempted.setdefault(function, []).append(
                {
                    "log_index": index,
                    "step_id": log.step_id,
                    "interface_name": log.interface_name,
                    "status_code": log.status_code,
                    "error": log.error,
                }
            )
    return attempted


def split_protocol_functions(value: str) -> list[str]:
    names: list[str] = []
    for token in str(value or "").replace("+", "/").replace(",", "/").split("/"):
        name = token.strip()
        if name:
            names.append(name)
    return names
