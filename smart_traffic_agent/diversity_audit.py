"""Offline, evidence-backed input and application-traffic diversity metrics."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .parameter_coverage import KEY as PARAMETER_CONTRACT_KEY, input_values
from .traffic_transactions import valid_session_segment_evidence


def build_diversity_audit(plan, logs, parameter_report=None, history=None) -> dict[str, Any]:
    """Measure real call, response, wire and session variety without gating execution."""
    contract = (plan.rag_context.get(PARAMETER_CONTRACT_KEY, {}) if plan else {}).get("functions", {})
    parameter_rows = (parameter_report or {}).get("functions", {})
    history = history or {}
    functions: dict[str, Any] = {}
    for function, rule in contract.items():
        dimensions = list(rule.get("dimensions", []))
        calls = [log for log in logs if log.protocol_function == function
                 and not any(token in log.step_id.upper() for token in
                             ("RESTORE", "VERIFY_SELECTION", "VERIFY_RESTORE", "MOTION_EVIDENCE"))]
        attempted: dict[tuple, dict[str, Any]] = {}
        eligible: dict[tuple, dict[str, Any]] = {}
        response_values: set[str] = set()
        request_payloads: set[str] = set()
        response_payloads: set[str] = set()
        valid_sessions: set[str] = set()
        request_segment_payloads: set[str] = set()
        response_segment_payloads: set[str] = set()
        mixed_request_payloads: set[str] = set()
        mixed_response_payloads: set[str] = set()
        for log in calls:
            values = input_values(log, function)
            config = {name: values[name] for name in dimensions if name in values}
            signature = _signature(config)
            attempted.setdefault(signature, config)
            session_valid = valid_session_segment_evidence(log)
            wire_valid = session_valid or log.traffic_status == "matched"
            successful = log.status_code == 0 and not log.error
            if successful and wire_valid and len(config) == len(dimensions):
                eligible.setdefault(signature, config)
                analysis = log.response.get("traffic_analysis", {})
                if not isinstance(analysis, dict):
                    analysis = {}
                data = log.response.get("data")
                if data not in (None, ""):
                    response_values.add(_stable_value(data))
                session = log.response.get("session_traffic") or {}
                direction = session.get("wire_direction") if isinstance(session, dict) else None
                for stream in analysis.get("tcp_streams", []) or []:
                    if not isinstance(stream, dict):
                        continue
                    request = _payload_digest(stream.get("request_payload_hex"))
                    response = _payload_digest(stream.get("response_payload_hex"))
                    if request:
                        request_payloads.add(request)
                        if session_valid and log.traffic_status == "request_only_segment":
                            request_segment_payloads.add(request)
                        elif session_valid and direction == "both":
                            mixed_request_payloads.add(request)
                    if response:
                        response_payloads.add(response)
                        if session_valid and direction == "response":
                            response_segment_payloads.add(response)
                        elif session_valid and direction == "both":
                            mixed_response_payloads.add(response)
                request = _payload_digest(analysis.get("request_payload_hex"))
                response = _payload_digest(analysis.get("response_payload_hex"))
                if request:
                    request_payloads.add(request)
                    if session_valid and log.traffic_status == "request_only_segment":
                        request_segment_payloads.add(request)
                if response:
                    response_payloads.add(response)
                    if session_valid and direction == "response":
                        response_segment_payloads.add(response)
                    elif session_valid and direction == "both":
                        mixed_response_payloads.add(response)
                if request and session_valid and direction == "both":
                    mixed_request_payloads.add(request)
            if session_valid:
                session = log.response.get("session_traffic") or log.response.get("traffic_analysis") or {}
                if isinstance(session, dict) and session.get("session_id"):
                    valid_sessions.add(str(session["session_id"]))

        previous = history.get(function, {}) if isinstance(history, dict) else {}
        previous_configs = {_signature(row) for row in previous.get("observed_input_configurations", [])
                             if isinstance(row, dict)}
        previous_requests = set(previous.get("request_payload_sha256s", []))
        previous_responses = set(previous.get("response_payload_sha256s", []))
        new_configs = [value for key, value in eligible.items() if key not in previous_configs]
        new_requests = request_payloads - previous_requests
        new_responses = response_payloads - previous_responses
        parameter = parameter_rows.get(function, {})
        input_applicable = bool(rule.get("input_diversity_applicable", bool(dimensions)))
        response_applicable = bool(rule.get("response_state_diversity_applicable", False))
        state_strategies = [item for item in rule.get("state_coverage", []) if isinstance(item, dict)]
        if input_applicable and len(eligible) < int(rule.get("minimum_distinct", 1)):
            next_action = "try_unseen_documented_input_configuration"
        elif response_applicable and len(response_values) < 2 and state_strategies:
            next_action = "consider_next_planned_state_strategy"
        elif response_applicable and len(response_values) < 2:
            next_action = "record_environment_limit_or_plan_supported_state_observation"
        else:
            next_action = "stop_or_move_to_next_target"
        functions[function] = {
            "variation_class": rule.get("variation_requirements", {}).get("variation_class", rule.get("mode")),
            "input_diversity": {
                "applicable": input_applicable,
                "required": bool(rule.get("input_diversity_required", False)),
                "dimensions": dimensions,
                "attempted_distinct_configurations": len(attempted),
                "eligible_distinct_configurations": len(eligible),
                "historical_distinct_configurations": len(previous_configs),
                "new_eligible_configurations": len(new_configs),
                "observed_input_configurations": list(eligible.values()),
                "configuration_budget": rule.get("configuration_budget"),
            },
            "response_diversity": {
                "applicable": response_applicable,
                "required": bool(rule.get("response_state_diversity_required", False)),
                "distinct_effective_responses": len(response_values),
                "effective_response_hashes": sorted(_hash(value) for value in response_values),
                "status": parameter.get("response_state_variation_status", "not_observed"),
            },
            "application_traffic": {
                "distinct_request_payloads": len(request_payloads),
                "distinct_response_payloads": len(response_payloads),
                "historical_request_payloads": len(previous_requests),
                "historical_response_payloads": len(previous_responses),
                "new_request_payloads": len(new_requests),
                "new_response_payloads": len(new_responses),
                "request_payload_sha256s": sorted(request_payloads),
                "response_payload_sha256s": sorted(response_payloads),
            },
            "sessions": {
                "successful_complete_sessions": len(valid_sessions),
                "request_only_segment_payload_variants": len(request_segment_payloads),
                "response_only_segment_payload_variants": len(response_segment_payloads),
                "mixed_session_request_payload_variants": len(mixed_request_payloads),
                "mixed_session_response_payload_variants": len(mixed_response_payloads),
                "session_variation_semantics": "segmented_or_mixed" if any((
                    request_segment_payloads, response_segment_payloads, mixed_request_payloads,
                    mixed_response_payloads,
                )) else "transaction",
            },
            "calls": {"actual_target_calls": len(calls), "successful_calls": sum(
                log.status_code == 0 and not log.error for log in calls
            )},
            "incremental_recommendation": {
                "next_action": next_action,
                "reason": _recommendation_reason(next_action, len(new_configs), len(new_requests), len(new_responses)),
            },
        }
    return {"version": 1, "status": "offline_only", "functions": functions}


def load_recent_diversity_feedback(runs_root: Path, functions, *, exclude_run: Path | None = None,
                                   limit_per_function: int = 3) -> dict[str, Any]:
    """Read recent offline audits as planning hints; never treats them as current evidence."""
    excluded = exclude_run.resolve() if exclude_run else None
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(runs_root.glob("*/batch_*/execution/offline_audit.json"), reverse=True):
        if excluded:
            try:
                if path.resolve().is_relative_to(excluded):
                    continue
            except OSError:
                pass
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for function in functions:
            row = (payload.get("diversity", {}).get("functions", {}).get(function, {}))
            if row and len(found[function]) < limit_per_function:
                found[function].append(row)
    result = {}
    for function, rows in found.items():
        configs, requests, responses = {}, set(), set()
        for row in rows:
            inputs = row.get("input_diversity", {})
            for config in inputs.get("observed_input_configurations", []):
                if isinstance(config, dict):
                    configs[_signature(config)] = config
            traffic = row.get("application_traffic", {})
            requests.update(traffic.get("request_payload_sha256s", []))
            responses.update(traffic.get("response_payload_sha256s", []))
        result[function] = {
            "recent_audits": len(rows),
            "observed_input_configurations": list(configs.values()),
            "request_payload_sha256s": sorted(requests),
            "response_payload_sha256s": sorted(responses),
            "latest_incremental_recommendation": rows[0].get("incremental_recommendation", {}),
        }
    return result


def _signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stable_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _payload_digest(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    try:
        payload = bytes.fromhex(text)
    except ValueError:
        return None
    return hashlib.sha256(payload).hexdigest() if payload else None


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _recommendation_reason(action: str, configs: int, requests: int, responses: int) -> str:
    if action == "try_unseen_documented_input_configuration":
        return f"{configs} new eligible configurations in this round; use only remaining documented values."
    if action == "consider_next_planned_state_strategy":
        return f"No effective response change yet; {requests} new request and {responses} new response payloads were observed."
    if action == "record_environment_limit_or_plan_supported_state_observation":
        return "Response state is applicable but unchanged; no further validated state strategy is available."
    return "No unmet applicable diversity objective remains, or the output is inherently stable."
