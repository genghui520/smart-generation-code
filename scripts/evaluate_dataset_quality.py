"""Evaluate Dataset Quality without invoking models, controllers, or simulators.

The evaluator never treats source code, Planner output, or reconstructed capture
events as proof of traffic.  A covered/aligned invocation needs a concrete,
TCP-reassembled application-layer transaction supplied by the recorded dataset.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_ROOT = ROOT / "evaluation" / "ground_truth" / "dataset_quality"
EXACT_API = re.compile(r"^(?:cnc|pmc)_[A-Za-z0-9_]+$")
EXPECTATIONS = {
    "fixed",
    "parameter_sensitive",
    "timing_sensitive",
    "parameter_and_timing_sensitive",
}
TRANSACTION_CANDIDATES = (
    "execution/data/protocol_transactions.jsonl",
    "execution/protocol_transactions.jsonl",
    "dataset_quality_transactions.jsonl",
)
PCAP_CANDIDATES = (
    "execution/data/focas_capture.pcap",
    "execution/data/focas_capture.pcapng",
    "execution/focas_capture.pcap",
    "execution/focas_capture.pcapng",
)

ProtocolSample = tuple[bytes, ...]
UnstableMask = dict[int, set[int]]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def load_ground_truth(scenario: str) -> dict[str, Any]:
    path = GROUND_TRUTH_ROOT / f"{scenario}.json"
    if not path.exists():
        raise ValueError(f"Unsupported Dataset Quality scenario: {scenario}")
    ground_truth = load_json(path)
    expectations = ground_truth.get("api_variation_expectations", {})
    invalid = sorted({str(value) for value in expectations.values()} - EXPECTATIONS)
    if invalid:
        raise ValueError(f"Invalid variation expectation(s) in {path}: {invalid}")
    return ground_truth


def normalize_parameters(value: Any) -> str:
    """Return a stable representation of an observed parameter configuration."""

    if isinstance(value, str):
        pieces: dict[str, str] = {}
        for part in value.split(";"):
            if "=" in part:
                key, item = part.split("=", 1)
                pieces[key.strip()] = item.strip()
        value = pieces if pieces else value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _decode_payload(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, list) and all(isinstance(item, int) and 0 <= item <= 255 for item in value):
        return bytes(value)
    if isinstance(value, dict):
        if "payload_hex" in value:
            return bytes.fromhex(str(value["payload_hex"]).replace(" ", ""))
        if "application_payload_hex" in value:
            return bytes.fromhex(str(value["application_payload_hex"]).replace(" ", ""))
        if "payload_base64" in value:
            return base64.b64decode(str(value["payload_base64"]), validate=True)
        if "payload" in value:
            return _decode_payload(value["payload"])
    if isinstance(value, str):
        compact = value.replace(" ", "").replace(":", "")
        if compact and len(compact) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", compact):
            return bytes.fromhex(compact)
    raise ValueError("Application payload must be bytes, byte values, hex, or base64.")


def build_protocol_sample(transaction: dict[str, Any]) -> ProtocolSample | None:
    """Build an ordered, transport-header-free, TCP-reassembled sample."""

    if not transaction or transaction.get("contains_transport_headers") is True:
        return None
    if transaction.get("tcp_reassembled") is False:
        return None
    if transaction.get("request_response_associated") is False:
        return None
    messages = transaction.get("application_messages", transaction.get("protocol_sample"))
    if not isinstance(messages, list) or not messages:
        return None
    try:
        sample = tuple(_decode_payload(message) for message in messages)
    except (ValueError, TypeError, binascii.Error):
        return None
    return sample if all(message for message in sample) else None


def group_invocations_by_api(invocations: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invocation in invocations:
        api = str(invocation.get("api", ""))
        if EXACT_API.fullmatch(api):
            grouped[api].append(invocation)
    return dict(grouped)


def group_by_parameter_configuration(invocations: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invocation in invocations:
        grouped[normalize_parameters(invocation.get("parameters", {}))].append(invocation)
    return dict(grouped)


def group_by_timing(invocations: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for invocation in invocations:
        grouped[str(invocation.get("timing", ""))].append(invocation)
    return dict(grouped)


def find_unstable_positions(samples: Iterable[ProtocolSample]) -> UnstableMask:
    """Find byte positions that vary under otherwise identical controls."""

    values = list(samples)
    if len(values) < 2:
        return {}
    first = values[0]
    if any(len(sample) != len(first) for sample in values):
        return {}
    mask: UnstableMask = defaultdict(set)
    for message_index, message in enumerate(first):
        if any(len(sample[message_index]) != len(message) for sample in values):
            continue
        for byte_index in range(len(message)):
            if len({sample[message_index][byte_index] for sample in values}) > 1:
                mask[message_index].add(byte_index)
    return dict(mask)


def mask_unstable_positions(sample: ProtocolSample, mask: UnstableMask) -> tuple[tuple[int | None, ...], ...]:
    return tuple(
        tuple(None if byte_index in mask.get(message_index, set()) else value for byte_index, value in enumerate(message))
        for message_index, message in enumerate(sample)
    )


def samples_differ(left: ProtocolSample, right: ProtocolSample, mask: UnstableMask | None = None) -> bool:
    if len(left) != len(right):
        return True
    if any(len(a) != len(b) for a, b in zip(left, right)):
        return True
    active_mask = mask or {}
    return mask_unstable_positions(left, active_mask) != mask_unstable_positions(right, active_mask)


def _successful_with_traffic(invocation: dict[str, Any]) -> bool:
    code = invocation.get("return_code")
    return code in (0, "0", "EW_OK") and invocation.get("protocol_sample") is not None


def evaluate_api_coverage(ground_truth: dict[str, Any], invocations: list[dict[str, Any]]) -> dict[str, Any]:
    successful_apis = {
        str(invocation.get("api"))
        for invocation in invocations
        if _successful_with_traffic(invocation) and EXACT_API.fullmatch(str(invocation.get("api", "")))
    }
    required: list[str] = []
    covered: list[str] = []
    missing: list[str] = []
    matches: dict[str, list[str]] = {}
    for group in ground_truth.get("required_api_groups", []):
        name = str(group["name"])
        accepted = [str(api) for api in group.get("accepted_apis", [])]
        mode = str(group.get("mode", "any"))
        matched = sorted(set(accepted) & successful_apis)
        is_covered = bool(matched) if mode == "any" else bool(accepted) and set(accepted) <= successful_apis
        required.append(name)
        matches[name] = matched
        (covered if is_covered else missing).append(name)
    score = len(covered) / len(required) if required else None
    return {
        "required_groups": required,
        "covered_groups": covered,
        "missing_groups": missing,
        "matched_actual_apis": matches,
        "matched_apis": sorted({api for apis in matches.values() for api in apis}),
        "coverage": score,
        "score": score,
    }


def _merge_masks(target: UnstableMask, source: UnstableMask) -> None:
    for message_index, positions in source.items():
        target.setdefault(message_index, set()).update(positions)


def _api_unstable_mask(invocations: list[dict[str, Any]]) -> UnstableMask:
    repeats: dict[tuple[str, str, str], list[ProtocolSample]] = defaultdict(list)
    for row in invocations:
        sample = row.get("protocol_sample")
        if sample is None:
            continue
        key = (
            normalize_parameters(row.get("parameters", {})),
            str(row.get("timing", "")),
            normalize_parameters(row.get("context", {})),
        )
        repeats[key].append(sample)
    mask: UnstableMask = {}
    for samples in repeats.values():
        if len(samples) >= 2:
            _merge_masks(mask, find_unstable_positions(samples))
    return mask


def _variation_result(
    invocations: list[dict[str, Any]],
    *,
    kind: str,
    mask: UnstableMask,
) -> dict[str, Any]:
    valid = [row for row in invocations if row.get("protocol_sample") is not None]
    if kind == "parameter":
        controlled_key = lambda row: (str(row.get("timing", "")), normalize_parameters(row.get("context", {})))
        condition_key = lambda row: normalize_parameters(row.get("parameters", {}))
        insufficient_reason = "fewer than two parameter configurations under the same timing/context"
    else:
        controlled_key = lambda row: (normalize_parameters(row.get("parameters", {})), normalize_parameters(row.get("context", {})))
        condition_key = lambda row: str(row.get("timing", ""))
        insufficient_reason = "fewer than two timings with identical parameters/context"
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        strata[controlled_key(row)].append(row)
    comparable_strata = [
        rows for rows in strata.values() if len({condition_key(row) for row in rows}) >= 2
    ]
    comparable = bool(comparable_strata)
    satisfied = False
    for rows in comparable_strata:
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if condition_key(left) == condition_key(right):
                    continue
                if samples_differ(left["protocol_sample"], right["protocol_sample"], mask):
                    satisfied = True
                    break
            if satisfied:
                break
        if satisfied:
            break
    distinct_conditions = len({condition_key(row) for row in valid})
    distinct_samples = len({mask_unstable_positions(row["protocol_sample"], mask) for row in valid})
    label = "distinct_parameter_configurations" if kind == "parameter" else "distinct_timings"
    return {
        "comparable": comparable,
        label: distinct_conditions,
        "distinct_protocol_samples": distinct_samples,
        "satisfied": satisfied if comparable else None,
        "reason": None if comparable else insufficient_reason,
    }


def evaluate_api_discriminability(
    api: str,
    expectation: str,
    invocations: list[dict[str, Any]],
) -> dict[str, Any]:
    if expectation == "fixed":
        return {
            "api": api,
            "expectation": expectation,
            "parameter_variation": {"comparable": False, "reason": "not_required"},
            "timing_variation": {"comparable": False, "reason": "not_required"},
            "unstable_positions": {},
            "score": None,
            "excluded": True,
        }
    mask = _api_unstable_mask(invocations)
    parameter_required = expectation in {"parameter_sensitive", "parameter_and_timing_sensitive"}
    timing_required = expectation in {"timing_sensitive", "parameter_and_timing_sensitive"}
    parameter = (
        _variation_result(invocations, kind="parameter", mask=mask)
        if parameter_required
        else {"comparable": False, "reason": "not_required"}
    )
    timing = (
        _variation_result(invocations, kind="timing", mask=mask)
        if timing_required
        else {"comparable": False, "reason": "not_required"}
    )
    required_results = [parameter] if parameter_required else []
    if timing_required:
        required_results.append(timing)
    score = (
        int(all(result.get("satisfied") is True for result in required_results))
        if required_results and all(result.get("comparable") for result in required_results)
        else None
    )
    return {
        "api": api,
        "expectation": expectation,
        "parameter_variation": parameter,
        "timing_variation": timing,
        "unstable_positions": {str(key): sorted(value) for key, value in mask.items()},
        "score": score,
        "excluded": False,
    }


def evaluate_scenario_discriminability(
    ground_truth: dict[str, Any], invocations: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped = group_invocations_by_api(invocations)
    results = [
        evaluate_api_discriminability(api, expectation, grouped.get(api, []))
        for api, expectation in ground_truth.get("api_variation_expectations", {}).items()
    ]
    sensitive = [row for row in results if not row["excluded"]]
    evaluable = [row for row in sensitive if row["score"] is not None]
    return {
        "apis": results,
        "expected_sensitive_apis": [row["api"] for row in sensitive],
        "evaluable_sensitive_apis": [row["api"] for row in evaluable],
        "insufficient_evidence_apis": [row["api"] for row in sensitive if row["score"] is None],
        "satisfied_apis": [row["api"] for row in evaluable if row["score"] == 1],
        "failed_apis": [row["api"] for row in evaluable if row["score"] == 0],
        "fixed_apis_excluded": [row["api"] for row in results if row["excluded"]],
        "score": sum(row["score"] for row in evaluable) / len(evaluable) if evaluable else None,
    }


def evaluate_semantic_consistency(invocations: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in invocations if row.get("valid", True)]
    aligned: list[dict[str, Any]] = []
    misaligned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in valid:
        checks = row.get("alignment_checks")
        if isinstance(checks, dict):
            is_aligned = bool(checks) and all(value is True for value in checks.values())
            failed = [name for name, value in checks.items() if value is not True]
        else:
            is_aligned = row.get("aligned") is True
            failed = [] if is_aligned else ["alignment_not_proven"]
        target = aligned if is_aligned else misaligned
        target.append(row)
        if failed:
            errors.append({"invocation_id": row.get("invocation_id"), "api": row.get("api"), "failed_checks": failed})
    score = len(aligned) / len(valid) if valid else None
    return {
        "valid_invocations": len(valid),
        "aligned_invocations": len(aligned),
        "misaligned_invocations": len(misaligned),
        "alignment_errors": errors,
        "semantic_consistency": score,
        "score": score,
    }


def _transaction_path(run_dir: Path) -> Path | None:
    for relative in TRANSACTION_CANDIDATES:
        path = run_dir / relative
        if path.exists():
            return path
    return None


def _pcap_path(run_dir: Path) -> Path | None:
    for relative in PCAP_CANDIDATES:
        path = run_dir / relative
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _return_codes_equal(left: Any, right: Any) -> bool:
    aliases = {"EW_OK": "0"}
    return aliases.get(str(left), str(left)) == aliases.get(str(right), str(right))


def _has_high_resolution_timestamp(value: Any) -> bool:
    text = str(value or "")
    return bool(re.search(r"T\d{2}:\d{2}:\d{2}\.\d+", text))


def _transaction_order_matches(transaction: dict[str, Any], log: dict[str, Any], order: int) -> bool:
    transaction_timestamp = transaction.get("log_timestamp", transaction.get("invocation_timestamp"))
    if transaction_timestamp is not None:
        return str(transaction_timestamp) == str(log.get("timestamp"))
    transaction_order = transaction.get("invocation_order", transaction.get("order"))
    if transaction_order is not None:
        try:
            return int(transaction_order) == order
        except (TypeError, ValueError):
            return False
    return False


def load_run_invocations(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logs = load_jsonl(run_dir / "execution" / "api_logs.jsonl")
    transaction_path = _transaction_path(run_dir)
    transactions = load_jsonl(transaction_path) if transaction_path else []
    transactions_by_id = {
        str(row.get("invocation_id", row.get("call_id"))): row
        for row in transactions
        if row.get("invocation_id", row.get("call_id")) is not None
    }
    pcap_path = _pcap_path(run_dir)
    concrete_logs = [row for row in logs if EXACT_API.fullmatch(str(row.get("protocol_function", "")))]
    invalid_api_logs = [row for row in logs if row not in concrete_logs]
    invocations: list[dict[str, Any]] = []
    support = {
        "api_name": bool(logs) and all(bool(row.get("protocol_function")) for row in logs),
        "concrete_api_identity": bool(concrete_logs),
        "input_parameters": bool(logs) and all(isinstance(row.get("input_parameters"), dict) for row in logs),
        "return_code": bool(logs) and all(row.get("status_code") is not None for row in logs),
        "timestamp": bool(logs) and all(bool(row.get("timestamp")) for row in logs),
        "high_resolution_timestamp": bool(logs) and all(
            (
                row.get("start_timestamp_ns") is not None
                and row.get("end_timestamp_ns") is not None
            )
            or _has_high_resolution_timestamp(row.get("timestamp"))
            for row in logs
        ),
        "invocation_id": bool(logs) and all(row.get("invocation_id", row.get("call_id")) is not None for row in logs),
        "timing": bool(logs) and all(bool(row.get("phase")) for row in logs),
        "device_context": bool(logs) and all(bool(row.get("device_context")) for row in logs),
        "annotation": bool(logs) and all(bool(row.get("semantic_label")) for row in logs),
        "pcap_file": pcap_path is not None,
        "request_response_transaction": bool(transactions) and all(
            row.get("invocation_id") is not None for row in transactions
        ),
        "traffic_observation_complete": bool(transactions) and all(
            row.get("traffic_observation_complete") is True for row in transactions
        ),
        "application_payload": bool(transactions) and any(
            build_protocol_sample(row) is not None for row in transactions
        ),
        "tcp_reassembly_metadata": bool(transactions) and all(row.get("tcp_reassembled") is True for row in transactions),
    }
    for order, log in enumerate(logs):
        api = str(log.get("protocol_function", ""))
        invocation_id_value = log.get("invocation_id", log.get("call_id"))
        invocation_id = str(invocation_id_value) if invocation_id_value is not None else None
        concrete_api = bool(EXACT_API.fullmatch(api))
        transaction = transactions_by_id.get(invocation_id or "")
        sample = build_protocol_sample(transaction or {})
        log_params = log.get("input_parameters", {})
        checks = {
            "api_identity": bool(transaction) and str(transaction.get("api", transaction.get("api_name", ""))) == api,
            "transaction_identity": bool(transaction)
            and log.get("transaction_id") is not None
            and transaction.get("transaction_id") == log.get("transaction_id"),
            "input_parameters": bool(transaction) and "decoded_parameters" in transaction and normalize_parameters(transaction.get("decoded_parameters")) == normalize_parameters(log_params),
            "return_code": bool(transaction) and _return_codes_equal(
                transaction.get("decoded_return_code"), log.get("status_code")
            ),
            "invocation_timestamp_order": bool(transaction)
            and invocation_id is not None
            and _transaction_order_matches(transaction, log, order),
            "request_response_association": bool(transaction) and transaction.get("request_response_associated") is True,
            "captured_traffic_transaction": sample is not None,
            "annotation": bool(transaction) and bool(log.get("semantic_label")) and transaction.get("annotation") == log.get("semantic_label"),
        }
        invocations.append({
            "invocation_id": invocation_id,
            "api": api,
            "parameters": log_params,
            "return_code": log.get("status_code"),
            "timestamp": log.get("timestamp"),
            "order": order,
            "timing": log.get("phase", ""),
            "context": log.get("device_context", {}),
            "annotation": log.get("semantic_label"),
            "protocol_sample": sample,
            "alignment_checks": checks,
            "valid": concrete_api and log.get("status_code") is not None,
            "invalid_reason": None if concrete_api else "not one concrete API invocation",
        })
    support["sufficient_for_parameter_comparison"] = (
        support["input_parameters"] and support["application_payload"]
        and support["invocation_id"] and support["timing"] and support["device_context"]
    )
    support["sufficient_for_timing_comparison"] = (
        support["timing"] and support["application_payload"]
        and support["invocation_id"] and support["device_context"]
    )
    support["sufficient_for_semantic_consistency"] = all(
        support[key]
        for key in (
            "api_name", "concrete_api_identity", "input_parameters", "return_code", "timestamp",
            "high_resolution_timestamp", "invocation_id", "timing", "device_context", "annotation",
            "request_response_transaction", "traffic_observation_complete", "tcp_reassembly_metadata",
        )
    )
    support["transaction_source"] = str(transaction_path) if transaction_path else None
    support["pcap_source"] = str(pcap_path) if pcap_path else None
    support["concrete_api_log_count"] = len(concrete_logs)
    support["non_focas_or_compound_log_count"] = len(invalid_api_logs)
    support["missing_fields"] = [key for key, value in support.items() if isinstance(value, bool) and not value]
    return invocations, support


def infer_scenario(run_dir: Path) -> str:
    plan_path = run_dir / "plan.json"
    if plan_path.exists():
        plan = load_json(plan_path)
        scenario = plan.get("scenario_type") or plan.get("scenario")
        if scenario in {"coordinate_motion", "program_management", "tool_management"}:
            return str(scenario)
    for part in reversed(run_dir.parts):
        if part in {"coordinate_motion", "program_management", "tool_management"}:
            return part
    raise ValueError(f"Cannot infer supported scenario for {run_dir}")


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    scenario = infer_scenario(run_dir)
    ground_truth = load_ground_truth(scenario)
    invocations, support = load_run_invocations(run_dir)
    coverage = evaluate_api_coverage(ground_truth, invocations)
    discriminability = evaluate_scenario_discriminability(ground_truth, invocations)
    semantic = evaluate_semantic_consistency(invocations)
    if not support["traffic_observation_complete"] or not support["request_response_transaction"]:
        coverage["score"] = None
        coverage["coverage"] = None
        coverage["measurement_status"] = "insufficient_traffic_transaction_evidence"
        discriminability["score"] = None
        discriminability["measurement_status"] = "insufficient_application_payload_evidence"
    if not support["sufficient_for_semantic_consistency"]:
        semantic["score"] = None
        semantic["semantic_consistency"] = None
        semantic["measurement_status"] = "insufficient_alignment_fields"
    return {
        "schema_version": "1.0",
        "scenario": scenario,
        "run_dir": str(run_dir),
        "api_coverage": coverage,
        "discriminability": discriminability,
        "semantic_consistency": semantic,
        "format_support": support,
        "notes": [
            "No Overall Dataset Quality score is computed.",
            "Coverage requires successful execution and a real application-layer protocol sample.",
            "Reconstructed capture_events without packet payloads are not treated as traffic evidence.",
        ],
    }


def find_run_dirs(path: Path) -> list[Path]:
    if path.name.startswith("run_") and path.is_dir():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("run_*")
        if candidate.is_dir()
        and (
            (candidate / "summary.json").exists()
            or (candidate / "execution" / "api_logs.jsonl").exists()
        )
    )


def macro_average(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def evaluate_dataset(path: Path) -> dict[str, Any]:
    run_results: list[dict[str, Any]] = []
    by_scenario_invocations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scenario_support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_dir in find_run_dirs(path):
        result = evaluate_run(run_dir)
        run_results.append(result)
        invocations, run_support = load_run_invocations(run_dir)
        by_scenario_invocations[result["scenario"]].extend(invocations)
        by_scenario_support[result["scenario"]].append(run_support)
        (run_dir / "dataset_quality.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    scenario_results: dict[str, Any] = {}
    for scenario in ("coordinate_motion", "program_management", "tool_management"):
        rows = by_scenario_invocations.get(scenario, [])
        if not rows:
            scenario_results[scenario] = None
            continue
        ground_truth = load_ground_truth(scenario)
        supports = by_scenario_support[scenario]
        coverage = evaluate_api_coverage(ground_truth, rows)
        discriminability = evaluate_scenario_discriminability(ground_truth, rows)
        semantic = evaluate_semantic_consistency(rows)
        if not any(
            item["request_response_transaction"] and item["traffic_observation_complete"]
            for item in supports
        ):
            coverage["score"] = None
            coverage["coverage"] = None
            coverage["measurement_status"] = "insufficient_traffic_transaction_evidence"
            discriminability["score"] = None
            discriminability["measurement_status"] = "insufficient_application_payload_evidence"
        if not all(item["sufficient_for_semantic_consistency"] for item in supports):
            semantic["score"] = None
            semantic["semantic_consistency"] = None
            semantic["measurement_status"] = "insufficient_alignment_fields"
        scenario_results[scenario] = {
            "api_coverage": coverage,
            "discriminability": discriminability,
            "semantic_consistency": semantic,
        }
    metric_names = ("api_coverage", "discriminability", "semantic_consistency")
    macro: dict[str, Any] = {}
    for metric in metric_names:
        scores = {
            scenario: (value[metric]["score"] if value else None)
            for scenario, value in scenario_results.items()
        }
        present = [score for score in scores.values() if score is not None]
        macro[metric] = {
            "scenario_scores": scores,
            "evaluable_scenario_count": len(present),
            "score": sum(present) / 3 if len(present) == 3 else None,
            "macro_over_evaluable_scenarios": sum(present) / len(present) if present else None,
        }
    return {
        "schema_version": "1.0",
        "runs": run_results,
        "scenarios": scenario_results,
        "dataset_macro": macro,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="One run directory or a root containing run_* directories.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_dataset(args.path)
    output = args.output or (args.path / "dataset_quality_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
