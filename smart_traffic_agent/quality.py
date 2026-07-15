from __future__ import annotations

import re
from typing import Any

from .models import ApiCallLog, ExecutionPlan, ExecutionResult, QualityAssessment


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
AXIS_NAME_RE = re.compile(r"(?:^|[_-])([xyz])(?:$|[_-])", re.IGNORECASE)
AXIS_INDEX_RE = re.compile(r"axis[_-]?(\d+)", re.IGNORECASE)


def build_quality_observation(plan: ExecutionPlan, result: ExecutionResult) -> QualityAssessment:
    metrics = collect_quality_metrics(result.api_logs)
    return QualityAssessment(
        passed=True,
        issues=[],
        metrics=metrics,
        recommendations=[
            "RouterAgent must evaluate these objective metrics against the PlannerAgent quality targets."
        ],
    )


def output_variation_is_sufficient(metrics: dict[str, Any]) -> bool:
    """Return whether collected outputs show useful traffic variation.

    This is an objective guardrail for RouterAgent: input parameter sweeps alone
    do not count. We require returned feed, position, and run/motion evidence.
    """
    return (
        bool(metrics.get("program_completed"))
        and int(metrics.get("changed_output_parameter_count") or 0) >= 3
        and int(metrics.get("feed_sample_count") or 0) >= 5
        and int(metrics.get("feed_unique_count") or 0) > 1
        and int(metrics.get("position_sample_count") or 0) >= 5
        and int(metrics.get("position_unique_count") or 0) > 1
        and (
            int(metrics.get("run_active_count") or 0) > 0
            or int(metrics.get("motion_active_count") or 0) > 0
        )
    )


def assess_traffic_quality(plan: ExecutionPlan, result: ExecutionResult) -> QualityAssessment:
    return build_quality_observation(plan, result)


def collect_quality_metrics(api_logs: list[ApiCallLog]) -> dict[str, Any]:
    feeds: list[float] = []
    positions: list[tuple[float, float, float]] = []
    run_values: list[float] = []
    motion_values: list[float] = []
    program_completion_gate_count = 0
    program_completed = False
    input_parameter_names: dict[str, list[str]] = {}
    return_parameter_names: dict[str, list[str]] = {}
    output_values: dict[str, dict[str, list[float | str]]] = {}

    for log in api_logs:
        input_text = input_data_text(log)
        data = response_data_text(log)
        fields = parse_semicolon_fields(data)
        record_parameter_names(input_parameter_names, log.interface_name, parse_semicolon_pairs(input_text))
        output_pairs = parse_semicolon_pairs(data)
        record_parameter_names(return_parameter_names, log.interface_name, output_pairs)
        record_output_values(output_values, log.interface_name, output_pairs)
        if log.interface_name == "ReadFeedSpeed":
            value = first_feed_value(input_text, data)
            if value is not None:
                feeds.append(value)
        if log.interface_name == "ReadPosition":
            xyz = first_position_tuple(input_text, data)
            if xyz is not None:
                positions.append(xyz)
        if log.interface_name == "ReadRunStatus":
            run = numeric_field(fields, "run")
            motion = numeric_field(fields, "motion")
            if run is not None:
                run_values.append(run)
            if motion is not None:
                motion_values.append(motion)
        if is_program_completion_gate_log(log, data):
            program_completion_gate_count += 1
            if program_completion_log_completed(fields, data):
                program_completed = True

    return {
        "api_log_count": len(api_logs),
        "feed_sample_count": len(feeds),
        "feed_unique_count": len(set(feeds)),
        "feed_values_preview": feeds[:10],
        "position_sample_count": len(positions),
        "position_unique_count": len(set(positions)),
        "position_values_preview": positions[:5],
        "run_active_count": sum(1 for value in run_values if value != 0),
        "motion_active_count": sum(1 for value in motion_values if value != 0),
        "program_completion_gate_count": program_completion_gate_count,
        "program_completed": program_completed,
        "input_parameter_names": input_parameter_names,
        "return_parameter_names": return_parameter_names,
        "output_parameter_variation": summarize_output_variation(output_values),
        "changed_output_parameter_count": count_changed_output_parameters(output_values),
    }


def input_data_text(log: ApiCallLog) -> str:
    raw = log.input_parameters.get("raw", "")
    if raw:
        return str(raw)
    return ";".join(f"{key}={value}" for key, value in log.input_parameters.items())


def response_data_text(log: ApiCallLog) -> str:
    data = log.response.get("data", "")
    if isinstance(data, str):
        return data
    return str(data)


def is_program_completion_gate_log(log: ApiCallLog, data: str) -> bool:
    haystack = " ".join(
        [
            log.step_id,
            log.interface_name,
            log.protocol_function,
            data,
        ]
    ).lower()
    return "program_completion_gate" in haystack or "waituntilprogramcomplete" in haystack


def program_completion_log_completed(fields: dict[str, str], data: str) -> bool:
    lowered = data.lower()
    if any(token in lowered for token in ["timeout=true", "timeout=1", "completed=false", "complete=false"]):
        return False
    if any(token in lowered for token in ["completed=true", "complete=true", "program_completed=true"]):
        return True
    run = numeric_field(fields, "run")
    motion = numeric_field(fields, "motion")
    if run is None:
        run = numeric_field(fields, "last_run")
    if motion is None:
        motion = numeric_field(fields, "last_motion")
    if run is None or motion is None or run != 0 or motion != 0:
        return False
    remaining_values = [
        numeric_field(fields, key)
        for key in [
            "distance_to_go",
            "remaining_distance",
            "remaining_move",
            "distance_to_go_axis1",
            "distance_to_go_axis2",
            "distance_to_go_axis3",
            "dist_axis1",
            "dist_axis2",
            "dist_axis3",
        ]
    ]
    numeric_remaining = [value for value in remaining_values if value is not None]
    return not numeric_remaining or all(abs(value) <= 0.001 for value in numeric_remaining)


def parse_semicolon_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in parse_semicolon_pairs(text):
        fields[key.strip()] = value.strip()
    return fields


def parse_semicolon_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def numeric_field(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if value is None:
        return None
    match = NUMBER_RE.search(value)
    if match is None:
        return None
    return float(match.group(0))


def first_feed_value(input_text: str, output_text: str) -> float | None:
    input_names = {key.lower() for key, _ in parse_semicolon_pairs(input_text)}
    values: list[float] = []
    for key, value in parse_semicolon_pairs(output_text):
        key_lower = key.lower()
        if is_metadata_key(key_lower) or key_lower in input_names:
            continue
        parsed = number_from_text(value)
        if parsed is not None:
            values.append(parsed)
    nonzero = [value for value in values if value != 0]
    if nonzero:
        return nonzero[-1]
    if values:
        return values[-1]
    return None


def first_position_tuple(input_text: str, output_text: str) -> tuple[float, float, float] | None:
    pairs = parse_semicolon_pairs(output_text)
    fields = parse_semicolon_fields(output_text)
    requested_axes = requested_axis_ids(input_text)
    axis_values: dict[str, list[float]] = {}
    axis_order: list[str] = []
    current_axis: str | None = None
    generated_axis_index = 0

    for key, value in pairs:
        key_lower = key.lower()
        direct_axis = axis_id_from_key(key_lower)
        if direct_axis is not None:
            current_axis = direct_axis
            register_axis(axis_order, current_axis)

        if key_lower.endswith("name"):
            named_axis = axis_id_from_value(value)
            if named_axis is not None:
                current_axis = named_axis
                register_axis(axis_order, current_axis)
                continue
            indexed_axis = axis_id_from_key(key_lower)
            if indexed_axis is not None:
                current_axis = indexed_axis
                register_axis(axis_order, current_axis)
                continue

        parsed = scaled_number_from_named_field(key_lower, value, fields)
        if parsed is None or is_metadata_key(key_lower):
            continue

        axis_id = direct_axis or current_axis
        if axis_id is None and looks_like_position_key(key_lower):
            generated_axis_index += 1
            axis_id = str(generated_axis_index)
            register_axis(axis_order, axis_id)
        if axis_id is not None and looks_like_position_key(key_lower):
            axis_values.setdefault(axis_id, []).append(parsed)

    ordered_requested_axes = [axis for axis in requested_axes if axis in axis_values]
    if len(ordered_requested_axes) >= 3:
        return tuple(representative_axis_value(axis_values[axis]) for axis in ordered_requested_axes[:3])  # type: ignore[return-value]

    if all(axis in axis_values for axis in ("x", "y", "z")):
        return tuple(representative_axis_value(axis_values[axis]) for axis in ("x", "y", "z"))  # type: ignore[return-value]

    indexed_axes = [axis for axis in axis_order if axis in axis_values and axis.isdigit()]
    if len(indexed_axes) >= 3:
        first_three = sorted(indexed_axes, key=int)[:3]
        return tuple(representative_axis_value(axis_values[axis]) for axis in first_three)  # type: ignore[return-value]

    ordered_axes = [axis for axis in axis_order if axis in axis_values]
    if len(ordered_axes) >= 3:
        return tuple(representative_axis_value(axis_values[axis]) for axis in ordered_axes[:3])  # type: ignore[return-value]

    plain_values = [
        parsed
        for key, value in pairs
        if looks_like_position_key(key.lower())
        for parsed in [scaled_number_from_named_field(key.lower(), value, fields)]
        if parsed is not None and not is_metadata_key(key.lower())
    ]
    if len(plain_values) >= 3:
        return tuple(plain_values[:3])  # type: ignore[return-value]
    return None


def record_parameter_names(
    grouped: dict[str, list[str]],
    interface_name: str,
    pairs: list[tuple[str, str]],
) -> None:
    names = grouped.setdefault(interface_name, [])
    for key, _ in pairs:
        if key not in names:
            names.append(key)


def record_output_values(
    grouped: dict[str, dict[str, list[float | str]]],
    interface_name: str,
    pairs: list[tuple[str, str]],
) -> None:
    interface_values = grouped.setdefault(interface_name, {})
    fields = {key.lower(): value for key, value in pairs}
    for key, value in pairs:
        key_lower = key.lower()
        if is_metadata_key(key_lower):
            continue
        parsed = scaled_number_from_named_field(key_lower, value, fields)
        stored: float | str
        if parsed is not None:
            stored = parsed
        else:
            stored = value.strip()
        interface_values.setdefault(key, []).append(stored)


def summarize_output_variation(
    grouped: dict[str, dict[str, list[float | str]]],
) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for interface_name, parameter_values in grouped.items():
        interface_summary: dict[str, dict[str, object]] = {}
        for parameter_name, values in parameter_values.items():
            unique_values = unique_preserving_order(values)
            interface_summary[parameter_name] = {
                "sample_count": len(values),
                "unique_count": len(unique_values),
                "changed": len(unique_values) > 1,
                "values_preview": unique_values[:5],
            }
        if interface_summary:
            summary[interface_name] = interface_summary
    return summary


def count_changed_output_parameters(grouped: dict[str, dict[str, list[float | str]]]) -> int:
    count = 0
    for parameter_values in grouped.values():
        for values in parameter_values.values():
            if len(set(values)) > 1:
                count += 1
    return count


def unique_preserving_order(values: list[float | str]) -> list[float | str]:
    seen: set[float | str] = set()
    rows: list[float | str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def requested_axis_ids(input_text: str) -> list[str]:
    axes: list[str] = []
    for key, value in parse_semicolon_pairs(input_text):
        combined = f"{key} {value}".lower()
        for axis in ("x", "y", "z"):
            if re.search(rf"(?:^|[^a-z0-9]){axis}(?:$|[^a-z0-9])", combined) and axis not in axes:
                axes.append(axis)
        for match in AXIS_INDEX_RE.finditer(combined):
            axis = match.group(1)
            if axis not in axes:
                axes.append(axis)
    return axes


def number_from_text(value: str) -> float | None:
    match = NUMBER_RE.search(value)
    if match is None:
        return None
    return float(match.group(0))


def scaled_number_from_named_field(key: str, value: str, fields: dict[str, str]) -> float | None:
    parsed = number_from_text(value)
    if parsed is None:
        return None
    dec = number_from_text(fields.get(f"{key}_dec", ""))
    if dec is None:
        return parsed
    scale = int(dec)
    while scale > 0:
        parsed /= 10.0
        scale -= 1
    return parsed


def is_metadata_key(key: str) -> bool:
    return any(token in key for token in ["raw", "dummy", "status", "return", "ret", "error", "name"]) or key.endswith(
        "_dec"
    ) or key in {
        "axes",
        "axis_count",
        "count",
        "unit",
        "dec",
        "type",
    }


def looks_like_position_key(key: str) -> bool:
    if is_metadata_key(key):
        return False
    return any(token in key for token in ["axis", "pos", "abs", "mach", "rel", "dist", "x", "y", "z"])


def axis_id_from_key(key: str) -> str | None:
    index_match = AXIS_INDEX_RE.search(key)
    if index_match is not None:
        return index_match.group(1)
    name_match = AXIS_NAME_RE.search(key)
    if name_match is not None:
        return name_match.group(1).lower()
    if key in {"x", "y", "z"}:
        return key
    return None


def axis_id_from_value(value: str) -> str | None:
    axis = value.strip().lower()
    if axis in {"x", "y", "z"}:
        return axis
    return None


def register_axis(axis_order: list[str], axis_id: str) -> None:
    if axis_id not in axis_order:
        axis_order.append(axis_id)


def representative_axis_value(values: list[float]) -> float:
    if not values:
        return 0.0
    nonzero = [value for value in values if value != 0]
    if nonzero:
        return nonzero[0]
    return values[0]
