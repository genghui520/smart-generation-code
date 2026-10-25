"""Shared helpers for paper-facing evaluation metrics."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _nonnegative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def normalize_token_usage(value: Any, *, source: str) -> dict[str, Any] | None:
    """Normalize persisted usage without inventing missing token data."""
    if not isinstance(value, Mapping):
        return None
    prompt = _nonnegative_int(value.get("prompt_tokens"))
    completion = _nonnegative_int(value.get("completion_tokens"))
    total = _nonnegative_int(value.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    request_count = _nonnegative_int(value.get("request_count"))
    if prompt is None and completion is None and total is None and request_count is None:
        return None
    return {
        "request_count": request_count,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "usage_source": source,
    }


def load_token_usage(*payloads: tuple[Mapping[str, Any] | None, str]) -> dict[str, Any] | None:
    """Return the first valid usage object from (payload, source) candidates."""
    for payload, source in payloads:
        if payload is None:
            continue
        usage = normalize_token_usage(payload.get("llm_usage"), source=source)
        if usage is not None:
            return usage
        usage = normalize_token_usage(payload.get("usage"), source=source)
        if usage is not None:
            return usage
    return None


def generation_time_seconds(metrics: Mapping[str, Any] | None) -> float | None:
    """Return planning plus code-generation time from persisted run metrics."""
    if not isinstance(metrics, Mapping):
        return None
    totals = metrics.get("agent_total_durations_seconds")
    if isinstance(totals, Mapping):
        if not any(name in totals for name in ("PlanningAgent", "CodeGenerationAgent")):
            return None
        values = [totals.get(name) for name in ("PlanningAgent", "CodeGenerationAgent")]
    else:
        durations = metrics.get("agent_durations_seconds")
        if not isinstance(durations, Mapping):
            return None
        if not any(name in durations for name in ("PlanningAgent", "CodeGenerationAgent")):
            return None
        values = []
        for name in ("PlanningAgent", "CodeGenerationAgent"):
            value = durations.get(name)
            values.append(sum(value) if isinstance(value, list) else value)
    try:
        return sum(float(value) for value in values if value is not None)
    except (TypeError, ValueError):
        return None


def generation_time_with_source(
    metrics: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[float | None, str | None]:
    """Return generation time and its evidence source.

    The metadata fallback is restricted to RQ1 ``--no-execute`` runs, where
    wrapper elapsed time covers code generation only (plus process overhead).
    """
    measured = generation_time_seconds(metrics)
    if measured is not None:
        return measured, "run_metrics.agent_total_durations_seconds"
    if isinstance(metadata, Mapping):
        command = metadata.get("command")
        no_execute = isinstance(command, list) and "--no-execute" in command
        if no_execute:
            try:
                elapsed = float(metadata.get("elapsed_seconds"))
            except (TypeError, ValueError):
                elapsed = None
            if elapsed is not None and elapsed >= 0:
                return elapsed, "rq1_run_metadata.elapsed_seconds_code_only"
    return None, None
