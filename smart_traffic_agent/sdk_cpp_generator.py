"""CLI-friendly SDK-to-C++ generation facade.

The input is the structured part of an ExecutionPlan.  This module deliberately
delegates C++ assembly to :func:`assemble_core_cpp`; it does not infer a
function name, argument order, SDK type, or buffer length from prose.
"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

from .deterministic_cpp_assembler import DeterministicAssemblyError, assemble_core_cpp


def _plan_steps(raw_steps: Any) -> list[SimpleNamespace]:
    if not isinstance(raw_steps, list):
        return []
    result: list[SimpleNamespace] = []
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}] must be an object")
        protocol = str(raw.get("protocol_function") or raw.get("api") or "").strip()
        if not protocol:
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}] lacks protocol_function")
        params = raw.get("parameters", {})
        if not isinstance(params, dict):
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}].parameters must be an object")
        result.append(SimpleNamespace(protocol_function=protocol, parameters=params))
    return result


def generate_core_cpp_from_plan(plan: dict[str, Any], *, sdk_header: str | Path | None = None) -> str:
    """Generate statements-only C++ from a JSON-compatible plan object."""
    if not isinstance(plan, dict):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: root must be an object")
    targets = plan.get("targets") or plan.get("target_apis")
    if not isinstance(targets, list) or not targets:
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: targets must be a non-empty array")
    planned = plan.get("planned_functions") or targets
    if not isinstance(planned, list):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: planned_functions must be an array")
    contract = plan.get("parameter_contract", {"functions": {}})
    if not isinstance(contract, dict):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: parameter_contract must be an object")
    return assemble_core_cpp(
        targets=[str(item) for item in targets],
        planned_functions=[str(item) for item in planned],
        steps=_plan_steps(plan.get("steps", [])),
        parameter_contract=contract,
        sdk_header=sdk_header,
    )


__all__ = ["generate_core_cpp_from_plan"]
