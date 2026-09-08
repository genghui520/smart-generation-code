"""Structured tool-call protocol shared by Planner and CodeGenerationAgent."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(slots=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    phase: str = "during"
    operation_id: str = ""


@dataclass(slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    return_code: int | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def tool_calls_from_step(step: Any) -> list[ToolCall]:
    """Convert a PlanStep into atomic tool calls without registry lookup."""
    rows = getattr(step, "api_calls", []) or []
    calls: list[ToolCall] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("protocol_function", row.get("tool_name", ""))).strip()
        if not tool_name:
            continue
        call_id = str(row.get("call_id", f"{step.step_id}-call-{index:02d}"))
        arguments = dict(row.get("parameters", row.get("arguments", {})) or {})
        # Planner may represent a host-side sequence explicitly. Normalize its
        # ordered members into the same atomic tool-call format as FOCAS APIs.
        if tool_name.lower().startswith("sequence:"):
            sequence = tool_name.split(":", 1)[1]
            members = [member.strip() for member in sequence.split("->") if member.strip()]
            for member_index, member in enumerate(members, 1):
                calls.append(
                    ToolCall(
                        call_id=f"{call_id}-{member_index:02d}",
                        tool_name=member,
                        arguments=arguments,
                        phase=step.phase,
                        operation_id=step.step_id,
                    )
                )
            continue
        if any(delimiter in tool_name for delimiter in ("/", "+", ";", ",")):
            members = [member.strip() for member in re.split(r"[/+;,]", tool_name) if member.strip()]
            for member_index, member in enumerate(members, 1):
                calls.append(
                    ToolCall(
                        call_id=f"{call_id}-{member_index:02d}",
                        tool_name=member,
                        arguments=arguments,
                        phase=step.phase,
                        operation_id=step.step_id,
                    )
                )
            continue
        calls.append(
            ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                phase=step.phase,
                operation_id=step.step_id,
            )
        )
    if not calls and getattr(step, "protocol_function", ""):
        tool_name = str(step.protocol_function).strip()
        arguments = dict(getattr(step, "parameters", {}) or {})
        if tool_name.lower().startswith("sequence:"):
            members = [member.strip() for member in tool_name.split(":", 1)[1].split("->") if member.strip()]
            calls.extend(
                ToolCall(
                    call_id=f"{step.step_id}-call-01-{member_index:02d}",
                    tool_name=member,
                    arguments=arguments,
                    phase=step.phase,
                    operation_id=step.step_id,
                )
                for member_index, member in enumerate(members, 1)
            )
        else:
            if any(delimiter in tool_name for delimiter in ("/", "+", ";", ",")):
                members = [member.strip() for member in re.split(r"[/+;,]", tool_name) if member.strip()]
                calls.extend(
                    ToolCall(
                        call_id=f"{step.step_id}-call-01-{member_index:02d}",
                        tool_name=member,
                        arguments=arguments,
                        phase=step.phase,
                        operation_id=step.step_id,
                    )
                    for member_index, member in enumerate(members, 1)
                )
                return calls
            calls.append(
                ToolCall(
                    call_id=f"{step.step_id}-call-01",
                    tool_name=tool_name,
                    arguments=arguments,
                    phase=step.phase,
                    operation_id=step.step_id,
                )
            )
    return calls


def validate_tool_calls(calls: list[ToolCall]) -> list[str]:
    errors: list[str] = []
    for call in calls:
        if any(delimiter in call.tool_name for delimiter in ("/", "+", ";", "->", ",")):
            errors.append(f"{call.call_id}: tool_name must contain exactly one API: {call.tool_name}")
        # Tool names are Planner-produced atomic identifiers.  FOCAS/PMC APIs
        # and NCGuide/operator helpers share this protocol; admission must not
        # depend on a locally maintained registry or prefix allow-list.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", call.tool_name):
            errors.append(f"{call.call_id}: invalid atomic tool name: {call.tool_name}")
    return errors
