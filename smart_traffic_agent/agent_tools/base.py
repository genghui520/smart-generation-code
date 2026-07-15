from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from ..models import ToolCallRecord, utc_now


ToolInput = TypeVar("ToolInput")
ToolOutput = TypeVar("ToolOutput")
ToolProgressCallback = Callable[[str, dict[str, Any]], None]


class AgentTool(ABC, Generic[ToolInput, ToolOutput]):
    """Small, testable unit of deterministic work invoked by an Agent."""

    name: str
    description: str

    @abstractmethod
    def invoke(self, tool_input: ToolInput) -> ToolOutput:
        raise NotImplementedError

    def input_summary(self, tool_input: ToolInput) -> dict[str, Any]:
        return {}

    def output_summary(self, tool_output: ToolOutput) -> dict[str, Any]:
        return {}

    def output_succeeded(self, tool_output: ToolOutput) -> bool:
        return True

    def output_error(self, tool_output: ToolOutput) -> str | None:
        return None


def invoke_tool(
    tool: AgentTool[ToolInput, ToolOutput],
    tool_input: ToolInput,
    progress_callback: ToolProgressCallback | None = None,
) -> tuple[ToolOutput, ToolCallRecord]:
    """Invoke one tool and return both its typed output and an audit trace."""

    started_at = utc_now()
    started = time.perf_counter()
    input_summary = tool.input_summary(tool_input)
    if progress_callback is not None:
        progress_callback("tool_start", {"tool": tool.name, "input": input_summary})

    try:
        output = tool.invoke(tool_input)
        success = tool.output_succeeded(output)
        error = tool.output_error(output)
        output_summary = tool.output_summary(output)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        record = ToolCallRecord(
            tool_name=tool.name,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=duration_ms,
            success=False,
            input_summary=input_summary,
            output_summary={},
            error=str(exc),
        )
        if progress_callback is not None:
            progress_callback("tool_complete", record.to_dict())
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)
    record = ToolCallRecord(
        tool_name=tool.name,
        started_at=started_at,
        completed_at=utc_now(),
        duration_ms=duration_ms,
        success=success,
        input_summary=input_summary,
        output_summary=output_summary,
        error=error,
    )
    if progress_callback is not None:
        progress_callback("tool_complete", record.to_dict())
    return output, record
