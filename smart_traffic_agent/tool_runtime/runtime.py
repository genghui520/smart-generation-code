from __future__ import annotations

import time
from typing import Any

from ..agent_tools.base import AgentTool, ToolProgressCallback, invoke_tool
from ..models import ToolCallRecord, utc_now
from .policy import check_permission
from .registry import ToolRegistry


class ToolExecutionError(RuntimeError):
    """Raised when a registered tool fails during invocation."""

    def __init__(self, agent_name: str, tool_name: str, cause: Exception) -> None:
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool {tool_name} failed for {agent_name}: {cause}")


class ToolRuntime:
    """Common policy, execution, progress, and audit boundary for AgentTool."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        progress_callback: ToolProgressCallback | None = None,
        history: list[ToolCallRecord] | None = None,
    ) -> None:
        self.registry = registry
        self.progress_callback = progress_callback
        self.history = history if history is not None else []

    def call(self, agent_name: str, tool_name: str, tool_input: Any) -> tuple[Any, ToolCallRecord]:
        check_permission(agent_name, tool_name)
        tool: AgentTool = self.registry.get(tool_name)
        try:
            output, record = invoke_tool(
                tool,
                tool_input,
                self.progress_callback,
                agent_name=agent_name,
            )
        except Exception as exc:
            # invoke_tool already emitted the progress event and preserved the
            # original failure. Runtime adds the same failure to shared audit.
            input_summary = tool.input_summary(tool_input)
            record = ToolCallRecord(
                tool_name=tool_name,
                started_at=utc_now(),
                completed_at=utc_now(),
                duration_ms=0,
                success=False,
                input_summary=input_summary,
                error=str(exc),
                agent_name=agent_name,
            )
            self.history.append(record)
            raise ToolExecutionError(agent_name, tool_name, exc) from exc
        self.history.append(record)
        return output, record
