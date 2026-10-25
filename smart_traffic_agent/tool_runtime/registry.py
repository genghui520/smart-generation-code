from __future__ import annotations

from ..agent_tools.base import AgentTool


class ToolNotFoundError(LookupError):
    """Raised when a tool name is not registered in the harness."""


class ToolRegistry:
    """Process-local registry shared by all agents in one workflow."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        name = str(getattr(tool, "name", "")).strip()
        if not name:
            raise ValueError("Cannot register a tool without a name.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, tool_name: str) -> AgentTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool is not registered: {tool_name}") from exc

    def list_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)
