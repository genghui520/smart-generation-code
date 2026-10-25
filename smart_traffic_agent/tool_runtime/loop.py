"""Model-to-tool loop used inside one Agent without changing LangGraph."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import inspect
from typing import Any, Callable

from ..llm import LlmClient
from .runtime import ToolRuntime


class ToolCallLoopError(RuntimeError):
    """Raised when an Agent tool loop cannot produce a final response."""


ToolInputFactory = Callable[[str, dict[str, Any]], Any]


class ToolCallLoop:
    """Run bounded LLM -> tool_call -> tool_result turns for one Agent."""

    def __init__(self, llm_client: LlmClient, tool_runtime: ToolRuntime, *, max_turns: int = 6) -> None:
        self.llm_client = llm_client
        self.tool_runtime = tool_runtime
        self.max_turns = max(1, max_turns)
        self.last_tool_outputs: list[tuple[str, Any, Any]] = []

    def run(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        *,
        input_factory: ToolInputFactory,
    ) -> dict[str, Any]:
        transcript = user_prompt
        self.last_tool_outputs = []
        for turn in range(self.max_turns):
            invoke_json = self.llm_client.invoke_json
            try:
                supports_agent_name = "agent_name" in inspect.signature(invoke_json).parameters
            except (TypeError, ValueError):
                supports_agent_name = True
            if supports_agent_name:
                payload = invoke_json(system_prompt, transcript, agent_name=agent_name)
            else:
                payload = invoke_json(system_prompt, transcript)
            calls = self._extract_calls(payload)
            if not calls:
                return payload

            results: list[dict[str, Any]] = []
            for index, call in enumerate(calls, 1):
                tool_name = str(call.get("tool_name", call.get("name", ""))).strip()
                arguments = call.get("arguments", call.get("input", {}))
                if not tool_name or not isinstance(arguments, dict):
                    raise ToolCallLoopError(f"Invalid tool call returned by {agent_name}: {call}")
                tool_input = input_factory(tool_name, arguments)
                output, record = self.tool_runtime.call(agent_name, tool_name, tool_input)
                self.last_tool_outputs.append((tool_name, output, record))
                results.append(
                    {
                        "call_index": index,
                        "tool_name": tool_name,
                        "success": record.success,
                        "record": record.to_dict(),
                        "output": serialize_tool_output(output),
                    }
                )
            transcript = (
                f"{user_prompt}\n\n"
                f"Tool loop turn {turn + 1} results (use these results and either call another tool or return final JSON):\n"
                f"{results}"
            )
        raise ToolCallLoopError(
            f"{agent_name} exceeded the maximum tool-loop turns ({self.max_turns})."
        )

    @staticmethod
    def _extract_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload.get("tool_calls"), list):
            return [item for item in payload["tool_calls"] if isinstance(item, dict)]
        if payload.get("type") == "tool_call" or payload.get("tool_name") or payload.get("name"):
            return [payload]
        return []


def serialize_tool_output(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [serialize_tool_output(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_tool_output(item) for key, item in value.items()}
    return value
