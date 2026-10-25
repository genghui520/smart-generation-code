from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory import LongTermMemoryStore
from ..models import WorkflowState
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class MemorySearchInput:
    query: str
    top_k: int = 5


class MemorySearchTool(AgentTool[MemorySearchInput, list[dict[str, Any]]]):
    name = "search_memory"
    description = "Search persistent repair memory using embeddings with a term fallback."

    def __init__(self, memory_store: LongTermMemoryStore) -> None:
        self.memory_store = memory_store

    def invoke(self, tool_input: MemorySearchInput) -> list[dict[str, Any]]:
        return self.memory_store.search(tool_input.query, top_k=max(1, min(tool_input.top_k, 50)))

    def input_summary(self, tool_input: MemorySearchInput) -> dict[str, Any]:
        return {"query": tool_input.query[:500], "top_k": tool_input.top_k}

    def output_summary(self, tool_output: list[dict[str, Any]]) -> dict[str, Any]:
        return {"memory_count": len(tool_output), "memory_ids": [item.get("memory_id") for item in tool_output]}


class RememberWorkflowTool(AgentTool[WorkflowState, dict[str, Any]]):
    name = "remember_workflow"
    description = "Persist a completed workflow repair outcome in long-term memory."

    def __init__(self, memory_store: LongTermMemoryStore) -> None:
        self.memory_store = memory_store

    def invoke(self, tool_input: WorkflowState) -> dict[str, Any]:
        self.memory_store.remember_workflow(tool_input)
        return {"remembered": bool(tool_input.repair_history)}

    def input_summary(self, tool_input: WorkflowState) -> dict[str, Any]:
        return {"task_id": tool_input.request.task_id, "repair_attempts": tool_input.repair_attempts}

    def output_summary(self, tool_output: dict[str, Any]) -> dict[str, Any]:
        return dict(tool_output)
