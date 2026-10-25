from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..knowledge import KnowledgeBase
from ..models import RetrievedChunk
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class KnowledgeSearchInput:
    query: str
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KnowledgeSearchOutput:
    results: list[RetrievedChunk]
    retrieval_method: str


class KnowledgeSearchTool(AgentTool[KnowledgeSearchInput, KnowledgeSearchOutput]):
    name = "search_knowledge"
    description = "Search the configured RAG knowledge base with optional metadata filters."

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def invoke(self, tool_input: KnowledgeSearchInput) -> KnowledgeSearchOutput:
        filters = dict(tool_input.filters or {})
        query = tool_input.query
        top_k = max(1, min(int(tool_input.top_k), 50))
        if filters.get("source_type") == "api" and hasattr(self.knowledge_base, "search_api"):
            results = self.knowledge_base.search_api(query, top_k=top_k)
        elif filters.get("source_type") == "rule" and hasattr(self.knowledge_base, "search_rules"):
            results = self.knowledge_base.search_rules(query, top_k=top_k)
        elif filters.get("source_type") == "alarm" and hasattr(self.knowledge_base, "search_alarms"):
            results = self.knowledge_base.search_alarms(query, top_k=top_k)
        elif filters.get("rule_type") and hasattr(self.knowledge_base, "search_rules_by_type"):
            results = self.knowledge_base.search_rules_by_type(
                query, str(filters["rule_type"]), top_k=top_k
            )
        elif filters.get("scenario") and hasattr(self.knowledge_base, "search_by_scenario"):
            results = self.knowledge_base.search_by_scenario(
                query, str(filters["scenario"]), top_k=top_k
            )
        else:
            results = self.knowledge_base.search(query, top_k=top_k)
        method = "embedding" if getattr(self.knowledge_base, "vector_search_available", False) else "term_fallback"
        return KnowledgeSearchOutput(results, method)

    def input_summary(self, tool_input: KnowledgeSearchInput) -> dict[str, Any]:
        return {"query": tool_input.query[:500], "top_k": tool_input.top_k, "filters": tool_input.filters}

    def output_summary(self, tool_output: KnowledgeSearchOutput) -> dict[str, Any]:
        return {
            "result_count": len(tool_output.results),
            "chunk_ids": [item.chunk.chunk_id for item in tool_output.results],
            "retrieval_method": tool_output.retrieval_method,
        }
