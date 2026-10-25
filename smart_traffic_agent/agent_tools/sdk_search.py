"""Read-only, query-driven search over the configured official FOCAS header."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..integrations.ncguide import default_focas_header_dir
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class FocasSdkSearchInput:
    query: str
    top_k: int = 8


@dataclass(slots=True)
class FocasSdkSearchOutput:
    query: str
    header_path: Path
    snippets: list[str] = field(default_factory=list)


class FocasSdkSearchTool(AgentTool[FocasSdkSearchInput, FocasSdkSearchOutput]):
    name = "search_focas_sdk"
    description = "Search the official Fwlib32.h using a query chosen by the calling agent."

    def __init__(self, header_path: Path | None = None) -> None:
        self.header_path = header_path or (default_focas_header_dir() / "Fwlib32.h")

    def invoke(self, tool_input: FocasSdkSearchInput) -> FocasSdkSearchOutput:
        if not self.header_path.exists():
            return FocasSdkSearchOutput(tool_input.query, self.header_path, [])
        text = self.header_path.read_text(encoding="latin-1", errors="replace")
        query_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", tool_input.query)
        }
        exact_symbols = {
            token.lower()
            for token in query_terms
            if token.startswith(("cnc_", "pmc_"))
        }
        blocks = re.split(r"(?<=;)[\r\n]+", text)
        scored: list[tuple[int, int, str]] = []
        for index, block in enumerate(blocks):
            lowered = block.lower()
            exact_score = 20 * sum(symbol in lowered for symbol in exact_symbols)
            term_score = sum(term in lowered for term in query_terms)
            score = exact_score + term_score
            if score:
                scored.append((score, -index, block.strip()[:2400]))
        scored.sort(reverse=True)
        top_k = max(1, min(int(tool_input.top_k), 30))
        return FocasSdkSearchOutput(
            tool_input.query,
            self.header_path,
            [block for _score, _index, block in scored[:top_k] if block],
        )

    def input_summary(self, tool_input: FocasSdkSearchInput) -> dict[str, object]:
        return {"query": tool_input.query[:500], "top_k": tool_input.top_k}

    def output_summary(self, tool_output: FocasSdkSearchOutput) -> dict[str, object]:
        return {
            "query": tool_output.query[:500],
            "header_path": str(tool_output.header_path),
            "snippet_count": len(tool_output.snippets),
        }
