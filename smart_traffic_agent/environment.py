"""Shared MetaGPT-style environment for all workflow roles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .code_spec import build_official_abi_context
from .integrations.ncguide import default_focas_header_dir
from .knowledge import KnowledgeBase


class AgentEnvironment:
    """Shared knowledge and controller context visible to every Agent."""

    def __init__(self, knowledge_base: KnowledgeBase, header_dir: Path | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.header_dir = header_dir or default_focas_header_dir()
        self.shared_rules: dict[str, Any] = {
            "official_header": str(self.header_dir / "Fwlib32.h"),
            "source": "RAG plus configured official Fwlib32.h",
        }

    def official_abi(self, function_names: list[str]) -> str:
        return build_official_abi_context(function_names, self.header_dir / "Fwlib32.h")
