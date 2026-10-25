"""Dynamic Agent Soul documents and evidence-backed experience access.

Core and skills are versioned files owned by the project. Runtime experience is
kept in the existing long-term memory store and is loaded on demand for one
agent and one task context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import LongTermMemoryStore


DEFAULT_SOUL_ROOT = Path("agent_souls")
AGENT_SOUL_NAMES = ("router", "planner", "code_generator", "executor")


@dataclass(frozen=True, slots=True)
class AgentSoul:
    agent_name: str
    core: str
    skills: dict[str, Any]
    skills_text: str
    source_paths: tuple[str, str]


class AgentSoulStore:
    """Load only the soul belonging to the requesting Agent."""

    def __init__(self, root: Path = DEFAULT_SOUL_ROOT) -> None:
        self.root = Path(root)

    def load(self, agent_name: str) -> AgentSoul:
        key = normalize_agent_name(agent_name)
        directory = self.root / key
        core_path = directory / "core.md"
        skills_path = directory / "skills.yaml"
        if not core_path.is_file() or not skills_path.is_file():
            raise FileNotFoundError(
                f"Agent Soul is incomplete for {key}: expected {core_path} and {skills_path}"
            )
        core = core_path.read_text(encoding="utf-8")
        skills_text = skills_path.read_text(encoding="utf-8")
        skills = parse_skills_yaml(skills_text)
        return AgentSoul(
            agent_name=key,
            core=core,
            skills=skills,
            skills_text=skills_text,
            source_paths=(str(core_path), str(skills_path)),
        )

    def prompt_context(
        self,
        agent_name: str,
        *,
        task_context: str = "",
        error_type: str = "",
        memory_store: "LongTermMemoryStore | None" = None,
        max_experiences: int = 5,
        max_chars: int = 10000,
    ) -> str:
        soul = self.load(agent_name)
        query = " ".join(item for item in (agent_name, task_context, error_type) if item).strip()
        experiences: list[dict[str, Any]] = []
        if memory_store is not None and query:
            search = getattr(memory_store, "search_agent_experiences", None)
            if callable(search):
                experiences = search(agent_name, query, top_k=max_experiences)
            else:
                experiences = [
                    row for row in memory_store.search(query, top_k=max_experiences)
                    if str(row.get("agent_name", "")).lower() in {"", normalize_agent_name(agent_name)}
                ]
        payload = (
            "# Agent Soul (immutable role guidance)\n"
            f"agent={soul.agent_name}\n"
            f"core:\n{soul.core.strip()}\n\n"
            "skills:\n"
            f"{soul.skills_text.strip()}\n\n"
            "# Evidence-backed Agent Experience (conditional reference)\n"
            f"{json.dumps(experiences, ensure_ascii=False, separators=(',', ':'))}\n"
            "Experience with status proposed is a hypothesis and cannot become a mandatory rule."
        )
        return payload[:max_chars]


def normalize_agent_name(agent_name: str) -> str:
    value = str(agent_name or "").strip().lower()
    aliases = {
        "routerrole": "router",
        "routeragent": "router",
        "planningrole": "planner",
        "planningagent": "planner",
        "codeengineerrole": "code_generator",
        "codegenerationagent": "code_generator",
        "executionrole": "executor",
        "executionagent": "executor",
    }
    value = aliases.get(value, value.replace("-", "_"))
    if value not in AGENT_SOUL_NAMES:
        raise ValueError(f"Unknown Agent Soul: {agent_name}")
    return value


def parse_skills_yaml(text: str) -> dict[str, Any]:
    """Parse skills YAML without making PyYAML a runtime requirement."""
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return payload if isinstance(payload, dict) else {"raw": text}
    except Exception:
        # The fallback preserves the complete source for prompt use and gives
        # callers a stable structured shape when optional YAML support is absent.
        return {"raw": text}


def soul_prompt_context(
    agent_name: str,
    *,
    task_context: str = "",
    error_type: str = "",
    memory_store: "LongTermMemoryStore | None" = None,
    root: Path = DEFAULT_SOUL_ROOT,
) -> str:
    return AgentSoulStore(root).prompt_context(
        agent_name,
        task_context=task_context,
        error_type=error_type,
        memory_store=memory_store,
    )
