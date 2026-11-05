"""Dynamic Agent Soul documents and evidence-backed experience access.

Core and skills are versioned files owned by the project. Runtime experience is
kept in the existing long-term memory store and is loaded on demand for one
agent and one task context.
"""
from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .models import utc_now

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
        evolution = self.search_evolution(
            agent_name,
            query,
            top_k=max_experiences,
        )
        payload = (
            "# Agent Soul (immutable role guidance)\n"
            f"agent={soul.agent_name}\n"
            f"core:\n{soul.core.strip()}\n\n"
            "skills:\n"
            f"{soul.skills_text.strip()}\n\n"
            "# Evidence-backed Agent Experience (conditional reference)\n"
            f"{json.dumps(experiences, ensure_ascii=False, separators=(',', ':'))}\n"
            "# Dynamic Agent Evolution (automatically updated from workflow evidence)\n"
            f"{json.dumps(evolution, ensure_ascii=False, separators=(',', ':'))}\n"
            "Experience with status proposed is a hypothesis and cannot become a mandatory rule. "
            "Only runtime_verified evolution may guide a repeated deterministic choice."
        )
        return payload[:max_chars]

    def evolution_path(self, agent_name: str) -> Path:
        """Return the append-only, agent-scoped learning ledger path."""

        return self.root / normalize_agent_name(agent_name) / "evolution.jsonl"

    def read_evolution(self, agent_name: str) -> list[dict[str, Any]]:
        path = self.evolution_path(agent_name)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and item.get("experience_id"):
                        rows.append(item)
        except OSError:
            return []
        return rows

    def search_evolution(self, agent_name: str, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve recent relevant self-updates without treating proposals as rules."""

        rows = self.read_evolution(agent_name)
        if not rows:
            return []
        terms = {part.lower() for part in str(query).split() if part}
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for row in rows:
            text = json.dumps(row, ensure_ascii=False).lower()
            score = sum(1 for term in terms if term in text)
            scored.append((score, str(row.get("timestamp", "")), row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [row for _, _, row in scored[: max(1, min(int(top_k), 20))]]

    def record_evolution(
        self,
        agent_name: str,
        evidence: dict[str, Any],
        *,
        verified: bool = False,
    ) -> dict[str, Any]:
        """Upsert one evidence-backed self-update for an Agent.

        The identity is stable for the same target/outcome, so repeated runs
        refine one record instead of endlessly appending duplicate lessons.
        """

        key = normalize_agent_name(agent_name)
        stable = dict(evidence)
        stable.pop("timestamp", None)
        canonical = json.dumps({"agent_name": key, "evidence": stable}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        experience_id = str(evidence.get("experience_id") or "soul-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20])
        rows = self.read_evolution(key)
        existing = next((row for row in rows if row.get("experience_id") == experience_id), None)
        status = "runtime_verified" if verified else "proposed"
        if existing and existing.get("status") == "runtime_verified":
            status = "runtime_verified"
        record = {
            "experience_id": experience_id,
            "agent_name": key,
            "record_type": "agent_soul_evolution",
            "status": status,
            "version": int(existing.get("version", 0)) + 1 if existing else 1,
            "timestamp": utc_now(),
            "evidence": evidence,
        }
        by_id = {str(row.get("experience_id")): row for row in rows}
        by_id[experience_id] = record
        path = self.evolution_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temp.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in by_id.values()),
            encoding="utf-8",
        )
        temp.replace(path)
        return record

    def learn_from_workflow(self, state: Any, *, workflow_success: bool) -> list[dict[str, Any]]:
        """Extract bounded lessons for all four Agents after a terminal run."""

        request = getattr(state, "request", None)
        plan = getattr(state, "plan", None)
        result = getattr(state, "result", None)
        permissions = getattr(request, "permissions", {}) if request is not None else {}
        targets = list(dict.fromkeys(
            permissions.get("coverage_current_batch_functions")
            or permissions.get("required_api_functions")
            or [getattr(log, "protocol_function", "") for log in (getattr(result, "api_logs", []) or [])]
        ))
        targets = [str(item) for item in targets if str(item)]
        errors = list(dict.fromkeys([
            *(getattr(state, "errors", []) or []),
            *(getattr(result, "errors", []) or [] if result is not None else []),
        ]))[-8:]
        logs = getattr(result, "api_logs", []) if result is not None else []
        log_facts = [
            {
                "api": str(getattr(log, "protocol_function", "")),
                "return_code": int(getattr(log, "status_code", 0)),
                "traffic_status": str(getattr(log, "traffic_status", "")),
            }
            for log in logs[-30:]
            if getattr(log, "protocol_function", "")
        ]
        common = {
            "task_id": str(getattr(request, "task_id", "")),
            "target_apis": targets,
            "target_environment": str(getattr(request, "target_environment", "")),
            "scenario_type": str(getattr(plan, "scenario_type", "")),
            "workflow_success": bool(workflow_success),
            "repair_attempts": int(getattr(state, "repair_attempts", 0)),
        }
        records: list[dict[str, Any]] = []
        for agent in AGENT_SOUL_NAMES:
            evidence = dict(common)
            evidence["learning_scope"] = {
                "planner": "plan and prerequisite applicability",
                "code_generator": "ABI, target invocation, session and logging generation",
                "executor": "actual calls, return codes, PCAP and terminal evidence",
                "router": "error classification, repair route and convergence",
            }[agent]
            if agent == "planner" and plan is not None:
                evidence["scene_plan"] = getattr(plan, "rag_context", {}).get("scene_plan", {})
                evidence["planning_errors"] = [
                    error for error in errors
                    if any(token in str(error).lower() for token in ("plan", "prerequisite", "parameter", "dependency"))
                ]
            elif agent == "code_generator":
                evidence["artifact_present"] = getattr(state, "artifacts", None) is not None
                evidence["generation_errors"] = [
                    error for error in errors
                    if any(token in str(error).lower() for token in ("compile", "abi", "code", "target api", "logged invocation", "sdk"))
                ]
                evidence["repair_history"] = list(getattr(state, "repair_history", [])[-3:])
            elif agent == "executor":
                evidence["collection_status"] = str(getattr(result, "collection_status", "not_run")) if result else "not_run"
                evidence["capture_event_count"] = len(getattr(result, "capture_events", []) or []) if result else 0
                evidence["observed_calls"] = log_facts
                evidence["execution_errors"] = [
                    error for error in errors
                    if any(token in str(error).lower() for token in ("ew_", "pcap", "capture", "runtime", "socket", "handle", "execution"))
                ]
            else:
                evidence["final_stage"] = str(getattr(state, "stage", ""))
                evidence["routing_errors"] = errors
                evidence["repair_history"] = list(getattr(state, "repair_history", [])[-3:])
            evidence["experience_id"] = "workflow-" + agent + "-" + hashlib.sha256(
                json.dumps({"targets": targets, "errors": errors, "scope": evidence["learning_scope"]}, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            records.append(self.record_evolution(agent, evidence, verified=bool(workflow_success and log_facts)))
        return records


def normalize_agent_name(agent_name: str) -> str:
    value = str(agent_name or "").strip().lower()
    aliases = {
        "routerrole": "router",
        "routeragent": "router",
        "plannerrole": "planner",
        "planningrole": "planner",
        "planningagent": "planner",
        "generatorrole": "code_generator",
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
