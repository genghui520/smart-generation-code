from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import WorkflowState, utc_now
from .quality import output_variation_is_sufficient
from .utils import ensure_dir, tokenize


DEFAULT_MEMORY_PATH = Path("rag_indexes/focas/repair_memory.jsonl")


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    timestamp: str
    task_description: str
    task_id: str
    target_environment: str
    scenario_type: str | None
    final_success: bool
    repair_attempts: int
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    final_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "task_description": self.task_description,
            "task_id": self.task_id,
            "target_environment": self.target_environment,
            "scenario_type": self.scenario_type,
            "final_success": self.final_success,
            "repair_attempts": self.repair_attempts,
            "repair_history": self.repair_history,
            "final_errors": self.final_errors,
            "notes": self.notes,
        }


class LongTermMemoryStore:
    """Persistent local memory for repair outcomes.

    This is intentionally lightweight: JSONL storage plus token-overlap search.
    It gives the workflow durable experience without requiring an LLM or vector DB.
    """

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = path

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        if not query_terms or not self.path.exists():
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.read_all():
            text = memory_search_text(record)
            score = len(query_terms.intersection(tokenize(text)))
            if score > 0:
                enriched = dict(record)
                enriched["score"] = score
                scored.append((score, enriched))
        scored.sort(key=lambda item: (item[0], item[1].get("timestamp", "")), reverse=True)
        return [record for _, record in scored[:top_k]]

    def remember_workflow(self, state: WorkflowState) -> None:
        if not state.repair_history:
            return
        record = MemoryRecord(
            memory_id=f"{state.request.task_id}-{utc_now()}",
            timestamp=utc_now(),
            task_description=state.request.description,
            task_id=state.request.task_id,
            target_environment=state.request.target_environment,
            scenario_type=state.plan.scenario_type if state.plan else None,
            final_success=workflow_succeeded(state),
            repair_attempts=state.repair_attempts,
            repair_history=state.repair_history,
            final_errors=state.errors,
            notes=memory_notes_from_state(state),
        )
        self.append(record)

    def append(self, record: MemoryRecord) -> None:
        ensure_dir(self.path.parent)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False))
            file.write("\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows


def memory_search_text(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("task_description", "")),
        str(record.get("target_environment", "")),
        str(record.get("scenario_type", "")),
        " ".join(str(error) for error in record.get("final_errors", [])),
        " ".join(str(note) for note in record.get("notes", [])),
    ]
    for repair in record.get("repair_history", []):
        if isinstance(repair, dict):
            parts.append(str(repair.get("repair_stage", "")))
            parts.append(str(repair.get("action", "")))
            parts.extend(str(error) for error in repair.get("errors", []))
    return "\n".join(parts)


def workflow_succeeded(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    quality_ok = state.quality_assessment is None or state.quality_assessment.passed
    if state.result.success and quality_ok:
        if state.plan is not None and any(step.interface_name == "StartProgram" for step in state.plan.steps):
            return bool(state.quality_assessment and state.quality_assessment.metrics.get("program_completed"))
        return True
    if state.quality_assessment is not None and output_variation_is_sufficient(state.quality_assessment.metrics):
        return True
    return False


def memory_notes_from_state(state: WorkflowState) -> list[str]:
    notes: list[str] = []
    for repair in state.repair_history:
        repair_stage = repair.get("repair_stage", "")
        action = repair.get("action", "")
        errors = "; ".join(str(error) for error in repair.get("errors", []))
        note = f"{repair_stage}: {action}"
        if errors:
            note = f"{note}; errors={errors}"
        notes.append(note)
    return notes
