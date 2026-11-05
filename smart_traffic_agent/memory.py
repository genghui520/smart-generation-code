from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from .knowledge import _BGEEmbedding
from .models import WorkflowState, utc_now
from .quality import output_variation_is_sufficient
from .utils import ensure_dir, tokenize


DEFAULT_MEMORY_PATH = Path("rag_indexes/focas/repair_memory.jsonl")
DEFAULT_MEMORY_VECTOR_DIR = Path("rag_indexes/focas/repair_memory_vector_db")
DEFAULT_EXPERIENCE_PATH = Path("rag_indexes/focas/error_experiences.jsonl")
MEMORY_COLLECTION = "focas_repair_memory"


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

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH, vector_db_dir: Path | None = None) -> None:
        self.path = path
        self.experience_path = (
            DEFAULT_EXPERIENCE_PATH
            if path == DEFAULT_MEMORY_PATH
            else path.parent / "error_experiences.jsonl"
        )
        self.vector_db_dir = vector_db_dir or (
            DEFAULT_MEMORY_VECTOR_DIR
            if path == DEFAULT_MEMORY_PATH
            else path.parent / f"{path.stem}_vector_db"
        )
        self._chroma_client = None
        self._vector_collection = None
        self._embed_fn = None
        self._vector_sync_attempted = False

    @property
    def vector_search_available(self) -> bool:
        return self._vector_collection is not None and self._embed_fn is not None

    def close(self) -> None:
        """Release local Chroma and embedding resources."""
        self._vector_collection = None
        self._embed_fn = None
        self._chroma_client = None

    def _ensure_vector_store(self) -> bool:
        if self._vector_sync_attempted:
            return self.vector_search_available
        self._vector_sync_attempted = True
        try:
            self._embed_fn = _BGEEmbedding()
            client = chromadb.PersistentClient(path=str(self.vector_db_dir))
            self._chroma_client = client
            self._vector_collection = client.get_or_create_collection(MEMORY_COLLECTION)
            self._sync_vector_store()
        except Exception as exc:
            self._vector_collection = None
            self._embed_fn = None
            print(f"[MemoryStore] embedding index unavailable, using term search: {exc}")
        return self.vector_search_available

    def _sync_vector_store(self) -> None:
        if not self.vector_search_available:
            return
        records = self.read_all() + self.read_experiences()
        if not records:
            return
        existing = set(self._vector_collection.get(include=[]).get("ids", []))
        pending = [record for record in records if record.get("memory_id") not in existing]
        if not pending:
            return
        documents = [memory_search_text(record) for record in pending]
        embeddings = self._embed_fn(documents)
        self._vector_collection.add(
            ids=[str(record["memory_id"]) for record in pending],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {
                    "memory_id": str(record.get("memory_id", "")),
                    "scenario_type": str(record.get("scenario_type", "")),
                    "final_success": bool(record.get("final_success", False)),
                }
                for record in pending
            ],
        )

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        if self._ensure_vector_store():
            try:
                query_embedding = self._embed_fn([query])
                result = self._vector_collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(top_k, 50),
                    include=["distances", "metadatas"],
                )
                records_by_id = {
                    str(item.get("memory_id")): item
                    for item in self.read_all() + self.read_experiences()
                }
                matches = []
                for memory_id, distance, metadata in zip(
                    result.get("ids", [[]])[0],
                    result.get("distances", [[]])[0],
                    result.get("metadatas", [[]])[0],
                ):
                    record = dict(records_by_id.get(str(memory_id), {}))
                    if not record:
                        continue
                    record["score"] = round(max(0.0, 1.0 - float(distance) / 2.0), 4)
                    record["retrieval_method"] = "embedding"
                    matches.append(record)
                return matches
            except Exception as exc:
                print(f"[MemoryStore] embedding search failed, using term search: {exc}")
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scored: list[tuple[int, dict[str, Any]]] = []
        for record in self.read_all() + self.read_experiences():
            text = memory_search_text(record)
            score = len(query_terms.intersection(tokenize(text)))
            if score > 0:
                enriched = dict(record)
                enriched["score"] = score
                enriched["retrieval_method"] = "term_fallback"
                scored.append((score, enriched))
        scored.sort(key=lambda item: (item[0], item[1].get("timestamp", "")), reverse=True)
        return [record for _, record in scored[:top_k]]

    def search_agent_experiences(
        self, agent_name: str, query: str, *, top_k: int = 5, include_shared: bool = False
    ) -> list[dict[str, Any]]:
        """Return experiences scoped to one Agent and task context.

        Unscoped legacy records are excluded by default so one Agent does not
        inherit another Agent's learned behavior. Callers that explicitly need
        shared historical facts may opt in with ``include_shared=True``.
        """
        agent = str(agent_name or "").strip().lower()
        rows = []
        for record in self.read_experiences():
            owner = str(record.get("agent_name", "")).strip().lower()
            if owner != agent and not (include_shared and owner == ""):
                continue
            rows.append(record)
        query_terms = set(tokenize(query))
        scored: list[tuple[int, dict[str, Any]]] = []
        for record in rows:
            score = len(query_terms.intersection(tokenize(memory_search_text(record))))
            if score or not query_terms:
                enriched = dict(record)
                enriched["score"] = score
                enriched["retrieval_method"] = "term_fallback"
                scored.append((score, enriched))
        scored.sort(key=lambda item: (item[0], item[1].get("timestamp", "")), reverse=True)
        return [record for _, record in scored[: max(1, min(top_k, 50))]]

    def submit_agent_experience(
        self,
        agent_name: str,
        evidence: dict[str, Any],
        *,
        verified: bool = False,
    ) -> dict[str, Any]:
        """Persist a versioned experience without silently making it trusted."""
        import hashlib

        canonical = json.dumps(
            {"agent_name": agent_name, "evidence": evidence},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        experience_id = str(evidence.get("experience_id") or "exp-" + hashlib.sha256(canonical.encode()).hexdigest()[:16])
        existing = next(
            (row for row in self.read_experiences() if str(row.get("experience_id")) == experience_id),
            None,
        )
        version = int(existing.get("version", 0)) + 1 if existing else 1
        status = "runtime_verified" if verified else "proposed"
        # A later hypothesis must never downgrade an experience that already
        # has runtime evidence. Promotion is monotonic until an explicit
        # superseding record is created under a new experience id.
        if existing and existing.get("status") == "runtime_verified":
            status = "runtime_verified"
        record = {
            "experience_id": experience_id,
            "memory_id": f"experience:{experience_id}",
            "record_type": "agent_experience",
            "agent_name": str(agent_name),
            "status": status,
            "version": version,
            "timestamp": utc_now(),
            "evidence": evidence,
        }
        self.upsert_experience(record)
        return record

    def upsert_experience(self, record: dict[str, Any]) -> None:
        ensure_dir(self.experience_path.parent)
        rows = self.read_experiences()
        experience_id = str(record.get("experience_id", ""))
        updated = False
        for index, row in enumerate(rows):
            if str(row.get("experience_id", "")) == experience_id:
                rows[index] = dict(record)
                updated = True
                break
        if not updated:
            rows.append(dict(record))
        self.experience_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        if self._ensure_vector_store():
            self._sync_vector_store()

    def read_experiences(self) -> list[dict[str, Any]]:
        """Read curated, evidence-backed troubleshooting experiences."""
        if not self.experience_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.experience_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("experience_id"):
                    payload.setdefault("memory_id", f"experience:{payload['experience_id']}")
                    payload.setdefault("record_type", "curated_error_experience")
                    rows.append(payload)
        return rows

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

    def remember_error(self, state: WorkflowState, *, status: str = "pending") -> str | None:
        """Persist an error episode as soon as the Router observes it."""
        errors = list(dict.fromkeys([*state.errors, *(state.result.errors if state.result else [])]))
        if not errors:
            return state.error_memory_id
        if state.error_memory_id is None:
            state.error_memory_id = f"error-{state.request.task_id}-{uuid.uuid4().hex[:10]}"
        record = {
            "memory_id": state.error_memory_id,
            "record_type": "workflow_error_episode",
            "error_category": "SDK_TYPE_MISMATCH" if any("SDK_TYPE_MISMATCH" in str(error) for error in errors) else "",
            "timestamp": utc_now(),
            "status": status,
            "task_description": state.request.description,
            "task_id": state.request.task_id,
            "target_environment": state.request.target_environment,
            "scenario_type": state.plan.scenario_type if state.plan else None,
            "stage": state.stage,
            "errors": errors,
            "repair_attempts": state.repair_attempts,
            "repair_history": list(state.repair_history),
            "evidence": error_episode_evidence(state),
            "final_success": False,
        }
        self.upsert(record)
        return state.error_memory_id

    def upsert(self, record: dict[str, Any]) -> None:
        """Append new memories and replace an existing episode on later updates."""
        ensure_dir(self.path.parent)
        records = self.read_all()
        memory_id = str(record["memory_id"])
        if not any(str(row.get("memory_id")) == memory_id for row in records):
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            updated = [record if str(row.get("memory_id")) == memory_id else row for row in records]
            self.path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated), encoding="utf-8")
        if self._ensure_vector_store():
            document = memory_search_text(record)
            embedding = self._embed_fn([document])[0]
            metadata = {
                "memory_id": memory_id,
                "scenario_type": str(record.get("scenario_type", "")),
                "final_success": bool(record.get("final_success", False)),
            }
            existing_ids = set(self._vector_collection.get(include=[]).get("ids", []))
            operation = self._vector_collection.update if memory_id in existing_ids else self._vector_collection.add
            operation(ids=[memory_id], documents=[document], embeddings=[embedding], metadatas=[metadata])

    def append(self, record: MemoryRecord) -> None:
        ensure_dir(self.path.parent)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False))
            file.write("\n")
        if self._ensure_vector_store():
            self._sync_vector_store()

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
        str(record.get("error_category", "")),
        str(record.get("title", "")),
        str(record.get("trigger", "")),
        str(record.get("root_cause", "")),
        str(record.get("fix", "")),
        str(record.get("prevention", "")),
        " ".join(str(item) for item in record.get("related_apis", [])),
        " ".join(str(item) for item in record.get("evidence", [])),
        str(record.get("task_description", "")),
        str(record.get("target_environment", "")),
        str(record.get("scenario_type", "")),
        str(record.get("status", "")),
        str(record.get("stage", "")),
        " ".join(str(error) for error in record.get("errors", [])),
        str(record.get("router_reason", "")),
        " ".join(str(error) for error in record.get("final_errors", [])),
        " ".join(str(note) for note in record.get("notes", [])),
    ]
    for repair in record.get("repair_history", []):
        if isinstance(repair, dict):
            parts.append(str(repair.get("repair_stage", "")))
            parts.append(str(repair.get("action", "")))
            parts.extend(str(error) for error in repair.get("errors", []))
            parts.append(str(repair.get("repair_instruction", "")))
            parts.append(str(repair.get("router_reason", "")))
    return "\n".join(parts)


def error_episode_evidence(state: WorkflowState) -> list[dict[str, Any]]:
    if state.result is None:
        return []
    return [
        {
            "interface_name": log.interface_name,
            "status_code": log.status_code,
            "error": log.error,
            "traffic_status": log.traffic_status,
            "input_parameters": log.input_parameters,
        }
        for log in state.result.api_logs[-20:]
    ]


def workflow_succeeded(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    quality_ok = state.quality_assessment is None or state.quality_assessment.passed
    if state.result.success and quality_ok:
        if state.request.quality_gate_enabled and state.plan is not None and any(step.interface_name == "StartProgram" for step in state.plan.steps):
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
