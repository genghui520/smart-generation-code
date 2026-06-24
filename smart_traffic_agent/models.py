from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


WorkflowStage = Literal[
    "planning",
    "code_generation",
    "execution",
    "annotation",
    "complete",
    "repair_plan",
    "repair_code",
    "repair_execution",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TaskRequest:
    description: str
    task_id: str
    protocol: str = "cnc"
    target_environment: str = "simulator"
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: KnowledgeChunk
    score: float


@dataclass(slots=True)
class PlanStep:
    step_id: str
    phase: Literal["before", "during", "after"]
    action: str
    interface_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    repeat: int = 1
    interval_seconds: float = 0.0
    expected_state: str = ""


@dataclass(slots=True)
class ExecutionPlan:
    plan_id: str
    task_id: str
    scenario_type: str
    scenario_goal: str
    target_environment: str
    nc_program_type: str
    nc_program_requirements: list[str]
    steps: list[PlanStep]
    expected_outputs: list[str]
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedArtifacts:
    api_script: str
    nc_program: str
    api_script_path: Path | None = None
    nc_program_path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApiCallLog:
    timestamp: str
    task_id: str
    step_id: str
    phase: str
    interface_name: str
    input_parameters: dict[str, Any]
    status_code: int
    response: dict[str, Any]
    semantic_label: str = ""
    error: str | None = None


@dataclass(slots=True)
class CaptureEvent:
    timestamp: str
    task_id: str
    interface_name: str
    direction: Literal["request", "response"]
    endpoint: str
    payload_summary: dict[str, Any]


@dataclass(slots=True)
class ExecutionResult:
    task_id: str
    success: bool
    api_logs: list[ApiCallLog]
    capture_events: list[CaptureEvent]
    output_dir: Path
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MappingRecord:
    task_id: str
    log_timestamp: str
    interface_name: str
    semantic_label: str
    related_capture_indexes: list[int]
    input_parameters: dict[str, Any]
    status_code: int


@dataclass(slots=True)
class WorkflowState:
    request: TaskRequest
    stage: WorkflowStage = "planning"
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    plan: ExecutionPlan | None = None
    artifacts: GeneratedArtifacts | None = None
    result: ExecutionResult | None = None
    mapping: list[MappingRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

