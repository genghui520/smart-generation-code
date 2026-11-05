from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


WorkflowStage = Literal[
    "planning",
    "code_generation",
    "execution",
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
    permissions: dict[str, Any] = field(default_factory=dict)
    # Dynamic output-quality checks can be disabled for local simulator tests.
    quality_gate_enabled: bool = True
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
    protocol_function: str = ""
    operation_kind: Literal["focas_api", "focas_sequence", "host_helper", "ncguide_ui"] = "focas_api"
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    # Structured execution-plan fields.  The legacy fields above remain the
    # source-compatible interface used by existing templates and tests.
    api: str = ""
    role: Literal["target", "auxiliary"] = "auxiliary"
    parameter_sources: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    required_inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    failure_policy: str = "block_dependents"
    blocks_dependents: bool = True


@dataclass(slots=True)
class NcProgramSpec:
    program_name: str
    purpose: str = ""
    block_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    generation_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionPlan:
    plan_id: str
    task_id: str
    scenario_type: str
    scenario_goal: str
    target_environment: str
    nc_program_type: str
    nc_program_requirements: list[str]
    nc_program_spec: NcProgramSpec
    steps: list[PlanStep]
    expected_outputs: list[str]
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    rag_context: dict[str, Any] = field(default_factory=dict)
    llm_notes: list[str] = field(default_factory=list)
    code_spec: Any | None = None
    # Versioned structured hand-off.  These fields are populated from the
    # existing TRCM/RAG plan and are additive to the original schema.
    target_api: str = ""
    scenario: str = ""
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    runtime_bindings: list[dict[str, Any]] = field(default_factory=list)
    prerequisites: dict[str, Any] = field(default_factory=dict)
    session: dict[str, Any] = field(default_factory=dict)
    collection: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2


@dataclass(slots=True)
class GeneratedArtifacts:
    api_script: str
    nc_program: str
    api_script_path: Path | None = None
    nc_program_path: Path | None = None
    diagnostics: list[str] = field(default_factory=list)
    harness_path: Path | None = None
    fragment_path: Path | None = None
    bindings_path: Path | None = None


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
    protocol_function: str = ""
    semantic_label: str = ""
    error: str | None = None
    invocation_id: str = ""
    transaction_id: str | None = None
    start_timestamp_ns: int | None = None
    end_timestamp_ns: int | None = None
    device_context: dict[str, Any] = field(default_factory=dict)
    traffic_status: str = "unavailable"


@dataclass(slots=True)
class CaptureEvent:
    timestamp: str
    task_id: str
    interface_name: str
    direction: Literal["request", "response"]
    endpoint: str
    payload_summary: dict[str, Any]


@dataclass(slots=True)
class ToolCallRecord:
    tool_name: str
    started_at: str
    completed_at: str
    duration_ms: int
    success: bool
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    agent_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    task_id: str
    success: bool
    api_logs: list[ApiCallLog]
    capture_events: list[CaptureEvent]
    output_dir: Path
    errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    collection_status: str = "failed"
    collection_evidence: dict[str, Any] = field(default_factory=dict)
    offline_audit_status: str = "not_run"


@dataclass(slots=True)
class ExecutionContext:
    """Runtime values shared by dependent API calls and repair decisions.

    The C++ harness remains responsible for the real call and value selection;
    this Python context is the persisted evidence view used by later steps or
    a bounded repair.  It never fabricates a value when an API did not return
    one.
    """

    values_by_api: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    statuses_by_api: dict[str, list[int]] = field(default_factory=dict)
    inputs_by_api: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    invocation_ids_by_api: dict[str, list[str]] = field(default_factory=dict)
    evidence_by_step: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    selected_parameters: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, log: ApiCallLog) -> None:
        api = str(log.protocol_function or log.interface_name or "").strip()
        if not api:
            return
        self.values_by_api.setdefault(api, []).append(dict(log.response or {}))
        self.statuses_by_api.setdefault(api, []).append(int(log.status_code))
        self.inputs_by_api.setdefault(api, []).append(dict(log.input_parameters or {}))
        self.invocation_ids_by_api.setdefault(api, []).append(str(log.invocation_id or ""))
        step_id = str(log.step_id or "").strip()
        if step_id:
            self.evidence_by_step.setdefault(step_id, []).append({
                "api": api,
                "invocation_id": str(log.invocation_id or ""),
                "status_code": int(log.status_code),
                "input": dict(log.input_parameters or {}),
                "output": dict(log.response or {}),
                "transaction_id": log.transaction_id,
                "start_timestamp_ns": log.start_timestamp_ns,
                "end_timestamp_ns": log.end_timestamp_ns,
                "traffic_status": log.traffic_status,
            })

    def successful_apis(self) -> set[str]:
        return {api for api, statuses in self.statuses_by_api.items() if any(status == 0 for status in statuses)}


@dataclass(slots=True)
class QualityAssessment:
    passed: bool
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MappingRecord:
    task_id: str
    log_timestamp: str
    interface_name: str
    protocol_function: str
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
    quality_assessment: QualityAssessment | None = None
    mapping: list[MappingRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    error_memory_id: str | None = None
    prerequisite_analysis_attempted: bool = False
    long_term_memories: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = field(default_factory=list)
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
