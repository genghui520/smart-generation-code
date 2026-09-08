"""Small Role/Action protocol inspired by MetaGPT's agent organization.

The workflow scheduler remains LangGraph. Roles own named actions and publish
typed hand-off messages so each stage has an explicit artifact boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .orchestration.roles import ActionSpec, AgentMessage, AgentTemplate, Role, message_to_dict


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    artifact_type: str
    artifact_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanSpec:
    plan_id: str
    step_count: int
    rag_chunk_count: int


@dataclass(frozen=True, slots=True)
class ApiContract:
    call_id: str
    tool_name: str
    phase: str
    operation_id: str
    arguments: dict[str, Any] = field(default_factory=dict)
    prototype: str = ""
    return_type: str = ""
    parameter_types: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)
    required_calls: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CodeSpec:
    scenario: str
    api_contracts: list[ApiContract] = field(default_factory=list)
    nc_program_name: str = ""
    source_requirements: list[str] = field(default_factory=list)
    official_abi_context: str = ""


@dataclass(frozen=True, slots=True)
class CodeArtifactSpec:
    source_path: str
    nc_program_path: str
    diagnostic_count: int


@dataclass(frozen=True, slots=True)
class ReviewReport:
    passed: bool
    diagnostic_count: int


@dataclass(frozen=True, slots=True)
class ExecutionResultSpec:
    success: bool
    api_log_count: int
    capture_event_count: int
