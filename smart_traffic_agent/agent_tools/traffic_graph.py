"""Structured AgentTool interface for TRG and parameter evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..memory import LongTermMemoryStore
from ..parameter_rules import (
    ParameterToolResult,
    discover_parameter_values,
    filter_parameter_candidates,
    generate_parameter_combinations,
    normalize_rules,
)
from ..traffic_graph import ApiNode, TrafficRelation, TrafficRelationshipService, relation_from_dict
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class ApiNameInput:
    api_name: str


@dataclass(frozen=True, slots=True)
class ChainInput:
    api_name: str
    max_depth: int = 3


@dataclass(frozen=True, slots=True)
class ParameterRulesInput:
    api_name: str


@dataclass(frozen=True, slots=True)
class DiscoverParameterInput:
    api_name: str
    parameter_name: str
    observations: tuple[dict[str, Any], ...] = ()
    source_api: str = ""


@dataclass(frozen=True, slots=True)
class FilterParameterInput:
    api_name: str
    parameter_name: str
    context: dict[str, Any] = field(default_factory=dict)
    rules: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class GenerateCombinationsInput:
    api_name: str
    parameter_domains: dict[str, tuple[Any, ...]]
    budget: int = 3
    constraint_descriptions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidateChainInput:
    api_name: str
    relation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProposeRelationInput:
    relation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VerifyRelationInput:
    relation_id: str
    evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AgentExperienceInput:
    agent_name: str
    task_context: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    verified: bool = False


class TrafficGraphToolBase(AgentTool[Any, dict[str, Any]]):
    def __init__(self, service: TrafficRelationshipService) -> None:
        self.service = service

    def output_summary(self, tool_output: dict[str, Any]) -> dict[str, Any]:
        return {
            "available": tool_output.get("available", tool_output.get("ok", False)),
            "backend": tool_output.get("backend", ""),
            "keys": sorted(tool_output),
        }


class GetApiInfoTool(TrafficGraphToolBase):
    name = "get_api_info"
    description = "Get one API node and its registered SDK metadata from TRG."

    def invoke(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return self.service.call("get_api_info", tool_input.api_name)

    def input_summary(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name}


class GetTrafficChainTool(TrafficGraphToolBase):
    name = "get_traffic_chain"
    description = "Query API-only traffic relationship paths without treating missing data as verified."

    def invoke(self, tool_input: ChainInput) -> dict[str, Any]:
        return self.service.call("get_traffic_chain", tool_input.api_name, tool_input.max_depth)

    def input_summary(self, tool_input: ChainInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "max_depth": tool_input.max_depth}


class GetApiDependenciesTool(TrafficGraphToolBase):
    name = "get_api_dependencies"
    description = "Get documented or runtime-verified dependencies for an API."

    def invoke(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return self.service.call("get_api_dependencies", tool_input.api_name)

    def input_summary(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name}


class GetParameterRulesTool(TrafficGraphToolBase):
    name = "get_parameter_rules"
    description = "Get PROVIDES_INPUT rules and parameter applicability for an API."

    def invoke(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return self.service.call("get_parameter_rules", tool_input.api_name)

    def input_summary(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name}


class DiscoverParameterValuesTool(AgentTool[DiscoverParameterInput, dict[str, Any]]):
    name = "discover_parameter_values"
    description = "Extract parameter candidates from actual API observations and evidence IDs."

    def invoke(self, tool_input: DiscoverParameterInput) -> dict[str, Any]:
        return discover_parameter_values(
            tool_input.api_name,
            tool_input.parameter_name,
            tool_input.observations,
            source_api=tool_input.source_api,
        ).to_dict()

    def input_summary(self, tool_input: DiscoverParameterInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "parameter_name": tool_input.parameter_name, "observation_count": len(tool_input.observations)}


class FilterParameterCandidatesTool(AgentTool[FilterParameterInput, dict[str, Any]]):
    name = "filter_parameter_candidates"
    description = "Apply evidence-backed parameter rules, protected-value exclusions, and operation context."

    def invoke(self, tool_input: FilterParameterInput) -> dict[str, Any]:
        rules = normalize_rules(tool_input.rules, tool_input.api_name)
        return filter_parameter_candidates(
            tool_input.api_name, tool_input.parameter_name, tool_input.context, rules
        ).to_dict()

    def input_summary(self, tool_input: FilterParameterInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "parameter_name": tool_input.parameter_name, "context_keys": sorted(tool_input.context)}


class GenerateParameterCombinationsTool(AgentTool[GenerateCombinationsInput, dict[str, Any]]):
    name = "generate_parameter_combinations"
    description = "Generate bounded unique combinations from already discovered legal domains."

    def invoke(self, tool_input: GenerateCombinationsInput) -> dict[str, Any]:
        return generate_parameter_combinations(
            tool_input.api_name,
            tool_input.parameter_domains,
            tool_input.budget,
            constraint_descriptions=tool_input.constraint_descriptions,
        ).to_dict()

    def input_summary(self, tool_input: GenerateCombinationsInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "parameters": sorted(tool_input.parameter_domains), "budget": tool_input.budget}


class ValidateTrafficChainTool(TrafficGraphToolBase):
    name = "validate_traffic_chain"
    description = "Check whether a chain contains only documented or runtime-verified relations."

    def invoke(self, tool_input: ValidateChainInput) -> dict[str, Any]:
        return self.service.call("validate_traffic_chain", tool_input.api_name, tool_input.relation_ids)

    def input_summary(self, tool_input: ValidateChainInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "relation_count": len(tool_input.relation_ids)}


class ProposeTrafficRelationTool(TrafficGraphToolBase):
    name = "propose_traffic_relation"
    description = "Store a relation hypothesis as proposed with evidence and an applicability scope."

    def invoke(self, tool_input: ProposeRelationInput) -> dict[str, Any]:
        relation = relation_from_dict(tool_input.relation)
        if relation.confidence_status != "proposed":
            relation = TrafficRelation(
                relation_id=relation.relation_id, source_api=relation.source_api, target_api=relation.target_api,
                relation_type=relation.relation_type, rule_id=relation.rule_id, confidence_status="proposed",
                evidence_ids=relation.evidence_ids, applicability=relation.applicability,
                attributes=relation.attributes, version=relation.version,
            )
        return self.service.call("upsert_relation", relation)


class VerifyTrafficRelationTool(TrafficGraphToolBase):
    name = "verify_traffic_relation"
    description = "Promote a proposed relation only after supplied execution evidence passes validation."

    def invoke(self, tool_input: VerifyRelationInput) -> dict[str, Any]:
        return self.service.call("verify_relation", tool_input.relation_id, list(tool_input.evidence))


class GetAgentExperienceTool(AgentTool[AgentExperienceInput, dict[str, Any]]):
    name = "get_agent_experience"
    description = "Retrieve one Agent's scoped long-term experience for the current task."

    def __init__(self, memory_store: LongTermMemoryStore) -> None:
        self.memory_store = memory_store

    def invoke(self, tool_input: AgentExperienceInput) -> dict[str, Any]:
        rows = self.memory_store.search_agent_experiences(tool_input.agent_name, tool_input.task_context, top_k=5)
        return {"success": True, "agent_name": tool_input.agent_name, "experiences": rows}

    def input_summary(self, tool_input: AgentExperienceInput) -> dict[str, Any]:
        return {"agent_name": tool_input.agent_name, "task_context": tool_input.task_context[:300]}


class SubmitAgentExperienceTool(AgentTool[AgentExperienceInput, dict[str, Any]]):
    name = "submit_agent_experience"
    description = "Persist a proposed or verified Agent experience with version and evidence."

    def __init__(self, memory_store: LongTermMemoryStore) -> None:
        self.memory_store = memory_store

    def invoke(self, tool_input: AgentExperienceInput) -> dict[str, Any]:
        return self.memory_store.submit_agent_experience(
            tool_input.agent_name, tool_input.evidence, verified=tool_input.verified
        )


def register_traffic_tools(registry: Any, service: TrafficRelationshipService, memory_store: LongTermMemoryStore) -> None:
    for tool in (
        GetApiInfoTool(service), GetTrafficChainTool(service), GetApiDependenciesTool(service),
        GetParameterRulesTool(service), DiscoverParameterValuesTool(), FilterParameterCandidatesTool(),
        GenerateParameterCombinationsTool(), ValidateTrafficChainTool(service),
        ProposeTrafficRelationTool(service), VerifyTrafficRelationTool(service),
        GetAgentExperienceTool(memory_store), SubmitAgentExperienceTool(memory_store),
    ):
        registry.register(tool)
