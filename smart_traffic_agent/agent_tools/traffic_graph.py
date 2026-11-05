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
    select_parameter_values,
    normalize_rules,
)
from ..traffic_graph import ApiNode, TrafficRelation, TrafficRelationshipService, relation_from_dict
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class ApiNameInput:
    api_name: str
    device_scope: str = ""
    allowed_api_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChainInput:
    api_name: str
    max_depth: int = 3
    device_scope: str = ""
    allowed_api_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DependencyPlanInput:
    api_name: str
    max_depth: int = 8
    device_scope: str = ""
    allowed_api_names: tuple[str, ...] = ()
    satisfied: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParameterRulesInput:
    api_name: str
    device_scope: str = ""
    allowed_api_names: tuple[str, ...] = ()


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
class SelectParameterInput:
    values: tuple[Any, ...]
    full_enum_threshold: int = 20
    middle_sample_count: int = 5


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
class RecordTrafficRelationsInput:
    """Structured relation observations produced by an Agent or Executor."""

    payload: dict[str, Any] = field(default_factory=dict)


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
        return self.service.call("get_api_info", tool_input.api_name, device_scope=tool_input.device_scope, allowed_api_names=tool_input.allowed_api_names or None)

    def input_summary(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "device_scope": tool_input.device_scope}


class GetTrafficChainTool(TrafficGraphToolBase):
    name = "get_traffic_chain"
    description = "Query API-only traffic relationship paths without treating missing data as verified."

    def invoke(self, tool_input: ChainInput) -> dict[str, Any]:
        return self.service.call("get_traffic_chain", tool_input.api_name, tool_input.max_depth, device_scope=tool_input.device_scope, allowed_api_names=tool_input.allowed_api_names or None)

    def input_summary(self, tool_input: ChainInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "max_depth": tool_input.max_depth, "device_scope": tool_input.device_scope}


class GetDependencyPlanTool(TrafficGraphToolBase):
    name = "get_dependency_plan"
    description = "Resolve confirmed filters/prepares dependencies layer by layer with bounded depth, timeout, and cycle detection."

    def invoke(self, tool_input: DependencyPlanInput) -> dict[str, Any]:
        return self.service.call(
            "get_dependency_plan", tool_input.api_name, tool_input.max_depth,
            device_scope=tool_input.device_scope,
            allowed_api_names=tool_input.allowed_api_names or None,
            satisfied=tool_input.satisfied or None,
        )

    def input_summary(self, tool_input: DependencyPlanInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "max_depth": tool_input.max_depth, "device_scope": tool_input.device_scope, "satisfied_count": len(tool_input.satisfied)}


class GetApiDependenciesTool(TrafficGraphToolBase):
    name = "get_api_dependencies"
    description = "Get documented or runtime-verified dependencies for an API."

    def invoke(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return self.service.call("get_api_dependencies", tool_input.api_name, device_scope=tool_input.device_scope, allowed_api_names=tool_input.allowed_api_names or None)

    def input_summary(self, tool_input: ApiNameInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "device_scope": tool_input.device_scope}


class GetParameterRulesTool(TrafficGraphToolBase):
    name = "get_parameter_rules"
    description = "Get filters rules and parameter applicability for an API."

    def invoke(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return self.service.call("get_parameter_rules", tool_input.api_name, device_scope=tool_input.device_scope, allowed_api_names=tool_input.allowed_api_names or None)

    def input_summary(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "device_scope": tool_input.device_scope}


class GetParameterPrerequisitesTool(TrafficGraphToolBase):
    name = "get_parameter_prerequisites"
    description = "Get controller-scoped CNC parameter bit prerequisites and manual-only read/write/readback paths for an API."

    def invoke(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return self.service.call(
            "get_parameter_prerequisites",
            tool_input.api_name,
            device_scope=tool_input.device_scope,
            allowed_api_names=tool_input.allowed_api_names or None,
        )

    def input_summary(self, tool_input: ParameterRulesInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "device_scope": tool_input.device_scope}


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


class SelectParameterValuesTool(AgentTool[SelectParameterInput, dict[str, Any]]):
    name = "select_parameter_values"
    description = "Select fixed, fully enumerated, or representative values from a filtered legal domain."

    def invoke(self, tool_input: SelectParameterInput) -> dict[str, Any]:
        return select_parameter_values(
            tool_input.values,
            full_enum_threshold=tool_input.full_enum_threshold,
            middle_sample_count=tool_input.middle_sample_count,
        )

    def input_summary(self, tool_input: SelectParameterInput) -> dict[str, Any]:
        return {"domain_size": len(tool_input.values), "full_enum_threshold": tool_input.full_enum_threshold, "middle_sample_count": tool_input.middle_sample_count}


class ValidateTrafficChainTool(TrafficGraphToolBase):
    name = "validate_traffic_chain"
    description = "Check whether a chain contains only documented or runtime-verified relations."

    def invoke(self, tool_input: ValidateChainInput) -> dict[str, Any]:
        return self.service.call("validate_traffic_chain", tool_input.api_name, tool_input.relation_ids)

    def input_summary(self, tool_input: ValidateChainInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "relation_count": len(tool_input.relation_ids)}


class ProposeTrafficRelationTool(TrafficGraphToolBase):
    name = "propose_traffic_relation"
    description = "Validate a relation hypothesis without storing an unconfirmed Neo4j edge."

    def invoke(self, tool_input: ProposeRelationInput) -> dict[str, Any]:
        try:
            relation = relation_from_dict(tool_input.relation)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "stored": False, "status": "unconfirmed", "reason": str(exc)}
        return {
            "ok": False,
            "stored": False,
            "status": "unconfirmed",
            "relation_id": relation.relation_id,
            "source_api": relation.source_api,
            "target_api": relation.target_api,
            "relation_type": str(relation.relation_type),
            "relation": {
                "relation_id": relation.relation_id,
                "source_api": relation.source_api,
                "target_api": relation.target_api,
                "relation_type": str(relation.relation_type),
                "rule_id": relation.rule_id,
                "confidence_status": relation.confidence_status,
                "evidence_ids": list(relation.evidence_ids),
                "applicability": relation.applicability,
                "attributes": relation.attributes,
                "version": relation.version,
            },
            "reason": "relation hypotheses are not persisted; provide confirmed evidence through the audited migration path",
        }


class VerifyTrafficRelationTool(TrafficGraphToolBase):
    name = "verify_traffic_relation"
    description = "Promote an existing confirmed relation only after supplied execution evidence passes validation."

    def invoke(self, tool_input: VerifyRelationInput) -> dict[str, Any]:
        return self.service.call("verify_relation", tool_input.relation_id, list(tool_input.evidence))


class RecordTrafficRelationsTool(TrafficGraphToolBase):
    name = "record_traffic_relations"
    description = (
        "Parse a structured TRCM relation-observation payload, validate its evidence, "
        "and persist only accepted documented, sdk_validated, or runtime_verified relations."
    )

    def invoke(self, tool_input: RecordTrafficRelationsInput) -> dict[str, Any]:
        payload = dict(tool_input.payload or {})
        observations = payload.get("relations", payload.get("observations", []))
        if not isinstance(observations, list):
            return {"ok": False, "accepted": [], "rejected": [{"reason": "relations must be a list"}]}
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for raw in observations:
            if not isinstance(raw, dict):
                rejected.append({"reason": "relation observation must be an object"})
                continue
            relation_payload = raw.get("relation", raw)
            if not isinstance(relation_payload, dict):
                rejected.append({"reason": "relation must be an object"})
                continue
            evidence = raw.get("evidence", payload.get("evidence", []))
            if not isinstance(evidence, list):
                evidence = []
            evidence_ids = [
                str(item.get("evidence_id", "")).strip()
                for item in evidence if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
            ]
            declared_ids = [str(item).strip() for item in relation_payload.get("evidence_ids", []) if str(item).strip()]
            if not evidence_ids and not declared_ids:
                rejected.append({"relation": relation_payload, "reason": "relation requires evidence_ids"})
                continue
            try:
                relation = relation_from_dict({
                    **relation_payload,
                    "evidence_ids": list(dict.fromkeys([*declared_ids, *evidence_ids])),
                })
            except (TypeError, ValueError) as exc:
                rejected.append({"relation": relation_payload, "reason": str(exc)})
                continue
            verification = raw.get("verification", {})
            if relation.confidence_status not in {"documented", "sdk_validated", "runtime_verified"}:
                rejected.append({"relation": relation_payload, "status": "unconfirmed", "reason": "unconfirmed relations are never persisted"})
                continue
            if relation.confidence_status == "runtime_verified":
                verified = isinstance(verification, dict) and bool(verification.get("success"))
                if not verified or not evidence:
                    rejected.append({"relation": relation_payload, "reason": "runtime_verified requires successful execution evidence"})
                    continue
            evidence_task_id = str(payload.get("task_id", "")).strip()
            node_contracts = payload.get("api_nodes", {})
            if not isinstance(node_contracts, dict):
                node_contracts = {}
            for api_name in (relation.source_api, relation.target_api):
                contract = node_contracts.get(api_name, {})
                if not isinstance(contract, dict):
                    contract = {}
                self.service.call(
                    "upsert_api",
                    ApiNode(
                        api_id=f"focas:{api_name}",
                        name=api_name,
                        protocol="FOCAS",
                        metadata={"evidence_task_id": evidence_task_id} if evidence_task_id else {},
                        parameters=dict(contract.get("parameters", {}) or {}) if isinstance(contract.get("parameters", {}), dict) else {},
                        prerequisites=dict(contract.get("prerequisites", {}) or {}) if isinstance(contract.get("prerequisites", {}), dict) else {},
                    ),
                )
            result = self.service.call("upsert_relation", relation)
            if result.get("ok"):
                accepted.append(result.get("relation", relation_payload))
            else:
                rejected.append({
                    "relation": relation_payload,
                    "stored": False,
                    "status": result.get("status", "unconfirmed"),
                    "reason": result.get("reason", result.get("error", "relation was not stored")),
                })
        return {
            "ok": not rejected,
            "accepted": accepted,
            "rejected": rejected,
            "relation_count": len(accepted),
            "backend": "configured" if self.service.available else "none",
        }

    def input_summary(self, tool_input: RecordTrafficRelationsInput) -> dict[str, Any]:
        payload = tool_input.payload or {}
        rows = payload.get("relations", payload.get("observations", []))
        return {"relation_count": len(rows) if isinstance(rows, list) else 0, "schema_version": payload.get("schema_version", "")}


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
        GetApiInfoTool(service), GetTrafficChainTool(service), GetDependencyPlanTool(service), GetApiDependenciesTool(service),
        GetParameterRulesTool(service), GetParameterPrerequisitesTool(service), DiscoverParameterValuesTool(), FilterParameterCandidatesTool(),
        GenerateParameterCombinationsTool(), SelectParameterValuesTool(), ValidateTrafficChainTool(service),
        ProposeTrafficRelationTool(service), VerifyTrafficRelationTool(service), RecordTrafficRelationsTool(service),
        GetAgentExperienceTool(memory_store), SubmitAgentExperienceTool(memory_store),
    ):
        registry.register(tool)
