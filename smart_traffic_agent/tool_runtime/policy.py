from __future__ import annotations


class ToolPermissionError(PermissionError):
    """Raised when an agent calls a tool outside its explicit policy."""


# Role names are the stable public identity already used by AgentTemplate.
AGENT_TOOL_POLICY: dict[str, set[str]] = {
    "SingleAgentRole": {
        "search_knowledge",
        "search_focas_sdk",
        "write_generated_artifact",
        "render_focas_cpp",
        "compile_generated_cpp",
    },
    "RouterRole": {
        "search_knowledge", "search_memory", "get_traffic_chain", "get_dependency_plan", "get_api_dependencies", "get_parameter_prerequisites", "get_agent_experience",
    },
    "PlanningRole": {
        "search_knowledge", "search_memory", "get_api_info", "get_traffic_chain", "get_dependency_plan", "get_api_dependencies", "get_parameter_prerequisites",
        "get_parameter_rules", "discover_parameter_values", "filter_parameter_candidates",
        "generate_parameter_combinations", "propose_traffic_relation", "get_agent_experience",
    },
    "CodeEngineerRole": {
        "search_knowledge",
        "search_memory",
        "write_generated_artifact",
        "render_focas_cpp",
        "validate_focas_cpp",
        "compile_generated_cpp",
        "get_api_info", "get_dependency_plan", "get_parameter_prerequisites",
        "get_traffic_chain",
        "get_parameter_rules",
        "get_sdk_type_contract",
        "filter_parameter_candidates",
        "generate_parameter_combinations",
        "get_agent_experience",
    },
    "ExecutionRole": {
        "compile_generated_cpp",
        "run_generated_executable",
        "collect_execution_artifacts",
        "search_knowledge",
        "search_memory",
        "get_agent_experience",
        "propose_traffic_relation",
        "verify_traffic_relation",
        "record_traffic_relations",
    },
    "TrafficGenerationWorkflow": {"search_memory", "remember_workflow", "submit_agent_experience"},
}


def authorized_tools(agent_name: str) -> set[str]:
    return set(AGENT_TOOL_POLICY.get(agent_name, set()))


def check_permission(agent_name: str, tool_name: str) -> None:
    allowed = authorized_tools(agent_name)
    if tool_name not in allowed:
        raise ToolPermissionError(
            f"Agent {agent_name} is not authorized to call tool {tool_name}."
        )
