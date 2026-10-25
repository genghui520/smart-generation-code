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
    "RouterRole": {"search_knowledge", "search_memory"},
    "PlanningRole": {"search_knowledge", "search_memory"},
    "CodeEngineerRole": {
        "search_knowledge",
        "search_memory",
        "write_generated_artifact",
        "render_focas_cpp",
        "validate_focas_cpp",
        "compile_generated_cpp",
    },
    "SingleAgentRole": {
        "search_knowledge",
        "search_focas_sdk",
        "write_generated_artifact",
        "render_focas_cpp",
        "compile_generated_cpp",
    },
    "ExecutionRole": {
        "compile_generated_cpp",
        "run_generated_executable",
        "collect_execution_artifacts",
        "search_knowledge",
        "search_memory",
    },
    "TrafficGenerationWorkflow": {"search_memory", "remember_workflow"},
}


def authorized_tools(agent_name: str) -> set[str]:
    return set(AGENT_TOOL_POLICY.get(agent_name, set()))


def check_permission(agent_name: str, tool_name: str) -> None:
    allowed = authorized_tools(agent_name)
    if tool_name not in allowed:
        raise ToolPermissionError(
            f"Agent {agent_name} is not authorized to call tool {tool_name}."
        )
