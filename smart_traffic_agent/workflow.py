from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from .agents.code_generator import CodeGenerationAgent
from .agents.executor import ExecutionAgent
from .agents.planner import PlanningAgent
from .agents.router import RouterAgent
from .knowledge import KnowledgeBase
from .llm import LlmClient
from .memory import LongTermMemoryStore
from .models import TaskRequest, WorkflowState, utc_now
from .quality import build_quality_observation, output_variation_is_sufficient
from .utils import ensure_dir, write_json


@dataclass
class GraphState:
    workflow: WorkflowState
    output_dir: Path


class TrafficGenerationWorkflow:
    """LangGraph-backed multi-agent workflow for CNC traffic generation."""

    max_repair_attempts = 3

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_client: LlmClient | None = None,
        memory_store: LongTermMemoryStore | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm_client = llm_client or LlmClient()
        self.memory_store = memory_store or LongTermMemoryStore()
        self.progress_callback = progress_callback
        self.knowledge_base = knowledge_base
        self.router = RouterAgent(llm_client=self.llm_client, knowledge_base=knowledge_base)
        self.planner = PlanningAgent(knowledge_base, llm_client=self.llm_client)
        self.generator = CodeGenerationAgent(llm_client=self.llm_client, knowledge_base=knowledge_base)
        self.executor = ExecutionAgent(
            knowledge_base=knowledge_base,
            tool_progress_callback=self._emit,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("router", self._router_node)
        graph.add_node("planning", self._planning_node)
        graph.add_node("code_generation", self._code_generation_node)
        graph.add_node("execution", self._execution_node)
        graph.add_node("repair_plan", self._repair_node("repair_plan"))
        graph.add_node("repair_code", self._repair_node("repair_code"))
        graph.add_node("repair_execution", self._repair_node("repair_execution"))

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._route_from_router_node,
            {
                "planning": "planning",
                "code_generation": "code_generation",
                "execution": "execution",
                "repair_plan": "repair_plan",
                "repair_code": "repair_code",
                "repair_execution": "repair_execution",
                "complete": END,
            },
        )
        for node_name in [
            "planning",
            "code_generation",
            "execution",
        ]:
            graph.add_edge(node_name, "router")
        graph.add_edge("repair_plan", "planning")
        graph.add_edge("repair_code", "code_generation")
        graph.add_edge("repair_execution", "execution")
        return graph.compile()

    def run(self, request: TaskRequest, output_dir: Path) -> WorkflowState:
        output_dir = ensure_dir(output_dir)
        self._emit(
            "workflow_start",
            {
                "task_id": request.task_id,
                "target_environment": request.target_environment,
                "output_dir": str(output_dir),
            },
        )
        workflow_state = WorkflowState(request=request)
        workflow_state.long_term_memories = self.memory_store.search(request.description)
        self._emit(
            "memory_loaded",
            {
                "count": len(workflow_state.long_term_memories),
                "memory_path": str(self.memory_store.path),
            },
        )
        initial = GraphState(workflow=workflow_state, output_dir=output_dir)
        recursion_limit = max(128, 24 * (self.max_repair_attempts + 1))
        result = self.graph.invoke(initial, config={"recursion_limit": recursion_limit})
        state = graph_result_to_state(result)
        write_json(output_dir / "summary.json", workflow_summary(state))
        try:
            self.memory_store.remember_workflow(state)
        except Exception as exc:
            state.errors.append(f"MemoryStore failed after workflow completion: {exc}")
            write_json(output_dir / "summary.json", workflow_summary(state))
        self._emit(
            "workflow_complete",
            {
                "stage": state.stage,
                "success": workflow_success(state),
                "repair_attempts": state.repair_attempts,
                "summary_path": str(output_dir / "summary.json"),
            },
        )
        return state

    def _router_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state.workflow
        if state.errors:
            query = f"{state.request.description}\n" + "\n".join(state.errors[-5:])
            state.long_term_memories = self.memory_store.search(query)
            self._emit(
                "memory_reloaded_for_error",
                {"count": len(state.long_term_memories), "errors": state.errors[-5:]},
            )
        state.stage = self.router.route(state)
        self._emit(
            "router_decision",
            {
                "next_stage": state.stage,
                "has_plan": state.plan is not None,
                "has_artifacts": state.artifacts is not None,
                "has_result": state.result is not None,
                "mapping_count": len(state.mapping),
                "errors": state.errors[-5:],
                "route_source": self.router.last_route_source,
                "route_reason": self.router.last_route_reason,
            },
        )
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _planning_node(self, graph_state: GraphState) -> GraphState:
        self._emit("agent_start", {"agent": "PlanningAgent", "task": graph_state.workflow.request.description})
        try:
            state = self.planner.run(graph_state.workflow)
            state.artifacts = None
            state.result = None
            state.quality_assessment = None
            state.mapping = []
            state.errors = []
            write_json(graph_state.output_dir / "plan.json", state.plan)
            assert state.plan is not None
            self._emit(
                "agent_complete",
                {
                    "agent": "PlanningAgent",
                    "scenario_type": state.plan.scenario_type,
                    "plan_steps": len(state.plan.steps),
                    "retrieved_chunks": len(state.retrieved_chunks),
                    "plan_path": str(graph_state.output_dir / "plan.json"),
                    "nc_program_spec": {
                        "program_name": state.plan.nc_program_spec.program_name,
                        "purpose": state.plan.nc_program_spec.purpose,
                        "block_goals": state.plan.nc_program_spec.block_goals,
                        "constraints": state.plan.nc_program_spec.constraints,
                        "generation_notes": state.plan.nc_program_spec.generation_notes,
                    },
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "phase": step.phase,
                            "interface_name": step.interface_name,
                            "protocol_function": step.protocol_function,
                            "repeat": step.repeat,
                            "interval_seconds": step.interval_seconds,
                            "action": step.action,
                        }
                        for step in state.plan.steps
                    ],
                },
            )
        except Exception as exc:
            state = graph_state.workflow
            state.errors.append(f"PlanningAgent failed: {exc}")
            state.stage = "repair_plan"
            self._emit("agent_error", {"agent": "PlanningAgent", "error": str(exc)})
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _code_generation_node(self, graph_state: GraphState) -> GraphState:
        self._emit("agent_start", {"agent": "CodeGenerationAgent"})
        try:
            state = self.generator.run(graph_state.workflow, graph_state.output_dir)
            state.result = None
            state.quality_assessment = None
            state.mapping = []
            state.errors = []
            assert state.artifacts is not None
            self._emit(
                "agent_complete",
                {
                    "agent": "CodeGenerationAgent",
                    "api_script_path": str(state.artifacts.api_script_path),
                    "nc_program_path": str(state.artifacts.nc_program_path),
                    "diagnostics": state.artifacts.diagnostics,
                },
            )
        except Exception as exc:
            state = graph_state.workflow
            state.errors.append(f"CodeGenerationAgent failed: {exc}")
            state.stage = "repair_code"
            self._emit("agent_error", {"agent": "CodeGenerationAgent", "error": str(exc)})
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _execution_node(self, graph_state: GraphState) -> GraphState:
        self._emit("agent_start", {"agent": "ExecutionAgent"})
        try:
            state = self.executor.run(graph_state.workflow, graph_state.output_dir)
            assert state.result is not None
            if state.plan is not None:
                state.quality_assessment = build_quality_observation(state.plan, state.result)
                if plan_requires_program_completion(state) and not quality_metrics_show_program_completed(state):
                    if "PROGRAM_NOT_COMPLETED" not in state.result.errors:
                        state.result.errors.append("PROGRAM_NOT_COMPLETED")
                    if "PROGRAM_NOT_COMPLETED" not in state.errors:
                        state.errors.append("PROGRAM_NOT_COMPLETED")
                    state.result.success = False
                if plan_requires_uploaded_program_verification(state) and not uploaded_program_is_verified(state):
                    if "PROGRAM_NOT_VERIFIED" not in state.result.errors:
                        state.result.errors.append("PROGRAM_NOT_VERIFIED")
                    if "PROGRAM_NOT_VERIFIED" not in state.errors:
                        state.errors.append("PROGRAM_NOT_VERIFIED")
                    state.result.success = False
                if state.result.success:
                    state.errors = []
            self._emit(
                "agent_complete",
                {
                    "agent": "ExecutionAgent",
                    "success": state.result.success,
                    "api_logs": len(state.result.api_logs),
                    "capture_events": len(state.result.capture_events),
                    "tool_calls": len(state.result.tool_calls),
                    "output_dir": str(state.result.output_dir),
                    "errors": state.result.errors,
                    "quality_assessment": state.quality_assessment,
                },
            )
        except Exception as exc:
            state = graph_state.workflow
            state.errors.append(f"ExecutionAgent failed: {exc}")
            state.stage = "repair_execution"
            self._emit("agent_error", {"agent": "ExecutionAgent", "error": str(exc)})
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _repair_node(self, stage: str):
        def node(graph_state: GraphState) -> GraphState:
            state = graph_state.workflow
            failure_errors = list(state.result.errors if state.result else state.errors)
            state.repair_attempts += 1
            state.repair_history.append(
                {
                    "timestamp": utc_now(),
                    "repair_stage": stage,
                    "attempt": state.repair_attempts,
                    "errors": failure_errors,
                    "router_reason": self.router.last_route_reason,
                    "repair_instruction": self.router.last_repair_instruction,
                    "action": repair_action_for(stage),
                    "previous_state": repair_state_snapshot(state),
                }
            )
            self._emit(
                "repair_start",
                {
                    "repair_stage": stage,
                    "attempt": state.repair_attempts,
                    "errors": failure_errors,
                    "router_reason": self.router.last_route_reason,
                    "repair_instruction": self.router.last_repair_instruction,
                    "action": repair_action_for(stage),
                },
            )

            if stage == "repair_plan":
                state.stage = "planning"
            elif stage == "repair_code":
                state.stage = "code_generation"
            elif stage == "repair_execution":
                state.stage = "execution"
            else:
                state.errors.append(f"Unknown repair stage: {stage}")
                state.stage = "complete"
            self._emit("repair_complete", {"repair_stage": stage, "next_stage": state.stage})
            return GraphState(workflow=state, output_dir=graph_state.output_dir)

        return node

    def _route_from_router_node(self, graph_state: GraphState | dict[str, Any]) -> str:
        state = graph_result_to_state(graph_state)
        stage = state.stage
        if stage in {"repair_plan", "repair_code", "repair_execution"} and state.repair_attempts >= self.max_repair_attempts:
            state.errors.append(
                f"Stopped repair loop after {state.repair_attempts} attempts; last routed stage was {stage}."
            )
            self._emit(
                "repair_stopped",
                {"repair_attempts": state.repair_attempts, "last_stage": stage, "errors": state.errors[-5:]},
            )
            return "complete"
        return stage

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(event, payload)


def graph_result_to_state(result: GraphState | dict[str, Any]) -> WorkflowState:
    if isinstance(result, GraphState):
        return result.workflow
    workflow = result.get("workflow")
    if isinstance(workflow, WorkflowState):
        return workflow
    raise TypeError("LangGraph result does not contain a WorkflowState")


def workflow_summary(state: WorkflowState) -> dict:
    rag_context = state.plan.rag_context if state.plan else {}
    artifacts_summary: dict[str, Any] | None = None
    if state.artifacts is not None:
        artifacts_summary = {
            "api_script_path": str(state.artifacts.api_script_path) if state.artifacts.api_script_path else None,
            "nc_program_path": str(state.artifacts.nc_program_path) if state.artifacts.nc_program_path else None,
            "diagnostics": state.artifacts.diagnostics,
        }
    return {
        "task_id": state.request.task_id,
        "stage": state.stage,
        "success": workflow_success(state),
        "scenario_type": state.plan.scenario_type if state.plan else None,
        "plan_steps": len(state.plan.steps) if state.plan else 0,
        "retrieved_chunks": len(state.retrieved_chunks),
        "rag_context_counts": {key: len(value) for key, value in rag_context.items()},
        "api_log_count": len(state.result.api_logs) if state.result else 0,
        "capture_event_count": len(state.result.capture_events) if state.result else 0,
        "execution_tool_calls": [call.to_dict() for call in state.result.tool_calls] if state.result else [],
        "artifacts": artifacts_summary,
        "mapping_count": len(state.mapping),
        "errors": state.errors,
        "quality_assessment": state.quality_assessment,
        "repair_attempts": state.repair_attempts,
        "repair_history": state.repair_history,
        "long_term_memory_count": len(state.long_term_memories),
        "orchestrator": "langgraph",
    }


def workflow_success(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    quality_ok = state.quality_assessment is None or state.quality_assessment.passed
    if plan_requires_uploaded_program_verification(state) and not uploaded_program_is_verified(state):
        return False
    if state.result.success and quality_ok:
        if plan_requires_program_completion(state) and not quality_metrics_show_program_completed(state):
            return False
        return True
    if state.quality_assessment is not None and output_variation_is_sufficient(state.quality_assessment.metrics):
        return True
    return False


def plan_requires_program_completion(state: WorkflowState) -> bool:
    if state.plan is None:
        return False
    return any(step.interface_name == "StartProgram" for step in state.plan.steps)


def quality_metrics_show_program_completed(state: WorkflowState) -> bool:
    if state.quality_assessment is None:
        return False
    return bool(state.quality_assessment.metrics.get("program_completed"))


def plan_requires_uploaded_program_verification(state: WorkflowState) -> bool:
    if state.plan is None or state.request.target_environment != "ncguide-generated-cpp":
        return False
    interfaces = {step.interface_name for step in state.plan.steps}
    return {"UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram"}.issubset(interfaces)


def uploaded_program_is_verified(state: WorkflowState) -> bool:
    if state.plan is None or state.result is None:
        return False
    match = re.search(r"\d+", state.plan.nc_program_spec.program_name)
    if match is None:
        return False
    expected = int(match.group(0))
    observed_expected_program = False
    for log in state.result.api_logs:
        identity = " ".join([log.interface_name, log.protocol_function, log.step_id]).lower()
        data = str(log.response.get("data", ""))
        lowered_data = data.lower()
        if any(token in identity for token in ["uploadprogram", "cnc_dwn", "selectprogram", "cnc_search"]):
            if log.status_code != 0 or log.error:
                return False
        if "program_not_verified" in lowered_data:
            return False
        if "program_verified=true" in lowered_data:
            observed_expected_program = True
        if any(token in identity for token in ["readprogramnumber", "cnc_rdprgnum"]):
            if log.status_code != 0 or log.error:
                return False
            numbers = {
                int(value)
                for value in re.findall(r"(?:current_program|running_program|main_program)\s*=\s*(\d+)", lowered_data)
            }
            if expected in numbers:
                observed_expected_program = True
            elif numbers:
                return False
    return observed_expected_program


def repair_state_snapshot(state: WorkflowState) -> dict[str, Any]:
    plan_summary: dict[str, Any] | None = None
    if state.plan is not None:
        plan_summary = {
            "scenario_type": state.plan.scenario_type,
            "scenario_goal": state.plan.scenario_goal,
            "nc_program_spec": {
                "program_name": state.plan.nc_program_spec.program_name,
                "purpose": state.plan.nc_program_spec.purpose,
                "block_goals": state.plan.nc_program_spec.block_goals,
                "constraints": state.plan.nc_program_spec.constraints,
                "generation_notes": state.plan.nc_program_spec.generation_notes,
            },
            "steps": [
                {
                    "step_id": step.step_id,
                    "phase": step.phase,
                    "interface_name": step.interface_name,
                    "repeat": step.repeat,
                    "interval_seconds": step.interval_seconds,
                    "action": step.action,
                    "parameters": step.parameters,
                    "expected_state": step.expected_state,
                }
                for step in state.plan.steps[:20]
            ],
            "quality_analysis": state.plan.rag_context.get("planning_quality_analysis", {}),
            "quality_targets": state.plan.rag_context.get("quality_targets", {}),
        }

    artifacts_summary: dict[str, Any] | None = None
    if state.artifacts is not None:
        artifacts_summary = {
            "nc_program": state.artifacts.nc_program[:2000],
            "api_script_preview": state.artifacts.api_script[:3000],
            "diagnostics": state.artifacts.diagnostics[-10:],
        }

    result_summary: dict[str, Any] | None = None
    if state.result is not None:
        result_summary = {
            "success": state.result.success,
            "errors": state.result.errors[-10:],
            "api_log_count": len(state.result.api_logs),
            "capture_event_count": len(state.result.capture_events),
            "tool_calls": [call.to_dict() for call in state.result.tool_calls],
            "sample_api_logs": [
                {
                    "step_id": log.step_id,
                    "interface_name": log.interface_name,
                    "status_code": log.status_code,
                    "response": log.response,
                    "error": log.error,
                }
                for log in state.result.api_logs[:10]
            ],
        }

    quality_summary: dict[str, Any] | None = None
    if state.quality_assessment is not None:
        quality_summary = {
            "passed": state.quality_assessment.passed,
            "issues": state.quality_assessment.issues,
            "metrics": state.quality_assessment.metrics,
            "recommendations": state.quality_assessment.recommendations,
        }

    return {
        "plan": plan_summary,
        "artifacts": artifacts_summary,
        "result": result_summary,
        "quality_assessment": quality_summary,
        "active_errors": state.errors[-10:],
    }


def repair_action_for(stage: str) -> str:
    actions = {
        "repair_plan": "Preserve the previous failed state as repair context, then send the task back to PlanningAgent to modify the plan.",
        "repair_code": "Preserve the previous failed artifacts as repair context, then send the task back to CodeGenerationAgent to modify generated NC/C++.",
        "repair_execution": "Preserve the previous failed execution as repair context, then send the task back to ExecutionAgent to retry or adjust execution.",
    }
    return actions.get(stage, "No repair action is registered for this stage.")
