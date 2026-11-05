from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END

from .agents.code_generator import CodeGenerationAgent
from .agents.executor import ExecutionAgent
from .agents.planner import PlanningAgent
from .agents.router import RouterAgent
from .orchestration.graph import AgentGraph
from .function_coverage import (
    DEFAULT_FUNCTION_COVERAGE_STATE,
    build_function_coverage_metrics,
    merge_workflow_function_coverage_state,
)
from .knowledge import KnowledgeBase
from .environment import AgentEnvironment
from .llm import LlmClient, TransientLlmError
from .memory import LongTermMemoryStore
from .agent_tools import (
    CollectExecutionArtifactsTool,
    CompileGeneratedCppTool,
    KnowledgeSearchTool,
    MemorySearchInput,
    MemorySearchTool,
    RememberWorkflowTool,
    RunGeneratedExecutableTool,
    WriteGeneratedArtifactTool,
    RenderFocasCppTool,
    ValidateFocasCppTool,
    GetSdkTypeContractTool,
    register_traffic_tools,
)
from .tool_runtime import ToolRuntime, ToolRegistry
from .models import TaskRequest, WorkflowState, utc_now
from .quality import build_quality_observation, output_variation_is_sufficient
from .rag.focas_function_manifest import DEFAULT_COVERAGE_BATCH_SIZE, DEFAULT_COVERAGE_SEGMENT_COUNT
from .utils import ensure_dir, write_json
from .agent_soul import AgentSoulStore
from .traffic_graph import TrafficRelationshipService, build_default_relationship_service
from .display_text import write_human_summary


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
        coverage_state_path: Path = DEFAULT_FUNCTION_COVERAGE_STATE,
        coverage_batch_size: int = DEFAULT_COVERAGE_BATCH_SIZE,
        coverage_segment_count: int = DEFAULT_COVERAGE_SEGMENT_COUNT,
        traffic_relationships: TrafficRelationshipService | None = None,
        soul_store: AgentSoulStore | None = None,
        max_repair_attempts: int = 3,
    ) -> None:
        self.max_repair_attempts = max(0, int(max_repair_attempts))
        self.llm_client = llm_client or LlmClient()
        self.memory_store = memory_store or LongTermMemoryStore()
        self.progress_callback = progress_callback
        self.coverage_state_path = coverage_state_path
        self.knowledge_base = knowledge_base
        self.traffic_relationships = traffic_relationships or build_default_relationship_service()
        self.soul_store = soul_store or AgentSoulStore(Path(os.getenv("SMPAGENT_SOUL_ROOT", "agent_souls")))
        self.environment = AgentEnvironment(knowledge_base)
        self.tool_registry = ToolRegistry()
        compile_tool = CompileGeneratedCppTool()
        run_tool = RunGeneratedExecutableTool()
        collect_tool = CollectExecutionArtifactsTool()
        for tool in (
            compile_tool,
            run_tool,
            collect_tool,
            KnowledgeSearchTool(knowledge_base),
            MemorySearchTool(self.memory_store),
            RememberWorkflowTool(self.memory_store),
            WriteGeneratedArtifactTool(),
            RenderFocasCppTool(),
            ValidateFocasCppTool(),
            GetSdkTypeContractTool(),
        ):
            self.tool_registry.register(tool)
        register_traffic_tools(self.tool_registry, self.traffic_relationships, self.memory_store)
        self.tool_runtime = ToolRuntime(self.tool_registry, progress_callback=self._emit)
        self.router = RouterAgent(llm_client=self.llm_client, knowledge_base=knowledge_base)
        self.planner = PlanningAgent(
            knowledge_base,
            llm_client=self.llm_client,
            coverage_state_path=coverage_state_path,
            coverage_batch_size=coverage_batch_size,
            coverage_segment_count=coverage_segment_count,
        )
        self.generator = CodeGenerationAgent(llm_client=self.llm_client, knowledge_base=knowledge_base)
        self.executor = ExecutionAgent(
            knowledge_base=knowledge_base,
            compile_tool=compile_tool,
            run_tool=run_tool,
            collect_tool=collect_tool,
            tool_runtime=self.tool_runtime,
            tool_progress_callback=self._emit,
            llm_client=self.llm_client,
        )
        for agent in (self.router, self.planner, self.generator, self.executor):
            agent.environment = self.environment
            agent.tool_runtime = self.tool_runtime
            agent.traffic_relationships = self.traffic_relationships
            agent.soul_store = self.soul_store
            agent.memory_store = self.memory_store
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = AgentGraph(GraphState)
        graph.add_agent_node("router", self.router, self._router_node)
        graph.add_agent_node("planning", self.planner, self._planning_node)
        graph.add_agent_node("code_generation", self.generator, self._code_generation_node)
        graph.add_agent_node("execution", self.executor, self._execution_node)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            self._route_from_router_node,
            {
                "planning": "planning",
                "code_generation": "code_generation",
                "execution": "execution",
                "repair_plan": "planning",
                "repair_code": "code_generation",
                "repair_execution": "execution",
                "complete": END,
            },
        )
        for node_name in [
            "planning",
            "code_generation",
            "execution",
        ]:
            graph.add_edge(node_name, "router")
        self.agent_graph = graph
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
        self.tool_runtime.history = workflow_state.tool_call_history
        historical_memory_enabled = bool(request.permissions.get("historical_memory_enabled", True))
        if historical_memory_enabled:
            memories, _ = self.tool_runtime.call(
                "TrafficGenerationWorkflow", "search_memory", MemorySearchInput(request.description)
            )
            workflow_state.long_term_memories = memories
        else:
            workflow_state.long_term_memories = []
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
        # LangGraph may materialize a new WorkflowState instance while the
        # shared ToolRuntime keeps the authoritative process-level audit list.
        # Reattach that history before summaries and persistence are produced.
        state.tool_call_history = list(self.tool_runtime.history)
        coverage_state_update = merge_workflow_function_coverage_state(state, self.coverage_state_path)
        summary = workflow_summary(state)
        summary["agent_graph"] = self.agent_graph.describe()
        if coverage_state_update:
            summary["coverage_state_update"] = coverage_state_update
        soul_updates: list[dict[str, Any]] = []
        try:
            soul_updates = self.soul_store.learn_from_workflow(
                state,
                workflow_success=workflow_success(state),
            )
            summary["agent_soul_updates"] = [
                {
                    "agent_name": item.get("agent_name"),
                    "experience_id": item.get("experience_id"),
                    "status": item.get("status"),
                    "version": item.get("version"),
                }
                for item in soul_updates
            ]
        except Exception as exc:
            # Learning must never change the execution result.  Preserve the
            # failure in the summary so the self-improvement path is auditable.
            summary["agent_soul_update_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output_dir / "summary.json", summary)
        write_human_summary(output_dir / "summary_zh.txt", summary)
        try:
            if historical_memory_enabled:
                if state.error_memory_id:
                    record = next(
                        (item for item in self.memory_store.read_all() if item.get("memory_id") == state.error_memory_id),
                        {},
                    )
                    record.update({
                        "status": "resolved" if workflow_success(state) else "unresolved",
                        "final_success": workflow_success(state),
                        "repair_history": list(state.repair_history),
                        "repair_attempts": state.repair_attempts,
                        "final_errors": list(state.errors or (state.result.errors if state.result else [])),
                        "timestamp_updated": utc_now(),
                    })
                    self.memory_store.upsert(record)
                self.tool_runtime.call("TrafficGenerationWorkflow", "remember_workflow", state)
        except Exception as exc:
            state.errors.append(f"MemoryStore failed after workflow completion: {exc}")
            summary = workflow_summary(state)
            summary["agent_graph"] = self.agent_graph.describe()
            if coverage_state_update:
                summary["coverage_state_update"] = coverage_state_update
            write_json(output_dir / "summary.json", summary)
            write_human_summary(output_dir / "summary_zh.txt", summary)
        finally:
            self.memory_store.close()
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
        historical_memory_enabled = bool(state.request.permissions.get("historical_memory_enabled", True))
        observed_errors = list(dict.fromkeys([*state.errors, *(state.result.errors if state.result else [])]))
        if observed_errors and historical_memory_enabled:
            query = f"{state.request.description}\n" + "\n".join(observed_errors[-5:])
            memories, _ = self.tool_runtime.call(
                "TrafficGenerationWorkflow", "search_memory", MemorySearchInput(query)
            )
            state.long_term_memories = [
                memory for memory in memories
                if memory.get("memory_id") != state.error_memory_id
            ]
            self.memory_store.remember_error(state)
            self._emit(
                "memory_reloaded_for_error",
                {"count": len(state.long_term_memories), "errors": observed_errors[-5:]},
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
                "repair_instruction": self.router.last_repair_instruction,
            },
        )
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _planning_node(self, graph_state: GraphState) -> GraphState:
        self._emit("agent_start", {"agent": "PlanningAgent", "task": graph_state.workflow.request.description})
        try:
            state = graph_state.workflow
            self._prepare_repair(state, "repair_plan")
            state = self.planner.run(state)
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
            if isinstance(exc, TransientLlmError):
                state.errors.append(f"transient_llm_error: {exc}")
                state.stage = "complete"
                self._emit(
                    "agent_error",
                    {
                        "agent": "PlanningAgent",
                        "error": str(exc),
                        "failure_category": "transient_llm_error",
                        "terminal": True,
                    },
                )
            else:
                state.errors.append(f"PlanningAgent failed: {exc}")
                state.stage = "repair_plan"
                self._emit("agent_error", {"agent": "PlanningAgent", "error": str(exc)})
        return GraphState(workflow=state, output_dir=graph_state.output_dir)

    def _code_generation_node(self, graph_state: GraphState) -> GraphState:
        self._emit("agent_start", {"agent": "CodeGenerationAgent"})
        try:
            state = graph_state.workflow
            self._prepare_repair(state, "repair_code")
            state = self.generator.run(state, graph_state.output_dir)
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
            state = graph_state.workflow
            self._prepare_repair(state, "repair_execution")
            state = self.executor.run(state, graph_state.output_dir)
            assert state.result is not None
            if state.plan is not None and state.request.quality_gate_enabled:
                # The executor reports transport/process success only.  Keep a
                # separate objective quality observation so RouterAgent can
                # decide whether the returned traffic is actually useful.
                state.quality_assessment = build_quality_observation(state.plan, state.result)
                if plan_requires_uploaded_program_verification(state) and not uploaded_program_is_verified(state):
                    if "PROGRAM_NOT_VERIFIED" not in state.result.errors:
                        state.result.errors.append("PROGRAM_NOT_VERIFIED")
                    if "PROGRAM_NOT_VERIFIED" not in state.errors:
                        state.errors.append("PROGRAM_NOT_VERIFIED")
                    state.result.success = False
                if state.result.success:
                    state.errors = []
            elif state.result.success:
                # Local simulator mode validates transport/API execution only;
                # real returned-output quality is deferred to NCGuide runs.
                state.quality_assessment = None
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

    def _prepare_repair(self, state: WorkflowState, stage: str) -> None:
        """Record a repair intent when control enters its target Agent node."""
        if state.stage != stage:
            return
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
        state.stage = {
            "repair_plan": "planning",
            "repair_code": "code_generation",
            "repair_execution": "execution",
        }[stage]
        self._emit("repair_complete", {"repair_stage": stage, "next_stage": state.stage})

    def _route_from_router_node(self, graph_state: GraphState | dict[str, Any]) -> str:
        state = graph_result_to_state(graph_state)
        stage = state.stage
        if stage in {"repair_plan", "repair_code", "repair_execution"} and state.repair_attempts >= self.max_repair_attempts:
            state.errors.append(
                f"Stopped repair loop after {state.repair_attempts} attempts; last routed stage was {stage}."
            )
            state.stage = "complete"
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
    collection_evidence = state.result.collection_evidence if state.result else {}
    status_fields = collection_evidence.get("status_fields", {}) if isinstance(collection_evidence, dict) else {}
    def context_size(value: Any) -> int:
        """Return a stable summary size for list-like and scalar context values."""
        if value is None:
            return 0
        if isinstance(value, (str, bytes, list, tuple, set, dict)):
            return len(value)
        # RAG context may contain scalar flags such as ``enabled``.  A scalar
        # is present, but it is not a collection whose length can be measured.
        return 1

    return {
        "task_id": state.request.task_id,
        "stage": state.stage,
        "success": workflow_success(state),
        "scenario_type": state.plan.scenario_type if state.plan else None,
        "plan_steps": len(state.plan.steps) if state.plan else 0,
        "retrieved_chunks": len(state.retrieved_chunks),
        "rag_context_counts": {key: context_size(value) for key, value in rag_context.items()},
        "api_log_count": len(state.result.api_logs) if state.result else 0,
        "capture_event_count": len(state.result.capture_events) if state.result else 0,
        "collection_status": state.result.collection_status if state.result else "not_run",
        "collection_evidence": collection_evidence,
        "workflow_status": status_fields.get("workflow_status", "not_run"),
        "api_execution_status": status_fields.get("api_execution_status", "not_run"),
        "capture_status": status_fields.get("capture_status", "not_run"),
        "annotation_status": status_fields.get("annotation_status", "not_run"),
        "capture_reason": status_fields.get("capture_reason", ""),
        "offline_audit_status": state.result.offline_audit_status if state.result else "not_run",
        "execution_tool_calls": [call.to_dict() for call in state.result.tool_calls] if state.result else [],
        "tool_call_history": [call.to_dict() for call in state.tool_call_history],
        "artifacts": artifacts_summary,
        "mapping_count": len(state.mapping),
        "errors": state.errors,
        "quality_assessment": state.quality_assessment,
        "scene_plan": rag_context.get("scene_plan", {}),
        "scene_execution_evidence": rag_context.get("scene_execution_evidence", {}),
        "plan_execution_evidence": rag_context.get("plan_execution_evidence", {}),
        "function_coverage": build_function_coverage_metrics(state.plan, state.result),
        "repair_attempts": state.repair_attempts,
        "repair_history": state.repair_history,
        "long_term_memory_count": len(state.long_term_memories),
        "orchestrator": "langgraph",
    }


def workflow_success(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    if plan_requires_uploaded_program_verification(state) and not uploaded_program_is_verified(state):
        return False
    if not state.result.success:
        return False
    # Output variation and transaction quality are persisted for offline audit.
    # They do not decide whether the compile/run/capture workflow executed.
    return True


def requires_dynamic_quality_gate(state: WorkflowState) -> bool:
    if not state.request.quality_gate_enabled:
        return False
    if state.plan is None:
        return False
    from .quality import plan_requires_motion_quality
    return plan_requires_motion_quality(state.plan)


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
    contract = state.plan.rag_context.get("execution_contract", {})
    if contract.get("motion"):
        return True
    interfaces = {step.interface_name for step in state.plan.steps}
    return {"UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram"}.issubset(interfaces)


def uploaded_program_is_verified(state: WorkflowState) -> bool:
    if state.plan is None or state.result is None:
        return False
    # The O number in the plan is only a preferred starting point. Runtime
    # collision handling may safely select another unused number, so verify
    # the program that was actually uploaded/selected instead of comparing it
    # with the planner's original value.
    observed_runtime_verification = False
    verified_runtime_numbers: set[int] = set()
    read_program_numbers: set[int] = set()
    for log in state.result.api_logs:
        identity = " ".join([log.interface_name, log.protocol_function, log.step_id]).lower()
        data = str(log.response.get("data", ""))
        lowered_data = data.lower()
        if any(token in identity for token in ["uploadprogram", "cnc_dwn", "selectprogram", "cnc_search"]):
            # Buffer backpressure is recoverable when the fixed transfer loop
            # subsequently closes and verifies the same upload successfully.
            buffer_retry = log.protocol_function == "cnc_download3" and log.status_code == 10
            if not buffer_retry and (log.status_code != 0 or log.error):
                return False
        if "program_not_verified" in lowered_data:
            return False
        if "program_verified=true" in lowered_data:
            observed_runtime_verification = True
            verified_runtime_numbers.update(
                int(value)
                for value in re.findall(
                    r"(?:expected_program_number|uploaded_program_number|selected_program_number)\s*=\s*(\d+)",
                    lowered_data,
                )
                if int(value) > 0
            )
        if any(token in identity for token in ["readprogramnumber", "cnc_rdprgnum"]):
            if log.status_code != 0 or log.error:
                return False
            read_program_numbers.update(
                int(value)
                for value in re.findall(r"(?:current_program|running_program|main_program)\s*=\s*(\d+)", lowered_data)
                if int(value) > 0
            )
    if not observed_runtime_verification:
        return False
    if verified_runtime_numbers and read_program_numbers:
        return bool(verified_runtime_numbers & read_program_numbers)
    return True


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
            "scene_plan": state.plan.rag_context.get("scene_plan", {}),
            "scene_execution_evidence": state.plan.rag_context.get("scene_execution_evidence", {}),
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
        "repair_code": "Preserve the previous failed artifacts as repair context, then send the task back to CodeGenerationAgent to modify only scenario_fragment.inc and re-run Render/Validate/Compile.",
        "repair_execution": "Preserve the previous failed execution as repair context, then send the task back to ExecutionAgent to retry or adjust execution.",
    }
    return actions.get(stage, "No repair action is registered for this stage.")
