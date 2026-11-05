from __future__ import annotations

import time
import os
import uuid
import json
from pathlib import Path
from shutil import copy2
from typing import Any, Iterable

from ..agent_tools import (
    CollectExecutionArtifactsInput,
    CollectExecutionArtifactsTool,
    CompileCppInput,
    CompileGeneratedCppTool,
    RecordTrafficRelationsInput,
    RunExecutableInput,
    RunGeneratedExecutableTool,
)
from ..agent_tools.base import ToolProgressCallback
from ..focas_contracts import resolve_api_return_code
from ..orchestration.roles import ActionSpec, AgentTemplate, message_to_dict
from ..agent_roles import ExecutionResultSpec
from ..agent_tools.cpp_execution import (
    compile_cpp_script as _compile_cpp_script,
    csv_header as _csv_header,
    find_msvc_x86_toolchain as _find_msvc_x86_toolchain,
    interface_from_focas as _interface_from_focas,
    latest_existing_dir as _latest_existing_dir,
    read_cpp_csv_logs as _read_cpp_csv_logs,
    read_csv_by_index as _read_csv_by_index,
    safe_status_code as _safe_status_code,
    validate_cpp_csv_schema as _validate_cpp_csv_schema,
)
from ..integrations.ncguide import (
    DEFAULT_FOCAS_PORTS,
    FocasBridgeClient,
    FocasCppBridgeClient,
    default_focas_runtime_dir,
)
from ..integrations.simulator_client import SimulatedCncClient
from ..knowledge import KnowledgeBase
from ..api_prerequisites import classify_prerequisite_failure
from ..scenario_capabilities import assess_scene_execution
from ..traffic_relationship_mining import build_runtime_relation_payload, mine_runtime_relations
from ..models import ApiCallLog, CaptureEvent, ExecutionContext, ExecutionResult, GeneratedArtifacts, WorkflowState, utc_now
from ..parameter_rules import bind_plan_runtime_parameters
from ..execution_plan_schema import execution_plan_view, normalize_execution_plan, validate_execution_plan
from ..function_coverage import is_planned_expected_error
from ..tool_runtime import ToolRuntime, ToolRegistry
from ..tool_runtime import ToolCallLoop
from ..llm import LlmClient
from ..tools import semantic_label_for, supported_ncguide_readonly_interfaces
from ..utils import write_json, write_jsonl


def copy_generated_sources(
    artifacts: GeneratedArtifacts,
    execution_dir: Path,
    source_path: Path,
    local_source: Path,
) -> None:
    """Copy the main source and any adjacent include fragments for compilation.

    Fragment paths are copied by basename so relative includes remain valid in
    the execution working directory. Additional include artifacts can be added
    later as fields on GeneratedArtifacts without changing the compile flow.
    """
    copy2(source_path, local_source)
    for attribute in ("fragment_path", "bindings_path"):
        artifact_path = getattr(artifacts, attribute, None)
        if artifact_path is None:
            continue
        if not artifact_path.exists():
            raise ValueError(f"Generated include artifact was not found: {artifact_path}")
        copy2(artifact_path, execution_dir / artifact_path.name)


SUPPORTED_READONLY_INTERFACES = supported_ncguide_readonly_interfaces()


def execution_timeout_for_source(source: Path) -> int | None:
    """Return the process timeout, preserving explicit operator waits.

    The generated harness only uses an unbounded wait for the REMOTE gate.
    An explicit environment override remains available for diagnostics and
    test runs; ordinary API programs keep the existing bounded timeout.
    """
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if ('wait_for_operator(api, handle' in text
            or 'wait_for_cnc_mode(api, handle' in text):
        return None
    configured = os.environ.get("SMPAGENT_CPP_TIMEOUT_SECONDS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    # Keep the process watchdog aligned with the deterministic NC preflight
    # budget.  The extra minute covers connection, upload, logging and cleanup;
    # it must not turn an overlong NC program into an unbounded wait.
    try:
        nc_budget = float(os.environ.get("SMP_MAX_NC_RUNTIME_SECONDS", "90"))
    except ValueError:
        nc_budget = 90.0
    return max(120, min(3600, int(nc_budget + 60)))


class ExecutionAgent(AgentTemplate):
    role_name = "ExecutionRole"
    profile = "FOCAS execution agent"
    goal = "Compile and execute the generated artifact when the selected environment requires it."
    constraints = ("Honor the target environment.", "Collect only evidence permitted by the task.")

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        compile_tool: CompileGeneratedCppTool | None = None,
        run_tool: RunGeneratedExecutableTool | None = None,
        collect_tool: CollectExecutionArtifactsTool | None = None,
        tool_progress_callback: ToolProgressCallback | None = None,
        tool_runtime: ToolRuntime | None = None,
        llm_client: LlmClient | None = None,
    ) -> None:
        super().__init__()
        self.register_actions(ActionSpec("execute_artifact", "CodeArtifact", "ExecutionResult"))
        self.knowledge_base = knowledge_base
        self.compile_tool = compile_tool or CompileGeneratedCppTool()
        self.run_tool = run_tool or RunGeneratedExecutableTool()
        self.collect_tool = collect_tool or CollectExecutionArtifactsTool()
        self.tool_progress_callback = tool_progress_callback
        self.tool_runtime = tool_runtime
        self.llm_client = llm_client

    def _execution_runtime(self) -> ToolRuntime:
        if self.tool_runtime is not None:
            return self.tool_runtime
        registry = ToolRegistry()
        for tool in (self.compile_tool, self.run_tool, self.collect_tool):
            registry.register(tool)
        return ToolRuntime(registry, progress_callback=self.tool_progress_callback)

    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None or state.artifacts is None:
            raise ValueError("Cannot execute before plan and generated artifacts exist.")
        normalize_execution_plan(state.plan, state.plan.rag_context.get("traffic_relationships"))
        state.plan.rag_context["execution_plan"] = execution_plan_view(state.plan)
        state.plan.rag_context["execution_plan_validation"] = validate_execution_plan(state.plan)
        write_json(output_dir / "execution_plan.json", state.plan.rag_context["execution_plan"])
        if state.plan.code_spec is None or not getattr(state.plan.code_spec, "official_abi_context", ""):
            state.errors.append("ExecutionAgent received no Planner CodeSpec official ABI context.")
        if self.environment is not None:
            state.plan.rag_context["environment_shared_rules"] = self.environment.shared_rules
        soul_store = getattr(self, "soul_store", None)
        if soul_store is not None:
            state.plan.rag_context["executor_soul"] = soul_store.prompt_context(
                self.role_name,
                task_context=f"{state.request.description}; target_environment={state.request.target_environment}",
                error_type="; ".join(str(item) for item in state.errors[-4:]),
                memory_store=getattr(self, "memory_store", None),
            )

        target_environment = state.request.target_environment
        if target_environment == "ncguide-generated-cpp":
            result = run_generated_cpp_api_script(
                state,
                output_dir,
                self.knowledge_base,
                compile_tool=self.compile_tool,
                run_tool=self.run_tool,
                collect_tool=self.collect_tool,
                tool_progress_callback=self.tool_progress_callback,
                tool_runtime=self._execution_runtime(),
                llm_client=self.llm_client,
                use_model_tool_loop=bool(state.request.permissions.get("execution_tool_loop")),
            )
            self._record_relationship_feedback(result)
            self._publish_execution_message(result)
            return result

        client = make_execution_client(target_environment)
        api_logs: list[ApiCallLog] = []
        capture_events: list[CaptureEvent] = []
        execution_dir = output_dir / "execution"
        skipped_steps: list[str] = []
        skip_is_error = target_environment != "ncguide-bridge-readonly"

        for step in executable_steps(state.plan.steps, target_environment, skipped_steps):
            for _ in range(step.repeat):
                invocation_id = f"inv-{uuid.uuid4()}"
                start_timestamp_ns = time.time_ns()
                timestamp = utc_now()
                capture_events.append(
                    CaptureEvent(
                        timestamp=timestamp,
                        task_id=state.request.task_id,
                        interface_name=step.interface_name,
                        direction="request",
                        endpoint=endpoint_for(state.request.target_environment, "cnc"),
                        payload_summary=step.parameters,
                    )
                )
                api_name = str(step.api or step.protocol_function or step.interface_name).strip()
                # Semantic interfaces remain the simulator/bridge dispatch
                # key; the exact API is preserved in protocol_function and
                # the structured execution evidence.
                response = client.call(step.interface_name or api_name, step.parameters)
                end_timestamp_ns = time.time_ns()
                status_code = int(response.get("status_code", 500))
                response_timestamp = utc_now()
                capture_events.append(
                    CaptureEvent(
                        timestamp=response_timestamp,
                        task_id=state.request.task_id,
                        interface_name=step.interface_name,
                        direction="response",
                        endpoint=endpoint_for(state.request.target_environment, "agent"),
                        payload_summary=response,
                    )
                )
                api_logs.append(
                    ApiCallLog(
                        timestamp=timestamp,
                        task_id=state.request.task_id,
                        step_id=step.step_id,
                        phase=normalize_execution_phase(step.phase),
                        interface_name=step.interface_name,
                        input_parameters=step.parameters,
                        status_code=status_code,
                        response=response,
                        protocol_function=step.protocol_function or api_name,
                        semantic_label=semantic_label(step.interface_name),
                        error=response.get("error"),
                        invocation_id=invocation_id,
                        transaction_id=None,
                        start_timestamp_ns=start_timestamp_ns,
                        end_timestamp_ns=end_timestamp_ns,
                        device_context={
                            "target_environment": target_environment,
                            "phase": normalize_execution_phase(step.phase),
                            "plan_step_role": step.role,
                            "dependency_paths": [item.get("dependency_path") for item in step.dependencies if isinstance(item, dict)],
                        },
                        traffic_status="no_real_traffic_capture",
                    )
                )
                if step.interval_seconds:
                    time.sleep(min(step.interval_seconds, 0.05))

        errors = [
            log.error
            for log in api_logs
            if log.error and not planned_expected_error_for_log(state.plan, log)
        ]
        if skip_is_error:
            errors.extend(skipped_steps)
        success = not errors and all(
            log.status_code == 0
            or planned_expected_error_for_log(state.plan, log)
            for log in api_logs
        )
        write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
        write_jsonl(execution_dir / "capture_events.jsonl", capture_events)
        if skipped_steps:
            write_jsonl(execution_dir / "skipped_steps.jsonl", [{"message": item} for item in skipped_steps])

        state.execution_context = build_execution_context(api_logs)
        bind_plan_from_execution(state)
        state.result = ExecutionResult(
            task_id=state.request.task_id,
            success=success,
            api_logs=api_logs,
            capture_events=capture_events,
            output_dir=execution_dir,
            errors=[error for error in errors if error],
        )
        record_scene_evidence(state)
        self._record_relationship_feedback(state)
        state.errors.extend(state.result.errors)
        attach_execution_error_knowledge(state, self.knowledge_base)
        state.stage = "complete"
        self._publish_execution_message(state)
        return state

    def _record_relationship_feedback(self, state: WorkflowState) -> None:
        """Feed complete execution evidence back to TRCM without affecting collection.

        Relationship mining is a knowledge update, not an execution gate. A
        missing graph backend or malformed evidence is persisted as a
        diagnostic and never changes ``ExecutionResult.success``.
        """
        if state.plan is None or state.result is None:
            return
        service = getattr(self, "traffic_relationships", None)
        if service is None:
            return
        feedback: dict[str, object]
        if not state.result.success:
            feedback = {
                "status": "skipped",
                "reason": "runtime evidence is incomplete or execution failed",
                "relation_count": 0,
            }
        else:
            try:
                payload = build_runtime_relation_payload(state.plan, state.result)
                relation_tool = getattr(self, "tool_runtime", None)
                if relation_tool is not None:
                    recorded, _trace = relation_tool.call(
                        self.role_name,
                        "record_traffic_relations",
                        RecordTrafficRelationsInput(payload),
                    )
                    relations = recorded.get("accepted", []) if isinstance(recorded, dict) else []
                    rejected = recorded.get("rejected", []) if isinstance(recorded, dict) else []
                else:
                    relations = mine_runtime_relations(service, state.plan, state.result)
                    rejected = []
                feedback = {
                    "status": "recorded" if service.available else "unavailable",
                    "relation_count": len(relations),
                    "discovered_relations": payload.get("discovered_relations", []),
                    "relations": relations,
                    "structured_payload": payload,
                    "rejected_relations": rejected,
                }
                if payload.get("discovered_relations"):
                    print(
                        "[TRCM] 新发现关系（待人工确认，未写入 Neo4j）："
                        + str(payload["discovered_relations"]),
                        flush=True,
                    )
            except Exception as exc:  # pragma: no cover - defensive backend isolation
                feedback = {
                    "status": "failed",
                    "relation_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        state.plan.rag_context["traffic_relationship_feedback"] = feedback
        try:
            write_json(state.result.output_dir / "traffic_relationship_feedback.json", feedback)
        except OSError:
            # Artifact persistence is best effort; primary API logs remain intact.
            pass

    def _publish_execution_message(self, state: WorkflowState) -> None:
        if state.result is None:
            return
        message = self.publish(
            "execute_artifact", "ExecutionResult",
            success=state.result.success,
            api_log_count=len(state.result.api_logs),
            capture_event_count=len(state.result.capture_events),
        )
        state.messages.append(message_to_dict(
            message,
            ExecutionResultSpec(
                state.result.success,
                len(state.result.api_logs),
                len(state.result.capture_events),
            ),
        ))


def semantic_label(interface_name: str) -> str:
    return semantic_label_for(interface_name)


def build_execution_context(
    api_logs: Iterable[ApiCallLog],
    *,
    parameter_rules_by_target: dict[str, Iterable[dict[str, Any]]] | None = None,
) -> ExecutionContext:
    """Convert real API logs into the shared runtime context for repairs."""
    context = ExecutionContext()
    for log in api_logs:
        context.record(log)
        session_id = str(log.device_context.get("session_id", "")) if isinstance(log.device_context, dict) else ""
        if session_id:
            session = context.sessions.setdefault(session_id, {"calls": [], "complete": False})
            session["calls"].append({"api_name": log.protocol_function, "status_code": log.status_code, "invocation_id": log.invocation_id})
            if str(log.device_context.get("session_completion_status", "")).lower() in {"complete", "completed", "success"}:
                session["complete"] = True
    # Resolve filters only after all calls have been recorded.  This keeps the
    # operation deterministic and ensures a later API can consume real values
    # from an earlier successful call without another LLM round trip.
    if parameter_rules_by_target:
        from ..parameter_rules import resolve_runtime_parameter_bindings
        for target_api, rules in parameter_rules_by_target.items():
            binding = resolve_runtime_parameter_bindings(str(target_api), context, rules)
            if binding.get("selected_parameters"):
                context.selected_parameters[str(target_api)] = dict(binding["selected_parameters"])
    return context


def parameter_rules_by_target_from_plan(plan: Any) -> dict[str, list[dict[str, Any]]]:
    """Extract graph filters rules without making the executor query Neo4j."""
    result: dict[str, list[dict[str, Any]]] = {}
    context = getattr(plan, "rag_context", {}) if plan is not None else {}
    graph = context.get("traffic_relationships", {}) if isinstance(context, dict) else {}
    for target_api, facts in (graph.get("apis", {}) or {}).items() if isinstance(graph, dict) else ():
        if not isinstance(facts, dict):
            continue
        payload = facts.get("parameter_rules", {})
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        if isinstance(rules, list) and rules:
            result[str(target_api)] = [item for item in rules if isinstance(item, dict)]
    # Normalized plans already carry the same graph facts in runtime_bindings;
    # use them directly so execution never needs a second graph/LLM pass.
    for binding in getattr(plan, "runtime_bindings", ()) or ():
        if not isinstance(binding, dict) or binding.get("filter_mode") not in {"include", "exclude"}:
            continue
        target = str(binding.get("target_api", "")).strip()
        if not target:
            continue
        result.setdefault(target, []).append(dict(binding))
    return result


def parameter_domains_by_target_from_plan(plan: Any) -> dict[str, dict[str, list[Any]]]:
    """Read explicit legal domains from the plan/Neo4j API contracts only."""
    output: dict[str, dict[str, list[Any]]] = {}
    graph = getattr(plan, "rag_context", {}).get("traffic_relationships", {}) if plan is not None else {}
    apis = graph.get("apis", {}) if isinstance(graph, dict) else {}
    for api_name, facts in apis.items() if isinstance(apis, dict) else ():
        facts = facts if isinstance(facts, dict) else {}
        node = facts.get("node_contract") or (facts.get("api_info", {}) or {}).get("api") or {}
        parameters = node.get("parameters", {}) if isinstance(node, dict) else {}
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except (TypeError, ValueError):
                parameters = {}
        if not isinstance(parameters, dict):
            continue
        domains: dict[str, list[Any]] = {}
        for name, metadata in parameters.items():
            if not isinstance(metadata, dict):
                continue
            values = metadata.get("allowed_values", metadata.get("domain"))
            if values is None and isinstance(metadata.get("constraints"), dict):
                values = metadata["constraints"].get("allowed_values")
            if isinstance(values, (list, tuple, set)):
                domains[str(name)] = list(values)
        if domains:
            output[str(api_name)] = domains
    for step in getattr(plan, "steps", ()) or ():
        api = str(getattr(step, "api", "") or getattr(step, "protocol_function", "") or "").strip()
        params = getattr(step, "parameters", {})
        declared = params.get("parameter_domains") if isinstance(params, dict) else None
        if isinstance(declared, dict):
            output.setdefault(api, {}).update({str(key): list(value) for key, value in declared.items() if isinstance(value, (list, tuple, set))})
    return output


def bind_plan_from_execution(state: WorkflowState) -> dict[str, Any]:
    """Bind filters from current execution evidence and persist the result."""
    if state.plan is None or state.execution_context is None:
        return {"success": False, "errors": [{"code": "execution_context_missing", "message": "current execution evidence is unavailable"}]}
    result = bind_plan_runtime_parameters(
        state.plan,
        state.execution_context,
        parameter_rules_by_target_from_plan(state.plan),
        parameter_domains_by_target=parameter_domains_by_target_from_plan(state.plan),
        update_plan=False,
    )
    state.plan.rag_context["runtime_parameter_bindings"] = result.get("results", {})
    return result


def normalize_execution_phase(value: str) -> str:
    lowered = str(value or "during").strip().lower()
    return {"before": "Before", "after": "After"}.get(lowered, "During")


def record_scene_evidence(state: WorkflowState) -> None:
    """Persist observed scene outcomes independently of workflow success."""
    if state.plan is None or state.result is None:
        return
    scene_plan = state.plan.rag_context.get("scene_plan")
    if not isinstance(scene_plan, dict) or not scene_plan.get("target_api"):
        return
    evidence = assess_scene_execution(scene_plan, state.result.api_logs)
    state.plan.rag_context["scene_execution_evidence"] = evidence
    write_json(state.result.output_dir / "scene_execution_evidence.json", evidence)


def assess_plan_step_execution(plan: Any, api_logs: Iterable[ApiCallLog]) -> dict[str, Any]:
    """Compare planned protocol steps with observed calls.

    This is evidence reporting, not an online success gate.  In particular,
    destructive cleanup steps are reported as authorization-blocked when they
    were not executed; they are never inferred from a plan or run implicitly
    by the fixed transfer harness.
    """
    logs = list(api_logs)
    scene_plan = plan.rag_context.get("scene_plan", {}) if plan is not None else {}
    target_api = str(scene_plan.get("target_api", ""))
    if not target_api:
        targets = plan.rag_context.get("execution_contract", {}).get("targets", []) if plan is not None else []
        target_api = str(targets[0]) if targets else ""
    observed_counts: dict[str, int] = {}
    for log in logs:
        name = str(log.protocol_function or log.interface_name or "").strip()
        if name:
            observed_counts[name] = observed_counts.get(name, 0) + 1

    steps: list[dict[str, Any]] = []
    missing_step_ids: list[str] = []
    missing_protocol_functions: list[str] = []
    conditional_skipped_steps: list[str] = []
    authorization_blocked_steps: list[str] = []
    planned_protocol_steps: dict[str, list[Any]] = {}
    for planned_step in getattr(plan, "steps", []) or []:
        protocol = str(getattr(planned_step, "protocol_function", "") or "").strip()
        if protocol and not protocol.startswith("N/A"):
            planned_protocol_steps.setdefault(protocol, []).append(planned_step)
    # Assign duplicate protocol calls to duplicate planned steps. Generated
    # harnesses carry the owning step in the log (for repeated calls this is
    # usually ``S005-1`` ... ``S005-30``), so use that evidence first. The
    # repeat-capacity fallback below is kept for older harnesses that emitted
    # only the shared protocol name.
    step_matches: dict[str, list[ApiCallLog]] = {}
    for protocol, protocol_steps in planned_protocol_steps.items():
        protocol_logs = [log for log in logs if str(log.protocol_function or "").strip() == protocol]
        unassigned: list[ApiCallLog] = []
        for log in protocol_logs:
            log_step_id = str(getattr(log, "step_id", "") or "").strip()
            owner = next(
                (
                    planned_step
                    for planned_step in protocol_steps
                    if log_step_id == str(getattr(planned_step, "step_id", ""))
                    or log_step_id.startswith(f"{getattr(planned_step, 'step_id', '')}-")
                ),
                None,
            )
            if owner is None:
                unassigned.append(log)
            else:
                step_matches.setdefault(str(getattr(owner, "step_id", "")), []).append(log)
        # Older logs may have no useful step suffix. Allocate those calls to
        # planned steps in order, respecting each step's repeat capacity.
        for planned_step in protocol_steps:
            step_id = str(getattr(planned_step, "step_id", ""))
            capacity = max(1, int(getattr(planned_step, "repeat", 1) or 1))
            already_assigned = len(step_matches.get(step_id, []))
            take = max(0, capacity - already_assigned)
            if take:
                step_matches.setdefault(step_id, []).extend(unassigned[:take])
                del unassigned[:take]
        # If a legacy harness emitted more calls than the declared repeat,
        # preserve them under the last planned step instead of dropping data.
        if unassigned and protocol_steps:
            last_id = str(getattr(protocol_steps[-1], "step_id", ""))
            step_matches.setdefault(last_id, []).extend(unassigned)
    for step in getattr(plan, "steps", []) or []:
        protocol = str(getattr(step, "protocol_function", "") or "").strip()
        interface_name = str(getattr(step, "interface_name", "") or "").strip()
        step_id = str(getattr(step, "step_id", "") or "")
        if not protocol or protocol.strip().casefold().startswith("n/a"):
            steps.append({"step_id": step_id, "protocol_function": protocol, "status": "not_applicable", "observed_count": 0})
            continue
        matches = list(step_matches.get(step_id, []))
        if not matches:
            matches = [
                log for log in logs
                if not log.protocol_function and str(log.interface_name or "").strip() == interface_name
            ]
        destructive = (
            protocol in {"cnc_delete", "cnc_wrparam", "cnc_wrmacro", "pmc_wrpmc"}
            or protocol.startswith("pmc_wr")
            or "delete" in (interface_name + " " + str(getattr(step, "action", ""))).lower()
        )
        conditional = protocol == "cnc_getdtailerr"
        if not matches and conditional and not any(log.status_code != 0 or log.error for log in logs):
            status = "conditional_skipped"
            conditional_skipped_steps.append(step_id)
        elif not matches and destructive:
            status = "authorization_blocked"
            authorization_blocked_steps.append(step_id)
            missing_step_ids.append(step_id)
            missing_protocol_functions.append(protocol)
        elif not matches:
            status = "not_executed"
            missing_step_ids.append(step_id)
            missing_protocol_functions.append(protocol)
        else:
            # repeat is a planning upper bound for data transfer steps.  A
            # bounded session with fewer chunks is still an observed step.
            status = "executed"
        steps.append({
            "step_id": step_id,
            "phase": str(getattr(step, "phase", "")),
            "protocol_function": protocol,
            "interface_name": interface_name,
            "planned_repeat": int(getattr(step, "repeat", 1) or 1),
            "observed_count": len(matches),
            "status": status,
            "destructive": destructive,
        })

    successful_protocols = {
        str(log.protocol_function): log
        for log in logs
        if log.status_code == 0 and not log.error and log.protocol_function
    }
    session_chain = [str(item) for item in scene_plan.get("session_chain", []) if str(item).strip()]
    target_logs = [log for log in logs if log.protocol_function == target_api]
    target_called = bool(target_logs)
    target_succeeded = bool(target_logs) and all(log.status_code == 0 and not log.error for log in target_logs)
    session_complete = bool(session_chain) and all(name in successful_protocols for name in session_chain)
    if not session_chain:
        session_complete = target_succeeded
    nonconditional_missing = [
        item for item in steps
        if item["status"] in {"not_executed", "authorization_blocked"}
        and item["protocol_function"] not in {"cnc_getdtailerr"}
    ]
    return {
        "target_api": target_api,
        "target_called": target_called,
        "target_succeeded": target_succeeded,
        "target_session_complete": session_complete,
        "plan_execution_complete": not nonconditional_missing,
        "observed_call_counts": observed_counts,
        "steps": steps,
        "missing_step_ids": list(dict.fromkeys(missing_step_ids)),
        "missing_protocol_functions": list(dict.fromkeys(missing_protocol_functions)),
        "conditional_skipped_steps": conditional_skipped_steps,
        "authorization_blocked_steps": authorization_blocked_steps,
        "note": "Plan completeness is diagnostic; target collection success is based on compile, execution, return codes and raw evidence.",
    }


def run_generated_cpp_api_script(
    state: WorkflowState,
    output_dir: Path,
    knowledge_base: KnowledgeBase | None = None,
    *,
    compile_tool: CompileGeneratedCppTool | None = None,
    run_tool: RunGeneratedExecutableTool | None = None,
    collect_tool: CollectExecutionArtifactsTool | None = None,
    tool_progress_callback: ToolProgressCallback | None = None,
    tool_runtime: ToolRuntime | None = None,
    llm_client: LlmClient | None = None,
    use_model_tool_loop: bool = False,
) -> WorkflowState:
    assert state.plan is not None
    assert state.artifacts is not None
    execution_dir = output_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    source_path = state.artifacts.api_script_path
    if source_path is None or not source_path.exists():
        raise ValueError("Generated C++ API script was not found.")

    local_source = execution_dir / "api_script.cpp"
    exe_path = execution_dir / "api_script.exe"
    copy_generated_sources(state.artifacts, execution_dir, source_path, local_source)

    compile_tool = compile_tool or CompileGeneratedCppTool()
    run_tool = run_tool or RunGeneratedExecutableTool()
    collect_tool = collect_tool or CollectExecutionArtifactsTool()
    if tool_runtime is None:
        registry = ToolRegistry()
        for tool in (compile_tool, run_tool, collect_tool):
            registry.register(tool)
        tool_runtime = ToolRuntime(registry, progress_callback=tool_progress_callback)
    if use_model_tool_loop and llm_client is not None and llm_client.enabled:
        return run_generated_cpp_api_script_with_tool_loop(
            state,
            output_dir,
            tool_runtime,
            llm_client,
            self.knowledge_base,
            compile_tool,
            run_tool,
            collect_tool,
        )
    tool_calls = []
    compile_result, compile_trace = tool_runtime.call(
        "ExecutionRole",
        "compile_generated_cpp",
        CompileCppInput(local_source, exe_path, execution_dir),
    )
    tool_calls.append(compile_trace)
    errors: list[str] = []
    if not compile_result.success:
        errors.append("C++ API script compilation failed.")
        status_fields = collection_status_fields(state, [], False, "failed")
        write_jsonl(execution_dir / "tool_calls.jsonl", tool_calls)
        state.execution_context = build_execution_context([])
        state.result = ExecutionResult(
            task_id=state.request.task_id,
            success=False,
            api_logs=[],
            capture_events=[],
            output_dir=execution_dir,
            errors=errors,
            tool_calls=tool_calls,
            collection_status="failed",
        )
        write_json(execution_dir / "collection_result.json", {
            "execution_id": state.request.task_id,
            "status": "failed",
            **status_fields,
            "errors": errors,
            "api_logs_saved": False,
        })
        record_scene_evidence(state)
        state.errors.extend(errors)
        attach_execution_error_knowledge(state, knowledge_base)
        state.stage = "complete"
        return state

    timeout_seconds = execution_timeout_for_source(local_source)
    run_result, run_trace = tool_runtime.call(
        "ExecutionRole",
        "run_generated_executable",
        RunExecutableInput(
            executable_path=exe_path,
            work_dir=execution_dir,
            environment={
                "FOCAS_DLL_DIR": os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir())),
                "SMPAGENT_BATCH_ID": state.request.task_id,
                "SMP_MAX_NC_RUNTIME_SECONDS": os.environ.get("SMP_MAX_NC_RUNTIME_SECONDS", "90"),
            },
            timeout_seconds=timeout_seconds,
        ),
    )
    tool_calls.append(run_trace)
    if run_result.timed_out:
        errors.append(f"C++ API script timed out after {timeout_seconds} seconds.")
    elif run_result.exit_code != 0:
        errors.append(f"C++ API script execution failed with exit code {run_result.exit_code}.")

    collected, collect_trace = tool_runtime.call(
        "ExecutionRole",
        "collect_execution_artifacts",
        CollectExecutionArtifactsInput(state.request.task_id, execution_dir),
    )
    tool_calls.append(collect_trace)
    api_logs = collected.api_logs
    capture_events = collected.capture_events
    plan_execution_evidence = assess_plan_step_execution(state.plan, api_logs)
    state.plan.rag_context["plan_execution_evidence"] = plan_execution_evidence
    write_json(execution_dir / "plan_execution_evidence.json", plan_execution_evidence)
    if collected.schema_errors:
        errors.extend(f"CAPTURE_ARTIFACT_ERROR: {item}" for item in collected.schema_errors)
    for log in api_logs:
        if (
            log.status_code != 0
            and log.error
            and not generated_cpp_log_is_nonfatal(log)
            and not planned_expected_error_for_log(state.plan, log)
        ):
            errors.append(log.error)
    prerequisite_failures = [
        {
            "function": log.protocol_function,
            "status_code": log.status_code,
            "error": log.error,
            "analysis": classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or ""),
        }
        for log in api_logs
        if log.status_code != 0 and classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or "")
    ]
    if prerequisite_failures and state.plan is not None:
        state.plan.rag_context["api_prerequisite_runtime"] = {
            "failures": prerequisite_failures,
            "analysis_attempted": False,
        }
    errors.extend(required_traffic_errors(state, api_logs))
    success = not errors and bool(api_logs)
    # Persist the primary collection record before running any quality audit.
    write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
    write_jsonl(execution_dir / "capture_events.jsonl", capture_events)
    write_jsonl(execution_dir / "tool_calls.jsonl", tool_calls)
    audit_status = "failed"
    try:
        from ..offline_audit import build_offline_audit
        audit = build_offline_audit(
            state.plan, api_logs, state.plan.rag_context.get("scene_execution_evidence")
        )
        audit_status = str(audit.get("status", "failed"))
        write_json(execution_dir / "offline_audit.json", audit)
        # Retain the legacy filename for existing consumers; it is audit-only.
        write_json(execution_dir / "parameter_coverage.json", audit.get("parameter_coverage", {}))
    except Exception as exc:
        audit_status = "failed"
        try:
            write_json(execution_dir / "offline_audit.json", {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}"
            })
        except OSError:
            pass
    collection_status, collection_evidence = classify_collection_status(state, api_logs, success)
    status_fields = collection_status_fields(state, api_logs, success, collection_status)
    collection_evidence.update({
        "execution_id": state.request.task_id,
        "api_return_codes": [
            {"api_name": log.protocol_function, "call_id": log.invocation_id,
             "return_code": log.status_code, "error": log.error}
            for log in api_logs if log.protocol_function
        ],
        "pcap_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in sorted((execution_dir / "data").rglob("*.pcap"))
        ],
        "pcap_mode": (
            "runtime_per_api_dumper"
            if (execution_dir / "data" / "by_api" / ".runtime_per_api_capture").is_file()
            else "batch_total_or_fallback"
        ),
        "api_capture_summary": build_api_capture_summary(api_logs),
        "generated_code": str(local_source),
        "generated_code_sha256": __import__("hashlib").sha256(local_source.read_bytes()).hexdigest()
        if local_source.is_file() else None,
        "plan_execution_evidence": plan_execution_evidence,
        "status_fields": status_fields,
        "errors": [error for error in errors if error],
    })
    collection_evidence["traffic_classification"] = write_api_traffic_classification(
        execution_dir, api_logs
    )
    try:
        write_json(execution_dir / "collection_result.json", {
            "status": collection_status,
            **status_fields,
            "evidence": collection_evidence,
            "offline_audit_status": audit_status,
        })
    except OSError as exc:
        collection_evidence["collection_manifest_write_error"] = f"{type(exc).__name__}: {exc}"

    state.execution_context = build_execution_context(
        api_logs,
        parameter_rules_by_target=parameter_rules_by_target_from_plan(state.plan),
    )
    binding_result = bind_plan_from_execution(state)
    if not binding_result.get("success", True):
        state.plan.rag_context["parameter_binding_errors"] = binding_result.get("results", {})
    if state.execution_context.selected_parameters:
        state.plan.rag_context["runtime_parameter_bindings"] = state.execution_context.selected_parameters
        write_json(
            execution_dir / "runtime_parameter_bindings.json",
            state.execution_context.selected_parameters,
        )
    state.result = ExecutionResult(
        task_id=state.request.task_id,
        success=success,
        api_logs=api_logs,
        capture_events=capture_events,
        output_dir=execution_dir,
        errors=[error for error in errors if error],
        tool_calls=tool_calls,
        collection_status=collection_status,
        collection_evidence=collection_evidence,
        offline_audit_status=audit_status,
    )
    record_scene_evidence(state)
    state.errors.extend(state.result.errors)
    attach_execution_error_knowledge(state, knowledge_base)
    state.stage = "complete"
    return state


def run_generated_cpp_api_script_with_tool_loop(
    state: WorkflowState,
    output_dir: Path,
    tool_runtime: ToolRuntime,
    llm_client: LlmClient,
    knowledge_base: KnowledgeBase | None,
    compile_tool: CompileGeneratedCppTool,
    run_tool: RunGeneratedExecutableTool,
    collect_tool: CollectExecutionArtifactsTool,
) -> WorkflowState:
    """Use an Agent-local tool loop while preserving the safe execution order."""
    assert state.artifacts is not None
    execution_dir = output_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    source_path = state.artifacts.api_script_path
    if source_path is None or not source_path.exists():
        raise ValueError("Generated C++ API script was not found.")
    local_source = execution_dir / "api_script.cpp"
    exe_path = execution_dir / "api_script.exe"
    copy_generated_sources(state.artifacts, execution_dir, source_path, local_source)
    timeout_seconds = execution_timeout_for_source(local_source)

    def input_factory(tool_name: str, arguments: dict[str, object]) -> object:
        if tool_name == "compile_generated_cpp":
            return CompileCppInput(local_source, exe_path, execution_dir)
        if tool_name == "run_generated_executable":
            return RunExecutableInput(
                executable_path=exe_path,
                work_dir=execution_dir,
                environment={
                    "FOCAS_DLL_DIR": os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir())),
                    "SMPAGENT_BATCH_ID": state.request.task_id,
                    "SMP_MAX_NC_RUNTIME_SECONDS": os.environ.get("SMP_MAX_NC_RUNTIME_SECONDS", "90"),
                },
                timeout_seconds=timeout_seconds,
            )
        if tool_name == "collect_execution_artifacts":
            return CollectExecutionArtifactsInput(state.request.task_id, execution_dir)
        raise ValueError(f"ExecutionAgent tool loop does not support {tool_name}")

    loop = ToolCallLoop(llm_client, tool_runtime, max_turns=5)
    loop.run(
        "ExecutionRole",
        (
            "You are the ExecutionAgent tool coordinator. Execute exactly these tools in this order: "
            "compile_generated_cpp, run_generated_executable, collect_execution_artifacts. "
            "Pass no invented paths or environment values; the harness supplies typed inputs. "
            "If compilation fails, stop and return final JSON with success=false. "
            "Return final JSON only after the required tool calls."
            f"\n\nDynamic Agent Soul:\n{state.plan.rag_context.get('executor_soul', '')}\n"
            "Treat graph proposals as diagnostics only; preserve the typed tool order and raw evidence."
        ),
        f"Execute generated artifact for task {state.request.task_id}. Required ordered tools: "
        "compile_generated_cpp -> run_generated_executable -> collect_execution_artifacts.",
        input_factory=input_factory,
    )
    ordered = [item for item in loop.last_tool_outputs if item[0] in {
        "compile_generated_cpp", "run_generated_executable", "collect_execution_artifacts"
    }]
    names = [item[0] for item in ordered]
    tool_calls = [item[2] for item in ordered]
    errors: list[str] = []
    expected_prefix = ["compile_generated_cpp", "run_generated_executable", "collect_execution_artifacts"]
    if names != expected_prefix[: len(names)]:
        errors.append(
            "ExecutionAgent tool loop violated required order: "
            "compile_generated_cpp -> run_generated_executable -> collect_execution_artifacts."
        )
    compile_output = next((item[1] for item in ordered if item[0] == "compile_generated_cpp"), None)
    if compile_output is not None and not compile_output.success:
        errors.append("C++ API script compilation failed.")
    run_output = next((item[1] for item in ordered if item[0] == "run_generated_executable"), None)
    if run_output is not None:
        if run_output.timed_out:
            errors.append(f"C++ API script timed out after {run_output.timeout_seconds} seconds.")
        elif run_output.exit_code != 0:
            errors.append(f"C++ API script execution failed with exit code {run_output.exit_code}.")
    collected = next((item[1] for item in ordered if item[0] == "collect_execution_artifacts"), None)
    api_logs = collected.api_logs if collected is not None else []
    capture_events = collected.capture_events if collected is not None else []
    plan_execution_evidence = assess_plan_step_execution(state.plan, api_logs)
    state.plan.rag_context["plan_execution_evidence"] = plan_execution_evidence
    write_json(execution_dir / "plan_execution_evidence.json", plan_execution_evidence)
    if collected is None:
        errors.append("ExecutionAgent tool loop did not collect execution artifacts.")
    elif collected.schema_errors:
        errors.extend(f"CAPTURE_ARTIFACT_ERROR: {item}" for item in collected.schema_errors)
    for log in api_logs:
        if (
            log.status_code != 0
            and log.error
            and not generated_cpp_log_is_nonfatal(log)
            and not planned_expected_error_for_log(state.plan, log)
        ):
            errors.append(log.error)
    prerequisite_failures = [
        {
            "function": log.protocol_function,
            "status_code": log.status_code,
            "error": log.error,
            "analysis": classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or ""),
        }
        for log in api_logs
        if log.status_code != 0 and classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or "")
    ]
    if prerequisite_failures and state.plan is not None:
        state.plan.rag_context["api_prerequisite_runtime"] = {
            "failures": prerequisite_failures,
            "analysis_attempted": False,
        }
    errors.extend(required_traffic_errors(state, api_logs))
    success = not errors and bool(api_logs)
    write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
    write_jsonl(execution_dir / "capture_events.jsonl", capture_events)
    write_jsonl(execution_dir / "tool_calls.jsonl", tool_calls)
    audit_status = "failed"
    try:
        from ..offline_audit import build_offline_audit
        audit = build_offline_audit(
            state.plan, api_logs, state.plan.rag_context.get("scene_execution_evidence")
        )
        audit_status = str(audit.get("status", "failed"))
        write_json(execution_dir / "offline_audit.json", audit)
        write_json(execution_dir / "parameter_coverage.json", audit.get("parameter_coverage", {}))
    except Exception as exc:
        audit_status = "failed"
        try:
            write_json(execution_dir / "offline_audit.json", {
                "status": "failed", "error": f"{type(exc).__name__}: {exc}"
            })
        except OSError:
            pass
    collection_status, collection_evidence = classify_collection_status(state, api_logs, success)
    status_fields = collection_status_fields(state, api_logs, success, collection_status)
    collection_evidence.update({
        "execution_id": state.request.task_id,
        "api_return_codes": [
            {"api_name": log.protocol_function, "call_id": log.invocation_id,
             "return_code": log.status_code, "error": log.error}
            for log in api_logs if log.protocol_function
        ],
        "pcap_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in sorted((execution_dir / "data").rglob("*.pcap"))
        ],
        "pcap_mode": (
            "runtime_per_api_dumper"
            if (execution_dir / "data" / "by_api" / ".runtime_per_api_capture").is_file()
            else "batch_total_or_fallback"
        ),
        "api_capture_summary": build_api_capture_summary(api_logs),
        "generated_code": str(local_source),
        "generated_code_sha256": __import__("hashlib").sha256(local_source.read_bytes()).hexdigest()
        if local_source.is_file() else None,
        "plan_execution_evidence": plan_execution_evidence,
        "status_fields": status_fields,
        "errors": list(errors),
    })
    collection_evidence["traffic_classification"] = write_api_traffic_classification(
        execution_dir, api_logs
    )
    try:
        write_json(execution_dir / "collection_result.json", {
            "status": collection_status,
            **status_fields,
            "evidence": collection_evidence,
            "offline_audit_status": audit_status,
        })
    except OSError as exc:
        collection_evidence["collection_manifest_write_error"] = f"{type(exc).__name__}: {exc}"
    state.execution_context = build_execution_context(
        api_logs,
        parameter_rules_by_target=parameter_rules_by_target_from_plan(state.plan),
    )
    binding_result = bind_plan_from_execution(state)
    if not binding_result.get("success", True):
        state.plan.rag_context["parameter_binding_errors"] = binding_result.get("results", {})
    if state.execution_context.selected_parameters:
        state.plan.rag_context["runtime_parameter_bindings"] = state.execution_context.selected_parameters
        write_json(
            execution_dir / "runtime_parameter_bindings.json",
            state.execution_context.selected_parameters,
        )
    state.result = ExecutionResult(
        task_id=state.request.task_id,
        success=success,
        api_logs=api_logs,
        capture_events=capture_events,
        output_dir=execution_dir,
        errors=errors,
        tool_calls=tool_calls,
        collection_status=collection_status,
        collection_evidence=collection_evidence,
        offline_audit_status=audit_status,
    )
    record_scene_evidence(state)
    state.errors.extend(errors)
    attach_execution_error_knowledge(state, knowledge_base)
    state.stage = "complete"
    return state


def generated_cpp_log_is_nonfatal(log: ApiCallLog) -> bool:
    """Return True for optional generated-C++ coverage warnings.

    The generated NCGuide runner may include extra planned/read-only FOCAS APIs
    to broaden traffic coverage.  Unsupported adapter stubs and controller
    option/mode returns should remain visible in CSV artifacts, but they should
    not make an otherwise completed compile/run/capture lifecycle fail.
    """

    text = " ".join(
        [
            str(log.error or ""),
            str(log.response.get("return_text", "")),
            str(log.response.get("data", "")),
            log.protocol_function,
            log.step_id,
        ]
    ).lower()
    if "program_completion_gate" in log.protocol_function.lower() or "program_completion_gate" in log.step_id.lower():
        return False
    # For broad API coverage, an unsupported-function response is itself a
    # useful controller response *only when the real request/response was
    # captured*. It must not turn a missing-PCAP run into success.
    if log.traffic_status == "matched" and "ew_func" in text:
        return True
    # cnc_dncstart2 is a documented observation of an already active
    # DNC/M198 operation.  EW_REJECT means that no such operation is active;
    # it is a valid negative observation when the real request/response was
    # captured.  Keep the return code in the raw log, but do not route the
    # completed observation into code repair.
    if (
        log.protocol_function == "cnc_dncstart2"
        and log.traffic_status == "matched"
        and any(code in text for code in ("ew_reject", "ew_busy", "ew_noopt", "ew_param"))
    ):
        return True
    return any(
        marker in text
        for marker in [
            "skipped_unsupported_by_fixed_adapter",
            "unsupported_symbol",
            "ew_noopt",
            "ew_mode",
            "focas_return_-7",
        ]
    )


def planned_expected_error_for_log(plan: Any, log: ApiCallLog) -> bool:
    """Return whether a nonzero log is the exact error branch in the plan."""
    if plan is None:
        return False
    return is_planned_expected_error(
        plan,
        log.protocol_function,
        {
            "step_id": log.step_id,
            "status_code": log.status_code,
            "input_parameters": log.input_parameters,
            "error": log.error,
            "response": log.response,
        },
    )


def required_traffic_errors(state: WorkflowState, api_logs: list[ApiCallLog]) -> list[str]:
    """Keep missing target calls fatal; wire matching belongs to offline audit."""
    required = set(state.request.permissions.get("required_api_functions") or [])
    required.update(state.request.permissions.get("coverage_current_batch_functions") or [])
    if state.request.permissions.get("require_planned_target_traffic") and state.plan is not None:
        manifest = state.plan.rag_context.get("function_coverage_manifest", {})
        for item in manifest.get("selected_batch", []) if isinstance(manifest, dict) else []:
            if isinstance(item, dict) and item.get("coverage_role", "target") == "target":
                required.add(str(item.get("function", "")))
    called = {log.protocol_function for log in api_logs}
    return [
        f"TARGET_API_NOT_CALLED: {function}"
        for function in sorted(name for name in required
                               if name.startswith(("cnc_", "pmc_")) and name not in called)
    ]


def classify_collection_status(
    state: WorkflowState,
    api_logs: list[ApiCallLog],
    execution_succeeded: bool,
) -> tuple[str, dict[str, object]]:
    """Classify online collection independently from offline traffic quality."""
    from ..function_coverage import is_non_wire_observable

    targets = set(state.request.permissions.get("required_api_functions") or [])
    targets.update(state.request.permissions.get("coverage_current_batch_functions") or [])
    if state.plan is not None:
        manifest = state.plan.rag_context.get("function_coverage_manifest", {})
        targets.update(
            str(row.get("function", ""))
            for row in manifest.get("selected_batch", [])
            if isinstance(row, dict) and row.get("coverage_role", "target") == "target"
        )
    targets.discard("")
    by_target = {
        name: [log for log in api_logs if log.protocol_function == name]
        for name in sorted(targets)
    }
    called = {name: len(logs) for name, logs in by_target.items()}
    def accepted(log: ApiCallLog) -> bool:
        return (
            log.status_code == 0
            and not log.error
        ) or planned_expected_error_for_log(state.plan, log)

    local_targets = {
        name for name, logs in by_target.items()
        if is_non_wire_observable(name) and logs
        and all(accepted(log) for log in logs)
    }
    wire_targets = {
        name for name, logs in by_target.items()
        if any(log.traffic_status in {"matched", "request_only_segment"} for log in logs)
    }
    blocked = any(log.protocol_function in {"host_api_prerequisite", "host_runtime_exception"}
                  and log.status_code != 0 for log in api_logs)
    if blocked and not wire_targets:
        status = "blocked"
    elif not execution_succeeded:
        status = "failed"
    elif targets and targets <= local_targets:
        status = "local_success"
    elif targets and targets <= wire_targets:
        status = "collected"
    elif targets and targets <= (wire_targets | local_targets) and wire_targets:
        status = "collected"
    elif targets:
        status = "partial"
    else:
        status = "local_success" if api_logs and all(log.status_code == 0 for log in api_logs) else "failed"
    return status, {
        "target_apis": sorted(targets),
        "target_call_counts": called,
        "target_apis_with_wire_evidence": sorted(wire_targets),
        "target_apis_without_wire_evidence": sorted(targets - wire_targets - local_targets),
        "wire_evidence_status": (
            "not_required" if targets and targets <= local_targets
            else "observed" if wire_targets
            else "insufficient"
        ),
        "target_api_calls": [
            {"api_name": log.protocol_function, "call_id": log.invocation_id,
             "step_id": log.step_id, "return_code": log.status_code,
             "input_parameters": log.input_parameters, "error": log.error}
            for log in api_logs if log.protocol_function in targets
        ],
        "auxiliary_api_calls": [
            {"api_name": log.protocol_function, "call_id": log.invocation_id,
             "step_id": log.step_id, "return_code": log.status_code,
             "input_parameters": log.input_parameters, "error": log.error}
            for log in api_logs if log.protocol_function and log.protocol_function not in targets
        ],
    }


def collection_status_fields(
    state: WorkflowState,
    api_logs: list[ApiCallLog],
    execution_succeeded: bool,
    collection_status: str,
) -> dict[str, str]:
    """Expose independent workflow, API, capture, and annotation states.

    A successful FOCAS return is deliberately independent from wire evidence:
    a local/cache-served call can be ``api_execution_status=success`` while
    its capture status remains ``partial``. This keeps the online result
    truthful and gives the desktop monitor a concrete reason for partial runs.
    """
    targets = set(state.request.permissions.get("required_api_functions") or [])
    targets.update(state.request.permissions.get("coverage_current_batch_functions") or [])
    if state.plan is not None:
        manifest = state.plan.rag_context.get("function_coverage_manifest", {})
        targets.update(
            str(row.get("function", ""))
            for row in manifest.get("selected_batch", [])
            if isinstance(row, dict) and row.get("coverage_role", "target") == "target"
        )
    targets.discard("")
    target_logs = [log for log in api_logs if log.protocol_function in targets]
    def accepted(log: ApiCallLog) -> bool:
        return (
            log.status_code == 0
            and not log.error
        ) or planned_expected_error_for_log(state.plan, log)

    api_ok = bool(target_logs) and all(accepted(log) for log in target_logs)
    from ..function_coverage import is_non_wire_observable
    local_only = bool(target_logs) and bool(targets) and targets <= {
        log.protocol_function for log in target_logs if is_non_wire_observable(log.protocol_function)
        and accepted(log)
    }
    wire_ok = any(log.traffic_status in {"matched", "request_only_segment"} for log in target_logs)
    all_raw_fields = bool(target_logs) and all(
        log.invocation_id and log.protocol_function and log.traffic_status
        and isinstance(log.response, dict)
        for log in target_logs
    )
    if not execution_succeeded:
        workflow_status = "failed"
    else:
        workflow_status = "completed"
    if not target_logs:
        api_execution_status = "target_not_called"
    else:
        api_execution_status = "success" if api_ok else "failed"
    if not target_logs:
        capture_status = "not_observed"
        capture_reason = "没有目标 API 调用记录"
    elif local_only:
        capture_status = "not_required"
        capture_reason = (
            "目标 API 是 FOCAS 句柄本地路径选择操作；本批次已真实调用并返回 EW_OK，"
            "该函数本身不要求独立 CNC 网络请求，已保留无报文 PCAP 和调用时间证据"
        )
    elif wire_ok:
        capture_status = "matched"
        capture_reason = "至少一条目标调用已关联真实 PCAP 证据"
    else:
        capture_status = "partial"
        capture_reason = (
            "目标 API 已执行，但没有目标调用在严格调用时间窗口内关联到真实 PCAP 请求/响应；"
            "不得使用相邻辅助调用补配 transaction_id"
        )
    annotation_status = "complete" if all_raw_fields else "partial"
    return {
        "workflow_status": workflow_status,
        "api_execution_status": api_execution_status,
        "capture_status": capture_status,
        "annotation_status": annotation_status,
        "capture_reason": capture_reason,
        "collection_status": collection_status,
    }


def build_api_capture_summary(api_logs: list[ApiCallLog]) -> dict[str, dict[str, Any]]:
    """Aggregate all calls for each API without treating file ownership as a match."""
    summary: dict[str, dict[str, Any]] = {}
    for log in api_logs:
        api = str(log.protocol_function or "").strip()
        if not api or not api.startswith(("cnc_", "pmc_")):
            continue
        item = summary.setdefault(api, {
            "call_count": 0,
            "success_count": 0,
            "traffic_matched_count": 0,
            "request_only_segment_count": 0,
            "unmatched_call_count": 0,
            "parameter_configurations": [],
            "pcap_files": [],
            "pcap_call_count": 0,
        })
        item["call_count"] += 1
        if log.status_code == 0 and not log.error:
            item["success_count"] += 1
        if log.traffic_status == "matched":
            item["traffic_matched_count"] += 1
        elif log.traffic_status == "request_only_segment":
            # A segment is real call-linked traffic, but it is not an
            # independent request/response transaction.
            item["request_only_segment_count"] += 1
        elif log.traffic_status not in {"not_applicable", "pending_transaction_mapping"}:
            item["unmatched_call_count"] += 1
        configuration = json.dumps(log.input_parameters, ensure_ascii=False, sort_keys=True, default=str)
        if configuration not in item["parameter_configurations"]:
            item["parameter_configurations"].append(configuration)
        pcap_file = log.response.get("api_pcap_file") if isinstance(log.response, dict) else None
        if pcap_file and pcap_file not in item["pcap_files"]:
            item["pcap_files"].append(pcap_file)
        if pcap_file:
            item["pcap_call_count"] += 1
    for item in summary.values():
        item["distinct_parameter_configurations"] = len(item["parameter_configurations"])
    return summary


def write_api_traffic_classification(
    execution_dir: Path,
    api_logs: list[ApiCallLog],
) -> dict[str, Any]:
    """Persist explicit with-traffic and without-traffic API manifests.

    Network evidence is intentionally independent from API execution status.
    A successful local FOCAS handle operation belongs in the no-traffic file
    with ``local_success``; a failed call remains visible there with its real
    execution status and is not converted into a success.
    """
    from ..function_coverage import is_non_wire_observable

    grouped: dict[str, list[ApiCallLog]] = {}
    for log in api_logs:
        api = str(log.protocol_function or "").strip()
        if api.startswith(("cnc_", "pmc_")):
            grouped.setdefault(api, []).append(log)

    with_traffic: list[dict[str, Any]] = []
    without_traffic: list[dict[str, Any]] = []
    for api, logs in sorted(grouped.items()):
        matched_count = sum(log.traffic_status == "matched" for log in logs)
        segment_count = sum(log.traffic_status == "request_only_segment" for log in logs)
        success_count = sum(log.status_code == 0 and not log.error for log in logs)
        pcap_files = sorted({
            str(log.response.get("api_pcap_file"))
            for log in logs
            if isinstance(log.response, dict) and log.response.get("api_pcap_file")
        })
        calls = [
            {
                "invocation_id": log.invocation_id,
                "step_id": log.step_id,
                "return_code": log.status_code,
                "traffic_status": log.traffic_status,
                "transaction_id": log.transaction_id,
            }
            for log in logs
        ]
        item = {
            "api_name": api,
            "call_count": len(logs),
            "success_count": success_count,
            "failed_count": len(logs) - success_count,
            "traffic_matched_count": matched_count,
            "request_only_segment_count": segment_count,
            "no_traffic_call_count": len(logs) - matched_count - segment_count,
            "pcap_files": pcap_files,
            "calls": calls,
        }
        if matched_count or segment_count:
            item["classification"] = "with_traffic"
            item["traffic_evidence"] = "matched_or_request_only_segment"
            with_traffic.append(item)
        else:
            item["classification"] = "without_traffic"
            item["traffic_evidence"] = "not_required_local_operation" if is_non_wire_observable(api) else "no_wire_evidence"
            item["execution_classification"] = (
                "local_success" if success_count == len(logs) and is_non_wire_observable(api)
                else "api_execution_success_without_wire_evidence" if success_count == len(logs)
                else "api_execution_failed"
            )
            without_traffic.append(item)

    with_path = execution_dir / "apis_with_traffic.json"
    without_path = execution_dir / "apis_without_traffic.json"
    write_json(with_path, {
        "schema_version": "api-traffic-classification-v1",
        "classification": "with_traffic",
        "api_count": len(with_traffic),
        "apis": with_traffic,
    })
    write_json(without_path, {
        "schema_version": "api-traffic-classification-v1",
        "classification": "without_traffic",
        "api_count": len(without_traffic),
        "apis": without_traffic,
    })
    return {
        "with_traffic_file": str(with_path),
        "without_traffic_file": str(without_path),
        "with_traffic_api_count": len(with_traffic),
        "without_traffic_api_count": len(without_traffic),
    }


def attach_execution_error_knowledge(state: WorkflowState, knowledge_base: KnowledgeBase | None) -> None:
    if state.plan is None or state.result is None or not state.result.errors:
        return
    failed_logs = []
    for log in state.result.api_logs:
        if not (log.error or log.status_code != 0):
            continue
        function = log.protocol_function.strip()
        diagnosis = (
            resolve_api_return_code(function, log.status_code)
            if function.startswith(("cnc_", "pmc_")) and "/" not in function
            else {}
        )
        failed_logs.append({
            "step_id": log.step_id,
            "interface_name": log.interface_name,
            "protocol_function": log.protocol_function,
            "status_code": log.status_code,
            "error": log.error,
            "data": str(log.response.get("data", ""))[:220],
            "return_code_names": diagnosis.get("api_code_names") or diagnosis.get("global_code_names") or [],
            "api_return_code_explanations": diagnosis.get("api_explanations", []),
            "return_code_documented_for_api": diagnosis.get("documented_for_api", False),
        })
    failed_logs = failed_logs[-8:]
    state.plan.rag_context["execution_return_code_diagnostics"] = failed_logs
    if knowledge_base is None:
        return
    query = "\n".join(
        [
            state.request.description,
            f"target_environment={state.request.target_environment}",
            f"errors={state.result.errors[-8:]}",
            f"failed_logs={failed_logs}",
            "FOCAS return code compile execution failure troubleshooting repair",
        ]
    )
    rows = knowledge_base.search_rules(query, top_k=4) + knowledge_base.search_api(query, top_k=4)
    state.plan.rag_context["execution_error_knowledge"] = summarize_retrieved_for_execution(rows)


def summarize_retrieved_for_execution(rows) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in rows:
        chunk_id = item.chunk.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        summary.append(
            {
                "chunk_id": chunk_id,
                "score": item.score,
                "source_type": item.chunk.metadata.get("source_type") or item.chunk.metadata.get("type"),
                "rule_type": item.chunk.metadata.get("rule_type"),
                "function": item.chunk.metadata.get("function"),
                "scenario": item.chunk.metadata.get("scenario") or item.chunk.metadata.get("scene"),
                "preview": item.chunk.text[:360],
            }
        )
        if len(summary) >= 6:
            break
    return summary


def compile_cpp_script(source_path: Path, exe_path: Path, work_dir: Path):
    """Compatibility export for callers that have not migrated to CompileGeneratedCppTool."""

    return _compile_cpp_script(source_path, exe_path, work_dir)


def find_msvc_x86_toolchain() -> tuple[Path, dict[str, str]] | None:
    return _find_msvc_x86_toolchain()


def latest_existing_dir(roots: list[Path]) -> Path | None:
    return _latest_existing_dir(roots)


def read_cpp_csv_logs(state: WorkflowState, execution_dir: Path) -> tuple[list[ApiCallLog], list[CaptureEvent]]:
    collected = _read_cpp_csv_logs(state.request.task_id, execution_dir)
    return collected.api_logs, collected.capture_events


def validate_cpp_csv_schema(input_path: Path, output_path: Path) -> list[str]:
    return _validate_cpp_csv_schema(input_path, output_path)


def csv_header(path: Path) -> list[str]:
    return _csv_header(path)


def read_csv_by_index(path: Path) -> dict[str, dict[str, str]]:
    return _read_csv_by_index(path)


def safe_status_code(value: str | None) -> int:
    return _safe_status_code(value)


def interface_from_focas(api_name: str) -> str:
    return _interface_from_focas(api_name)


def make_execution_client(target_environment: str):
    if target_environment in {"ncguide-bridge", "ncguide-bridge-readonly"}:
        cpp_bridge = Path(os.getenv("FOCAS_CPP_BRIDGE_EXE", ".tools/focas_bridge_cpp/focas_bridge.exe"))
        focas_runtime_dir = Path(os.getenv("FOCAS_DLL_DIR", str(default_focas_runtime_dir())))
        if cpp_bridge.exists():
            return FocasCppBridgeClient(
                bridge_exe=cpp_bridge,
                install_dir=focas_runtime_dir,
                host="127.0.0.1",
                port=int(os.getenv("FOCAS_BRIDGE_PORT", str(DEFAULT_FOCAS_PORTS[0]))),
            )
        return FocasBridgeClient(
            python_exe=Path(os.getenv("FOCAS_BRIDGE_PYTHON", "C:/Python32/python.exe")),
            install_dir=focas_runtime_dir,
            host="127.0.0.1",
            port=int(os.getenv("FOCAS_BRIDGE_PORT", str(DEFAULT_FOCAS_PORTS[0]))),
        )
    return SimulatedCncClient()


def endpoint_for(target_environment: str, side: str) -> str:
    if target_environment in {"ncguide-bridge", "ncguide-bridge-readonly"}:
        return "ncguide://focas" if side == "cnc" else "smpagent://bridge"
    return "simulator://cnc" if side == "cnc" else "simulator://agent"


def executable_steps(steps, target_environment: str, skipped_steps: list[str]):
    if target_environment != "ncguide-bridge-readonly":
        return steps
    selected = []
    for step in steps:
        if step.interface_name in SUPPORTED_READONLY_INTERFACES:
            selected.append(step)
        else:
            skipped_steps.append(
                f"Skipped {step.step_id} {step.interface_name}: ncguide-bridge-readonly supports only {sorted(SUPPORTED_READONLY_INTERFACES)}"
            )
    if selected:
        return selected
    skipped_steps.append("No supported read-only step was available; inserted read-only NCGuide status probe.")
    from ..models import PlanStep

    return [PlanStep("NCG001", "before", "read NCGuide run status", "ReadRunStatus", {})]
