from __future__ import annotations

from ..knowledge import KnowledgeBase
from ..focas_contracts import resolve_api_return_code
from ..agent_tools.knowledge import KnowledgeSearchInput
from ..agent_tools.memory import MemorySearchInput
from ..llm import LlmClient
from ..models import ApiCallLog, WorkflowStage, WorkflowState
from ..quality import output_variation_is_sufficient, plan_requires_motion_quality
from ..api_prerequisites import classify_prerequisite_failure
from ..orchestration.roles import ActionSpec, AgentTemplate
from ..tool_runtime import ToolRuntime
from ..agent_tools.traffic_graph import ApiNameInput

class RouterAgent(AgentTemplate):
    role_name = "RouterRole"
    profile = "Workflow router"
    goal = "Select one valid next stage and provide a concrete repair instruction when needed."
    constraints = ("Only choose a stage from the workflow's valid transitions.",)

    def __init__(self, llm_client: LlmClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        super().__init__()
        self.register_actions(ActionSpec("route_stage", "WorkflowState", "WorkflowStage"))
        self.llm_client = llm_client or LlmClient()
        self.knowledge_base = knowledge_base
        self.last_route_source = "llm"
        self.last_route_reason = ""
        self.last_repair_instruction = ""
        self.tool_runtime: ToolRuntime | None = None

    def route(self, state: WorkflowState) -> WorkflowStage:
        if state.stage == "repair_code" and any(
            str(error).startswith("CodeGenerationAgent failed:")
            or "compilation failed" in str(error).lower()
            or "compile failed" in str(error).lower()
            for error in state.errors
        ) and "repair_code" in valid_next_stages(state):
            self.last_route_source = "deterministic_code_generation_failure"
            self.last_route_reason = next(
                str(error) for error in state.errors
                if str(error).startswith("CodeGenerationAgent failed:")
                or "compilation failed" in str(error).lower()
                or "compile failed" in str(error).lower()
            )
            self.last_repair_instruction = (
                "Repair the current generated C++ or its compile error while preserving the validated plan and target calls."
            )
            return "repair_code"
        repeated = repeated_target_failure(state)
        if repeated and "repair_plan" in valid_next_stages(state):
            self.last_route_source = "deterministic_repeated_target_failure"
            self.last_route_reason = repeated
            self.last_repair_instruction = (
                "Persist the failure evidence for this target, mark it blocked for the current batch, "
                "and continue planning/executing independent target APIs. Do not regenerate the unchanged session."
            )
            return "repair_plan"
        sdk_error = next((str(error) for error in state.errors if "SDK_TYPE_MISMATCH" in str(error)), "")
        if sdk_error and "repair_code" in valid_next_stages(state):
            self.last_route_source = "deterministic_sdk_type_mismatch"
            self.last_route_reason = sdk_error
            self.last_repair_instruction = (
                "Use the SDK-derived Fwlib32.h type index to replace the invalid member access "
                "with an actual field or a documented alternative FOCAS API; preserve the target API set."
            )
            return "repair_code"
        capture_error = next((str(error) for error in state.errors
                              if "CAPTURE_ARTIFACT_ERROR:" in str(error)
                              or "PCAP_CAPTURE_FAILED" in str(error)), "")
        if capture_error and "repair_execution" in valid_next_stages(state):
            self.last_route_source = "deterministic_capture_tool_failure"
            self.last_route_reason = capture_error
            self.last_repair_instruction = "Repair or reconfigure the capture tool path; preserve the generated C++ and API evidence."
            return "repair_execution"
        prerequisite = terminal_capture_prerequisite(state)
        if prerequisite:
            self.last_route_source = "deterministic_capture_prerequisite"
            self.last_route_reason = prerequisite
            self.last_repair_instruction = ""
            return "complete"
        missing_target_log = next(
            (str(error) for error in state.errors
             if "no logged invocation" in str(error).lower()
             or "TARGET_API_NOT_CALLED:" in str(error)),
            "",
        )
        if missing_target_log and "repair_code" in valid_next_stages(state):
            self.last_route_source = "deterministic_missing_target_invocation"
            self.last_route_reason = missing_target_log
            self.last_repair_instruction = (
                "Repair only the target invocation path: restore the target ctx.call and exact api.<target> call, "
                "preserve valid helper calls, then revalidate and compile."
            )
            return "repair_code"
        if planning_transient_error(state):
            self.last_route_source = "deterministic_transient_llm_error"
            self.last_route_reason = "PlanningAgent provider/network failure; terminate without repair_plan"
            self.last_repair_instruction = ""
            return "complete"
        # Flow-style terminal guard.  Once execution has succeeded and the
        # objective evidence is complete, do not let an LLM choose a repair
        # branch again.  This prevents the historical "success -> repair_code"
        # loop caused by exposing terminal and repair edges simultaneously.
        if state.result is not None and state.result.success:
            if valid_next_stages(state) == {"complete"}:
                self.last_route_source = "deterministic_quality_gate"
                self.last_route_reason = "execution_success_and_quality_gate_passed"
                self.last_repair_instruction = ""
                return "complete"
        allowed = valid_next_stages(state)
        if len(allowed) == 1:
            self.last_route_source = "deterministic_single_transition"
            self.last_route_reason = "only_valid_next_stage"
            self.last_repair_instruction = ""
            return next(iter(allowed))
        llm_stage = self.route_with_llm(state)
        self.last_route_source = "llm"
        return llm_stage

    def route_with_llm(self, state: WorkflowState) -> WorkflowStage:
        if not self.llm_client.enabled:
            raise RuntimeError("RouterAgent requires an LLM decision in agent-only mode.")
        all_stages = {
            "planning",
            "code_generation",
            "execution",
            "complete",
            "repair_plan",
            "repair_code",
            "repair_execution",
        }
        allowed = valid_next_stages(state)
        router_knowledge_context = retrieve_router_knowledge_context(
            self.knowledge_base, state, tool_runtime=self.tool_runtime, agent_name=self.role_name
        )
        router_graph_context: dict[str, object] = {}
        target_names = []
        if state.plan is not None:
            target_names = list(state.plan.rag_context.get("explicit_target_functions", []) or [])
            if not target_names:
                target_names = list(state.request.permissions.get("required_api_functions", []) or [])
        if self.tool_runtime is not None:
            for api_name in dict.fromkeys(str(item).strip() for item in target_names if str(item).strip()):
                try:
                    dependencies, _ = self.tool_runtime.call(
                        self.role_name, "get_api_dependencies", ApiNameInput(api_name)
                    )
                    router_graph_context[api_name] = dependencies
                except Exception as exc:
                    router_graph_context[api_name] = {
                        "available": False,
                        "knowledge_status": "unknown",
                        "error": str(exc),
                    }
        system_prompt = (
            "# Identity\n"
            "You are RouterAgent in a multi-agent CNC FOCAS traffic-generation workflow.\n\n"
            "# Instructions\n"
            "Choose exactly one next stage from the valid next stages for the current state. "
            "Use repair_plan for planning/spec problems demonstrated by downstream failure evidence, "
            "repair_code for generated C++/NC/code/schema problems, repair_execution for runtime, "
            "NCGuide, port, click, DLL, timeout, or toolchain problems. "
            "The fixed NCGuide driver executes one block per Cycle Start: single_block=1 is required. "
            "Never instruct clearing single block or bypassing panel gates. A UI trigger failure is not a sampling-plan defect. "
            "A successful execution result only means the run completed without fatal execution errors; "
            "it is not by itself sufficient for completion. Before choosing complete, verify high-quality "
            "traffic evidence in the raw returned-output API logs and matched PCAP transactions. "
            "Require completed program execution and position/feed variation only when the target API batch "
            "actually requires NC motion. For program transfer, directory, file, parameter, tool, PMC, and "
            "option APIs, assess their own target calls, returned fields, parameter changes, and traffic. "
            "If UploadProgram fails with evidence of a target O-number collision, for example "
            "TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED after an exact directory or detail-error check, "
            "prefer repair_plan so PlanningAgent chooses a different O number. A numeric return 5 is EW_DATA "
            "in the configured SDK and does not by itself prove a program-number collision. "
            "If LOAD_DLL and CONNECT succeeded but UploadProgram, SelectProgram, ReadProgramNumber verification, "
            "or generated-artifact CSV/schema behavior failed, prefer repair_code because CodeGenerationAgent must "
            "modify the generated NC/C++ lifecycle implementation. "
            "Windows process exits 3221225725/0xC00000FD (stack overflow) and 3221225477/0xC0000005 (access violation) indicate generated C++ memory/ABI defects; route them to repair_code and never repair_execution/re-run the unchanged binary. "
            "If traffic quality failed because feed/position/run-state did not vary, prefer repair_plan "
            "so PlanningAgent can redesign the NC specification and sampling strategy. "
            "Transaction matching, parameter coverage, and output/state variation are offline audit results. "
            "They must never trigger repair_code or make a completed compile/run fail. Preserve audit diagnostics "
            "and allow a later independent collection run to fill missing coverage. "
            "Use api_log_quality_evidence for output/state quality, including "
            "program_verified, program_completion_gate/local_quality_evaluation, feed samples, position samples, "
            "run/motion samples, and representative raw rows. Choose complete when the raw returned-output logs show "
            "useful FOCAS traffic variation and completed execution, even if local quality_metrics missed a field name. "
            "If result_success is false but raw api_log_quality_evidence shows sufficient returned-output variation, "
            "choose complete only when program completion is evidenced and the remaining errors are nonfatal optional warnings. "
            "Do not complete when CONNECT failed, required artifacts are missing, the target API was not called, "
            "or a required NC-motion operation failed its explicit runtime safety gates. Missing matched traffic "
            "with otherwise successful target calls is partial collection evidence, not a code defect. "
            "Do not impose motion-only gates on non-motion APIs. "
            "Use quality_metrics as hints, not as hard gates. If quality_metrics and raw API-log evidence conflict, "
            "prefer the raw API-log evidence and mention the mismatch in the route reason. "
            "When choosing a repair stage, also write a concrete repair_instruction for the next agent. "
            "For failed FOCAS calls, use recent_failed_api_logs' API-specific return-code explanation. "
            "A generic code name is not proof of a unique root cause; request a detail-error read or "
            "controller state evidence when the API-specific explanation is insufficient. "
            "For LLM timeout/empty-output failures, instruct the next agent to reduce output scope, use the provided template, "
            "return strict JSON, and avoid optional review or placeholder code. "
            "Do not choose execution before generated artifacts exist. "
            "A newly generated plan proceeds directly to code_generation without a separate approval/replanning loop. "
            "plan_review is diagnostic context, not a pre-execution gate. Only choose repair_plan after an actual "
            "generation, compilation, execution, or result-validation failure provides evidence that the plan's "
            "dependencies, operation sequence, or sampling strategy caused that failure. Explain that causal evidence "
            "in the repair_instruction. Use repair_code for implementation/type/schema mistakes; do not replan "
            "merely for interface naming, parameter-coverage polish, or hypothetical missing steps. "
            "Do not choose a repair stage when there is no current failure. Return JSON only.\n\n"
            "# Output Schema\n"
            "{\"next_stage\":\"planning\",\"reason\":\"short reason\",\"repair_instruction\":\"concrete instruction for next agent, empty if no repair\"}"
        )
        soul_context = ""
        soul_store = getattr(self, "soul_store", None)
        if soul_store is not None:
            try:
                soul_context = soul_store.prompt_context(
                    self.role_name,
                    task_context=f"{state.request.description}; stage={state.stage}",
                    error_type="; ".join(str(item) for item in state.errors[-4:]),
                    memory_store=getattr(self, "memory_store", None),
                )
            except (FileNotFoundError, ValueError) as exc:
                soul_context = f"Agent Soul unavailable: {exc}"
        if soul_context:
            system_prompt += (
                "\n\n# Dynamic Agent Soul\n"
                + soul_context
                + "\nTreat proposed or unknown traffic relationships as hypotheses; route only from concrete evidence.\n"
            )
        user_prompt = (
            f"Current state:\n"
            f"stage={state.stage}\n"
            f"has_plan={state.plan is not None}\n"
            f"has_artifacts={state.artifacts is not None}\n"
            f"has_result={state.result is not None}\n"
            f"errors={state.errors[-5:]}\n"
            f"quality_assessment={state.quality_assessment}\n"
            f"quality_metrics={summarize_quality_for_router(state)}\n"
            f"api_log_quality_evidence={summarize_api_log_quality_evidence(state)}\n"
            f"result_success={state.result.success if state.result else None}\n"
            f"result_errors={state.result.errors if state.result else []}\n"
            f"recent_failed_api_logs={summarize_failed_api_logs(state)}\n"
            f"retrieved_error_experience_memories={state.long_term_memories}\n"
            f"plan_review={summarize_plan_for_router(state)}\n"
            f"router_knowledge_context={router_knowledge_context}\n"
            f"router_agent_soul={soul_context}\n"
            f"traffic_relationships={state.plan.rag_context.get('traffic_relationships', {}) if state.plan else {}}\n"
            f"router_graph_context={router_graph_context}\n"
            f"repair_attempts={state.repair_attempts}\n"
            f"All stages: {sorted(all_stages)}\n"
            f"Valid next stages for this state: {sorted(allowed)}"
        )
        fast_code_only = bool(
            state.request.permissions.get("code_only_evaluation")
            and state.request.permissions.get("representative_scenario")
        )
        if self.tool_runtime is not None and not fast_code_only:
            system_prompt += (
                "\nIf the current evidence is insufficient, you may call search_knowledge or search_memory "
                "through the ToolCallLoop before returning the final route JSON."
            )
            payload = self.tool_loop(max_turns=3).run(
                self.role_name,
                system_prompt,
                user_prompt,
                input_factory=self._tool_input_factory,
            )
        else:
            payload = self.llm_client.invoke_json(system_prompt, user_prompt)
        stage = self.validated_stage_from_payload(payload, allowed)
        if stage is not None:
            return stage
        corrected_payload = self.request_corrected_route(system_prompt, user_prompt, payload, allowed)
        stage = self.validated_stage_from_payload(corrected_payload, allowed)
        if stage is not None:
            return stage
        raise ValueError(
            "RouterAgent LLM returned a route outside the valid next stages: "
            f"initial={payload.get('next_stage')!r}, corrected={corrected_payload.get('next_stage')!r}, "
            f"valid={sorted(allowed)}"
        )

    @staticmethod
    def _tool_input_factory(tool_name: str, arguments: dict[str, object]) -> object:
        if tool_name == "search_knowledge":
            return KnowledgeSearchInput(
                query=str(arguments.get("query", "")),
                top_k=int(arguments.get("top_k", 6)),
                filters=dict(arguments.get("filters", {}) or {}),
            )
        if tool_name == "search_memory":
            return MemorySearchInput(
                query=str(arguments.get("query", "")),
                top_k=int(arguments.get("top_k", 5)),
            )
        raise ValueError(f"RouterAgent tool loop does not support {tool_name}")

    def validated_stage_from_payload(
        self,
        payload: dict,
        allowed: set[WorkflowStage],
    ) -> WorkflowStage | None:
        next_stage = str(payload.get("next_stage", "")).strip()
        if next_stage in allowed:
            self.last_route_reason = str(payload.get("reason", "")).strip()
            self.last_repair_instruction = str(payload.get("repair_instruction", "")).strip()
            return next_stage  # type: ignore[return-value]
        return None

    def request_corrected_route(
        self,
        system_prompt: str,
        user_prompt: str,
        payload: dict,
        allowed: set[WorkflowStage],
    ) -> dict:
        correction_prompt = (
            f"{user_prompt}\n\n"
            f"Previous RouterAgent decision was invalid: {payload}\n"
            f"Choose exactly one next_stage from this valid set only: {sorted(allowed)}\n"
            "Return JSON only."
        )
        return self.llm_client.invoke_json(system_prompt, correction_prompt)


def valid_next_stages(state: WorkflowState) -> set[WorkflowStage]:
    if any(
        "PlanningAgent Markdown is missing required fields:" in str(error)
        for error in state.errors
    ):
        # Schema omissions are deterministic Planner-output failures. Repeating
        # the same planning request without new evidence only burns retries.
        return {"complete"}
    if any(marker in str(error) for error in state.errors
           for marker in ("CODEGEN_REPAIR_EXHAUSTED:", "EXECUTION_CONTRACT_ERROR:")):
        # The generator already used its bounded diagnostic repairs. Restarting
        # it from scratch repeats the same contract without new evidence.
        return {"complete"}
    if terminal_capture_prerequisite(state):
        return {"complete"}
    if planning_transient_error(state):
        return {"complete"}
    if target_program_selected_requires_replan(state):
        return {"repair_plan"}
    if generated_cpp_crash_requires_code_repair(state):
        return {"repair_code"}
    if failed_program_lifecycle_call_requires_code_repair(state):
        return {"repair_code"}
    if state.result is not None and state.result.success:
        # Quality and coverage audits are persisted independently after capture.
        return {"complete"}
    # A quality-looking log cannot override a failed transport/process
    # result.  Flow-style completion is monotonic: only a successful execution
    # may enter the terminal edge; failed executions must be classified and
    # retried (or reported failed after the repair budget is exhausted).
    if state.result is not None and state.quality_assessment is not None and state.result.api_logs:
        return {"repair_plan", "repair_code", "repair_execution"}
    if state.errors:
        if state.plan is None:
            return {"repair_plan"}
        if state.artifacts is None:
            return {"repair_plan", "repair_code"}
        if state.result is None:
            return {"repair_plan", "repair_code", "repair_execution"}
        if not state.result.success:
            return {"repair_plan", "repair_code", "repair_execution"}

    if state.plan is None:
        return {"planning"}
    if state.artifacts is None:
        # Attempt implementation first. Plan-review hints must not create an
        # approval loop before any downstream failure has been observed.
        # The error/result branches above retain evidence-based replanning.
        return {"code_generation"}
    if state.result is None:
        return {"execution"}
    if not state.result.success:
        return {"repair_plan", "repair_code", "repair_execution"}
    if state.quality_assessment is not None:
        return {"complete"}
    return {"complete"}


def planning_transient_error(state: WorkflowState) -> bool:
    """Planning provider failures are terminal for this run, never re-planned."""
    return any(str(error).startswith("transient_llm_error:") for error in state.errors)


def terminal_capture_prerequisite(state: WorkflowState) -> str:
    """Code regeneration cannot install controller options or change its mode."""
    if state.result is None or state.result.success or generated_cpp_crash_requires_code_repair(state):
        return ""
    prerequisite_failure = next(
        (
            classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or "")
            for log in state.result.api_logs
            if log.protocol_function.startswith(("cnc_", "pmc_")) and log.status_code != 0
            and classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or "")
        ),
        None,
    )
    if prerequisite_failure and state.plan is not None:
        context = state.plan.rag_context.setdefault("api_prerequisite_runtime", {})
        if state.prerequisite_analysis_attempted:
            return (
                "API prerequisite analysis already attempted without a changed controller state; "
                "stop repeated retries and preserve the evidence: "
                + str(prerequisite_failure)
            )
        state.prerequisite_analysis_attempted = True
        context["analysis_attempted"] = True
        context["failure"] = prerequisite_failure
        context["failed_api"] = next(
            log.protocol_function for log in state.result.api_logs
            if log.protocol_function.startswith(("cnc_", "pmc_"))
            and log.status_code != 0
            and classify_prerequisite_failure(log.protocol_function, log.status_code, log.error or "")
        )
    for log in state.result.api_logs:
        if log.protocol_function == "host_runtime_exception" and log.status_code != 0:
            detail = str(log.response.get("data", log.error or ""))
            if any(marker in detail for marker in (
                "CYCLE_START_CURSOR_FAILED", "CYCLE_START_CLICK_FAILED", "CYCLE_START_TARGET_OBSCURED",
                "NCGUIDE_WINDOW_NOT_FOREGROUND", "NCGUIDE_WINDOW_NOT_FOUND",
                "CYCLE_START_CONFIGURATION_MISSING", "CYCLE_START_OUTSIDE_WINDOW",
                "NCGUIDE_CLIENT_COORDINATE_FAILED", "NCGUIDE_WINDOW_COORDINATE_FAILED", "NCGUIDE_CLICK_MODE_INVALID",
            )):
                return "NCGuide UI prerequisite unavailable; stop unchanged retries and preserve single-block mode: " + detail
        if log.protocol_function == "host_api_prerequisite" and log.status_code != 0:
            if not _has_analyzable_api_evidence(state):
                return "API prerequisite unavailable: " + str(log.response.get("data", log.error or ""))
        if log.protocol_function.startswith(("cnc_", "pmc_")) and log.status_code in {6, 12}:
            if not _has_analyzable_api_evidence(state):
                return f"{log.protocol_function} returned {log.status_code}; required controller option/mode must be provided before retry"
    return ""


def _has_analyzable_api_evidence(state: WorkflowState) -> bool:
    """Whether the Router LLM has enough wire evidence to diagnose the failure."""
    if state.result is None:
        return False
    return any(
        log.traffic_status == "matched"
        or log.traffic_status == "request_only_segment"
        or log.response.get("traffic_analysis")
        for log in state.result.api_logs
    )


def generated_cpp_crash_requires_code_repair(state: WorkflowState) -> bool:
    errors = list(state.errors)
    if state.result is not None:
        errors.extend(state.result.errors)
    text = " ".join(errors).lower()
    return any(
        marker in text
        for marker in [
            "3221225725",
            "0xc00000fd",
            "3221225477",
            "0xc0000005",
            "stack overflow",
            "access violation",
        ]
    )


def failed_program_lifecycle_call_requires_code_repair(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    lifecycle_tokens = [
        "uploadprogram",
        "selectprogram",
        "deleteprogram",
        "programlifecyclegate",
        "cnc_dwn",
        "cnc_download",
        "cnc_search",
        "cnc_delete",
        "cnc_rdprgnum",
    ]
    for log in state.result.api_logs:
        if log.status_code == 0 and not log.error:
            continue
        if is_nonfatal_optional_api_warning(log):
            continue
        identity = " ".join([log.interface_name, log.protocol_function, log.step_id]).lower()
        if any(token in identity for token in lifecycle_tokens):
            return True
    return False


def is_nonfatal_optional_api_warning(log: ApiCallLog) -> bool:
    text = " ".join(
        [
            str(log.error or ""),
            str(log.response.get("return_text", "")),
            str(log.response.get("data", "")),
        ]
    ).lower()
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


def target_program_selected_requires_replan(state: WorkflowState) -> bool:
    errors = list(state.errors)
    if state.result is not None:
        errors.extend(state.result.errors)
        for log in state.result.api_logs:
            errors.append(str(log.response.get("data", "")))
    text = " ".join(errors).lower()
    return any(
        marker in text
        for marker in [
            "target_program_selected_replan_required",
            "target_program_exists_replan_required",
        ]
    )


def result_has_sufficient_output_variation(state: WorkflowState) -> bool:
    if state.result is None or state.quality_assessment is None:
        return False
    if "PROGRAM_NOT_VERIFIED" in state.errors or "PROGRAM_NOT_VERIFIED" in state.result.errors:
        return False
    metrics = state.quality_assessment.metrics
    return (
        bool(metrics.get("program_completed"))
        and int(metrics.get("changed_output_parameter_count") or 0) >= 3
        and int(metrics.get("feed_sample_count") or 0) >= 5
        and int(metrics.get("feed_unique_count") or 0) > 1
        and int(metrics.get("position_sample_count") or 0) >= 5
        and int(metrics.get("position_unique_count") or 0) > 1
        and (
            int(metrics.get("run_active_count") or 0) > 0
            or int(metrics.get("motion_active_count") or 0) > 0
        )
    )


def requires_dynamic_quality_gate(state: WorkflowState) -> bool:
    """Return whether completion needs a returned-output quality gate."""
    if not state.request.quality_gate_enabled:
        return False
    if state.plan is None:
        return False
    return plan_requires_motion_quality(state.plan)


def summarize_failed_api_logs(state: WorkflowState) -> list[dict[str, object]]:
    if state.result is None:
        return []
    rows: list[dict[str, object]] = []
    for log in state.result.api_logs:
        if log.status_code == 0 and not log.error:
            continue
        function = log.protocol_function.strip()
        explanation = (
            resolve_api_return_code(function, log.status_code)
            if function.startswith(("cnc_", "pmc_")) and "/" not in function
            else {}
        )
        rows.append(
            {
                "step_id": log.step_id,
                "interface_name": log.interface_name,
                "protocol_function": log.protocol_function,
                "status_code": log.status_code,
                "error": log.error,
                "data": str(log.response.get("data", ""))[:240],
                "return_code_names": explanation.get("api_code_names") or explanation.get("global_code_names") or [],
                "api_return_code_explanations": [
                    str(item)[:320] for item in explanation.get("api_explanations", [])[:2]
                ],
                "return_code_documented_for_api": explanation.get("documented_for_api", False),
            }
        )
    return rows[-8:]


def summarize_quality_for_router(state: WorkflowState) -> dict[str, object]:
    if state.quality_assessment is None:
        return {}
    metrics = state.quality_assessment.metrics
    keys = [
        "feed_sample_count",
        "feed_unique_count",
        "feed_values_preview",
        "position_sample_count",
        "position_unique_count",
        "position_values_preview",
        "run_active_count",
        "motion_active_count",
        "program_completion_gate_count",
        "program_completed",
        "input_parameter_names",
        "return_parameter_names",
        "output_parameter_variation",
        "changed_output_parameter_count",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def summarize_plan_for_router(state: WorkflowState) -> dict[str, object]:
    """Return a compact, reviewable Planner artifact for RouterAgent."""
    if state.plan is None:
        return {}
    plan = state.plan
    prototype_by_function: dict[str, str] = {}
    if plan.code_spec is not None:
        for contract in getattr(plan.code_spec, "api_contracts", []) or []:
            name = str(getattr(contract, "tool_name", "")).strip()
            if name:
                prototype_by_function[name] = str(getattr(contract, "prototype", "")).strip()

    step_rows: list[dict[str, object]] = []
    empty_parameter_strategies: list[str] = []
    protocol_functions: list[str] = []
    for step in plan.steps[:40]:
        parameters = dict(step.parameters)
        parameter_generation = parameters.get("parameter_generation", [])
        substantive_parameters = {
            key: value for key, value in parameters.items() if key != "parameter_generation"
        }
        if substantive_parameters and not parameter_generation:
            empty_parameter_strategies.append(step.step_id)
        if step.protocol_function:
            # Lifecycle actions can be represented as an atomic sequence such
            # as cnc_dwnstart3/cnc_download3/cnc_dwnend3.  Prototype evidence
            # is stored per exported FOCAS symbol, so review each member rather
            # than treating the whole sequence as one nonexistent function.
            protocol_functions.extend(
                name.strip()
                for name in step.protocol_function.split("/")
                if name.strip()
            )
        step_rows.append(
            {
                "step_id": step.step_id,
                "phase": step.phase,
                "interface_name": step.interface_name,
                "protocol_function": step.protocol_function,
                "parameters": substantive_parameters,
                "parameter_generation": parameter_generation,
                "repeat": step.repeat,
                "interval_seconds": step.interval_seconds,
                "action": step.action[:240],
            }
        )

    missing_official_prototypes = sorted(
        {
            name
            for name in protocol_functions
            if name.startswith(("cnc_", "pmc_")) and not prototype_by_function.get(name)
        }
    )
    from ..execution_contracts import lifecycle_interfaces
    interfaces = lifecycle_interfaces(plan.steps)
    manifest = plan.rag_context.get("function_coverage_manifest")
    if isinstance(manifest, dict) and manifest.get("enabled"):
        motion_plan = any(
            segment.get("cycle_start_required") and segment.get("target_batch")
            for segment in manifest.get("segments", [])
            if isinstance(segment, dict)
        )
    else:
        motion_plan = (
            str(plan.scenario_type).lower() == "coordinate_motion"
            or any(token in " ".join(plan.nc_program_spec.block_goals).lower() for token in ["g00", "g01", "g02", "g03", "motion", "运动"])
        )
    lifecycle_required = (
        state.request.target_environment == "ncguide-generated-cpp"
        and motion_plan
        and not bool(state.request.permissions.get("planner_only_test"))
    )
    required_lifecycle = {"UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram"}
    missing_lifecycle = sorted(required_lifecycle - interfaces) if lifecycle_required else []
    blocking_plan_issues: list[str] = []
    if missing_official_prototypes:
        blocking_plan_issues.append(
            "FOCAS functions without official configured-header prototypes: "
            + ", ".join(missing_official_prototypes)
        )
    if missing_lifecycle:
        blocking_plan_issues.append(
            "NC motion plan lacks required executable lifecycle interfaces: "
            + ", ".join(missing_lifecycle)
        )
    nonblocking_plan_warnings: list[str] = []
    if empty_parameter_strategies:
        nonblocking_plan_warnings.append(
            "Input parameter coverage strategies are incomplete for steps: "
            + ", ".join(empty_parameter_strategies)
        )
    return {
        "scenario_type": plan.scenario_type,
        "scenario_goal": plan.scenario_goal,
        "nc_program_spec": {
            "program_name": plan.nc_program_spec.program_name,
            "purpose": plan.nc_program_spec.purpose,
            "block_goals": plan.nc_program_spec.block_goals,
            "constraints": plan.nc_program_spec.constraints,
        },
        "steps": step_rows,
        "missing_official_prototypes": missing_official_prototypes,
        "steps_with_input_parameters_but_no_parameter_strategy": empty_parameter_strategies,
        "missing_required_program_lifecycle": missing_lifecycle,
        "blocking_plan_issues": blocking_plan_issues,
        "nonblocking_plan_warnings": nonblocking_plan_warnings,
        "planning_quality_analysis": plan.rag_context.get("planning_quality_analysis", {}),
        "quality_targets": plan.rag_context.get("quality_targets", {}),
    }


def summarize_api_log_quality_evidence(state: WorkflowState) -> dict[str, object]:
    if state.result is None:
        return {}
    evidence: dict[str, object] = {
        "program_verified": False,
        "program_completed": False,
        "local_quality_pass": False,
        "feed_rows": [],
        "position_rows": [],
        "run_motion_rows": [],
        "nonfatal_warning_rows": [],
    }
    feed_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    run_motion_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for log in state.result.api_logs:
        data = str(log.response.get("data", ""))
        haystack = " ".join([log.step_id, log.interface_name, log.protocol_function, data]).lower()
        if "program_verified=true" in haystack:
            evidence["program_verified"] = True
        if "program_completion_gate" in haystack and any(token in haystack for token in ["completed=true", "complete=true"]):
            evidence["program_completed"] = True
        if "local_quality_evaluation" in haystack and any(token in haystack for token in ["quality_pass", "quality_pass=true"]):
            evidence["local_quality_pass"] = True
        row = {
            "step_id": log.step_id,
            "api": log.protocol_function or log.interface_name,
            "status_code": log.status_code,
            "data": data[:260],
        }
        if any(token in haystack for token in ["feed=", "actual_feed"]):
            feed_rows.append(row)
        if any(token in haystack for token in ["position_axis", "absolute_axis", "abs_axis", "axis1=", "axis2=", "axis3="]):
            position_rows.append(row)
        if "run=" in haystack or "motion=" in haystack:
            run_motion_rows.append(row)
        if log.status_code != 0:
            warning_rows.append(row)
    evidence["feed_rows"] = representative_rows(feed_rows)
    evidence["position_rows"] = representative_rows(position_rows)
    evidence["run_motion_rows"] = representative_rows(run_motion_rows)
    evidence["nonfatal_warning_rows"] = warning_rows[:6]
    return evidence


def representative_rows(rows: list[dict[str, object]], limit: int = 8) -> list[dict[str, object]]:
    if len(rows) <= limit:
        return rows
    head_count = max(1, limit // 2)
    tail_count = limit - head_count
    return rows[:head_count] + rows[-tail_count:]


def retrieve_router_knowledge_context(
    knowledge_base: KnowledgeBase | None,
    state: WorkflowState,
    *,
    tool_runtime: ToolRuntime | None = None,
    agent_name: str = "RouterRole",
) -> list[dict[str, object]]:
    if knowledge_base is None:
        return []
    failed_logs = summarize_failed_api_logs(state)
    query = "\n".join(
        [
            state.request.description,
            f"errors={state.errors[-8:]}",
            f"result_errors={state.result.errors[-8:] if state.result else []}",
            f"failed_api_logs={failed_logs}",
            f"quality_assessment={state.quality_assessment}",
        ]
    )
    if tool_runtime is None:
        rows = knowledge_base.search_rules(query, top_k=4) + knowledge_base.search_api(query, top_k=4)
    else:
        rules, _ = tool_runtime.call(
            agent_name, "search_knowledge", KnowledgeSearchInput(query, 4, {"source_type": "rule"})
        )
        apis, _ = tool_runtime.call(
            agent_name, "search_knowledge", KnowledgeSearchInput(query, 4, {"source_type": "api"})
        )
        rows = rules.results + apis.results
    return summarize_retrieved_for_router(rows)


def summarize_retrieved_for_router(rows) -> list[dict[str, object]]:
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


def repeated_target_failure(state: WorkflowState) -> str | None:
    """Detect an unchanged API-specific failure across repair attempts."""
    from collections import Counter
    import re
    signatures = []
    for error in state.errors:
        text = str(error)
        if "BLOCKING:" not in text and "CODEGEN_REPAIR_EXHAUSTED" not in text:
            continue
        match = re.search(r"(?:target API|required target API)\s+([a-zA-Z0-9_]+).*", text)
        if match:
            signatures.append(f"{match.group(1)}|{text.split(';')[0].strip()}")
    counts = Counter(signatures)
    repeated = next((key for key, count in counts.items() if count >= 2), None)
    return f"Repeated unchanged target failure: {repeated}" if repeated else None
