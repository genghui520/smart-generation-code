from __future__ import annotations

from ..knowledge import KnowledgeBase
from ..llm import LlmClient
from ..models import WorkflowStage, WorkflowState

class RouterAgent:
    def __init__(self, llm_client: LlmClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        self.llm_client = llm_client or LlmClient()
        self.knowledge_base = knowledge_base
        self.last_route_source = "llm"
        self.last_route_reason = ""
        self.last_repair_instruction = ""

    def route(self, state: WorkflowState) -> WorkflowStage:
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
        router_knowledge_context = retrieve_router_knowledge_context(self.knowledge_base, state)
        system_prompt = (
            "# Identity\n"
            "You are RouterAgent in a multi-agent CNC FOCAS traffic-generation workflow.\n\n"
            "# Instructions\n"
            "Choose exactly one next stage from the valid next stages for the current state. "
            "Use repair_plan for planning/spec/coverage problems, "
            "repair_code for generated C++/NC/code/schema problems, repair_execution for runtime, "
            "NCGuide, port, click, DLL, timeout, or toolchain problems. "
            "If UploadProgram fails with evidence of a target O-number collision, for example "
            "TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED or cnc_dwnend3/cnc_download3 returning FOCAS_RET_5 after a "
            "non-destructive upload attempt, prefer repair_plan so PlanningAgent chooses a different O number. "
            "If LOAD_DLL and CONNECT succeeded but UploadProgram, SelectProgram, ReadProgramNumber verification, "
            "or generated-artifact CSV/schema behavior failed, prefer repair_code because CodeGenerationAgent must "
            "modify the generated NC/C++ lifecycle implementation. "
            "Windows process exits 3221225725/0xC00000FD (stack overflow) and 3221225477/0xC0000005 (access violation) indicate generated C++ memory/ABI defects; route them to repair_code and never repair_execution/re-run the unchanged binary. "
            "If traffic quality failed because feed/position/run-state did not vary, prefer repair_plan "
            "so PlanningAgent can redesign the NC specification and sampling strategy. "
            "RouterAgent owns the traffic-quality decision. Inspect the raw API-log evidence directly against the "
            "PlannerAgent quality targets; quality_metrics are auxiliary summaries and may be incomplete when generated "
            "C++ uses new output field names. Quality is evaluated by whether returned output parameters vary, not by "
            "whether input parameters vary. Use api_log_quality_evidence as the primary evidence, including "
            "program_verified, program_completion_gate/local_quality_evaluation, feed samples, position samples, "
            "run/motion samples, and representative raw rows. Choose complete when the raw returned-output logs show "
            "useful FOCAS traffic variation and completed execution, even if local quality_metrics missed a field name. "
            "If result_success is false but raw api_log_quality_evidence shows sufficient returned-output variation, "
            "choose complete only when program completion is evidenced and the remaining errors are nonfatal optional warnings. "
            "Do not complete when CONNECT failed, artifacts are missing, no NC program was verified/started, "
            "program_completed is false/missing, program_completion_gate timed out, or feed/position/run-motion outputs did not vary. "
            "Use quality_metrics as hints, not as hard gates. If quality_metrics and raw API-log evidence conflict, "
            "prefer the raw API-log evidence and mention the mismatch in the route reason. "
            "When choosing a repair stage, also write a concrete repair_instruction for the next agent. "
            "For LLM timeout/empty-output failures, instruct the next agent to reduce output scope, use the provided template, "
            "return strict JSON, and avoid optional review or placeholder code. "
            "Do not choose execution before generated artifacts exist. "
            "Do not choose a repair stage when there is no current failure. Return JSON only.\n\n"
            "# Output Schema\n"
            "{\"next_stage\":\"planning\",\"reason\":\"short reason\",\"repair_instruction\":\"concrete instruction for next agent, empty if no repair\"}"
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
            f"router_knowledge_context={router_knowledge_context}\n"
            f"repair_attempts={state.repair_attempts}\n"
            f"All stages: {sorted(all_stages)}\n"
            f"Valid next stages for this state: {sorted(allowed)}"
        )
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
    if target_program_selected_requires_replan(state):
        return {"repair_plan"}
    if upload_program_number_conflict_requires_replan(state):
        return {"repair_plan"}
    if generated_cpp_crash_requires_code_repair(state):
        return {"repair_code"}
    if failed_program_lifecycle_call_requires_code_repair(state):
        return {"repair_code"}
    if result_has_sufficient_output_variation(state):
        return {"complete", "repair_plan", "repair_code", "repair_execution"}
    if state.result is not None and state.quality_assessment is not None and state.result.api_logs:
        return {"complete", "repair_plan", "repair_code", "repair_execution"}
    if state.quality_assessment is not None and not state.quality_assessment.passed:
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
        return {"code_generation"}
    if state.result is None:
        return {"execution"}
    if not state.result.success:
        return {"repair_plan", "repair_code", "repair_execution"}
    if state.quality_assessment is not None:
        return {"complete", "repair_plan", "repair_code", "repair_execution"}
    return {"complete"}


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
        identity = " ".join([log.interface_name, log.protocol_function, log.step_id]).lower()
        if any(token in identity for token in lifecycle_tokens):
            return True
    return False


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


def upload_program_number_conflict_requires_replan(state: WorkflowState) -> bool:
    if state.result is None:
        return False
    for log in state.result.api_logs:
        identity = " ".join([log.interface_name, log.protocol_function, log.step_id]).lower()
        data = str(log.response.get("data", "")).lower()
        if log.status_code == 5 and any(token in identity for token in ["cnc_dwnend3", "cnc_download3", "uploadprogram"]):
            return True
        if "target_program_exists_replan_required" in data or "target_program_selected_replan_required" in data:
            return True
    return False


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


def summarize_failed_api_logs(state: WorkflowState) -> list[dict[str, object]]:
    if state.result is None:
        return []
    rows: list[dict[str, object]] = []
    for log in state.result.api_logs:
        if log.status_code == 0 and not log.error:
            continue
        rows.append(
            {
                "step_id": log.step_id,
                "interface_name": log.interface_name,
                "protocol_function": log.protocol_function,
                "status_code": log.status_code,
                "error": log.error,
                "data": str(log.response.get("data", ""))[:240],
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
    rows = knowledge_base.search_rules(query, top_k=4) + knowledge_base.search_api(query, top_k=4)
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
