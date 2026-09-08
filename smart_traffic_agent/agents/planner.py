from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..knowledge import KnowledgeBase
from ..llm import LlmClient
from ..models import ExecutionPlan, NcProgramSpec, PlanStep, RetrievedChunk, WorkflowState
from ..tools import focas_function_for, interface_to_focas_mapping
from ..utils import tokenize
from ..function_coverage import DEFAULT_FUNCTION_COVERAGE_STATE, load_terminal_covered_functions
from ..orchestration.roles import ActionSpec, AgentTemplate, message_to_dict
from ..agent_roles import PlanSpec
from ..code_spec import build_code_spec, build_official_abi_context
from ..rag.focas_function_manifest import (
    DEFAULT_COVERAGE_BATCH_SIZE,
    DEFAULT_COVERAGE_SEGMENT_COUNT,
    DEFAULT_FOCAS_FUNCTION_MANIFEST,
    build_function_coverage_context,
)
from ..rag.scenario_templates import DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH
from .prompts import PLANNING_REVIEW_JSON_SCHEMA, PLANNING_REVIEW_SYSTEM_PROMPT


RULE_TYPES = ["nc_rule", "operation_rule", "collection_rule", "safety_rule"]

SCENARIO_ALIASES = {
    "spindle_start_stop": "spindle_speed_change",
    "spindle_state": "spindle_speed_change",
    "parameter_read_write": "parameter_write_simulated",
}

INTERFACE_TO_FOCAS = interface_to_focas_mapping()


class PlanningAgent(AgentTemplate):
    role_name = "PlanningRole"
    profile = "RAG planning agent"
    goal = "Retrieve authoritative knowledge and produce an executable structured plan."
    constraints = ("Use RAG evidence and official ABI facts.", "Keep API calls atomic.")

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_client: LlmClient | None = None,
        scenario_templates_path: Path = DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH,
        function_manifest_path: Path = DEFAULT_FOCAS_FUNCTION_MANIFEST,
        coverage_state_path: Path = DEFAULT_FUNCTION_COVERAGE_STATE,
        coverage_batch_size: int = DEFAULT_COVERAGE_BATCH_SIZE,
        coverage_segment_count: int = DEFAULT_COVERAGE_SEGMENT_COUNT,
    ) -> None:
        super().__init__()
        self.register_actions(
            ActionSpec("retrieve_focas_knowledge", "TaskRequest", "KnowledgeSpec"),
            ActionSpec("write_execution_plan", "KnowledgeSpec", "PlanSpec"),
        )
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client or LlmClient()
        self.scenario_templates = load_scenario_templates(scenario_templates_path)
        self.function_manifest_path = function_manifest_path
        self.coverage_state_path = coverage_state_path
        self.coverage_batch_size = coverage_batch_size
        self.coverage_segment_count = coverage_segment_count

    def run(self, state: WorkflowState) -> WorkflowState:
        request = state.request
        scenario = normalize_scenario(infer_scenario(request.description))
        selected_templates = select_scenario_templates(
            request.description,
            scenario,
            self.scenario_templates,
        )
        retrieval = retrieve_planning_context(
            self.knowledge_base,
            request.description,
            scenario,
        )
        plan = build_plan(
            request.task_id,
            scenario,
            request.target_environment,
            retrieval,
            selected_templates=selected_templates,
        )
        if state.long_term_memories:
            plan.rag_context["long_term_memories"] = summarize_long_term_memories(state.long_term_memories)
            plan.llm_notes.append(f"Loaded {len(state.long_term_memories)} long-term repair memories.")
        repair_context = summarize_repair_context(state)
        if repair_context:
            plan.rag_context["repair_context"] = repair_context
        plan.rag_context["task_permissions"] = dict(request.permissions)
        plan.rag_context["coverage_intent"] = infer_coverage_intent(request.description, request.target_environment)
        if (
            plan.rag_context["coverage_intent"].get("full_function_coverage")
            and not request.permissions.get("representative_scenario", False)
        ):
            terminal_covered_functions = load_terminal_covered_functions(self.coverage_state_path)
            coverage_context = build_function_coverage_context(
                self.function_manifest_path,
                covered_functions=terminal_covered_functions,
                batch_size=self.coverage_batch_size,
                scenario_type=scenario,
                max_segments=self.coverage_segment_count,
            )
            plan.rag_context["function_coverage_manifest"] = coverage_context
            plan.rag_context["function_coverage_state"] = {
                "state_path": str(self.coverage_state_path),
                "terminal_covered_function_count": len(terminal_covered_functions),
            }
            if coverage_context.get("enabled"):
                plan.llm_notes.append(
                    "Loaded FOCAS function manifest multi-segment batch "
                    f"segments={coverage_context.get('segment_count')} targets={coverage_context.get('target_batch_size')} "
                    f"support={coverage_context.get('support_batch_size')} / "
                    f"{coverage_context.get('target_function_count')} functions."
                )
        plan.rag_context["api_candidate_pool"] = build_api_candidate_pool(plan)
        candidate_functions = [
            str(row.get("protocol_function", ""))
            for row in plan.rag_context["api_candidate_pool"]
            if isinstance(row, dict) and str(row.get("protocol_function", "")).strip()
        ]
        plan.rag_context["official_abi_context"] = build_official_abi_context(candidate_functions)
        if self.environment is not None:
            plan.rag_context["environment_shared_rules"] = self.environment.shared_rules
            plan.rag_context["official_abi_context"] = self.environment.official_abi(candidate_functions)
        if not self.llm_client.enabled:
            raise RuntimeError("PlanningAgent requires an LLM planning decision in agent-only mode.")
        apply_llm_planning_decision(
            plan,
            self.llm_client,
            request.description,
            representative_scenario=request.permissions.get("representative_scenario", False),
        )
        if request.permissions.get("representative_scenario", False):
            plan.llm_notes.append(
                "Representative benchmark skipped the optional planning review request to reduce generation latency."
            )
        else:
            enrich_plan_with_llm(
                plan,
                self.llm_client,
                request.description,
            )
        state.retrieved_chunks = flatten_retrieval(retrieval)
        state.plan = plan
        plan.code_spec = build_code_spec(plan, state.retrieved_chunks)
        handoff = self.publish(
            "write_execution_plan", "PlanSpec",
            plan_id=plan.plan_id, step_count=len(plan.steps), rag_chunk_count=len(state.retrieved_chunks),
            code_spec=asdict(plan.code_spec),
        )
        state.messages.append(message_to_dict(
            handoff,
            PlanSpec(plan.plan_id, len(plan.steps), len(state.retrieved_chunks)),
        ))
        state.stage = "code_generation"
        return state


def load_scenario_templates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    templates = payload.get("templates", [])
    return [template for template in templates if isinstance(template, dict)]


def select_scenario_templates(
    task_description: str,
    scenario: str,
    templates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not templates:
        return []
    if scenario == "comprehensive_focas_traffic":
        return sorted(templates, key=template_priority)

    target_scene = scenario_to_template_scene(scenario, task_description)
    matched = [
        template
        for template in templates
        if template.get("scenario_name") == target_scene or template.get("scenario_id") == scenario
    ]
    if matched:
        return sorted(matched, key=template_priority)

    scored = [(template_match_score(task_description, scenario, template), template) for template in templates]
    scored = [(score, template) for score, template in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [template for _, template in scored[:2]]


def scenario_to_template_scene(scenario: str, task_description: str) -> str:
    joined = task_description.lower()
    mapping = {
        "coordinate_motion": "coordinate_feed_motion",
        "feed_speed_change": "coordinate_feed_motion",
        "spindle_speed_change": "spindle_control",
        "parameter_read": "parameter_macro_access",
        "parameter_write_simulated": "parameter_macro_access",
        "alarm_query": "alarm_diagnosis",
        "diagnostic_query": "alarm_diagnosis",
        "pmc_signal_read": "pmc_signal_monitoring",
        "program_lifecycle": "program_lifecycle",
        "tool_offset_setting": "tool_offset_management",
        "macro_variable_read_write": "parameter_macro_access",
        "general_status_collection": "general_status_collection",
    }
    if "work coordinate" in joined or "g54" in joined:
        return "work_coordinate_setting"
    return mapping.get(scenario, scenario)


def template_priority(template: dict[str, Any]) -> tuple[int, int]:
    priority = {"high": 0, "medium": 1, "normal": 2}
    return (
        priority.get(str(template.get("coverage_priority", "normal")), 3),
        -int(template.get("cluster_member_count", 0)),
    )


def template_match_score(task_description: str, scenario: str, template: dict[str, Any]) -> int:
    query_terms = set(tokenize(task_description + " " + scenario))
    text = " ".join(
        [
            str(template.get("scenario_id", "")),
            str(template.get("scenario_name", "")),
            str(template.get("goal", "")),
            " ".join(template.get("main_objects", [])),
            " ".join(template.get("main_apis", [])),
            " ".join(template.get("main_nc_or_operation_features", [])),
            " ".join(template.get("expected_signals", [])),
        ]
    )
    template_terms = set(tokenize(text))
    return len(query_terms.intersection(template_terms))


def retrieve_planning_context(
    knowledge_base: KnowledgeBase,
    query: str,
    scenario: str,
) -> dict[str, list[RetrievedChunk]]:
    retrieval_query = f"{query}\nscenario: {scenario}"
    context: dict[str, list[RetrievedChunk]] = {}
    context["scenario_organization"] = knowledge_base.search_scenario_organization(
        retrieval_query,
        scenario,
        top_k=2,
    )
    for rule_type in RULE_TYPES:
        context[rule_type] = knowledge_base.search_rules_by_type(
            retrieval_query,
            rule_type,
            top_k=3,
        )
    context["api"] = knowledge_base.search_api(retrieval_query, top_k=20)
    if not any(context.values()):
        context["general"] = knowledge_base.search(retrieval_query, top_k=6)
    return context


def flatten_retrieval(context: dict[str, list[RetrievedChunk]]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    rows: list[RetrievedChunk] = []
    for items in context.values():
        for item in items:
            key = item.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    return rows


def summarize_retrieval(context: dict[str, list[RetrievedChunk]]) -> dict:
    summary: dict[str, object] = {}
    for key, items in context.items():
        summary[key] = [
            {
                "chunk_id": item.chunk.chunk_id,
                "score": item.score,
                "source_type": item.chunk.metadata.get("source_type"),
                "rule_type": item.chunk.metadata.get("rule_type"),
                "scenario": item.chunk.metadata.get("scenario"),
                "function": item.chunk.metadata.get("function"),
                "source_file": item.chunk.metadata.get("source_file"),
                "page_start": item.chunk.metadata.get("page_start"),
                "text_preview": item.chunk.text[:240],
            }
            for item in items
        ]
    return summary


def apply_llm_planning_decision(
    plan: ExecutionPlan,
    llm_client: LlmClient,
    task_description: str,
    *,
    representative_scenario: bool = False,
) -> None:
    system_prompt = (
        "# Identity\n"
        "You are PlanningAgent in a multi-agent FANUC FOCAS traffic-generation system.\n"
        "You generate the execution plan specification, not concrete C++ code and not concrete G-code.\n\n"
        "# Instructions\n"
        "- Decide the NC program specification, the set of related FOCAS APIs to cover, each API's input parameter strategy, the execution order, timing, and safety constraints.\n"
        "- First analyze the NC program scenario described by the task, then infer which FOCAS APIs are semantically related by reading the retrieved knowledge-base facts: function names, descriptions, prototypes, arguments, return data, examples, and error notes.\n"
        "- Do not rely on a local tool/interface registry as the API universe. The knowledge base is the source of API facts; you may plan any relevant FOCAS protocol_function found or inferable from retrieved API knowledge.\n"
        "- Build an API coverage set from api_candidate_pool plus any additional relevant API facts in RAG: include APIs that can produce relevant returned outputs, state transitions, lifecycle evidence, control/synchronization evidence, or intentionally varied simulator-side mutation evidence for this NC program scenario. Exclude APIs only with a brief reason.\n"
        "- When coverage_intent.full_traffic_coverage=true, optimize for broad/full traffic coverage rather than a minimal safe API subset. For a simulator/NCGuide target, do not exclude a RAG-retrieved write or control API merely because it mutates simulator state, creates lifecycle side effects, is not the shortest path, or is outside the narrowest scene core. Include it when it can generate distinguishable traffic and can be bracketed by setup/read-back/cleanup or bounded simulator values.\n"
        "- When coverage_intent.full_function_coverage=true and function_coverage_manifest.enabled=true, treat function_coverage_manifest as the global FOCAS API universe from focas_funcs.xlsx. The current run may contain multiple scenario segments, not one isolated scenario and not one isolated function list. For each segment in function_coverage_manifest.segments, target_batch functions are the coverage objectives for that segment; support_batch functions are allowed to repeat because they create, verify, synchronize, or restore that segment's state. Include executable steps for each target_batch function when ABI/prototype and bounded arguments can be established; otherwise record a concrete defer/exclude reason such as missing prototype/export, unsupported controller/simulator, or unsafe/unbounded argument construction. Use support_batch APIs where needed, but do not count support-only repetition as progress on target coverage.\n"
        "- A single generated C++ script may execute multiple scenario segments sequentially. Segments that require NC programs may use distinct bounded O-number payloads; segments that do not require NC motion should be implemented as direct FOCAS probe/read/write-restore call blocks. Keep segment boundaries explicit in step_id, action, CSV data, and diagnostics.\n"
        "- For full-coverage simulator runs, treat semantically related APIs as complementary unless RAG evidence shows an exact incompatibility. For example, position families such as cnc_rdposition, cnc_absolute, cnc_machine, cnc_relative, and cnc_distance may all contribute different returned fields; lifecycle/program APIs such as cnc_rdmdiprgstat, cnc_wractpt, cnc_pdf_wractpt, cnc_setpglock, cnc_search, and cnc_rdprgnum may contribute different program-state traffic. Do not exclude one solely because another API already gives partial evidence.\n"
        "- Valid exclusion reasons in full-coverage simulator mode are narrow: absent official prototype/header evidence, DLL/export unavailable, unsupported by the simulator/controller per RAG, argument construction cannot be made bounded, API conflicts with a required lifecycle gate, or the user/permissions explicitly forbid it. 'write operation', 'unnecessary mutation', 'not MDI-focused', or 'covered by another API' are not sufficient by themselves.\n"
        "- For each selected API, create at least one executable step with protocol_function set to the exact FOCAS function name or function sequence. If no existing semantic interface name fits, create a concise interface_name such as ReadDistanceToGo or ReadAxisLoad while keeping protocol_function exact.\n"
        "- Keep each selected protocol_function consistent with its retrieved prototype and arguments. Do not label a step cnc_rdprogdir while using cnc_rdprogdir2/cnc_rdprogdir3-style top_prog/num_prog pointer parameters; choose the exact documented family member whose prototype matches the planned parameter strategy.\n"
        "- First analyze what will make the generated traffic high quality, distinguishable, and broad enough to cover the selected related API set.\n"
        "- For NC motion used to create observable traffic, do not make the program excessively slow. Prefer multiple motion blocks whose main observable portion lasts about 2-5 seconds each under simulation, with enough sampling frequency to observe returned-output changes. If prior output was too fast to observe, reduce feed or increase distance moderately; if runtime was excessive, increase feed or shorten travel.\n"
        "- For single-block StartProgram/Cycle Start steps, explicitly require a trigger-readiness gate: before each Cycle Start, "
        "the generated C++ must poll relevant completion evidence from the selected API set and wait until the previous block is no longer active or until a bounded "
        "timeout is reached. Do not plan blind consecutive Cycle Start clicks.\n"
        "- In the confirmed NCGuide Single Block behavior, every non-empty executable NC line consumes one Cycle Start, including the O program-number line, setup/modal-only lines, motion lines, and the final M30 line. Plan enough guarded Cycle Start operations to advance the entire uploaded NC payload, not merely its motion blocks.\n"
        "- Require the generated C++ to derive or embed the effective NC segment count from the exact generated payload, log expected_nc_segment_count and cycle_start_click_count, and execute through the M30 segment before the final completion gate. Non-motion segments are lifecycle/warmup evidence and must not be counted as motion samples.\n"
        "- Upload/select/program-number verification is a hard lifecycle gate. If cnc_dwnend3 or cnc_search fails, or cnc_rdprgnum does not match the uploaded O number, execution must stop before any Cycle Start and report PROGRAM_NOT_VERIFIED.\n"
        "- Before UploadProgram, require a documented program-directory/read API for exact target-program existence checks. PlannerAgent may provide a base nc_program_spec.program_name, but runtime collision handling belongs to CodeGenerationAgent/C++ after it reads the actual controller directory. Do not use repair planning merely to guess another O number.\n"
        "- Single target-program deletion is allowed when the generated C++ has positively identified an exact conflicting O number and deletes only that O number before upload. Delete-all remains permission-gated: consider cnc_delall only when task_permissions.allow_delete_all_programs=true and only if RAG-supported reasoning shows it is appropriate for the isolated NCGuide experiment.\n"
        "- If prior execution quality failed, read the previous failed plan/artifacts/result and modify that design; do not ignore it and start from scratch.\n"
        "- Do not output concrete NC blocks such as G01 X...; output block_goals and constraints instead.\n"
        "- For every executable API step, plan input parameters with parameter_generation. "
        "Classify each input parameter as fixed, enum, or range: fixed uses one constant value from the task/API context; "
        "enum lists all safe finite values to traverse; range gives min, max, and representative samples such as min/mid/max. "
        "The parameter strategy is for input coverage only; traffic quality is later evaluated from returned output parameter variation.\n"
        "- Return the complete planned API steps. The local draft is context only.\n"
        "- Return JSON only.\n\n"
        "# Output Schema\n"
        "{"
        "\"scenario_goal\":\"short goal\","
        "\"quality_analysis\":{\"traffic_goal\":\"what must vary\",\"risk\":\"why first run may fail\","
        "\"strategy\":[\"moderate-duration motion blocks / sufficient sampling frequency / avoid excessive runtime\"],"
        "\"repair_hypothesis\":\"what to change if output shows no variation\"},"
        "\"api_coverage_analysis\":[{\"protocol_function\":\"cnc_example\",\"relationship\":\"why this API relates to the NC scenario\","
        "\"coverage_role\":\"what traffic/evidence it contributes\",\"decision\":\"include\",\"parameter_strategy_summary\":\"fixed/enum/range plan\"}],"
        "\"quality_targets\":{\"min_feed_samples\":5,\"expect_feed_variation\":true,"
        "\"expect_position_variation\":true,\"preferred_motion_duration_seconds_per_block\":\"about 2-5 seconds\",\"preferred_feed\":\"moderate low feed, not extremely slow\"},"
        "\"coverage_segments\":[{\"segment_id\":\"01_programmed_coordinate_motion\",\"main_state_driver\":\"driver\","
        "\"nc_program_required\":true,\"target_functions\":[\"cnc_absolute\"],\"support_functions\":[\"cnc_statinfo\"],"
        "\"quality_target\":\"what this segment proves\"}],"
        "\"nc_program_spec\":{\"program_name\":\"O1234\",\"purpose\":\"short purpose\","
        "\"block_goals\":[\"goal\"],\"constraints\":[\"constraint\"],\"generation_notes\":[\"note\"]},"
        "\"steps\":[{\"step_id\":\"S001\",\"phase\":\"before\",\"action\":\"read status\","
        "\"interface_name\":\"ReadRunStatus\",\"protocol_function\":\"cnc_statinfo\",\"parameters\":{},"
        "\"parameter_generation\":[{\"name\":\"type\",\"kind\":\"enum\",\"values\":[-1,0,1],\"reason\":\"cover documented finite modes\"},"
        "{\"name\":\"axis_count\",\"kind\":\"range\",\"min\":1,\"max\":8,\"samples\":[1,4,8],\"reason\":\"cover min/mid/max\"},"
        "{\"name\":\"host\",\"kind\":\"fixed\",\"value\":\"127.0.0.1\",\"reason\":\"NCGuide simulator endpoint\"}],"
        "\"repeat\":1,"
        "\"interval_seconds\":0.2,\"expected_state\":\"short expected state\"}],"
        "\"notes\":[\"short note\"]"
        "}"
    )
    user_prompt = (
        f"Task:\n{task_description}\n\n"
        f"Current scenario={plan.scenario_type}\n"
        f"Current goal={plan.scenario_goal}\n"
        f"Current NC spec={plan.nc_program_spec}\n"
        f"Explicit task permissions={plan.rag_context.get('task_permissions', {})}\n"
        f"Current steps={[{'id': s.step_id, 'phase': s.phase, 'interface': s.interface_name, 'repeat': s.repeat, 'interval': s.interval_seconds, 'action': s.action} for s in plan.steps]}\n"
        f"RAG context={compact_rag_context(plan.rag_context)}\n"
        f"Official ABI context for candidate APIs:\n{plan.rag_context.get('official_abi_context', '')}\n"
        f"Prior repair history and quality failures={plan.rag_context.get('repair_context', [])}\n"
    )
    code_only = bool(plan.rag_context.get("task_permissions", {}).get("code_only_evaluation"))
    if code_only:
        # RQ1 uses a compact MetaGPT-style requirements/design artifact. The
        # API set still comes from task requirements and RAG evidence, not a
        # local scenario registry.
        system_prompt = (
            "You are the PlanningAgent for a code-only FANUC FOCAS benchmark. "
            "Return JSON only. Produce a complete CodeSpec for one C++ source file. "
            "Use task requirements and RAG evidence to select every required atomic FOCAS API; "
            "never invent symbols or substitute a locally maintained API list. "
            "Each step must contain exactly one protocol_function, documented parameters, "
            "phase, and purpose. Include connection and cleanup when required by the task. "
            "Do not include PCAP, CSV, traffic capture, simulator UI, or runtime logging. "
            "Required JSON fields: scenario_goal, nc_program_spec, steps, notes."
        )
        user_prompt = (
            f"Task:\n{task_description}\n\n"
            f"Scenario:\n{plan.scenario_type}\n\n"
            f"RAG evidence:\n{compact_rag_context(plan.rag_context)}\n\n"
            f"Official ABI context for candidate APIs:\n{plan.rag_context.get('official_abi_context', '')}\n\n"
            f"Initial plan context:\n{plan.steps}\n{plan.nc_program_spec}\n\n"
            "Return a complete code-generation specification. Include all API calls explicitly required by the task, "
            "and preserve their exact official names and argument semantics."
        )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    required_planning_keys = (
        ("scenario_goal", "nc_program_spec", "steps")
        if code_only
        else ("scenario_goal", "quality_analysis", "quality_targets", "nc_program_spec", "steps")
    )
    missing_planning_keys = [key for key in required_planning_keys if not payload.get(key)]
    nc_spec_payload = payload.get("nc_program_spec")
    if isinstance(nc_spec_payload, dict):
        for nested_key in ("program_name", "purpose", "block_goals", "constraints"):
            if not nc_spec_payload.get(nested_key):
                missing_planning_keys.append(f"nc_program_spec.{nested_key}")
    if missing_planning_keys:
        # Repair the structured artifact once, keeping the task/RAG response as
        # context. Do not synthesize missing scenario or API facts locally.
        repair_system_prompt = (
            "You are repairing a PlanningAgent JSON artifact. Return JSON only. "
            "Preserve all valid content from the previous response and add only "
            "the missing required fields. Use the task and RAG context as evidence; "
            "do not invent APIs or replace exact protocol names. Required fields: "
            "scenario_goal, quality_analysis, quality_targets, nc_program_spec, steps."
        )
        repair_user_prompt = (
            f"Task:\n{task_description}\n\n"
            f"RAG context:\n{compact_rag_context(plan.rag_context)}\n\n"
            f"Previous planning artifact:\n{payload}\n\n"
            f"Missing or invalid required fields: {missing_planning_keys}\n"
            "Return one complete planning JSON object."
        )
        repaired_payload = llm_client.invoke_json(repair_system_prompt, repair_user_prompt)
        if isinstance(repaired_payload, dict):
            original_spec = payload.get("nc_program_spec")
            repaired_spec = repaired_payload.get("nc_program_spec")
            payload = {**payload, **repaired_payload}
            if isinstance(original_spec, dict) and isinstance(repaired_spec, dict):
                payload["nc_program_spec"] = {**original_spec, **repaired_spec}

    scenario_goal = str(payload.get("scenario_goal", "")).strip()
    if not scenario_goal:
        raise ValueError("PlanningAgent LLM response must include scenario_goal.")
    plan.scenario_goal = scenario_goal

    spec_payload = payload.get("nc_program_spec", {})
    if not isinstance(spec_payload, dict):
        raise ValueError("PlanningAgent LLM response must include nc_program_spec.")
    apply_llm_nc_spec(plan, spec_payload, code_only=code_only)

    quality_analysis = payload.get("quality_analysis", {})
    if isinstance(quality_analysis, dict) and quality_analysis:
        plan.rag_context["planning_quality_analysis"] = quality_analysis
        for item in quality_analysis_to_notes(quality_analysis):
            plan.llm_notes.append(f"quality_analysis: {item}")
    elif not code_only:
        raise ValueError("PlanningAgent LLM response must include quality_analysis.")

    quality_targets = payload.get("quality_targets", {})
    if isinstance(quality_targets, dict) and quality_targets:
        plan.rag_context["quality_targets"] = quality_targets
    elif not code_only:
        raise ValueError("PlanningAgent LLM response must include quality_targets.")

    api_coverage = payload.get("api_coverage_analysis", payload.get("api_coverage", []))
    if isinstance(api_coverage, list):
        plan.rag_context["api_coverage_analysis"] = normalize_api_coverage_analysis(api_coverage)
    else:
        plan.rag_context["api_coverage_analysis"] = []

    coverage_segments = payload.get("coverage_segments", [])
    if isinstance(coverage_segments, list):
        plan.rag_context["coverage_segments"] = normalize_coverage_segments(coverage_segments)
    if not plan.rag_context.get("coverage_segments") and "function_coverage_manifest" in plan.rag_context:
        plan.rag_context["coverage_segments"] = coverage_segments_from_manifest_context(
            plan.rag_context["function_coverage_manifest"]
        )

    step_payload = payload.get("steps", [])
    if not isinstance(step_payload, list):
        raise ValueError("PlanningAgent LLM response must include steps.")
    plan.steps = plan_steps_from_llm_rows(step_payload)
    if not plan.steps:
        if code_only:
            repaired_steps = llm_client.invoke_json(
                "You repair invalid PlanningAgent steps. Return JSON only with a non-empty steps list. "
                "Each step must have step_id, phase (before/during/after), action, interface_name, "
                "and exactly one atomic protocol_function. Use only task and RAG evidence; do not invent APIs.",
                f"Task: {task_description}\nRAG: {compact_rag_context(plan.rag_context)}\n"
                f"Invalid steps: {step_payload}\nReturn corrected atomic steps.",
            )
            if isinstance(repaired_steps, dict):
                candidate_steps = repaired_steps.get("steps", [])
                if isinstance(candidate_steps, list):
                    plan.steps = plan_steps_from_llm_rows(candidate_steps)
        if not plan.steps:
            raise ValueError("PlanningAgent LLM response did not include any valid steps.")
    if code_only:
        # Preserve the generic NC lifecycle after the LLM plan is parsed. The
        # lifecycle helper expands UploadProgram into atomic FOCAS calls so
        # CodeSpec can carry the actual cnc_download3 payload contract.
        ensure_nc_program_lifecycle_steps(
            plan.scenario_type,
            plan.steps,
            plan.nc_program_spec.program_name,
        )
        attach_protocol_functions(plan.steps)
        plan.steps = audit_and_repair_task_api_coverage(
            plan,
            task_description,
            llm_client,
        )
    if representative_scenario:
        plan.steps = compact_representative_steps(plan.steps, max_steps=18)
        plan.llm_notes.append(
            "Representative benchmark retained RAG-selected interfaces without TOOL_REGISTRY admission filtering."
        )
    attach_protocol_functions(plan.steps)

    notes = payload.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if isinstance(notes, list):
        for note in notes:
            text = str(note).strip()
        if text:
            plan.llm_notes.append(f"llm_planning_generation: {text}")


def audit_and_repair_task_api_coverage(
    plan: ExecutionPlan,
    task_description: str,
    llm_client: LlmClient,
) -> list[PlanStep]:
    """Keep explicit task operations from disappearing during plan compression."""
    # Code-only C++ contracts contain controller APIs only. NC payload design
    # remains in nc_program_spec; host/UI pseudo-actions must not become API
    # calls that CodeEngineer tries to implement.
    plan.steps = [step for step in plan.steps if is_code_only_protocol_step(step)]
    current = [
        {
            "step_id": step.step_id,
            "phase": step.phase,
            "action": step.action,
            "interface_name": step.interface_name,
            "protocol_function": step.protocol_function,
            "parameters": step.parameters,
        }
        for step in plan.steps
    ]
    prompt = (
        "You are auditing a PlannerAgent plan for task/API coverage. Return JSON only with keys "
        "required_protocol_functions, missing_steps, exclusions. required_protocol_functions must "
        "list every exact API needed by an operation explicitly stated in the task, even when it is "
        "already present in current steps. Use only APIs supported by the task "
        "and the supplied RAG evidence; never use a local registry and never invent symbols. "
        "A function explicitly required by the task must appear in required_protocol_functions. "
        "For every missing function, provide one atomic missing_steps item with an exact single "
        "protocol_function, interface_name, phase, action, and documented parameters. "
        "Do not add PCAP, CSV, UI automation, or runtime logging for code-only evaluation."
    )
    user_prompt = (
        f"Task:\n{task_description}\n\n"
        f"RAG API evidence and candidates:\n{compact_rag_context(plan.rag_context)}\n\n"
        f"Current parsed steps:\n{current}\n\n"
        "Audit explicit task operations and return any omitted atomic API steps."
    )
    try:
        payload = llm_client.invoke_json(prompt, user_prompt)
    except Exception as exc:
        plan.llm_notes.append(f"task API coverage audit failed: {exc}")
        return plan.steps
    if not isinstance(payload, dict):
        return plan.steps
    missing_rows = payload.get("missing_steps", [])
    missing = plan_steps_from_llm_rows(missing_rows) if isinstance(missing_rows, list) else []
    missing = [step for step in missing if is_code_only_protocol_step(step)]
    existing_names = {
        name
        for step in plan.steps
        for name in protocol_function_names_for_planner(step.protocol_function)
    }
    candidate_names = {
        str(row.get("protocol_function", "")).strip()
        for row in plan.rag_context.get("api_candidate_pool", [])
        if isinstance(row, dict) and str(row.get("protocol_function", "")).strip()
    }
    raw_required_names = payload.get("required_protocol_functions", [])
    if isinstance(raw_required_names, str):
        raw_required_names = [raw_required_names]
    required_names = {
        str(value).strip()
        for value in raw_required_names
        if str(value).strip() and str(value).strip().startswith(("cnc_", "pmc_"))
    } if isinstance(raw_required_names, list) else set()
    evidence_names = {
        str(row.get("function", "")).strip()
        for row in plan.rag_context.get("api", [])
        if isinstance(row, dict) and str(row.get("function", "")).strip()
    }
    abi_context = str(plan.rag_context.get("official_abi_context", ""))
    abi_supported_names = set(re.findall(r"\b(cnc_[A-Za-z0-9_]+|pmc_[A-Za-z0-9_]+)\s*\(", abi_context))
    missing = [
        step for step in missing
        if all(
            name in abi_supported_names and (name in candidate_names or name in evidence_names)
            for name in protocol_function_names_for_planner(step.protocol_function)
        )
    ]
    # The audit may identify a required API without returning a complete row.
    # Materialize that decision as an atomic step only when the API is backed by
    # the current RAG candidate/evidence set; parameters remain for CodeSpec ABI
    # resolution and are never invented here.
    for name in sorted(required_names - existing_names):
        if name not in abi_supported_names or (name not in candidate_names and name not in evidence_names):
            continue
        missing.append(
            PlanStep(
                step_id=f"AUDIT-{len(missing) + 1:03d}",
                phase="during",
                action=f"Execute required documented API {name}.",
                interface_name=f"FOCAS_{name}",
                protocol_function=name,
                operation_kind="focas_api",
                api_calls=[{"call_id": f"audit-call-{len(missing) + 1:02d}", "protocol_function": name, "parameters": {}}],
            )
        )
    # Deterministic evidence fallback: recover candidates whose RAG description
    # shares multiple meaningful terms with the task when the audit model omits
    # the required_protocol_functions field. This is evidence-driven, not an
    # API registry or scenario-specific function list.
    evidence_rows = list(plan.rag_context.get("api", []))
    candidate_rows = list(plan.rag_context.get("api_candidate_pool", []))
    task_terms = {
        term for term in re.findall(r"[a-z][a-z0-9_]{3,}", task_description.lower())
        if term not in {"generate", "complete", "client", "must", "include", "using", "documented", "return", "with"}
    }
    evidence_by_name: dict[str, str] = {}
    for row in evidence_rows:
        if isinstance(row, dict):
            name = str(row.get("function", "")).strip()
            if name:
                evidence_by_name[name] = " ".join(str(row.get(key, "")) for key in ("preview", "text_preview"))
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("protocol_function", "")).strip()
        if not name.startswith(("cnc_", "pmc_")) or name not in abi_supported_names:
            continue
        if not name or name in existing_names or name in {step.protocol_function for step in missing}:
            continue
        description = " ".join([
            " ".join(str(value) for value in row.get("relationships", [])),
            evidence_by_name.get(name, ""),
        ]).lower()
        matched_terms = {term for term in task_terms if term in description}
        if len(matched_terms) < 1:
            continue
        missing.append(
            PlanStep(
                step_id=f"EVIDENCE-{len(missing) + 1:03d}",
                phase="during",
                action=f"Execute RAG-supported API {name} for the requested operation.",
                interface_name=f"FOCAS_{name}",
                protocol_function=name,
                operation_kind="focas_api",
                api_calls=[{"call_id": f"evidence-call-{len(missing) + 1:02d}", "protocol_function": name, "parameters": {}}],
            )
        )
    added = 0
    for step in missing:
        names = protocol_function_names_for_planner(step.protocol_function)
        if not names or any(name in existing_names for name in names):
            continue
        plan.steps.append(step)
        existing_names.update(names)
        added += 1
    if added:
        plan.llm_notes.append(f"Task/API coverage audit restored {added} omitted atomic step(s).")
    exclusions = payload.get("exclusions", [])
    if isinstance(exclusions, list):
        for exclusion in exclusions[:12]:
            text = str(exclusion).strip()
            if text:
                plan.llm_notes.append(f"task API coverage exclusion: {text}")
    return plan.steps


def is_code_only_protocol_step(step: PlanStep) -> bool:
    """Accept only atomic controller symbols in the RQ1 C++ contract."""
    names = protocol_function_names_for_planner(step.protocol_function)
    return bool(names) and all(name.startswith(("cnc_", "pmc_")) for name in names)


def apply_llm_nc_spec(plan: ExecutionPlan, payload: dict[str, Any], *, code_only: bool = False) -> None:
    program_name = str(payload.get("program_name", "")).strip()
    if not valid_program_name(program_name):
        raise ValueError(f"PlanningAgent LLM returned invalid NC program name: {program_name!r}")
    block_goals = string_list(payload.get("block_goals"))
    if not block_goals and not code_only:
        raise ValueError("PlanningAgent LLM response must include nc_program_spec.block_goals.")
    constraints = string_list(payload.get("constraints"))
    if not constraints and not code_only:
        raise ValueError("PlanningAgent LLM response must include nc_program_spec.constraints.")
    generation_notes = string_list(payload.get("generation_notes"))
    purpose = str(payload.get("purpose", "")).strip()
    if not purpose and not code_only:
        raise ValueError("PlanningAgent LLM response must include nc_program_spec.purpose.")
    plan.nc_program_spec = NcProgramSpec(
        program_name=program_name,
        purpose=purpose,
        block_goals=block_goals,
        constraints=constraints,
        generation_notes=generation_notes,
    )
    for step in plan.steps:
        if "program_name" in step.parameters:
            step.parameters["program_name"] = program_name


def plan_steps_from_llm_rows(rows: list[Any]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        interface_name = str(row.get("interface_name", "")).strip()
        if not interface_name:
            continue
        parameters = row.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        parameters = dict(parameters)
        parameter_generation = row.get("parameter_generation", row.get("parameter_strategy"))
        if isinstance(parameter_generation, list):
            parameters["parameter_generation"] = normalize_parameter_generation(parameter_generation)
        phase = phase_or_default(row.get("phase"))
        protocol_function = str(row.get("protocol_function", "")).strip()
        raw_calls = row.get("api_calls", [])
        api_calls = normalize_api_calls(raw_calls, protocol_function, parameters)
        if api_calls and not protocol_function:
            protocol_function = str(api_calls[0]["protocol_function"])
        steps.append(
            PlanStep(
                step_id=str(row.get("step_id", f"LLM-{index:03d}")).strip() or f"LLM-{index:03d}",
                phase=phase,  # type: ignore[arg-type]
                action=str(row.get("action", interface_name)).strip() or interface_name,
                interface_name=interface_name,
                parameters={str(key): value for key, value in parameters.items()},
                repeat=max(1, min(safe_int(row.get("repeat"), 1), 50)),
                interval_seconds=max(0.0, min(safe_float(row.get("interval_seconds"), 0.0), 10.0)),
                expected_state=str(row.get("expected_state", "")).strip(),
                protocol_function=protocol_function,
                operation_kind=str(row.get("operation_kind", "focas_api")).strip() or "focas_api",
                api_calls=api_calls,
            )
        )
    return steps


def normalize_api_calls(raw_calls: Any, protocol_function: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Planner output into one exact FOCAS function per call object."""
    calls: list[dict[str, Any]] = []
    if isinstance(raw_calls, list):
        for index, item in enumerate(raw_calls, 1):
            if not isinstance(item, dict):
                continue
            function = str(item.get("protocol_function", item.get("function", ""))).strip()
            if not function:
                continue
            call_parameters = item.get("parameters", {})
            if not isinstance(call_parameters, dict):
                call_parameters = {}
            calls.append({
                "call_id": str(item.get("call_id", f"call-{index:02d}")),
                "protocol_function": function,
                "parameters": dict(call_parameters),
            })
    if calls:
        return calls
    functions = [part.strip() for part in re.split(r"\s*(?:/|\\+|;|->)\s*", protocol_function) if part.strip()]
    return [
        {"call_id": f"call-{index:02d}", "protocol_function": function, "parameters": dict(parameters)}
        for index, function in enumerate(functions, 1)
    ]


def compact_representative_steps(steps: list[PlanStep], *, max_steps: int) -> list[PlanStep]:
    """Keep a bounded, executable subset for code-generation benchmarks."""
    selected: list[PlanStep] = []
    seen: set[tuple[str, str]] = set()
    for step in steps:
        key = (step.interface_name, step.phase)
        if key in seen:
            continue
        seen.add(key)
        selected.append(step)
        if len(selected) >= max_steps:
            break
    return selected


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_parameter_generation(rows: list[Any]) -> list[dict[str, Any]]:
    strategies: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        kind = str(row.get("kind", "")).strip().lower()
        if not name or kind not in {"fixed", "enum", "range"}:
            continue
        item: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "reason": str(row.get("reason", "")).strip(),
        }
        if kind == "fixed":
            item["value"] = row.get("value")
        elif kind == "enum":
            values = row.get("values", [])
            item["values"] = values if isinstance(values, list) else [values]
        elif kind == "range":
            item["min"] = row.get("min")
            item["max"] = row.get("max")
            samples = row.get("samples", [])
            item["samples"] = samples if isinstance(samples, list) else [samples]
        strategies.append(item)
    return strategies


def normalize_api_coverage_analysis(rows: list[Any]) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        protocol_function = str(row.get("protocol_function", row.get("function", ""))).strip()
        if not protocol_function:
            continue
        coverage.append(
            {
                "protocol_function": protocol_function,
                "relationship": str(row.get("relationship", row.get("reason", ""))).strip(),
                "coverage_role": str(row.get("coverage_role", "")).strip(),
                "decision": str(row.get("decision", "include")).strip() or "include",
                "parameter_strategy_summary": str(row.get("parameter_strategy_summary", "")).strip(),
            }
        )
    return coverage


def normalize_coverage_segments(rows: list[Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        segment_id = str(row.get("segment_id", f"segment_{index:02d}")).strip() or f"segment_{index:02d}"
        segments.append(
            {
                "segment_id": segment_id,
                "main_state_driver": str(row.get("main_state_driver", "")).strip(),
                "nc_program_required": bool(row.get("nc_program_required", False)),
                "target_functions": string_list(row.get("target_functions")),
                "support_functions": string_list(row.get("support_functions")),
                "quality_target": str(row.get("quality_target", "")).strip(),
            }
        )
    return segments


def coverage_segments_from_manifest_context(context: Any) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    rows: list[dict[str, Any]] = []
    for segment in context.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        rows.append(
            {
                "segment_id": segment.get("segment_id", ""),
                "main_state_driver": segment.get("main_state_driver", ""),
                "nc_program_required": bool(segment.get("nc_program_required", False)),
                "target_functions": [
                    row.get("function")
                    for row in segment.get("target_batch", [])
                    if isinstance(row, dict) and row.get("function")
                ],
                "support_functions": [
                    row.get("function")
                    for row in segment.get("support_batch", [])
                    if isinstance(row, dict) and row.get("function")
                ],
                "quality_target": segment.get("quality_target", ""),
            }
        )
    return rows


def valid_program_name(value: str) -> bool:
    return len(value) == 5 and value.startswith("O") and value[1:].isdigit()


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def quality_analysis_to_notes(value: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for key in ["traffic_goal", "risk", "repair_hypothesis"]:
        text = str(value.get(key, "")).strip()
        if text:
            notes.append(f"{key}: {text}")
    strategy = value.get("strategy", [])
    if isinstance(strategy, str):
        strategy = [strategy]
    if isinstance(strategy, list):
        notes.extend(f"strategy: {str(item).strip()}" for item in strategy if str(item).strip())
    return notes


def enrich_plan_with_llm(plan: ExecutionPlan, llm_client: LlmClient, task_description: str) -> None:
    system_prompt = PLANNING_REVIEW_SYSTEM_PROMPT
    user_prompt = (
        "Task:\n"
        f"{task_description}\n\n"
        "Plan summary:\n"
        f"scenario={plan.scenario_type}\n"
        f"goal={plan.scenario_goal}\n"
        f"nc_requirements={plan.nc_program_requirements}\n"
        f"planned_nc_program_spec={{'program_name': '{plan.nc_program_spec.program_name}', 'block_goals': {plan.nc_program_spec.block_goals}, 'constraints': {plan.nc_program_spec.constraints}, 'purpose': '{plan.nc_program_spec.purpose}'}}\n"
        f"steps={[{'id': s.step_id, 'phase': s.phase, 'action': s.action, 'interface': s.interface_name, 'focas': s.protocol_function} for s in plan.steps]}\n\n"
        "RAG context keys and first previews:\n"
        f"{compact_rag_context(plan.rag_context)}\n\n"
        "Return JSON with this shape:\n"
        f"{PLANNING_REVIEW_JSON_SCHEMA}"
    )
    try:
        payload = llm_client.invoke_json(system_prompt, user_prompt)
    except Exception as exc:
        plan.llm_notes.append(f"LLM planning review failed: {exc}")
        return

    if payload.get("plan_ok") is False:
        plan.llm_notes.append("LLM marked the plan as needing review.")
    for field in ["notes", "recommended_api_functions", "safety_constraints"]:
        values = payload.get(field, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            for value in values:
                text = str(value).strip()
                if text:
                    plan.llm_notes.append(f"{field}: {text}")


def compact_rag_context(rag_context: dict) -> dict:
    compact = {}
    for key, rows in rag_context.items():
        if key == "function_coverage_manifest" and isinstance(rows, dict):
            compact[key] = {
                "enabled": rows.get("enabled"),
                "source_manifest": rows.get("source_manifest"),
                "target_function_count": rows.get("target_function_count"),
                "covered_function_count": rows.get("covered_function_count"),
                "remaining_function_count": rows.get("remaining_function_count"),
                "selected_batch_size": rows.get("selected_batch_size"),
                "target_batch_size": rows.get("target_batch_size"),
                "support_batch_size": rows.get("support_batch_size"),
                "scenario_batch_id": rows.get("scenario_batch_id"),
                "main_state_driver": rows.get("main_state_driver"),
                "quality_target": rows.get("quality_target"),
                "selection_strategy": rows.get("selection_strategy"),
                "segment_count": rows.get("segment_count"),
                "function_family_counts": rows.get("function_family_counts"),
                "remaining_family_counts": rows.get("remaining_family_counts"),
                "segments": [
                    {
                        "segment_id": segment.get("segment_id"),
                        "scenario_batch_id": segment.get("scenario_batch_id"),
                        "main_state_driver": segment.get("main_state_driver"),
                        "quality_target": segment.get("quality_target"),
                        "nc_program_required": segment.get("nc_program_required"),
                        "target_functions": [
                            row.get("function")
                            for row in segment.get("target_batch", [])[:40]
                            if isinstance(row, dict)
                        ],
                        "support_functions": [
                            row.get("function")
                            for row in segment.get("support_batch", [])[:24]
                            if isinstance(row, dict)
                        ],
                    }
                    for segment in rows.get("segments", [])[:6]
                    if isinstance(segment, dict)
                ],
                "selected_batch": [
                    {
                        "function": row.get("function"),
                        "raw_function": row.get("raw_function"),
                        "coverage_role": row.get("coverage_role"),
                        "category": row.get("category"),
                        "description": str(row.get("description", ""))[:160],
                        "symbol_candidates": row.get("symbol_candidates", [])[:3],
                        "selection_reason": row.get("selection_reason"),
                    }
                    for row in rows.get("selected_batch", [])[:60]
                    if isinstance(row, dict)
                ],
            }
            continue
        if isinstance(rows, dict):
            compact[key] = rows
            continue
        if not isinstance(rows, list):
            compact[key] = str(rows)[:240]
            continue
        row_limit = 12 if key == "api" else 2
        compact[key] = [
            {
                "rule_type": row.get("rule_type"),
                "scenario": row.get("scenario"),
                "function": row.get("function"),
                "preview": str(row.get("text_preview", ""))[:260],
            }
            for row in rows[:row_limit]
        ]
    return compact


def infer_coverage_intent(task_description: str, target_environment: str) -> dict[str, Any]:
    text = task_description.lower()
    full_markers = [
        "全量",
        "全部",
        "都需要",
        "都要",
        "囊括",
        "覆盖所有",
        "高覆盖",
        "全面",
        "完整",
        "full",
        "all",
        "complete",
        "comprehensive",
        "high coverage",
        "maximum coverage",
        "full coverage",
        "cover all",
        "全覆盖",
        "全量",
        "全部",
        "所有",
        "每个",
    ]
    full_function_markers = [
        "full function coverage",
        "cover every function",
        "cover all functions",
        "all functions",
        "function manifest",
        "focas_funcs",
        "函数全覆盖",
        "全覆盖",
        "全量函数",
        "全部函数",
        "所有函数",
        "每个函数",
        "里面的函数",
    ]
    simulator_markers = ["simulator", "simulation", "ncguide", "仿真", "仿真器"]
    target_text = target_environment.lower()
    simulator_target = any(marker in target_text for marker in simulator_markers)
    explicit_simulator_context = simulator_target or any(marker in text for marker in simulator_markers)
    explicit_full_coverage = any(marker in text for marker in full_markers)
    explicit_full_function_coverage = any(marker in text for marker in full_function_markers)
    full_coverage = True
    full_function_coverage = True
    return {
        "full_traffic_coverage": full_coverage,
        "full_function_coverage": full_function_coverage,
        "coverage_default": "focas_system_default_full_high_quality_coverage",
        "explicit_full_coverage_request": explicit_full_coverage,
        "explicit_full_function_coverage_request": explicit_full_function_coverage,
        "target_environment": target_environment,
        "simulator_target": explicit_simulator_context,
        "planner_policy": (
            "FOCAS system default is full, high-quality function coverage. PlannerAgent should analyze and include "
            "manifest-batch functions plus semantically related RAG-retrieved APIs that can generate distinguishable traffic. "
            "Simulator write/control side effects are acceptable when bounded and logged; exclude only with concrete "
            "incompatibility, unsupported ABI/export/controller evidence, explicit user prohibition, or a separate hard "
            "permission gate."
            if (full_coverage or full_function_coverage) and explicit_simulator_context
            else "FOCAS system default is full, high-quality function coverage. Plan from the global function manifest plus RAG evidence and justify include/exclude/defer decisions."
        ),
    }


def build_api_candidate_pool(plan: ExecutionPlan) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}

    def add(function_name: Any, source: str, relationship: str = "") -> None:
        function = str(function_name or "").strip()
        if not function:
            return
        entry = candidates.setdefault(
            function,
            {
                "protocol_function": function,
                "sources": [],
                "relationships": [],
            },
        )
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
        if relationship and relationship not in entry["relationships"]:
            entry["relationships"].append(relationship)

    for step in plan.steps:
        for function_name in protocol_function_names_for_planner(step.protocol_function):
            add(function_name, "draft_plan_step", f"{step.interface_name}: {step.action}")

    for row in plan.rag_context.get("api", []):
        if not isinstance(row, dict):
            continue
        add(row.get("function"), "retrieved_api", str(row.get("preview", ""))[:180])

    for template in plan.rag_context.get("selected_scenario_templates", []):
        if not isinstance(template, dict):
            continue
        template_id = str(template.get("template_id") or template.get("scenario_id") or "scenario_template")
        for function_name in template.get("main_apis", []) or []:
            add(function_name, f"selected_template:{template_id}", str(template.get("scenario_name", "")))

    manifest_context = plan.rag_context.get("function_coverage_manifest", {})
    if isinstance(manifest_context, dict):
        for row in manifest_context.get("selected_batch", []) or []:
            if not isinstance(row, dict):
                continue
            add(
                row.get("function"),
                "function_manifest_target_batch"
                if row.get("coverage_role", "target") == "target"
                else "function_manifest_support_batch",
                f"{row.get('coverage_role', 'target')} | {row.get('category', '')}: {row.get('description', '')}",
            )

    return sorted(
        candidates.values(),
        key=lambda item: (0 if "draft_plan_step" in item["sources"] else 1, item["protocol_function"]),
    )


def protocol_function_names_for_planner(value: str) -> list[str]:
    names: list[str] = []
    for part in str(value or "").replace("+", "/").split("/"):
        name = part.strip()
        if name:
            names.append(name)
    return names


def summarize_long_term_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for memory in memories[:5]:
        rows.append(
            {
                "memory_id": memory.get("memory_id"),
                "score": memory.get("score"),
                "task_description": str(memory.get("task_description", ""))[:160],
                "scenario_type": memory.get("scenario_type"),
                "target_environment": memory.get("target_environment"),
                "final_success": memory.get("final_success"),
                "repair_attempts": memory.get("repair_attempts"),
                "notes": [str(note)[:200] for note in memory.get("notes", [])[:3]],
            }
        )
    return rows


def summarize_repair_context(state: WorkflowState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if state.quality_assessment is not None:
        rows.append(
            {
                "source": "quality_assessment",
                "passed": state.quality_assessment.passed,
                "issues": state.quality_assessment.issues,
                "metrics": state.quality_assessment.metrics,
                "recommendations": state.quality_assessment.recommendations,
            }
        )
    for item in state.repair_history[-3:]:
        rows.append(
            {
                "source": "repair_history",
                "repair_stage": item.get("repair_stage"),
                "attempt": item.get("attempt"),
                "errors": item.get("errors", [])[:5],
                "router_reason": item.get("router_reason", ""),
                "repair_instruction": item.get("repair_instruction", ""),
                "previous_state": item.get("previous_state", {}),
            }
        )
    if state.errors:
        rows.append({"source": "current_errors", "errors": state.errors[-5:]})
    return rows


def infer_scenario(description: str) -> str:
    terms = set(tokenize(description))
    joined = description.lower()
    has_specific_scene_hint = any(
        word in joined
        for word in [
            "坐标",
            "主轴",
            "转速",
            "参数",
            "报警",
            "诊断",
            "程序",
            "宏",
            "刀具",
            "spindle",
            "parameter",
            "alarm",
            "diagnostic",
            "program",
            "coordinate",
            "position",
            "axis",
        ]
    )
    if (
        any(word in joined for word in ["全面", "多样", "覆盖", "综合"])
        or "diverse" in joined
        or "comprehensive" in joined
        or ("focas" in joined and ("traffic" in joined or "流量" in joined) and not has_specific_scene_hint)
    ):
        return "comprehensive_focas_traffic"
    if terms.intersection({"坐标", "coordinate", "position", "axis"}) or "坐标" in joined or "x/y/z" in joined:
        return "coordinate_motion"
    if terms.intersection({"主轴", "spindle"}) or "主轴" in joined or "spindle" in joined or "g96" in joined or "g97" in joined:
        return "spindle_speed_change"
    if terms.intersection({"参数", "parameter", "config"}) or "参数" in joined:
        if any(word in joined for word in ["write", "写", "修改", "设置"]):
            return "parameter_write_simulated"
        return "parameter_read"
    if terms.intersection({"报警", "alarm", "fault"}) or "报警" in joined:
        return "alarm_query"
    if terms.intersection({"诊断", "diagnostic", "diagnosis"}) or "诊断" in joined:
        return "diagnostic_query"
    if terms.intersection({"pmc", "di", "do"}):
        return "pmc_signal_read"
    if terms.intersection({"程序", "program", "nc", "gcode"}) or "程序" in joined:
        return "program_lifecycle"
    if terms.intersection({"宏", "macro"}) or "宏" in joined:
        return "macro_variable_read_write"
    if terms.intersection({"刀具", "tool", "offset"}) or "刀具" in joined:
        return "tool_offset_setting"
    return "general_status_collection"


def normalize_scenario(scenario: str) -> str:
    return SCENARIO_ALIASES.get(scenario, scenario)


def build_plan(
    task_id: str,
    scenario: str,
    target_environment: str,
    retrieval: dict[str, list[RetrievedChunk]],
    selected_templates: list[dict[str, Any]] | None = None,
) -> ExecutionPlan:
    selected_templates = selected_templates or []
    if scenario == "coordinate_motion":
        nc_program_spec = plan_nc_program_spec(scenario, selected_templates, task_id)
        steps = coordinate_motion_stepwise_steps(nc_program_spec)
    else:
        nc_program_spec = plan_nc_program_spec(scenario, selected_templates, task_id)
        steps = steps_from_templates(selected_templates) if selected_templates else steps_for(scenario)
        ensure_nc_program_lifecycle_steps(scenario, steps, nc_program_spec.program_name)
    attach_protocol_functions(steps)
    requirements = merge_requirements(
        nc_requirements_from_templates(selected_templates) or nc_requirements_for(scenario),
        retrieval.get("nc_rule", []),
    )
    rag_context = summarize_retrieval(retrieval)
    if selected_templates:
        rag_context["selected_scenario_templates"] = summarize_selected_templates(selected_templates)
    return ExecutionPlan(
        plan_id=f"plan-{task_id}",
        task_id=task_id,
        scenario_type=scenario,
        scenario_goal=scenario_goal_from_templates(selected_templates) or scenario_goal_for(scenario),
        target_environment=target_environment,
        nc_program_type=nc_type_from_templates(selected_templates) or nc_type_for(scenario),
        nc_program_requirements=requirements,
        nc_program_spec=nc_program_spec,
        steps=steps,
        expected_outputs=[
            "api_call_script",
            "nc_program",
            "capture_events",
            "api_logs",
            "quality_assessment",
        ],
        retrieved_chunk_ids=[item.chunk.chunk_id for item in flatten_retrieval(retrieval)],
        rag_context=rag_context,
    )


def plan_nc_program_spec(scenario: str, selected_templates: list[dict[str, Any]], task_id: str = "") -> NcProgramSpec:
    program_name = program_name_for_scenario(scenario, task_id)
    if scenario == "coordinate_motion":
        return NcProgramSpec(
            program_name=program_name,
            purpose="Ask CodeGenerationAgent to create observable single-block coordinate motion for position/feed/status traffic without excessive runtime.",
            block_goals=[
                "set absolute mode and work coordinate system",
                "move to a safe initial XYZ position",
                "perform X-axis feed motion with an observable 2-5 second sampling window",
                "perform XY motion with small Z change and an observable 2-5 second sampling window",
                "perform reverse X motion with an observable 2-5 second sampling window",
                "return to the start area with safe Z",
                "end the program",
            ],
            constraints=[
                "use a valid O program number",
                "prefer G90 and G54",
                "use G01 feed moves for observable coordinate changes",
                "choose feed values and travel distances so main motion blocks are observable but not excessively slow",
                "prefer multiple moderate-duration motion blocks over one very slow or very long block",
                "end with M30",
            ],
            generation_notes=[
                "Target about 2-5 seconds of observable motion per main G01 block; CodeGenerationAgent chooses exact coordinates and F words.",
                "Single Block mode is expected so generated blocks can align with Cycle Start and API capture timing.",
            ],
        )
    if scenario == "comprehensive_focas_traffic":
        return NcProgramSpec(
            program_name=program_name,
            purpose="Ask CodeGenerationAgent to cover motion, spindle, status, feed, alarm, and program lifecycle traffic in one safe simulator program.",
            block_goals=[
                "set absolute mode and work coordinate system",
                "start spindle at one speed",
                "perform observable feed motion",
                "change spindle speed",
                "perform second observable feed motion",
                "stop spindle",
                "return to a safe point",
                "end the program",
            ],
            constraints=[
                "include observable axis motion",
                "include spindle start, speed change, and stop behavior",
                "avoid unsafe machine travel",
                "end with M30",
            ],
            generation_notes=["Planner composed this specification from comprehensive scenario requirements and selected templates."],
        )
    if scenario in {"spindle_state", "spindle_speed_change"}:
        return NcProgramSpec(
            program_name=program_name,
            purpose="Ask CodeGenerationAgent to create observable spindle start, speed change, dwell, and stop states.",
            block_goals=[
                "set absolute mode and work coordinate system",
                "start spindle at baseline speed",
                "dwell for sampling",
                "change spindle speed",
                "dwell for sampling",
                "stop spindle",
                "end the program",
            ],
            constraints=["include M03 or M04", "include at least two S values", "include M05", "end with M30"],
            generation_notes=["Planner requests dwell blocks so spindle state can be sampled between speed changes."],
        )
    if scenario == "program_lifecycle":
        return NcProgramSpec(
            program_name=program_name,
            purpose="Ask CodeGenerationAgent to create a short executable program for upload, selection, start, and completion traffic.",
            block_goals=["set absolute mode", "perform one safe short operation", "dwell briefly", "end the program"],
            constraints=["valid program number", "safe short execution", "end with M30"],
        )
    return NcProgramSpec(
        program_name=program_name,
        purpose="Ask CodeGenerationAgent to create a minimal safe NC program for baseline collection.",
        block_goals=["set absolute mode", "dwell briefly", "end the program"],
        constraints=["valid program number", "avoid unsafe motion", "end with M30"],
        generation_notes=[
            "Planner used a minimal specification because this scenario does not require a dedicated machining sequence."
        ],
    )


def steps_from_templates(templates: list[dict[str, Any]]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for template_index, template in enumerate(templates, 1):
        prefix = f"C{template_index:02d}"
        for step_index, row in enumerate(template.get("operation_template", []), 1):
            if not isinstance(row, dict):
                continue
            interface_name = str(row.get("interface_name", "ReadRunStatus"))
            steps.append(
                PlanStep(
                    step_id=f"{prefix}-{step_index:03d}",
                    phase=phase_or_default(row.get("phase")),
                    action=f"{template.get('scenario_name', template.get('scenario_id', 'scenario'))}: {row.get('action', interface_name)}",
                    interface_name=interface_name,
                    parameters=dict(row.get("parameters", {}) or {}),
                    repeat=int(row.get("repeat", 1) or 1),
                    interval_seconds=float(row.get("interval_seconds", 0.0) or 0.0),
                    expected_state="; ".join(template.get("expected_signals", [])[:3]),
                    protocol_function=str(row.get("protocol_function", "")),
                )
            )
    return steps


def ensure_nc_program_lifecycle_steps(scenario: str, steps: list[PlanStep], program_name: str | None = None) -> None:
    if scenario not in {"coordinate_motion", "spindle_speed_change", "program_lifecycle", "comprehensive_focas_traffic"}:
        return
    program_name = program_name or program_name_for_scenario(scenario)
    missing_prefix: list[PlanStep] = []
    if not any(step.interface_name == "UploadProgram" for step in steps):
        missing_prefix.append(
            PlanStep("NC-001", "before", "upload generated NC program", "UploadProgram", {"program_name": program_name})
        )
    if not any(step.interface_name == "SelectProgram" for step in steps):
        missing_prefix.append(
            PlanStep("NC-002", "before", "select generated NC program", "SelectProgram", {"program_name": program_name})
        )
    if not any(step.interface_name == "ReadProgramNumber" for step in steps):
        missing_prefix.append(
            PlanStep("NC-003", "before", "verify current program number", "ReadProgramNumber", {})
        )
    if not any(step.interface_name == "StartProgram" for step in steps):
        missing_prefix.append(
            PlanStep(
                "NC-004",
                "during",
                "trigger NCGuide cycle start",
                "StartProgram",
                {
                    "window_title": "FANUC CNC GUIDE",
                    "button_text": "",
                    "click_mode": "screen",
                    "cycle_start_x": 989,
                    "cycle_start_y": 914,
                },
                interval_seconds=0.5,
            )
        )
    if missing_prefix:
        steps[:0] = missing_prefix


def program_name_for_scenario(scenario: str, task_id: str = "") -> str:
    number = random.SystemRandom().randint(1000, 8999)
    return f"O{number:04d}"


def classify_block_goal(goal: str) -> str:
    text = goal.lower()
    if "end" in text:
        return "program_end"
    if "spindle" in text and "start" in text:
        return "spindle_start"
    if "spindle" in text and "stop" in text:
        return "spindle_stop"
    if "spindle" in text and "speed" in text:
        return "spindle_speed_change"
    if "dwell" in text:
        return "dwell"
    if "feed" in text or "motion" in text or "move" in text:
        if "feed" in text or "slow" in text:
            return "feed_motion"
        return "positioning"
    if "coordinate" in text or "absolute" in text:
        return "coordinate_setup"
    return "planned_block"


def expected_feed_from_goal(goal: str) -> str:
    text = goal.lower()
    if "very slow" in text:
        return "very_low"
    if "slow" in text or "feed" in text:
        return "low"
    return ""


def phase_or_default(value: Any) -> str:
    return str(value) if value in {"before", "during", "after"} else "during"


def nc_requirements_from_templates(templates: list[dict[str, Any]]) -> list[str]:
    requirements: list[str] = []
    for template in templates:
        scene = template.get("scenario_name", template.get("scenario_id", "scenario"))
        for requirement in template.get("nc_program_requirements", []):
            text = f"{scene}: {requirement}"
            if text not in requirements:
                requirements.append(text)
    return requirements


def scenario_goal_from_templates(templates: list[dict[str, Any]]) -> str:
    if not templates:
        return ""
    if len(templates) == 1:
        return str(templates[0].get("goal", ""))
    return f"Cover {len(templates)} clustered FOCAS scenario templates for diverse traffic generation."


def nc_type_from_templates(templates: list[dict[str, Any]]) -> str:
    if not templates:
        return ""
    if len(templates) > 1:
        return "template_composed_focas_program"
    scene = str(templates[0].get("scenario_name", ""))
    if "coordinate" in scene or "motion" in scene:
        return "template_coordinate_motion"
    if "spindle" in scene:
        return "template_spindle_control"
    if "program" in scene:
        return "template_program_lifecycle"
    return "template_minimal_safe_program"


def summarize_selected_templates(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "template_id": template.get("template_id"),
            "scenario_id": template.get("scenario_id"),
            "scenario_name": template.get("scenario_name"),
            "coverage_priority": template.get("coverage_priority"),
            "cluster_member_count": template.get("cluster_member_count"),
            "main_apis": template.get("main_apis", [])[:5],
            "main_objects": template.get("main_objects", [])[:5],
            "steps": len(template.get("operation_template", [])),
        }
        for template in templates
    ]


def attach_protocol_functions(steps: list[PlanStep]) -> None:
    for step in steps:
        if step.protocol_function.strip():
            if not step.api_calls:
                step.api_calls = normalize_api_calls([], step.protocol_function, step.parameters)
            continue
        step.protocol_function = focas_function_for(step.interface_name)
        step.api_calls = normalize_api_calls([], step.protocol_function, step.parameters)


def merge_requirements(defaults: list[str], nc_rules: list[RetrievedChunk]) -> list[str]:
    requirements = list(defaults)
    for item in nc_rules[:3]:
        source = item.chunk.metadata.get("source_chunk_id") or item.chunk.chunk_id
        scenario = item.chunk.metadata.get("scenario", "")
        requirement = f"consult retrieved nc_rule {source} for {scenario}".strip()
        if requirement not in requirements:
            requirements.append(requirement)
    return requirements


def steps_for(scenario: str) -> list[PlanStep]:
    builders = {
        "comprehensive_focas_traffic": comprehensive_focas_steps,
        "coordinate_motion": coordinate_motion_steps,
        "spindle_speed_change": spindle_steps,
        "parameter_read": parameter_read_steps,
        "parameter_write_simulated": parameter_write_steps,
        "alarm_query": alarm_steps,
        "diagnostic_query": diagnostic_steps,
        "pmc_signal_read": pmc_steps,
        "program_lifecycle": program_steps,
        "tool_offset_setting": tool_offset_steps,
        "macro_variable_read_write": macro_steps,
        "general_status_collection": status_steps,
    }
    return builders.get(scenario, status_steps)()


def coordinate_motion_steps() -> list[PlanStep]:
    program_name = program_name_for_scenario("coordinate_motion")
    return [
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {"program_name": program_name}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {"program_name": program_name}),
        PlanStep("S003", "before", "read initial status", "ReadRunStatus", {}),
        PlanStep("S004", "during", "start NC program", "StartProgram", {}),
        PlanStep(
            "S005",
            "during",
            "sample axis positions",
            "ReadPosition",
            {"axes": ["X", "Y", "Z"], "coordinate_system": "machine"},
            repeat=5,
            interval_seconds=0.2,
            expected_state="coordinates change over time",
        ),
        PlanStep("S006", "during", "sample running status", "ReadRunStatus", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S007", "during", "sample feed speed", "ReadFeedSpeed", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S008", "after", "stop or confirm program end", "StopProgram", {}),
        PlanStep("S009", "after", "read final status", "ReadRunStatus", {}),
    ]


def coordinate_motion_stepwise_steps(nc_program_spec: NcProgramSpec) -> list[PlanStep]:
    program_name = nc_program_spec.program_name
    blocks = [
        (
            f"B{index:02d}",
            goal,
            classify_block_goal(goal),
            expected_feed_from_goal(goal),
        )
        for index, goal in enumerate(nc_program_spec.block_goals, 1)
    ]
    steps: list[PlanStep] = [
        PlanStep("NC-001", "before", "upload generated NC program", "UploadProgram", {"program_name": program_name}),
        PlanStep("NC-002", "before", "select generated NC program", "SelectProgram", {"program_name": program_name}),
        PlanStep("NC-003", "before", "verify current program number", "ReadProgramNumber", {}),
        PlanStep(
            "NC-004",
            "before",
            "verify single block mode is enabled before stepwise capture",
            "ReadRunStatus",
            {"execution_mode": "single_block_required"},
            expected_state="operator panel SINGLE BLOCK should be enabled",
        ),
    ]
    for index, (block_id, block_goal, block_scene, expected_feed) in enumerate(blocks, 1):
        block_params = {
            "program_name": program_name,
            "block_index": index,
            "block_id": block_id,
            "block_goal": block_goal,
            "block_scene": block_scene,
            "execution_mode": "single_block",
            "expected_feed": expected_feed,
        }
        steps.extend(
            [
                PlanStep(
                    f"{block_id}-PRE-STAT",
                    "before",
                    f"before block {index}: read run status",
                    "ReadRunStatus",
                    block_params,
                    expected_state=f"before executing planned block goal: {block_goal}",
                ),
                PlanStep(
                    f"{block_id}-PRE-POS",
                    "before",
                    f"before block {index}: read machine coordinates",
                    "ReadPosition",
                    {**block_params, "axes": ["X", "Y", "Z"], "coordinate_system": "machine"},
                    expected_state="baseline coordinate snapshot",
                ),
                PlanStep(
                    f"{block_id}-TRIGGER",
                    "during",
                    f"trigger single block {index}: {block_goal}",
                    "StartProgram",
                    {
                        **block_params,
                        "window_title": "FANUC CNC GUIDE",
                        "button_text": "",
                        "click_mode": "screen",
                        "cycle_start_x": 989,
                        "cycle_start_y": 914,
                        "pre_start_gate": "wait_until_previous_block_not_active",
                        "gate_status_api": "ReadRunStatus",
                        "max_wait_seconds": 8.0,
                        "poll_interval_seconds": 0.2,
                    },
                    interval_seconds=0.2,
                    expected_state="NC block advances after Cycle Start",
                ),
                PlanStep(
                    f"{block_id}-DURING-STAT",
                    "during",
                    f"during block {index}: motion-gated status sampling",
                    "ReadRunStatus",
                    {**block_params, "sample_role": "during_motion_gate"},
                    repeat=3,
                    interval_seconds=0.1,
                    expected_state="confirm motion state immediately after Cycle Start",
                ),
                PlanStep(
                    f"{block_id}-DURING-FEED",
                    "during",
                    f"during block {index}: focused feed sampling",
                    "ReadFeedSpeed",
                    {**block_params, "sample_role": "during_focused"},
                    repeat=5,
                    interval_seconds=0.2,
                    expected_state="capture active feed speed while motion state is active",
                ),
                PlanStep(
                    f"{block_id}-DURING-POS",
                    "during",
                    f"during block {index}: focused position sampling",
                    "ReadPosition",
                    {**block_params, "axes": ["X", "Y", "Z"], "coordinate_system": "machine", "sample_role": "during_focused"},
                    repeat=5,
                    interval_seconds=0.2,
                    expected_state="capture transient coordinate changes immediately after Cycle Start",
                ),
                PlanStep(
                    f"{block_id}-POST-STAT",
                    "after",
                    f"after block {index}: read run status",
                    "ReadRunStatus",
                    block_params,
                    repeat=1,
                    interval_seconds=0.1,
                    expected_state=f"status after executing planned block goal: {block_goal}",
                ),
                PlanStep(
                    f"{block_id}-POST-POS",
                    "after",
                    f"after block {index}: read machine coordinates",
                    "ReadPosition",
                    {**block_params, "axes": ["X", "Y", "Z"], "coordinate_system": "machine"},
                    repeat=1,
                    interval_seconds=0.1,
                    expected_state="coordinate snapshot aligned to completed block",
                ),
                PlanStep(
                    f"{block_id}-POST-FEED",
                    "after",
                    f"after block {index}: read feed speed",
                    "ReadFeedSpeed",
                    block_params,
                    expected_state="feed/status sample aligned to completed block",
                ),
            ]
        )
    return steps


def comprehensive_focas_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload comprehensive NC program", "UploadProgram", {"program_name": "O8001"}),
        PlanStep("S002", "before", "select comprehensive NC program", "SelectProgram", {"program_name": "O8001"}),
        PlanStep("S003", "before", "read baseline run status", "ReadRunStatus", {}),
        PlanStep("S004", "before", "read baseline parameter", "ReadParameter", {"parameter_no": 1001}),
        PlanStep("S005", "during", "start comprehensive scene", "StartProgram", {}),
        PlanStep(
            "S006",
            "during",
            "sample changing axis positions",
            "ReadPosition",
            {"axes": ["X", "Y", "Z"], "coordinate_system": "machine"},
            repeat=5,
            interval_seconds=0.2,
            expected_state="machine coordinates change over time",
        ),
        PlanStep("S007", "during", "sample feed speed", "ReadFeedSpeed", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S008", "during", "sample spindle speed", "ReadSpindleSpeed", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S009", "during", "sample run status", "ReadRunStatus", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S010", "during", "query alarms during execution", "ReadAlarm", {}, repeat=2, interval_seconds=0.2),
        PlanStep("S011", "after", "stop comprehensive scene", "StopProgram", {}),
        PlanStep("S012", "after", "read final run status", "ReadRunStatus", {}),
    ]


def spindle_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload spindle NC program", "UploadProgram", {"program_name": "O2001"}),
        PlanStep("S002", "before", "select spindle NC program", "SelectProgram", {"program_name": "O2001"}),
        PlanStep("S003", "during", "start spindle scene", "StartProgram", {}),
        PlanStep("S004", "during", "read spindle speed", "ReadSpindleSpeed", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S005", "during", "read running status", "ReadRunStatus", {}, repeat=2, interval_seconds=0.2),
        PlanStep("S006", "after", "stop spindle scene", "StopProgram", {}),
    ]


def parameter_read_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read parameter", "ReadParameter", {"parameter_no": 1001}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def parameter_write_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline parameter", "ReadParameter", {"parameter_no": 1001}),
        PlanStep("S002", "during", "write safe simulated parameter value", "WriteParameter", {"parameter_no": 1001, "value": 1}),
        PlanStep("S003", "after", "read parameter after write", "ReadParameter", {"parameter_no": 1001}),
    ]


def alarm_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "query active alarms", "ReadAlarm", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def diagnostic_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read diagnostic-like parameter", "ReadParameter", {"parameter_no": 300}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def pmc_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read simulated PMC signal range", "ReadParameter", {"parameter_no": 1200}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def program_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {"program_name": "O3001"}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {"program_name": "O3001"}),
        PlanStep("S003", "during", "start program", "StartProgram", {}),
        PlanStep("S004", "during", "read program status", "ReadRunStatus", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S005", "after", "stop program", "StopProgram", {}),
    ]


def tool_offset_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read simulated tool offset parameter", "ReadParameter", {"parameter_no": 5013}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def macro_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read simulated macro variable", "ReadParameter", {"parameter_no": 500}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def status_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read initial status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read position", "ReadPosition", {"axes": ["X", "Y", "Z"]}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def scenario_goal_for(scenario: str) -> str:
    goals = {
        "comprehensive_focas_traffic": "Generate broad and diverse FOCAS traffic across program lifecycle, motion, spindle, status, parameter, and alarm queries.",
        "coordinate_motion": "Generate traffic with changing coordinates, run state, and feed speed.",
        "spindle_speed_change": "Generate traffic related to spindle speed and spindle state changes.",
        "parameter_read": "Generate traffic for CNC parameter reading behavior.",
        "parameter_write_simulated": "Generate safe simulated traffic for parameter write behavior.",
        "alarm_query": "Generate traffic for alarm and status queries.",
        "diagnostic_query": "Generate traffic for diagnostic data queries.",
        "pmc_signal_read": "Generate traffic for PMC/DI/DO-like signal reads.",
        "program_lifecycle": "Generate traffic for NC program upload, selection, run, and stop.",
        "tool_offset_setting": "Generate traffic for tool offset and compensation data access.",
        "macro_variable_read_write": "Generate traffic for macro variable read/write-like operations.",
        "general_status_collection": "Generate baseline CNC status collection traffic.",
    }
    return goals.get(scenario, goals["general_status_collection"])


def nc_type_for(scenario: str) -> str:
    if scenario == "comprehensive_focas_traffic":
        return "comprehensive_focas_program"
    if scenario == "coordinate_motion":
        return "straight_interpolation_motion"
    if scenario == "spindle_speed_change":
        return "spindle_speed_change"
    if scenario == "program_lifecycle":
        return "basic_program_lifecycle"
    return "minimal_safe_program"


def nc_requirements_for(scenario: str) -> list[str]:
    common = ["include a valid program number", "end with M30", "keep execution short"]
    if scenario == "comprehensive_focas_traffic":
        return common + [
            "include observable axis motion",
            "include spindle start, speed change, and stop behavior",
            "support status, parameter, and alarm collection before/during/after execution",
        ]
    if scenario == "coordinate_motion":
        return common + ["move X/Y/Z through observable points", "use G90 and G01 feed motion"]
    if scenario == "spindle_speed_change":
        return common + ["start spindle with M03", "change spindle speed", "stop spindle with M05"]
    if scenario == "program_lifecycle":
        return common + ["include a short executable program body"]
    return common
