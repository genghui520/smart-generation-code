from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..knowledge import KnowledgeBase
from ..agent_tools.knowledge import KnowledgeSearchInput, KnowledgeSearchOutput
from ..llm import LlmClient, TransientLlmError, temporary_llm_timeout
from ..models import ExecutionPlan, NcProgramSpec, PlanStep, RetrievedChunk, WorkflowState
from ..tools import focas_function_for, interface_to_focas_mapping
from ..utils import tokenize
from ..function_coverage import DEFAULT_FUNCTION_COVERAGE_STATE, load_resolved_functions
from ..api_prerequisites import build_prerequisite_plan, load_manifest_rows, manifest_functions_by_category
from ..orchestration.roles import ActionSpec, AgentTemplate, message_to_dict
from ..tool_runtime import ToolRuntime
from ..agent_roles import PlanSpec
from ..code_spec import build_code_spec, build_official_abi_context
from ..execution_contracts import MOTION_TARGET_APIS, motion_required_for_targets
from ..rag.focas_function_manifest import (
    DEFAULT_COVERAGE_BATCH_SIZE,
    DEFAULT_COVERAGE_SEGMENT_COUNT,
    DEFAULT_FOCAS_FUNCTION_MANIFEST,
    build_function_coverage_context,
)
from ..rag.scenario_templates import DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH
from ..fast_path import eligible_read_fast_path, build_fast_path_steps
from ..scenario_capabilities import analyze_output_dependencies, build_scene_capability_plan, load_scenario_capabilities
from ..integrations.ncguide import default_focas_header_dir
from .prompts import PLANNING_REVIEW_JSON_SCHEMA, PLANNING_REVIEW_SYSTEM_PROMPT


RULE_TYPES = ["nc_rule", "operation_rule", "collection_rule", "safety_rule", "api_prerequisite"]

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
        self.tool_runtime: ToolRuntime | None = None

    def run(self, state: WorkflowState) -> WorkflowState:
        # Planning is the slowest LLM phase because it carries RAG and ABI
        # context. A single provider timeout is retried as a wait/retry event;
        # it is never turned into a repair_plan request. The outer bound keeps
        # a broken provider from holding the workflow forever.
        max_transient_attempts = 3
        for attempt in range(1, max_transient_attempts + 1):
            try:
                print(
                    f"[PlanningAgent] LLM planning request attempt "
                    f"{attempt}/{max_transient_attempts}; timeout window=240s",
                    flush=True,
                )
                with temporary_llm_timeout(self.llm_client, 240.0):
                    return self._run_with_timeout(state)
            except TransientLlmError as exc:
                if attempt >= max_transient_attempts:
                    print(
                        f"[PlanningAgent] transient_llm_error attempts exhausted "
                        f"({attempt}/{max_transient_attempts}); ending run without repair_plan: {exc}",
                        flush=True,
                    )
                    raise
                delay = 2 ** (attempt - 1)
                print(
                    f"[PlanningAgent] waiting after transient LLM failure; "
                    f"retry={attempt + 1}/{max_transient_attempts} in {delay}s "
                    f"(timeout window=240s): {exc}",
                    flush=True,
                )
                import time
                time.sleep(delay)

    def _run_with_timeout(self, state: WorkflowState) -> WorkflowState:
        request = state.request
        scenario = normalize_scenario(infer_scenario(request.description))
        selected_templates = select_scenario_templates(
            request.description,
            scenario,
            self.scenario_templates,
        )
        explicit_ids = request.permissions.get("selected_scenario_ids")
        if explicit_ids:
            selected_templates = [t for t in self.scenario_templates if t.get("scenario_id") in explicit_ids]
            missing = set(explicit_ids) - {t["scenario_id"] for t in selected_templates}
            if missing:
                raise ValueError(f"Unknown explicitly selected scenarios: {sorted(missing)}")
            if len(selected_templates) == 1:
                scene_name = selected_templates[0].get("scenario_name", "")
                scenario = {
                    "coordinate_feed_motion": "coordinate_motion",
                    "alarm_diagnosis": "alarm_query",
                    "parameter_macro_access": "parameter_read",
                    "pmc_signal_monitoring": "pmc_signal_read",
                    "tool_life_management": "tool_management",
                    "tool_offset_management": "tool_offset_setting",
                    "spindle_control": "spindle_speed_change",
                }.get(scene_name, scene_name or scenario)
        # The deterministic local RAG pass below is sufficient for planning.
        # Do not run a second LLM->search_knowledge loop: it duplicated context
        # and consumed extra HTTP attempts before the actual plan request.
        tool_retrieval = []
        retrieval = retrieve_planning_context(
            self.knowledge_base,
            request.description,
            scenario,
            tool_runtime=self.tool_runtime,
            agent_name=self.role_name,
        )
        requested_targets = [str(item).strip() for item in (
            request.permissions.get("coverage_current_batch_functions")
            or request.permissions.get("required_api_functions") or ()
        ) if str(item).strip()]
        # Retrieve exact API evidence for every explicitly selected target.
        # Broad scenario similarity is not reliable enough to supply argument
        # domains or output dependencies for a specific function.
        if requested_targets and hasattr(self.knowledge_base, "lookup_api_exact"):
            exact_api_rows = self.knowledge_base.lookup_api_exact(requested_targets)
            retrieval.setdefault("api", []).extend(exact_api_rows or [])
        retrieval = deduplicate_retrieval_context(retrieval)
        print(
            "[PlanningAgent] RAG results "
            + ", ".join(
                f"{key}={len(items)} items/{sum(len(item.chunk.text) for item in items) // 4} tokens_est"
                for key, items in retrieval.items()
            ),
            flush=True,
        )
        if tool_retrieval:
            retrieval["agent_tool_search"] = tool_retrieval
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
        plan.rag_context["explicit_target_functions"] = requested_targets
        plan.rag_context["coverage_intent"] = infer_coverage_intent(request.description, request.target_environment)
        if (
            plan.rag_context["coverage_intent"].get("full_function_coverage")
            and not request.permissions.get("representative_scenario", False)
        ):
            resolved_functions = load_resolved_functions(self.coverage_state_path)
            resolved_functions.update(
                str(item) for item in request.permissions.get("coverage_skip_functions", []) if str(item).strip()
            )
            coverage_context = build_function_coverage_context(
                self.function_manifest_path,
                covered_functions=resolved_functions,
                target_functions=(
                    set(request.permissions["coverage_target_functions"])
                    if "coverage_target_functions" in request.permissions else None
                ),
                batch_size=self.coverage_batch_size,
                scenario_type=scenario,
                max_segments=self.coverage_segment_count,
            )
            plan.rag_context["function_coverage_manifest"] = coverage_context
            if coverage_context.get("enabled") and not request.permissions.get("code_only_evaluation"):
                # The generic comprehensive-scene skeleton contains NC upload,
                # Cycle Start, and axis sampling. It is not a valid starting
                # plan for arbitrary manifest-selected API families.
                plan.steps = []
                plan.rag_context["coverage_segments"] = coverage_segments_from_manifest_context(coverage_context)
                plan.llm_notes.append("Manifest targets, not a coordinate-motion template, define this API batch.")
            plan.rag_context["function_coverage_state"] = {
                "state_path": str(self.coverage_state_path),
                "resolved_function_count": len(resolved_functions),
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
        if candidate_functions and hasattr(self.knowledge_base, "lookup_api_exact"):
            exact_candidate_docs = self.knowledge_base.lookup_api_exact(candidate_functions)
            retrieval.setdefault("api", []).extend(exact_candidate_docs)
            retrieval = deduplicate_retrieval_context(retrieval)
            plan.rag_context["api"] = summarize_retrieval({"api": retrieval.get("api", [])}).get("api", [])
            priority_functions = set(requested_targets)
            priority_functions.update(
                str(row.get("function", "")).strip()
                for row in plan.rag_context.get("function_coverage_manifest", {}).get("selected_batch", [])
                if isinstance(row, dict) and row.get("coverage_role", "target") == "target"
            )
            priority_functions.update(
                name for step in plan.steps
                for name in protocol_function_names_for_planner(step.protocol_function)
            )
            plan.rag_context["exact_api_contracts"] = [
                {
                    "function": item.chunk.metadata.get("function", ""),
                    "chunk_type": item.chunk.metadata.get("chunk_type", ""),
                    "chunk_id": item.chunk.chunk_id,
                    "source_url": item.chunk.metadata.get("source_url", ""),
                    "text": item.chunk.text,
                }
                for item in exact_candidate_docs
                if str(item.chunk.metadata.get("function", "")).strip() in priority_functions
            ]
        plan.rag_context["official_abi_context"] = build_official_abi_context(candidate_functions)
        if self.environment is not None:
            plan.rag_context["environment_shared_rules"] = self.environment.shared_rules
            plan.rag_context["official_abi_context"] = self.environment.official_abi(candidate_functions)
        manifest_rows = load_manifest_rows(self.function_manifest_path)
        prerequisite_functions = set(candidate_functions)
        prerequisite_functions.update(
            str(row.get("function", "")).strip()
            for row in plan.rag_context.get("function_coverage_manifest", {}).get("selected_batch", [])
            if isinstance(row, dict) and row.get("function")
        )
        # The manifest batch may be selected after the generic scenario query;
        # retrieve prerequisite rules again using exact target names so a broad
        # full-coverage request cannot miss a function-specific rule.
        if prerequisite_functions and hasattr(self.knowledge_base, "lookup_api_prerequisites_exact"):
            exact_prerequisites = self.knowledge_base.lookup_api_prerequisites_exact(prerequisite_functions)
            existing_ids = {
                item.chunk.chunk_id for item in retrieval.get("api_prerequisite", [])
            }
            retrieval.setdefault("api_prerequisite", []).extend(
                item for item in exact_prerequisites if item.chunk.chunk_id not in existing_ids
            )
        prerequisite_rules = []
        for item in retrieval.get("api_prerequisite", []):
            rule = dict(item.chunk.metadata)
            rule["text"] = item.chunk.text
            for key in ("applies_to", "conditions", "preparation_operations", "verification_operations"):
                raw = rule.get(key)
                if isinstance(raw, str):
                    try:
                        rule[key] = json.loads(raw)
                    except json.JSONDecodeError:
                        rule[key] = []
            prerequisite_rules.append(rule)
        tool_life_functions = manifest_functions_by_category(manifest_rows, "tool life management data")
        for rule in prerequisite_rules:
            if rule.get("rule_id") == "FANUC-tool-life-management-prerequisite":
                rule["applies_to"] = tool_life_functions
        plan.rag_context["api_prerequisites"] = build_prerequisite_plan(
            prerequisite_functions, prerequisite_rules,
            environment=request.target_environment, permissions=request.permissions,
        )
        fast_targets = [str(item).strip() for item in (
            request.permissions.get("coverage_current_batch_functions")
            or request.permissions.get("required_api_functions") or ()
        ) if str(item).strip()]
        manifest_names = [str(row.get("function", "")).strip() for row in manifest_rows if isinstance(row, dict)]
        plan.rag_context.setdefault("scenario_capabilities", load_scenario_capabilities())
        requested_scene_plans = [
            build_scene_capability_plan(name, plan.rag_context["scenario_capabilities"])
            for name in fast_targets
        ]
        if requested_scene_plans:
            plan.rag_context["scene_plans"] = requested_scene_plans
            if len(requested_scene_plans) == 1:
                plan.rag_context["scene_plan"] = requested_scene_plans[0]
        requires_event_planning = any(
            analyze_output_dependencies(name)["dependency_kind"] == "external_event_and_runtime_state"
            for name in fast_targets
        )
        fast_ok, fast_facts = eligible_read_fast_path(
            fast_targets,
            header_path=default_focas_header_dir() / "Fwlib32.h",
            manifest_functions=manifest_names,
        )
        if requires_event_planning and self.llm_client.enabled:
            fast_ok = False
        if fast_ok:
            target = fast_targets[0]
            plan.steps = build_fast_path_steps(target)
            plan.scenario_goal = f"Deterministic read-only capture of {target}"
            plan.nc_program_type = "not_required"
            plan.nc_program_requirements = []
            plan.nc_program_spec = NcProgramSpec(
                program_name="", purpose="Read-only API capture; no NC program or motion required.",
                constraints=["No parameter writes", "No Cycle Start", "Preserve AUTO mode"],
            )
            plan.rag_context["known_knowledge"] = fast_facts
            plan.rag_context["fast_path"] = {
                "enabled": True, "kind": "read_only_api", "target": target,
                "llm_calls_avoided": ["planning_decision", "nc_program_generation"],
                "fallback": "Planner LLM when SDK/controller/parameter facts are incomplete",
            }
            plan.llm_notes.append("Deterministic fast path selected from SDK/manifest/read-only recipe; Planner LLM decision skipped.")
            if requires_event_planning:
                plan.nc_program_type = "required_for_variation_after_gates"
                plan.nc_program_requirements = [
                    "bounded G31 NC program",
                    "documented SKIP event behavior",
                    "no execution until NCGuide isolation, permission, and motion bounds are verified",
                ]
                plan.nc_program_spec = NcProgramSpec(
                    program_name="",
                    purpose="Plan a bounded G31/SKIP scenario for cnc_skip response variation; do not upload or execute until all simulator gates are verified.",
                    block_goals=["bounded G31 move that can observe a documented SKIP event", "end with M30"],
                    constraints=[
                        "do not guess SKIP signal injection or write PMC/system parameters",
                        "validate NCGuide isolation, execution permission, syntax, and motion bounds before upload or Cycle Start",
                    ],
                    generation_notes=["Current run retains only the read-only cnc_skip baseline."],
                )
        else:
            if not self.llm_client.enabled:
                raise RuntimeError("PlanningAgent requires an LLM planning decision in agent-only mode.")
            apply_llm_planning_decision(
                plan,
                self.llm_client,
                request.description,
                representative_scenario=request.permissions.get("representative_scenario", False),
            )
        plan.rag_context.setdefault("scenario_capabilities", load_scenario_capabilities())
        scene_targets = list(dict.fromkeys(
            fast_targets or [
                name for step in plan.steps
                for name in protocol_function_names_for_planner(step.protocol_function)
            ]
        ))
        plan.rag_context["scene_plans"] = [
            build_scene_capability_plan(name, plan.rag_context["scenario_capabilities"])
            for name in scene_targets
        ]
        if len(plan.rag_context["scene_plans"]) == 1:
            plan.rag_context["scene_plan"] = plan.rag_context["scene_plans"][0]
        if not request.permissions.get("code_only_evaluation"):
            gated_scenes = [
                scene for scene in plan.rag_context["scene_plans"]
                if isinstance(scene, dict)
                and scene.get("nc_execution_decision", {}).get("cycle_start_required")
                and not scene.get("nc_execution_decision", {}).get("cycle_start_allowed")
                and scene.get("operator_interaction", {}).get("cycle_start_action")
            ]
            if gated_scenes:
                plan.steps = [
                    step for step in plan.steps
                    if step.interface_name not in {
                        "UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram", "StopProgram"
                    }
                ]
                plan.rag_context["deferred_operator_actions"] = [
                    {
                        "action": "manual_cycle_start",
                        "target": scene.get("target_api"),
                        "required_when": scene.get("operator_interaction", {}).get("required_when", []),
                        "execution_gate": scene.get("execution_policy"),
                        "status": "deferred_until_execution_gates_verified",
                    }
                    for scene in gated_scenes
                ]
                plan.llm_notes.append(
                    "Scene rule requires NC execution for response variation, but current capability/permission gates are unverified; retain the read-only target sample and defer Cycle Start."
                )
        plan.rag_context["environment_preflight_policy"] = {
            "mode": "read_current_mode_before_any_mode_dependent_operation",
            "mode_change": "never_automatic; pause_and_request_operator_when_mismatched",
            "parameter_write": "only exact sourced parameter/bit/value and applicability plus explicit allow_environment_writes permission; snapshot/write/readback; stop on mismatch or restart requirement",
            "nc_execution": "only when output dependency requires runtime execution; do not infer Cycle Start permission from diversity goals",
            "operator_resume": "require operator confirmation followed by fresh read-only verification",
        }
        from ..execution_contracts import execution_contract, normalize_lifecycle_steps, analyze_api_interactions
        normalize_lifecycle_steps(plan.steps)
        planned_functions = {
            name for step in plan.steps
            for name in protocol_function_names_for_planner(step.protocol_function)
        }
        manifest_rows = plan.rag_context.get("api", [])
        interaction_plan = analyze_api_interactions(planned_functions, manifest_rows if isinstance(manifest_rows, list) else None)
        if interaction_plan:
            plan.rag_context["api_interaction_plan"] = interaction_plan
            plan.llm_notes.append(
                "Planner interaction analysis added discovery/lifecycle/restore helpers for: "
                + ", ".join(sorted(interaction_plan))
            )
        plan.rag_context["execution_contract"] = execution_contract(
            plan, request.permissions.get("coverage_current_batch_functions")
            or request.permissions.get("required_api_functions") or (),
        ).to_dict()
        plan.llm_notes.append("Planner review LLM pass disabled; CodeGenerationAgent owns implementation details.")
        state.retrieved_chunks = flatten_retrieval(retrieval)
        from ..parameter_coverage import attach_parameter_contract
        attach_parameter_contract(plan, request.permissions.get("coverage_current_batch_functions")
                                  or request.permissions.get("required_api_functions") or ())
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

    def _model_tool_retrieval(self, task_description: str, scenario: str) -> list[RetrievedChunk]:
        """Let the planning model request RAG evidence through the shared harness."""
        if self.tool_runtime is None or not self.llm_client.enabled:
            return []
        loop = self.tool_loop(max_turns=2)
        system_prompt = (
            "You are the knowledge-retrieval substep of PlanningAgent. "
            "Before producing a final response, call search_knowledge at least once. "
            "Use the returned official API, ABI, structure, and scenario rules as evidence. "
            "After retrieval, return JSON only with {\"retrieval_complete\":true}."
        )
        user_prompt = (
            f"Task: {task_description}\n"
            f"Scenario: {scenario}\n"
            "Call search_knowledge with a focused query for the APIs, official prototypes, "
            "structure definitions, safety rules, and scenario constraints needed for planning."
        )
        loop.run(
            self.role_name,
            system_prompt,
            user_prompt,
            input_factory=self._tool_input_factory,
        )
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for tool_name, output, _ in loop.last_tool_outputs:
            if tool_name != "search_knowledge" or not isinstance(output, KnowledgeSearchOutput):
                continue
            for item in output.results:
                if item.chunk.chunk_id in seen:
                    continue
                seen.add(item.chunk.chunk_id)
                chunks.append(item)
        return chunks

    def _tool_input_factory(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "search_knowledge":
            raise ValueError(f"PlanningAgent retrieval loop only supports search_knowledge, got {tool_name}")
        return KnowledgeSearchInput(
            query=str(arguments.get("query", "")),
            top_k=int(arguments.get("top_k", 8)),
            filters=dict(arguments.get("filters", {}) or {}),
        )


def load_scenario_templates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    templates = payload.get("templates", [])
    templates = [template for template in templates if isinstance(template, dict)]
    if payload.get("knowledge_type") == "scenario_execution_templates":
        from ..rag.scenario_templates import consolidate_scenario_templates
        return consolidate_scenario_templates(templates)
    return templates


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
        "tool_management": "tool_life_management",
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
    *,
    tool_runtime: ToolRuntime | None = None,
    agent_name: str = "PlanningRole",
) -> dict[str, list[RetrievedChunk]]:
    retrieval_query = f"{query}\nscenario: {scenario}"
    context: dict[str, list[RetrievedChunk]] = {}
    def search(search_query: str, top_k: int, filters: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        if tool_runtime is None:
            if filters and filters.get("source_type") == "api" and hasattr(knowledge_base, "search_api"):
                results = knowledge_base.search_api(search_query, top_k=top_k)
                print_rag_search_stats(filters, results)
                return results
            if filters and filters.get("source_type") == "rule" and hasattr(knowledge_base, "search_rules"):
                results = knowledge_base.search_rules(search_query, top_k=top_k)
                print_rag_search_stats(filters, results)
                return results
            if filters and filters.get("rule_type") and hasattr(knowledge_base, "search_rules_by_type"):
                results = knowledge_base.search_rules_by_type(search_query, str(filters["rule_type"]), top_k=top_k)
                print_rag_search_stats(filters, results)
                return results
            if filters and filters.get("scenario") and hasattr(knowledge_base, "search_by_scenario"):
                results = knowledge_base.search_by_scenario(search_query, str(filters["scenario"]), top_k=top_k)
                print_rag_search_stats(filters, results)
                return results
            if filters and filters.get("scenario") and hasattr(knowledge_base, "search_scenario_organization"):
                results = knowledge_base.search_scenario_organization(
                    search_query, str(filters["scenario"]), top_k=top_k
                )
                print_rag_search_stats(filters, results)
                return results
            results = knowledge_base.search(search_query, top_k=top_k) if hasattr(knowledge_base, "search") else []
            print_rag_search_stats(filters, results)
            return results
        output, _ = tool_runtime.call(
            agent_name,
            "search_knowledge",
            KnowledgeSearchInput(search_query, top_k, filters or {}),
        )
        print_rag_search_stats(filters, output.results)
        return output.results

    context["scenario_organization"] = search(retrieval_query, 1, {"scenario": scenario})
    for rule_type in RULE_TYPES:
        context[rule_type] = search(retrieval_query, 2, {"rule_type": rule_type})
    context["api"] = search(retrieval_query, 8, {"source_type": "api"})
    if not any(context.values()):
        context["general"] = search(retrieval_query, 6)
    return context


def print_rag_search_stats(filters: dict[str, Any] | None, results: list[RetrievedChunk]) -> None:
    raw_ids = [str(item.chunk.chunk_id) for item in results]
    print(
        f"[PlanningAgent] search_knowledge filters={filters or {}} returned={len(results)} "
        f"unique={len(set(raw_ids))} content_tokens_est={sum(len(item.chunk.text) for item in results) // 4}",
        flush=True,
    )


def deduplicate_retrieval_context(context: dict[str, list[RetrievedChunk]]) -> dict[str, list[RetrievedChunk]]:
    """Remove duplicate chunks returned by overlapping scenario/rule/API searches."""
    seen: set[str] = set()
    result: dict[str, list[RetrievedChunk]] = {}
    for key, items in context.items():
        unique: list[RetrievedChunk] = []
        for item in items:
            chunk_id = str(item.chunk.chunk_id)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique.append(item)
        result[key] = unique
    return result


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


def parse_planning_markdown(text: str) -> dict[str, Any]:
    """Convert the fixed Planner Markdown contract into the existing payload."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("PlanningAgent returned empty Markdown.")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    if cleaned.startswith("{"):
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("PlanningAgent output must describe one plan object.")
        return payload

    section_aliases = {
        "scenario goal": "scenario_goal",
        "quality analysis": "quality_analysis",
        "quality targets": "quality_targets",
        "nc program": "nc_program_spec",
        "scene requirements": "scene_requirements",
        "steps": "steps",
        "api coverage analysis": "api_coverage_analysis",
        "coverage segments": "coverage_segments",
        "notes": "notes",
    }
    payload: dict[str, Any] = {
        "quality_analysis": {},
        "quality_targets": {},
        "nc_program_spec": {},
        "steps": [],
    }
    current_section = ""
    current_step: dict[str, Any] | None = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        step_heading = re.match(r"^#{2,6}\s+(.+?)\s*$", line)
        if step_heading and current_section == "steps":
            current_step = {"step_id": step_heading.group(1).strip()}
            payload["steps"].append(current_step)
            continue
        # Providers sometimes emit the required sections as ``## NC Program``
        # or ``### NC Program`` and wrap keys in Markdown bold markers.  Treat
        # those as the same contract instead of silently dropping the section.
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            heading_name = re.sub(r"^\*+|\*+$", "", heading.group(1).strip()).strip().lower()
            current_section = section_aliases.get(heading_name, "")
            current_step = None
            continue
        key_value = re.match(
            r"^(?:[-*]\s+)?(?:\*\*)?([A-Za-z][A-Za-z0-9_]*)(?:\*\*)?\s*[:=]\s*(.*)$",
            line,
        )
        if not key_value or not current_section:
            continue
        key, raw_value = key_value.groups()
        value = _parse_planner_markdown_value(raw_value)
        if current_section == "scenario_goal":
            if key in {"goal", "scenario_goal"}:
                payload["scenario_goal"] = str(value)
        elif current_section == "steps":
            if current_step is None:
                raise ValueError("PlanningAgent Markdown step fields require a preceding ## step_id heading.")
            current_step[key] = value
        elif current_section in {"api_coverage_analysis", "coverage_segments", "notes", "scene_requirements"}:
            rows = payload.setdefault(current_section, [])
            if key in {"item", "note"}:
                rows.append(value)
        else:
            payload[current_section][key] = value

    return payload


def _parse_planner_markdown_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == "`" and value[-1] == "`":
        value = value[1:-1].strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
            return float(value)
        return value


def apply_llm_planning_decision(
    plan: ExecutionPlan,
    llm_client: LlmClient,
    task_description: str,
    *,
    representative_scenario: bool = False,
) -> None:
    # Keep the initial semantic contract as a guardrail. The LLM may add legal
    # APIs, but it must not replace a task-level interface with a different
    # function merely because the latter appeared in RAG evidence.
    semantic_baseline = {
        step.interface_name: step
        for step in plan.steps
        if step.interface_name in SEMANTIC_INTERFACE_TO_FOCAS
    }
    system_prompt = (
        "# Identity\n"
        "You are PlanningAgent in a multi-agent FANUC FOCAS traffic-generation system.\n"
        "You generate the execution plan specification, not concrete C++ code and not concrete G-code.\n\n"
        "# Instructions\n"
        "- Decide the NC program structure, the set of related FOCAS APIs to cover, each API's input parameter strategy, the execution order, timing, and safety constraints. Do not choose an O program number; CodeGenerationAgent allocates it.\n"
        "- First analyze the NC program scenario described by the task, then infer which FOCAS APIs are semantically related by reading the retrieved knowledge-base facts: function names, descriptions, prototypes, arguments, return data, examples, and error notes.\n"
        "- Before selecting each target API, inspect rag_context.api_prerequisites. For every matched rule, include the documented prerequisite conditions, source evidence, current-state checks, expected state, preparation operations, verification operations, restart requirement, and failure policy in the plan. Reuse one preparation/verification sequence for targets sharing a rule. Never invent a parameter number, bit, value, restart requirement, or controller capability when the rule does not provide evidence.\n"
        "- Environment writes are allowed only when task permissions explicitly allow isolated-simulator writes and the rule marks them safe. Use read-original -> modify only the documented bit -> readback -> restore/snapshot; if restart is required, stop for operator/environment reinitialization and do not execute the target early.\n"
        "- Do not rely on a local tool/interface registry as the API universe. The knowledge base is the source of API facts; you may plan any relevant FOCAS protocol_function found or inferable from retrieved API knowledge.\n"
        "- Build an API coverage set from api_candidate_pool plus any additional relevant API facts in RAG: include APIs that can produce relevant returned outputs, state transitions, lifecycle evidence, control/synchronization evidence, or intentionally varied simulator-side mutation evidence for this NC program scenario. Exclude APIs only with a brief reason.\n"
        "- When coverage_intent.full_traffic_coverage=true, optimize for broad/full traffic coverage rather than a minimal safe API subset. For a simulator/NCGuide target, do not exclude a RAG-retrieved write or control API merely because it mutates simulator state, creates lifecycle side effects, is not the shortest path, or is outside the narrowest scene core. Include it when it can generate distinguishable traffic and can be bracketed by setup/read-back/cleanup or bounded simulator values.\n"
        "- When coverage_intent.full_function_coverage=true and function_coverage_manifest.enabled=true, treat function_coverage_manifest as the global FOCAS API universe from focas_funcs.xlsx. The current run may contain multiple scenario segments, not one isolated scenario and not one isolated function list. For each segment in function_coverage_manifest.segments, target_batch functions are the coverage objectives for that segment; support_batch functions are allowed to repeat because they create, verify, synchronize, or restore that segment's state. Include executable steps for each target_batch function when ABI/prototype and bounded arguments can be established; otherwise record a concrete defer/exclude reason such as missing prototype/export, unsupported controller/simulator, or unsafe/unbounded argument construction. Use support_batch APIs where needed, but do not count support-only repetition as progress on target coverage.\n"
        "- A single generated C++ script may execute multiple scenario segments sequentially. nc_program_required means a valid NC text payload is needed; only cycle_start_required means NC execution/motion is needed. Program-directory, transfer, file, parameter, tool, PMC, and option-probe segments must not use Cycle Start unless an explicit target API needs an executing program. Keep segment boundaries explicit in step_id, action, CSV data, and diagnostics.\n"
        "- For full-coverage simulator runs, treat semantically related APIs as complementary unless RAG evidence shows an exact incompatibility. For example, position families such as cnc_rdposition, cnc_absolute, cnc_machine, cnc_relative, and cnc_distance may all contribute different returned fields; lifecycle/program APIs such as cnc_rdmdiprgstat, cnc_wractpt, cnc_pdf_wractpt, cnc_setpglock, cnc_search, and cnc_rdprgnum may contribute different program-state traffic. Do not exclude one solely because another API already gives partial evidence.\n"
        "- Valid exclusion reasons in full-coverage simulator mode are narrow: absent official prototype/header evidence, DLL/export unavailable, unsupported by the simulator/controller per RAG, argument construction cannot be made bounded, API conflicts with a required lifecycle gate, or the user/permissions explicitly forbid it. 'write operation', 'unnecessary mutation', 'not MDI-focused', or 'covered by another API' are not sufficient by themselves.\n"
        "- For each selected API, create at least one executable step with protocol_function set to the exact FOCAS function name or function sequence. If no existing semantic interface name fits, create a concise interface_name such as ReadDistanceToGo or ReadAxisLoad while keeping protocol_function exact.\n"
        "- Keep each selected protocol_function consistent with its retrieved prototype and arguments. Do not label a step cnc_rdprogdir while using cnc_rdprogdir2/cnc_rdprogdir3-style top_prog/num_prog pointer parameters; choose the exact documented family member whose prototype matches the planned parameter strategy.\n"
        "- Treat download lifecycles as versioned contracts: cnc_download3 requires cnc_dwnstart3 then cnc_dwnend3; cnc_download4 requires cnc_dwnstart4 then cnc_dwnend4. A non-EW_OK start blocks the data-transfer call.\n"
        "- Expand every documented api_sequence into an independent atomic session. If a batch contains both version 3 and version 4 targets, emit two separate sessions with separate target/support roles and one normalized input/output log pair per real call; never interleave or cross-pair versions.\n"
        "- cnc_dncstart2 is a state query for an already active CNC-side DNC/M198 operation, not a command that starts DNC. Plan cnc_dnc2/cnc_dncend2 only behind an observed EW_OK result. DNC/M198 requires its own verified CNC mode/option/signal prerequisites; do not infer NCGuide incompatibility from EW_REJECT, which means no DNC/M198 was active at query time.\n"
        "- First analyze what will make the generated traffic high quality, distinguishable, and broad enough to cover the selected related API set.\n"
        "- Only for an actual NC-motion segment, use observable 2-5 second motion blocks and adequate sampling. Non-motion API quality is judged by target calls, return values, parameter variation, and matched PCAP transactions, not axis/feed variation.\n"
        "- Only if an NC-motion segment actually requires StartProgram/Cycle Start, explicitly require a trigger-readiness gate: before each Cycle Start, "
        "the generated C++ must poll relevant completion evidence from the selected API set and wait until the previous block is no longer active or until a bounded "
        "timeout is reached. In this NCGuide setup, run=1 and motion=0 means the previous Single Block is complete and the next click is ready; run=0/motion=0 is initial idle. Do not wait for run=0 between blocks. Do not plan blind consecutive Cycle Start clicks.\n"
        "- In the confirmed NCGuide Single Block behavior, every non-empty executable NC line consumes one Cycle Start, including the O program-number line, setup/modal-only lines, motion lines, and the final M30 line. Plan enough guarded Cycle Start operations to advance the entire uploaded NC payload, not merely its motion blocks.\n"
        "- Require the generated C++ to derive or embed the effective NC segment count from the exact generated payload, log expected_nc_segment_count and cycle_start_click_count, and execute through the M30 segment before the final completion gate. Non-motion segments are lifecycle/warmup evidence and must not be counted as motion samples.\n"
        "- Upload/select/program-number verification is a hard lifecycle gate. If cnc_dwnend3 or cnc_search fails, or cnc_rdprgnum does not match the uploaded O number, execution must stop before any Cycle Start and report PROGRAM_NOT_VERIFIED.\n"
        "- Before UploadProgram, require a documented program-directory/read API for exact target-program existence checks. O-number allocation belongs to CodeGenerationAgent and runtime collision handling belongs to generated C++ after it reads the actual controller directory. Do not choose or repair an O number in the plan.\n"
        "- Single target-program deletion is allowed when the generated C++ has positively identified an exact conflicting O number and deletes only that O number before upload. Delete-all remains permission-gated: consider cnc_delall only when task_permissions.allow_delete_all_programs=true and only if RAG-supported reasoning shows it is appropriate for the isolated NCGuide experiment.\n"
        "- If prior execution quality failed, read the previous failed plan/artifacts/result and modify that design; do not ignore it and start from scratch.\n"
        "- Do not output concrete NC blocks such as G01 X...; output block_goals and constraints instead.\n"
        "- For every executable API step, describe input constraints, not concrete test values. "
        "For each input identify its semantic role, documented domain source, bounds/enum constraints in words, "
        "dependencies on other inputs, whether it is fixed by the ABI or discoverable at runtime, and evidence. "
        "Do not emit parameter_generation.value/values/min/max/samples or concrete parameter_cases; "
        "CodeGenerationAgent derives safe values from the official contract and discovers device-dependent values. "
        "Declare coupled parameter_constraints as relationships that generated cases must preserve, not as value tuples. "
        "For multi-API steps identify the owning function on each constraint. "
        "Input coverage and returned-output variation are audited separately; repeated inputs are one combination.\n"
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
        "\"quality_targets\":{\"input_diversity_required\":false,\"response_state_diversity_required\":false,\"response_variation_strategy\":\"only when target classification requires it\"},"
        "\"coverage_segments\":[{\"segment_id\":\"01_programmed_coordinate_motion\",\"main_state_driver\":\"driver\","
        "\"nc_program_required\":true,\"target_functions\":[\"cnc_absolute\"],\"support_functions\":[\"cnc_statinfo\"],"
        "\"quality_target\":\"what this segment proves\"}],"
        "\"nc_program_spec\":{\"purpose\":\"short purpose\","
        "\"block_goals\":[\"goal\"],\"constraints\":[\"constraint\"],\"generation_notes\":[\"note\"]},"
        "\"steps\":[{\"step_id\":\"S001\",\"phase\":\"before\",\"action\":\"read status\","
        "\"interface_name\":\"ReadRunStatus\",\"protocol_function\":\"cnc_statinfo\",\"parameters\":{},"
        "\"parameter_generation\":[{\"name\":\"type\",\"kind\":\"enum\",\"constraints\":[\"values documented by target API; generator resolves exact set\"],\"evidence\":\"API docs\"},"
        "{\"name\":\"axis\",\"kind\":\"discovered\",\"constraints\":[\"must be a controlled axis\"],\"discovery\":{\"source\":\"device observation\"}}],"
        "\"parameter_constraints\":[{\"relation\":\"length must equal buffer capacity required by ABI\",\"inputs\":[\"length\",\"buffer\"],\"evidence\":\"SDK/API contract\"}],"
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
        f"Evidence-scoped target scene rules={json.dumps(plan.rag_context.get('scene_plans', []), ensure_ascii=False, separators=(',', ':'))}\n"
        f"Explicit task permissions={plan.rag_context.get('task_permissions', {})}\n"
        f"Current steps={[{'id': s.step_id, 'phase': s.phase, 'interface': s.interface_name, 'repeat': s.repeat, 'interval': s.interval_seconds, 'action': s.action} for s in plan.steps]}\n"
        f"RAG context={compact_rag_context(plan.rag_context)}\n"
        f"Official ABI context for candidate APIs:\n{plan.rag_context.get('official_abi_context', '')}\n"
        f"Prior repair history and quality failures={plan.rag_context.get('repair_context', [])}\n"
        f"Retrieved error-experience knowledge (use only entries whose status/evidence fit this task; treat hypotheses as conditional): {plan.rag_context.get('long_term_memories', [])}\n"
    )
    manifest_context = plan.rag_context.get("function_coverage_manifest")
    if isinstance(manifest_context, dict) and manifest_context.get("enabled") and not any(
        segment.get("cycle_start_required") and segment.get("target_batch")
        for segment in manifest_context.get("segments", [])
        if isinstance(segment, dict)
    ):
        system_prompt += (
            "\n# Binding for this selected API batch\n"
            "This batch contains no API requiring NC axis motion. Do not add StartProgram, "
            "Cycle Start, motion sampling, program_completion_gate, or feed/coordinate quality targets. "
            "A valid NC text buffer may still be needed for a transfer/verify API. "
            "Execute and assess the target APIs directly, with matched PCAP traffic and returned fields.\n"
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
    compact_context = compact_rag_context(plan.rag_context)
    rag_text = json.dumps(compact_context, ensure_ascii=False, separators=(",", ":"))
    memory_text = json.dumps(plan.rag_context.get("long_term_memories", []), ensure_ascii=False)
    task_context_text = (
        f"{task_description}\n{plan.scenario_type}\n{plan.scenario_goal}\n"
        f"goals={plan.nc_program_spec.block_goals}\nconstraints={plan.nc_program_spec.constraints}\n"
        f"steps={[{'id': s.step_id, 'phase': s.phase, 'interface': s.interface_name, 'protocol_function': s.protocol_function, 'repeat': s.repeat, 'interval_seconds': s.interval_seconds, 'action': s.action} for s in plan.steps]}"
    )
    diagnostics_context = {
        "rag_tokens_est": len(rag_text) // 4,
        "memory_tokens_est": len(memory_text) // 4,
        "task_context_tokens_est": len(task_context_text) // 4,
        "retrieved_chunks": len(plan.retrieved_chunk_ids),
        "deduplicated_chunk_count": len(set(plan.retrieved_chunk_ids)),
    }
    # Planner owns only the structured decision. ABI declarations and C++ /
    # capture implementation details are intentionally deferred to codegen.
    system_prompt = (
        "You are PlanningAgent for a FANUC FOCAS traffic-generation task. "
        "Return Markdown using exactly the supplied headings and key/value fields. "
        "Decide scenario_goal, quality_targets, "
        "NC program goals/constraints, API selection and exact order, before/during/after timing, "
        "safe parameter strategies, and execution constraints. Use only the supplied RAG evidence. "
        "You OWN parameter-domain and coverage-constraint planning for EVERY target API; CodeGenerationAgent chooses "
        "and implements concrete test values from those constraints and official evidence. "
        "For each real input use parameter_generation to describe kind, semantic role, constraints, evidence, "
        "dependencies, and discovery method when needed. Do not put concrete values in the plan. "
        "Do not guess device-specific axes, program numbers, parameter numbers or write values. "
        "Include evidence/source, constraints and dependencies for each strategy. "
        "Use parameter_constraints for coupled-input relationships, never concrete value tuples. "
        "Specify parameter_coverage_goal with "
        "minimum_distinct, coverage_scope, evidence, restore_strategy and limitation handling. "
        "Use historical_diversity_feedback in Explicit task permissions as prior observations, not as current device state. "
        "Exclude previously sampled valid input configurations when choosing new cases, unless replay is needed to verify reproducibility. "
        "Set a bounded maximum configuration budget and distinguish the planned target count from the hard execution budget. "
        "For independent inputs prefer controlled one-factor-at-a-time cases; for coupled inputs specify valid combinations and their relation. "
        "Name output fields expected to change and fields expected to remain stable; record a source-backed reason when response-state variation is not applicable. "
        "Escalate response variation only through the lowest-cost supported strategy: input change, existing business data, authorized isolated data setup, validated NC execution, then supported event simulation. "
        "Never treat low diversity as execution failure, never retry unchanged configurations, and stop when the documented domain, planned budget, or verified environment capability is exhausted. "
        "Do not default minimum_distinct to 1 for a documented multi-value input domain: the parameter "
        "coverage contract enforces multiple successful combinations for axis inputs that allow ALL_AXES "
        "and controller-reported axis numbers. Use 1 only for callable-only coverage or a documented "
        "single-value domain; do not lower an enforced contract because discovery is missing. "
        "Repeated identical calls are NOT different combinations. Handles, timestamps and output pointers "
        "are not diversity dimensions. For no-variable-input APIs plan state_coverage with meaningful "
        "device states, transitions and expected output variation or explain why variation is inapplicable. "
        "Classify every target explicitly with variation_class=call_only|input_only|response_observation_only|response_state_only|input_and_response, plus input_diversity {applicable,required,reason} and response_state_diversity {applicable,required,reason}. Enforce input diversity only when the documented input domain is applicable; enforce response-state diversity only when the returned data depends on an observable state/event. For call-only and inherently stable APIs require only a real successful call and matched traffic. "
        "For every target, decide whether output diversity needs an NC program and whether that program must execute; justify this from output dependencies, not generic templates. cnc_skip is a read API: vary its documented axis selector using controller-reported axis count; it does not require a G31 program or SKIP signal to call. For runtime-motion targets, if execution gates are unverified, retain a read-only baseline, omit upload/select/start steps, and record a deferred manual Cycle Start action before any state-changing operation. Never infer execution from a plan. Identify required CNC mode/parameter conditions only from exact applicable evidence. Observe mode first; on mismatch request operator action and stop that target, never switch automatically. Parameter writes require exact sourced number/bit/value/scope, explicit allow_environment_writes permission, snapshot/write/read-back, and stop on mismatch or restart requirement. Unknown requirements remain unknown and block only dependent state-changing operations while retaining safe read-only samples. Never request Cycle Start just to increase diversity. "
        "For cnc_skip, plan cnc_rdaxisname discovery and aggregate plus per-axis reads; no G31/SKIP event is a prerequisite for the API call. "
        "Do not delegate selection of coverage dimensions to CodeGenerationAgent, but delegate concrete value "
        "selection within the documented constraints. Do not generate unsafe "
        "Cartesian products; state changes require snapshot, verification and restoration. "
        "Do not write C++, G-code blocks, Fwlib declarations, capture code, validators, or full API documentation. "
        "Do not omit required headings. Do not wrap the whole response in a code fence."
    )
    user_prompt = (
        f"Task:\n{task_description}\n\n"
        f"Scenario={plan.scenario_type}\n"
        f"Initial plan={[
            {'id': s.step_id, 'phase': s.phase, 'interface': s.interface_name,
             'protocol_function': s.protocol_function, 'repeat': s.repeat,
             'interval_seconds': s.interval_seconds, 'action': s.action}
            for s in plan.steps
        ]}\n"
        f"NC goals={plan.nc_program_spec.block_goals}\n"
        f"NC constraints={plan.nc_program_spec.constraints}\n"
        f"RAG evidence={json.dumps(compact_context, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Return this fixed Markdown structure. Values after parameters, parameter_generation, parameter_constraints, parameter_coverage_goal and state_coverage "
        "must be compact JSON values; all other fields are normal Markdown text or JSON scalars/lists.\n\n"
        "# Scenario Goal\n"
        "goal: <short goal>\n\n"
        "# Quality Analysis\n"
        "- traffic_goal: <what returned outputs must vary>\n"
        "- risk: <main risk>\n"
        "- strategy: [\"strategy 1\", \"strategy 2\"]\n\n"
        "# Quality Targets\n"
        "- input_diversity_required: false\n"
        "- response_state_diversity_required: false\n"
        "- max_configurations: 1\n"
        "- response_fields_expected_to_change: []\n"
        "- response_fields_expected_stable: []\n"
        "- response_variation_strategy: \"only when classification requires it\"\n\n"
        "# NC Program\n"
        "- purpose: <short purpose>\n"
        "- block_goals: [\"goal 1\", \"goal 2\"]\n"
        "- constraints: [\"constraint 1\"]\n"
        "- generation_notes: []\n\n"
        "# Scene Requirements\n"
        "- item: {\"target_api\":\"<exact target>\",\"variation_class\":\"call_only|input_only|response_observation_only|response_state_only|input_and_response\",\"input_diversity\":{\"applicable\":false,\"required\":false,\"reason\":\"...\"},\"response_state_diversity\":{\"applicable\":false,\"required\":false,\"reason\":\"...\"},\"output_dependencies\":[\"<field and dependency>\"],\"nc_program_required\":false,\"execution_required\":false,\"nc_behavior\":[],\"required_mode\":\"unknown or documented mode\",\"mode_evidence\":\"<exact source or unknown>\",\"parameter_preconditions\":[],\"operator_action_if_mismatch\":true,\"safe_fallback\":\"<read-only sample or stop reason>\"}\n\n"
        "# Steps\n"
        "## S001\n"
        "- phase: before\n"
        "- action: <action>\n"
        "- interface_name: <interface>\n"
        "- protocol_function: <exact FOCAS function>\n"
        "- parameters: {}\n"
        "- parameter_generation: [{\"name\":\"<actual input name>\",\"kind\":\"fixed|enum|range|discovered\",\"reason\":\"<coverage purpose>\",\"evidence\":\"<RAG source>\",\"dependencies\":[],\"constraints\":[\"domain rule; no concrete values\"]}]\n"
        "- parameter_constraints: [{\"relation\":\"<relationship between inputs>\",\"inputs\":[\"<name>\"],\"evidence\":\"<source>\"}]\n"
        "- parameter_coverage_goal: {\"minimum_distinct\":1,\"coverage_scope\":\"<evidence-backed valid domain>\",\"evidence\":\"<source and applicability>\",\"restore_strategy\":\"<if state changing>\",\"on_insufficient\":\"retain samples and report limitation\"}\n"
        "- state_coverage: {}\n"
        "- repeat: 1\n"
        "- interval_seconds: 0.2\n"
        "- expected_state: <expected state>\n\n"
        "Repeat one ## step block per API operation. Optional final sections are "
        "# API Coverage Analysis, # Coverage Segments, and # Notes; put each object in the first two "
        "as '- item: {compact JSON object}' and each note as '- note: text'."
    )
    body_bytes_est = len(json.dumps({"model": "", "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]}, ensure_ascii=False).encode("utf-8"))
    print(
        f"[PlanningAgent] prompt input_tokens_est={(len(system_prompt) + len(user_prompt)) // 4} "
        f"body_bytes={body_bytes_est} "
        f"rag_tokens_est={diagnostics_context['rag_tokens_est']} "
        f"memory_tokens_est={diagnostics_context['memory_tokens_est']} "
        f"task_context_tokens_est={diagnostics_context['task_context_tokens_est']} "
        f"retrieved_chunks={diagnostics_context['retrieved_chunks']} "
        f"deduplicated_chunk_count={diagnostics_context['deduplicated_chunk_count']}",
        flush=True,
    )
    response_text = llm_client.invoke_text(
        system_prompt,
        user_prompt,
        agent_name="PlanningAgent",
        diagnostics_context=diagnostics_context,
    )
    payload = parse_planning_markdown(response_text)
    # Keep the structured contract usable when a provider omits the NC section
    # or returns only part of it.  The initial deterministic plan is already
    # evidence-scoped; it is a safe source for missing scaffolding, while the
    # LLM still owns target selection and operation planning.
    nc_spec_payload = payload.get("nc_program_spec")
    if not isinstance(nc_spec_payload, dict):
        nc_spec_payload = {}
    explicit_fallback_targets = {
        str(name).strip()
        for name in plan.rag_context.get("explicit_target_functions", [])
        if str(name).strip()
    }
    returned_step_targets = {
        name
        for row in payload.get("steps", [])
        if isinstance(row, dict)
        for name in protocol_function_names_for_planner(str(row.get("protocol_function", "")))
    }
    manifest_fallback_targets = {
        str(row.get("function", "")).strip()
        for row in plan.rag_context.get("function_coverage_manifest", {}).get("selected_batch", [])
        if isinstance(row, dict) and row.get("coverage_role", "target") == "target" and row.get("function")
    }
    fallback_targets = explicit_fallback_targets | manifest_fallback_targets | returned_step_targets
    fallback_motion_required = motion_required_for_targets(fallback_targets) if fallback_targets else motion_required_for_targets(
        name for step in plan.steps for name in protocol_function_names_for_planner(step.protocol_function)
    )
    fallback_spec = {
        "purpose": (
            plan.nc_program_spec.purpose
            if fallback_motion_required
            else "No NC program is required for the selected API batch; preserve read-only execution constraints."
        ),
        "block_goals": list(plan.nc_program_spec.block_goals) if fallback_motion_required else [],
        "constraints": list(plan.nc_program_spec.constraints) if fallback_motion_required else [],
        "generation_notes": list(plan.nc_program_spec.generation_notes) if fallback_motion_required else [],
    }
    if not fallback_spec["block_goals"]:
        fallback_spec["block_goals"] = (
            ["follow the validated NC execution scene", "end the program with M30"]
            if fallback_motion_required else ["no NC program or Cycle Start is required for this API batch"]
        )
    if not fallback_spec["constraints"]:
        fallback_spec["constraints"] = [
            "do not execute NC motion unless the target classification requires it",
            "preserve AUTO mode and the existing CNC state",
        ]
    if not fallback_spec["generation_notes"]:
        fallback_spec["generation_notes"] = [
            "Planner field fallback was filled from the deterministic initial plan because the provider omitted part of the NC spec."
        ]
    missing_nc_fields = []
    for key, value in fallback_spec.items():
        if not nc_spec_payload.get(key):
            nc_spec_payload[key] = value
            missing_nc_fields.append(key)
    payload["nc_program_spec"] = nc_spec_payload
    if missing_nc_fields:
        plan.llm_notes.append(
            "Planner Markdown omitted NC spec fields; deterministic fallback filled: "
            + ", ".join(missing_nc_fields)
        )
    if not payload.get("quality_analysis"):
        payload["quality_analysis"] = {
            "traffic_goal": "capture the selected target and its evidence-backed support calls",
            "risk": "response variation may be inapplicable or environment-gated for this target",
            "strategy": ["apply the target variation classification", "retain real calls and matched traffic"],
        }
        plan.llm_notes.append("Planner Markdown omitted quality_analysis; deterministic fallback filled it.")
    if not payload.get("quality_targets"):
        payload["quality_targets"] = {
            "input_diversity_required": False,
            "response_state_diversity_required": False,
            "response_variation_strategy": "only when target classification requires it",
        }
        plan.llm_notes.append("Planner Markdown omitted quality_targets; deterministic fallback filled it.")
    required_planning_keys = (
        ("scenario_goal", "nc_program_spec", "steps")
        if code_only
        else ("scenario_goal", "quality_analysis", "quality_targets", "nc_program_spec", "steps")
    )
    missing_planning_keys = [key for key in required_planning_keys if not payload.get(key)]
    nc_spec_payload = payload.get("nc_program_spec")
    if isinstance(nc_spec_payload, dict):
        for nested_key in ("purpose", "block_goals", "constraints"):
            if not nc_spec_payload.get(nested_key):
                missing_planning_keys.append(f"nc_program_spec.{nested_key}")
    if missing_planning_keys:
        raise ValueError(
            "PlanningAgent Markdown is missing required fields: "
            + ", ".join(missing_planning_keys)
        )

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
    scene_requirements = payload.get("scene_requirements", [])
    if isinstance(scene_requirements, list):
        plan.rag_context["planner_scene_requirements"] = [
            item for item in scene_requirements if isinstance(item, dict)
        ]
    if "function_coverage_manifest" in plan.rag_context:
        # The selected function manifest is authoritative. A model must not
        # invent a motion segment or change a non-motion segment into one.
        plan.rag_context["coverage_segments"] = coverage_segments_from_manifest_context(
            plan.rag_context["function_coverage_manifest"]
        )
    elif isinstance(coverage_segments, list):
        plan.rag_context["coverage_segments"] = normalize_coverage_segments(coverage_segments)

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
                agent_name="PlanningAgent",
            )
            if isinstance(repaired_steps, dict):
                candidate_steps = repaired_steps.get("steps", [])
                if isinstance(candidate_steps, list):
                    plan.steps = plan_steps_from_llm_rows(candidate_steps)
        if not plan.steps:
            raise ValueError("PlanningAgent LLM response did not include any valid steps.")
    if not code_only:
        plan.steps = materialize_included_api_candidates(plan)
    # Enforce the same motion prerequisite used by ExecutionContract. A target
    # such as cnc_rdaxisdata may be selected inside the comprehensive batch,
    # where the scenario name alone is not coordinate_motion. In that case the
    # plan must still contain the NC upload/select/verify/Cycle Start lifecycle.
    permissions = plan.rag_context.get("task_permissions", {})
    planned_targets = set(permissions.get("coverage_current_batch_functions") or ())
    if not planned_targets:
        manifest = plan.rag_context.get("function_coverage_manifest", {})
        planned_targets = {
            str(row.get("function", ""))
            for row in manifest.get("selected_batch", [])
            if isinstance(row, dict) and row.get("coverage_role", "target") == "target"
        }
    if motion_required_for_targets(planned_targets):
        ensure_nc_program_lifecycle_steps("coordinate_motion", plan.steps, plan.nc_program_spec.program_name)
        plan.rag_context["motion_required_by_targets"] = sorted(planned_targets & MOTION_TARGET_APIS)
    if code_only:
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
        restore_semantic_api_bindings(plan.steps, semantic_baseline, plan.llm_notes)
    else:
        # The model owns the scenario plan and NC specification, while these
        # four infrastructure steps are mandatory execution plumbing for an
        # NC program.  Enforce them after parsing so a model response cannot
        # omit the upload/select/verify/UI-start path and trap Router in a
        # repair loop.
        if "function_coverage_manifest" in plan.rag_context:
            motion_required = motion_required_for_targets(planned_targets) or any(
                segment.get("cycle_start_required") and segment.get("execution_allowed", True)
                and segment.get("target_functions")
                for segment in plan.rag_context.get("coverage_segments", [])
            )
            if motion_required:
                ensure_nc_program_lifecycle_steps("coordinate_motion", plan.steps)
            else:
                plan.steps = [
                    step for step in plan.steps
                    if step.interface_name not in {"StartProgram", "StopProgram"}
                ]
                plan.rag_context["quality_targets"] = {
                    **plan.rag_context.get("quality_targets", {}),
                    "expect_feed_variation": False,
                    "expect_position_variation": False,
                    "require_program_completion": False,
                    "require_target_api_traffic": True,
                }
                plan.llm_notes.append("Selected API batch has no motion target; Cycle Start and motion-only quality gates removed.")
        else:
            ensure_nc_program_lifecycle_steps(
                plan.scenario_type,
                plan.steps,
                plan.nc_program_spec.program_name,
            )
        attach_protocol_functions(plan.steps)
    ensure_tool_management_operation(plan)
    if representative_scenario:
        plan.steps = compact_representative_steps(plan.steps, max_steps=18)
        plan.llm_notes.append(
            "Representative benchmark retained RAG-selected interfaces without TOOL_REGISTRY admission filtering."
        )
    # Compaction is allowed to reduce breadth, but never to remove the
    # Ground Truth tool-data operation from the final Planner hand-off.
    ensure_tool_management_operation(plan)
    attach_protocol_functions(plan.steps)

    notes = payload.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if isinstance(notes, list):
        for note in notes:
            text = str(note).strip()
            if text:
                plan.llm_notes.append(f"llm_planning_generation: {text}")
    strip_planner_program_number_decisions(plan.nc_program_spec, plan.steps)


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
        f"Retrieved error-experience memories (reference prior outcomes; apply only when trigger and evidence match, and preserve uncertainty):\n{plan.rag_context.get('long_term_memories', [])}\n\n"
        f"Current parsed steps:\n{current}\n\n"
        "Audit explicit task operations and return any omitted atomic API steps."
    )
    try:
        payload = llm_client.invoke_json(prompt, user_prompt, agent_name="PlanningAgent")
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
        program_name="",
        purpose=purpose,
        block_goals=block_goals,
        constraints=constraints,
        generation_notes=generation_notes,
    )


def strip_planner_program_number_decisions(nc_program_spec: NcProgramSpec, steps: list[PlanStep]) -> None:
    """Keep concrete O-number allocation out of the Planner contract."""
    nc_program_spec.program_name = ""
    program_keys = {
        "program_name",
        "program_number",
        "preferred_program_number",
        "selected_program_number",
        "target_program",
    }
    for step in steps:
        for key in program_keys:
            step.parameters.pop(key, None)
        strategies = step.parameters.get("parameter_generation")
        if isinstance(strategies, list):
            step.parameters["parameter_generation"] = [
                row
                for row in strategies
                if not isinstance(row, dict) or str(row.get("name", "")) not in program_keys
            ]


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
        protocol_function = str(row.get("protocol_function", "")).strip()
        if protocol_function:
            try:
                from ..focas_contracts import documented_parameter_names
                api_input_names = {name.lower() for name in documented_parameter_names(protocol_function)}
            except (KeyError, ValueError):
                api_input_names = set()
            # API argument values are selected by CodeGenerationAgent from the
            # active SDK, docs and runtime discovery, never copied from a plan.
            for key in list(parameters):
                if key.lower() in api_input_names and key not in {
                    "parameter_generation", "parameter_constraints", "parameter_cases",
                    "parameter_coverage_goal", "state_coverage",
                }:
                    parameters.pop(key)
        parameter_generation = row.get("parameter_generation", row.get("parameter_strategy"))
        if isinstance(parameter_generation, list):
            parameters["parameter_generation"] = normalize_parameter_generation(parameter_generation)
        if isinstance(row.get("parameter_cases"), list):
            # The Planner specifies relations; concrete input tuples belong to
            # code generation after resolving the SDK/API domain.
            parameters["parameter_constraints"] = normalize_parameter_constraints(
                row.get("parameter_constraints", row["parameter_cases"])
            )
        elif isinstance(row.get("parameter_constraints"), list):
            parameters["parameter_constraints"] = normalize_parameter_constraints(row["parameter_constraints"])
        for key in ("parameter_coverage_goal", "state_coverage"):
            if isinstance(row.get(key), dict):
                parameters[key] = row[key]
        phase = phase_or_default(row.get("phase"))
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
        if not name or kind not in {"fixed", "enum", "range", "discovered"}:
            continue
        item: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "reason": str(row.get("reason", "")).strip(),
        }
        if row.get("function"):
            item["function"] = str(row["function"])
        for key in ("evidence", "constraints", "dependencies", "discovery", "selection_rule", "on_insufficient"):
            if key in row:
                item[key] = row[key]
        strategies.append(item)
    return strategies


def normalize_parameter_constraints(rows: list[Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        relation = str(row.get("relation", row.get("constraint", ""))).strip()
        if not relation:
            continue
        item = {"relation": relation}
        for key in ("function", "inputs", "evidence", "source", "applicability"):
            if key in row:
                item[key] = row[key]
        constraints.append(item)
    return constraints


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


def materialize_included_api_candidates(plan: ExecutionPlan) -> list[PlanStep]:
    """Turn evidence-backed Planner include decisions into executable steps."""
    candidates = {
        str(row.get("protocol_function", "")).strip()
        for row in plan.rag_context.get("api_candidate_pool", [])
        if isinstance(row, dict) and str(row.get("protocol_function", "")).strip()
    }
    abi_context = str(plan.rag_context.get("official_abi_context", ""))
    abi_functions = set(re.findall(r"\b(cnc_[A-Za-z0-9_]+|pmc_[A-Za-z0-9_]+)\s*\(", abi_context))
    existing = {
        name
        for step in plan.steps
        for name in protocol_function_names_for_planner(step.protocol_function)
    }
    added: list[PlanStep] = []
    for row in plan.rag_context.get("api_coverage_analysis", []):
        if not isinstance(row, dict) or str(row.get("decision", "")).strip().lower() != "include":
            continue
        name = str(row.get("protocol_function", "")).strip()
        if not name or name not in candidates or name not in abi_functions or name in existing:
            continue
        relationship = str(row.get("relationship", "")).strip()
        role = str(row.get("coverage_role", "")).strip()
        action = f"Capture documented API {name}"
        if relationship:
            action += f": {relationship}"
        if role:
            action += f" ({role})"
        added.append(PlanStep(
            step_id=f"RAG-API-{len(added) + 1:03d}",
            phase="during",
            action=action,
            interface_name=f"FOCAS_{name}",
            protocol_function=name,
            operation_kind="focas_api",
        ))
        existing.add(name)
    if added:
        plan.llm_notes.append(
            f"Materialized {len(added)} Planner-included API candidate(s) after SDK/RAG evidence validation."
        )
    return [*plan.steps, *added]


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
                "cycle_start_required": bool(row.get("cycle_start_required", False)),
                "execution_allowed": bool(row.get("execution_allowed", True)),
                "execution_gate": str(row.get("execution_gate", "")),
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
                "cycle_start_required": bool(segment.get("cycle_start_required", False)),
                "execution_allowed": bool(segment.get("execution_allowed", True)),
                "execution_gate": str(segment.get("execution_gate", "")),
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
        payload = llm_client.invoke_json(system_prompt, user_prompt, agent_name="PlanningAgent")
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
        if key == "exact_api_contracts" and isinstance(rows, list):
            compact[key] = [
                {
                    "function": row.get("function"),
                    "chunk_type": row.get("chunk_type"),
                    "source": row.get("source_url") or row.get("chunk_id"),
                    "text": str(row.get("text", ""))[:6000],
                }
                for row in rows[:48]
                if isinstance(row, dict)
            ]
            continue
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
                        "cycle_start_required": segment.get("cycle_start_required"),
                        "target_functions": [
                            row.get("function")
                             for row in segment.get("target_batch", [])[:12]
                            if isinstance(row, dict)
                        ],
                        "support_functions": [
                            row.get("function")
                             for row in segment.get("support_batch", [])[:8]
                            if isinstance(row, dict)
                        ],
                    }
                     for segment in rows.get("segments", [])[:2]
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
                     for row in rows.get("selected_batch", [])[:24]
                    if isinstance(row, dict)
                ],
            }
            continue
        if key == "official_abi_context":
            # ABI details are consumed by CodeSpec/code generation, not the
            # planner's scenario/API/timing decision prompt.
            compact[key] = "omitted_for_planner_prompt"
            continue
        if key == "api_candidate_pool" and isinstance(rows, list):
            compact[key] = [
                {
                    "protocol_function": row.get("protocol_function"),
                    "sources": row.get("sources", [])[:3],
                    "relationships": [str(item)[:180] for item in row.get("relationships", [])[:3]],
                }
                for row in rows[:32]
                if isinstance(row, dict)
            ]
            continue
        if isinstance(rows, dict):
            compact[key] = compact_prompt_mapping(rows, max_depth=2)
            continue
        if not isinstance(rows, list):
            compact[key] = str(rows)[:240]
            continue
        row_limit = 12 if key == "api" else 2
        compact[key] = [
            {
                "rule_type": retrieval_row_metadata(row).get("rule_type"),
                "scenario": retrieval_row_metadata(row).get("scenario"),
                "function": retrieval_row_metadata(row).get("function"),
                "preview": str(retrieval_row_preview(row))[:260],
            }
            for row in rows[:row_limit]
            if isinstance(row, (dict, RetrievedChunk))
        ]
    return compact


def retrieval_row_metadata(row: dict[str, Any] | RetrievedChunk) -> dict[str, Any]:
    if isinstance(row, RetrievedChunk):
        return row.chunk.metadata
    return row


def retrieval_row_preview(row: dict[str, Any] | RetrievedChunk) -> str:
    if isinstance(row, RetrievedChunk):
        return row.chunk.text
    return str(row.get("text_preview", row.get("preview", "")))


def compact_prompt_mapping(value: dict[str, Any], *, max_depth: int, max_items: int = 24) -> dict[str, Any]:
    """Bound nested RAG metadata before it is interpolated into an LLM prompt."""
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_items:
            break
        if isinstance(item, str):
            result[str(key)] = item[:800]
        elif isinstance(item, list):
            result[str(key)] = [str(part)[:400] for part in item[:12]]
        elif isinstance(item, dict) and max_depth > 0:
            result[str(key)] = compact_prompt_mapping(item, max_depth=max_depth - 1, max_items=12)
        else:
            result[str(key)] = item
    return result


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

    for function_name in plan.rag_context.get("explicit_target_functions", []) or []:
        add(function_name, "explicit_target", "explicitly selected by task")

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
                "record_type": memory.get("record_type", "workflow_repair_memory"),
                "experience_id": memory.get("experience_id"),
                "title": str(memory.get("title", ""))[:180],
                "status": memory.get("status"),
                "confidence": memory.get("confidence"),
                "trigger": str(memory.get("trigger", ""))[:320],
                "root_cause": str(memory.get("root_cause", ""))[:400],
                "evidence": [str(item)[:240] for item in memory.get("evidence", [])[:3]],
                "fix": str(memory.get("fix", ""))[:500],
                "prevention": str(memory.get("prevention", ""))[:320],
                "related_apis": memory.get("related_apis", [])[:12],
                "limitations": str(memory.get("limitations", ""))[:320],
                "final_errors": [str(error)[:240] for error in memory.get("final_errors", [])[:3]],
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
        # The Tool Management benchmark description may mention an NC
        # context, but its required semantic operation is tool-data access.
        # Classify this explicit scene before the generic program lifecycle.
        if ("tool" in joined and "management" in joined) or "刀具管理" in joined:
            return "tool_management"
        return "program_lifecycle"
    if terms.intersection({"宏", "macro"}) or "宏" in joined:
        return "macro_variable_read_write"
    if terms.intersection({"刀具", "tool", "offset"}) or "刀具" in joined:
        if "tool" in joined and "management" in joined:
            return "tool_management"
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
    strip_planner_program_number_decisions(nc_program_spec, steps)
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
    if scenario == "coordinate_motion":
        return NcProgramSpec(
            program_name="",
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
            program_name="",
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
            program_name="",
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
            program_name="",
            purpose="Ask CodeGenerationAgent to create a short executable program for upload, selection, start, and completion traffic.",
            block_goals=["set absolute mode", "perform one safe short operation", "dwell briefly", "end the program"],
            constraints=["valid program number", "safe short execution", "end with M30"],
        )
    return NcProgramSpec(
        program_name="",
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
    if scenario not in {"coordinate_motion", "spindle_speed_change"}:
        return
    missing_prefix: list[PlanStep] = []
    from ..execution_contracts import normalize_lifecycle_steps, step_functions
    functions = set().union(*(step_functions(step) for step in steps))
    if not any(step.interface_name == "UploadProgram" for step in steps) and not {
        "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"
    }.issubset(functions):
        missing_prefix.append(
            PlanStep("NC-001", "before", "upload generated NC program", "UploadProgram", {})
        )
    if not any(step.interface_name == "SelectProgram" for step in steps) and "cnc_search" not in functions:
        missing_prefix.append(
            PlanStep("NC-002", "before", "select generated NC program", "SelectProgram", {})
        )
    if not any(step.interface_name == "ReadProgramNumber" for step in steps) and "cnc_rdprgnum" not in functions:
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
                {"configuration_source": "NCGUIDE_* environment variables"},
                interval_seconds=0.5,
            )
        )
    if missing_prefix:
        steps[:0] = missing_prefix
    normalize_lifecycle_steps(steps)


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


SEMANTIC_INTERFACE_TO_FOCAS = {
    "ReadRunStatus": "cnc_statinfo",
    "ReadPosition": "cnc_rdposition",
    "ReadFeedSpeed": "cnc_actf",
    "ReadSpindleSpeed": "cnc_acts",
    "ReadParameter": "cnc_rdparam",
    "ReadAlarm": "cnc_alarm2",
}

TOOL_DATA_ACCESS_APIS = (
    "cnc_rdtool",
    "cnc_rdtofs",
    "cnc_rdtofsr",
    "cnc_rdtofsinfo",
    "cnc_rdtofsinfo2",
)


def ensure_tool_management_operation(plan: ExecutionPlan) -> None:
    """Keep the Ground Truth tool-data operation in every Tool Management plan."""
    if plan.scenario_type != "tool_management":
        return
    for step in plan.steps:
        names = protocol_function_names_for_planner(step.protocol_function)
        if step.interface_name == "ToolDataAccess" and any(name in TOOL_DATA_ACCESS_APIS for name in names):
            return
    plan.steps.append(
        PlanStep(
            "TM-TOOL-DATA",
            "during",
            "read tool-management table data",
            "ToolDataAccess",
            {"start_tool": 1, "tool_count": 1},
            expected_state="read one bounded tool-management record",
            protocol_function="cnc_rdtool",
            operation_kind="focas_api",
        )
    )
    plan.llm_notes.append("Tool Management semantic coverage enforced: ToolDataAccess -> cnc_rdtool.")


def restore_semantic_api_bindings(
    steps: list[PlanStep],
    baseline: dict[str, PlanStep],
    notes: list[str] | None = None,
) -> None:
    """Restore canonical semantic APIs while retaining unrelated legal APIs."""
    by_interface = {step.interface_name: step for step in steps}
    for interface_name, protocol_function in SEMANTIC_INTERFACE_TO_FOCAS.items():
        baseline_step = baseline.get(interface_name)
        current_step = by_interface.get(interface_name)
        if current_step is None and baseline_step is not None:
            current_step = PlanStep(
                step_id=f"SEM-{interface_name}",
                phase=baseline_step.phase,
                action=baseline_step.action,
                interface_name=interface_name,
                parameters=dict(baseline_step.parameters),
                repeat=baseline_step.repeat,
                interval_seconds=baseline_step.interval_seconds,
                expected_state=baseline_step.expected_state,
                protocol_function=protocol_function,
                operation_kind=baseline_step.operation_kind,
            )
            steps.append(current_step)
            by_interface[interface_name] = current_step
            if notes is not None:
                notes.append(f"Restored semantic binding {interface_name} -> {protocol_function} after LLM plan parsing.")
        if current_step is None:
            continue
        if current_step.protocol_function != protocol_function:
            current_step.protocol_function = protocol_function
            current_step.api_calls = []
            if notes is not None:
                notes.append(f"Canonicalized semantic binding {interface_name} -> {protocol_function}.")
        current_step.api_calls = normalize_api_calls([], protocol_function, current_step.parameters)


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
        "tool_management": tool_management_steps,
        "tool_offset_setting": tool_offset_steps,
        "macro_variable_read_write": macro_steps,
        "general_status_collection": status_steps,
    }
    return builders.get(scenario, status_steps)()


def coordinate_motion_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {}),
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
        PlanStep("NC-001", "before", "upload generated NC program", "UploadProgram", {}),
        PlanStep("NC-002", "before", "select generated NC program", "SelectProgram", {}),
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
        PlanStep("S001", "before", "upload comprehensive NC program", "UploadProgram", {}),
        PlanStep("S002", "before", "select comprehensive NC program", "SelectProgram", {}),
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
        PlanStep("S001", "before", "upload spindle NC program", "UploadProgram", {}),
        PlanStep("S002", "before", "select spindle NC program", "SelectProgram", {}),
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
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {}),
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


def tool_management_steps() -> list[PlanStep]:
    """Plan the Ground Truth tool-data operation with its official ABI API."""
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep(
            "S002",
            "during",
            "read tool-management table data",
            "ToolDataAccess",
            {"start_tool": 1, "tool_count": 1},
            expected_state="read one bounded tool-management record",
            protocol_function="cnc_rdtool",
        ),
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
        "tool_management": "Generate traffic for tool-management table or tool-offset data access.",
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
