from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import os
import re
import secrets
from typing import Any

from ..integrations.ncguide import default_focas_header_dir, default_focas_runtime_dir
from ..focas_harness import render_focas_harness_template
from ..focas_contracts import named_contract_context, validate_return_code_labels
from ..focas_capture_runtime import (
    capture_functions, guarded_core, render_capture_harness, validate_capture_fragment,
    wrap_capture_fragment,
)
from ..execution_contracts import SESSION_APIS, execution_contract
from ..orchestration.roles import ActionSpec, AgentTemplate, message_to_dict
from ..agent_roles import CodeArtifactSpec, CodeSpec, ReviewReport
from ..knowledge import KnowledgeBase
from ..agent_tools.artifacts import WriteGeneratedArtifactInput
from ..agent_tools.base import invoke_tool
from ..agent_tools.render_focas_cpp import RenderFocasCppInput, RenderFocasCppOutput, RenderFocasCppTool
from ..agent_tools.validate_focas_cpp import (
    ValidateFocasCppInput,
    ValidateFocasCppOutput,
    ValidateFocasCppTool,
)
from ..agent_tools.knowledge import KnowledgeSearchInput, KnowledgeSearchOutput
from ..agent_tools.sdk_type_contract import SdkTypeContractInput
from ..tool_runtime import ToolRegistry, ToolRuntime
from ..llm import LlmClient
from ..models import ExecutionPlan, GeneratedArtifacts, NcProgramSpec, PlanStep, WorkflowState
from ..execution_plan_schema import execution_plan_view, normalize_execution_plan, validate_execution_plan
from ..tool_protocol import ToolCall, tool_calls_from_step, validate_tool_calls
from ..utils import ensure_dir
from ..sdk_type_index import (
    load_or_build_sdk_type_index, validate_struct_member_access,
    sdk_index_prompt_context, sdk_environment_identity,
)
from ..api_log_templates import api_metadata
from ..history_examples import retrieve_verified_api_examples
from ..deterministic_cpp_assembler import assemble_core_cpp, DeterministicAssemblyError
from ..sdk_cpp_generator import assemble_capture_call, SdkCallAssemblyError, output_symbol_name


def sdk_type_context_for_abi(abi: str) -> str:
    header = default_focas_header_dir() / "Fwlib32.h"
    if not header.is_file():
        return f"SDK header unavailable: {header}"
    names = set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", abi))
    return sdk_index_prompt_context(
        load_or_build_sdk_type_index(header, Path("rag_indexes/focas/sdk_type_index_0iD.json")), names
    )


def code_generator_soul_context(rag_context: dict[str, Any]) -> str:
    """Return only CodeGenerationAgent's scoped Soul context.

    The fallback keeps direct helper tests and older saved plans compatible;
    production workflows populate ``code_generator_soul`` explicitly.
    """
    return str(
        rag_context.get("code_generator_soul")
        or rag_context.get("agent_soul", "")
    ).strip()
from .prompts import CODE_REVIEW_JSON_SCHEMA, CODE_REVIEW_SYSTEM_PROMPT, FOCAS_CPP_GENERATION_SYSTEM_PROMPT


class CodeGenerationAgent(AgentTemplate):
    role_name = "CodeEngineerRole"
    profile = "FOCAS code generation agent"
    goal = "Turn the structured plan and CodeSpec into a complete compilable artifact."
    constraints = ("Use official declarations.", "Preserve the Planner atomic tool-call contract.")

    def __init__(self, llm_client: LlmClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        super().__init__()
        self.register_actions(
            ActionSpec("write_cpp_code", "PlanSpec", "CodeArtifact"),
            ActionSpec("repair_cpp_code", "ReviewReport", "CodeArtifact"),
        )
        self.llm_client = llm_client or LlmClient()
        self.knowledge_base = knowledge_base
        self.tool_runtime: ToolRuntime | None = None
        self.review_role = CodeReviewRole()

    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None:
            raise ValueError("Cannot generate code before a plan exists.")

        # Make the current run root available to the deterministic history
        # retriever; it never becomes an instruction for the model.
        state.plan.rag_context["output_dir"] = str(output_dir)
        normalize_execution_plan(state.plan, state.plan.rag_context.get("traffic_relationships"))
        state.plan.rag_context["execution_plan"] = execution_plan_view(state.plan)
        state.plan.rag_context["execution_plan_validation"] = validate_execution_plan(state.plan)

        # Query the authoritative SDK type contract before asking the model or
        # deterministic renderer to produce production C++. This is read-only;
        # unresolved types block generation instead of allowing local structs.
        if self.tool_runtime is not None and "get_sdk_type_contract" in self.tool_runtime.registry.names():
            target_api = str(state.plan.target_api or "").strip()
            if target_api:
                sdk_contract, _record = self.tool_runtime.call(
                    self.role_name,
                    "get_sdk_type_contract",
                    SdkTypeContractInput(target_api),
                )
                if not isinstance(sdk_contract, dict) or not sdk_contract.get("ok"):
                    raise ValueError(
                        "SDK_TYPE_CONTRACT_UNRESOLVED: "
                        + str((sdk_contract or {}).get("error", target_api))
                    )
                state.plan.rag_context["sdk_type_contract"] = sdk_contract

        soul_store = getattr(self, "soul_store", None)
        if soul_store is not None:
            state.plan.rag_context["code_generator_soul"] = soul_store.prompt_context(
                self.role_name,
                task_context=f"{state.request.description}; scenario={state.plan.scenario_type}",
                error_type="; ".join(str(item) for item in state.errors[-4:]),
                memory_store=getattr(self, "memory_store", None),
            )

        repair_context = summarize_codegen_repair_context(state)
        if repair_context:
            state.plan.rag_context["code_generation_repair_context"] = repair_context
        program_contract_diagnostics = normalize_plan_nc_program_contract(state.plan)
        planned_steps = list(state.plan.steps)
        planned_steps = normalize_program_lifecycle_step_parameters(
            planned_steps,
            state.plan.nc_program_spec.program_name,
        )
        executable_steps, skipped_steps = select_codegen_executable_steps(planned_steps)
        if not executable_steps:
            raise ValueError("CodeGenerationAgent could not generate an executable C++ API script: no executable planned steps.")
        tool_calls = [call for step in executable_steps for call in tool_calls_from_step(step)]
        tool_contract_errors = validate_tool_calls(tool_calls)
        if tool_contract_errors:
            raise ValueError("Invalid Planner tool-call contract: " + "; ".join(tool_contract_errors))
        if state.plan.code_spec is None:
            from ..code_spec import build_code_spec
            state.plan.code_spec = build_code_spec(state.plan, state.retrieved_chunks)
        code_spec = state.plan.code_spec
        state.plan.rag_context["sdk_environment_identity"] = sdk_environment_identity(
            default_focas_header_dir() / "Fwlib32.h",
            Path(os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir()))) / "Fwlib32.dll",
        )
        # Production API operations are generated as small model-authored
        # fragments.  The model receives the exact SDK ABI/rule context and
        # the immutable harness still owns connection, capture, logging and
        # cleanup.  Deterministic validation/compilation remains the admission
        # gate; the old recipe renderer is retained for compatibility callers.
        deterministic_cpp_path = False
        deterministic_fast_path = bool(state.plan.rag_context.get("fast_path", {}).get("enabled"))
        state.plan.rag_context["generation_mode"] = "llm_api_fragments"
        state.plan.rag_context["logs_generated_by"] = "deterministic_harness"
        # Retrieve rule/context evidence once for the complete plan.  The
        # subsequent bundle generation sends it together with the active SDK
        # excerpts, rather than issuing one model request per API.
        code_generation_tool_search = [] if deterministic_cpp_path else self._model_tool_retrieval(
            state.request.description,
            state.plan.scenario_type,
            code_spec,
            state.plan.rag_context,
        )
        if code_generation_tool_search:
            state.plan.rag_context["code_generation_tool_search"] = code_generation_tool_search
        if self.environment is not None:
            state.plan.rag_context["environment_shared_rules"] = self.environment.shared_rules
        codegen_knowledge = [] if deterministic_cpp_path else retrieve_codegen_knowledge_context(
            self.knowledge_base, state, executable_steps
        )
        if codegen_knowledge:
            state.plan.rag_context["code_generation_knowledge"] = codegen_knowledge
        generated_dir = ensure_dir(output_dir / "generated")
        previous_fragment_path = (
            state.artifacts.fragment_path
            if state.artifacts is not None and state.artifacts.fragment_path is not None
            else None
        )
        previous_harness_path = (
            state.artifacts.harness_path
            if state.artifacts is not None and state.artifacts.harness_path is not None
            else None
        )
        repairing_existing_fragment = bool(
            state.repair_history
            and state.repair_history[-1].get("repair_stage") == "repair_code"
            and previous_fragment_path is not None
            and previous_fragment_path.exists()
        )
        runtime_capture = (
            state.request.target_environment == "ncguide-generated-cpp"
            and not bool(state.request.permissions.get("code_only_evaluation"))
        )
        # One production shell for every API batch. Motion is a capability of
        # that shell, not a different generator selected by a scenario label.
        fixed_runtime_capture = False
        generated_runtime_capture = runtime_capture and not fixed_runtime_capture
        runtime_contract = execution_contract(
            state.plan, state.request.permissions.get("coverage_current_batch_functions")
            or state.request.permissions.get("required_api_functions") or (),
        ) if runtime_capture else None
        if runtime_contract:
            state.plan.rag_context["execution_contract"] = runtime_contract.to_dict()
        prerequisite_plan = state.plan.rag_context.get("api_prerequisites", {})
        if prerequisite_plan.get("enabled") and runtime_contract is not None:
            state.plan.rag_context["execution_contract"]["api_prerequisites"] = prerequisite_plan
        if runtime_capture:
            repairing_existing_fragment = False
        deferred_motion = any(
            isinstance(scene, dict)
            and scene.get("nc_execution_decision", {}).get("cycle_start_required")
            and not scene.get("nc_execution_decision", {}).get("cycle_start_allowed")
            for scene in state.plan.rag_context.get("scene_plans", [])
        )
        # A running-NC plan may still be waiting for the operator/Cycle Start
        # gate, but it must retain a valid O-number payload for upload and
        # later execution.  Clearing it here while the running rule has
        # already restored StartProgram creates an internally contradictory
        # contract and fails before capture.
        if should_defer_nc_program_generation(state.plan, deferred_motion):
            nc_program = ""
            nc_generation_diagnostics = [
                "NC generation/execution deferred: Cycle Start is required but its simulator and operator gates are not verified."
            ]
        elif should_skip_nc_program_generation(state.plan):
            nc_program = ""
            nc_generation_diagnostics = [
                "CodeGenerationAgent used deterministic read-only fast path; NC program generation skipped."
            ]
        else:
            # NC generation needs the same graph contract that Planner used.
            # Keep it in the shared context so the generator and its validator
            # cannot silently downgrade a running-state scene to O-number/M30.
            state.plan.rag_context["nc_program_requirements"] = list(
                dict.fromkeys(str(item).strip() for item in state.plan.nc_program_requirements if str(item).strip())
            )
            state.plan.rag_context["nc_target_api"] = str(
                state.plan.target_api
                or next((step.api or step.protocol_function.split("/")[0] for step in executable_steps if step.role == "target"), "")
            ).strip()
            nc_program, nc_generation_diagnostics = generate_nc_program(
                state.request.description,
                state.plan.scenario_type,
                state.plan.nc_program_spec,
                state.plan.rag_context,
                self.llm_client,
            )
        nc_generation_diagnostics.extend(program_contract_diagnostics)
        if fixed_runtime_capture:
            harness_text = render_fixed_ncguide_cpp_script(state, executable_steps, nc_program)
            fragment_text = build_scenario_fragment("    // Runtime capture uses the deterministic NCGuide runner.\n")
            cpp_generation_diagnostics = [
                "CodeGenerationAgent used the deterministic NCGuide capture runner with invocation/transaction boundaries."
            ]
        elif generated_runtime_capture:
            fragment_text = generate_capture_fragment(
                state, executable_steps, nc_program, self.llm_client,
            )
            harness_text = render_capture_harness(
                runtime_contract.functions,
                (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1"),
                fragment_text,
                contract=runtime_contract, nc_program=nc_program,
            )
            cpp_generation_diagnostics = [
                "Production capture uses the immutable SMPAgent harness; only API operations are generated/repaired."
            ]
            if deterministic_fast_path:
                cpp_generation_diagnostics.extend([
                    "Planner fast path selected for scenario/NC decisions; API fragments still use SDK-context LLM generation.",
                    "Input/output CSV rows, invocation IDs, timestamps and return-code text are generated by CaptureContext and checked by ValidateFocasCppTool.",
                ])
        elif repairing_existing_fragment:
            # Rebuild from the current ExecutionPlan and SDK recipes.  A
            # previous fragment is evidence for diagnostics only; it is never
            # handed back to a model to rewrite ABI-critical C++.
            harness_text, fragment_text = render_code_only_cpp_artifacts(
                state, executable_steps, nc_program, self.llm_client,
            )
            cpp_generation_diagnostics = [
                "CodeGenerationAgent regenerated only the affected API fragments from the ExecutionPlan after a repair request."
            ]
        else:
            harness_text, fragment_text = render_code_only_cpp_artifacts(
                state,
                executable_steps,
                nc_program,
                self.llm_client,
            )
            cpp_generation_diagnostics = [
                "CodeGenerationAgent generated scenario_fragment.inc and assembled it with the fixed FOCAS harness."
            ]
        api_script = harness_text

        api_script_path = generated_dir / "api_script.py"
        legacy_cpp_path = generated_dir / "focas_test.cpp"
        if api_script_path.exists():
            api_script_path.unlink()
        if legacy_cpp_path.exists():
            legacy_cpp_path.unlink()
        api_script_path = generated_dir / "api_script.cpp"
        harness_path = generated_dir / "focas_harness.cpp"
        fragment_path = generated_dir / "scenario_fragment.inc"
        nc_program_path = generated_dir / "program.nc"
        self._write_artifact(nc_program_path, nc_program)
        self._write_artifact(harness_path, harness_text)
        self._write_artifact(fragment_path, fragment_text)
        self._render_focas_cpp(harness_path, fragment_path, api_script_path)
        api_script = api_script_path.read_text(encoding="utf-8")
        cpp_generation_diagnostics.append(
            "RenderFocasCppTool assembled focas_harness.cpp and scenario_fragment.inc into api_script.cpp."
        )

        diagnostics: list[str] = []
        max_internal_repair_attempts = 2
        seen_repair_diagnostics: set[str] = set()
        for compile_attempt in range(max_internal_repair_attempts + 1):
            validation_output = self._validate_focas_cpp(
                api_script_path,
                fragment_path,
                nc_program,
                executable_steps,
                state,
                code_spec,
            )
            diagnostics = [*validation_output.errors, *validation_output.warnings]
            diagnostics.extend(nc_generation_diagnostics)
            if state.request.permissions.get("ablation_variant") == "single-agent":
                diagnostics.append(
                    "RQ3 Single-Agent used the shared deterministic task/RAG scaffold; "
                    "no PlanningAgent or RouterAgent LLM decision was invoked."
                )
            else:
                diagnostics.append("CodeGenerationAgent used PlannerAgent steps directly; step planning remains owned by PlanningAgent.")
            diagnostics.extend(cpp_generation_diagnostics)
            diagnostics.extend(skipped_steps)
            diagnostics.append(f"C++ FOCAS API script saved to {api_script_path}")
            state.artifacts = GeneratedArtifacts(
                api_script=api_script,
                nc_program=nc_program,
                api_script_path=api_script_path,
                nc_program_path=nc_program_path,
                diagnostics=diagnostics,
                harness_path=harness_path,
                fragment_path=fragment_path,
            )
            validation_errors = validation_output.errors
            if validation_errors:
                validation_fingerprint = re.sub(r"\s+", " ", "; ".join(validation_errors)).strip()
                if validation_fingerprint in seen_repair_diagnostics:
                    raise ValueError(
                        "CODEGEN_REPAIR_STALLED: repeated validation diagnostic; "
                        + validation_fingerprint
                    )
                seen_repair_diagnostics.add(validation_fingerprint)
                if compile_attempt < max_internal_repair_attempts and generated_runtime_capture and self.llm_client.enabled:
                    fragment_text = generate_capture_fragment(
                        state, executable_steps, nc_program, self.llm_client,
                        previous=fragment_text, diagnostics="; ".join(validation_errors),
                    )
                    self._write_artifact(fragment_path, fragment_text)
                    self._write_artifact(harness_path, render_capture_harness(
                        runtime_contract.functions,
                        (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1"), fragment_text,
                        contract=runtime_contract, nc_program=nc_program,
                    ))
                    self._render_focas_cpp(harness_path, fragment_path, api_script_path)
                    api_script = api_script_path.read_text(encoding="utf-8")
                    continue
                if not generated_runtime_capture:
                    raise ValueError(
                        "CODEGEN_DETERMINISTIC_VALIDATION_FAILED: "
                        + "; ".join(validation_errors)
                    )
                raise ValueError(("CODEGEN_REPAIR_EXHAUSTED: " if runtime_capture else "") +
                                 "ValidateFocasCppTool rejected generated C++: " + "; ".join(validation_errors))
            if state.request.target_environment != "ncguide-generated-cpp":
                break
            compile_error = preflight_compile_generated_cpp(
                api_script_path,
                generated_dir,
                tool_runtime=self.tool_runtime,
            )
            if not compile_error:
                if compile_attempt:
                    diagnostics.append(f"CodeGenerationAgent fixed MSVC preflight compilation after {compile_attempt} internal repair attempt(s).")
                break
            compile_fingerprint = re.sub(r"\s+", " ", compile_error).strip()
            if compile_fingerprint in seen_repair_diagnostics:
                raise ValueError(
                    "CODEGEN_REPAIR_STALLED: repeated compiler diagnostic; "
                    + compile_fingerprint
                )
            seen_repair_diagnostics.add(compile_fingerprint)
            diagnostics.append(f"MSVC preflight compilation failed on CodeGenerationAgent internal attempt {compile_attempt + 1}: {compile_error}")
            state.artifacts.diagnostics = diagnostics
            if compile_attempt >= max_internal_repair_attempts or not self.llm_client.enabled or fixed_runtime_capture or not generated_runtime_capture:
                raise ValueError(("CODEGEN_REPAIR_EXHAUSTED: " if runtime_capture else "") +
                                 "Generated C++ failed MSVC preflight compilation: " + compile_error)
            if generated_runtime_capture:
                fragment_text = generate_capture_fragment(
                    state, executable_steps, nc_program, self.llm_client,
                    previous=fragment_text, diagnostics=compile_error,
                )
                self._write_artifact(fragment_path, fragment_text)
                self._write_artifact(harness_path, render_capture_harness(
                    runtime_contract.functions,
                    (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1"), fragment_text,
                    contract=runtime_contract, nc_program=nc_program,
                ))
                self._render_focas_cpp(harness_path, fragment_path, api_script_path)
                api_script = api_script_path.read_text(encoding="utf-8")
                cpp_generation_diagnostics.append("CodeGenerationAgent repaired only API operations from MSVC diagnostics.")
                continue
            fragment_text = repair_scenario_fragment_after_compile_error(
                state, executable_steps, nc_program, fragment_text,
                compile_error, self.llm_client,
            )
            self._write_artifact(fragment_path, fragment_text)
            self._render_focas_cpp(harness_path, fragment_path, api_script_path)
            api_script = api_script_path.read_text(encoding="utf-8")
            cpp_generation_diagnostics.append("CodeGenerationAgent repaired scenario_fragment.inc from MSVC preflight compiler diagnostics.")
        # Runtime fragments already pass the deterministic SDK/contract/MSVC
        # gates below.  Do not spend a second model call on an advisory whole
        # program review for the same live capture; keep that review for the
        # non-runtime compatibility path only.
        if self.llm_client.enabled and not deterministic_cpp_path and not generated_runtime_capture and not bool(state.request.permissions.get("code_only_evaluation")):
            review_diagnostics = review_generated_with_llm(
                    self.llm_client,
                    state.request.description,
                    state.plan.scenario_type,
                    nc_program,
                    api_script + ("\n" + fragment_text if generated_runtime_capture else ""),
                    planned_api_names=list(runtime_contract.functions) if runtime_contract else [
                        call.tool_name
                        for step in executable_steps
                        for call in tool_calls_from_step(step)
                    ],
                    code_spec=code_spec,
                    code_only=bool(state.request.permissions.get("code_only_evaluation")),
                )
            diagnostics.extend(review_diagnostics)
            review_message = self.review_role.publish(
                "review_cpp_code", "ReviewReport",
                passed=not any("marked generated artifacts" in item for item in review_diagnostics),
                diagnostic_count=len(review_diagnostics),
            )
            state.messages.append(message_to_dict(
                review_message,
                ReviewReport(
                    not any("marked generated artifacts" in item for item in review_diagnostics),
                    len(review_diagnostics),
                ),
            ))
            if generated_runtime_capture and review_diagnostics:
                diagnostics.append("LLM review is advisory; typed runtime contracts and the real SDK compiler own execution admission.")
        state.artifacts.diagnostics = diagnostics
        handoff = self.publish(
            "write_cpp_code", "CodeArtifact",
            api_script_path=str(api_script_path),
            nc_program_path=str(nc_program_path),
            diagnostic_count=len(diagnostics),
        )
        state.messages.append(message_to_dict(
            handoff,
            CodeArtifactSpec(str(api_script_path), str(nc_program_path), len(diagnostics)),
        ))
        if self.tool_runtime is not None:
            state.tool_call_history = list(self.tool_runtime.history)
        state.stage = "execution"
        return state

    def _model_tool_retrieval(
        self,
        task_description: str,
        scenario: str,
        code_spec: CodeSpec,
        rag_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Use the CodeGenerationAgent-owned loop for authoritative evidence."""
        if self.tool_runtime is None or not self.llm_client.enabled:
            return []
        loop = self.tool_loop(max_turns=3)
        api_names = [contract.tool_name for contract in code_spec.api_contracts]
        system_prompt = (
            "You are the evidence-retrieval substep of CodeGenerationAgent. "
            "Before returning, call search_knowledge at least once for official FOCAS prototypes, "
            "structure definitions, and generation rules. Prefer one broad query covering all requested APIs; "
            "after receiving its results, return the final JSON without redundant searches. "
            "Do not generate C++ in this substep. "
            "After retrieval, return JSON only with {\"retrieval_complete\":true}."
        )
        user_prompt = (
            f"Task: {task_description}\nScenario: {scenario}\n"
            f"Authoritative API set: {api_names}\n"
            f"Agent Soul:\n{code_generator_soul_context(rag_context or {})}\n"
            f"Traffic Relationship Graph context:\n{(rag_context or {}).get('traffic_relationships', {})}\n"
            "Search the RAG knowledge base for exact declarations and ABI evidence for these APIs."
        )
        soul_context = code_generator_soul_context(rag_context or {})
        if soul_context:
            system_prompt += (
                "\nScoped Agent Soul:\n"
                + soul_context
                + "\nUse graph relationships only as evidence; proposed relations are not mandatory."
            )
        loop.run(
            self.role_name,
            system_prompt,
            user_prompt,
            input_factory=self._tool_input_factory,
        )
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tool_name, output, _ in loop.last_tool_outputs:
            if tool_name != "search_knowledge" or not isinstance(output, KnowledgeSearchOutput):
                continue
            for item in output.results:
                chunk_id = item.chunk.chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "score": item.score,
                        "source_type": item.chunk.metadata.get("source_type"),
                        "function": item.chunk.metadata.get("function"),
                        "text_preview": item.chunk.text[:520],
                    }
                )
        return rows

    @staticmethod
    def _tool_input_factory(tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "search_knowledge":
            raise ValueError(f"CodeGenerationAgent retrieval loop only supports search_knowledge, got {tool_name}")
        return KnowledgeSearchInput(
            query=str(arguments.get("query", "")),
            top_k=int(arguments.get("top_k", 8)),
            filters=dict(arguments.get("filters", {}) or {}),
        )

    def _render_focas_cpp(
        self,
        harness_path: Path,
        fragment_path: Path,
        output_cpp_path: Path,
    ) -> RenderFocasCppOutput:
        tool_input = RenderFocasCppInput(harness_path, fragment_path, output_cpp_path)
        if self.tool_runtime is not None:
            output, _record = self.tool_runtime.call(self.role_name, "render_focas_cpp", tool_input)
        else:
            # Dependency-injection path for isolated unit tests only. The
            # production workflow always supplies the shared ToolRuntime.
            output, record = invoke_tool(RenderFocasCppTool(), tool_input, agent_name=self.role_name)
        if not output.success:
            raise ValueError(output.error or "RenderFocasCppTool failed to assemble api_script.cpp.")
        return output

    def _validate_focas_cpp(
        self,
        cpp_path: Path,
        fragment_path: Path | None,
        nc_program: str,
        executable_steps: list[PlanStep],
        state: WorkflowState,
        code_spec: CodeSpec,
    ) -> ValidateFocasCppOutput:
        tool_input = ValidateFocasCppInput(
            cpp_path=cpp_path,
            fragment_path=fragment_path,
            nc_program=nc_program,
            executable_steps=tuple(executable_steps),
            allow_delete_all_programs=bool(state.request.permissions.get("allow_delete_all_programs")),
            require_official_focas_header=True,
            # TaskRequest has no ``execute`` attribute; that check silently
            # disabled real-capture validation for every live desktop run.
            require_real_pcap_capture=(
                state.request.target_environment == "ncguide-generated-cpp"
                and not bool(state.request.permissions.get("code_only_evaluation"))
            ),
            code_only_evaluation=bool(state.request.permissions.get("code_only_evaluation")),
            code_spec=code_spec,
            production_contract=(execution_contract(
                state.plan, state.request.permissions.get("coverage_current_batch_functions")
                or state.request.permissions.get("required_api_functions") or (),
            ) if state.request.target_environment == "ncguide-generated-cpp"
                and not state.request.permissions.get("code_only_evaluation") else None),
        )
        if self.tool_runtime is not None:
            output, _record = self.tool_runtime.call(self.role_name, "validate_focas_cpp", tool_input)
        else:
            # Dependency-injection path for isolated unit tests only. The
            # production workflow always supplies the shared ToolRuntime.
            output, _record = invoke_tool(ValidateFocasCppTool(), tool_input, agent_name=self.role_name)
        return output

    def _write_artifact(self, path: Path, content: str) -> None:
        if self.tool_runtime is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return
        self.tool_runtime.call(
            self.role_name,
            "write_generated_artifact",
            WriteGeneratedArtifactInput(path, content),
        )


def preflight_compile_generated_cpp(
    source_path: Path,
    work_dir: Path,
    *,
    tool_runtime: ToolRuntime | None = None,
) -> str:
    from ..agent_tools import CompileCppInput, CompileGeneratedCppTool

    exe_path = work_dir / "preflight_api_script.exe"
    if tool_runtime is None:
        registry = ToolRegistry()
        registry.register(CompileGeneratedCppTool())
        tool_runtime = ToolRuntime(registry)
    result, _record = tool_runtime.call(
        "CodeEngineerRole",
        "compile_generated_cpp",
        CompileCppInput(source_path, exe_path, work_dir),
    )
    success = result.success
    for path in [exe_path, work_dir / f"{source_path.stem}.obj"]:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
    if success:
        return ""
    output = "\n".join(part.strip() for part in [result.stdout or "", result.stderr or ""] if part.strip())
    # Keep the primary diagnostic; template overload notes can consume the entire tail.
    if len(output) > 3000:
        output = output[:1800] + "\n... compiler notes truncated ...\n" + output[-1200:]
    return output or f"compiler exit code {result.return_code}"


def select_codegen_executable_steps(steps: list[PlanStep]) -> tuple[list[PlanStep], list[str]]:
    selected = []
    skipped = []
    for step in steps:
        if is_non_executable_codegen_step(step):
            reason = (
                "Packet capture is managed by the deterministic runner, not a FOCAS API."
                if "npcap" in f"{step.interface_name} {step.protocol_function}".lower()
                else "Planner step is an analysis/evaluation step, not a direct C++ API/control operation."
            )
            skipped.append(
                f"Skipped {step.step_id} {step.interface_name}/{step.protocol_function}: "
                f"{reason}"
            )
            continue
        selected.append(step)
    return selected, skipped


def is_non_executable_codegen_step(step: PlanStep) -> bool:
    text = f"{step.interface_name} {step.protocol_function} {step.action}".lower()
    if "npcap" in text or "capture preflight" in text or "capture stop" in text:
        return True
    if step.protocol_function.strip():
        return False
    return any(word in text for word in ["evaluate", "evaluation", "assess", "quality", "annotate", "mapping"])


def steps_for_prompt(steps: list[PlanStep]) -> list[dict[str, Any]]:
    result = []
    for step in steps:
        calls = tool_calls_from_step(step)
        result.append({
            "step_id": step.step_id,
            "phase": step.phase,
            "action": step.action,
            "interface_name": step.interface_name,
            "parameters": step.parameters,
            "repeat": step.repeat,
            "interval_seconds": step.interval_seconds,
            "expected_state": step.expected_state,
            "protocol_function": step.protocol_function,
            "operation_kind": step.operation_kind,
            "api_calls": step.api_calls,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "phase": call.phase,
                    "operation_id": call.operation_id,
                }
                for call in calls
            ],
        })
    return result


def summarize_codegen_repair_context(state: WorkflowState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in state.repair_history[-3:]:
        previous_state = item.get("previous_state", {})
        row: dict[str, Any] = {
            "repair_stage": item.get("repair_stage"),
            "attempt": item.get("attempt"),
            "errors": item.get("errors", [])[:10],
            "router_reason": item.get("router_reason", ""),
            "repair_instruction": item.get("repair_instruction", ""),
        }
        if isinstance(previous_state, dict):
            artifacts = previous_state.get("artifacts")
            result = previous_state.get("result")
            quality = previous_state.get("quality_assessment")
            if artifacts:
                row["previous_artifacts"] = artifacts
            if result:
                row["previous_result"] = result
            if quality:
                row["previous_quality_assessment"] = quality
        rows.append(row)
    if state.errors:
        rows.append({"current_errors": state.errors[-10:]})
    return rows


def normalize_program_lifecycle_step_parameters(steps: list[PlanStep], program_name: str) -> list[PlanStep]:
    for step in steps:
        if step.interface_name in {"UploadProgram", "SelectProgram"}:
            step.parameters["program_name"] = program_name
    return steps


def normalize_plan_nc_program_contract(plan: ExecutionPlan) -> list[str]:
    requested_name = plan.nc_program_spec.program_name.strip()
    allocated_by_codegen = not requested_name
    effective_name = (
        f"O{1000 + secrets.randbelow(7000):04d}"
        if allocated_by_codegen
        else safe_nc_program_name(requested_name)
    )
    if effective_name == requested_name:
        return []

    plan.nc_program_spec.program_name = effective_name
    normalize_program_lifecycle_step_parameters(plan.steps, effective_name)
    # CodeSpec copies the program name, so a previously built instance must be
    # regenerated after safety normalization.
    plan.code_spec = None
    record_key = "code_generation_program_allocation" if allocated_by_codegen else "nc_program_name_normalization"
    plan.rag_context[record_key] = {
        "requested": requested_name or None,
        "effective": effective_name,
        "reason": "allocated_by_code_generation" if allocated_by_codegen else "protected_or_invalid_program_number",
    }
    if allocated_by_codegen:
        return [
            f"CodeGenerationAgent allocated candidate NC program name {effective_name}; runtime may choose another "
            "unused O number after reading the controller directory."
        ]
    return [
        f"CodeGenerationAgent normalized NC program name from {requested_name} to {effective_name} "
        "before building CodeSpec and artifacts."
    ]


def protocol_function_for_interface(interface_name: str) -> str:
    for step_interface, function_name in [
        ("UploadProgram", "cnc_dwnstart3/cnc_download3/cnc_dwnend3"),
        ("SelectProgram", "cnc_search"),
        ("ReadProgramNumber", "cnc_rdprgnum"),
        ("StartProgram", "ncguide_ui_cycle_start"),
        ("ReadRunStatus", "cnc_statinfo"),
        ("ReadPosition", "cnc_rdposition"),
        ("ReadDistanceToGo", "cnc_distance"),
        ("ReadFeedSpeed", "cnc_actf"),
        ("ReadSpindleSpeed", "cnc_acts"),
        ("ReadAlarm", "cnc_alarm2"),
    ]:
        if interface_name == step_interface:
            return function_name
    return ""


def safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def generate_nc_program(
    task_description: str,
    scenario: str,
    nc_program_spec: NcProgramSpec,
    rag_context: dict[str, Any],
    llm_client: LlmClient,
) -> tuple[str, list[str]]:
    if not llm_client.enabled:
        raise RuntimeError("CodeGenerationAgent requires an LLM NC program generation in agent-only mode.")
    generated = generate_nc_program_with_llm(task_description, scenario, nc_program_spec, rag_context, llm_client)
    if not generated:
        raise ValueError("CodeGenerationAgent LLM returned an empty or unsafe NC program.")
    framing_errors = (
        validate_nc_payload_framing(generated)
        + validate_nc_controller_modes(generated, rag_context)
        + validate_nc_program_requirements(generated, rag_context)
        + validate_nc_runtime_budget(generated, rag_context)
    )
    if framing_errors:
        repaired = repair_nc_program_with_llm(
            task_description,
            scenario,
            nc_program_spec,
            rag_context,
            generated,
            framing_errors,
            llm_client,
        )
        if not repaired:
            raise ValueError("CodeGenerationAgent NC repair returned an empty or unsafe NC program.")
        framing_errors = (
            validate_nc_payload_framing(repaired)
            + validate_nc_controller_modes(repaired, rag_context)
            + validate_nc_program_requirements(repaired, rag_context)
            + validate_nc_runtime_budget(repaired, rag_context)
        )
        if framing_errors:
            raise ValueError("CodeGenerationAgent NC payload repair failed: " + "; ".join(framing_errors))
        return repaired, [
            "LLM generated NC program from PlannerAgent specification.",
            "CodeGenerationAgent repaired NC payload framing from validation feedback.",
            f"Deterministic NC runtime estimate: {rag_context.get('nc_runtime_estimate', {}).get('estimated_seconds', 0)}s "
            f"within budget {rag_context.get('nc_runtime_budget_seconds', nc_runtime_budget_seconds())}s.",
        ]
    return generated, [
        "LLM generated NC program from PlannerAgent specification.",
        f"Deterministic NC runtime estimate: {rag_context.get('nc_runtime_estimate', {}).get('estimated_seconds', 0)}s "
        f"within budget {rag_context.get('nc_runtime_budget_seconds', nc_runtime_budget_seconds())}s.",
    ]


def generate_nc_program_with_llm(
    task_description: str,
    scenario: str,
    nc_program_spec: NcProgramSpec,
    rag_context: dict[str, Any],
    llm_client: LlmClient,
) -> str:
    system_prompt = (
        "# Identity\n"
        "You are CodeGenerationAgent in a multi-agent FANUC FOCAS traffic-generation system.\n"
        "You generate concrete, safe NC program text from PlannerAgent's NC program specification.\n\n"
        "# Instructions\n"
        "- Generate concrete FANUC-style NC program blocks using the safe candidate O program name allocated by CodeGenerationAgent.\n"
        "- Use safe simulator-scale motion only.\n"
        "- Do not emit redundant G97 when no G96 constant-surface-speed mode is used. For a G17 milling program, use S with M03/M04 directly; do not introduce G96.\n"
        "- If rag_context.coverage_segments contains multiple segments, generate one short NC text payload for the first segment with nc_program_required=true. Only cycle_start_required=true calls for executable axis motion. A program-transfer segment may need NC text but must not start it. Other non-NC segments use direct FOCAS calls.\n"
        "- For coordinate-motion traffic, prefer several observable G01 motion blocks whose main motion lasts about 2-5 seconds each in simulation.\n"
        "- Use a documented positive feed and short travel that keeps the complete payload within the deterministic runtime budget. Do not force a slow feed merely to obtain longer sampling windows.\n"
        "- Keep the complete payload within the deterministic global runtime budget. Prefer short motion blocks and documented feed values; never use long low-feed motion that can hold run=3/motion=1 until the executor timeout.\n"
        "- Use PlannerAgent's quality analysis: if feed/position variation is required, balance motion duration, sampling frequency, and total runtime.\n"
        "- If a previous NC program failed or produced poor traffic, revise that program based on the failure context instead of ignoring it.\n"
        "- End with M30.\n"
        "- Return JSON only. Do not include Markdown.\n\n"
        "# Output Schema\n"
        "{\"nc_program\":\"O1234\\nG90 G54\\nG01 X...\\nM30\\n\",\"notes\":[\"short note\"],\"alarm_trigger\":{\"alarm_code\":\"controller code\",\"condition\":\"exact NC condition\",\"evidence\":\"RAG/manual evidence id\"}}"
    )
    required_nc = [str(item).strip() for item in rag_context.get("nc_program_requirements", []) if str(item).strip()]
    target_api = str(rag_context.get("nc_target_api", "")).strip()
    if required_nc:
        system_prompt += (
            "\n# Mandatory graph NC requirements\n"
            f"Target API: {target_api or 'unspecified'}\n"
            f"Requirements: {required_nc}\n"
            "These requirements are executable constraints, not explanatory notes. "
            "Do not return an idle O-number/M30-only program when they require a running state. "
            "For spindle variation, include real M03 or M04 start and multiple distinct S speed blocks "
            "(minimum two; additional values are allowed and preferred when the runtime budget permits), "
            "M05 stop, and M30.\n"
        )
    if target_api == "cnc_rdspcss":
        system_prompt += (
            "\n# Constant-surface-speed requirement\n"
            "cnc_rdspcss reads CSS state, so the NC payload must execute in lathe/CSS mode: "
            "include G96 before the spindle speed stages, use multiple distinct S values "
            "(minimum two; do not cap the program at two when more values fit the runtime budget), "
            "and do not emit G17 or G94 milling-mode blocks. Keep M03/M04, M05 and M30.\n"
        )
        system_prompt += (
            "Use a short, moderate-speed motion window: prefer G91 incremental moves of about "
            "Z-2.0 and X1.0 (never an absolute far-away X target such as X30 when the current "
            "position is unknown), use short incremental moves and a documented positive feed selected "
            "from the active controller limits, then restore G90 before M05/M30. The global runtime "
            "estimate, rather than an API-specific slow-feed rule, controls the duration.\n"
        )
        system_prompt += (
            "The attached FANUC 0i-D manual distinguishes M and T semantics: "
            "for M-series machining centers use G92 S_ for the maximum spindle-speed clamp; "
            "G50 S_ is a T-series G-code-system-A option and must not be emitted unless "
            "rag_context explicitly confirms T series and G-code system A. If the series is "
            "unknown, record the missing clamp as an advisory state and allow the observation "
            "to proceed; do not turn smax=0 into a code-generation rejection.\n"
        )
    alarm_trigger = rag_context.get("alarm_history_trigger", {})
    if isinstance(alarm_trigger, dict) and alarm_trigger.get("required"):
        system_prompt += (
            "\n# Required alarm-producing NC behavior\n"
            "The selected target requires a fresh alarm before alarm-history reads. "
            "Choose one controller-specific alarm-producing NC condition from the retrieved, "
            "verified RAG/manual evidence for this controller and scenario. Include the exact "
            "alarm code/condition and its evidence in notes. Do not invent, hard-code, or reuse "
            "a generic G/M code when evidence does not support it; if no verified NC trigger is "
            "available, return an explicit generation error instead of a normal no-alarm program.\n"
        )
    segments = rag_context.get("coverage_segments", [])
    if segments and not any(segment.get("cycle_start_required") for segment in segments if isinstance(segment, dict)) and not (isinstance(alarm_trigger, dict) and alarm_trigger.get("required")):
        system_prompt += (
            "\n# Non-motion API batch\n"
            "The NC text, if needed, is only a program-transfer or verification input. "
            "Generate a minimal valid O-number/M30 payload without G00/G01 axis motion, "
            "spindle start, or Cycle Start. Do not impose coordinate/feed variation.\n"
        )
    user_prompt = (
        f"Task:\n{task_description}\n\n"
        f"Scenario: {scenario}\n"
        f"Planner NC structural spec plus CodeGenerationAgent allocation:\n"
        f"codegen_candidate_program_name={nc_program_spec.program_name}\n"
        f"purpose={nc_program_spec.purpose}\n"
        f"block_goals={nc_program_spec.block_goals}\n"
        f"constraints={nc_program_spec.constraints}\n"
        f"generation_notes={nc_program_spec.generation_notes}\n"
        f"coverage_segments={rag_context.get('coverage_segments', [])}\n"
        f"function_coverage_manifest={rag_context.get('function_coverage_manifest', {})}\n"
        f"quality_analysis={rag_context.get('planning_quality_analysis', {})}\n"
        f"quality_targets={rag_context.get('quality_targets', {})}\n"
        f"repair_context={rag_context.get('repair_context', [])}\n"
        f"code_generation_repair_context={rag_context.get('code_generation_repair_context', [])}\n"
        f"code_generation_knowledge={rag_context.get('code_generation_knowledge', [])}\n"
        f"agent_soul={code_generator_soul_context(rag_context)}\n"
        f"traffic_relationships={rag_context.get('traffic_relationships', {})}\n"
    )
    soul_context = code_generator_soul_context(rag_context)
    if soul_context:
        system_prompt += (
            "\n\n# Dynamic Agent Soul\n"
            + soul_context
            + "\nUse only documented or runtime_verified relationships as mandatory dependencies; "
            "proposed and unknown relationships are hypotheses only.\n"
        )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    if isinstance(alarm_trigger, dict) and alarm_trigger.get("required"):
        selected = payload.get("alarm_trigger") if isinstance(payload, dict) else None
        if not isinstance(selected, dict) or not all(
            str(selected.get(key, "")).strip() for key in ("alarm_code", "condition", "evidence")
        ):
            raise ValueError(
                "Alarm-history NC generation requires a controller-specific alarm_trigger "
                "with alarm_code, condition, and RAG evidence; no generic trigger is accepted."
            )
    nc_program = str(payload.get("nc_program", "")).strip()
    if not nc_program:
        blocks = payload.get("blocks", [])
        if isinstance(blocks, list):
            nc_program = "\n".join(str(block).strip() for block in blocks if str(block).strip())
    return normalize_llm_nc_program(nc_program, nc_program_spec.program_name)


def validate_nc_payload_framing(nc_program: str) -> list[str]:
    """Validate the exact byte framing required by cnc_download3."""
    errors: list[str] = []
    if not nc_program.startswith("\n"):
        errors.append("payload must begin with LF before the O program number")
    if not nc_program.endswith("\n%"):
        errors.append("payload must end with LF followed by a single percent character")
    if nc_program.count("%") != 1:
        errors.append("payload must contain exactly one percent character")
    return errors


def validate_nc_controller_modes(nc_program: str, rag_context: dict[str, Any] | None = None) -> list[str]:
    blocks = [re.sub(r"\([^)]*\)", "", line).upper() for line in nc_program.splitlines()]
    has_g17 = any(re.search(r"(?<![A-Z0-9.])G0?17(?![0-9.])", block) for block in blocks)
    has_g96 = any(re.search(r"(?<![A-Z0-9.])G0?96(?![0-9.])", block) for block in blocks)
    errors = []
    if has_g17 and has_g96:
        errors.append("G17 milling program must not use unverified G96 constant-surface-speed mode")
    if (has_g17 or not has_g96) and any(
        re.search(r"(?<![A-Z0-9.])G0?97(?![0-9.])", block) for block in blocks
    ):
        errors.append("NC program must omit unneeded G97; use S with M03/M04 directly")
    rag_context = rag_context or {}
    has_g50_speed_clamp = any(re.search(r"(?<![A-Z0-9.])G0?50\s*S\s*[+-]?\d", block) for block in blocks)
    series = str(rag_context.get("controller_series") or rag_context.get("machine_series") or "").strip().upper()
    g_code_system = str(rag_context.get("g_code_system") or "").strip().upper()
    if has_g50_speed_clamp and not (series.startswith("T") and g_code_system in {"A", "GCODE_A", "G CODE SYSTEM A"}):
        errors.append("G50 S maximum-spindle-speed clamp requires confirmed T series and G-code system A; use G92 S_ for M series or request confirmation")
    return errors


def validate_nc_program_requirements(nc_program: str, rag_context: dict[str, Any]) -> list[str]:
    """Validate graph-declared NC behavior before upload.

    The check is deliberately based on the plan contract, while the spindle
    rule is expressed by capability (a running spindle with speed variation),
    so new APIs can reuse it without adding API-specific branches.
    """
    requirements = [str(item).strip().lower() for item in rag_context.get("nc_program_requirements", []) if str(item).strip()]
    target = str(rag_context.get("nc_target_api", "")).strip()
    if not requirements:
        return []
    text = re.sub(r"\([^)]*\)", "", nc_program).upper()
    errors: list[str] = []
    running = any("running" in item or "主轴启动" in item or "spindle" in item for item in requirements)
    spindle_variation = any(
        marker in item
        for item in requirements
        for marker in ("恒表面速度", "主轴速度变化", "spindle speed", "speed variation")
    )
    css_target = target == "cnc_rdspcss"
    if running and spindle_variation or css_target:
        if not re.search(r"\bM0?[34]\b", text):
            errors.append(f"NC requirement for {target or 'target API'} requires spindle start with M03 or M04")
        speeds = re.findall(r"\bS(?:[0-9]+(?:\.[0-9]+)?)\b", text)
        if len(set(speeds)) < 2:
            errors.append(
                f"NC requirement for {target or 'target API'} requires multiple distinct spindle speed S values "
                "(minimum two; additional values are allowed)"
            )
        if not re.search(r"\bM0?5\b", text):
            errors.append(f"NC requirement for {target or 'target API'} requires spindle stop with M05")
    if css_target:
        if not re.search(r"\bG0?96\b", text):
            errors.append("NC requirement for cnc_rdspcss requires G96 constant-surface-speed mode")
        if re.search(r"\bG0?17\b", text) or re.search(r"\bG0?94\b", text):
            errors.append("NC requirement for cnc_rdspcss forbids G17/G94 milling mode; use G96 CSS mode")
        feeds = [float(value) for value in re.findall(r"\bF([0-9]+(?:\.[0-9]+)?)\b", text)]
        if feeds and any(value <= 0 for value in feeds):
            errors.append("NC requirement for cnc_rdspcss requires positive feed values")
        series = str(rag_context.get("controller_series") or rag_context.get("machine_series") or "").strip().upper()
        g_code_system = str(rag_context.get("g_code_system") or "").strip().upper()
        has_g92_clamp = any(re.search(r"\bG0?92\s*S\s*[1-9][0-9]*(?:\.[0-9]+)?\b", block) for block in text.splitlines())
        has_t_g50_clamp = (
            series.startswith("T")
            and g_code_system in {"A", "GCODE_A", "G CODE SYSTEM A"}
            and any(re.search(r"\bG0?50\s*S\s*[1-9][0-9]*(?:\.[0-9]+)?\b", block) for block in text.splitlines())
        )
        # smax=0 is a controller state observation.  Record it for graph/RAG
        # diagnostics instead of rejecting the NC payload; the applicable
        # G92/G50 rule depends on the machine series and code system.
    return errors


def repair_nc_program_with_llm(
    task_description: str,
    scenario: str,
    nc_program_spec: NcProgramSpec,
    rag_context: dict[str, Any],
    previous_program: str,
    framing_errors: list[str],
    llm_client: LlmClient,
) -> str:
    system_prompt = (
        "You are CodeGenerationAgent repairing an NC download payload. Return JSON only with key nc_program. "
        "Preserve the valid NC blocks, program number, safety constraints, and M30. "
        "The repaired payload must fit the deterministic global runtime budget; shorten motion blocks or use a "
        "documented feed instead of adding waits or long low-feed motion. "
        "Repair all reported framing and controller-mode errors. For a G17 milling program, remove G97 "
        "and use S with M03/M04 directly; do not introduce G96. The exact payload must start with one LF before O-number, "
        "For cnc_rdspcss, a positive maximum spindle-speed clamp must appear before G96: use G92 S#### for an M-series "
        "machining center; use G50 S#### only when the context explicitly confirms T series and G-code system A. "
        "use LF between lines, and end with one LF followed by exactly one percent character: \nO1234\n...\nM30\n%. "
        "Do not add a percent line at the beginning or any content after the final percent."
    )
    user_prompt = (
        f"Task: {task_description}\nScenario: {scenario}\n"
        f"Planner NC specification: {nc_program_spec}\n"
        f"Previous NC payload: {previous_program!r}\n"
        f"Validation errors: {framing_errors}\n"
        "Return the complete corrected payload, not a patch."
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    repaired = str(payload.get("nc_program", "")).replace("\r\n", "\n").replace("\r", "\n")
    return repaired


def normalize_llm_nc_program(nc_program: str, program_name: str) -> str:
    program_name = safe_nc_program_name(program_name)
    lines = [line.strip() for line in nc_program.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line and not line.startswith("%")]
    # Standalone parenthesized lines are generator commentary, not executable
    # NC blocks.  They also consume a Single Block click in NCGuide, so keep
    # them out of the uploaded payload entirely.
    lines = [line for line in lines if not re.fullmatch(r"\([^()]*\)", line)]
    if not lines:
        return ""
    if not lines[0].startswith("O"):
        lines.insert(0, program_name)
    else:
        lines[0] = program_name
    if not any(line.upper().startswith("M30") for line in lines):
        lines.append("M30")
    has_g17 = any(re.search(r"(?<![A-Z0-9.])G0?17(?![0-9.])", line, re.IGNORECASE) for line in lines)
    has_g96 = any(re.search(r"(?<![A-Z0-9.])G0?96(?![0-9.])", line, re.IGNORECASE) for line in lines)
    if has_g17 and has_g96:
        return ""
    if not has_g96:
        lines = [re.sub(r"(?<![A-Z0-9.])G0?97(?![0-9.])", "", line, flags=re.IGNORECASE).strip()
                 for line in lines]
        lines = [line for line in lines if line]
    if not is_safe_nc_program(lines):
        return ""
    return "\n".join(lines + [""])


def safe_nc_program_name(program_name: str) -> str:
    return f"O{safe_nc_program_number(nc_program_number(program_name, 5711)):04d}"


def should_skip_nc_program_generation(plan: ExecutionPlan) -> bool:
    """Return whether the selected path is genuinely read-only.

    A running-NC fast path still needs a generated O-number program; it only
    skips the Planner LLM decision.  Treating it as a read-only path produces
    an empty payload and is rejected by the execution contract.
    """
    fast_path = plan.rag_context.get("fast_path", {}) if plan is not None else {}
    return bool(
        isinstance(fast_path, dict)
        and fast_path.get("enabled")
        and fast_path.get("kind") != "running_nc_program"
        and not plan.rag_context.get("alarm_history_trigger", {}).get("required")
    )


def should_defer_nc_program_generation(plan: ExecutionPlan, deferred_motion: bool) -> bool:
    """Defer only plans that do not already require a running NC payload."""
    if not deferred_motion:
        return False
    fast_path = plan.rag_context.get("fast_path", {})
    running_plan = (
        plan.nc_program_type == "required_for_running_state"
        or (isinstance(fast_path, dict) and fast_path.get("kind") == "running_nc_program")
    )
    return not running_plan


def safe_nc_program_number(program_number: int) -> int:
    """Avoid FANUC protected O8000/O9000 ranges for generated upload payloads."""

    if 1 <= program_number < 8000:
        return program_number
    return 5700 + (abs(program_number) % 2000)


def is_safe_nc_program(lines: list[str]) -> bool:
    blocked_tokens = ["G28", "G30", "G53", "M98", "M99", "G10"]
    joined = " ".join(line.upper() for line in lines)
    return not any(token in joined for token in blocked_tokens)


def generate_cpp_api_script(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
) -> tuple[str, list[str]]:
    if state.plan is None:
        raise ValueError("CodeGenerationAgent cannot generate C++ before a plan exists.")
    # Keep this public compatibility entry point on the same statements-only
    # fragment path as production capture.  The LLM may author API statements,
    # while the fixed harness and validators own the surrounding C++.
    script = render_code_only_cpp_template(state, executable_steps, nc_program, llm_client)
    if not script:
        raise ValueError("CodeGenerationAgent deterministic C++ renderer returned an empty script.")
    return script, [
        "CodeGenerationAgent generated ordered SDK-context API fragments; the fixed harness and deterministic validators own execution admission."
    ]



def render_code_only_cpp_template(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
) -> str:
    """Render the assembled source for compatibility with existing callers."""
    return render_code_only_cpp_artifacts(state, executable_steps, nc_program, llm_client)[0]


def render_code_only_cpp_artifacts(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
) -> tuple[str, str]:
    """Return ``(focas_harness.cpp, scenario_fragment.inc)``."""
    if state.plan is None:
        raise ValueError("CodeGenerationAgent cannot render a template before a plan exists.")
    selected_functions = sorted(
        {
            function_name
            for step in executable_steps
            for function_name in protocol_function_names(step.protocol_function)
        }
    )
    contract = execution_contract(
        state.plan,
        state.request.permissions.get("coverage_current_batch_functions")
        or state.request.permissions.get("required_api_functions") or (),
    )
    if llm_client.enabled:
        # Keep the same statements-only API fragment contract for simulator,
        # code-only and live-capture callers.  The production harness still
        # owns its main function and dynamic bindings.
        core_cpp = generate_api_fragments_with_llm(
            state, executable_steps, nc_program, llm_client,
            wrap=False,
        )
        fragment = build_scenario_fragment(core_cpp)
    else:
        from ..parameter_coverage import attach_parameter_contract
        parameter_contract = attach_parameter_contract(state.plan, contract.targets)
        try:
            core_cpp = assemble_core_cpp(
                targets=contract.targets,
                planned_functions=contract.planned_functions or selected_functions,
                steps=executable_steps,
                parameter_contract=parameter_contract,
            )
        except DeterministicAssemblyError as exc:
            raise ValueError(str(exc)) from exc
        core_cpp = normalize_capture_core(core_cpp)
        fragment = build_scenario_fragment(core_cpp)
    return render_focas_harness(state, executable_steps, fragment), fragment


def render_focas_harness(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    fragment: str,
) -> str:
    """Assemble the immutable harness with the selected official bindings."""
    if state.plan is None:
        raise ValueError("CodeGenerationAgent cannot render a template before a plan exists.")
    selected_functions = sorted(
        {
            function_name
            for step in executable_steps
            for function_name in protocol_function_names(step.protocol_function)
        }
        | {"cnc_allclibhndl3", "cnc_freelibhndl"}
    )
    return render_focas_harness_template(selected_functions)


def _normalize_model_core(core_cpp: str) -> str:
    """Strip accidental wrapper/state declarations and normalize axis length.

    Repair prompts sometimes return the surrounding function despite the
    fragment contract. Remove that wrapper before the fixed harness adds its
    own one; also prevent sizeof(ODBAXIS) from reaching axis APIs where the
    documented argument is an axis count.
    """
    import re
    text = str(core_cpp or '').strip()
    match = re.search(r"short\s+run_generated_scenario\s*\([^)]*\)\s*\{(.*)\}\s*$", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    text = re.sub(r"(?m)^\s*(?:short\s+ret|bool\s+ok)\s*=.*?;\s*$", "", text)
    text = re.sub(r"sizeof\s*\(\s*ODBAXIS\s*\)", "(4 + 4 * MAX_AXIS)", text, flags=re.IGNORECASE)
    # The fixed harness owns the FOCAS handle lifecycle.  Models sometimes
    # copy a cleanup call into the fragment; remove that complete conditional
    # block before validation/compilation so it cannot double-free the handle
    # or trigger the fragment-boundary validator.
    cursor = 0
    while True:
        match = re.search(r"(?m)^\s*if\s*\([^\n]*matched_targets[^\n]*\)\s*\{", text[cursor:])
        if not match:
            break
        start = cursor + match.start()
        brace = cursor + match.end() - 1
        depth, index, quoted, escaped = 0, brace, None, False
        while index < len(text):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quoted:
                    quoted = None
            elif char in {"'", '"'}:
                quoted = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    block = text[start:end]
                    if "cnc_freelibhndl" in block:
                        text = text[:start] + "/* framework owns cnc_freelibhndl cleanup */" + text[end:]
                    cursor = start
                    break
            index += 1
        else:
            break
    # A standalone release operation is another common model leak. Remove
    # the complete call up to its following return-code check; the immutable
    # harness performs the release after the fragment returns.
    text = re.sub(
        r'(?s)ctx\.call\([^;]*"cnc_freelibhndl".*?\}\);\s*(?=if\s*\(ret)',
        '/* framework owns cnc_freelibhndl cleanup */\n',
        text,
        count=4,
    )
    # Log payload fields are semicolon-delimited.  Normalize the common
    # generated string form `",field="` without touching C++ argument lists.
    text = re.sub(r'(["\']),(?=[A-Za-z_]\w*=)', r'\1;', text)
    return text

def build_scenario_fragment(core_cpp: str) -> str:
    """Wrap model statements in the only fragment interface allowed by the harness."""
    core_cpp = _normalize_model_core(core_cpp)
    return (
        "short run_generated_scenario(\n"
        "    FocasApi& api,\n"
        "    unsigned short handle\n"
        ")\n"
        "{\n"
        "    CaptureContext ctx;\n"
        "    bool ok = true;\n"
        "    std::string nc_payload;\n"
        "    std::vector<char> nc_buffer;\n"
        "    long nc_length = 0;\n"
        "    short ret = EW_OK;\n"
        "    // SCENARIO_CORE_BEGIN\n"
        f"{core_cpp.rstrip()}\n"
        "    // SCENARIO_CORE_END\n"
        "    return ok ? EW_OK : ret;\n"
        "}\n"
    )


def capture_functions_for_steps(steps: list[PlanStep]) -> list[str]:
    return sorted({name for step in steps for name in protocol_function_names(step.protocol_function)
                   if name.startswith(("cnc_", "pmc_"))})


def normalize_capture_core(core: str) -> str:
    """Remove wrapper-owned state declarations from model-generated code.

    A repair can return the complete callback despite the body-only contract.
    Keep the first declaration in that case, and let the wrapper supply both
    declarations when the response contains body statements only.
    """
    complete = bool(re.search(r"\bshort\s+run_generated_scenario\s*\(", core))
    seen = set()
    lines = []
    for line in core.splitlines(keepends=True):
        match = re.fullmatch(r"\s*(short\s+ret\s*=\s*EW_OK|bool\s+ok\s*=\s*true)\s*;\s*", line)
        if match:
            name = "ret" if match.group(1).startswith("short") else "ok"
            if not complete or name in seen:
                continue
            seen.add(name)
        lines.append(line)
    return "".join(lines)


def _core_cpp_from_payload(payload: Any) -> str:
    """Accept the canonical core_cpp key and common provider aliases."""
    if not isinstance(payload, dict):
        return ""
    for key in ("core_cpp", "fragment_cpp", "scenario_fragment", "cpp_code"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return strip_cpp_fence(value).strip()
    return ""


def _record_generation_response(state: WorkflowState, llm_client: LlmClient, *, stage: str) -> None:
    """Keep structured diagnostics without confusing them with API evidence."""
    if state.plan is None:
        return
    raw_text = getattr(llm_client, "last_json_text", "") or ""
    text = raw_text if isinstance(raw_text, str) else ""
    payload = getattr(llm_client, "last_json_payload", {}) or {}
    state.plan.rag_context.setdefault("code_generation_responses", []).append({
        "stage": stage,
        "response_chars": len(text),
        "raw_response": text,
        "response_keys": sorted(payload) if isinstance(payload, dict) else [],
        "parse_error": str(getattr(llm_client, "last_json_parse_error", "") or ""),
        "contains_cpp_marker": "core_cpp" in text or "fragment_cpp" in text,
        "contains_bare_call_marker": "call_expr" in text or "api_fragments" in text,
    })


def _ensure_parameter_combinations(plan: ExecutionPlan, contract: Any, targets: set[str]) -> None:
    """Reject only a completed empty domain, never a pending dependency."""
    # Direct recipe/unit callers may construct a partial plan before the
    # normal CodeGenerationAgent validation hand-off. Defer this guard until
    # that hand-off has produced its structured validation record.
    if "execution_plan_validation" not in plan.rag_context:
        return
    coverage = plan.rag_context.get("parameter_coverage_contract", {})
    entries = coverage.get("functions", {}) if isinstance(coverage, dict) else {}
    pending_targets: set[str] = set()
    for binding in getattr(plan, "runtime_bindings", ()) or ():
        if isinstance(binding, dict) and str(binding.get("target_api", "")).strip():
            pending_targets.add(str(binding["target_api"]).strip())
    for step in getattr(plan, "steps", ()) or ():
        params = step.parameters if isinstance(step.parameters, dict) else {}
        # Deterministic running-state recipes may declare a domain directly
        # on the step (for example cnc_accdecdly axis=1..N sourced from
        # cnc_sysinfo.sysinfo.axes) rather than using a graph runtime_binding.
        # This is a resolvable runtime domain, not an empty combination set.
        if (
            str(step.protocol_function or "").strip() == "cnc_accdecdly"
            and str(params.get("axis_domain", "")).strip() == "1..N"
            and str(params.get("axis_source", "")).strip()
        ):
            pending_targets.add("cnc_accdecdly")
        for binding in params.get("runtime_bindings", ()) or ():
            if isinstance(binding, dict) and str(binding.get("target_api", "")).strip():
                pending_targets.add(str(binding["target_api"]).strip())
    runtime_results = plan.rag_context.get("runtime_parameter_bindings", {})
    runtime_results = runtime_results if isinstance(runtime_results, dict) else {}
    for target in sorted(targets):
        runtime_result = runtime_results.get(target)
        if target in pending_targets and runtime_result is None:
            # The first generation pass is allowed to contain the prerequisite
            # call. The target is resolved after Executor records its output.
            continue
        if target in pending_targets and isinstance(runtime_result, dict):
            status = str(runtime_result.get("status", "")).strip()
            if status in {"pending_dependencies", "source_pending"}:
                continue
            if runtime_result.get("success") is False:
                errors = runtime_result.get("errors", [])
                detail = "; ".join(
                    str(item.get("message", item)) if isinstance(item, dict) else str(item)
                    for item in errors
                ) or "runtime parameter binding failed"
                raise ValueError(f"PARAMETER_BINDING_FAILED: {target}; {detail}")
            if status == "empty_result" or runtime_result.get("empty_result"):
                raise ValueError(f"PARAMETER_COMBINATIONS_EMPTY: {target}; filtered legal domain is empty")
        entry = entries.get(target, {}) if isinstance(entries, dict) else {}
        dimensions = entry.get("dimensions", []) if isinstance(entry, dict) else []
        cases = entry.get("cases", []) if isinstance(entry, dict) else []
        strategies = entry.get("planner_strategies", []) if isinstance(entry, dict) else []
        if dimensions and not cases and not strategies and target not in pending_targets:
            raise ValueError(
                f"PARAMETER_COMBINATIONS_EMPTY: {target}; parameter domain/selection is unresolved "
                "(no bound values, legal cases, or deterministic selection strategy)"
            )


def active_sdk_symbol_evidence(functions: list[str] | set[str] | tuple[str, ...]) -> dict[str, str]:
    """Return prototype evidence for planned APIs from the active header.

    This is deliberately independent of the optional SDK type-candidate index:
    an empty type-candidate list does not mean that the API itself is missing.
    """
    header = default_focas_header_dir() / "Fwlib32.h"
    if not header.is_file():
        return {str(name): "header_unavailable" for name in functions}
    try:
        text = header.read_text(encoding="latin-1")
    except OSError:
        return {str(name): "header_unreadable" for name in functions}
    result: dict[str, str] = {}
    for name in sorted({str(item).strip() for item in functions if str(item).strip()}):
        if re.search(rf"\b{re.escape(name)}\s*\(", text):
            result[name] = "confirmed_header_prototype"
        else:
            result[name] = "not_found_in_active_header"
    return result


def generate_capture_fragment(
    state: WorkflowState, steps: list[PlanStep], nc_program: str, llm_client: LlmClient,
    *, previous: str = "", diagnostics: str = "",
) -> str:
    """Generate the production fragment from ordered, SDK-scoped API pieces.

    A configured LLM receives the complete ExecutionPlan once and returns an
    ordered bundle of statements-only fragments, one per planned API step. The
    fixed harness and :class:`ValidateFocasCppTool` remain the owners of
    connection, capture, logging, ABI checks and compilation. The deterministic
    recipe path is retained only for offline/legacy callers that do not
    configure an LLM; it is no longer the production admission path and its
    missing-recipe error is never used when the model is available.
    """
    assert state.plan is not None
    if not diagnostics and state.errors and state.repair_history and state.repair_history[-1].get("repair_stage") == "repair_code":
        diagnostics = "; ".join(str(item) for item in state.errors[-10:])
    contract = execution_contract(state.plan, state.request.permissions.get("coverage_current_batch_functions")
                                  or state.request.permissions.get("required_api_functions") or ())
    targets = set(contract.targets)
    from ..parameter_coverage import attach_parameter_contract
    attach_parameter_contract(state.plan, contract.targets)
    _ensure_parameter_combinations(state.plan, contract, targets)
    if llm_client.enabled:
        return generate_api_fragments_with_llm(
            state,
            steps,
            nc_program,
            llm_client,
            previous=previous,
            diagnostics=diagnostics,
        )
    raise ValueError(
        "CODEGEN_LLM_UNAVAILABLE: production API fragment generation requires "
        "a configured CodeGenerationAgent model; no deterministic recipe fallback is used"
    )

def validate_scenario_fragment(fragment: str) -> list[str]:
    """Enforce the narrow source boundary between the model and the harness."""
    import re

    if "CaptureContext& ctx" in fragment:
        return validate_capture_fragment(fragment)
    forbidden = []
    # The harness owns the single scenario function and its state variables.
    # Model output must contain statements only; duplicate wrappers/declarations
    # otherwise compile as C2374/C2086 after an otherwise valid repair.
    if len(re.findall(r"\bshort\s+run_generated_scenario\s*\(", fragment)) > 1:
        forbidden.append("duplicate run_generated_scenario wrapper")
    core = fragment
    core_start = fragment.find("// SCENARIO_CORE_BEGIN")
    core_end = fragment.find("// SCENARIO_CORE_END")
    if core_start >= 0 and core_end > core_start:
        core = fragment[core_start + len("// SCENARIO_CORE_BEGIN"):core_end]
    if re.search(r"(?m)^\s*(?:short\s+ret|bool\s+ok)\s*=", core):
        forbidden.append("harness-owned ret/ok redeclaration")
    checks = {
        r"\bmain\s*\(": "main()",
        r"\b(?:struct|class)\s+FocasApi\b": "FocasApi definition",
        r"\bLoadLibraryW\s*\(": "LoadLibraryW",
        r"\bGetProcAddress\s*\(": "GetProcAddress",
        r"\bcnc_allclibhndl3\b": "cnc_allclibhndl3",
        r"\bcnc_freelibhndl\b": "cnc_freelibhndl",
        r"(?m)^\s*#\s*include\b": "#include",
        r"\btypedef\s+struct\b": "official structure redefinition",
        r"\b(?:struct|class)\s+(?:ODB|POSELM|IODB)\w*\b": "official structure redefinition",
    }
    for pattern, label in checks.items():
        if re.search(pattern, fragment, re.IGNORECASE):
            forbidden.append(label)
    if not re.search(r"\bshort\s+run_generated_scenario\s*\(", fragment):
        forbidden.append("run_generated_scenario signature")
    if not re.search(r"\bFocasApi\s*&\s*api\b", fragment):
        forbidden.append("FocasApi& api parameter")
    if not re.search(r"\bunsigned\s+short\s+handle\b", fragment):
        forbidden.append("unsigned short handle parameter")
    if not forbidden:
        return []
    return [
        "BLOCKING: scenario_fragment.inc violates the fragment boundary: "
        + ", ".join(sorted(set(forbidden)))
        + "."
    ]


def _split_cpp_call_arguments(source: str, open_index: int) -> tuple[list[str], int] | None:
    """Split one C++ call at top-level commas, preserving lambda/map bodies."""
    args: list[str] = []
    start = open_index + 1
    paren = brace = bracket = 0
    i = start
    state = "code"
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if state == "line_comment":
            if ch in "\r\n":
                state = "code"
            i += 1
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                i += 2
            else:
                i += 1
            continue
        if state in {"string", "char"}:
            if ch == "\\":
                i += 2
                continue
            if (state == "string" and ch == '"') or (state == "char" and ch == "'"):
                state = "code"
            i += 1
            continue
        if ch == "/" and nxt == "/":
            state = "line_comment"
            i += 2
            continue
        if ch == "/" and nxt == "*":
            state = "block_comment"
            i += 2
            continue
        if ch == '"':
            state = "string"
            i += 1
            continue
        if ch == "'":
            state = "char"
            i += 1
            continue
        if ch == "(":
            paren += 1
        elif ch == ")":
            if paren == 0 and brace == 0 and bracket == 0:
                args.append(source[start:i].strip())
                return args, i
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        elif ch == "," and paren == 0 and brace == 0 and bracket == 0:
            args.append(source[start:i].strip())
            start = i + 1
        i += 1
    return None


def validate_capture_context_call_contract(fragment: str) -> list[str]:
    """Check the fixed six-argument CaptureContext::call contract before MSVC."""
    diagnostics: list[str] = []
    masked = strip_cpp_comments_and_literals(fragment)
    call_pattern = re.compile(r"\bctx\s*\.\s*call\s*\(")
    for match in call_pattern.finditer(masked):
        open_index = match.end() - 1
        parsed = _split_cpp_call_arguments(fragment, open_index)
        line = fragment.count("\n", 0, match.start()) + 1
        if parsed is None:
            diagnostics.append(
                "BLOCKING: CAPTURE_CALL_SIGNATURE_PARSE_ERROR: "
                f"line={line}; could not find the closing parenthesis for ctx.call"
            )
            continue
        args, _close_index = parsed
        api_match = re.search(r"\b((?:cnc|pmc|ds|ncguide)_\w+)\b", " ".join(args[:3]))
        api_name = api_match.group(1) if api_match else "unknown"
        step_arg = args[0].strip() if args else ""
        step_literal = re.fullmatch(r'"([^"\\]*(?:\\.[^"\\]*)*)"', step_arg)
        step_id = step_literal.group(1) if step_literal else (step_arg or "unknown")
        if len(args) != 6:
            diagnostics.append(
                "BLOCKING: CAPTURE_CALL_SIGNATURE_MISMATCH: "
                f"api={api_name}; step_id={step_id}; expected=6; actual={len(args)}; line={line}"
            )
            continue
        function_match = re.fullmatch(r"\s*api\s*\.\s*((?:cnc|pmc|ds)_\w+)\s*", args[3])
        if not function_match:
            diagnostics.append(
                "BLOCKING: CAPTURE_CALL_FUNCTION_ARGUMENT_INVALID: "
                f"api={api_name}; step_id={step_id}; line={line}; "
                "argument_4 must be the loaded api.<function> member"
            )
        invoked = re.search(r"\breturn\s+api\s*\.\s*((?:cnc|pmc|ds)_\w+)\s*\(", args[4])
        if not invoked:
            diagnostics.append(
                "BLOCKING: CAPTURE_CALL_INVOKER_INVALID: "
                f"api={api_name}; step_id={step_id}; line={line}; "
                "argument_5 must invoke the real SDK function"
            )
        if function_match and invoked:
            function_name = function_match.group(1)
            invoked_name = invoked.group(1)
            if function_name != invoked_name:
                diagnostics.append(
                    "BLOCKING: CAPTURE_CALL_API_NAME_MISMATCH: "
                    f"api={api_name}; step_id={step_id}; line={line}; "
                    f"argument_4={function_name}; argument_5={invoked_name}"
                )
            api_literal = re.fullmatch(r'\s*"((?:cnc|pmc|ds)_\w+)"\s*', args[1])
            if api_literal and api_literal.group(1) != function_name:
                diagnostics.append(
                    "BLOCKING: CAPTURE_CALL_API_NAME_MISMATCH: "
                    f"api={api_literal.group(1)}; step_id={step_id}; line={line}; "
                    f"argument_4={function_name}"
                )
    return diagnostics


def _validate_api_fragment_statements(fragment: str) -> list[str]:
    """Reject wrapper and ownership leaks before rendering or compiling."""
    source = str(fragment or "")
    masked = strip_cpp_comments_and_literals(source)
    forbidden: list[str] = []
    checks = {
        r"(?m)^\s*#\s*(?:include|define|pragma)\b": "preprocessor directive",
        r"\bmain\s*\(": "main() wrapper",
        r"\b(?:LoadLibraryW|GetProcAddress)\s*\(": "loader ownership",
        r"\b(?:cnc_allclibhndl3|cnc_freelibhndl)\b": "connection lifecycle",
        r"\b(?:typedef\s+struct|struct\s+FocasApi|class\s+FocasApi)\b": "type definition",
    }
    for pattern, label in checks.items():
        if re.search(pattern, masked, re.IGNORECASE):
            forbidden.append(label)
    if re.search(
        r"\b(?:static\s+)?(?:short|int|long|bool|void|std::string)\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{",
        masked,
    ):
        forbidden.append("additional function definition")
    if not forbidden:
        return []
    return [
        "BLOCKING: API_FRAGMENT_STATEMENTS_ONLY: "
        + ", ".join(sorted(set(forbidden)))
        + "."
    ]


def generate_scenario_core_with_llm(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
    selected_functions: list[str],
    *,
    context_steps: list[PlanStep] | None = None,
    repair_diagnostics: str = "",
    existing_fragment: str = "",
) -> str:
    """Ask the model for statements for the selected API fragment only.

    ``executable_steps`` identifies the fragment being generated.  The full
    ``context_steps`` list is supplied as read-only ordering/dependency
    context, so a later fragment can consume variables produced by an earlier
    planned call without inventing a return value.
    """
    assert state.plan is not None
    # Compatibility callers now use the same bare-call bundle path as
    # production. The legacy free-form core prompt below is intentionally
    # unreachable and is retained only for source compatibility during the
    # migration of older integrations.
    return generate_api_fragments_with_llm(
        state,
        executable_steps,
        nc_program,
        llm_client,
        previous=existing_fragment,
        diagnostics=repair_diagnostics,
        wrap=False,
    )
    abi = official_focas_abi_context(selected_functions)
    sdk_types = sdk_type_context_for_abi(abi)
    required_semantics = []
    if state.plan.scenario_type == "tool_management" and any(
        name in {"cnc_rdtool", "cnc_rdtofs", "cnc_rdtofsr", "cnc_rdtofsinfo", "cnc_rdtofsinfo2"}
        for name in selected_functions
    ):
        required_semantics.append(
            "tool_data_access: make a real call to one of cnc_rdtool, cnc_rdtofs, cnc_rdtofsr, "
            "cnc_rdtofsinfo, or cnc_rdtofsinfo2; for cnc_rdtool use IODBTLMNG and check its return code."
        )
    history_target = selected_functions[0] if selected_functions else state.plan.target_api or next(
        (name for step in executable_steps if getattr(step, "role", "") == "target"
         for name in protocol_function_names(step.protocol_function)),
        "",
    )
    historical_examples = retrieve_verified_api_examples(
        history_target,
        output_dir=Path(state.plan.rag_context.get("output_dir", "")) if state.plan.rag_context.get("output_dir") else None,
        limit=3,
        sdk_identity=str(sdk_environment_identity(
            default_focas_header_dir(),
            Path(os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir()))) / "Fwlib32.dll",
        )),
    )
    state.plan.rag_context["verified_api_examples"] = historical_examples
    system_prompt = (
        "You are CodeGenerationAgent generating only the scenario core for a fixed FANUC FOCAS C++17 template. "
        "Return JSON only with key core_cpp. Do not return includes, structs, typedefs, templates, main(), "
        "LoadLibrary/GetProcAddress, connection, cleanup, or markdown. The surrounding template already provides "
        "FocasApi api, unsigned short handle, bool ok, std::string nc_payload, std::vector<char> nc_buffer, "
        "long nc_length, short ret, and check_rc(ret,name). Use only api.<official_function>, official Fwlib32.h "
        "structures, and the Planner-selected APIs. Never emit run_generated_scenario, short ret, or bool ok; "
        "the fixed wrapper owns them. For cnc_absolute/cnc_absolute2/cnc_machine/cnc_relative/cnc_relative2/cnc_distance, "
        "the length argument is the documented ODBAXIS data-block size; for 0i-D all-axis reads use 4 + 4 * MAX_AXIS (MAX_AXIS=32), never a guessed axis count. Do not return early from the core; update ok and continue so "
        "the fixed cleanup always runs. For cnc_rdposition, use ODBPOS positions[count] and scalar "
        "positions[i].abs.data/mach.data/rel.data/dist.data; never POSELM.data[]. For cnc_actf, use ODBACT "
        "and actual_feed.data. For cnc_download3, use nc_buffer.data() and nc_length; the fixed template already "
        "provides the correctly framed NC payload. Every Planner semantic operation is mandatory and must have a "
        "real executable call in this fragment, never a comment, dead branch, or declaration-only reference. "
        "When using CaptureContext, call it only as ctx.call(step_id, api_name, parameter_text, api.<function>, "
        "invoker_lambda, output_lambda); the four-argument legacy form is forbidden. "
        "Resolve every selected API to its concrete api.<function_name> member from the supplied Fwlib32.h prototype; "
        "api.function, target_api, TODO and ellipsis are forbidden output tokens. "
        "Keep the fragment concise and compilable."
    )
    ordered_context_steps = context_steps if context_steps is not None else executable_steps
    fragment_steps = steps_for_prompt(executable_steps)
    user_prompt = (
        f"Task: {state.request.description}\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"ExecutionPlan order and dependency context: {steps_for_prompt(ordered_context_steps)}\n"
        f"Current API fragment steps (implement only these now): {fragment_steps}\n"
        f"Official APIs for this fragment: {selected_functions}\n"
        f"Official ABI excerpts:\n{abi}\n"
        f"SDK type definitions from the active header:\n{sdk_types}\n"
        f"Parameter/API rules from the local knowledge context:\n{state.plan.rag_context.get('code_generation_knowledge', [])}\n"
        f"Exact API contracts and runtime bindings:\n{state.plan.rag_context.get('exact_api_contracts', {})}\n"
        f"Runtime execution evidence (only values already returned by upstream calls):\n{getattr(state, 'execution_context', {})}\n"
        f"Verified historical examples (wire-matched successful calls only; values are comparison evidence, not authority):\n{json.dumps(historical_examples, ensure_ascii=False, indent=2)}\n"
        f"Required semantic operations (must be implemented): {required_semantics}\n"
        f"Agent Soul:\n{code_generator_soul_context(state.plan.rag_context)}\n"
        f"Traffic Relationship Graph context:\n{state.plan.rag_context.get('traffic_relationships', {})}\n"
        f"NC program is already embedded by the shell:\n{nc_program}\n"
        + (f"Repair diagnostics for this fragment:\n{repair_diagnostics}\nExisting fragment to repair:\n{existing_fragment}\n" if repair_diagnostics or existing_fragment else "")
        + "Initialize nc_payload/nc_buffer/nc_length inside the fragment when an upload API needs them, using the supplied NC program. "
        "When a verified historical example is present, reuse its call shape, structure access, prerequisite order, and log format only after checking them against the current Fwlib32.h prototype and SDK type index; never copy its device-specific values blindly. "
        "Implement only the current API fragment, preserve the ExecutionPlan order, use runtime-bound variables from earlier steps, and never invent an upstream return value. Return the smallest complete core fragment that implements the current API step."
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    _record_generation_response(state, llm_client, stage="scenario_core_generation")
    core = _core_cpp_from_payload(payload)
    if not core:
        retry_payload = llm_client.invoke_json(
            "Return JSON only with a non-empty core_cpp string containing the requested FOCAS statements. "
            "Do not return markdown or an empty value.",
            user_prompt,
        )
        _record_generation_response(state, llm_client, stage="scenario_core_generation_retry")
        core = _core_cpp_from_payload(retry_payload)
    if not core:
        keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        category = "model_no_code" if not keys else "response_format_error"
        raise ValueError("CODEGEN_CORE_EMPTY[" + category + "]: CodeGenerationAgent scenario-core generation returned an empty core_cpp fragment (response_keys=" + ",".join(keys) + ").")
    return _normalize_model_core(strip_cpp_fence(core))


def _validate_api_fragment_call(fragment: str, functions: list[str]) -> None:
    """Require each requested API fragment to contain a real SDK call."""
    normalized = _normalize_model_core(fragment)
    boundary_errors = _validate_api_fragment_statements(normalized)
    contract_errors = validate_capture_context_call_contract(normalized)
    if boundary_errors:
        raise ValueError("; ".join(boundary_errors))
    if contract_errors:
        raise ValueError("; ".join(contract_errors))
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", normalized, flags=re.DOTALL)
    missing = [
        name for name in functions
        if not re.search(rf"\b(?:api\.)?{re.escape(name)}(?:_fn|_p)?\s*\(", source)
    ]
    if missing:
        raise ValueError(
            "API_FRAGMENT_MISSING_CALL: " + ", ".join(missing)
            + "; generated fragment must call the requested SDK API"
        )
    # Apply the same SDK declaration checks used by ValidateFocasCppTool while
    # this fragment is still isolated. The final validator remains mandatory;
    # this early check only prevents invalid code from being rendered/compiled.
    try:
        from ..integrations.ncguide import default_focas_header_dir
        from ..sdk_type_index import (
            load_or_build_sdk_type_index,
            validate_sdk_type_declarations,
            validate_struct_member_access,
        )
        header = default_focas_header_dir() / "Fwlib32.h"
        if header.is_file():
            index = load_or_build_sdk_type_index(
                header, Path("rag_indexes/focas/sdk_type_index_0iD.json")
            )
            sdk_errors = validate_struct_member_access(normalized, index)
            sdk_errors.extend(validate_sdk_type_declarations(normalized, index))
            if sdk_errors:
                raise ValueError("; ".join(sdk_errors))
    except ImportError:
        # The final production validator remains authoritative if the optional
        # SDK index module is unavailable in a stripped-down test installation.
        pass


def generate_api_fragments_with_llm(
    state: WorkflowState,
    steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
    *,
    previous: str = "",
    diagnostics: str = "",
    wrap: bool = True,
) -> str:
    """Generate an ordered locally assembled bundle with one model invocation.

    The model returns only bare API calls. The SDK assembler owns declarations,
    parameter logs, output serializers, and the fixed ``ctx.call`` wrapper.
    """
    assert state.plan is not None
    fragment_rows: list[tuple[PlanStep, list[str]]] = []
    for step in steps:
        names = [name for name in protocol_function_names(step.protocol_function) if name not in SESSION_APIS]
        if names:
            fragment_rows.append((step, names))

    if not fragment_rows:
        raise ValueError("CODEGEN_NO_API_FRAGMENTS: ExecutionPlan contains no API operation for CodeGenerationAgent")

    cache = state.plan.rag_context.setdefault("codegen_api_fragments", {})
    if not isinstance(cache, dict):
        cache = {}
        state.plan.rag_context["codegen_api_fragments"] = cache
    elif not previous and not diagnostics and not (
        state.repair_history and state.repair_history[-1].get("repair_stage") == "repair_code"
    ):
        # A fresh generation must not reuse fragments left by an earlier run
        # that happened to reuse the same in-memory ExecutionPlan.
        cache.clear()

    mentioned = set(re.findall(r"\b(?:cnc|pmc)_\w+\b", str(diagnostics)))
    mentioned_step_ids = set(re.findall(r"\bS[A-Za-z0-9_.-]+\b", str(diagnostics)))
    repair_step_ids = {
        step.step_id
        for step, names in fragment_rows
        if mentioned.intersection(names)
    }
    if diagnostics and not repair_step_ids:
        # Prefer an explicit step marker when a compiler/validator reports
        # one.  If only a generic target diagnostic is available, repair the
        # planned target step; never discard unrelated cached fragments.
        repair_step_ids = {
            step.step_id
            for step, _names in fragment_rows
            if str(step.step_id) in mentioned_step_ids
        }
    if diagnostics and not repair_step_ids:
        target_name = str(getattr(state.plan, "target_api", "") or "").strip()
        repair_step_ids = {
            step.step_id
            for step, names in fragment_rows
            if target_name and target_name in names
        }
    if diagnostics and not repair_step_ids and len(fragment_rows) == 1:
        # A single-fragment plan has an unambiguous repair owner even when a
        # mock/compiler diagnostic omitted both the API name and step marker.
        repair_step_ids = {fragment_rows[0][0].step_id}
    if diagnostics and not repair_step_ids:
        raise ValueError(
            "API_FRAGMENT_REPAIR_SCOPE_UNKNOWN: diagnostics do not identify a planned API or step; "
            "refusing to regenerate unrelated fragments"
        )

    requested_rows = [
        (step, names)
        for step, names in fragment_rows
        if not diagnostics or step.step_id in repair_step_ids or not cache.get(str(step.step_id))
    ]
    if requested_rows:
        requested_ids = {str(step.step_id) for step, _names in requested_rows}
        try:
            generated = generate_api_fragment_bundle_with_llm(
                state,
                steps,
                requested_rows,
                nc_program,
                llm_client,
                diagnostics=diagnostics,
                existing_fragments={
                    str(key): value.get("call_specs", [])
                    for key, value in cache.items()
                    if isinstance(value, dict)
                    and str(key) in requested_ids
                    and value.get("call_specs")
                },
            )
        except Exception as exc:
            api_label = ",".join(
                name for _step, names in requested_rows for name in names
            )
            raise ValueError(f"API_FRAGMENT_GENERATION_FAILED[{api_label}]: {exc}") from exc
        fragment_errors: list[str] = []
        for step, names in requested_rows:
            key = str(step.step_id)
            core = generated.get(key, "")
            if not core:
                fragment_errors.append(
                    f"API_FRAGMENT_MISSING_FRAGMENT[{','.join(names)}]: model did not return step_id={key}"
                )
                continue
            try:
                _validate_api_fragment_call(core, names)
            except Exception as exc:
                fragment_errors.append(
                    f"API_FRAGMENT_GENERATION_FAILED[{','.join(names)}]: {exc}"
                )
                continue
            cache[key] = {"step_id": key, "functions": names, "core_cpp": core}
        if fragment_errors:
            raise ValueError("; ".join(fragment_errors))

    rows: list[str] = []
    for step, names in fragment_rows:
        cached = cache.get(str(step.step_id))
        if not isinstance(cached, dict) or not str(cached.get("core_cpp", "")).strip():
            raise ValueError(
                f"API_FRAGMENT_MISSING_FRAGMENT[{','.join(names)}]: no cached fragment for step_id={step.step_id}"
            )
        rows.append(str(cached["core_cpp"]))

    combined = "\n\n".join(item for item in rows if item.strip())
    if not combined.strip():
        raise ValueError("CODEGEN_CORE_EMPTY[no_api_fragments]: no API fragment was generated")
    state.plan.rag_context["code_generation_fragments_order"] = [
        str(step.step_id) for step, _names in fragment_rows
    ]
    normalized = normalize_capture_core(combined)
    return wrap_capture_fragment(normalized) if wrap else normalized


def _runtime_symbols_before_step(all_steps: list[PlanStep], step_id: str) -> dict[str, str]:
    """Expose only deterministic output symbols from earlier plan steps."""
    symbols: dict[str, str] = {}
    for step in all_steps:
        if str(step.step_id) == str(step_id):
            break
        for api_name in protocol_function_names(step.protocol_function):
            metadata = api_metadata(api_name)
            for parameter in metadata.output_parameters:
                if "*" in parameter.type:
                    symbols[parameter.name] = output_symbol_name(step.step_id, parameter.name)
                    symbols[f"{step.step_id}.{parameter.name}"] = output_symbol_name(step.step_id, parameter.name)
    return symbols


def generate_api_fragment_bundle_with_llm(
    state: WorkflowState,
    all_steps: list[PlanStep],
    requested_rows: list[tuple[PlanStep, list[str]]],
    nc_program: str,
    llm_client: LlmClient,
    *,
    diagnostics: str = "",
    existing_fragments: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Call the model once and parse an ordered ``step_id`` fragment bundle."""
    assert state.plan is not None
    requested_functions = [name for _step, names in requested_rows for name in names]
    all_functions = [
        name for step in all_steps
        for name in protocol_function_names(step.protocol_function)
        if name not in SESSION_APIS
    ]
    all_functions = list(dict.fromkeys(all_functions))
    abi = official_focas_abi_context(all_functions)
    sdk_types = sdk_type_context_for_abi(abi)
    history: dict[str, Any] = {}
    for name in requested_functions:
        history[name] = retrieve_verified_api_examples(
            name,
            output_dir=Path(state.plan.rag_context.get("output_dir", ""))
            if state.plan.rag_context.get("output_dir") else None,
            limit=3,
            sdk_identity=str(sdk_environment_identity(
                default_focas_header_dir(),
                Path(os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir()))) / "Fwlib32.dll",
            )),
        )
    state.plan.rag_context["verified_api_examples"] = history
    requested_prompt_rows = []
    for step, names in requested_rows:
        requested_prompt_rows.append({
            "step_id": str(step.step_id),
            "apis": names,
            "phase": step.phase,
            "action": step.action,
            "parameters": step.parameters,
            "available_runtime_symbols": _runtime_symbols_before_step(all_steps, step.step_id),
        })
    existing = existing_fragments or {}
    system_prompt = (
        "You are CodeGenerationAgent generating bare FANUC FOCAS API expressions for a fixed C++17 harness. "
        "Return JSON only with key api_fragments. Each item must contain step_id and calls; each call must contain "
        "api_name, call_expr, parameter_values, and optional struct_values. parameter_values may contain only "
        "formal SDK parameter names; structure or union fields must be nested under struct_values using the pointer "
        "parameter name as the root. Use an empty parameter_values object only when the API has no named inputs. "
        "The call_expr must be exactly one bare expression such as "
        "api.cnc_statinfo(handle, &__out_statinfo). The local assembler will create SDK declarations, parameter logs, "
        "output serializers, lambdas, and the six-argument ctx.call wrapper. "
        "Never emit ctx.call, lambda syntax, semicolons, declarations, main(), includes, structs, typedefs, connection, "
        "cleanup, logging code, Markdown, api.function(...), target_api, TODO, or ellipsis. "
        "Use only the supplied Fwlib32.h prototype and SDK type context. Current structure pointers must use the placeholder "
        "&__out_<formal_parameter_name>; earlier output values must use only the available_runtime_symbols supplied for "
        "that step. Do not invent an upstream output or replace a missing runtime value with a guessed literal. "
        "Preserve the complete ExecutionPlan order and return at least one call entry for each planned API. "
        "When parameter combinations contain multiple cases, return one call entry per case in deterministic order; "
        "repeated api_name values within one step are valid and must not be collapsed. "
        "Parameter expressions may contain documented constants, literals, handle, and previously bound runtime symbols. "
        "For a structure pointer, put field values in struct_values, for example "
        "struct_values:{\"buffer\":{\"type\":1,\"data\":[12]}}; do not emit buffer.type or buffer.data[0] as parameter keys. "
        "The local assembler, not the model, owns return-code handling, input/output CSV format and PCAP boundaries. "
        "Repair only requested step_ids and leave unrelated cached calls unchanged."
    )
    if state.plan.rag_context.get("agent_soul"):
        system_prompt += "\nAgent operating guidance:\n" + str(state.plan.rag_context["agent_soul"])
    user_prompt = (
        f"Task: {state.request.description}\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"Target environment: {state.request.target_environment}\n"
        f"Complete ExecutionPlan in authoritative order:\n{steps_for_prompt(all_steps)}\n"
        f"Fragments to generate or repair now:\n{requested_prompt_rows}\n"
        f"Official ABI declarations and referenced structures:\n{abi}\n"
        f"SDK type index context:\n{sdk_types}\n"
        f"Parameter/rule knowledge:\n{state.plan.rag_context.get('code_generation_knowledge', [])}\n"
        f"Exact API contracts:\n{state.plan.rag_context.get('exact_api_contracts', {})}\n"
        f"Traffic Relationship Graph context:\n{state.plan.rag_context.get('traffic_relationships', {})}\n"
        f"Runtime execution evidence (only successful upstream outputs):\n{getattr(state, 'execution_context', {})}\n"
        f"Verified same-API examples:\n{json.dumps(history, ensure_ascii=False, indent=2)}\n"
        f"NC program already embedded by the harness:\n{nc_program}\n"
        f"Repair diagnostics:\n{diagnostics}\n"
        f"Existing fragments for the requested steps:\n{json.dumps(existing, ensure_ascii=False, indent=2)}\n"
        "Return JSON with this shape: {\"api_fragments\":[{\"step_id\":\"S001\",\"calls\":[{\"api_name\":\"cnc_statinfo\",\"call_expr\":\"api.cnc_statinfo(handle, &__out_statinfo)\",\"parameter_values\":{},\"struct_values\":{}}]}]}"
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    _record_generation_response(state, llm_client, stage="api_fragment_bundle_generation")
    rows = payload.get("api_fragments") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        retry_payload = llm_client.invoke_json(
            "Return JSON only with api_fragments. Include every requested step_id exactly once; each call must contain one bare api.<function>(...) expression and no ctx.call.",
            user_prompt,
        )
        _record_generation_response(state, llm_client, stage="api_fragment_bundle_generation_retry")
        payload = retry_payload
        rows = payload.get("api_fragments") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("CODEGEN_FRAGMENT_BUNDLE_FORMAT: expected JSON api_fragments array of bare calls")
    expected = {str(step.step_id): names for step, names in requested_rows}
    result: dict[str, str] = {}
    seen_step_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        step_id = str(row.get("step_id", "")).strip()
        if step_id not in expected:
            raise ValueError(f"CODEGEN_FRAGMENT_BUNDLE_UNPLANNED_STEP: {step_id or '<missing>'}")
        if step_id in seen_step_ids:
            raise ValueError(f"CODEGEN_FRAGMENT_BUNDLE_DUPLICATE_STEP: step_id={step_id}")
        seen_step_ids.add(step_id)
        returned_apis = row.get("apis", expected[step_id])
        if not isinstance(returned_apis, list) or set(str(item) for item in returned_apis) != set(expected[step_id]):
            raise ValueError(
                f"CODEGEN_FRAGMENT_BUNDLE_API_MISMATCH: step_id={step_id}; "
                f"expected={expected[step_id]}; returned={returned_apis}"
            )
        if row.get("core_cpp") or row.get("fragment_cpp"):
            raise ValueError(
                f"CODEGEN_MODEL_OUTPUT_CONTRACT[step={step_id}]: core_cpp is forbidden; return bare call_expr"
            )
        calls = row.get("calls")
        if calls is None and row.get("call_expr"):
            calls = [{
                "api_name": row.get("api_name") or (expected[step_id][0] if len(expected[step_id]) == 1 else ""),
                "call_expr": row.get("call_expr"),
                "parameter_values": row.get("parameter_values", {}),
                "struct_values": row.get("struct_values", {}),
            }]
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"CODEGEN_CALLS_EMPTY[step={step_id}]: return calls with bare call_expr entries")
        returned_call_names: list[str] = []
        assembled: list[str] = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError(f"CODEGEN_CALL_SPEC_INVALID[step={step_id}]: call entry must be an object")
            api_name = str(call.get("api_name", "")).strip()
            call_expr = str(call.get("call_expr", "")).strip()
            if api_name not in expected[step_id]:
                raise ValueError(
                    f"CODEGEN_CALL_API_MISMATCH: step_id={step_id}; expected={expected[step_id]}; returned={api_name}"
                )
            parameter_values = call.get("parameter_values", {})
            if not isinstance(parameter_values, dict):
                raise ValueError(f"CODEGEN_CALL_PARAMETERS_INVALID: step_id={step_id}; api={api_name}")
            struct_values = call.get("struct_values", {})
            if not isinstance(struct_values, dict):
                raise ValueError(f"CODEGEN_STRUCT_VALUES_INVALID: step_id={step_id}; api={api_name}")
            try:
                # Each combination needs a unique local output object name,
                # while the externally visible step_id remains the plan's
                # step_id for logging and PCAP correlation.
                combination_index = len(assembled) + 1
                fragment, _outputs = assemble_capture_call(
                    step_id=step_id,
                    api_name=api_name,
                    call_expr=call_expr,
                    parameter_values=parameter_values,
                    struct_values=struct_values,
                    available_symbols=_runtime_symbols_before_step(all_steps, step_id),
                    symbol_step_id=(step_id if combination_index == 1 else f"{step_id}_C{combination_index:03d}"),
                )
            except SdkCallAssemblyError as exc:
                raise ValueError(f"CODEGEN_CALL_ASSEMBLY_FAILED[step={step_id},api={api_name}]: {exc}") from exc
            returned_call_names.append(api_name)
            assembled.append(fragment)
        if not set(expected[step_id]).issubset(set(returned_call_names)):
            raise ValueError(
                f"CODEGEN_CALL_API_MISMATCH: step_id={step_id}; expected={expected[step_id]}; returned={returned_call_names}"
            )
        result[step_id] = "\n".join(assembled)
        # Persist each valid assembled step immediately. If a later step in
        # the same batch is malformed, repair can reuse this step unchanged.
        state.plan.rag_context.setdefault("codegen_api_call_specs", {})[step_id] = calls
        state.plan.rag_context.setdefault("codegen_api_fragments", {})[step_id] = {
            "step_id": step_id,
            "functions": list(expected[step_id]),
            "call_specs": calls,
            "core_cpp": result[step_id],
        }
    missing = [step_id for step_id in expected if step_id not in result]
    if missing:
        raise ValueError("CODEGEN_FRAGMENT_BUNDLE_MISSING_STEPS: " + ", ".join(missing))
    return result


def repair_scenario_fragment_after_compile_error(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    fragment: str,
    compile_error: str,
    llm_client: LlmClient,
) -> str:
    return repair_scenario_fragment(
        state, executable_steps, nc_program, fragment,
        "MSVC compiler diagnostics:\n" + compile_error,
        llm_client,
    )


def repair_scenario_fragment_after_contract_error(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    fragment: str,
    contract_error: str,
    llm_client: LlmClient,
) -> str:
    return repair_scenario_fragment(
        state, executable_steps, nc_program, fragment,
        "Contract diagnostics:\n" + contract_error,
        llm_client,
    )


def repair_scenario_fragment(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    fragment: str,
    diagnostics: str,
    llm_client: LlmClient,
) -> str:
    """Repair only scenario_fragment.inc; the harness is never regenerated."""
    assert state.plan is not None
    sdk_types = sdk_type_context_for_abi(official_focas_abi_context(
        [name for step in executable_steps for name in protocol_function_names(step.protocol_function)]
    ))
    prompt = (
        "You are repairing only a FANUC FOCAS scenario fragment. Return JSON only with key fragment_cpp. "
        "Return the complete fragment including exactly the function short run_generated_scenario(FocasApi& api, unsigned short handle). "
        "Do not return main(), FocasApi, LoadLibraryW, GetProcAddress, cnc_allclibhndl3, cnc_freelibhndl, #include, typedefs, "
        "or any official structure redefinition. The fixed harness owns connection, dynamic API binding, cleanup, and exception handling. "
        "Use only official Fwlib32.h types and the Planner APIs. POSELM.data is scalar; use positions[i].abs.data, never data[i]. "
        "Every generated ctx.call must use exactly six arguments: step_id, api_name, parameter_text, api.<function>, "
        "an invoker lambda returning api.<function>(...), and an output lambda. Never use the legacy four-argument form. "
        "For ToolDataAccess/tool_management, preserve or add a real call to one accepted official tool-data API "
        "(prefer the Planner-selected API, and for cnc_rdtool use IODBTLMNG); never satisfy it with a comment or dead code. "
        "Do not change the Planner API set. Keep NC payload local to the fragment if needed.\n\n"
        f"Task: {state.request.description}\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"Planner steps: {steps_for_prompt(executable_steps)}\n"
        f"NC program: {nc_program}\n"
        f"{diagnostics}\n"
        f"SDK type definitions from the active header:\n{sdk_types}\n"
        f"Current fragment:\n{fragment}"
    )
    payload = llm_client.invoke_json(prompt, prompt)
    repaired = str(payload.get("fragment_cpp", "")).strip()
    if not repaired:
        raise ValueError("CodeGenerationAgent fragment repair returned an empty fragment_cpp value.")
    repaired = strip_cpp_fence(repaired)
    fragment_diagnostics = validate_scenario_fragment(repaired)
    if fragment_diagnostics:
        raise ValueError("Repaired scenario fragment is invalid: " + "; ".join(fragment_diagnostics))
    return repaired


def nc_program_number(nc_program: str, default: int = 1000) -> int:
    import re

    for line in nc_program.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = re.match(r"\s*O\s*(\d+)", line, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return default


def nc_program_body_without_o(nc_program: str) -> str:
    lines = [
        line.strip()
        for line in nc_program.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip() and not line.strip().startswith("%")
    ]
    if lines and lines[0].upper().startswith("O"):
        lines = lines[1:]
    return "\n".join(lines)


def effective_nc_segment_count(nc_program: str) -> int:
    count = 0
    for line in nc_program.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("("):
            continue
        count += 1
    return max(count, 1)


def estimate_nc_runtime(nc_program: str) -> dict[str, Any]:
    """Estimate bounded NC execution time from modal feed and axis distance.

    This is a preflight estimate, not a claim about controller timing.  It is
    deliberately conservative: every executable line gets a small observation
    allowance, while motion time is computed from the line distance and the
    active feed.  Unknown-feed motion is reported instead of silently treating
    it as instantaneous.
    """
    position = {axis: 0.0 for axis in ("X", "Y", "Z", "A", "B", "C")}
    absolute = True
    feed: float | None = None
    segments: list[dict[str, Any]] = []
    total = 0.0
    unknown_feed = False
    for raw in nc_program.replace("\r", "").splitlines():
        line = re.sub(r"\([^)]*\)", "", raw).strip().upper()
        if not line or line.startswith("%") or line.startswith("O"):
            continue
        if re.search(r"\bG0?91\b", line):
            absolute = False
        elif re.search(r"\bG0?90\b", line):
            absolute = True
        feed_match = re.search(r"\bF([0-9]+(?:\.[0-9]+)?)", line)
        if feed_match:
            parsed_feed = float(feed_match.group(1))
            if parsed_feed > 0:
                feed = parsed_feed
        deltas: list[float] = []
        for axis in position:
            match = re.search(rf"\b{axis}(-?[0-9]+(?:\.[0-9]+)?)", line)
            if not match:
                continue
            value = float(match.group(1))
            delta = value if not absolute else value - position[axis]
            deltas.append(delta)
            position[axis] = position[axis] + delta if not absolute else value
        distance = sum(delta * delta for delta in deltas) ** 0.5
        if distance > 0:
            if feed is None:
                unknown_feed = True
                motion_seconds = 8.0
            else:
                motion_seconds = (distance / feed) * 60.0
            segment_seconds = max(0.8, motion_seconds) + 0.4
        else:
            segment_seconds = 0.35
        total += segment_seconds
        segments.append({"line": line, "distance": round(distance, 4), "feed": feed,
                         "estimated_seconds": round(segment_seconds, 3)})
    return {
        "estimated_seconds": round(total, 3),
        "segment_count": len(segments),
        "segments": segments,
        "unknown_feed_motion": unknown_feed,
    }


def nc_runtime_budget_seconds() -> float:
    try:
        configured = float(os.environ.get("SMP_MAX_NC_RUNTIME_SECONDS", "90"))
    except ValueError:
        configured = 90.0
    return max(10.0, min(configured, 3600.0))


def validate_nc_runtime_budget(nc_program: str, rag_context: dict[str, Any]) -> list[str]:
    estimate = estimate_nc_runtime(nc_program)
    budget = nc_runtime_budget_seconds()
    rag_context["nc_runtime_estimate"] = estimate
    rag_context["nc_runtime_budget_seconds"] = budget
    errors: list[str] = []
    if estimate["unknown_feed_motion"]:
        errors.append("NC runtime estimate cannot verify a motion line because no feed value is available")
    if estimate["estimated_seconds"] > budget:
        errors.append(
            "NC runtime estimate exceeds global budget: "
            f"estimated={estimate['estimated_seconds']:.1f}s;budget={budget:.1f}s; "
            "shorten motion distances or use a documented higher feed while preserving the target condition"
        )
    return errors


def planned_api_names_by_phase(steps: list[PlanStep]) -> dict[str, list[str]]:
    by_phase: dict[str, list[str]] = {"before": [], "during": [], "after": []}
    for step in steps:
        phase = step.phase if step.phase in by_phase else "during"
        for name in protocol_function_names(step.protocol_function):
            if name not in by_phase[phase]:
                by_phase[phase].append(name)
    return by_phase


def cpp_string_vector(values: list[str]) -> str:
    if not values:
        return "{}"
    return "{" + ", ".join(f'"{cpp_string_literal(value)}"' for value in values) + "}"


FIXED_NCGUIDE_APIS = {
    "cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2",
    "cnc_sysinfo",
    "cnc_rdalmmsg", "cnc_rdprogdir3", "cnc_dwnstart3", "cnc_download3",
    "cnc_dwnend3", "cnc_search", "cnc_rdprgnum", "cnc_actf",
    "cnc_rdposition", "cnc_rdsvmeter", "cnc_rdparam", "cnc_distance",
    "cnc_absolute", "cnc_absolute2", "cnc_machine", "cnc_relative",
    "cnc_relative2", "cnc_skip", "cnc_srvdelay", "cnc_accdecdly",
    "cnc_rddynamic", "cnc_rdaxisdata", "cnc_rd3dtooltip",
    "cnc_rdmdiprgstat", "cnc_rdmdipntr", "cnc_rdtool", "cnc_rdtofs",
    "cnc_rdtofsr", "cnc_rdtofsinfo", "cnc_rdtofsinfo2",
}


def use_fixed_ncguide_runner(state: WorkflowState, steps: list[PlanStep]) -> bool:
    """The fixed runner executes a complete Single Block motion lifecycle."""
    if state.plan is None or state.plan.scenario_type != "coordinate_motion":
        return False
    if "StartProgram" not in {step.interface_name for step in steps}:
        return False
    if not set(state.request.permissions.get("required_api_functions") or []).issubset(FIXED_NCGUIDE_APIS):
        return False
    return all(
        name in FIXED_NCGUIDE_APIS
        for step in steps
        for name in protocol_function_names(step.protocol_function)
        if name.startswith(("cnc_", "pmc_"))
    )


def render_fixed_ncguide_cpp_script(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
) -> str:
    """Render a deterministic C++ runner; LLM output is limited to NC payload text."""

    assert state.plan is not None
    requested_program = nc_program_number(nc_program, 5711)
    preferred_program = safe_nc_program_number(requested_program)
    body = nc_program_body_without_o(nc_program)
    expected_segments = effective_nc_segment_count(nc_program)
    # Coordinate Motion never requires a destructive full program-memory
    # cleanup. Keep delete-all available only to the Program Management fixed
    # runner and only when the existing explicit permission is present.
    allow_delete_all = bool(
        state.plan.scenario_type == "program_management"
        and state.request.permissions.get("allow_delete_all_programs")
    )
    planned_api_by_phase = planned_api_names_by_phase(executable_steps)
    delete_all_resolve = (
        '    auto cnc_delall_fn = reinterpret_cast<decltype(&::cnc_delall)>(GetProcAddress(focasDll, "cnc_delall"));\n'
        if allow_delete_all
        else ""
    )
    delete_all_missing_check = " || !cnc_delall_fn" if allow_delete_all else ""
    delete_all_block = (
        """
    if (!programAvailable) {
        LogHost(inputCsv, outputCsv, idx, "S007_DELETE_ALL_AUTHORIZED", "host_delete_all_gate",
                0, "DELETE_ALL_AUTHORIZED",
                "delete_all_authorized=true;collision_strategy=delete_all_authorized;preferred_program_number=" + std::to_string(preferredProgram));
        short delRet = CallFocas(inputCsv, outputCsv, idx, sniffer, "S007_DELETE_ALL", "cnc_delall",
            "delete_all_authorized=true;collision_strategy=delete_all_authorized",
            [&]() { return cnc_delall_fn(handle); },
            "delete_all_authorized=true;collision_strategy=delete_all_authorized");
        if (delRet != 0) {
            LogHost(inputCsv, outputCsv, idx, "PROGRAM_REPLACEMENT_FAILED", "host_lifecycle_gate",
                    delRet, "PROGRAM_REPLACEMENT_FAILED",
                    "delete_all_authorized=true;delete_all_failed=true;no_cycle_start=true");
            cleanup();
            return 6;
        }
        selectedProgram = preferredProgram;
        programAvailable = true;
    }
"""
        if allow_delete_all
        else """
    if (!programAvailable) {
        LogHost(inputCsv, outputCsv, idx, "TARGET_PROGRAM_EXISTS_NO_SLOT", "host_collision_policy",
                1, "TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED",
                "program_number_available=false;delete_all_authorized=false;no_cycle_start=true");
        cleanup();
        return 6;
    }
"""
    )
    pre_upload_delete_all_block = (
        """
    LogHost(inputCsv, outputCsv, idx, "S003A_DELETE_ALL_BEFORE_UPLOAD_GATE", "host_delete_all_gate",
            0, "DELETE_ALL_AUTHORIZED",
            "delete_all_authorized=true;delete_all_before_upload=true;reason=avoid_old_program_collision_and_protection_confusion");
    short preDeleteRet = CallFocas(inputCsv, outputCsv, idx, sniffer, "S003A_DELETE_ALL_BEFORE_UPLOAD", "cnc_delall",
        "delete_all_authorized=true;delete_all_before_upload=true",
        [&]() { return cnc_delall_fn(handle); },
        "delete_all_authorized=true;delete_all_before_upload=true");
    if (preDeleteRet != 0) {
        ODBST protectStatus{};
        CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
            "S003A_DELETE_ALL_BEFORE_UPLOAD_STATUS", "cnc_statinfo",
            "delete_all_failed=true;delete_return=" + std::to_string(preDeleteRet),
            [&]() { return cnc_statinfo_fn(handle, &protectStatus); },
            [&]() { return "delete_all_failed=true;delete_return=" + std::to_string(preDeleteRet) +
                ";delete_return_text=" + RetText(preDeleteRet) + ";" + StatinfoData(protectStatus) +
                ";upload_protection_or_mode_gate=true"; }, "Before");
        cleanup();
        return 6;
    }
"""
        if allow_delete_all
        else ""
    )
    template = r'''
// Fixed NCGuide FOCAS runner generated by CodeGenerationAgent.
// LLM-fillable content is restricted to NC payload/configuration constants below.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include <Fwlib32.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using std::string;

using u_char = unsigned char;
using bpf_u_int32 = unsigned int;
struct pcap;
struct pcap_dumper;
struct pcap_pkthdr { timeval ts; bpf_u_int32 caplen; bpf_u_int32 len; };
using pcap_t = pcap;
using pcap_dumper_t = pcap_dumper;
using pcap_handler = void(__cdecl *)(u_char*, const pcap_pkthdr*, const u_char*);
using pcap_open_live_t = pcap_t*(__cdecl *)(const char*, int, int, int, char*);
using pcap_dump_open_t = pcap_dumper_t*(__cdecl *)(pcap_t*, const char*);
using pcap_dispatch_t = int(__cdecl *)(pcap_t*, int, pcap_handler, u_char*);
using pcap_dump_t = void(__cdecl *)(u_char*, const pcap_pkthdr*, const u_char*);
using pcap_dump_flush_t = int(__cdecl *)(pcap_dumper_t*);
using pcap_dump_close_t = void(__cdecl *)(pcap_dumper_t*);
using pcap_close_t = void(__cdecl *)(pcap_t*);

static const int kPreferredProgramNumber = __PREFERRED_PROGRAM__;
static const int kExpectedNcSegmentCount = __EXPECTED_SEGMENTS__;
static const char* kNcBodyWithoutO = "__NC_BODY__";
static const std::vector<string> kPlannedBeforeApis = __PLANNED_BEFORE_APIS__;
static const std::vector<string> kPlannedDuringApis = __PLANNED_DURING_APIS__;
static const std::vector<string> kPlannedAfterApis = __PLANNED_AFTER_APIS__;

string Timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    tm local{};
    localtime_s(&local, &t);
    std::ostringstream os;
    os << std::put_time(&local, "%Y-%m-%dT%H:%M:%S") << "."
       << std::setw(9) << std::setfill('0') << (ns % 1000000000LL);
    return os.str();
}

long long TimestampNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

string RunNonce() {
    static const string nonce = std::to_string(GetCurrentProcessId()) + "-" + std::to_string(TimestampNs());
    return nonce;
}

string InvocationId(int index) { return "inv-" + RunNonce() + "-" + std::to_string(index); }
string TransactionId(int index) { return "tx-" + RunNonce() + "-" + std::to_string(index); }

string NormalizePhase(const string& value) {
    string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered == "before") return "Before";
    if (lowered == "after") return "After";
    return "During";
}

string gCurrentPhase = "Before";
int gSysinfoAxes = 0;
std::vector<long> gAccdecReferenceData;
bool gAccdecVariationObserved = false;

int ParseSysinfoAxes(const char raw[2]) {
    const unsigned char first = static_cast<unsigned char>(raw[0]);
    const unsigned char second = static_cast<unsigned char>(raw[1]);
    if (first >= '0' && first <= '9' && second >= '0' && second <= '9') {
        return static_cast<int>(first - '0') * 10 + static_cast<int>(second - '0');
    }
    if (first >= '0' && first <= '9') return static_cast<int>(first - '0');
    if (second >= '0' && second <= '9') return static_cast<int>(second - '0');
    if (first > 0 && second == 0) return static_cast<int>(first);
    return 0;
}

string StatinfoData(const ODBST& status) {
    return "aut=" + std::to_string(status.aut) +
        ";run=" + std::to_string(status.run) +
        ";motion=" + std::to_string(status.motion) +
        ";emergency=" + std::to_string(status.emergency) +
        ";alarm=" + std::to_string(status.alarm) +
        ";edit=" + std::to_string(status.edit);
}

string EscapeCsv(const string& value) {
    if (value.find(',') == string::npos && value.find('"') == string::npos && value.find('\n') == string::npos) {
        return value;
    }
    string out = "\"";
    for (char c : value) out += (c == '"') ? "\"\"" : string(1, c);
    out += "\"";
    return out;
}

int ApiParameterCount(const string& api) {
    // Legacy CSV compatibility field: SDK/ABI argument count. FlibHndl is
    // session context, while ODBST* is an output buffer for cnc_statinfo.
    if (api == "cnc_allclibhndl3") return 4;
    if (api == "cnc_freelibhndl") return 1;
    if (api == "cnc_statinfo") return 2;
    if (api == "cnc_sysinfo") return 2;
    if (api == "cnc_alarm2") return 2;
    if (api == "cnc_rdalmmsg") return 4;
    if (api == "cnc_rdprogdir3") return 5;
    if (api == "cnc_dwnstart3") return 2;
    if (api == "cnc_download3") return 3;
    if (api == "cnc_dwnend3") return 1;
    if (api == "cnc_search") return 2;
    if (api == "cnc_rdprgnum") return 2;
    if (api == "cnc_actf") return 2;
    if (api == "cnc_rdposition") return 4;
    if (api == "cnc_distance") return 4;
    if (api == "cnc_absolute") return 4;
    if (api == "cnc_absolute2") return 4;
    if (api == "cnc_machine") return 4;
    if (api == "cnc_relative") return 4;
    if (api == "cnc_relative2") return 4;
    if (api == "cnc_skip") return 4;
    if (api == "cnc_srvdelay") return 4;
    if (api == "cnc_accdecdly") return 4;
    if (api == "cnc_rddynamic") return 4;
    if (api == "cnc_rdaxisdata") return 6;
    if (api == "cnc_rd3dtooltip") return 2;
    if (api == "cnc_rdmdiprgstat") return 2;
    if (api == "cnc_rdmdipntr") return 2;
    if (api == "cnc_rdtool") return 4;
    if (api == "cnc_rdtofs") return 5;
    if (api == "cnc_rdtofsr") return 6;
    if (api == "cnc_rdtofsinfo") return 2;
    if (api == "cnc_rdtofsinfo2") return 2;
    if (api == "cnc_getdtailerr") return 2;
__DELETE_ALL_PARAMETER_COUNT__
    if (api == "pcap_open_live") return 5;
    if (api == "LoadLibraryW") return 1;
    if (api == "ncguide_ui_cycle_start") return 2;
    return 0;
}

void WriteBom(std::ofstream& f) {
    f.put(char(0xEF)); f.put(char(0xBB)); f.put(char(0xBF));
}

string RetText(short code) {
    switch (code) {
    case 0: return "EW_OK";
    case -17: return "EW_PROTOCOL";
    case -16: return "EW_SOCKET";
    case -15: return "EW_NODLL";
    case -14: return "EW_INIERR";
    case -8: return "EW_HANDLE";
    case -6: return "EW_UNEXP";
    case -2: return "EW_RESET";
    case -1: return "EW_BUSY";
    case 1: return "EW_FUNC";
    case 2: return "EW_LENGTH";
    case 3: return "EW_NUMBER";
    case 4: return "EW_ATTRIB";
    case 5: return "EW_DATA";
    case 6: return "EW_NOOPT";
    case 7: return "EW_PROT";
    case 8: return "EW_OVRFLOW";
    case 9: return "EW_PARAM";
    case 10: return "EW_BUFFER";
    case 11: return "EW_PATH";
    case 12: return "EW_MODE";
    case 13: return "EW_REJECT";
    case 14: return "EW_DTSRVR";
    case 15: return "EW_ALARM";
    case 16: return "EW_STOP";
    case 17: return "EW_PASSWD";
    default: return "FOCAS_RETURN_" + std::to_string(code);
    }
}

void LogInput(std::ofstream& csv, int index, const string& step, const string& phase,
              const string& iface, const string& fn, const string& params, const string& ts,
              const string& invocationId, const string& transactionId, long long startNs,
              const string& deviceContext) {
    csv << index << "," << EscapeCsv(ts) << "," << EscapeCsv(step)
        << "," << EscapeCsv(phase) << "," << EscapeCsv(iface) << "," << EscapeCsv(fn)
        << "," << EscapeCsv(params) << "," << ApiParameterCount(fn)
        << "," << EscapeCsv(invocationId) << "," << EscapeCsv(transactionId)
        << "," << startNs << "," << EscapeCsv(deviceContext) << "\n";
    csv.flush();
}

void LogOutput(std::ofstream& csv, int index, const string& step, const string& api,
               short ret, const string& text, const string& data, const string& ts,
               const string& invocationId, const string& transactionId, long long endNs) {
    csv << index << "," << EscapeCsv(ts) << "," << EscapeCsv(step) << "," << EscapeCsv(api)
        << "," << ret << "," << EscapeCsv(text) << "," << EscapeCsv(data)
        << "," << ApiParameterCount(api) << "," << EscapeCsv(invocationId)
        << "," << EscapeCsv(transactionId) << "," << endNs << "\n";
    csv.flush();
}

string ConsoleShort(const string& value, size_t limit = 240) {
    if (value.size() <= limit) return value;
    return value.substr(0, limit) + "...";
}

void PrintApiStart(int index, const string& step, const string& api, const string& params) {
    (void)index;
    (void)step;
    (void)api;
    (void)params;
}

void PrintApiEnd(int index, const string& api, short ret, const string& text, const string& data) {
    (void)index;
    if (ret == 0) {
        std::cout << api << " -> " << ConsoleShort(data) << std::endl;
    } else {
        std::cout << api << " -> ret=" << ret << " " << text
                  << "; " << ConsoleShort(data) << std::endl;
    }
}

string EnvString(const char* name, const string& fallback = "") {
    char buffer[1024]{};
    DWORD n = GetEnvironmentVariableA(name, buffer, DWORD(sizeof(buffer)));
    if (n == 0 || n >= sizeof(buffer)) return fallback;
    return string(buffer, n);
}

std::wstring EnvWideRequired(const wchar_t* name) {
    wchar_t buffer[MAX_PATH]{};
    DWORD n = GetEnvironmentVariableW(name, buffer, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return L"";
    return std::wstring(buffer, n);
}

int EnvInt(const char* name, int fallback = 0) {
    string value = EnvString(name);
    if (value.empty()) return fallback;
    try { return std::stoi(value); } catch (...) { return fallback; }
}

class PacketSniffer {
public:
    PacketSniffer(const string& networkDevice, const string& pcapFile,
                  const string& apiDirectory, const string& batchId)
        : device_(networkDevice), file_(pcapFile), apiDirectory_(apiDirectory), batchId_(batchId) {}
    bool Start() {
        // Prefer Npcap explicitly. A legacy WinPcap DLL can also be present
        // in SysWOW64 and loads successfully, but it cannot open Npcap's
        // loopback adapter. LOAD_WITH_ALTERED_SEARCH_PATH also resolves
        // Npcap's sibling Packet.dll from the same directory.
        char windowsDir[MAX_PATH]{};
        GetWindowsDirectoryA(windowsDir, MAX_PATH);
        string npcapPath = string(windowsDir) + "\\SysWOW64\\Npcap\\wpcap.dll";
        dll_ = LoadLibraryExA(npcapPath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!dll_) dll_ = LoadLibraryA("wpcap.dll");
        if (!dll_) { error_ = "LoadLibraryA(wpcap.dll) failed"; return false; }
        openLive_ = reinterpret_cast<pcap_open_live_t>(GetProcAddress(dll_, "pcap_open_live"));
        dumpOpen_ = reinterpret_cast<pcap_dump_open_t>(GetProcAddress(dll_, "pcap_dump_open"));
        dispatch_ = reinterpret_cast<pcap_dispatch_t>(GetProcAddress(dll_, "pcap_dispatch"));
        dump_ = reinterpret_cast<pcap_dump_t>(GetProcAddress(dll_, "pcap_dump"));
        dumpFlush_ = reinterpret_cast<pcap_dump_flush_t>(GetProcAddress(dll_, "pcap_dump_flush"));
        dumpClose_ = reinterpret_cast<pcap_dump_close_t>(GetProcAddress(dll_, "pcap_dump_close"));
        close_ = reinterpret_cast<pcap_close_t>(GetProcAddress(dll_, "pcap_close"));
        if (!openLive_ || !dumpOpen_ || !dispatch_ || !dump_ || !dumpFlush_ || !dumpClose_ || !close_) {
            error_ = "GetProcAddress pcap functions failed";
            return false;
        }
        char errbuf[512]{};
        handle_ = openLive_(device_.c_str(), 65535, 1, 1, errbuf);
        if (!handle_) { error_ = string("pcap_open_live failed: ") + errbuf; return false; }
        dumper_ = dumpOpen_(handle_, file_.c_str());
        if (!dumper_) { error_ = "pcap_dump_open failed"; return false; }
        return true;
    }
    bool BeginApi(const string& api) {
        if (!handle_ || !dumpOpen_) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = apiDumpers_.find(api);
        if (found == apiDumpers_.end()) {
            string safe = api;
            for (char& c : safe) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
            }
            string safeBatch = batchId_;
            for (char& c : safeBatch) {
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) c = '_';
            }
            string path = apiDirectory_ + "\\" + safeBatch + "_" + safe + ".pcap";
            pcap_dumper_t* dumper = dumpOpen_(handle_, path.c_str());
            if (!dumper) { error_ = "pcap_dump_open failed for API " + api; failed_ = true; return false; }
            found = apiDumpers_.emplace(api, dumper).first;
        }
        activeApi_ = api;
        return true;
    }
    void EndApi(const string& api) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = apiDumpers_.find(api);
        if (found != apiDumpers_.end() && dumpFlush_) dumpFlush_(found->second);
        if (activeApi_ == api) activeApi_.clear();
    }
    void CapturePackets(int packetCount = 12) {
        if (!handle_ || !dumper_ || !dispatch_) return;
        dispatch_(handle_, packetCount, &PacketSniffer::DumpThunk, reinterpret_cast<u_char*>(this));
        if (dumpFlush_) dumpFlush_(dumper_);
    }
    void Stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : apiDumpers_) {
            if (dumpFlush_) dumpFlush_(item.second);
            if (dumpClose_) dumpClose_(item.second);
        }
        apiDumpers_.clear();
        activeApi_.clear();
        if (dumper_ && dumpClose_) { dumpFlush_(dumper_); dumpClose_(dumper_); dumper_ = nullptr; }
        if (handle_ && close_) { close_(handle_); handle_ = nullptr; }
        if (dll_) { FreeLibrary(dll_); dll_ = nullptr; }
    }
    string Error() const { return error_; }
private:
    static void DumpThunk(u_char* user, const pcap_pkthdr* header, const u_char* packet) {
        auto* self = reinterpret_cast<PacketSniffer*>(user);
        if (self->dump_ && self->dumper_) {
            self->dump_(reinterpret_cast<u_char*>(self->dumper_), header, packet);
        }
        std::lock_guard<std::mutex> lock(self->mutex_);
        auto found = self->apiDumpers_.find(self->activeApi_);
        if (found != self->apiDumpers_.end() && self->dump_) {
            self->dump_(reinterpret_cast<u_char*>(found->second), header, packet);
        }
    }
    string device_;
    string file_;
    string apiDirectory_;
    string batchId_;
    string error_;
    HMODULE dll_ = nullptr;
    pcap_t* handle_ = nullptr;
    pcap_dumper_t* dumper_ = nullptr;
    std::map<string, pcap_dumper_t*> apiDumpers_;
    string activeApi_;
    std::mutex mutex_;
    bool failed_ = false;
    pcap_open_live_t openLive_ = nullptr;
    pcap_dump_open_t dumpOpen_ = nullptr;
    pcap_dispatch_t dispatch_ = nullptr;
    pcap_dump_t dump_ = nullptr;
    pcap_dump_flush_t dumpFlush_ = nullptr;
    pcap_dump_close_t dumpClose_ = nullptr;
    pcap_close_t close_ = nullptr;
};

template <typename Callable, typename OutputBuilder>
short CallFocasLazy(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx, PacketSniffer& sniffer,
                    const string& step, const string& api, const string& params, Callable&& fn,
                    OutputBuilder&& buildOutput, const string& phase = "") {
    string effectivePhase = phase.empty() ? gCurrentPhase : NormalizePhase(phase);
    sniffer.CapturePackets();
    sniffer.BeginApi(api);
    long long startNs = TimestampNs();
    string ts = Timestamp();
    string invocationId = InvocationId(idx);
    string transactionId = TransactionId(idx);
    string deviceContext = "target=NCGuide;endpoint=" + EnvString("FOCAS_HOST", "127.0.0.1") +
        ":" + EnvString("FOCAS_PORT", "8193") + ";phase=" + effectivePhase;
    PrintApiStart(idx, step, api, params);
    LogInput(inputCsv, idx, step, effectivePhase, api, api, params, ts,
             invocationId, transactionId, startNs, deviceContext);
    short ret = fn();
    sniffer.CapturePackets();
    long long endNs = TimestampNs();
    sniffer.EndApi(api);
    string outputData = buildOutput();
    PrintApiEnd(idx, api, ret, RetText(ret), outputData);
    LogOutput(outputCsv, idx, step, api, ret, RetText(ret), outputData, Timestamp(),
              invocationId, transactionId, endNs);
    ++idx;
    return ret;
}

template <typename Callable>
short CallFocas(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx, PacketSniffer& sniffer,
                const string& step, const string& api, const string& params, Callable&& fn,
                const string& dataPrefix, const string& phase = "") {
    return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, step, api, params, std::forward<Callable>(fn),
                         [&]() { return dataPrefix; }, phase);
}

void LogHost(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx,
             const string& step, const string& api, short ret, const string& text, const string& data,
             const string& phase = "") {
    string effectivePhase = phase.empty() ? gCurrentPhase : NormalizePhase(phase);
    long long startNs = TimestampNs();
    string ts = Timestamp();
    string invocationId = InvocationId(idx);
    PrintApiStart(idx, step, api, data);
    LogInput(inputCsv, idx, step, effectivePhase, api, api, data, ts, invocationId, "", startNs,
             "target=NCGuide;host_event=true;phase=" + effectivePhase);
    PrintApiEnd(idx, api, ret, text, data);
    LogOutput(outputCsv, idx, step, api, ret, text, data, Timestamp(), invocationId, "", TimestampNs());
    ++idx;
}

bool ClickScreenPoint(int x, int y) {
    if (!SetCursorPos(x, y)) return false;
    Sleep(80);
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    Sleep(40);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
    Sleep(120);
    return true;
}

HWND FindTopLevelWindowContaining(const string& requestedTitle) {
    if (requestedTitle.empty()) return nullptr;
    struct SearchContext {
        string requested;
        HWND found = nullptr;
    } context{requestedTitle, nullptr};
    std::transform(context.requested.begin(), context.requested.end(), context.requested.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    EnumWindows([](HWND hwnd, LPARAM rawContext) -> BOOL {
        if (!IsWindowVisible(hwnd)) return TRUE;
        auto* context = reinterpret_cast<SearchContext*>(rawContext);
        char candidateBuffer[256]{};
        GetWindowTextA(hwnd, candidateBuffer, 255);
        string candidate = candidateBuffer;
        std::transform(candidate.begin(), candidate.end(), candidate.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (!candidate.empty() && candidate.find(context->requested) != string::npos) {
            context->found = hwnd;
            return FALSE;
        }
        return TRUE;
    }, reinterpret_cast<LPARAM>(&context));
    return context.found;
}

bool TriggerNcGuideCycleStart(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx, int clickCount) {
    if (EnvString("NCGUIDE_ENABLE_UI_START", "1") != "1") {
        LogHost(inputCsv, outputCsv, idx, "S_CYCLE_START_DISABLED", "ncguide_ui_cycle_start",
                13, "UI_START_DISABLED", "ui_click_dispatched=false;no_click=true");
        return false;
    }
    int x = EnvInt("NCGUIDE_CYCLE_START_X", 0);
    int y = EnvInt("NCGUIDE_CYCLE_START_Y", 0);
    string mode = EnvString("NCGUIDE_CLICK_MODE", "screen");
    string title = EnvString("NCGUIDE_WINDOW_TITLE", "Main Panel");
    HWND configuredWindow = FindTopLevelWindowContaining(title);
    if (configuredWindow) {
        if (IsIconic(configuredWindow)) ShowWindow(configuredWindow, SW_RESTORE);
        SetForegroundWindow(configuredWindow);
        BringWindowToTop(configuredWindow);
        Sleep(250);
    }
    POINT pt{};
    pt.x = x;
    pt.y = y;
    HWND pointWindow = WindowFromPoint(pt);
    HWND rootWindow = pointWindow ? GetAncestor(pointWindow, GA_ROOT) : nullptr;
    HWND targetWindow = rootWindow ? rootWindow : pointWindow;
    char pointTitle[256]{};
    if (targetWindow) GetWindowTextA(targetWindow, pointTitle, 255);
    auto lower = [](string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return value;
    };
    string actualTitle = pointTitle;
    bool pointWindowLooksRight = targetWindow &&
        (title.empty() || lower(actualTitle).find(lower(title)) != string::npos);
    bool clicked = configuredWindow && pointWindowLooksRight && ClickScreenPoint(x, y);
    LogHost(inputCsv, outputCsv, idx, "S_CYCLE_START_CLICK", "ncguide_ui_cycle_start", clicked ? 0 : 13,
            clicked ? "EW_OK" : "CYCLE_START_TARGET_MISMATCH",
            "window_title=" + title + ";click_mode=" + mode + ";cycle_start_x=" + std::to_string(x) +
            ";cycle_start_y=" + std::to_string(y) + ";cycle_start_click_count=" + std::to_string(clickCount) +
            ";expected_nc_segment_count=" + std::to_string(kExpectedNcSegmentCount) +
            ";configured_window_found=" + string(configuredWindow ? "true" : "false") +
            ";point_window_title=" + string(pointTitle) +
            ";point_window_matches_target=" + string(pointWindowLooksRight ? "true" : "false") +
            ";ui_click_dispatched=" + string(clicked ? "true" : "false"));
    return clicked;
}

string BuildUploadPayload(int programNumber) {
    std::ostringstream payload;
    payload << "\nO" << programNumber << "\n" << kNcBodyWithoutO << "\n%";
    return payload.str();
}

int CountNcEffectiveSegments() {
    return kExpectedNcSegmentCount; // effective_nc_segment_count, CountNcEffectiveSegments, includes O-line and M30
}

int main() {
    CreateDirectoryA("data", nullptr);
    CreateDirectoryA("data\\by_api", nullptr);
    {
        std::ofstream marker("data\\by_api\\.runtime_per_api_capture", std::ios::binary);
        marker << "mode=runtime_per_api_dumper;batch_id="
               << EnvString("SMPAGENT_BATCH_ID", "batch_unknown") << "\n";
    }
    std::ofstream inputCsv("data\\focas_api_input.csv", std::ios::binary);
    std::ofstream outputCsv("data\\focas_api_output.csv", std::ios::binary);
    WriteBom(inputCsv);
    WriteBom(outputCsv);
    inputCsv << "index,timestamp,step_id,phase,interface_name,protocol_function,parameters,api_parameter_count,invocation_id,transaction_id,start_timestamp_ns,device_context\n";
    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count,invocation_id,transaction_id,end_timestamp_ns\n";
    int idx = 1;

    string networkDevice = EnvString("PCAP_NETWORK_DEVICE", "\\Device\\NPF_Loopback");
    string pcapFile = "data\\focas_capture.pcap";
    PacketSniffer sniffer(networkDevice, pcapFile, "data\\by_api",
                          EnvString("SMPAGENT_BATCH_ID", "batch_unknown"));
    {
        string ts = Timestamp();
        string invocationId = InvocationId(idx);
        long long startNs = TimestampNs();
        LogInput(inputCsv, idx, "PCAP_START", "Before", "PacketCapture", "pcap_open_live",
                 "device=" + networkDevice, ts, invocationId, "", startNs,
                 "target=NCGuide;capture_setup=true;phase=Before");
        if (!sniffer.Start()) {
            LogOutput(outputCsv, idx, "PCAP_START", "pcap_open_live", -1, "PCAP_CAPTURE_FAILED",
                      "pcap_capture_enabled=false;error=" + sniffer.Error(), Timestamp(),
                      invocationId, "", TimestampNs());
            return 2;
        }
        LogOutput(outputCsv, idx, "PCAP_START", "pcap_open_live", 0, "PCAP_CAPTURE_STARTED",
                  "pcap_capture_enabled=true;pcap_capture_started=true;file=" + pcapFile, Timestamp(),
                  invocationId, "", TimestampNs());
        ++idx;
    }

    HMODULE focasDll = nullptr;
    unsigned short handle = 0;
    auto cleanup = [&]() {
        sniffer.Stop();
        if (focasDll) FreeLibrary(focasDll);
    };

    std::wstring dllDir = EnvWideRequired(L"FOCAS_DLL_DIR");
    if (dllDir.empty()) {
        LogHost(inputCsv, outputCsv, idx, "LOAD_DLL", "LoadLibraryW", -1, "FOCAS_DLL_DIR_MISSING",
                "FOCAS_DLL_DIR missing;no FOCAS APIs called");
        cleanup();
        return 3;
    }
    SetDllDirectoryW(dllDir.c_str());
    std::wstring dllPath = dllDir + L"\\Fwlib32.dll";
    focasDll = LoadLibraryW(dllPath.c_str());
    if (!focasDll) {
        LogHost(inputCsv, outputCsv, idx, "LOAD_DLL", "LoadLibraryW", -1, "LoadLibraryW_FAILED",
                "dll=Fwlib32.dll;required_symbols_resolved=false");
        cleanup();
        return 3;
    }

    auto cnc_allclibhndl3_fn = reinterpret_cast<decltype(&::cnc_allclibhndl3)>(GetProcAddress(focasDll, "cnc_allclibhndl3"));
    auto cnc_freelibhndl_fn = reinterpret_cast<decltype(&::cnc_freelibhndl)>(GetProcAddress(focasDll, "cnc_freelibhndl"));
    auto cnc_statinfo_fn = reinterpret_cast<decltype(&::cnc_statinfo)>(GetProcAddress(focasDll, "cnc_statinfo"));
    auto cnc_sysinfo_fn = reinterpret_cast<decltype(&::cnc_sysinfo)>(GetProcAddress(focasDll, "cnc_sysinfo"));
    auto cnc_alarm2_fn = reinterpret_cast<decltype(&::cnc_alarm2)>(GetProcAddress(focasDll, "cnc_alarm2"));
    auto cnc_rdprogdir3_fn = reinterpret_cast<decltype(&::cnc_rdprogdir3)>(GetProcAddress(focasDll, "cnc_rdprogdir3"));
    auto cnc_dwnstart3_fn = reinterpret_cast<decltype(&::cnc_dwnstart3)>(GetProcAddress(focasDll, "cnc_dwnstart3"));
    auto cnc_download3_fn = reinterpret_cast<decltype(&::cnc_download3)>(GetProcAddress(focasDll, "cnc_download3"));
    auto cnc_dwnend3_fn = reinterpret_cast<decltype(&::cnc_dwnend3)>(GetProcAddress(focasDll, "cnc_dwnend3"));
    auto cnc_search_fn = reinterpret_cast<decltype(&::cnc_search)>(GetProcAddress(focasDll, "cnc_search"));
    auto cnc_rdprgnum_fn = reinterpret_cast<decltype(&::cnc_rdprgnum)>(GetProcAddress(focasDll, "cnc_rdprgnum"));
    auto cnc_actf_fn = reinterpret_cast<decltype(&::cnc_actf)>(GetProcAddress(focasDll, "cnc_actf"));
    auto cnc_rdposition_fn = reinterpret_cast<decltype(&::cnc_rdposition)>(GetProcAddress(focasDll, "cnc_rdposition"));
    auto cnc_rdsvmeter_fn = reinterpret_cast<decltype(&::cnc_rdsvmeter)>(GetProcAddress(focasDll, "cnc_rdsvmeter"));
    auto cnc_rdparam_fn = reinterpret_cast<decltype(&::cnc_rdparam)>(GetProcAddress(focasDll, "cnc_rdparam"));
    auto cnc_distance_fn = reinterpret_cast<decltype(&::cnc_distance)>(GetProcAddress(focasDll, "cnc_distance"));
    auto cnc_getdtailerr_fn = reinterpret_cast<decltype(&::cnc_getdtailerr)>(GetProcAddress(focasDll, "cnc_getdtailerr"));
    auto cnc_rdalmmsg_fn = reinterpret_cast<decltype(&::cnc_rdalmmsg)>(GetProcAddress(focasDll, "cnc_rdalmmsg"));
    auto cnc_absolute_fn = reinterpret_cast<decltype(&::cnc_absolute)>(GetProcAddress(focasDll, "cnc_absolute"));
    auto cnc_absolute2_fn = reinterpret_cast<decltype(&::cnc_absolute2)>(GetProcAddress(focasDll, "cnc_absolute2"));
    auto cnc_machine_fn = reinterpret_cast<decltype(&::cnc_machine)>(GetProcAddress(focasDll, "cnc_machine"));
    auto cnc_relative_fn = reinterpret_cast<decltype(&::cnc_relative)>(GetProcAddress(focasDll, "cnc_relative"));
    auto cnc_relative2_fn = reinterpret_cast<decltype(&::cnc_relative2)>(GetProcAddress(focasDll, "cnc_relative2"));
    auto cnc_skip_fn = reinterpret_cast<decltype(&::cnc_skip)>(GetProcAddress(focasDll, "cnc_skip"));
    auto cnc_srvdelay_fn = reinterpret_cast<decltype(&::cnc_srvdelay)>(GetProcAddress(focasDll, "cnc_srvdelay"));
    auto cnc_rdaxisname_fn = reinterpret_cast<decltype(&::cnc_rdaxisname)>(GetProcAddress(focasDll, "cnc_rdaxisname"));
    auto cnc_accdecdly_fn = reinterpret_cast<decltype(&::cnc_accdecdly)>(GetProcAddress(focasDll, "cnc_accdecdly"));
    auto cnc_rddynamic_fn = reinterpret_cast<decltype(&::cnc_rddynamic)>(GetProcAddress(focasDll, "cnc_rddynamic"));
    auto cnc_rdaxisdata_fn = reinterpret_cast<decltype(&::cnc_rdaxisdata)>(GetProcAddress(focasDll, "cnc_rdaxisdata"));
    auto cnc_rd3dtooltip_fn = reinterpret_cast<decltype(&::cnc_rd3dtooltip)>(GetProcAddress(focasDll, "cnc_rd3dtooltip"));
    auto cnc_rdmdiprgstat_fn = reinterpret_cast<decltype(&::cnc_rdmdiprgstat)>(GetProcAddress(focasDll, "cnc_rdmdiprgstat"));
    auto cnc_rdmdipntr_fn = reinterpret_cast<decltype(&::cnc_rdmdipntr)>(GetProcAddress(focasDll, "cnc_rdmdipntr"));
    auto cnc_rdtool_fn = reinterpret_cast<decltype(&::cnc_rdtool)>(GetProcAddress(focasDll, "cnc_rdtool"));
    auto cnc_rdtofs_fn = reinterpret_cast<decltype(&::cnc_rdtofs)>(GetProcAddress(focasDll, "cnc_rdtofs"));
    auto cnc_rdtofsr_fn = reinterpret_cast<decltype(&::cnc_rdtofsr)>(GetProcAddress(focasDll, "cnc_rdtofsr"));
    auto cnc_rdtofsinfo_fn = reinterpret_cast<decltype(&::cnc_rdtofsinfo)>(GetProcAddress(focasDll, "cnc_rdtofsinfo"));
    auto cnc_rdtofsinfo2_fn = reinterpret_cast<decltype(&::cnc_rdtofsinfo2)>(GetProcAddress(focasDll, "cnc_rdtofsinfo2"));
__DELETE_ALL_RESOLVE__
    bool plannedNeedsSysinfo = false;
    for (const auto& plannedApi : kPlannedBeforeApis) plannedNeedsSysinfo = plannedNeedsSysinfo || plannedApi == "cnc_sysinfo";
    for (const auto& plannedApi : kPlannedDuringApis) plannedNeedsSysinfo = plannedNeedsSysinfo || plannedApi == "cnc_sysinfo";
    for (const auto& plannedApi : kPlannedAfterApis) plannedNeedsSysinfo = plannedNeedsSysinfo || plannedApi == "cnc_sysinfo";
    if (!cnc_allclibhndl3_fn || !cnc_freelibhndl_fn || !cnc_statinfo_fn || !cnc_alarm2_fn ||
        (plannedNeedsSysinfo && !cnc_sysinfo_fn) ||
        !cnc_rdprogdir3_fn || !cnc_dwnstart3_fn || !cnc_download3_fn || !cnc_dwnend3_fn ||
        !cnc_search_fn || !cnc_rdprgnum_fn || !cnc_actf_fn || !cnc_rdposition_fn || !cnc_distance_fn ||
        !cnc_getdtailerr_fn__DELETE_ALL_MISSING_CHECK__) {
        LogHost(inputCsv, outputCsv, idx, "LOAD_DLL", "LoadLibraryW", -1, "GetProcAddress_FAILED",
                "required_symbols_resolved=false");
        cleanup();
        return 3;
    }
    LogHost(inputCsv, outputCsv, idx, "LOAD_DLL", "LoadLibraryW", 0, "EW_OK", "required_symbols_resolved=true");

    int focasPort = EnvInt("FOCAS_PORT", 8193);
    string focasHost = EnvString("FOCAS_HOST", "127.0.0.1");
    short ret = CallFocas(inputCsv, outputCsv, idx, sniffer, "S001_CONNECT", "cnc_allclibhndl3",
        "host=" + focasHost + ";port=" + std::to_string(focasPort) + ";timeout_seconds=10;coverage_role=support",
        [&]() { return cnc_allclibhndl3_fn(focasHost.c_str(), static_cast<unsigned short>(focasPort), 10, &handle); },
        "host=" + focasHost + ";port=" + std::to_string(focasPort) + ";coverage_role=support");
    if (ret != 0) { cleanup(); return 4; }

    auto freeHandle = [&]() {
        if (handle && cnc_freelibhndl_fn) {
            CallFocas(inputCsv, outputCsv, idx, sniffer, "S015_RELEASE_HANDLE", "cnc_freelibhndl",
                "handle_valid=true", [&]() { return cnc_freelibhndl_fn(handle); },
                "handle_released=true", "After");
            handle = 0;
        }
    };

    auto logDetailError = [&](const string& step) {
        ODBERR err{};
        CallFocasLazy(inputCsv, outputCsv, idx, sniffer, step, "cnc_getdtailerr", "",
            [&]() { return cnc_getdtailerr_fn(handle, &err); },
            [&]() { return "err_no=" + std::to_string(err.err_no) +
                ";err_dtno=" + std::to_string(err.err_dtno); });
    };

    auto runPlannedApi = [&](const string& stepBase, const string& api, int segment, const string& phase) -> short {
        (void)segment;
        gCurrentPhase = NormalizePhase(phase);
        string noParams = "";
        if (api == "cnc_allclibhndl3" || api == "cnc_freelibhndl" ||
            api == "cnc_dwnstart3" || api == "cnc_download3" || api == "cnc_dwnend3" ||
            api == "cnc_search" || api == "cnc_rdprgnum" || api == "cnc_rdprogdir3") {
            LogHost(inputCsv, outputCsv, idx, stepBase + "_" + api, "host_structural_api_reference", 0,
                    "STRUCTURAL_API_HANDLED_BY_MAIN_CHAIN", "referenced_api=" + api);
            return 0;
        }
        if (api == "cnc_statinfo") {
            ODBST st{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_statinfo", "cnc_statinfo",
                noParams, [&]() { return cnc_statinfo_fn(handle, &st); },
                [&]() { return StatinfoData(st); });
        }
        if (api == "cnc_sysinfo") {
            if (!cnc_sysinfo_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_sysinfo", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_sysinfo");
                return 1;
            }
            ODBSYS sysinfo{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_sysinfo", "cnc_sysinfo",
                noParams, [&]() {
                    short result = cnc_sysinfo_fn(handle, &sysinfo);
                    if (result == 0) {
                        gSysinfoAxes = ParseSysinfoAxes(sysinfo.axes);
                        if (gSysinfoAxes < 1 || gSysinfoAxes > MAX_AXIS) gSysinfoAxes = 0;
                    }
                    return result;
                },
                [&]() { return "axes=" + std::to_string(gSysinfoAxes) +
                    ";axes_raw0=" + std::to_string(static_cast<unsigned char>(sysinfo.axes[0])) +
                    ";axes_raw1=" + std::to_string(static_cast<unsigned char>(sysinfo.axes[1])) +
                    ";axis_domain=1..N;axis_count_source=cnc_sysinfo.sysinfo.axes"; });
        }
        if (api == "cnc_alarm2") {
            long alarmBits = 0;
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_alarm2", "cnc_alarm2",
                noParams, [&]() { return cnc_alarm2_fn(handle, &alarmBits); },
                [&]() { return "alarm_bits=" + std::to_string(alarmBits); });
        }
        if (api == "cnc_rdalmmsg") {
            if (!cnc_rdalmmsg_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdalmmsg", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdalmmsg");
                return 1;
            }
            short count = 4;
            ODBALMMSG alarms[4]{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdalmmsg", "cnc_rdalmmsg",
                "alarm_type=-1;message_count=4",
                [&]() { return cnc_rdalmmsg_fn(handle, -1, &count, alarms); },
                [&]() { return "returned_alarm_messages=" + std::to_string(count); });
        }
        if (api == "cnc_actf") {
            ODBACT feed{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_actf", "cnc_actf",
                noParams, [&]() { return cnc_actf_fn(handle, &feed); },
                [&]() { return "feed_speed=" + std::to_string(feed.data) + ";actual_feed=" + std::to_string(feed.data); });
        }
        if (api == "cnc_rdposition") {
            ODBPOS pos[3]{};
            short posCount = 3;
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdposition", "cnc_rdposition",
                "type=-1;axis_count=3",
                [&]() { return cnc_rdposition_fn(handle, -1, &posCount, pos); },
                [&]() { return "position_axis1=" + std::to_string(pos[0].abs.data) +
                ";position_axis2=" + std::to_string(pos[1].abs.data) +
                ";position_axis3=" + std::to_string(pos[2].abs.data) +
                ";distance_to_go=" + std::to_string(pos[0].dist.data); });
        }
        if (api == "cnc_rdsvmeter") {
            if (!cnc_rdsvmeter_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdsvmeter", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdsvmeter");
                return 1;
            }
            short axisCount = 3;
            ODBSVLOAD loads[3]{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdsvmeter", "cnc_rdsvmeter",
                "axis_count=3",
                [&]() { return cnc_rdsvmeter_fn(handle, &axisCount, loads); },
                [&]() { return "returned_axis_count=" + std::to_string(axisCount) +
                ";servo_load_axis1=" + std::to_string(loads[0].svload.data) +
                ";servo_load_axis2=" + std::to_string(loads[1].svload.data) +
                ";servo_load_axis3=" + std::to_string(loads[2].svload.data); });
        }
        if (api == "cnc_rdparam") {
            if (!cnc_rdparam_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdparam", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdparam;parameter_no=1005");
                return 1;
            }
            constexpr short parameterNo = 1005;
            constexpr short axis = -1;
            short length = static_cast<short>(sizeof(IODBPSD));
            IODBPSD parameter{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdparam", "cnc_rdparam",
                "parameter_no=1005;axis=-1;length=" + std::to_string(length),
                [&]() { return cnc_rdparam_fn(handle, parameterNo, axis, length, &parameter); },
                [&]() { return "parameter_no=" + std::to_string(parameter.datano) +
                ";axis=" + std::to_string(parameter.type) +
                ";cdata=" + std::to_string(static_cast<int>(parameter.u.cdata)) +
                ";idata=" + std::to_string(parameter.u.idata) +
                ";ldata=" + std::to_string(parameter.u.ldata); });
        }
        if (api == "cnc_distance") {
            ODBAXIS axis{};
            // For FS0i-D, ODBAXIS length is the all-axis data block size.
            // MAX_AXIS is 32 in the official 0i-D header: 4 + 4*MAX_AXIS.
            constexpr short len = static_cast<short>(4 + 4 * MAX_AXIS);
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_distance", "cnc_distance",
                "axis=-1;length=" + std::to_string(len),
                [&]() { return cnc_distance_fn(handle, -1, len, &axis); },
                [&]() { return "distance_to_go=" + std::to_string(axis.data[0]) +
                ";dist_axis1=" + std::to_string(axis.data[0]) +
                ";dist_axis2=" + std::to_string(axis.data[1]) +
                ";dist_axis3=" + std::to_string(axis.data[2]); });
        }
        auto runAxisApi = [&](const string& axisApi, auto fn) -> short {
            if (!fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_" + axisApi, "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=" + axisApi);
                return 1;
            }
            short axisCount = MAX_AXIS;
            ODBAXISNAME axisNames[MAX_AXIS]{};
            short discoveryRet = 1;
            if (cnc_rdaxisname_fn) {
                discoveryRet = CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
                    stepBase + "_cnc_rdaxisname", "cnc_rdaxisname",
                    "data_num=" + std::to_string(axisCount),
                    [&]() { return cnc_rdaxisname_fn(handle, &axisCount, axisNames); },
                    [&]() { return "controlled_axis_count=" + std::to_string(axisCount); });
            } else {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdaxisname", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdaxisname");
            }
            auto callAxis = [&](short selectedAxis) -> short {
                ODBAXIS axis{};
                // FOCAS length is the logical ODBAXIS data block, not the
                // C struct size: one axis is 4+4, ALL_AXES is 4+4*MAX_AXIS.
                const short len = selectedAxis == static_cast<short>(-1)
                    ? static_cast<short>(4 + 4 * MAX_AXIS)
                    : static_cast<short>(8);
                return CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
                    stepBase + "_" + axisApi + "_axis_" + std::to_string(selectedAxis), axisApi,
                    "axis=" + std::to_string(selectedAxis) +
                        ";length=" + std::to_string(len),
                    [&]() { return fn(handle, selectedAxis, len, &axis); },
                    [&]() { return "axis_data1=" + std::to_string(axis.data[0]) +
                    ";axis_data2=" + std::to_string(axis.data[1]) +
                    ";axis_data3=" + std::to_string(axis.data[2]); });
            };
            short result = callAxis(-1);
            if (discoveryRet != 0 || axisCount <= 0 || axisCount > MAX_AXIS) return result;
            // These APIs document the selector as an axis number.  Use the
            // controller-reported count; do not reinterpret it as a position
            // type domain.
            for (short selectedAxis = 1; selectedAxis <= axisCount; ++selectedAxis) {
                const short axisRet = callAxis(selectedAxis);
                if (result == 0 && axisRet != 0) result = axisRet;
                if (axisRet == 0) {
                    if (gAccdecReferenceData.empty()) {
                        gAccdecReferenceData.assign(axis.data, axis.data + MAX_AXIS);
                    } else if (!std::equal(gAccdecReferenceData.begin(), gAccdecReferenceData.end(), axis.data)) {
                        gAccdecVariationObserved = true;
                    }
                }
            }
            return result;
        };
        if (api == "cnc_absolute") return runAxisApi("cnc_absolute", cnc_absolute_fn);
        if (api == "cnc_absolute2") return runAxisApi("cnc_absolute2", cnc_absolute2_fn);
        if (api == "cnc_machine") return runAxisApi("cnc_machine", cnc_machine_fn);
        if (api == "cnc_relative") return runAxisApi("cnc_relative", cnc_relative_fn);
        if (api == "cnc_relative2") return runAxisApi("cnc_relative2", cnc_relative2_fn);
        if (api == "cnc_skip") return runAxisApi("cnc_skip", cnc_skip_fn);
        if (api == "cnc_srvdelay") return runAxisApi("cnc_srvdelay", cnc_srvdelay_fn);
        if (api == "cnc_accdecdly") {
            if (!cnc_accdecdly_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_accdecdly", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_accdecdly");
                return 1;
            }
            if (gSysinfoAxes < 1 || gSysinfoAxes > MAX_AXIS) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_accdecdly", "host_runtime_binding", 1,
                        "AXIS_COUNT_UNAVAILABLE", "axis_count_source=cnc_sysinfo.sysinfo.axes;no_axis_call=true");
                return 1;
            }
            short result = 0;
            for (short selectedAxis = 1; selectedAxis <= gSysinfoAxes; ++selectedAxis) {
                ODBAXIS axis{};
                const short length = static_cast<short>(8);
                const short axisRet = CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
                    stepBase + "_cnc_accdecdly_axis_" + std::to_string(selectedAxis), "cnc_accdecdly",
                    "axis=" + std::to_string(selectedAxis) + ";length=" + std::to_string(length) + ";axis_count=" + std::to_string(gSysinfoAxes),
                    [&]() { return cnc_accdecdly_fn(handle, selectedAxis, length, &axis); },
                    [&]() {
                        string data = "axis=" + std::to_string(selectedAxis) + ";length=" + std::to_string(length) + ";data=[";
                        for (int dataIndex = 0; dataIndex < MAX_AXIS; ++dataIndex) {
                            if (dataIndex) data += ",";
                            data += std::to_string(axis.data[dataIndex]);
                        }
                        return data + "];axis_count_source=cnc_sysinfo.sysinfo.axes";
                    });
                if (result == 0 && axisRet != 0) result = axisRet;
            }
            return result;
        }
        if (api == "cnc_rddynamic") {
            if (!cnc_rddynamic_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rddynamic", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rddynamic");
                return 1;
            }
            ODBDY dy{};
            short len = sizeof(ODBDY);
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rddynamic", "cnc_rddynamic",
                "axis=-1;length=" + std::to_string(len),
                [&]() { return cnc_rddynamic_fn(handle, -1, len, &dy); },
                [&]() { return "dynamic_feed=" + std::to_string(dy.actf) +
                ";dynamic_program=" + std::to_string(dy.prgnum) +
                ";dynamic_distance=" + std::to_string(dy.pos.faxis.distance[0]); });
        }
        if (api == "cnc_rdaxisdata") {
            if (!cnc_rdaxisdata_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdaxisdata", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdaxisdata");
                return 1;
            }
            short types[1] = {0};
            short num = 1;
            short len = sizeof(ODBAXDT);
            ODBAXDT data[1]{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdaxisdata", "cnc_rdaxisdata",
                "class=1;type_count=1;length=" + std::to_string(len),
                [&]() { return cnc_rdaxisdata_fn(handle, 1, types, num, &len, data); },
                [&]() { return "axisdata_value=" + std::to_string(data[0].data) + ";axisdata_dec=" + std::to_string(data[0].dec); });
        }
        if (api == "cnc_rd3dtooltip") {
            if (!cnc_rd3dtooltip_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rd3dtooltip", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rd3dtooltip");
                return 1;
            }
            ODB3DHDL tip{};
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rd3dtooltip", "cnc_rd3dtooltip",
                noParams, [&]() { return cnc_rd3dtooltip_fn(handle, &tip); }, "tooltip_read=true");
        }
        if (api == "cnc_rdmdiprgstat") {
            if (!cnc_rdmdiprgstat_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdmdiprgstat", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdmdiprgstat");
                return 1;
            }
            unsigned short mdiStatus = 0;
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdmdiprgstat", "cnc_rdmdiprgstat",
                noParams, [&]() { return cnc_rdmdiprgstat_fn(handle, &mdiStatus); },
                "mdi_program_status=" + std::to_string(mdiStatus));
        }
        if (api == "cnc_rdmdipntr") {
            if (!cnc_rdmdipntr_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdmdipntr", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdmdipntr");
                return 1;
            }
            ODBMDIP mdi{};
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdmdipntr", "cnc_rdmdipntr",
                noParams, [&]() { return cnc_rdmdipntr_fn(handle, &mdi); }, "mdi_pointer_read=true");
        }
        if (api == "cnc_rdtool") {
            if (!cnc_rdtool_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdtool", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdtool");
                return 1;
            }
            short startTool = static_cast<short>(1 + (segment % 3));
            short count = 1;
            IODBTLMNG toolData[1]{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdtool", "cnc_rdtool",
                "start_tool=" + std::to_string(startTool) + ";count=1",
                [&]() { return cnc_rdtool_fn(handle, startTool, &count, toolData); },
                [&]() { return "returned_count=" + std::to_string(count) +
                    ";T_code=" + std::to_string(toolData[0].T_code); });
        }
        if (api == "cnc_rdtofs") {
            if (!cnc_rdtofs_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdtofs", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdtofs");
                return 1;
            }
            short offsetNumber = static_cast<short>(1 + (segment % 3));
            ODBTOFS offset{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdtofs", "cnc_rdtofs",
                "offset_number=" + std::to_string(offsetNumber) + ";offset_type=0;length=" + std::to_string(sizeof(offset)),
                [&]() { return cnc_rdtofs_fn(handle, offsetNumber, 0, static_cast<short>(sizeof(offset)), &offset); },
                [&]() { return "datano=" + std::to_string(offset.datano) +
                    ";type=" + std::to_string(offset.type) + ";data=" + std::to_string(offset.data); });
        }
        if (api == "cnc_rdtofsr") {
            if (!cnc_rdtofsr_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdtofsr", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdtofsr");
                return 1;
            }
            short startOffset = static_cast<short>(1 + (segment % 3));
            short endOffset = static_cast<short>(startOffset + 1);
            IODBTO offsets{};
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdtofsr", "cnc_rdtofsr",
                "start_offset=" + std::to_string(startOffset) + ";offset_type=0;end_offset=" +
                    std::to_string(endOffset) + ";length=" + std::to_string(sizeof(offsets)),
                [&]() { return cnc_rdtofsr_fn(handle, startOffset, 0, endOffset,
                    static_cast<short>(sizeof(offsets)), &offsets); }, "offset_range_read=true");
        }
        if (api == "cnc_rdtofsinfo") {
            if (!cnc_rdtofsinfo_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdtofsinfo", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdtofsinfo");
                return 1;
            }
            ODBTLINF info{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdtofsinfo", "cnc_rdtofsinfo",
                noParams, [&]() { return cnc_rdtofsinfo_fn(handle, &info); },
                [&]() { return "ofs_type=" + std::to_string(info.ofs_type) +
                    ";use_no=" + std::to_string(info.use_no); });
        }
        if (api == "cnc_rdtofsinfo2") {
            if (!cnc_rdtofsinfo2_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdtofsinfo2", "host_unsupported_api", 1,
                        "UNSUPPORTED_SYMBOL", "requested_api=cnc_rdtofsinfo2");
                return 1;
            }
            ODBTLINF2 info{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdtofsinfo2", "cnc_rdtofsinfo2",
                noParams, [&]() { return cnc_rdtofsinfo2_fn(handle, &info); },
                [&]() { return "ofs_type=" + std::to_string(info.ofs_type) +
                    ";use_no=" + std::to_string(info.use_no) +
                    ";ofs_enable=" + std::to_string(info.ofs_enable); });
        }
        LogHost(inputCsv, outputCsv, idx, stepBase + "_" + api, "host_unsupported_api", 1,
                "SKIPPED_UNSUPPORTED_BY_FIXED_ADAPTER", "requested_api=" + api);
        return 1;
    };

    auto runPlannedApiList = [&](const std::vector<string>& apis, const string& stepBase, int segment, const string& phase) {
        if (apis.empty()) {
            LogHost(inputCsv, outputCsv, idx, stepBase + "_none", "host_planned_api_list", 0, "NO_PLANNED_API",
                    "");
            return;
        }
        for (const auto& api : apis) {
            runPlannedApi(stepBase, api, segment, phase);
        }
    };

    auto runEarlyAbsoluteAndFeedReads = [&](const string& stepBase, int segment) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            string repeatStep = stepBase + "_early_abs_actf_repeat" + std::to_string(repeat + 1);
            runPlannedApi(repeatStep, "cnc_absolute", segment, "during");
            runPlannedApi(repeatStep, "cnc_actf", segment, "during");
            Sleep(50);
        }
    };

    ODBST initialStatus{};
    ret = CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S002_INITIAL_STATUS", "cnc_statinfo",
        "coverage_role=support",
        [&]() { return cnc_statinfo_fn(handle, &initialStatus); },
        [&]() { return StatinfoData(initialStatus) + ";coverage_role=support"; });

    long alarmBits = 0;
    CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S003_INITIAL_ALARM", "cnc_alarm2",
        "coverage_role=support",
        [&]() { return cnc_alarm2_fn(handle, &alarmBits); },
        [&]() { return "alarm_bits=" + std::to_string(alarmBits) + ";coverage_role=support"; });
    runPlannedApiList(kPlannedBeforeApis, "S003_PLANNED_BEFORE_API", 0, "before");
__PRE_UPLOAD_DELETE_ALL_BLOCK__

    int preferredProgram = kPreferredProgramNumber;
    int selectedProgram = preferredProgram;
    bool programAvailable = false;
    for (int candidate = preferredProgram; candidate < preferredProgram + 20; ++candidate) {
        long top = candidate;
        short count = 20;
        PRGDIR3 entries[20]{};
        bool exists = false;
        string dataPrefix = "preferred_program_number=" + std::to_string(preferredProgram) +
            ";candidate_program_number=" + std::to_string(candidate) +
            ";exact_match=true;target_program_exists=";
        ret = CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S004_PROGRAM_DIRECTORY_LOOKUP", "cnc_rdprogdir3",
            "type=0;top_program=" + std::to_string(candidate) + ";page_size=20;target_program=O" + std::to_string(candidate),
            [&]() { return cnc_rdprogdir3_fn(handle, 0, &top, &count, entries); },
            [&]() { return dataPrefix + "unknown;returned_num=" + std::to_string(count); });
        if (ret != 0) {
            logDetailError("S004_PROGRAM_DIRECTORY_DETAIL_ERROR");
            cleanup();
            return 5;
        }
        for (int i = 0; i < count && i < 20; ++i) {
            if (entries[i].number == candidate) exists = true;
        }
        LogHost(inputCsv, outputCsv, idx, "S004_PROGRAM_DIRECTORY_RESULT", "program_directory_lookup", 0, "EW_OK",
                "preferred_program_number=" + std::to_string(preferredProgram) +
                ";selected_program_number=" + std::to_string(candidate) +
                ";target_program_exists=" + string(exists ? "true" : "false") +
                ";program_number_available=" + string(exists ? "false" : "true") +
                ";collision_strategy=choose_unused;exact_match=true");
        if (!exists) {
            selectedProgram = candidate;
            programAvailable = true;
            break;
        }
    }
__DELETE_ALL_BLOCK__
    LogHost(inputCsv, outputCsv, idx, "S005_COLLISION_POLICY_GATE", "host_collision_policy", 0, "EW_OK",
            "preferred_program_number=" + std::to_string(preferredProgram) +
            ";selected_program_number=" + std::to_string(selectedProgram) +
            ";target_program_exists=" + string(selectedProgram == preferredProgram ? "false" : "true") +
            ";program_number_available=true;collision_strategy=choose_unused");

    string payload = BuildUploadPayload(selectedProgram);
    ret = 0;
    for (int attempt = 1; attempt <= 3; ++attempt) {
        ret = CallFocas(inputCsv, outputCsv, idx, sniffer, "S006_UPLOAD_START_attempt" + std::to_string(attempt), "cnc_dwnstart3",
            "download_type=0;download_type_name=NC_PROGRAM;selected_program_number=" + std::to_string(selectedProgram),
            [&]() { return cnc_dwnstart3_fn(handle, 0); },
            "selected_program_number=" + std::to_string(selectedProgram) + ";attempt=" + std::to_string(attempt));
        if (ret == 0) break;
        logDetailError("S006_UPLOAD_START_DETAIL_ERROR");
        if (ret == -16 && attempt < 3) {
            freeHandle();
            Sleep(500);
            CallFocas(inputCsv, outputCsv, idx, sniffer, "S006_RECONNECT_AFTER_SOCKET", "cnc_allclibhndl3",
                "host=" + focasHost + ";port=" + std::to_string(focasPort) + ";timeout_seconds=10;reason=EW_SOCKET_retry",
                [&]() { return cnc_allclibhndl3_fn(focasHost.c_str(), static_cast<unsigned short>(focasPort), 10, &handle); },
                "host=" + focasHost + ";port=" + std::to_string(focasPort) + ";reason=EW_SOCKET_retry");
        } else {
            break;
        }
    }
    if (ret != 0) {
        LogHost(inputCsv, outputCsv, idx, "PROGRAM_NOT_VERIFIED", "host_lifecycle_gate", ret, "PROGRAM_NOT_VERIFIED",
                "upload_failed=true;no_cycle_start=true;selected_program_number=" + std::to_string(selectedProgram));
        freeHandle(); cleanup(); return 6;
    }

    long len = static_cast<long>(payload.size());
    std::vector<char> buffer(payload.begin(), payload.end());
    buffer.push_back('\0');
    ret = CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S007_UPLOAD_PAYLOAD", "cnc_download3",
        "payload_format=LF_O_blocks_LF_percent;length=" + std::to_string(len),
        [&]() { return cnc_download3_fn(handle, &len, buffer.data()); },
        [&]() { return "requested_bytes=" + std::to_string(payload.size()) + ";accepted_length=" + std::to_string(len) +
        ";selected_program_number=" + std::to_string(selectedProgram); });
    if (ret != 0) { logDetailError("S007_UPLOAD_PAYLOAD_DETAIL_ERROR"); freeHandle(); cleanup(); return 7; }

    ret = CallFocas(inputCsv, outputCsv, idx, sniffer, "S008_UPLOAD_END", "cnc_dwnend3",
        "selected_program_number=" + std::to_string(selectedProgram),
        [&]() { return cnc_dwnend3_fn(handle); },
        "selected_program_number=" + std::to_string(selectedProgram));
    if (ret != 0) {
        logDetailError("S008_UPLOAD_END_DETAIL_ERROR");
        ODBST uploadEndStatus{};
        CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
            "S008_UPLOAD_END_STATUS_GATE", "cnc_statinfo",
            "upload_end_failed=true;upload_end_return=" + std::to_string(ret),
            [&]() { return cnc_statinfo_fn(handle, &uploadEndStatus); },
            [&]() { return "upload_end_failed=true;upload_end_return=" + std::to_string(ret) +
                ";upload_end_return_text=" + RetText(ret) +
                ";selected_program_number=" + std::to_string(selectedProgram) + ";" +
                StatinfoData(uploadEndStatus) +
                ";upload_protection_or_mode_gate=" + string(ret == 7 ? "true" : "false"); }, "Before");
        freeHandle(); cleanup(); return 8;
    }

    // Flow-style lifecycle retry: NCGuide can briefly report EW_BUSY after
    // upload finalization.  Retry only that transient return, with every
    // attempt logged; all other selection errors remain fatal.
    ret = -1;
    for (int attempt = 1; attempt <= 3; ++attempt) {
        ret = CallFocas(inputCsv, outputCsv, idx, sniffer,
            "S009_SELECT_PROGRAM_attempt" + std::to_string(attempt), "cnc_search",
            "program_number=" + std::to_string(selectedProgram) +
                ";attempt=" + std::to_string(attempt),
            [&]() { return cnc_search_fn(handle, static_cast<short>(selectedProgram)); },
            "selected_program_number=" + std::to_string(selectedProgram) +
                ";attempt=" + std::to_string(attempt));
        if (ret == 0 || ret != -1 || attempt == 3) break;
        Sleep(250);
    }
    if (ret != 0) { logDetailError("S009_SELECT_DETAIL_ERROR"); freeHandle(); cleanup(); return 9; }

    ODBPRO prg{};
    ret = CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S010_VERIFY_PROGRAM", "cnc_rdprgnum",
        "expected_program_number=" + std::to_string(selectedProgram),
        [&]() { return cnc_rdprgnum_fn(handle, &prg); },
        [&]() { return "running_program=" + std::to_string(prg.data) + ";main_program=" + std::to_string(prg.mdata) +
        ";expected_program_number=" + std::to_string(selectedProgram); });
    bool programVerified = (ret == 0 && (prg.data == selectedProgram || prg.mdata == selectedProgram));
    LogHost(inputCsv, outputCsv, idx, "S010_PROGRAM_VERIFICATION_GATE", "host_lifecycle_gate",
            programVerified ? 0 : 1, programVerified ? "PROGRAM_VERIFIED" : "PROGRAM_NOT_VERIFIED",
            "program_verified=" + string(programVerified ? "true" : "false") +
            ";expected_program_number=" + std::to_string(selectedProgram) +
            ";uploaded_program_number=" + std::to_string(selectedProgram));
    if (!programVerified) { freeHandle(); cleanup(); return 10; }

    int expected_nc_segment_count = CountNcEffectiveSegments();
    int cycle_start_click_count = 0;
    LogHost(inputCsv, outputCsv, idx, "S011_SINGLE_BLOCK_PLAN", "host_payload_parser", 0, "EW_OK",
            "effective_nc_segment_count=" + std::to_string(expected_nc_segment_count) +
            ";expected_nc_segment_count=" + std::to_string(expected_nc_segment_count) +
            ";cycle_start_click_count=0;M30=true");

    auto readDynamicSeq = [&](long& seq, long& prg, long& feed) -> short {
        seq = -1;
        prg = -1;
        feed = -1;
        if (!cnc_rddynamic_fn) return 1;
        ODBDY dy{};
        short len = sizeof(ODBDY);
        short dRet = CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
            "S012_READ_DYNAMIC_SEQUENCE", "cnc_rddynamic",
            "axis=-1;length=" + std::to_string(len),
            [&]() { return cnc_rddynamic_fn(handle, -1, len, &dy); },
            [&]() { return "seqnum=" + std::to_string(dy.seqnum) +
                ";prgnum=" + std::to_string(dy.prgnum) +
                ";actual_feed=" + std::to_string(dy.actf); }, "During");
        if (dRet == 0) {
            seq = dy.seqnum;
            prg = dy.prgnum;
            feed = dy.actf;
        }
        return dRet;
    };

    auto verifyCycleStartEffect = [&](int segment, int clickCount, long beforeSeq) -> bool {
        bool effectObserved = false;
        bool activeObserved = false;
        long lastSeq = beforeSeq;
        long lastPrg = -1;
        long lastFeed = -1;
        short lastStatRet = 0;
        short lastDynRet = 0;
        ODBST observedStatus{};
        for (int poll = 0; poll < 20; ++poll) {
            Sleep(50);
            lastStatRet = CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
                "S012_CYCLE_START_EFFECT_STATUS", "cnc_statinfo",
                "segment_index=" + std::to_string(segment) + ";poll=" + std::to_string(poll),
                [&]() { return cnc_statinfo_fn(handle, &observedStatus); },
                [&]() { return StatinfoData(observedStatus); }, "During");
            long seq = -1;
            long prg = -1;
            long feed = -1;
            lastDynRet = readDynamicSeq(seq, prg, feed);
            if (lastStatRet == 0 && (observedStatus.run != 0 || observedStatus.motion != 0)) {
                activeObserved = true;
                effectObserved = true;
            }
            if (lastDynRet == 0) {
                lastSeq = seq;
                lastPrg = prg;
                lastFeed = feed;
                if (beforeSeq >= 0 && seq != beforeSeq) {
                    effectObserved = true;
                }
            }
            if (effectObserved) break;
        }
        LogHost(inputCsv, outputCsv, idx, "S012_CYCLE_START_EFFECT_VERIFY", "cycle_start_effect_gate",
                effectObserved ? 0 : 1,
                effectObserved ? "CYCLE_START_EFFECT_OBSERVED" : "CYCLE_START_EFFECT_NOT_OBSERVED",
                "cycle_start_effect_gate=true;cycle_start_effect_observed=" + string(effectObserved ? "true" : "false") +
                ";active_state_observed=" + string(activeObserved ? "true" : "false") +
                ";segment_index=" + std::to_string(segment) +
                ";cycle_start_click_count=" + std::to_string(clickCount) +
                ";before_seqnum=" + std::to_string(beforeSeq) +
                ";last_seqnum=" + std::to_string(lastSeq) +
                ";last_prgnum=" + std::to_string(lastPrg) +
                ";last_dynamic_feed=" + std::to_string(lastFeed) +
                ";last_stat_ret=" + std::to_string(lastStatRet) +
                ";last_dynamic_ret=" + std::to_string(lastDynRet) +
                ";last_run=" + std::to_string(observedStatus.run) +
                ";last_motion=" + std::to_string(observedStatus.motion));
        return effectObserved;
    };

    auto waitUntilCycleStartReady = [&](int segment) -> bool {
        ODBST status{};
        int stableReadyPolls = 0;
        int waited = 0;
        short lastRet = 0;
        while (waited <= 15000) {
            lastRet = CallFocasLazy(inputCsv, outputCsv, idx, sniffer,
                "S012_CYCLE_START_READY_STATUS", "cnc_statinfo",
                "segment_index=" + std::to_string(segment) + ";waited_ms=" + std::to_string(waited),
                [&]() { return cnc_statinfo_fn(handle, &status); },
                [&]() { return StatinfoData(status); }, "During");
            bool ready = (lastRet == 0 && status.alarm == 0 && status.motion == 0 && (status.run == 0 || status.run == 1));
            if (ready) {
                ++stableReadyPolls;
                if (stableReadyPolls >= 3) break;
            } else {
                stableReadyPolls = 0;
            }
            Sleep(100);
            waited += 100;
        }
        bool ready = (lastRet == 0 && status.alarm == 0 && status.motion == 0 && (status.run == 0 || status.run == 1) && stableReadyPolls >= 3);
        LogHost(inputCsv, outputCsv, idx, "S012_cycle_start_ready_gate", "host_cycle_start_ready_gate",
                ready ? 0 : 1,
                ready ? "CYCLE_START_READY" : "CYCLE_START_READY_TIMEOUT",
                "cycle_start_ready_gate=true;ready=" + string(ready ? "true" : "false") +
                ";segment_index=" + std::to_string(segment) +
                ";waited_ms=" + std::to_string(waited) +
                ";stable_ready_polls=" + std::to_string(stableReadyPolls) +
                ";last_ret=" + std::to_string(lastRet) +
                ";last_run=" + std::to_string(status.run) +
                ";last_motion=" + std::to_string(status.motion) +
                ";last_alarm=" + std::to_string(status.alarm) +
                ";ready_rule=(run==0_or_run==1)_and_motion==0_and_alarm==0_stable");
        return ready;
    };

    for (int seg = 1; seg <= expected_nc_segment_count; ++seg) {
        // cycle_start_ready_gate: never click the next Single Block while the previous segment is still active.
        if (!waitUntilCycleStartReady(seg)) {
            LogHost(inputCsv, outputCsv, idx, "S012_cycle_start_ready_gate_abort", "host_cycle_start_guard",
                    1, "CYCLE_START_NOT_READY_ABORT",
                    "cycle_start_ready_gate=true;no_click=true;segment_index=" + std::to_string(seg) +
                    ";cycle_start_click_count=" + std::to_string(cycle_start_click_count));
            freeHandle(); cleanup(); return 12;
        }
        long beforeSeq = -1;
        long beforePrg = -1;
        long beforeFeed = -1;
        readDynamicSeq(beforeSeq, beforePrg, beforeFeed);
        int nextClickCount = cycle_start_click_count + 1;
        if (!TriggerNcGuideCycleStart(inputCsv, outputCsv, idx, nextClickCount)) {
            LogHost(inputCsv, outputCsv, idx, "S012_cycle_start_target_abort", "host_cycle_start_guard",
                    13, "CYCLE_START_TARGET_MISMATCH",
                    "cycle_start_target_guard=true;no_click=true;segment_index=" + std::to_string(seg) +
                    ";cycle_start_click_count=" + std::to_string(cycle_start_click_count));
            freeHandle(); cleanup(); return 13;
        }
        if (!verifyCycleStartEffect(seg, nextClickCount, beforeSeq)) {
            LogHost(inputCsv, outputCsv, idx, "S012_cycle_start_effect_abort", "host_cycle_start_guard",
                    13, "CYCLE_START_EFFECT_NOT_OBSERVED_ABORT",
                    "no_click=true;segment_index=" + std::to_string(seg) +
                    ";cycle_start_click_count=" + std::to_string(cycle_start_click_count));
            freeHandle(); cleanup(); return 13;
        }
        cycle_start_click_count = nextClickCount;
        for (int sample = 0; sample < 8; ++sample) {
            string sampleStep = "S013_PLANNED_DURING_API_sample" + std::to_string(sample);
            runEarlyAbsoluteAndFeedReads(sampleStep, seg);
            runPlannedApiList(kPlannedDuringApis, sampleStep, seg, "during");
            Sleep(200);
        }
    }

    gCurrentPhase = "After";
    bool completed = false;
    ODBST last{};
    int waited_ms = 0;
    auto startWait = std::chrono::steady_clock::now();
    while (waited_ms < 20000) {
        CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S014_program_completion_gate_poll", "cnc_statinfo",
            "program_completion_gate=true;expected_nc_segment_count=" + std::to_string(expected_nc_segment_count) +
            ";cycle_start_click_count=" + std::to_string(cycle_start_click_count),
            [&]() { return cnc_statinfo_fn(handle, &last); },
            [&]() { return "program_completion_gate=true;" + StatinfoData(last); });
        if (cycle_start_click_count >= expected_nc_segment_count && last.run == 0 && last.motion == 0) {
            completed = true;
            break;
        }
        Sleep(250);
        waited_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startWait).count());
    }
    LogHost(inputCsv, outputCsv, idx, "S014_PROGRAM_COMPLETION_GATE", "program_completion_gate",
            completed ? 0 : 1, completed ? "COMPLETED" : "TIMEOUT",
            "program_completion_gate=true;completed=" + string(completed ? "true" : "false") +
            ";timeout=" + string(completed ? "false" : "true") +
            ";waited_ms=" + std::to_string(waited_ms) +
            ";last_run=" + std::to_string(last.run) +
            ";last_motion=" + std::to_string(last.motion) +
            ";remaining_distance=sampled;expected_nc_segment_count=" + std::to_string(expected_nc_segment_count) +
            ";cycle_start_click_count=" + std::to_string(cycle_start_click_count));
    runPlannedApiList(kPlannedAfterApis, "S014_PLANNED_AFTER_API", expected_nc_segment_count, "after");

    LogHost(inputCsv, outputCsv, idx, "S015_EXECUTION_SUMMARY", "host_execution_summary", 0,
            gAccdecVariationObserved ? "VARIATION_OBSERVED" : "NO_RESPONSE_VARIATION",
            "program_completed=" + string(completed ? "true" : "false") +
            ";position_sample_count=generated_by_fixed_sampler;feed_sample_count=generated_by_fixed_sampler" +
            ";accdecdly_variation_observed=" + string(gAccdecVariationObserved ? "true" : "false") +
            ";variation_source=real_accdecdly.data[]");

    freeHandle();
    cleanup();
    return completed ? 0 : 11;
}
'''
    return (
        template.replace("__PREFERRED_PROGRAM__", str(preferred_program))
        .replace("__EXPECTED_SEGMENTS__", str(expected_segments))
        .replace("__NC_BODY__", cpp_string_literal(body))
        .replace("__PLANNED_BEFORE_APIS__", cpp_string_vector(planned_api_by_phase["before"]))
        .replace("__PLANNED_DURING_APIS__", cpp_string_vector(planned_api_by_phase["during"]))
        .replace("__PLANNED_AFTER_APIS__", cpp_string_vector(planned_api_by_phase["after"]))
        .replace("__PRE_UPLOAD_DELETE_ALL_BLOCK__", pre_upload_delete_all_block.rstrip())
        .replace("__DELETE_ALL_RESOLVE__", delete_all_resolve.rstrip())
        .replace("__DELETE_ALL_MISSING_CHECK__", delete_all_missing_check)
        .replace("__DELETE_ALL_BLOCK__", delete_all_block.rstrip())
        .replace(
            "__DELETE_ALL_PARAMETER_COUNT__",
            '    if (api == "cnc_delall") return 1;' if allow_delete_all else "",
        )
    )


def repair_cpp_api_script_after_compile_error(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    api_script: str,
    compile_error: str,
    llm_client: LlmClient,
    *,
    code_only: bool = False,
) -> str:
    if not llm_client.enabled:
        raise RuntimeError("CodeGenerationAgent cannot repair C++ without an LLM client.")
    assert state.plan is not None
    system_prompt = (
        f"{FOCAS_CPP_GENERATION_SYSTEM_PROMPT}\n\n"
        "# Task\n"
        "You are CodeGenerationAgent performing an internal compile-repair pass. "
        "Return one complete corrected C++17 source file as JSON with key cpp_code.\n\n"
        "# Rules\n"
        "- Fix the MSVC compiler errors exactly; do not replace the PlannerAgent plan or choose a different NC program number.\n"
        "- If diagnostics contain SDK_TYPE_UNKNOWN, use the supplied active Fwlib32.h prototype/type definitions; remove the invented type and select only the exact associated type. Never substitute by name similarity or define a local replacement typedef.\n"
        "- Preserve the uploaded NC payload, CSV schema, FOCAS lifecycle, selected protocol functions, and cleanup behavior unless the compiler error requires a local structural change.\n"
        "- Use the official Fwlib32.h declarations and dynamic LoadLibraryW/GetProcAddress pattern already required by the generation contract.\n"
        "- Avoid goto statements that cross initialization of C++ objects; prefer scoped cleanup helpers, early returns, or declarations before any jump target.\n"
        "- Return JSON only: {\"cpp_code\":\"// full corrected source\"}."
    )
    if code_only or state.request.permissions.get("code_only_evaluation"):
        system_prompt = (
            "You are repairing one complete FANUC FOCAS C++ source file for RQ1 code-only evaluation. "
            "Return JSON only with key cpp_code. Fix only the reported MSVC compiler error. "
            "Preserve the Planner CodeSpec API set and official Fwlib32.h ABI. "
            "Do not generate or preserve PCAP, Npcap, Wireshark, CSV, traffic collection, simulator UI, "
            "or runtime logging. Do not add or remove APIs. "
            "Every dynamically loaded API function pointer must use exactly "
            "reinterpret_cast<decltype(&::api_name)>(GetProcAddress(..., \"api_name\")); "
            "never manually spell a function-pointer typedef or signature. A generic loader must derive its cast type "
            "from the destination function-pointer member (for example std::remove_reference_t<decltype(fn)>); "
            "never hard-code decltype(&::cnc_allclibhndl3) or any one API type for all symbols. "
            "Use only the official structure type and fields recorded in CodeSpec. "
            "For cnc_rdposition, follow the official header: ODBPOS* points to an array of ODBPOS elements "
            "with official abs/mach/rel/dist POSELM fields; use a positive data_num and parse the requested axis elements. "
            "POSELM.data is one scalar long value, never an array. Put the axis subscript on the ODBPOS array: "
            "positions[i].abs.data, positions[i].mach.data, positions[i].rel.data, or positions[i].dist.data. "
            "Never generate p.data[i] or positions[i].abs.data[j]. A helper receiving one POSELM may print only p.data; "
            "a helper printing X/Y/Z must receive the ODBPOS array and index its ODBPOS elements. "
            "Never invent idata/ldata members. For cnc_absolute, "
            "use ODBAXIS directly through its official dummy/type/data[] fields; it has no abs member. If MSVC reports that ODBPOS has no dummy or data member, "
            "replace those accesses with the official ODBPOS abs/mach/rel/dist fields; do not define a replacement struct. "
            "The source must compile as C++17; do not use abbreviated function templates such as an auto parameter. "
            "Consult the SDK-derived type index for every structure member; state labels are not C++ fields. "
            "For ReadFeedSpeed, the source must contain an actual resolved cnc_actf call with an official ODBACT output object and ODBACT.data; loading or declaring cnc_actf alone is insufficient. "
            "Return one complete compilable C++17 source file."
        )
    user_prompt = (
        f"Task description:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"NC program, must preserve unless impossible:\n{nc_program}\n\n"
        f"Executable API steps:\n{steps_for_prompt(executable_steps)}\n\n"
        f"Structured CodeSpec with authoritative ABI contracts:\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
        f"SDK type definitions from the active header:\n{sdk_type_context_for_abi(state.plan.code_spec.official_abi_context if state.plan.code_spec is not None else '')}\n\n"
        f"MSVC preflight compiler diagnostics:\n{compile_error}\n\n"
        "Previous C++ source to repair:\n"
        f"{api_script}"
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    repaired = str(payload.get("cpp_code", "")).strip()
    if not repaired:
        repaired = str(payload.get("api_script", "")).strip()
    repaired = normalize_focas_header_include(strip_cpp_fence(repaired))
    if not repaired:
        raise ValueError("CodeGenerationAgent compile-repair LLM returned an empty C++ API script.")
    if state.request.permissions.get("code_only_evaluation"):
        import re

        repaired = normalize_nc_download_payload(repaired)
        repaired = normalize_official_odbpos_field_names(repaired)
        repaired = normalize_official_odbpos_axis_access(repaired)
        repaired = normalize_official_odbaxis_fields(repaired)
        repaired = repaired.replace("print_axis_data(", "print_axis_positions(")
        repaired = re.sub(
            r"print_rdposition\(\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*\)",
            r"print_rdposition(\1[0])",
            repaired,
        )
        repaired = normalize_cpp17_auto_parameters(repaired)
        if not validate_sdk_struct_members(api_script):
            repaired = preserve_code_only_abi_preamble(api_script, repaired)
        # Recheck ABI corrections after any verified preamble is restored.
        repaired = normalize_official_odbpos_field_names(repaired)
        repaired = normalize_official_odbpos_axis_access(repaired)
        repaired = normalize_official_odbaxis_fields(repaired)
        repaired = normalize_cpp17_auto_parameters(repaired)
    sdk_errors = validate_sdk_struct_members(repaired)
    if sdk_errors:
        raise ValueError("CodeGenerationAgent repair still has SDK_TYPE_MISMATCH: " + "; ".join(sdk_errors))
    return repaired


def preserve_code_only_abi_preamble(previous_source: str, repaired_source: str) -> str:
    """Keep the verified ABI/load section immutable during compile repair."""
    main_marker = "int main()"
    previous_main = previous_source.find(main_marker)
    repaired_main = repaired_source.find(main_marker)
    if previous_main < 0 or repaired_main < 0:
        return repaired_source
    return previous_source[:previous_main] + repaired_source[repaired_main:]


def normalize_official_odbpos_field_names(api_script: str) -> str:
    """Correct only fields on variables declared with the official ODBPOS type."""
    import re

    names = set(re.findall(r"\bODBPOS\s+(\w+)\s*(?:\[|\{|;|=)", api_script))
    for name in names:
        api_script = re.sub(rf"\b{re.escape(name)}\s*(\.|->)\s*dummy\b", rf"{name}\1abs", api_script)
        api_script = re.sub(rf"\b{re.escape(name)}\s*(\.|->)\s*data\b", rf"{name}\1abs.data", api_script)
    return api_script


def normalize_official_odbpos_axis_access(api_script: str) -> str:
    """Treat cnc_rdposition output as an array of per-axis ODBPOS objects."""
    import re

    fields = r"(?:abs|mach|rel|dist)"
    reference_parameters = set(re.findall(r"\bconst\s+ODBPOS\s*&\s*(\w+)", api_script))
    for name in reference_parameters:
        indexed_field = re.compile(
            rf"\b{re.escape(name)}\s*\.\s*({fields})\s*\.\s*data\s*\[\s*([^\]]+)\s*\]"
        )
        if not indexed_field.search(api_script):
            continue
        api_script = re.sub(
            rf"\bconst\s+ODBPOS\s*&\s*{re.escape(name)}\b",
            f"const ODBPOS* {name}",
            api_script,
        )
        api_script = indexed_field.sub(rf"{name}[\2].\1.data", api_script)

    array_names = set(re.findall(r"\bODBPOS\s+(\w+)\s*\[", api_script))
    for name in array_names:
        api_script = re.sub(
            rf"\b{re.escape(name)}\s*\.\s*({fields})\s*\.\s*data\s*\[\s*([^\]]+)\s*\]",
            rf"{name}[\2].\1.data",
            api_script,
        )
    return api_script


def validate_focas_loader_type_safety(api_script: str) -> list[str]:
    """Reject generic symbol loaders that cast every API to one fixed ABI type."""
    import re

    generic_loader = re.search(
        r"(?:auto|template\s*<[^>]+>)[\s\S]{0,500}?"
        r"(?:auto\s*&|[A-Za-z_]\w*\s*&)?\s*fn[\s\S]{0,500}?"
        r"using\s+\w+\s*=\s*decltype\s*\(\s*&\s*::\s*((?:cnc|pmc)_\w+)\s*\)"
        r"[\s\S]{0,500}?GetProcAddress\s*\([^;]+?\bname\b",
        api_script,
        re.IGNORECASE,
    )
    if generic_loader is None:
        return []
    fixed_api = generic_loader.group(1)
    symbol_names = set(
        re.findall(
            r"\b(?:load|resolve)\s*\([^;\n]*?\"((?:cnc|pmc)_[A-Za-z0-9_]+)\"",
            api_script,
            re.IGNORECASE,
        )
    )
    if len(symbol_names) <= 1:
        return []
    return [
        "BLOCKING: generic FOCAS symbol loader casts multiple APIs through the fixed type "
        f"decltype(&::{fixed_api}). Derive the cast from the destination member, for example "
        "std::remove_reference_t<decltype(fn)>, or use an explicit decltype(&::api_name) cast for each symbol."
    ]


def validate_poselem_scalar_access(api_script: str) -> list[str]:
    """Reject array subscripts applied to the scalar POSELM.data field."""
    import re

    violations: set[str] = set()
    nested_pattern = re.compile(
        r"\b\w+(?:\s*\[[^\]]+\])?\s*(?:\.|->)\s*"
        r"(?:abs|mach|rel|dist)\s*(?:\.|->)\s*data\s*\[[^\]]+\]",
        re.IGNORECASE,
    )
    violations.update(match.group(0) for match in nested_pattern.finditer(api_script))

    poselem_names = set(
        re.findall(
            r"\b(?:const\s+)?POSELM\s*(?:[&*]\s*)?(\w+)",
            api_script,
            re.IGNORECASE,
        )
    )
    for name in poselem_names:
        direct_pattern = re.compile(
            rf"\b{re.escape(name)}\s*(?:\.|->)\s*data\s*\[[^\]]+\]",
            re.IGNORECASE,
        )
        violations.update(match.group(0) for match in direct_pattern.finditer(api_script))

    if not violations:
        return []
    examples = ", ".join(sorted(violations)[:4])
    return [
        "BLOCKING: POSELM.data is a scalar long in the active Fwlib32.h and cannot be subscripted "
        f"({examples}). For multiple axes, index the ODBPOS output array and then access the scalar field, "
        "for example positions[i].abs.data or positions[i].mach.data."
    ]


def validate_odbact_scalar_access(api_script: str) -> list[str]:
    """Reject array subscripts applied to the scalar ODBACT.data field."""
    import re

    names = set(re.findall(r"\bODBACT\s+(\w+)", api_script, re.IGNORECASE))
    violations = {
        match.group(0)
        for name in names
        for match in re.finditer(
            rf"\b{re.escape(name)}\s*(?:\.|->)\s*data\s*\[[^\]]+\]",
            api_script,
            re.IGNORECASE,
        )
    }
    if not violations:
        return []
    return [
        "BLOCKING: ODBACT.data is a scalar long in the active Fwlib32.h and cannot be subscripted "
        f"({', '.join(sorted(violations)[:4])}). Use act.data."
    ]


def validate_official_odbst_fields(api_script: str) -> list[str]:
    """Compatibility wrapper for callers; validation is now SDK-derived."""
    header_path = default_focas_header_dir() / "Fwlib32.h"
    if not header_path.is_file():
        return [f"BLOCKING: SDK_TYPE_MISMATCH: active Fwlib32.h not found: {header_path}"]
    index_path = Path("rag_indexes/focas/sdk_type_index_0iD.json")
    index = load_or_build_sdk_type_index(header_path, index_path)
    return validate_struct_member_access(api_script, index)


def validate_sdk_struct_members(api_script: str) -> list[str]:
    """Validate all generated SDK struct/union member accesses from the active header."""
    return validate_official_odbst_fields(api_script)

def normalize_official_odbaxis_fields(api_script: str) -> str:
    """Correct ODBAXIS member hallucinations without touching other types."""
    import re

    names = set(re.findall(r"\bODBAXIS\s+(\w+)\s*(?:\{|;|=)", api_script))
    for name in names:
        api_script = re.sub(rf"\b{re.escape(name)}\s*(\.|->)\s*abs\s*\.\s*data\b", rf"{name}\1data", api_script)
    return api_script


def normalize_nc_download_payload(api_script: str) -> str:
    """Normalize the generated NC string to FOCAS download framing.

    This intentionally touches only the first assignment to a conventional NC
    payload variable. It does not add or replace scenario content.
    """
    import re

    assignment = re.search(
        r"\b(?:program|payload|nc_program|nc_program_text)\b\s*(?:\[[^\]]*\])?\s*=\s*",
        api_script,
        re.IGNORECASE,
    )
    if not assignment:
        return api_script

    semicolon = api_script.find(";", assignment.end())
    if semicolon < 0:
        return api_script
    segment = api_script[assignment.start() : semicolon]

    # A common model output is "%\\n" "O..." ... "%\\n". The documented
    # payload has no leading percent line and exactly one trailing percent.
    percent_literals = list(re.finditer(r'"%\\n"', segment))
    if len(percent_literals) >= 2:
        first = percent_literals[0]
        segment = segment[: first.start()] + '""' + segment[first.end() :]
        percent_literals = list(re.finditer(r'"%\\n"', segment))
        last = percent_literals[-1]
        segment = segment[: last.start()] + '"%"' + segment[last.end() :]
    elif len(percent_literals) == 1:
        only = percent_literals[0]
        first_o = re.search(r'"O[A-Za-z0-9_]+', segment, re.IGNORECASE)
        if first_o is not None and only.start() > first_o.start():
            segment = segment[: only.start()] + '"%"' + segment[only.end() :]
        else:
            segment = segment[: only.start()] + '""' + segment[only.end() :]

    # Ensure the first O-number literal is preceded by the required LF. This
    # handles both split literals and a single literal such as "O1234\\n".
    first_o = re.search(r'"O(?P<number>[A-Za-z0-9_]+)', segment, re.IGNORECASE)
    if first_o:
        before = segment[: first_o.start()]
        if r"\n" not in before[-12:]:
            segment = segment[: first_o.start()] + '"\\n"\n        ' + segment[first_o.start() :]

    return api_script[: assignment.start()] + segment + api_script[semicolon:]


def normalize_cpp17_auto_parameters(api_script: str) -> str:
    """Convert the common abbreviated loader template to valid C++17 syntax."""
    import re

    return re.sub(
        r"static\s+bool\s+load_symbol\(FocasApi\s*&api,\s*FARPROC\s+proc,\s*auto\s*&fn,\s*const\s+char\s*\*name\)",
        "template <typename Fn>\\nstatic bool load_symbol(FocasApi &api, FARPROC proc, Fn &fn, const char *name)",
        api_script,
    )


def repair_cpp_api_script_after_contract_error(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    api_script: str,
    contract_error: str,
    llm_client: LlmClient,
    *,
    code_only: bool = False,
) -> str:
    """Repair generated source against the atomic API/CSV communication contract."""
    assert state.plan is not None
    system_prompt = (
        f"{FOCAS_CPP_GENERATION_SYSTEM_PROMPT}\n\n"
        "You are performing a strict post-generation contract repair. Return JSON only with key cpp_code.\n"
        "Preserve the real FOCAS calls, parameters, NC payload, and cleanup. Do not remove required calls.\n"
        "Every LogInput/LogOutput API field must contain exactly one real API name or one host helper name.\n"
        "For a sequence such as cnc_dwnstart3, cnc_download3, cnc_dwnend3, emit separate input/output rows for each real call.\n"
        "Never put multiple API names joined by '/', '+', ';', ',', or '->' in api_name or protocol_function fields.\n"
        "Do not hide calls in aggregate telemetry rows; each real call must have its own row and exact API name.\n"
        "Keep the source complete, compilable C++17, and preserve the ExecutionAgent CSV headers and matching indices."
    )
    if code_only:
        system_prompt = (
            "You are repairing one complete FANUC FOCAS C++ source file for RQ1 code-only evaluation. "
            "Return JSON only with key cpp_code. Preserve the Planner atomic API contract and official ABI. "
            "Fix only the reported issue. Do not generate PCAP/Npcap/Wireshark, CSV files, traffic collection, "
            "simulator UI automation, runtime logging, placeholders, or fake results. Keep connection, exact API "
            "arguments, return-code handling, and cnc_freelibhndl cleanup. Return a complete compilable source file. "
            "The structured CodeSpec below is authoritative; do not infer requirements from the old source. "
            "For every dynamically loaded API, the function-pointer alias must be declared exactly as "
            "using Fn = decltype(&::api_name); never manually spell a function-pointer signature. "
            "If one generic loader accepts auto& fn, its cast type must be std::remove_reference_t<decltype(fn)>; "
            "never use one fixed API decltype for every loaded symbol. "
            "For cnc_rdposition, ODBPOS* points to an array of official objects with abs/mach/rel/dist POSELM fields; "
            "use a positive data_num and do not invent idata/ldata members. "
            "POSELM.data is a scalar long, so axis indexing belongs on the ODBPOS array: positions[i].abs.data. "
            "Never subscript POSELM.data and never treat one POSELM as an XYZ array."
            " For ODBPOS, read the official abs/mach/rel/dist fields; "
            "for Windows wide-character fwprintf diagnostics, use %ls for wchar_t*."
        )
    user_prompt = (
        f"Task:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"NC program:\n{nc_program}\n\n"
        f"Atomic executable steps:\n{steps_for_prompt(executable_steps)}\n\n"
        f"Structured CodeSpec:\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
        f"Named API argument contract:\n{named_contract_context([name for step in executable_steps for name in protocol_function_names(step.protocol_function)])}\n\n"
        f"Contract violation:\n{contract_error}\n\n"
        f"Source to repair:\n{api_script}"
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    repaired = str(payload.get("cpp_code", payload.get("api_script", ""))).strip()
    repaired = normalize_focas_header_include(strip_cpp_fence(repaired))
    if not repaired:
        raise ValueError("CodeGenerationAgent contract-repair LLM returned an empty C++ API script.")
    if code_only:
        repaired = normalize_nc_download_payload(repaired)
        repaired = normalize_official_odbpos_field_names(repaired)
        repaired = normalize_official_odbpos_axis_access(repaired)
    return repaired


def generate_cpp_api_script_with_llm(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
) -> str:
    assert state.plan is not None
    # Deprecated compatibility entry point: keep the model on the same bare
    # API contract as the production path.  The local SDK assembler produces
    # the statements-only capture fragment; this function no longer asks the
    # model for a complete source file or a ``ctx.call`` wrapper.
    return generate_api_fragments_with_llm(
        state, executable_steps, nc_program, llm_client, wrap=False,
    )
    selected_functions = sorted(
        {
            function_name
            for step in executable_steps
            for function_name in protocol_function_names(step.protocol_function)
        }
    )
    history_target = state.plan.target_api or next(
        (name for step in executable_steps if getattr(step, "role", "") == "target"
         for name in protocol_function_names(step.protocol_function)),
        (selected_functions[-1] if selected_functions else ""),
    )
    verified_api_examples = retrieve_verified_api_examples(
        history_target,
        output_dir=Path(state.plan.rag_context.get("output_dir", "")) if state.plan.rag_context.get("output_dir") else None,
        limit=3,
        sdk_identity=str(sdk_environment_identity(
            default_focas_header_dir(),
            Path(os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir()))) / "Fwlib32.dll",
        )),
    )
    state.plan.rag_context["verified_api_examples"] = verified_api_examples
    official_abi_context = official_focas_abi_context(selected_functions)
    sdk_type_context = sdk_type_context_for_abi(official_abi_context)
    code_only = bool(state.request.permissions.get("code_only_evaluation"))
    system_prompt = (
        f"{FOCAS_CPP_GENERATION_SYSTEM_PROMPT}\n\n"
        "# Current Output Contract\n"
        "- You are generating the complete C++ source file, not reviewing it.\n"
        "- Return JSON only with key cpp_code.\n"
        "- cpp_code must be a full single-file C++17 program containing main().\n"
        "- The generated C++ must compile with MSVC using Windows/User32, the official controller-specific FOCAS header, and dynamic DLL loading.\n"
        "- Because windows.h defines min/max macros, the C++ must either define NOMINMAX before including windows.h or avoid std::min/std::max entirely.\n"
        "- Include the official header exactly as #include <Fwlib32.h>. The configured header is the FANUC Series 0i-D SDK header.\n"
        "- Use official Fwlib32.h structures, constants, and declarations. Do not redeclare FOCAS structs or hand-write FOCAS function-pointer prototypes.\n"
        "- Derive every dynamically loaded FOCAS function type from the official declaration, for example: auto cnc_rdposition_fn = reinterpret_cast<decltype(&::cnc_rdposition)>(GetProcAddress(dll, \"cnc_rdposition\"));.\n"
        "- Continue using LoadLibraryW and GetProcAddress; do not call imported FOCAS functions directly and do not require Fwlib32.lib.\n"
        "- The user prompt contains official ABI excerpts extracted from the configured Fwlib32.h for Planner-selected functions. Treat those declarations and structure fields as exact; never invent fields that are absent.\n"
        "- The named API contract below supplies the meaning and order of each argument because some Fwlib32.h declarations omit names. If Planner parameter keys conflict with that contract, the documented named argument order wins. Do not pass a count such as data_num as a byte length; for a single output structure use sizeof(the actual structure) where the API documents a byte-length argument. Log the canonical argument names and actual values.\n"
        "- Before emitting each call, bind every expression to the ordered parameter list from Fwlib32.h and the SDK type index. Same-typed adjacent parameters are not interchangeable: for cnc_relative/cnc_absolute/cnc_machine/cnc_distance and their variants emit (handle, axis, length, &relative) exactly, then verify the logged values match those expressions.\n"
        "- Derive EW_* return labels from the configured Fwlib32.h constants. Never invent or reorder numeric error-code mappings; in particular a length error must be routed as a parameter/length repair, not an offset-number repair.\n"
        "- For FOCAS connection, follow cpp/focas_connect_demo.cpp exactly: read FOCAS_DLL_DIR with GetEnvironmentVariableW, call SetDllDirectoryW(dll_dir), load dll_dir + '\\\\Fwlib32.dll' with LoadLibraryW, resolve cnc_allclibhndl3/cnc_freelibhndl, then connect using FOCAS_HOST and FOCAS_PORT environment variables (defaults 127.0.0.1 and 8193) with timeout 10.\n"
        "- Do not hard-code mojibake/garbled Chinese DLL paths inside generated C++; use FOCAS_DLL_DIR and wide-character Windows APIs for DLL loading.\n"
        "- Write data/focas_api_input.csv and data/focas_api_output.csv with UTF-8 BOM.\n"
        "- Every planned executable step must produce one input CSV row and one output CSV row per repeat.\n"
        "- Every input/output CSV row must include a timestamp column recorded at call time. Use ISO-8601 local time, "
        "ISO-8601 UTC time, or epoch milliseconds consistently. Matching input/output rows for one API call should share "
        "the same timestamp when possible.\n"
        "- Planner parameter_generation describes input domains and constraints, not concrete values. Resolve "
        "actual values here from the exact API documentation and device observations. Preserve every coupling "
        "constraint and generate only valid combinations within documented bounds/enums. "
        "For each generated combination, emit a real API call and matching input/output CSV rows. "
        "Do not treat input variation as traffic quality; it is only the call-generation strategy.\n"
        "- The executable steps below are selected by PlannerAgent/LLM from task context and RAG; do not reject them merely because of local registry assumptions.\n"
        "- Every Planner semantic read step must include an actual invocation in main, not only a declaration or LoadProc. For ReadFeedSpeed, declare official ODBACT actualFeed{}, call the resolved cnc_actf pointer with (handle, &actualFeed), check the return code, and use actualFeed.data.\n"
        "- The listed interface meanings are examples, not the full API universe. If PlannerAgent provides an exact protocol_function that is not in the examples, use the retrieved API/rule knowledge to generate the required dynamic GetProcAddress typedef, argument construction, API call, output parsing, CSV logging, and cleanup.\n"
        "- FOCAS APIs with similar names are not ABI-compatible. Derive each resolved symbol's type from its exact official declaration in Fwlib32.h; never reuse another family member's type.\n"
        "- For cnc_rdposition, allocate a positive-count ODBPOS array and parse each requested element's abs/mach/rel/dist POSELM fields exactly as declared in Fwlib32.h. POSELM.data is one scalar long value, not an axis array. Read multiple axes as positions[i].abs.data, positions[i].mach.data, positions[i].rel.data, or positions[i].dist.data; never use p.data[i] or positions[i].abs.data[j]. Never use a negative axis selector as data_num.\n"
        "- For cnc_absolute, cnc_absolute2, cnc_machine, cnc_relative, cnc_relative2, and cnc_distance, the third argument is the documented ODBAXIS data-block length; for FS0i-D all-axis reads use 4 + 4 * MAX_AXIS (MAX_AXIS=32), not a guessed count or pointer. For cnc_wrrelpos, the documented read-back counterpart is cnc_relative or cnc_relative2, never cnc_rdrelpos.\n"
        "- Helper functions used across feed/position/status metric vectors must be type-correct. Prefer a function template such as template<class T> bool HasVariation(const std::vector<T>&) instead of a std::vector<long>-only helper called with std::vector<short>.\n"
        "- Avoid goto statements that jump over initialization of C++ objects such as std::string or std::vector. Prefer early returns, scoped cleanup helpers, or declare all such objects before any possible jump target.\n"
        "- If an API/control step is infeasible, generate a real best-effort implementation plus explicit diagnostics, or fail generation with a clear reason; do not silently skip required steps.\n"
        "- Do not emit SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN, not_executed_by_cpp_generator, placeholder skip rows, or non-fatal diagnostics in place of required executable steps.\n"
        "- The generated C++ must follow the ExecutionAgent CSV contract exactly.\n"
        "- focas_api_input.csv header must be: index,timestamp,step_id,interface_name,protocol_function,parameters,api_parameter_count\n"
        "- focas_api_output.csv header must be: index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n"
        "- The same monotonically increasing integer index must be written to matching input/output rows.\n"
        "- For operation_kind=focas_sequence, expand api_calls into atomic logging and execution. Each api_calls element is one real FOCAS call: write one input row with its exact single protocol_function, execute it, and write one output row with the same exact single API name. Never write slash, plus, semicolon, arrow, or other delimiters between multiple API names in any api_name or protocol_function field. For example, cnc_dwnstart3, cnc_download3, and cnc_dwnend3 require three separate input/output row pairs.\n"
        "- When execution_contract.api_prerequisites is present, implement its documented checks and operations in order before target calls. Log every preparation API separately, keep preparation/support traffic distinct from target coverage, and use only the exact sourced parameter value, type, scope, and SDK constraints. Do not infer snapshot, readback, restoration, or a single candidate from a write classification. If restart_required is true, record a pending reinitialization state and stop before the target; do not assume a successful write is active.\n"
        "- Download API versions are strict session pairs: cnc_download3 uses cnc_dwnstart3/cnc_dwnend3; cnc_download4 uses cnc_dwnstart4/cnc_dwnend4. Never mix versions. For an NC program downloaded by cnc_download4, use cnc_dwnstart4(handle, 0, nullptr) when selecting the current CNC folder, and log dir_name=NULL;destination=current_folder. Check the matching start return code and skip transfer on any non-EW_OK result; call the matching end only for a session that was successfully opened. Do not claim a specific absolute destination or successful registration without directory/read-back evidence.\n"
        "- cnc_dncstart2 is a CNC-side DNC/M198 activity query, not a DNC start command. It returns EW_OK only when DNC/M198 is already active and EW_REJECT when it is not. Call cnc_dnc2/cnc_dncend2 only after cnc_dncstart2 returns EW_OK; never make negative probes after EW_REJECT to manufacture EW_FUNC traffic. DNC requires independently verified CNC operating mode/options/signals; do not switch modes, change parameters, or claim NCGuide unsupported without direct evidence.\n"
        "- A high-level interface_name such as UploadProgram may describe a sequence, but it must never collapse several real calls into one CSV record.\n"
        "- Treat tool_calls as the authoritative communication format from PlannerAgent. Each tool_calls element has one tool_name and one arguments object. Preserve call_id, operation_id, phase, tool_name, and arguments when generating code and logs. Do not reconstruct a composite tool_name from multiple elements.\n"
        "- If a previous C++ script failed, revise it based on the previous script preview, diagnostics, compile/runtime errors, and quality assessment.\n"
        "- If repair_context reports LLM timeout, empty cpp_code, or invalid executable steps, simplify: keep only required selected steps, use the scaffold directly, and still return one complete strict JSON object.\n"
        "- UploadProgram must preserve the NC program body below, but the O-number is an allocation detail owned by CodeGenerationAgent/runtime. Treat CodeGenerationAgent's candidate as a preferred/base O number, then after reading the controller directory choose the actual upload O number in C++.\n"
        "- If PlannerAgent provided coverage_segments, implement them sequentially inside this one generated C++ program. Use clear segment markers in step_id/action/CSV data such as segment_id=01_programmed_coordinate_motion. A segment may have its own NC payload or no NC payload at all.\n"
        "- For coverage_segments with nc_program_required=true, the provided NC program is a text payload for transfer/verification or execution as specified. Only segments with cycle_start_required=true may upload/select/run it and use Cycle Start. A program-transfer segment with cycle_start_required=false may call its target download/verify APIs but must not run the payload. For segments with nc_program_required=false, use direct FOCAS calls and do not add NC upload or Cycle Start.\n"
        "- For each segment, distinguish target functions from support functions in CSV data fields: coverage_role=target or coverage_role=support. Support calls may repeat to create/verify/recover state; target calls are the coverage objectives.\n"
        "- Each CSV output row must represent exactly one real FOCAS/PMC API call or one host/UI helper. Do not put semicolon-separated API names such as cnc_a;cnc_b in one api_name/protocol_function field. Do not create LifecycleManifestCoverage/OffsetManifestCoverage summary rows, and do not mark coverage_role=target unless that specific function pointer was resolved and that specific API was actually called.\n"
        "- Implement real Wireshark/Npcap packet capture with a PacketSniffer class and instantiate it in main with syntax like PacketSniffer sniffer(networkDevice, pcapFile); call sniffer.Start(), sniffer.CapturePackets(...) immediately before and after each FOCAS API call, and sniffer.Stop() during cleanup. PacketSniffer must use Npcap/WinPcap APIs such as pcap_open_live, pcap_dump_open, pcap_dispatch, pcap_dump, and pcap_dump_flush, loaded dynamically if needed. Read networkDevice from the PCAP_NETWORK_DEVICE environment variable, defaulting to the calibrated Npcap loopback device \\\\Device\\\\NPF_Loopback; do not hardcode \\\\Device\\\\NPF_{CHANGE_ME}. Never create fake/minimal pcap files by writing only a pcap header. For this task, packet capture is a hard execution gate: if sniffer.Start() fails, log pcap_capture_enabled=false and PCAP_CAPTURE_FAILED, close resources, and exit nonzero before calling FOCAS APIs. On success, log pcap_capture_enabled=true or pcap_capture_started=true in focas_api_output.csv.\n"
        "- The generated MSVC executable is x86. Resolve Npcap from %WINDIR%\\\\SysWOW64\\\\Npcap\\\\wpcap.dll first with LOAD_WITH_ALTERED_SEARCH_PATH (sibling Packet.dll), then a safe fallback. Log GetLastError or pcap errbuf if it fails; never continue without capture.\n"
        "- Follow the documented cnc_download3 NC payload format exactly: the first byte must be LF, followed by the runtime-selected O-number and NC blocks separated by LF, with one trailing percent character at the end. Use a payload like \"\\nO1234\\nG90...\\nM30\\n%\". Do not prepend a percent line before the leading LF and do not append data after the final percent.\n"
        "- Before cnc_dwnstart3, implement the exact program-existence step selected by PlannerAgent from RAG using that API's documented prototype and arguments. Set target_program_exists=true only when a successful returned directory/read result contains an entry whose program number exactly equals the candidate O number. Do not replace PlannerAgent's API choice with a locally preferred directory API.\n"
        "- For cnc_rdprogdir-style target existence checks, use bounded windows near the target O number and documented positive entry counts. Do not issue a broad top=1/bottom=9999 request with num=1 as the only check. Return code 2 is EW_LENGTH in the configured SDK, not EW_NUMBER, and a nonzero return is not proof that the target exists; log the actual API-specific error and repair the directory-read parameter strategy rather than declaring target_program_exists=true.\n"
        "- Runtime O-number handling must be in generated C++ if logic after reading the directory: try the preferred/base O number first; if it exists, choose among these explicit branches: (1) choose a nearby unused valid O number and rebuild the NC payload with that selected_program_number; (2) delete exactly the conflicting program with cnc_delete and then upload the same selected_program_number; or (3) when Explicit task permissions contains allow_delete_all_programs=true, call cnc_delall to clear program memory before upload. Log preferred_program_number, selected_program_number, target_program_exists, program_number_available, and collision_strategy=choose_unused, collision_strategy=delete_exact_conflict, or collision_strategy=delete_all_authorized.\n"
        "- Single-program deletion is allowed only for an exact confirmed conflict. If using cnc_delete, call it only with the exact candidate O number that directory/search proved exists, log delete_called=true, delete_target_program=Oxxxx, exact_match=true, and abort with PROGRAM_REPLACEMENT_FAILED if cnc_delete fails. Never delete a different program number and never use deletion as a broad cleanup step.\n"
        "- cnc_delall is permission-gated. Generate it when Explicit task permissions contains allow_delete_all_programs=true and runtime collision/cleanup logic chooses full program-memory cleanup for this isolated NCGuide experiment. Otherwise its presence is a blocking safety violation. When authorized, log delete_all_authorized=true, collision_strategy=delete_all_authorized, check its return code, and abort with PROGRAM_REPLACEMENT_FAILED if the call fails.\n"
        "- Any existence-check error other than a documented empty result must abort before cnc_dwnstart3; do not infer that the candidate is available from an error return.\n"
        "- cnc_dwnend3 may report delayed cnc_download3 errors. If cnc_getdtailerr reports err_no=4 despite the verified replacement flow, abort and report that the target O number still exists; do not retry blindly. If err_no=5, the same program is selected and execution must stop for a safe re-selection/repair.\n"
        "- SelectProgram must select the same runtime selected_program_number as the uploaded NC program.\n"
        "- ReadProgramNumber must call cnc_rdprgnum and log the active/main program numbers.\n"
        "- Program lifecycle calls are hard gates. Check cnc_dwnstart3, every cnc_download3 call, cnc_dwnend3, and cnc_search return codes. Then call cnc_rdprgnum and compare the returned current/main program number with the uploaded O number. Log program_verified=true only on an exact match. On any lifecycle failure or mismatch, log PROGRAM_NOT_VERIFIED and exit nonzero before the first Cycle Start. Never log selected_program=<expected> as if selection succeeded when cnc_search returned an error.\n"
        "- StartProgram must use NCGuide UI Cycle Start coordinates from environment variables NCGUIDE_CYCLE_START_X, NCGUIDE_CYCLE_START_Y, NCGUIDE_CLICK_MODE, and NCGUIDE_WINDOW_TITLE, not hardcoded SetCursorPos literals. It may use screen/client click parameters from those environment variables, but it must not blindly click while the previous single-block segment is still running. "
        "Implement a helper named WaitUntilCycleStartReady or an equivalent block containing the marker cycle_start_ready_gate. "
        "Before every Cycle Start click, poll cnc_statinfo and remaining distance until the previous block is complete or a bounded timeout expires; accept run=0,motion=0 (initial idle) AND run=1,motion=0 (Single Block paused/ready for the next click) when no movement remains. Never require run=0 between Single Block lines. Prefer cnc_distance for remaining distance, or use cnc_rdposition(type=3)/ODBPOS.dist when cnc_distance is not available. Log waited_ms, last run/motion values, distance-to-go values, and ready/timeout in the CSV data. "
        "If the gate times out, do not click again immediately; log a diagnostic row and continue with sampling or return a clear nonzero status.\n"
        "- For this confirmed NCGuide Single Block setup, every non-empty NC line in the exact uploaded payload consumes one Cycle Start: the O program-number line, modal/setup-only lines, motion lines, and M30 all count. Count these effective NC segments from the exact payload (ignoring only blank/comment-only lines), log expected_nc_segment_count and cycle_start_click_count, and issue enough readiness-gated clicks to execute the M30 segment. Do not generate a fixed click list based only on motion blocks or Planner StartProgram step count.\n"
        "- Use a bounded whole-program Single Block loop (for example RunUploadedProgramToCompletion) whose click limit is at least the effective NC segment count. A ready-gate timeout must abort without clicking. Setup/program-number segments may produce no motion and must be treated as warmup/lifecycle segments rather than failed motion samples. Only declare the click loop complete after cycle_start_click_count reaches the expected effective segment count; then run the final program_completion_gate.\n"
        "- For NC motion timing, do not compensate for missing samples by making every move extremely slow. Prefer multiple 2-5 second observable blocks and 100-200 ms sampling intervals; if prior traffic was too sparse, adjust feed/travel moderately.\n"
        "- For coordinate-motion traffic, explicitly sample and log distance-to-go/remaining move data. Prefer a ReadDistanceToGo/cnc_distance call; cnc_rdposition(type=3) or ODBPOS.dist is acceptable as a fallback. "
        "Use parameter names such as distance_to_go, dist_axis1, dist_axis2, dist_axis3, remaining_move, or remaining_distance so RouterAgent can recognize them.\n"
        "- Before local evaluation, DISCONNECT, or normal process exit after any StartProgram step, wait for NC program completion using a helper named WaitUntilProgramComplete or a CSV marker program_completion_gate. "
        "This completion gate must keep sampling cnc_statinfo and cnc_distance/cnc_rdposition(type=3) until the program is idle/complete and remaining movement is zero/stable, or until a bounded timeout is logged as a failure. "
        "The final gate output row must include program_completion_gate, completed=true/false, timeout=true/false, waited_ms, last_run, last_motion, and distance_to_go or remaining_distance fields. "
        "Do not report overall success, do not run local evaluation as successful, and do not disconnect as a normal completed run while last_run/last_motion indicate active motion or the completion gate timed out.\n"
        "- Idle status between Single Block segments is not whole-program completion. The final program_completion_gate may only report completed=true after program_verified=true and after all effective NC segments, including M30, have received their guarded Cycle Start. Include expected_nc_segment_count and cycle_start_click_count in the final gate row.\n"
        "- Packet capture support is mandatory for this generated traffic run. Do not continue to FOCAS calls when Npcap/WinPcap capture cannot be opened.\n"
        "- Include resource cleanup: cnc_freelibhndl, packet capture close, FreeLibrary.\n\n"
        "# Recommended C++ Scaffold\n"
        "Use this scaffold as the starting structure. Fill in the step-specific FOCAS calls, CSV rows, sleeps, NC payload upload, "
        "program selection, and NCGuide Cycle Start logic according to the executable API steps. "
        "You may revise helper signatures if needed, but keep the same overall structure and resource cleanup.\n"
        f"{cpp_generation_scaffold()}\n\n"
        "# Output Schema\n"
        "{\"cpp_code\":\"// complete C++ source here\", \"notes\":[\"short note\"]}"
    )
    planned_interfaces = {step.interface_name for step in executable_steps}
    if "StartProgram" not in planned_interfaces:
        system_prompt += (
            "\n# Binding scenario boundary\n"
            "The executable plan has no StartProgram. Do not click Cycle Start, call SetCursorPos/mouse_event/SendInput, "
            "or insert a program-execution loop. Any motion helper in the recommended scaffold is inapplicable here.\n"
        )
    transfer_family = {"cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"}
    if "UploadProgram" not in planned_interfaces and not transfer_family.intersection(selected_functions):
        system_prompt += (
            "The executable plan has no UploadProgram. Do not call cnc_dwnstart3, cnc_download3, "
            "cnc_dwnend3, or upload any NC payload. Use only the planned API operations and required support calls.\n"
        )
    system_prompt += (
        "\nFinal API argument contract (documented names, installed SDK types):\n"
        + named_contract_context(selected_functions)
        + "\nIf a Planner parameter label conflicts, preserve the operation goal but correct the call arguments before execution.\n"
    )
    user_prompt = (
        f"Task:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"Scenario goal: {state.plan.scenario_goal}\n"
        f"Target environment: {state.request.target_environment}\n"
        f"Explicit task permissions: {state.request.permissions}\n"
        f"FOCAS runtime dir: {default_focas_runtime_dir()}\n"
        f"Official FOCAS header: {default_focas_header_dir() / 'Fwlib32.h'}\n"
        f"Official ABI excerpts for Planner-selected functions:\n{official_abi_context}\n\n"
        f"SDK-derived structure/union field index (authoritative; generated from the active header):\n{sdk_type_context}\n\n"
        f"Verified historical same-API examples (successful return + matched PCAP only; validate against the active SDK before reuse):\n{json.dumps(verified_api_examples, ensure_ascii=False, indent=2)}\n\n"
        f"Pre-generation SDK type contract lookup (read-only; reuse these existing types and never define local SDK structs):\n{json.dumps(state.plan.rag_context.get('sdk_type_contract', {}), ensure_ascii=False, indent=2)}\n\n"
        f"NC program:\n{nc_program}\n\n"
        f"Coverage segments for this run: {state.plan.rag_context.get('coverage_segments', [])}\n"
        f"Function coverage manifest context: {state.plan.rag_context.get('function_coverage_manifest', {})}\n"
        f"Quality analysis: {state.plan.rag_context.get('planning_quality_analysis', {})}\n"
        f"Quality targets: {state.plan.rag_context.get('quality_targets', {})}\n"
        f"Repair context from previous failed attempt: {summarize_codegen_repair_context(state)}\n"
        f"CodeGenerationAgent direct RAG knowledge: {state.plan.rag_context.get('code_generation_knowledge', [])}\n"
        f"Retrieved error-experience memories (historical evidence, not binding API documentation; apply only when trigger and environment match): {state.plan.rag_context.get('long_term_memories', [])}\n"
        f"Retrieved API/rule examples for these interfaces: {summarize_codegen_rag_examples(state.plan.rag_context, executable_steps)}\n"
        f"Dynamic Agent Soul and scoped experience:\n{code_generator_soul_context(state.plan.rag_context)}\n"
        f"Traffic Relationship Graph context (canonical filters/prepares facts and proposed hypotheses):\n"
        f"{state.plan.rag_context.get('traffic_relationships', {})}\n"
        f"Structured ExecutionPlan hand-off (authoritative for step order, runtime bindings, prerequisites and collection):\n"
        f"{json.dumps(execution_plan_view(state.plan), ensure_ascii=False, sort_keys=True)}\n"
        f"ExecutionPlan validation:\n{state.plan.rag_context.get('execution_plan_validation', {})}\n"
        f"Runtime ExecutionContext evidence (real prerequisite outputs only; never invent missing values):\n"
        f"{getattr(state, 'execution_context', {})}\n"
        f"Structured relation discovery plan (hypotheses only until execution evidence exists):\n"
        f"{state.plan.rag_context.get('relation_discovery_plan', [])}\n"
        f"Executable API steps:\n{steps_for_prompt(executable_steps)}\n\n"
        "Parameter generation policy:\n"
        "- fixed: use the provided constant value in every generated call.\n"
        "- enum: call the API once for each listed safe value or value combination.\n"
        "- range: call the API at representative points: min, max, and middle samples from the PlannerAgent strategy.\n"
        "- Log the actual generated input values in focas_api_input.csv parameters for each call.\n"
        "- The documented named API prototype overrides conflicting Planner argument labels. If a required value cannot be derived safely, stop before calling the API and report the missing prerequisite; do not sweep arbitrary values to guess an undocumented range.\n"
        "- For any step.parameters.runtime_bindings, use the deterministic bound_parameters and binding evidence produced by Executor after successful upstream calls; apply include/exclude and transform rules only through the local binding tool, then enumerate the resulting values in the target call. Never replace a missing/failed source result with a literal guess; emit a prerequisite failure and skip the dependent call.\n"
        "- Returned output parameter variation, not input variation, is what RouterAgent will evaluate for quality.\n\n"
        "Common C++ interface meanings, for convenience only:\n"
        "- UploadProgram -> cnc_dwnstart3/cnc_download3/cnc_dwnend3\n"
        "- SelectProgram -> cnc_search\n"
        "- ReadProgramNumber -> cnc_rdprgnum\n"
        "- ReadProgramDirectory -> cnc_rdprogdir3\n"
        "- DeleteProgram -> cnc_delete for the exact target O number only\n"
        "- StartProgram -> NCGuide UI Cycle Start helper\n"
        "- ReadRunStatus -> cnc_statinfo\n"
        "- ReadPosition -> cnc_rdposition\n"
        "- ReadDistanceToGo -> cnc_distance\n"
        "- ReadFeedSpeed -> cnc_actf\n"
        "- ReadSpindleSpeed -> cnc_acts\n"
        "- ReadAlarm -> cnc_alarm2\n"
        "\nCycle Start gating policy:\n"
        "- Single-block Cycle Start is a stateful trigger, not a normal repeatable API read.\n"
        "- Generated C++ must gate each click with cnc_statinfo polling so the next click happens only after the previous block is complete/ready.\n"
        "- The gate must also query remaining distance, preferably cnc_distance, or cnc_rdposition(type=3)/ODBPOS.dist as fallback, because run/motion status alone can say ready while a block or program is still settling.\n"
        "- Include the literal marker cycle_start_ready_gate in the helper name, comment, CSV data, or diagnostic so the workflow can verify this behavior.\n"
        "- After the final Cycle Start, do not immediately evaluate/disconnect. Add a final program_completion_gate/WaitUntilProgramComplete loop that samples status plus distance-to-go until the NC program has completed.\n"
        "- Count the exact uploaded NC payload's non-empty executable lines. In this NCGuide environment the O-number line and M30 each require their own Cycle Start, as do modal/setup and motion lines. Drive the whole payload, not only planned motion blocks.\n"
        "- Before the click loop, require program_verified=true from successful upload completion, selection, and cnc_rdprgnum equality with the expected O number; otherwise emit PROGRAM_NOT_VERIFIED and stop.\n"
        "- The program_completion_gate output must include completed=true only after cnc_statinfo reports idle run/motion and remaining movement is zero/stable; completion timeout must be a failure, not a nonfatal warning.\n"
    )
    if code_only:
        # RQ1 follows a compact MetaGPT-style artifact contract. Keep this
        # prompt separate from the execution/data-collection contract used by
        # RQ3; appending an override to that large prompt is unreliable.
        single_agent = state.request.permissions.get("ablation_variant") == "single-agent"
        contract_owner = "shared task/RAG scaffold" if single_agent else "Planner"
        system_prompt = (
            "You are a C++ engineer generating one complete FANUC FOCAS client for a code-only benchmark.\n"
            "Return JSON only with key cpp_code. The value must be a single compilable C++17 source file containing main().\n"
            "Use the supplied official Fwlib32.h declarations exactly. Do not invent structures, fields, signatures, or symbols.\n"
            "Use FOCAS_DLL_DIR with GetEnvironmentVariableW, SetDllDirectoryW, LoadLibraryW, and GetProcAddress.\n"
            f"Connect with cnc_allclibhndl3 to the requested endpoint, execute every {contract_owner} tool call exactly once or according to its repeat, check every return code, print concise diagnostics, and release cnc_freelibhndl safely.\n"
            "For every selected API, derive its function-pointer type from the exact official declaration, construct documented arguments, and use only the official output fields. A generic loader may use std::remove_reference_t<decltype(fn)> from its destination member; it must never hard-code one API decltype for every symbol.\n"
            "For cnc_rdposition, allocate ODBPOS positions[axis_count] and put the axis index on that array. Each positions[i].abs, mach, rel, and dist member is one POSELM, and POSELM.data is one scalar long. Use positions[i].abs.data (or mach/rel/dist), never POSELM.data[i], p.data[i], or positions[i].abs.data[j].\n"
            "Do not generate packet capture, PCAP/Npcap/Wireshark, CSV files, traffic collection, simulator UI automation, runtime logging, fake results, placeholders, or local registry substitutions.\n"
            "A bounded NC payload may be included as a clearly delimited constant when required by the task, but do not upload or execute it unless the supplied API contract explicitly requires and documents that lifecycle.\n"
            "If an API cannot be implemented from the supplied ABI, return a clear generation error rather than silently replacing or skipping it.\n"
            "The source must be self-contained apart from the configured official FOCAS SDK header and Windows system headers.\n"
            "Output schema: {\"cpp_code\":\"complete source\"}."
        )
        system_prompt += f"\n\nOfficial ABI excerpts:\n{official_abi_context}\n"
        user_prompt = (
            f"Task:\n{state.request.description}\n\n"
            f"Scenario: {state.plan.scenario_type}\n"
            f"Scenario goal: {state.plan.scenario_goal}\n"
            f"FOCAS runtime directory: {default_focas_runtime_dir()}\n"
            f"Official header: {default_focas_header_dir() / 'Fwlib32.h'}\n"
            f"NC payload:\n{nc_program}\n\n"
            f"{contract_owner} atomic tool calls:\n{steps_for_prompt(executable_steps)}\n\n"
            f"Structured CodeSpec (authoritative contract):\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
            f"Structured ExecutionPlan (authoritative step order and API contract):\n{json.dumps(execution_plan_view(state.plan), ensure_ascii=False, sort_keys=True)}\n"
            f"ExecutionPlan validation:\n{state.plan.rag_context.get('execution_plan_validation', {})}\n"
            f"RAG API/rule evidence:\n{state.plan.rag_context.get('code_generation_knowledge', [])}\n"
            f"Retrieved examples:\n{summarize_codegen_rag_examples(state.plan.rag_context, executable_steps)}\n\n"
            f"Agent Soul:\n{code_generator_soul_context(state.plan.rag_context)}\n"
            f"Traffic Relationship Graph context:\n{state.plan.rag_context.get('traffic_relationships', {})}\n\n"
            f"Generate the complete source now. Preserve the exact {contract_owner} API names and documented parameters."
        )
    soul_context = code_generator_soul_context(state.plan.rag_context)
    if soul_context:
        system_prompt += (
            "\n\n# Dynamic Agent Soul\n"
            + soul_context
            + "\n\n# Relationship evidence boundary\n"
            "Use documented or runtime_verified relationships as mandatory constraints. "
            "Treat proposed and unknown relationships as hypotheses only; do not invent dependencies.\n"
        )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    cpp_code = str(payload.get("cpp_code", "")).strip()
    if not cpp_code:
        cpp_code = str(payload.get("api_script", "")).strip()
    cpp_code = normalize_focas_header_include(strip_cpp_fence(cpp_code))
    if code_only:
        # Apply deterministic ABI/framing normalization to the initial
        # artifact as well as to later repair artifacts. These are constrained
        # corrections of model spelling/formatting, not scenario content.
        cpp_code = normalize_nc_download_payload(cpp_code)
        cpp_code = normalize_official_odbpos_field_names(cpp_code)
        cpp_code = normalize_official_odbpos_axis_access(cpp_code)
        cpp_code = normalize_official_odbaxis_fields(cpp_code)
        cpp_code = normalize_cpp17_auto_parameters(cpp_code)
    return cpp_code


class CodeReviewRole(AgentTemplate):
    role_name = "CodeReviewRole"
    profile = "C++ artifact review agent"
    goal = "Check generated code against compilation, ABI, completeness, and consistency requirements."
    constraints = ("Review the generated artifact and CodeSpec.", "Do not require runtime traffic for RQ1.")

    def __init__(self) -> None:
        super().__init__()
        self.register_actions(ActionSpec("review_cpp_code", "CodeArtifact", "ReviewReport"))


def summarize_codegen_rag_examples(
    rag_context: dict[str, Any],
    executable_steps: list[PlanStep],
) -> list[dict[str, Any]]:
    wanted_functions = {
        function_name
        for step in executable_steps
        for function_name in protocol_function_names(step.protocol_function)
    }
    wanted_interfaces = {step.interface_name for step in executable_steps}
    rows: list[dict[str, Any]] = []
    for section in ["api", "operation_rule", "collection_rule", "safety_rule", "nc_rule"]:
        for item in rag_context.get(section, [])[:8]:
            if not isinstance(item, dict):
                continue
            function_name = str(item.get("function", "")).strip()
            preview = str(item.get("text_preview", "")).strip()
            scenario = str(item.get("scenario", "")).strip()
            rule_type = str(item.get("rule_type", "")).strip()
            source_type = str(item.get("source_type", section)).strip()
            if function_name and wanted_functions and function_name not in wanted_functions:
                continue
            if not function_name and section == "api" and wanted_interfaces:
                continue
            if not preview:
                continue
            rows.append(
                {
                    "section": section,
                    "source_type": source_type,
                    "rule_type": rule_type,
                    "scenario": scenario,
                    "function": function_name,
                    "source_file": item.get("source_file"),
                    "page_start": item.get("page_start"),
                    "preview": preview[:420],
                }
            )
            if len(rows) >= 10:
                return rows
    return rows


def retrieve_codegen_knowledge_context(
    knowledge_base: KnowledgeBase | None,
    state: WorkflowState,
    executable_steps: list[PlanStep],
) -> list[dict[str, Any]]:
    if knowledge_base is None or state.plan is None:
        return []
    step_text = "\n".join(
        f"{step.step_id} {step.interface_name} {step.protocol_function} {step.action} {step.parameters}"
        for step in executable_steps
    )
    repair_text = str(summarize_codegen_repair_context(state))
    query = "\n".join(
        [
            state.request.description,
            f"scenario={state.plan.scenario_type}",
            f"target_environment={state.request.target_environment}",
            f"nc_program_spec={state.plan.nc_program_spec}",
            f"executable_steps={step_text}",
            f"errors={state.errors[-8:]}",
            f"repair_context={repair_text}",
        ]
    )
    rows = (
        knowledge_base.search_api(query, top_k=8)
        + knowledge_base.search_rules(query, top_k=6)
        + knowledge_base.search(query, top_k=4)
    )
    return summarize_retrieved_for_codegen(rows)


def summarize_retrieved_for_codegen(rows) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
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
                "interface": item.chunk.metadata.get("interface"),
                "scenario": item.chunk.metadata.get("scenario") or item.chunk.metadata.get("scene"),
                "source_file": item.chunk.metadata.get("source_file"),
                "page_start": item.chunk.metadata.get("page_start"),
                "preview": item.chunk.text[:520],
            }
        )
        if len(summary) >= 12:
            break
    return summary


def protocol_function_names(protocol_function: str) -> list[str]:
    import re

    names: list[str] = []
    for part in re.split(r"[,/;+]+", protocol_function):
        name = part.strip()
        if name and (name.startswith("cnc_") or name.startswith("pmc_")):
            names.append(name)
    return names


def official_focas_abi_context(function_names: list[str]) -> str:
    """Extract exact prototypes and directly referenced struct layouts from the configured FANUC header."""

    import re

    header_path = default_focas_header_dir() / "Fwlib32.h"
    if not header_path.exists():
        return f"Official header not found: {header_path}"
    header = header_path.read_text(encoding="latin-1")
    typedef_blocks = extract_focas_typedef_blocks(header)
    prototypes: list[str] = []
    referenced_types: set[str] = set()
    for function_name in function_names:
        match = re.search(
            rf"FWLIBAPI\s+[^;]*?\b{re.escape(function_name)}\s*\([^;]*?\)\s*;",
            header,
            re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            continue
        prototype = " ".join(match.group(0).split())
        prototypes.append(prototype)
        referenced_types.update(
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", prototype)
            if token not in {"FWLIBAPI", "WINAPI"}
        )

    struct_definitions: list[str] = []
    pending = sorted(referenced_types)
    seen_types: set[str] = set()
    while pending and len(struct_definitions) < 24:
        type_name = pending.pop(0)
        if type_name in seen_types:
            continue
        seen_types.add(type_name)
        definition = typedef_blocks.get(type_name.upper())
        if definition is None:
            continue
        struct_definitions.append(definition)
        nested = {
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", definition)
            if token not in seen_types and token not in {"MAX_AXIS", "MAX_SPINDLE"}
        }
        pending.extend(sorted(nested))

    sections = [f"header={header_path}"]
    if prototypes:
        sections.append("[official function declarations]\n" + "\n".join(prototypes))
    named_context = named_contract_context(function_names)
    if named_context:
        sections.append(
            "[documented named parameters and API-specific error meanings; "
            "Fwlib32.h remains authoritative for types]\n" + named_context
        )
    if struct_definitions:
        sections.append("[official referenced type definitions]\n" + "\n\n".join(struct_definitions))
    return "\n\n".join(sections)[:24000]


def extract_focas_typedef_blocks(header: str) -> dict[str, str]:
    """Parse balanced typedef struct/union blocks from the official C header."""

    import re

    blocks: dict[str, str] = {}
    start_pattern = re.compile(r"typedef\s+(?:struct|union)\s+\w+\s*\{", re.IGNORECASE)
    for start_match in start_pattern.finditer(header):
        open_brace = header.find("{", start_match.start())
        if open_brace < 0:
            continue
        depth = 0
        close_brace = -1
        for index in range(open_brace, len(header)):
            char = header[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    close_brace = index
                    break
        if close_brace < 0:
            continue
        alias_match = re.match(r"\s*([A-Za-z_]\w*)\s*;", header[close_brace + 1 :])
        if alias_match is None:
            continue
        alias = alias_match.group(1)
        end = close_brace + 1 + alias_match.end()
        definition = "\n".join(line.rstrip() for line in header[start_match.start() : end].splitlines())
        # The header contains conditional alternate definitions for some
        # aliases, including ODBPOS. Keep the first definition selected by
        # the default compiler configuration.
        blocks.setdefault(alias.upper(), definition)
    return blocks


def cpp_generation_scaffold() -> str:
    return r'''
```cpp
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <Fwlib32.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

using std::string;

using short_t = short;
using ushort_t = unsigned short;

struct FocasApi {
    HMODULE dll = nullptr;
    ushort_t handle = 0;

    // Derive ABI types from the official FANUC header; resolve only required functions.
    decltype(&::cnc_allclibhndl3) cnc_allclibhndl3_fn = nullptr;
    decltype(&::cnc_freelibhndl) cnc_freelibhndl_fn = nullptr;
    decltype(&::cnc_dwnstart3) cnc_dwnstart3_fn = nullptr;
    decltype(&::cnc_download3) cnc_download3_fn = nullptr;
    decltype(&::cnc_dwnend3) cnc_dwnend3_fn = nullptr;
    decltype(&::cnc_search) cnc_search_fn = nullptr;
    decltype(&::cnc_rdprgnum) cnc_rdprgnum_fn = nullptr;
    decltype(&::cnc_statinfo) cnc_statinfo_fn = nullptr;
    decltype(&::cnc_rdposition) cnc_rdposition_fn = nullptr;
    decltype(&::cnc_actf) cnc_actf_fn = nullptr;
    decltype(&::cnc_acts) cnc_acts_fn = nullptr;
    decltype(&::cnc_alarm2) cnc_alarm2_fn = nullptr;
};

static void WriteBom(std::ofstream& file) {
    const unsigned char bom[] = {0xEF, 0xBB, 0xBF};
    file.write(reinterpret_cast<const char*>(bom), sizeof(bom));
}

static string CsvEscape(const string& value) {
    string out = "\"";
    for (char c : value) out += (c == '"') ? "\"\"" : string(1, c);
    out += "\"";
    return out;
}

static string Timestamp() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buffer[64];
    sprintf_s(buffer, "%04d-%02d-%02dT%02d:%02d:%02d.%03d",
              st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return string(buffer);
}

static void LogInput(std::ofstream& csv, int index, const string& stepId, const string& phase,
                     const string& interfaceName, const string& protocolFunction,
                     const string& parameters, const string& timestamp = Timestamp()) {
    csv << index << "," << CsvEscape(timestamp) << "," << CsvEscape(stepId) << ","
        << CsvEscape(interfaceName) << "," << CsvEscape(protocolFunction) << ","
        << CsvEscape(parameters) << ",0\n";
}

static void LogOutput(std::ofstream& csv, int index, const string& stepId,
                      const string& apiName, short_t ret,
                      const string& returnText, const string& data = "", const string& timestamp = Timestamp()) {
    csv << index << "," << CsvEscape(timestamp) << "," << CsvEscape(stepId) << "," << CsvEscape(apiName) << ","
        << ret << "," << CsvEscape(returnText) << "," << CsvEscape(data) << "\n";
}

template <typename T>
static bool Resolve(HMODULE dll, const char* name, T& out) {
    out = reinterpret_cast<T>(GetProcAddress(dll, name));
    return out != nullptr;
}

static int ProgramNumberFromName(const string& programName) {
    if (programName.size() > 1 && (programName[0] == 'O' || programName[0] == 'o')) {
        return std::stoi(programName.substr(1));
    }
    return std::stoi(programName);
}

static bool LoadFocas(FocasApi& api, const wchar_t* dllPath) {
    api.dll = LoadLibraryW(dllPath);
    if (!api.dll) {
        return false;
    }
    bool ok = true;
    ok &= Resolve(api.dll, "cnc_allclibhndl3", api.cnc_allclibhndl3_fn);
    ok &= Resolve(api.dll, "cnc_freelibhndl", api.cnc_freelibhndl_fn);
    // Resolve additional functions required by selected executable steps here.
    return ok;
}

// If UploadProgram is selected, implement a real function here using:
// cnc_dwnstart3 -> cnc_download3 chunks -> cnc_dwnend3.
// Do not return success without calling those FOCAS functions.

static short_t SelectProgram(FocasApi& api, const string& programName) {
    return api.cnc_search_fn ? api.cnc_search_fn(api.handle, static_cast<short_t>(ProgramNumberFromName(programName))) : -1;
}

static void TriggerNcGuideCycleStart(int x, int y) {
    SetCursorPos(x, y);
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
}

int main() {
    std::ofstream inputCsv("data\\focas_api_input.csv", std::ios::binary);
    std::ofstream outputCsv("data\\focas_api_output.csv", std::ios::binary);
    WriteBom(inputCsv);
    WriteBom(outputCsv);
    inputCsv << "index,timestamp,step_id,interface_name,protocol_function,parameters,api_parameter_count\n";
    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n";
    int csvIndex = 1;

    FocasApi api;
    if (!LoadFocas(api, L"Fwlib32.dll")) {
        LogInput(inputCsv, csvIndex, "LOAD_DLL", "before", "FOCAS", "LoadLibraryW", "dll=Fwlib32.dll");
        LogOutput(outputCsv, csvIndex, "LOAD_DLL", "LoadLibraryW", -1, "LoadLibraryW failed", "");
        return 1;
    }

    char host[256] = "127.0.0.1";
    char portText[16] = "8193";
    GetEnvironmentVariableA("FOCAS_HOST", host, sizeof(host));
    GetEnvironmentVariableA("FOCAS_PORT", portText, sizeof(portText));
    short_t ret = api.cnc_allclibhndl3_fn
        ? api.cnc_allclibhndl3_fn(host, static_cast<unsigned short>(std::stoi(portText)), 10, &api.handle)
        : -1;
    LogInput(inputCsv, csvIndex, "CONNECT", "before", "FOCAS", "cnc_allclibhndl3", std::string("host=") + host + ";port=" + portText + ";timeout=10");
    LogOutput(outputCsv, csvIndex, "CONNECT", "cnc_allclibhndl3", ret, ret == 0 ? "EW_OK" : "connect failed", "cnc_allclibhndl3");
    ++csvIndex;
    if (ret != 0) {
        if (api.dll) FreeLibrary(api.dll);
        return 1;
    }

    // Emit one block per executable API step and repeat.
    // Each block must call LogInput before the FOCAS/UI operation using csvIndex,
    // call the real FOCAS/UI function or NCGuide UI action,
    // then call LogOutput with the same csvIndex and parsed response fields,
    // then increment csvIndex.
    // Do not leave placeholder success responses.

    if (api.cnc_freelibhndl_fn) api.cnc_freelibhndl_fn(api.handle);
    if (api.dll) FreeLibrary(api.dll);
    return 0;
}
```
'''.strip()


def strip_cpp_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_focas_header_include(api_script: str) -> str:
    """Force generated C++ to include FOCAS via the configured include path.

    The MSVC build already adds the controller-specific directory containing
    Fwlib32.h. Generated source must therefore include only the basename, not a
    hardcoded local path such as Fwlib/0iD/Fwlib32.h.
    """
    import re

    return re.sub(
        r"#\s*include\s*[<\"][^>\"\n]*Fwlib32\.h[>\"]",
        "#include <Fwlib32.h>",
        api_script,
        flags=re.IGNORECASE,
    )


def render_nc_program(scenario: str) -> str:
    if scenario == "comprehensive_focas_traffic":
        return "\n".join(
            [
                "O8001",
                "G90 G54",
                "M03 S800",
                "G01 X0 Y0 Z5 F300",
                "G01 X10 Y0 Z5 F300",
                "S1200",
                "G01 X10 Y10 Z3 F240",
                "G01 X0 Y10 Z2 F240",
                "M05",
                "G01 X0 Y0 Z5 F300",
                "M30",
                "",
            ]
        )
    if scenario == "coordinate_motion":
        return "\n".join(
            [
                "O1000",
                "G90 G54",
                "G00 X0 Y0 Z5",
                "G01 X20 Y0 Z5 F300",
                "G01 X20 Y20 Z4 F240",
                "G01 X0 Y20 Z4 F240",
                "G01 X0 Y0 Z5 F300",
                "M30",
                "",
            ]
        )
    if scenario in {"spindle_state", "spindle_speed_change"}:
        return "\n".join(["O2001", "G90 G54", "M03 S800", "G04 P1", "S1200", "G04 P1", "M05", "M30", ""])
    if scenario == "program_lifecycle":
        return "\n".join(["O3001", "G90 G54", "G01 X1 Y1 F120", "G04 P1", "M30", ""])
    return "\n".join(["O9001", "G90 G54", "G04 P1", "M30", ""])


def render_api_script(steps) -> str:
    return render_cpp_api_script_v2("generated", steps)


def windows_cpp_wide_string(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def cpp_string_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def render_cpp_api_script_v2(scenario: str, steps, nc_program: str | None = None) -> str:
    planned_functions = sorted({step.protocol_function for step in steps if step.protocol_function})
    runtime_dir = windows_cpp_wide_string(default_focas_runtime_dir())
    dll_path = windows_cpp_wide_string(default_focas_runtime_dir() / "Fwlib32.dll")
    nc_payload = "\n" + (nc_program or render_nc_program(scenario)).strip().replace("\r\n", "\n").replace("\r", "\n") + "\n%"
    lines = [
        f"// Auto-generated FOCAS C++ API script for scenario: {scenario}",
        "// This file is the API script: it connects to FANUC FOCAS and calls API functions with planned parameters.",
        "// Compile example:",
        "// cl /utf-8 /EHsc /std:c++17 api_script.cpp",
        "",
        "#define WIN32_LEAN_AND_MEAN",
        "#include <winsock2.h>",
        "#include <windows.h>",
        "#include <Fwlib32.h>",
        "#include <iostream>",
        "#include <fstream>",
        "#include <sstream>",
        "#include <string>",
        "#include <ctime>",
        "#include <iomanip>",
        "#include <thread>",
        "#include <chrono>",
        "#include <vector>",
        "#include <cctype>",
        "#include <cstring>",
        "#include <cstdlib>",
        "",
        "using namespace std;",
        "",
        "using cnc_allclibhndl3_t = decltype(&::cnc_allclibhndl3);",
        "using cnc_freelibhndl_t = decltype(&::cnc_freelibhndl);",
        "using cnc_dwnstart3_t = decltype(&::cnc_dwnstart3);",
        "using cnc_download3_t = decltype(&::cnc_download3);",
        "using cnc_dwnend3_t = decltype(&::cnc_dwnend3);",
        "using cnc_search_t = decltype(&::cnc_search);",
        "using cnc_rdprgnum_t = decltype(&::cnc_rdprgnum);",
        "using cnc_statinfo_t = decltype(&::cnc_statinfo);",
        "using cnc_actf_t = decltype(&::cnc_actf);",
        "using cnc_acts_t = decltype(&::cnc_acts);",
        "using cnc_rdposition_t = decltype(&::cnc_rdposition);",
        "using cnc_distance_t = decltype(&::cnc_distance);",
        "using cnc_alarm2_t = decltype(&::cnc_alarm2);",
        "",
        "using u_char = unsigned char;",
        "using bpf_u_int32 = unsigned int;",
        "struct pcap;",
        "struct pcap_dumper;",
        "struct pcap_addr;",
        "struct pcap_if { pcap_if* next; char* name; char* description; pcap_addr* addresses; unsigned int flags; };",
        "struct pcap_pkthdr { timeval ts; bpf_u_int32 caplen; bpf_u_int32 len; };",
        "using pcap_t = pcap;",
        "using pcap_dumper_t = pcap_dumper;",
        "using pcap_if_t = pcap_if;",
        "using pcap_handler = void (*)(u_char*, const pcap_pkthdr*, const u_char*);",
        "using pcap_findalldevs_t = int(__cdecl *)(pcap_if_t**, char*);",
        "using pcap_freealldevs_t = void(__cdecl *)(pcap_if_t*);",
        "using pcap_open_live_t = pcap_t*(__cdecl *)(const char*, int, int, int, char*);",
        "using pcap_dump_open_t = pcap_dumper_t*(__cdecl *)(pcap_t*, const char*);",
        "using pcap_dispatch_t = int(__cdecl *)(pcap_t*, int, pcap_handler, u_char*);",
        "using pcap_dump_t = void(__cdecl *)(u_char*, const pcap_pkthdr*, const u_char*);",
        "using pcap_dump_flush_t = int(__cdecl *)(pcap_dumper_t*);",
        "using pcap_dump_close_t = void(__cdecl *)(pcap_dumper_t*);",
        "using pcap_close_t = void(__cdecl *)(pcap_t*);",
        "using pcap_geterr_t = char*(__cdecl *)(pcap_t*);",
        "",
        "string GetTimestamp() {",
        "    time_t now = time(nullptr);",
        "    tm ltm;",
        "    localtime_s(&ltm, &now);",
        "    stringstream ss;",
        '    ss << put_time(&ltm, "%Y-%m-%dT%H:%M:%S");',
        "    return ss.str();",
        "}",
        "",
        "string EscapeCsvField(const string& field) {",
        "    if (field.find(',') == string::npos && field.find('\"') == string::npos && field.find('\\n') == string::npos) return field;",
        "    string escaped = \"\\\"\";",
        "    for (char c : field) escaped += (c == '\"') ? \"\\\"\\\"\" : string(1, c);",
        "    escaped += \"\\\"\";",
        "    return escaped;",
        "}",
        "",
        "string ReturnText(short code) {",
        "    switch (code) {",
        '    case 0: return "EW_OK";',
        '    case 1: return "EW_FUNC";',
        '    case 2: return "EW_LENGTH";',
        '    case 3: return "EW_NUMBER";',
        '    case 4: return "EW_ATTRIB";',
        '    case 5: return "EW_DATA";',
        '    case 6: return "EW_NOOPT";',
        '    case 7: return "EW_PROT";',
        '    case 8: return "EW_OVRFLOW";',
        '    case 9: return "EW_PARAM";',
        '    case 10: return "EW_BUFFER";',
        '    case 11: return "EW_PATH";',
        '    case 12: return "EW_MODE";',
        '    case 13: return "EW_REJECT";',
        '    case 14: return "EW_DTSRVR";',
        '    case 15: return "EW_ALARM";',
        '    case 16: return "EW_STOP";',
        '    case 17: return "EW_PASSWD";',
        '    default: return "EW_" + to_string(code);',
        "    }",
        "}",
        "",
        "void WriteInput(ofstream& csv, int index, const string& stepId, const string& phase, const string& api, const string& params) {",
        '    csv << index << "," << GetTimestamp() << "," << stepId << "," << phase << "," << api << "," << EscapeCsvField(params) << "\\n";',
        "}",
        "",
        "void WriteOutput(ofstream& csv, int index, const string& stepId, const string& api, short ret, const string& data) {",
        '    csv << index << "," << GetTimestamp() << "," << stepId << "," << api << "," << ret << "," << ReturnText(ret) << "," << EscapeCsvField(data) << "\\n";',
        "}",
        "",
        "int ProgramNumberFromName(const string& programName) {",
        "    string digits;",
        "    for (char c : programName) if (isdigit(static_cast<unsigned char>(c))) digits += c;",
        "    return digits.empty() ? 0 : stoi(digits);",
        "}",
        "",
        "string EnvString(const char* name, const string& defaultValue = \"\") {",
        "    char buffer[512] = {};",
        "    DWORD size = GetEnvironmentVariableA(name, buffer, static_cast<DWORD>(sizeof(buffer)));",
        "    if (size == 0 || size >= sizeof(buffer)) return defaultValue;",
        "    return string(buffer, size);",
        "}",
        "",
        "int EnvInt(const char* name, int defaultValue = 0) {",
        "    string value = EnvString(name);",
        "    if (value.empty()) return defaultValue;",
        "    try { return stoi(value); } catch (...) { return defaultValue; }",
        "}",
        "",
        "bool EnvFlag(const char* name) {",
        "    string value = EnvString(name);",
        "    for (char& c : value) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));",
        "    return value == \"1\" || value == \"true\" || value == \"yes\" || value == \"on\";",
        "}",
        "",
        "bool ClickClientPoint(HWND hwnd, int x, int y) {",
        "    if (!hwnd || x <= 0 || y <= 0) return false;",
        "    POINT pt{x, y};",
        "    if (!ClientToScreen(hwnd, &pt)) return false;",
        "    SetForegroundWindow(hwnd);",
        "    Sleep(120);",
        "    SetCursorPos(pt.x, pt.y);",
        "    INPUT inputs[2]{};",
        "    inputs[0].type = INPUT_MOUSE;",
        "    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;",
        "    inputs[1].type = INPUT_MOUSE;",
        "    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;",
        "    UINT sent = SendInput(2, inputs, sizeof(INPUT));",
        "    Sleep(120);",
        "    return sent == 2;",
        "}",
        "",
        "bool ClickScreenPoint(int x, int y) {",
        "    if (x <= 0 || y <= 0) return false;",
        "    SetCursorPos(x, y);",
        "    Sleep(120);",
        "    INPUT inputs[2]{};",
        "    inputs[0].type = INPUT_MOUSE;",
        "    inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;",
        "    inputs[1].type = INPUT_MOUSE;",
        "    inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;",
        "    UINT sent = SendInput(2, inputs, sizeof(INPUT));",
        "    Sleep(120);",
        "    return sent == 2;",
        "}",
        "",
        "struct ChildButtonSearch {",
        "    string targetText;",
        "    HWND found = nullptr;",
        "};",
        "",
        "BOOL CALLBACK FindChildButtonByTextProc(HWND child, LPARAM lParam) {",
        "    auto* search = reinterpret_cast<ChildButtonSearch*>(lParam);",
        "    char text[256] = {};",
        "    char className[256] = {};",
        "    GetWindowTextA(child, text, sizeof(text));",
        "    GetClassNameA(child, className, sizeof(className));",
        "    string childText = text;",
        "    string childClass = className;",
        "    if (childText == search->targetText && childClass.find(\"BUTTON\") != string::npos) {",
        "        search->found = child;",
        "        return FALSE;",
        "    }",
        "    return TRUE;",
        "}",
        "",
        "HWND FindChildButtonByText(HWND parent, const string& buttonText) {",
        "    if (!parent || buttonText.empty()) return nullptr;",
        "    ChildButtonSearch search{buttonText, nullptr};",
        "    EnumChildWindows(parent, FindChildButtonByTextProc, reinterpret_cast<LPARAM>(&search));",
        "    return search.found;",
        "}",
        "",
        "bool ClickChildButtonByText(HWND parent, const string& buttonText) {",
        "    HWND button = FindChildButtonByText(parent, buttonText);",
        "    if (!button) return false;",
        "    SetForegroundWindow(parent);",
        "    Sleep(120);",
        "    LRESULT result = SendMessageA(button, BM_CLICK, 0, 0);",
        "    Sleep(300);",
        "    return result == 0;",
        "}",
        "",
        "short TriggerNcGuideCycleStart(const string& requestedTitle, string& data) {",
        "    if (!EnvFlag(\"NCGUIDE_ENABLE_UI_START\")) {",
        "        data = \"enabled=0;triggered=0;reason=NCGUIDE_ENABLE_UI_START_not_set\";",
        "        return 0;",
        "    }",
        "    string title = EnvString(\"NCGUIDE_WINDOW_TITLE\", requestedTitle.empty() ? \"FANUC CNC GUIDE\" : requestedTitle);",
        "    HWND hwnd = FindWindowA(nullptr, title.c_str());",
        "    if (!hwnd) {",
        "        data = \"enabled=1;triggered=0;reason=window_not_found;window_title=\" + title;",
        "        return 14;",
        "    }",
        "    string buttonText = EnvString(\"NCGUIDE_START_BUTTON_TEXT\", \"\");",
        "    if (!buttonText.empty()) {",
        "        bool buttonClicked = ClickChildButtonByText(hwnd, buttonText);",
        "        if (buttonClicked) {",
        "            data = \"enabled=1;triggered=button_click;window_title=\" + title + \";button_text=\" + buttonText;",
        "            return 0;",
        "        }",
        "    }",
        "    int modeX = EnvInt(\"NCGUIDE_MODE_X\", 0);",
        "    int modeY = EnvInt(\"NCGUIDE_MODE_Y\", 0);",
        "    int startX = EnvInt(\"NCGUIDE_CYCLE_START_X\", 0);",
        "    int startY = EnvInt(\"NCGUIDE_CYCLE_START_Y\", 0);",
        "    if (startX <= 0 || startY <= 0) {",
        "        int manualWaitSeconds = EnvInt(\"NCGUIDE_MANUAL_START_WAIT_SECONDS\", 0);",
        "        if (manualWaitSeconds > 0) {",
        "            cout << \"Manual Cycle Start window: press Cycle Start within \" << manualWaitSeconds << \" seconds...\" << endl;",
        "            Sleep(static_cast<DWORD>(manualWaitSeconds) * 1000);",
        "            data = \"enabled=1;triggered=manual_wait;wait_seconds=\" + to_string(manualWaitSeconds) + \";window_title=\" + title;",
        "            return 0;",
        "        }",
        "        data = \"enabled=1;triggered=0;reason=cycle_start_coordinates_missing;window_title=\" + title;",
        "        return 14;",
        "    }",
        "    bool modeClicked = true;",
        "    string clickMode = EnvString(\"NCGUIDE_CLICK_MODE\", \"client\");",
        "    if (modeX > 0 && modeY > 0) {",
        "        modeClicked = (clickMode == \"screen\") ? ClickScreenPoint(modeX, modeY) : ClickClientPoint(hwnd, modeX, modeY);",
        "        Sleep(300);",
        "    }",
        "    bool startClicked = (clickMode == \"screen\") ? ClickScreenPoint(startX, startY) : ClickClientPoint(hwnd, startX, startY);",
        "    data = \"enabled=1;triggered=\" + string(startClicked ? \"1\" : \"0\") + \";mode_clicked=\" + string(modeClicked ? \"1\" : \"0\") + \";click_mode=\" + clickMode + \";window_title=\" + title + \";cycle_start_x=\" + to_string(startX) + \";cycle_start_y=\" + to_string(startY);",
        "    return (modeClicked && startClicked) ? 0 : 14;",
        "}",
        "",
        "class PacketSniffer {",
        "public:",
        "    bool Start(const string& outputPath) {",
        '        dll_ = LoadLibraryW(L"wpcap.dll");',
        "        if (!dll_) { cout << \"PCAP_CAPTURE_FAILED: Npcap wpcap.dll not found.\" << endl; return false; }",
        '        findalldevs_ = reinterpret_cast<pcap_findalldevs_t>(GetProcAddress(dll_, "pcap_findalldevs"));',
        '        freealldevs_ = reinterpret_cast<pcap_freealldevs_t>(GetProcAddress(dll_, "pcap_freealldevs"));',
        '        open_live_ = reinterpret_cast<pcap_open_live_t>(GetProcAddress(dll_, "pcap_open_live"));',
        '        dump_open_ = reinterpret_cast<pcap_dump_open_t>(GetProcAddress(dll_, "pcap_dump_open"));',
        '        dispatch_ = reinterpret_cast<pcap_dispatch_t>(GetProcAddress(dll_, "pcap_dispatch"));',
        '        dump_ = reinterpret_cast<pcap_dump_t>(GetProcAddress(dll_, "pcap_dump"));',
        '        dump_flush_ = reinterpret_cast<pcap_dump_flush_t>(GetProcAddress(dll_, "pcap_dump_flush"));',
        '        dump_close_ = reinterpret_cast<pcap_dump_close_t>(GetProcAddress(dll_, "pcap_dump_close"));',
        '        close_ = reinterpret_cast<pcap_close_t>(GetProcAddress(dll_, "pcap_close"));',
        '        geterr_ = reinterpret_cast<pcap_geterr_t>(GetProcAddress(dll_, "pcap_geterr"));',
        "        if (!findalldevs_ || !freealldevs_ || !open_live_ || !dump_open_ || !dispatch_ || !dump_ || !dump_flush_ || !dump_close_ || !close_) {",
        "            cout << \"PCAP_CAPTURE_FAILED: Npcap exports are incomplete.\" << endl;",
        "            return false;",
        "        }",
        "        char errbuf[512] = {};",
        "        pcap_if_t* devices = nullptr;",
        "        if (findalldevs_(&devices, errbuf) != 0 || !devices) { cout << \"pcap_findalldevs failed: \" << errbuf << endl; return false; }",
        "        string selected = EnvString(\"PCAP_NETWORK_DEVICE\", \"\\\\\\\\Device\\\\\\\\NPF_Loopback\");",
        "        bool foundSelected = false;",
        "        for (pcap_if_t* dev = devices; dev; dev = dev->next) {",
        "            string name = dev->name ? dev->name : \"\";",
        "            if (name == selected) { foundSelected = true; break; }",
        "        }",
        "        freealldevs_(devices);",
        "        if (selected.empty() || !foundSelected) { cout << \"PCAP_CAPTURE_FAILED: configured Npcap adapter not found: \" << selected << endl; return false; }",
        "        handle_ = open_live_(selected.c_str(), 65536, 1, 100, errbuf);",
        "        if (!handle_) { cout << \"pcap_open_live failed: \" << errbuf << endl; return false; }",
        "        dumper_ = dump_open_(handle_, outputPath.c_str());",
        "        if (!dumper_) { cout << \"pcap_dump_open failed\" << endl; close_(handle_); handle_ = nullptr; return false; }",
        "        cout << \"pcap_capture_adapter=\" << selected << endl;",
        "        enabled_ = true;",
        "        return true;",
        "    }",
        "    void Capture(int packetCount = 32) {",
        "        if (!enabled_ || !handle_ || !dumper_) return;",
        "        activeDump_ = dump_;",
        "        dispatch_(handle_, packetCount, &PacketCapture::DumpCallback, reinterpret_cast<u_char*>(dumper_));",
        "        if (dump_flush_) dump_flush_(dumper_);",
        "    }",
        "    void Stop() {",
        "        if (dumper_ && dump_close_) { dump_flush_(dumper_); dump_close_(dumper_); dumper_ = nullptr; }",
        "        if (handle_ && close_) { close_(handle_); handle_ = nullptr; }",
        "        if (dll_) { FreeLibrary(dll_); dll_ = nullptr; }",
        "        enabled_ = false;",
        "    }",
        "private:",
        "    static void DumpCallback(u_char* user, const pcap_pkthdr* header, const u_char* packet) {",
        "        if (activeDump_) activeDump_(user, header, packet);",
        "    }",
        "    inline static pcap_dump_t activeDump_ = nullptr;",
        "    HMODULE dll_ = nullptr;",
        "    pcap_t* handle_ = nullptr;",
        "    pcap_dumper_t* dumper_ = nullptr;",
        "    bool enabled_ = false;",
        "    pcap_findalldevs_t findalldevs_ = nullptr;",
        "    pcap_freealldevs_t freealldevs_ = nullptr;",
        "    pcap_open_live_t open_live_ = nullptr;",
        "    pcap_dump_open_t dump_open_ = nullptr;",
        "    pcap_dispatch_t dispatch_ = nullptr;",
        "    pcap_dump_t dump_ = nullptr;",
        "    pcap_dump_flush_t dump_flush_ = nullptr;",
        "    pcap_dump_close_t dump_close_ = nullptr;",
        "    pcap_close_t close_ = nullptr;",
        "    pcap_geterr_t geterr_ = nullptr;",
        "};",
        "",
        "int main() {",
        "    SetConsoleOutputCP(CP_UTF8);",
        f'    SetDllDirectoryW(L"{runtime_dir}");',
        f'    HMODULE dll = LoadLibraryW(L"{dll_path}");',
        '    if (!dll) { cerr << "LoadLibraryW(Fwlib32.dll) failed" << endl; return 1; }',
        '    auto cnc_allclibhndl3_fn = reinterpret_cast<cnc_allclibhndl3_t>(GetProcAddress(dll, "cnc_allclibhndl3"));',
        '    auto cnc_freelibhndl_fn = reinterpret_cast<cnc_freelibhndl_t>(GetProcAddress(dll, "cnc_freelibhndl"));',
        '    auto cnc_dwnstart3_fn = reinterpret_cast<cnc_dwnstart3_t>(GetProcAddress(dll, "cnc_dwnstart3"));',
        '    auto cnc_download3_fn = reinterpret_cast<cnc_download3_t>(GetProcAddress(dll, "cnc_download3"));',
        '    auto cnc_dwnend3_fn = reinterpret_cast<cnc_dwnend3_t>(GetProcAddress(dll, "cnc_dwnend3"));',
        '    auto cnc_search_fn = reinterpret_cast<cnc_search_t>(GetProcAddress(dll, "cnc_search"));',
        '    auto cnc_rdprgnum_fn = reinterpret_cast<cnc_rdprgnum_t>(GetProcAddress(dll, "cnc_rdprgnum"));',
        '    auto cnc_statinfo_fn = reinterpret_cast<cnc_statinfo_t>(GetProcAddress(dll, "cnc_statinfo"));',
        '    auto cnc_actf_fn = reinterpret_cast<cnc_actf_t>(GetProcAddress(dll, "cnc_actf"));',
        '    auto cnc_acts_fn = reinterpret_cast<cnc_acts_t>(GetProcAddress(dll, "cnc_acts"));',
        '    auto cnc_rdposition_fn = reinterpret_cast<cnc_rdposition_t>(GetProcAddress(dll, "cnc_rdposition"));',
        '    auto cnc_distance_fn = reinterpret_cast<cnc_distance_t>(GetProcAddress(dll, "cnc_distance"));',
        '    auto cnc_alarm2_fn = reinterpret_cast<cnc_alarm2_t>(GetProcAddress(dll, "cnc_alarm2"));',
        '    if (!cnc_allclibhndl3_fn || !cnc_freelibhndl_fn || !cnc_dwnstart3_fn || !cnc_download3_fn || !cnc_dwnend3_fn || !cnc_search_fn || !cnc_rdprgnum_fn || !cnc_statinfo_fn || !cnc_actf_fn || !cnc_acts_fn || !cnc_rdposition_fn || !cnc_distance_fn || !cnc_alarm2_fn) {',
        '        cerr << "Required FOCAS API export is missing from Fwlib32.dll" << endl;',
        "        return 1;",
        "    }",
        '    string cncHost = EnvString("FOCAS_HOST", "127.0.0.1");',
        '    const char* cncIp = cncHost.c_str();',
        '    unsigned short port = static_cast<unsigned short>(EnvInt("FOCAS_PORT", 8193));',
        "    unsigned short handle = 0;",
        "    short ret = cnc_allclibhndl3_fn(cncIp, port, 30, &handle);",
        '    cout << "cnc_allclibhndl3 ret=" << ret << " (" << ReturnText(ret) << ")" << endl;',
        "    if (ret != 0) return 1;",
        "",
        '    CreateDirectoryA("data", NULL);',
        '    PacketSniffer sniffer;',
        '    bool pcapEnabled = sniffer.Start("data\\\\focas_api_capture.pcap");',
        '    ofstream inputCsv("data\\\\focas_api_input.csv", ios::binary);',
        '    ofstream outputCsv("data\\\\focas_api_output.csv", ios::binary);',
            "    inputCsv.put(0xEF); inputCsv.put(0xBB); inputCsv.put(0xBF);",
            "    outputCsv.put(0xEF); outputCsv.put(0xBB); outputCsv.put(0xBF);",
        '    inputCsv << "index,timestamp,step_id,phase,api_name,parameters,api_parameter_count\\n";',
        '    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\\n";',
        "    int index = 0;",
        '    if (!pcapEnabled) { outputCsv << "1," << GetTimestamp() << ",PCAP_CAPTURE,PacketSniffer,-1,PCAP_CAPTURE_FAILED," << EscapeCsvField("pcap_capture_enabled=false") << "\\\\n"; cnc_freelibhndl_fn(handle); FreeLibrary(dll); return 1; }',
        '    outputCsv << "1," << GetTimestamp() << ",PCAP_CAPTURE,PacketSniffer,0,EW_OK," << EscapeCsvField("pcap_capture_enabled=true;pcap_file=data\\\\\\\\focas_api_capture.pcap") << "\\\\n";',
        f'    const string ncProgramPayload = "{cpp_string_literal(nc_payload)}";',
        "",
        f'    cout << "planned_functions={",".join(planned_functions)}" << endl;',
    ]
    for step in steps:
        lines.extend(cpp_step_block(step))
    lines.extend(
        [
            "",
            "    cnc_freelibhndl_fn(handle);",
            "    packetCapture.Stop();",
            "    FreeLibrary(dll);",
            '    cout << "FOCAS API script finished. CSV files are under .\\\\data" << endl;',
            "    return 0;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def compact_log_params(step) -> str:
    keep_by_interface = {
        "UploadProgram": ["program_name"],
        "SelectProgram": ["program_name"],
        "ReadProgramNumber": [],
        "StartProgram": [],
        "ReadRunStatus": [],
        "ReadPosition": [],
        "ReadDistanceToGo": [],
        "ReadFeedSpeed": [],
        "ReadSpindleSpeed": [],
        "ReadAlarm": [],
    }
    keys = keep_by_interface.get(step.interface_name, ["block_index", "nc_block"])
    pairs = []
    for key in keys:
        value = step.parameters.get(key)
        if value is None or value == "":
            continue
        pairs.append(f"{key}={value}")
    return ";".join(pairs) or "default"


def cpp_step_block(step) -> list[str]:
    blocks: list[str] = [
        f"    // {step.step_id}: {step.phase} - {step.action}",
        f"    for (int repeatIndex = 0; repeatIndex < {int(step.repeat)}; ++repeatIndex) {{",
        "        ++index;",
    ]
    api = step.protocol_function or step.interface_name
    params = compact_log_params(step)
    if step.interface_name == "UploadProgram":
        program_name = str(step.parameters.get("program_name", "O0"))
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_dwnstart3/cnc_download3/cnc_dwnend3", "{params}");',
                f'        short programNo = static_cast<short>(ProgramNumberFromName("{program_name}"));',
                "        packetCapture.Capture();",
                "        short preSearchRet = cnc_search_fn(handle, programNo);",
                "        packetCapture.Capture();",
                "        short startRet = 0;",
                "        short callRet = 0;",
                "        long totalBytes = 0;",
                "        long transferredBytes = 0;",
                "        bool existingProgram = (preSearchRet == 0);",
                "        if (!existingProgram) {",
                "            packetCapture.Capture();",
                "            startRet = cnc_dwnstart3_fn(handle, 0);",
                "            packetCapture.Capture();",
                "            callRet = startRet;",
                "        }",
                "        if (!existingProgram && startRet == 0) {",
                "            string remaining = ncProgramPayload;",
                "            char* cursor = remaining.data();",
                "            long remainingBytes = static_cast<long>(remaining.size());",
                "            totalBytes = remainingBytes;",
                "            while (remainingBytes > 0) {",
                "                long chunkBytes = remainingBytes;",
                "                packetCapture.Capture();",
                "                short downloadRet = cnc_download3_fn(handle, &chunkBytes, cursor);",
                "                packetCapture.Capture();",
                "                if (downloadRet == 11) continue;",
                "                if (downloadRet != 0) { callRet = downloadRet; break; }",
                "                cursor += chunkBytes;",
                "                remainingBytes -= chunkBytes;",
                "                transferredBytes += chunkBytes;",
                "            }",
                "            packetCapture.Capture();",
                "            short endRet = cnc_dwnend3_fn(handle);",
                "            packetCapture.Capture();",
                "            if (callRet == 0) callRet = endRet;",
                "        }",
                f'        string data = "program={program_name};pre_search_ret=" + to_string(preSearchRet) + ";existing=" + string(existingProgram ? "1" : "0") + ";start_ret=" + to_string(startRet) + ";bytes=" + to_string(transferredBytes) + "/" + to_string(totalBytes);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_dwnstart3/cnc_download3/cnc_dwnend3", callRet, data);',
            ]
        )
    elif step.interface_name == "SelectProgram":
        program_name = str(step.parameters.get("program_name", "O0"))
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_search", "{params}");',
                f'        short programNo = static_cast<short>(ProgramNumberFromName("{program_name}"));',
                "        packetCapture.Capture();",
                "        short callRet = cnc_search_fn(handle, programNo);",
                "        packetCapture.Capture();",
                '        string data = "program_no=" + to_string(programNo);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_search", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadProgramNumber":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_rdprgnum", "{params}");',
                "        ODBPRO prgnum{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_rdprgnum_fn(handle, &prgnum);",
                "        packetCapture.Capture();",
                '        string data = "running_program=" + to_string(prgnum.data) + ";main_program=" + to_string(prgnum.mdata);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_rdprgnum", callRet, data);',
            ]
        )
    elif step.interface_name == "StartProgram":
        window_title = str(step.parameters.get("window_title", "FANUC CNC GUIDE"))
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "ncguide_ui_cycle_start", "{params}");',
                "        // UI target and coordinates come only from the current run environment.",
                "        string data;",
                "        packetCapture.Capture();",
                f'        short callRet = TriggerNcGuideCycleStart("{window_title}", data);',
                "        packetCapture.Capture();",
                "        short verifyRet = 0;",
                "        short verifiedRun = 0;",
                "        short verifiedMotion = 0;",
                "        short verifiedAut = 0;",
                "        short verifiedAlarm = 0;",
                "        for (int verifyIndex = 0; verifyIndex < 15; ++verifyIndex) {",
                "            ODBST verifyStatus{};",
                "            packetCapture.Capture();",
                "            verifyRet = cnc_statinfo_fn(handle, &verifyStatus);",
                "            packetCapture.Capture();",
                "            verifiedAut = verifyStatus.aut;",
                "            verifiedRun = verifyStatus.run;",
                "            verifiedMotion = verifyStatus.motion;",
                "            verifiedAlarm = verifyStatus.alarm;",
                "            if (verifyRet == 0 && verifiedMotion != 0) break;",
                "            this_thread::sleep_for(chrono::milliseconds(300));",
                "        }",
                '        data = "triggered=" + string(callRet == 0 ? "1" : "0") + ";verified_run=" + to_string(verifiedRun) + ";verified_motion=" + to_string(verifiedMotion) + ";verified_alarm=" + to_string(verifiedAlarm);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "ncguide_ui_cycle_start", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadRunStatus":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_statinfo", "{params}");',
                "        ODBST status{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_statinfo_fn(handle, &status);",
                "        packetCapture.Capture();",
                '        string data = "aut=" + to_string(status.aut) + ";run=" + to_string(status.run) + ";motion=" + to_string(status.motion) + ";emergency=" + to_string(status.emergency) + ";alarm=" + to_string(status.alarm) + ";edit=" + to_string(status.edit);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_statinfo", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadPosition":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_rdposition", "{params};type=1;axis_count=3");',
                "        short type = 1;",
                "        short axisCount = 3;",
                "        ODBPOS positions[3]{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_rdposition_fn(handle, type, &axisCount, positions);",
                "        packetCapture.Capture();",
                '        string data = "X=" + to_string(axisCount > 0 ? positions[0].mach.data : 0) + ";Y=" + to_string(axisCount > 1 ? positions[1].mach.data : 0) + ";Z=" + to_string(axisCount > 2 ? positions[2].mach.data : 0);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_rdposition", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadDistanceToGo":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_distance", "{params};axis_count=3");',
                "        short axisCount = 3;",
                "        ODBAXIS distance[3]{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_distance_fn(handle, -1, &axisCount, distance);",
                "        packetCapture.Capture();",
                '        string data = "distance_to_go_axis1=" + to_string(axisCount > 0 ? distance[0].data : 0) + ";distance_to_go_axis2=" + to_string(axisCount > 1 ? distance[1].data : 0) + ";distance_to_go_axis3=" + to_string(axisCount > 2 ? distance[2].data : 0);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_distance", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadFeedSpeed":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_actf", "{params}");',
                "        ODBACT feed{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_actf_fn(handle, &feed);",
                "        packetCapture.Capture();",
                '        string data = "actual_feed=" + to_string(feed.data);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_actf", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadSpindleSpeed":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_acts", "{params}");',
                "        ODBACT spindle{};",
                "        packetCapture.Capture();",
                "        short callRet = cnc_acts_fn(handle, &spindle);",
                "        packetCapture.Capture();",
                '        string data = "spindle_speed=" + to_string(spindle.data);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_acts", callRet, data);',
            ]
        )
    elif step.interface_name == "ReadAlarm":
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "cnc_alarm2", "{params}");',
                "        long alarmBits = 0;",
                "        packetCapture.Capture();",
                "        short callRet = cnc_alarm2_fn(handle, &alarmBits);",
                "        packetCapture.Capture();",
                '        string data = "alarm_bits=" + to_string(alarmBits);',
                f'        WriteOutput(outputCsv, index, "{step.step_id}", "cnc_alarm2", callRet, data);',
            ]
        )
    else:
        raise ValueError(f"Unsupported C++ API generation step: {step.interface_name}/{api}")
    if step.interval_seconds:
        blocks.append(f"        this_thread::sleep_for(chrono::milliseconds({int(float(step.interval_seconds) * 1000)}));")
    blocks.append("    }")
    blocks.append("")
    return blocks


def render_cpp_api_script(scenario: str, steps) -> str:
    method_names = sorted({step.protocol_function for step in steps if step.protocol_function})
    primary_method = method_names[0] if method_names else "cnc_statinfo"
    interfaces = sorted({step.interface_name for step in steps})
    return f'''// Auto-generated FOCAS C++ API script for scenario: {scenario}
// Compile example:
// cl /utf-8 /EHsc /std:c++17 /I"C:\\\\Lib\\\\FOCAS2 Library\\\\Fwlib\\\\30i" /I"C:\\\\Lib\\\\WinPcapSDK\\\\Include" api_script.cpp /link /LIBPATH:"C:\\\\Lib\\\\FOCAS2 Library\\\\Fwlib" /LIBPATH:"C:\\\\Lib\\\\WinPcapSDK\\\\Lib" Fwlib32.lib wpcap.lib Packet.lib ws2_32.lib

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include "fwlib32.h"
#include <pcap.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "fwlib32.lib")
#pragma comment(lib, "wpcap.lib")
#pragma comment(lib, "Packet.lib")

using namespace std;

string GetTimestamp() {{
    time_t now = time(0);
    tm ltm;
    localtime_s(&ltm, &now);
    stringstream ss;
    ss << put_time(&ltm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}}

string GetReturnCodeMeaning(short code) {{
    switch(code) {{
        case 0: return "EW_OK: Success";
        case 1: return "EW_FUNC: Function not executed";
        case 2: return "EW_LENGTH: Data block length error";
        case 3: return "EW_NUMBER: Data number error";
        case 4: return "EW_ATTRIB: Data attribute error";
        case 5: return "EW_DATA: Data error";
        case 6: return "EW_NOOPT: No option";
        case 7: return "EW_PROT: Write protected";
        case 8: return "EW_OVRFLOW: Memory overflow";
        case 9: return "EW_PARAM: CNC parameter error";
        case 10: return "EW_BUFFER: Buffer full or empty";
        case 11: return "EW_PATH: Path error";
        case 12: return "EW_MODE: CNC mode error";
        case 13: return "EW_REJECT: Execution rejected";
        case 14: return "EW_DTSRVR: Data server error";
        case 15: return "EW_ALARM: Alarm state";
        case 16: return "EW_STOP: Stop state";
        case 17: return "EW_PASSWD: Data protection";
        default: return "Unknown error code: " + to_string(code);
    }}
}}

string EscapeCsvField(const string& field) {{
    if (field.find(',') != string::npos || field.find('"') != string::npos || field.find('\\n') != string::npos) {{
        string escaped = "\\"";
        for (char c : field) {{
            if (c == '"') escaped += "\\"\\"";
            else escaped += c;
        }}
        escaped += "\\"";
        return escaped;
    }}
    return field;
}}

class PacketSniffer {{
private:
    pcap_t* handle;
    pcap_dumper_t* dumper;
    bool capturing;
    string deviceName;
    string pcapFileName;

public:
    PacketSniffer(const string& device, const string& output)
        : handle(nullptr), dumper(nullptr), capturing(false), deviceName(device), pcapFileName(output) {{}}

    bool Start() {{
        char errbuf[PCAP_ERRBUF_SIZE] = {{0}};
        handle = pcap_open_live(deviceName.c_str(), 65536, 1, 1000, errbuf);
        if (handle == nullptr) {{
            cerr << "Error opening adapter: " << errbuf << endl;
            return false;
        }}
        dumper = pcap_dump_open(handle, pcapFileName.c_str());
        if (dumper == nullptr) {{
            cerr << "Error opening pcap dump file: " << pcap_geterr(handle) << endl;
            pcap_close(handle);
            handle = nullptr;
            return false;
        }}
        capturing = true;
        return true;
    }}

    void CapturePackets(int count = 20) {{
        if (!capturing || !handle || !dumper) return;
        int captured = pcap_dispatch(handle, count,
            [](u_char* user, const struct pcap_pkthdr* pkthdr, const u_char* packet) {{
                pcap_dumper_t* dumper = reinterpret_cast<pcap_dumper_t*>(user);
                pcap_dump(reinterpret_cast<u_char*>(dumper), pkthdr, packet);
            }}, reinterpret_cast<u_char*>(dumper));
        if (captured > 0) pcap_dump_flush(dumper);
    }}

    void Stop() {{
        if (!capturing) return;
        if (dumper) {{
            pcap_dump_flush(dumper);
            pcap_dump_close(dumper);
            dumper = nullptr;
        }}
        if (handle) {{
            pcap_close(handle);
            handle = nullptr;
        }}
        capturing = false;
    }}
}};

int main() {{
    SetConsoleOutputCP(CP_UTF8);

    const char* hostEnv = getenv("FOCAS_HOST");
    string cncIp = (hostEnv && hostEnv[0]) ? hostEnv : "127.0.0.1";
    const char* portEnv = getenv("FOCAS_PORT");
    unsigned short port = static_cast<unsigned short>(portEnv ? std::stoi(portEnv) : 8193);
    const char* envDevice = getenv("PCAP_NETWORK_DEVICE");
    string networkDevice = (envDevice && envDevice[0]) ? envDevice : "\\\\Device\\\\NPF_Loopback";
    string methodName = "{primary_method}";
    vector<string> plannedInterfaces = {{{", ".join(f'"{item}"' for item in interfaces)}}};

    CreateDirectoryA("data", NULL);
    string pcapFile = "data\\\\Fanuc_" + methodName + ".pcap";
    string inputCsvFile = "data\\\\Fanuc_" + methodName + "_input.csv";
    string outputCsvFile = "data\\\\Fanuc_" + methodName + "_output.csv";

    PacketSniffer sniffer(networkDevice, pcapFile);
    if (!sniffer.Start()) {{
        cerr << "PCAP_CAPTURE_FAILED: failed to open configured Npcap device" << endl;
        return 1;
        cerr << u8"启动流量捕获失败，请检查网卡设备名。" << endl;
    }}

    unsigned short focasHandle = 0;
    short ret = cnc_allclibhndl3(cncIp.c_str(), port, 30, &focasHandle);
    cout << "cnc_allclibhndl3: " << GetReturnCodeMeaning(ret) << endl;
    if (ret != EW_OK) {{
        cerr << u8"CNC连接失败。" << endl;
        sniffer.Stop();
        return 1;
    }}

    ofstream inputCsv(inputCsvFile, ios::binary);
    inputCsv.put(0xEF); inputCsv.put(0xBB); inputCsv.put(0xBF);
    inputCsv << "index,timestamp,api_name,description,param_count,parameters\\n";

    ofstream outputCsv(outputCsvFile, ios::binary);
    outputCsv.put(0xEF); outputCsv.put(0xBB); outputCsv.put(0xBF);
    outputCsv << "index,timestamp,api_name,return_code,return_desc,status,error_message,data\\n";

    int testIndex = 0;
    vector<short> statTypes = {{0}};
    for (short statType : statTypes) {{
        testIndex++;
        string timestamp = GetTimestamp();
        inputCsv << testIndex << "," << timestamp << "," << methodName << ","
                 << EscapeCsvField(u8"读取CNC运行状态") << ",1,type=" << statType << "\\n";

        sniffer.CapturePackets(20);
        ODBST statinfo;
        memset(&statinfo, 0, sizeof(statinfo));
        ret = cnc_statinfo(focasHandle, &statinfo);
        this_thread::sleep_for(chrono::milliseconds(50));
        sniffer.CapturePackets(20);

        string status = (ret == EW_OK) ? "Success" : "Failed";
        string data = "aut=" + to_string(statinfo.aut) + ";run=" + to_string(statinfo.run) + ";alarm=" + to_string(statinfo.alarm);
        outputCsv << testIndex << "," << timestamp << ",cnc_statinfo," << ret << ","
                  << EscapeCsvField(GetReturnCodeMeaning(ret)) << "," << status << ",,"
                  << EscapeCsvField(data) << "\\n";
    }}

    cout << u8"计划接口数量: " << plannedInterfaces.size() << endl;
    cout << u8"注意: 请根据 RAG 检索到的 API 原型扩展本文件中的具体 API 调用参数遍历。" << endl;

    inputCsv.close();
    outputCsv.close();
    cnc_freelibhndl(focasHandle);
    sniffer.Stop();

    cout << u8"测试完成。" << endl;
    cout << "input: " << inputCsvFile << endl;
    cout << "output: " << outputCsvFile << endl;
    cout << "pcap: " << pcapFile << endl;
    return 0;
}}
'''


def validate_generated(
    api_script: str,
    nc_program: str,
    executable_steps: list[PlanStep] | None = None,
    *,
    allow_delete_all_programs: bool = False,
    require_official_focas_header: bool = False,
    require_real_pcap_capture: bool = False,
    code_only_evaluation: bool = False,
) -> list[str]:
    import re

    diagnostics: list[str] = []
    if re.search(r"expected_return_code|error[- _]?path\s+probe|negative\s+probe", api_script, re.IGNORECASE):
        diagnostics.append(
            "BLOCKING: generated C++ contains a disabled error-path probe or expected_return_code; "
            "only successful documented API calls may be collected."
        )
    if "cnc_allclibhndl3" not in api_script:
        diagnostics.append("C++ API script does not connect with cnc_allclibhndl3")
    if "LoadLibraryW" not in api_script or "GetProcAddress" not in api_script:
        diagnostics.append("C++ API script does not dynamically load Fwlib32.dll")
    fixed_harness = (
        '#include "scenario_fragment.inc"' in api_script
        and "struct FocasApi" in api_script
        and "bool load()" in api_script
    )
    if (not fixed_harness and (
        "FOCAS_DLL_DIR" not in api_script
        or "GetEnvironmentVariableW" not in api_script
        or "SetDllDirectoryW" not in api_script
    )):
        diagnostics.append(
            "BLOCKING: C++ must load Fwlib32.dll using FOCAS_DLL_DIR, GetEnvironmentVariableW, and SetDllDirectoryW like cpp/focas_connect_demo.cpp."
        )
    if (
        "#include <windows.h>" in api_script
        and ("std::min" in api_script or "std::max" in api_script)
        and "NOMINMAX" not in api_script
    ):
        diagnostics.append(
            "BLOCKING: C++ uses std::min/std::max with windows.h but does not define NOMINMAX before including windows.h."
        )
    diagnostics.extend(validate_poselem_scalar_access(api_script))
    diagnostics.extend(validate_odbact_scalar_access(api_script))
    diagnostics.extend(validate_sdk_struct_members(api_script))
    diagnostics.extend(validate_focas_loader_type_safety(api_script))
    if require_official_focas_header:
        diagnostics.extend(validate_official_focas_header_usage(api_script))
        diagnostics.extend(validate_all_focas_symbols_against_header(api_script))
        header_path = default_focas_header_dir() / "Fwlib32.h"
        if header_path.is_file():
            diagnostics.extend(validate_return_code_labels(api_script, header_path.read_text(encoding="latin-1")))
    if require_real_pcap_capture:
        diagnostics.extend(validate_real_packet_capture_usage(api_script))
    diagnostics.extend(validate_no_aggregate_or_fake_manifest_coverage(api_script))
    if "M30" not in nc_program:
        diagnostics.append("NC program does not end with M30")
    if not nc_program.startswith("O"):
        diagnostics.append("NC program does not start with a program number")
    if executable_steps:
        planned_interfaces = {step.interface_name for step in executable_steps}
        planned_functions = {
            name for step in executable_steps for name in protocol_function_names(step.protocol_function)
        }
        # A target that is present in the active FOCAS header must not be
        # downgraded to a synthetic unavailable block by code generation.
        # Keep the normal target-invocation check intact; this diagnostic makes
        # the upstream false-availability decision explicit in the failure.
        confirmed_symbols = active_sdk_symbol_evidence(planned_functions)
        unavailable_block = re.compile(
            r'ctx\.block\s*\(\s*["\'](?P<api>(?:cnc|pmc|ds)_\w+)["\']\s*,\s*[^)]*'
            r'(?:no\s+prototype|dll\s+export|unsupported\s+api|api\s+unavailable)',
            re.IGNORECASE | re.DOTALL,
        )
        for match in unavailable_block.finditer(api_script):
            api_name = match.group("api")
            if confirmed_symbols.get(api_name) == "confirmed_header_prototype":
                diagnostics.append(
                    "BLOCKING: GENERATOR_FALSE_UNAVAILABLE: "
                    f"{api_name} is present in the active Fwlib32.h prototype set, "
                    "but generated code blocks it as unavailable; emit the real SDK call."
                )
        if not code_only_evaluation:
            if "StartProgram" not in planned_interfaces and re.search(
                r"\b(?:SetCursorPos|mouse_event|SendInput)\s*\(", api_script
            ):
                diagnostics.append("BLOCKING: unplanned NCGuide UI Cycle Start/cursor action")
            if "UploadProgram" not in planned_interfaces and not {
                "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"
            }.intersection(planned_functions) and re.search(
                r"\b(?:cnc_dwnstart3|cnc_download3|cnc_dwnend3)(?:_fn)?\s*\(", api_script
            ):
                diagnostics.append("BLOCKING: unplanned NC program download")
        forbidden_skip_markers = [
            "SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN",
            "not_executed_by_cpp_generator",
            "unsupported upload/select",
            "outside the currently permitted supported_by_cpp_codegen",
        ]
        for marker in forbidden_skip_markers:
            if marker in api_script:
                diagnostics.append(
                    f"BLOCKING: C++ script emits skip marker {marker!r} for planned executable steps."
                )
        required_snippets = {
            "UploadProgram": ["cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"],
            "SelectProgram": ["cnc_search"],
            "ReadProgramNumber": ["cnc_rdprgnum"],
            "ReadProgramDirectory": ["cnc_rdprogdir3"],
            "DeleteProgram": ["cnc_delete"],
            "StartProgram": ["SetCursorPos", "mouse_event"],
            "ReadRunStatus": ["cnc_statinfo"],
            "ReadPosition": ["cnc_rdposition"],
            "ReadDistanceToGo": ["cnc_distance"],
            "ReadFeedSpeed": ["cnc_actf"],
            "ReadSpindleSpeed": ["cnc_acts"],
            "ReadAlarm": ["cnc_alarm2"],
        }
        fallback_interfaces = {
            step.interface_name for step in executable_steps if not step.protocol_function.strip()
        }
        for interface_name in sorted(fallback_interfaces):
            for snippet in required_snippets.get(interface_name, []):
                if snippet not in api_script:
                    diagnostics.append(
                        f"BLOCKING: planned executable step {interface_name} is missing required implementation snippet {snippet}."
                    )
        if "UploadProgram" in planned_interfaces and re.search(
            r"(?:payload|nc_payload)\s*=\s*\"%\\n\"",
            api_script,
            re.IGNORECASE,
        ):
            diagnostics.append(
                "BLOCKING: cnc_download3 NC payload must begin with LF, not a leading percent line; "
                'use "\\nO...\\nM30\\n%".'
            )
        if require_official_focas_header:
            diagnostics.extend(validate_focas_position_call_safety(api_script, executable_steps))
        else:
            diagnostics.extend(validate_focas_program_directory_abi(api_script, executable_steps))
            diagnostics.extend(validate_focas_position_abi(api_script, executable_steps))
            diagnostics.extend(validate_focas_odbaxis_abi(api_script, executable_steps))
        # Function-pointer declarations can still compile when adjacent
        # same-typed SDK arguments are swapped.  Validate the call semantics
        # separately from the ABI shape checks above.
        diagnostics.extend(validate_focas_argument_semantics(api_script, executable_steps))
        diagnostics.extend(validate_metric_helper_types(api_script))
        diagnostics.extend(validate_semantic_api_bindings(api_script, executable_steps))
        uses_delete_all = "cnc_delall" in api_script.lower()
        if uses_delete_all and not allow_delete_all_programs:
            diagnostics.append(
                "BLOCKING: generated C++ uses cnc_delall without explicit allow_delete_all_programs permission."
            )
        if uses_delete_all and allow_delete_all_programs and not has_authorized_delete_all_gate(api_script):
            diagnostics.append(
                "BLOCKING: authorized cnc_delall must log delete_all_authorized=true, check its return code, and abort on failure."
            )
        if not code_only_evaluation and "UploadProgram" in planned_interfaces and not uses_delete_all and not has_target_program_availability_flow(api_script):
            diagnostics.append(
                "BLOCKING: NCGuide upload must read the program directory, determine target_program_exists exactly, "
                "and either choose an unused selected_program_number or safely delete the exact conflicting program before upload."
            )
        if "UploadProgram" in planned_interfaces and "cnc_delete" in api_script.lower():
            diagnostics.extend(validate_safe_single_program_delete_flow(api_script))
        if not code_only_evaluation and "StartProgram" in planned_interfaces and not has_cycle_start_ready_gate(api_script):
            diagnostics.append(
                "BLOCKING: planned StartProgram/Cycle Start steps must include a cycle_start_ready_gate or "
                "WaitUntilCycleStartReady helper that polls cnc_statinfo before clicking again."
            )
        if not code_only_evaluation and "StartProgram" in planned_interfaces:
            diagnostics.extend(validate_cycle_start_ready_states(api_script))
        if not code_only_evaluation and "StartProgram" in planned_interfaces:
            diagnostics.extend(validate_ncguide_cycle_start_env_usage(api_script))
        if not code_only_evaluation and "StartProgram" in planned_interfaces and not has_program_completion_gate(api_script):
            diagnostics.append(
                "BLOCKING: planned StartProgram/Cycle Start workflow must include a final program_completion_gate or "
                "WaitUntilProgramComplete helper before evaluation/disconnect."
            )
        if not code_only_evaluation and "StartProgram" in planned_interfaces and not has_program_completion_wait_logic(api_script):
            diagnostics.append(
                "BLOCKING: program_completion_gate must poll cnc_statinfo until idle completion and log completed/timeout, "
                "waited_ms, last_run, and last_motion before evaluation/disconnect."
            )
        if not code_only_evaluation and "StartProgram" in planned_interfaces and not has_whole_program_single_block_loop(api_script):
            diagnostics.append(
                "BLOCKING: Single Block execution must count and drive every effective NC segment, including the "
                "O-number line and M30, and log expected_nc_segment_count plus cycle_start_click_count."
            )
        lifecycle_interfaces = {"UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram"}
        if not code_only_evaluation and lifecycle_interfaces.issubset(planned_interfaces) and not has_program_verification_gate(api_script):
            diagnostics.append(
                "BLOCKING: uploaded-program execution must hard-gate Cycle Start on program_verified and emit "
                "PROGRAM_NOT_VERIFIED when upload/select fails or cnc_rdprgnum does not match the expected O number."
            )
        if not code_only_evaluation and ({"ReadPosition", "ReadDistanceToGo"} & planned_interfaces) and not logs_distance_to_go(api_script):
            diagnostics.append(
                "BLOCKING: coordinate sampling must log remaining movement using cnc_distance or "
                "cnc_rdposition(type=3)/ODBPOS.dist as distance-to-go data."
            )
    return diagnostics


def validate_real_packet_capture_usage(api_script: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    lowered = api_script.lower()
    # Contract checks must inspect executable source, not explanatory comments.
    # The generated dynamic-loader implementation may describe its declarations
    # as "minimal" while still performing real packet capture.
    source_without_comments = re.sub(r"//[^\r\n]*|/\*.*?\*/", "", lowered, flags=re.DOTALL)
    fake_markers = [
        "writeminimalpcap",
        "capturemarker",
        "minimal pcap",
        "pcap marker",
    ]
    if any(marker in source_without_comments for marker in fake_markers):
        diagnostics.append(
            "BLOCKING: generated C++ creates fake/minimal pcap marker files instead of attempting real packet capture."
        )

    if "packetsniffer" not in lowered or "packetsniffer sniffer" not in lowered:
        diagnostics.append(
            "BLOCKING: generated C++ must instantiate PacketSniffer sniffer(networkDevice, pcapFile) for real pcap capture attempts."
        )

    required_pcap_markers = [
        "pcap_open_live",
        "pcap_dump_open",
        "pcap_dispatch",
        "pcap_dump",
        "pcap_dump_flush",
    ]
    missing = [marker for marker in required_pcap_markers if marker not in lowered]
    if missing:
        diagnostics.append(
            "BLOCKING: PacketSniffer must use real Npcap/WinPcap capture APIs; missing "
            + ", ".join(missing)
            + "."
        )

    if ".pcap" in lowered and "pcap_capture_disabled" not in lowered and "pcap" in lowered:
        if "pcap_open_live" not in lowered or "pcap_dispatch" not in lowered:
            diagnostics.append(
                "BLOCKING: generated C++ references .pcap output but does not show a real capture-or-disabled path."
            )
    if "pcap_capture_disabled" in lowered:
        diagnostics.append(
            "BLOCKING: packet capture is mandatory for this task; generated C++ must abort on sniffer.Start() failure instead of logging pcap_capture_disabled and continuing."
        )
    if "pcap_unavailable_continuing" in lowered:
        diagnostics.append(
            "BLOCKING: generated C++ continues after Npcap failed; capture failure must abort before FOCAS calls."
        )
    if ("system32/npcap/wpcap.dll" in lowered or "system32\\\\npcap\\\\wpcap.dll" in lowered) and "syswow64" not in lowered:
        diagnostics.append(
            "BLOCKING: x86 generated C++ loads only the x64 System32 Npcap DLL; use SysWOW64/Npcap with a checked failure path."
        )
    compact = re.sub(r"\s+", "", lowered)
    if "bool pcapready" in lowered and "if(!pcapready)" not in compact:
        diagnostics.append(
            "BLOCKING: generated C++ must abort before any FOCAS call when PacketSniffer::Start() fails; pcapReady is currently allowed to continue."
        )
    elif "sniffer.start()" in lowered and "if(!sniffer.start())" not in compact:
        diagnostics.append(
            "BLOCKING: generated C++ must check PacketSniffer::Start() and exit nonzero before any FOCAS call when capture cannot start."
        )
    if "pcap_capture_enabled=true" not in lowered and "pcap_capture_started=true" not in lowered:
        diagnostics.append(
            "BLOCKING: generated C++ must log pcap_capture_enabled=true or pcap_capture_started=true when real packet capture starts."
        )
    if "pcap_network_device" not in lowered:
        diagnostics.append(
            "BLOCKING: generated C++ must read the Npcap adapter from PCAP_NETWORK_DEVICE instead of hardcoding the capture device."
        )
    if "npf_{change_me}" in lowered:
        diagnostics.append(
            "BLOCKING: generated C++ must not hardcode CHANGE_ME Npcap devices for Wireshark capture."
        )
    return diagnostics


def validate_code_only_source(api_script: str) -> list[str]:
    lowered = api_script.lower()
    forbidden = {
        "pcap": "packet-capture code",
        "wpcap": "packet-capture code",
        "focas_api_input.csv": "CSV collection output",
        "focas_api_output.csv": "CSV collection output",
        "loginput(": "runtime API logging",
        "logoutput(": "runtime API logging",
        "setcursorpos(": "NCGuide UI automation",
        "mouse_event(": "NCGuide UI automation",
    }
    found = sorted({label for marker, label in forbidden.items() if marker in lowered})
    if not found:
        return []
    return [
        "BLOCKING: RQ1 code-only output contains excluded runtime/data-collection content: "
        + ", ".join(found)
        + "."
    ]


def validate_ncguide_cycle_start_env_usage(api_script: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    required_markers = ["NCGUIDE_CYCLE_START_X", "NCGUIDE_CYCLE_START_Y", "NCGUIDE_CLICK_MODE", "NCGUIDE_WINDOW_TITLE"]
    missing = [marker for marker in required_markers if marker not in api_script]
    if missing:
        diagnostics.append(
            "BLOCKING: Cycle Start UI click must read NCGuide coordinates/window settings from environment variables; missing "
            + ", ".join(missing)
            + "."
        )
    if re.search(r"\bSetCursorPos\s*\(\s*\d+\s*,\s*\d+\s*\)", api_script):
        diagnostics.append(
            "BLOCKING: Cycle Start UI click uses hardcoded SetCursorPos numeric coordinates; use NCGUIDE_CYCLE_START_X/Y instead."
        )
    return diagnostics


def validate_no_aggregate_or_fake_manifest_coverage(api_script: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    string_literals = re.findall(r'"(?:\\.|[^"\\])*"', api_script)
    api_name_pattern = re.compile(r"\b(?:cnc|pmc)_[A-Za-z0-9_]+\b")
    for literal in string_literals:
        names = list(dict.fromkeys(api_name_pattern.findall(literal)))
        # Parameters and human-readable host-helper descriptions may mention
        # several APIs. Only fields whose value starts with an API name can be
        # api_name/protocol_function columns and need this validation.
        literal_value = literal[1:-1].lstrip() if len(literal) >= 2 else ""
        if not literal_value.startswith(("cnc_", "pmc_")):
            continue
        if len(names) >= 2 and (";" in literal or "/" in literal or "," in literal or "+" in literal):
            preview = literal[:120] + ("..." if len(literal) > 120 else "")
            diagnostics.append(
                "BLOCKING: CSV api_name/protocol_function fields must contain one real API per row; "
                f"found aggregate API literal {preview}."
            )
            break

    fake_coverage_markers = [
        "LifecycleManifestCoverage",
        "OffsetManifestCoverage",
        "ManifestCoverage",
        "best_effort_lifecycle_segment_recorded",
        "offset_mutation_not_performed_without_verified_offset_layout",
        "primary_motion_lifecycle_completed",
    ]
    found = [marker for marker in fake_coverage_markers if marker in api_script]
    if found:
        diagnostics.append(
            "BLOCKING: generated C++ must not emit fake manifest coverage rows without real API calls; found "
            + ", ".join(found[:4])
            + "."
        )
    return diagnostics


def validate_official_focas_header_usage(api_script: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    if not re.search(r"#\s*include\s*[<\"]Fwlib32\.h[>\"]", api_script, re.IGNORECASE):
        diagnostics.append(
            "BLOCKING: generated C++ must include the official controller-specific Fwlib32.h header."
        )
        return diagnostics

    resolved_functions = set(
        re.findall(
            r"GetProcAddress\s*\([^,]+,\s*\"(cnc_[A-Za-z0-9_]+)\"\s*\)",
            api_script,
            re.IGNORECASE,
        )
    )
    resolved_functions.update(
        re.findall(
            r"\bResolve\s*\([^,]+,\s*\"(cnc_[A-Za-z0-9_]+)\"\s*,",
            api_script,
            re.IGNORECASE,
        )
    )
    for function_name in sorted(resolved_functions):
        official_type = re.search(
            rf"decltype\s*\(\s*&\s*(?:::)?\s*{re.escape(function_name)}\s*\)",
            api_script,
            re.IGNORECASE,
        )
        if official_type is None:
            diagnostics.append(
                f"BLOCKING: dynamically resolved {function_name} must derive its function-pointer type "
                "from the official Fwlib32.h declaration with decltype(&::function_name)."
            )
    return diagnostics


def validate_all_focas_symbols_against_header(api_script: str) -> list[str]:
    """Require every generated FOCAS/PMC symbol to exist in the configured header.

    RAG and Planner output can provide context, but they are not authority for
    API existence or ABI. This gate deliberately runs against the same local
    Fwlib32.h used by the compiler.
    """
    import re

    header_path = default_focas_header_dir() / "Fwlib32.h"
    if not header_path.exists():
        return [f"BLOCKING: official Fwlib32.h is unavailable for strict API validation: {header_path}"]
    header = header_path.read_text(encoding="latin-1")
    declared = {
        name.lower()
        for name in re.findall(
            r"FWLIBAPI\s+[^;]*?\b((?:cnc|pmc)_[A-Za-z0-9_]+)\s*\(",
            header,
            re.IGNORECASE | re.DOTALL,
        )
    }
    referenced = extract_real_focas_symbol_references(api_script)
    unknown = sorted(referenced - declared)
    if not unknown:
        return []
    return [
        "BLOCKING: generated source references FOCAS/PMC symbols absent from the official Fwlib32.h: "
        + ", ".join(unknown)
        + "."
    ]


def extract_real_focas_symbol_references(api_script: str) -> set[str]:
    """Return actual FOCAS exports/calls, excluding local C++ identifiers.

    Fixed runners suffix local function-pointer variables with ``_fn``. A
    lexical scan of every ``cnc_*`` token therefore confuses calls through
    those pointers with SDK exports. Dynamic export strings and
    ``decltype(&::api)`` references are authoritative; direct call candidates
    are retained only when they are not local aliases, variables, or helpers.
    """
    import re

    symbol = r"((?:cnc|pmc)_[A-Za-z0-9_]+)"
    referenced = {
        name.lower()
        for name in re.findall(
            rf"GetProcAddress\s*\([^,]+,\s*\"{symbol}\"\s*\)",
            api_script,
            re.IGNORECASE,
        )
    }
    referenced.update(
        name.lower()
        for name in re.findall(
            rf"\bResolve\s*\([^,]+,\s*\"{symbol}\"\s*,",
            api_script,
            re.IGNORECASE,
        )
    )
    referenced.update(
        name.lower()
        for name in re.findall(
            rf"decltype\s*\(\s*&\s*(?:::)?\s*{symbol}\s*\)",
            api_script,
            re.IGNORECASE,
        )
    )

    lexical_source = strip_cpp_comments_and_literals(api_script)
    local_names = {
        name.lower()
        for name in re.findall(rf"\busing\s+{symbol}\s*=", lexical_source, re.IGNORECASE)
    }
    local_names.update(
        name.lower()
        for name in re.findall(
            rf"\btypedef\b[^;]*?\b{symbol}\s*;",
            lexical_source,
            re.IGNORECASE | re.DOTALL,
        )
    )
    local_names.update(
        name.lower()
        for name in re.findall(
            rf"\b(?:auto|decltype\s*\([^)]*\)|[A-Za-z_]\w*(?:::\w+)*(?:\s*[*&]+)?)\s+{symbol}\s*(?:=|;)",
            lexical_source,
            re.IGNORECASE,
        )
    )
    local_names.update(
        name.lower()
        for name in re.findall(
            rf"\b(?:void|bool|char|short|int|long|float|double|auto|string|std::\w+)\s+{symbol}\s*\([^;{{}}]*\)\s*(?:;|{{)",
            lexical_source,
            re.IGNORECASE,
        )
    )

    for name in re.findall(rf"\b{symbol}\s*\(", lexical_source, re.IGNORECASE):
        lowered = name.lower()
        if re.search(r"_(?:fn|p|t)$", lowered) or lowered in local_names:
            continue
        referenced.add(lowered)
    return referenced


def strip_cpp_comments_and_literals(source: str) -> str:
    """Mask C++ comments and quoted literals while preserving token spacing."""
    import re

    pattern = re.compile(
        r'//[^\r\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.DOTALL,
    )
    return pattern.sub(lambda match: " " * len(match.group(0)), source)


def validate_focas_position_call_safety(api_script: str, executable_steps: list[PlanStep]) -> list[str]:
    import re

    planned_functions = {
        name
        for step in executable_steps
        for name in protocol_function_names(step.protocol_function)
    }
    if "cnc_rdposition" not in planned_functions:
        return []
    if re.search(r"\b(?:axes|axisCount|data_num|num)\s*=\s*-\s*MAX_AXIS\b", api_script, re.IGNORECASE):
        return [
            "BLOCKING: cnc_rdposition data_num must be a positive allocated axis count, not -MAX_AXIS."
        ]
    return []


def has_cycle_start_ready_gate(api_script: str) -> bool:
    lowered = api_script.lower()
    return "cycle_start_ready_gate" in lowered or "waituntilcyclestartready" in lowered


def validate_cycle_start_ready_states(api_script: str) -> list[str]:
    """Reject the NCGuide gate that trapped run=1/motion=0 after one click."""
    source = re.sub(r"//[^\r\n]*|/\*.*?\*/", "", api_script, flags=re.DOTALL)
    gate = re.search(r"\bWaitUntilCycleStartReady\s*\([^;]*?\)\s*\{", source, re.DOTALL)
    if gate:
        depth = 1
        end = gate.end()
        while end < len(source) and depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        source = source[gate.end():end]
    run_zero = re.search(r"\b(?:\w+\.)?run\s*==\s*0\b", source)
    motion_zero = re.search(r"\b(?:\w+\.)?motion\s*==\s*0\b", source)
    run_one = re.search(r"\b(?:\w+\.)?run\s*==\s*1\b", source)
    if run_zero and motion_zero and not run_one:
        return [
            "BLOCKING: Cycle Start readiness treats only run=0/motion=0 as ready; "
            "NCGuide Single Block also uses run=1/motion=0 between executable lines."
        ]
    return []


def validate_focas_program_directory_abi(api_script: str, executable_steps: list[PlanStep]) -> list[str]:
    diagnostics: list[str] = []
    planned_functions = {
        name
        for step in executable_steps
        for name in protocol_function_names(step.protocol_function)
    }
    expected_parameter_counts = {
        "cnc_rdprogdir": 6,
        "cnc_rdprogdir2": 5,
        "cnc_rdprogdir3": 5,
    }
    for function_name, expected_count in expected_parameter_counts.items():
        if function_name not in planned_functions:
            continue
        actual_count = dynamic_function_pointer_parameter_count(api_script, function_name)
        if actual_count is None:
            diagnostics.append(
                f"BLOCKING: planned {function_name} is missing an inspectable dynamic function-pointer prototype."
            )
        elif actual_count != expected_count:
            diagnostics.append(
                f"BLOCKING: planned {function_name} uses a {actual_count}-argument function-pointer prototype; "
                f"the documented prototype requires {expected_count} arguments."
            )
    return diagnostics


def dynamic_function_pointer_parameter_count(api_script: str, function_name: str) -> int | None:
    import re

    pattern = re.compile(
        rf"\(\s*__(?:stdcall|cdecl)\s*\*\s*{re.escape(function_name)}\s*\)\s*\(([^)]*)\)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(api_script)
    if match is None:
        return None
    parameters = match.group(1).strip()
    if not parameters or parameters == "void":
        return 0
    return parameters.count(",") + 1


def validate_focas_position_abi(api_script: str, executable_steps: list[PlanStep]) -> list[str]:
    import re

    planned_functions = {
        name
        for step in executable_steps
        for name in protocol_function_names(step.protocol_function)
    }
    planned_interfaces = {step.interface_name for step in executable_steps}
    if "cnc_rdposition" not in planned_functions:
        return []
    diagnostics: list[str] = []
    parameter_count = dynamic_function_pointer_parameter_count(api_script, "cnc_rdposition")
    if parameter_count is not None and parameter_count != 4:
        diagnostics.append(
            f"BLOCKING: cnc_rdposition uses a {parameter_count}-argument function-pointer prototype; "
            "the documented prototype requires 4 arguments."
        )
    odbpos_match = re.search(
        r"struct\s+ODBPOS\w*\s*\{([^}]*)\}",
        api_script,
        re.IGNORECASE | re.DOTALL,
    )
    odbpos_body = odbpos_match.group(1).lower() if odbpos_match else ""
    if not all(re.search(rf"\b{name}\b", odbpos_body) for name in ["abs", "mach", "rel", "dist"]):
        diagnostics.append(
            "BLOCKING: cnc_rdposition output structure must contain documented ODBPOS abs/mach/rel/dist fields; "
            "an invented aggregate or reduced array can violate the official ABI."
        )
    if re.search(r"\b(?:axes|data_num|num)\s*=\s*-\s*MAX_AXIS\b", api_script, re.IGNORECASE) or re.search(
        r"CallPos\s*\([^;\n]*-\s*MAX_AXIS", api_script, re.IGNORECASE
    ):
        diagnostics.append(
            "BLOCKING: cnc_rdposition data_num must be a positive allocated axis count, not -MAX_AXIS or another negative selector."
        )
    return diagnostics


def validate_focas_odbaxis_abi(api_script: str, executable_steps: list[PlanStep]) -> list[str]:
    import re

    planned_functions = {
        name
        for step in executable_steps
        for name in protocol_function_names(step.protocol_function)
    }
    diagnostics: list[str] = []
    for function_name in ["cnc_absolute", "cnc_absolute2", "cnc_machine", "cnc_relative", "cnc_relative2", "cnc_distance"]:
        if function_name not in planned_functions:
            continue
        parameters = dynamic_function_pointer_parameters(api_script, function_name)
        if parameters is None:
            diagnostics.append(
                f"BLOCKING: planned {function_name} is missing an inspectable dynamic function-pointer prototype."
            )
            continue
        if len(parameters) != 4:
            diagnostics.append(
                f"BLOCKING: {function_name} requires 4 arguments but the generated prototype has {len(parameters)}."
            )
        elif "*" in parameters[2]:
            diagnostics.append(
                f"BLOCKING: {function_name} third argument is a length value, not a pointer."
            )
        if re.search(rf"{function_name}\s*\([^;\n]*,\s*sizeof\s*\(", api_script, re.IGNORECASE):
            diagnostics.append(
                f"BLOCKING: {function_name} third argument must use the documented ODBAXIS data-block size (4 + 4 * MAX_AXIS for 0i-D)."
            )
        if re.search(rf"{function_name}\s*\([^;\n]*,\s*&\s*\w+\s*,", api_script, re.IGNORECASE):
            diagnostics.append(
                f"BLOCKING: {function_name} call passes &length; pass the documented ODBAXIS data-block size."
            )
        if re.search(r"\b(?:const\s+)?(?:short|long)\s+length\s*=\s*sizeof\s*\(\s*ODBAXIS\s*\)", api_script, re.IGNORECASE):
            diagnostics.append(
                f"BLOCKING: {function_name} uses an unnormalized ODBAXIS length; use 4 + 4 * MAX_AXIS for 0i-D all-axis reads."
            )
    return diagnostics


def validate_focas_argument_semantics(api_script: str, executable_steps: list[PlanStep] | None = None) -> list[str]:
    """Check semantic ordering for SDK calls whose adjacent arguments share a type.

    C++ accepts a swapped ``short`` pair such as ``axis`` and ``length``.  The
    compiler therefore cannot catch this ABI error; inspect the call expression
    against the names and values in the active SDK contract before execution.
    This deliberately remains conservative and reports only an unambiguous
    length/axis (or type) swap.
    """
    import re

    planned = {
        name
        for step in (executable_steps or [])
        for name in protocol_function_names(step.protocol_function)
    }
    diagnostics: list[str] = []
    # The SDK position family is (FlibHndl, axis/type, length, ODBAXIS*).
    # Keep the check expression based so it also covers dynamically resolved
    # members (api.cnc_relative) and local function-pointer variables.
    functions = planned or {
        name for name in ("cnc_absolute", "cnc_absolute2", "cnc_machine",
                          "cnc_relative", "cnc_relative2", "cnc_distance")
        if re.search(rf"\b(?:api\.)?{name}\s*\(", api_script)
    }
    for function_name in sorted(functions):
        if function_name not in {"cnc_absolute", "cnc_absolute2", "cnc_machine",
                                 "cnc_relative", "cnc_relative2", "cnc_distance"}:
            continue
        # Count direct SDK call arguments as a second guard.  A missing axis
        # argument is otherwise easy to miss because all remaining values are
        # short/pointer compatible at the C++ call site.
        call_pattern = re.compile(
            rf"\b(?:api\.)?{re.escape(function_name)}(?:_fn)?\s*\(([^;\n]*)\)",
            re.IGNORECASE,
        )
        for call in call_pattern.finditer(api_script):
            raw_args = call.group(1)
            depth = 0
            commas = 0
            for char in raw_args:
                if char in "(<[{":
                    depth += 1
                elif char in ")>]}" and depth:
                    depth -= 1
                elif char == "," and depth == 0:
                    commas += 1
            if commas + 1 != 4:
                diagnostics.append(
                    f"BLOCKING: {function_name} SDK call has {commas + 1} arguments; "
                    "the official Fwlib32.h prototype requires (handle, axis, length, &ODBAXIS)."
                )
                break
        if function_name == "cnc_relative" and re.search(r"\bODBPOS\b", api_script):
            diagnostics.append(
                "BLOCKING: cnc_relative output must use the SDK ODBAXIS structure; "
                "ODBPOS/POSELM is the cnc_rdposition structure and cannot be used for relative snapshots."
            )
        # Capture one call at a time.  The generated calls use simple scalar
        # expressions for these two arguments, so nested commas are not valid
        # in the selector/length positions.
        pattern = re.compile(
            rf"\b(?:api\.)?{re.escape(function_name)}(?:_fn)?\s*\(\s*[^,]+,\s*([^,]+),\s*([^,]+),\s*&",
            re.IGNORECASE,
        )
        for match in pattern.finditer(api_script):
            second = match.group(1).strip().lower()
            third = match.group(2).strip().lower()
            second_is_length = bool(re.search(r"\b(?:len|length|rellen|axislength)\b|sizeof", second))
            third_is_selector = bool(re.search(r"(?:^|\W)-1\b|\b(?:axis|selectedaxis|type)\b", third))
            if second_is_length and third_is_selector:
                diagnostics.append(
                    f"BLOCKING: {function_name} SDK arguments are reversed; expected "
                    "(handle, axis/type, length, &ODBAXIS), but the generated call passes length before axis/type."
                )
                break
            # A literal negative selector in the length position is always
            # invalid for this family, even when the first expression is not
            # named in the usual way.
            if re.fullmatch(r"(?:static_cast\s*<\s*short\s*>\s*\(\s*)?-1\s*\)?", third):
                diagnostics.append(
                    f"BLOCKING: {function_name} third SDK argument is -1; this position is the positive ODBAXIS length."
                )
                break
    return diagnostics


def dynamic_function_pointer_parameters(api_script: str, function_name: str) -> list[str] | None:
    import re

    pattern = re.compile(
        rf"\(\s*__(?:stdcall|cdecl)\s*\*\s*{re.escape(function_name)}\s*\)\s*\(([^)]*)\)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(api_script)
    if match is None:
        return None
    parameters = match.group(1).strip()
    if not parameters or parameters == "void":
        return []
    return [item.strip() for item in parameters.split(",")]


def validate_metric_helper_types(api_script: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    long_only = re.search(
        r"HasVariation\s*\(\s*const\s+std::vector\s*<\s*(?:long|long_t)\s*>\s*&",
        api_script,
        re.IGNORECASE,
    )
    short_call = re.search(
        r"HasVariation\s*\(\s*(?:metrics\.)?(?:run|motion|status)",
        api_script,
        re.IGNORECASE,
    )
    if long_only and short_call:
        diagnostics.append(
            "BLOCKING: HasVariation accepts only vector<long> but is called with a short status vector; "
            "use a type-generic template or matching overload."
        )
    return diagnostics


def has_program_completion_gate(api_script: str) -> bool:
    lowered = api_script.lower()
    return "program_completion_gate" in lowered or "waituntilprogramcomplete" in lowered


def has_program_completion_wait_logic(api_script: str) -> bool:
    import re

    lowered = api_script.lower()
    if not has_program_completion_gate(api_script):
        return False
    required_markers = [
        "cnc_statinfo",
        "completed",
        "timeout",
        "waited",
        "last_run",
        "last_motion",
    ]
    if not all(marker in lowered for marker in required_markers):
        return False
    idle_markers = ["run==0", "run == 0", "last_run=0", "last_run == 0", "motion==0", "motion == 0", "last_motion=0", "last_motion == 0"]
    if any(marker in lowered for marker in idle_markers):
        return True
    has_polling_loop = re.search(r"\b(?:while|for)\s*\(", lowered) is not None
    has_completion_condition = re.search(r"\bcompleted\s*=", lowered) is not None
    has_wait_bound = any(marker in lowered for marker in ["steady_clock", "timeoutms", "timeout_ms", "sleep_for"])
    return has_polling_loop and has_completion_condition and has_wait_bound


def has_whole_program_single_block_loop(api_script: str) -> bool:
    import re

    lowered = api_script.lower()
    required_markers = [
        "expected_nc_segment_count",
        "cycle_start_click_count",
        "m30",
    ]
    if not all(marker in lowered for marker in required_markers):
        return False
    has_named_runner = any(
        marker in lowered
        for marker in [
            "runuploadedprogramtocompletion",
            "effective_nc_segment_count",
            "countnceffectivesegments",
            "count_nc_effective_segments",
        ]
    )
    if has_named_runner:
        return True

    has_segment_counter = re.search(r"\bcount[a-z0-9_]*segments?\b", lowered) is not None
    has_bounded_loop = re.search(r"\b(?:for|while)\s*\(", lowered) is not None
    has_cycle_start_action = any(
        marker in lowered
        for marker in ["triggerncguidecyclestart", "mouse_event", "setcursorpos", "cycle_start"]
    )
    click_counter_names = r"(?:clickcount|cycle_start_click_count)"
    increments_click_count = any(
        re.search(pattern, lowered) is not None
        for pattern in [
            rf"\b{click_counter_names}\s*(?:\+\+|\+=\s*1)",
            rf"\+\+\s*{click_counter_names}\b",
            rf"\b{click_counter_names}\s*=\s*{click_counter_names}\s*\+\s*1",
        ]
    )
    return has_segment_counter and has_bounded_loop and has_cycle_start_action and increments_click_count


def has_program_verification_gate(api_script: str) -> bool:
    lowered = api_script.lower()
    required_markers = [
        "program_verified",
        "program_not_verified",
        "cnc_rdprgnum",
    ]
    if not all(marker in lowered for marker in required_markers):
        return False
    has_expected_program = any(
        marker in lowered
        for marker in ["expected_program", "expectedprogram", "uploaded_program_number"]
    )
    has_failure_gate = any(
        marker in lowered
        for marker in ["return 3", "return 1", "exitcode", "exit_code", "return false"]
    )
    return has_expected_program and has_failure_gate


def has_target_program_availability_flow(api_script: str) -> bool:
    lowered = api_script.lower()
    required_markers = [
        "target_program_exists",
        "program_number_available",
    ]
    if not all(marker in lowered for marker in required_markers):
        return False
    has_existence_read = any(
        marker in lowered
        for marker in [
            "cnc_rdprogdir",
            "program_directory_lookup",
            "target_program_lookup",
        ]
    )
    has_exact_match = any(marker in lowered for marker in ["exact_match", "expected_program_number"])
    has_runtime_selection = all(
        marker in lowered
        for marker in ["selected_program_number", "preferred_program_number"]
    )
    has_collision_strategy = any(
        marker in lowered
        for marker in ["collision_strategy=choose_unused", "choose_unused", "findunused", "find_unused"]
    ) or has_safe_single_program_delete_flow(api_script)
    return (
        has_existence_read
        and has_exact_match
        and has_runtime_selection
        and has_collision_strategy
        and "cnc_delall" not in lowered
    )


def validate_safe_single_program_delete_flow(api_script: str) -> list[str]:
    if has_safe_single_program_delete_flow(api_script):
        return []
    return [
        "BLOCKING: cnc_delete is allowed only for an exact confirmed single-program collision; "
        "log delete_called=true, delete_target_program, exact_match=true, and abort on delete failure."
    ]


def has_safe_single_program_delete_flow(api_script: str) -> bool:
    lowered = api_script.lower()
    if "cnc_delete" not in lowered:
        return False
    required_markers = [
        "target_program_exists",
        "exact_match",
        "delete_called=true",
        "delete_target_program",
    ]
    if not all(marker in lowered for marker in required_markers):
        return False
    failure_markers = [
        "program_replacement_failed",
        "delete_failed",
        "delete_ret",
        "delete_return",
        "return 3",
        "return 2",
        "return false",
        "exit_code",
    ]
    return any(marker in lowered for marker in failure_markers)


def has_authorized_delete_all_gate(api_script: str) -> bool:
    lowered = api_script.lower()
    if (
        "cnc_delall" not in lowered
        or "delete_all_authorized=true" not in lowered
        or "collision_strategy=delete_all_authorized" not in lowered
    ):
        return False
    return any(
        marker in lowered
        for marker in ["program_replacement_failed", "return 3", "exit_code", "return false"]
    )


def logs_distance_to_go(api_script: str) -> bool:
    lowered = api_script.lower()
    distance_markers = [
        "cnc_distance",
        "distance_to_go",
        "remaining_move",
        "remaining_distance",
        "dist_axis",
        ".dist",
        "pos[i].dist",
    ]
    has_distance_api = "cnc_distance" in lowered or "cnc_rdposition" in lowered
    return has_distance_api and any(marker in lowered for marker in distance_markers)


def blocking_codegen_diagnostics(diagnostics: list[str]) -> list[str]:
    return [item for item in diagnostics if item.startswith("BLOCKING:")]


def hard_blocking_codegen_diagnostics(diagnostics: list[str]) -> list[str]:
    hard_markers = [
        "cnc_delall without explicit allow_delete_all_programs permission",
        "authorized cnc_delall must log delete_all_authorized=true",
        "preserve existing programs and replan",
        "fake/minimal pcap marker files",
        "PacketSniffer sniffer",
        "real Npcap/WinPcap capture APIs",
        "abort before any FOCAS call",
        "check PacketSniffer::Start()",
        "aggregate API literal",
        "fake manifest coverage rows",
        "RQ1 code-only output contains excluded runtime/data-collection content",
        "CodeSpec has no official prototype evidence",
        "CodeSpec API coverage missing from generated source",
        "handwritten FOCAS function-pointer signatures are forbidden",
        "fwprintf wchar_t output must use %ls",
        "cnc_download3 payload must begin with LF",
        "cnc_download3 payload must end with LF followed by a single percent character",
        "C++ must load Fwlib32.dll using FOCAS_DLL_DIR",
        "cnc_rdposition must use an official ODBPOS object",
        "scenario_fragment.inc violates the fragment boundary",
    ]
    return [
        item
        for item in blocking_codegen_diagnostics(diagnostics)
        if any(marker in item for marker in hard_markers)
    ]


def review_generated_with_llm(
    llm_client: LlmClient,
    task_description: str,
    scenario: str,
    nc_program: str,
    api_script: str,
    *,
    planned_api_names: list[str] | None = None,
    code_spec: CodeSpec | None = None,
    code_only: bool = False,
) -> list[str]:
    system_prompt = (
        "You are reviewing one generated FANUC FOCAS C++ source file for a code-only benchmark.\n"
        "Return JSON only with keys ok and diagnostics. Check compilation plausibility, official ABI use, exact Planner API coverage, documented arguments, return-code handling, and resource cleanup.\n"
        "Do not require packet capture, PCAP, CSV, traffic collection, simulator UI, or runtime logs. Do not reject a source because it does not execute a simulator.\n"
        "Only require program upload/selection when those exact APIs are present in the Planner contract; a documented NC payload constant is valid otherwise.\n"
        if code_only
        else CODE_REVIEW_SYSTEM_PROMPT
    )
    # Do not hide the middle of a generated artifact from the reviewer: doing
    # so creates false truncation/ABI findings on otherwise complete sources.
    script_preview = (
        "COMPLETE C++ SCRIPT:\n" + api_script
        if len(api_script) <= 30000
        else "BEGINNING OF C++ SCRIPT:\n"
        + api_script[:14000]
        + "\n\n[review input truncated by size limit]\n\nEND OF C++ SCRIPT:\n"
        + api_script[-14000:]
    )
    user_prompt = (
        f"C++ generation policy:\n{FOCAS_CPP_GENERATION_SYSTEM_PROMPT if not code_only else 'Code-only source generation contract: no packet capture, CSV, UI automation, or runtime data collection.'}\n\n"
        "Verified NCGuide/FOCAS facts for this review:\n"
        "- cnc_download3 NC program data starts with LF and ends with one percent character; it does not start with a percent line.\n"
        "- The official prototype is cnc_dwnstart3(unsigned short, short); its second argument is the documented download type, not the NC O-number. The official local example uses cnc_dwnstart3(h, 0). The O-number is carried by the NC payload and later selection/search call.\n"
        "- cnc_dwnend3 should be called after a successful cnc_dwnstart3 session; returning immediately when cnc_dwnstart3 itself fails is valid and must not be reported as missing cleanup.\n"
        "- For the active default Fwlib32.h, ODBPOS is an array element with POSELM abs/mach/rel/dist fields. Each POSELM.data member is a scalar long. Axis indexing belongs on the ODBPOS array (positions[i].abs.data), never on POSELM.data. Do not substitute the conditional alternate idata/ldata definition.\n"
        "- cnc_getdtailerr err_no=4 after download means the same O number is already registered.\n"
        "- In this confirmed Single Block setup, the O-number line and M30 each consume a Cycle Start, in addition to modal and motion lines. Include them when reviewing expected_nc_segment_count.\n\n"
        f"Task: {task_description}\n"
        f"Scenario: {scenario}\n\n"
        f"NC program:\n{nc_program}\n\n"
        f"C++ FOCAS API script length: {len(api_script)} characters\n"
        f"Planner atomic API names (summary): {planned_api_names or []}\n"
        f"Structured CodeSpec contract:\n{asdict(code_spec) if code_spec is not None else {}}\n"
        f"Official ABI declarations and referenced structures:\n"
        f"{official_focas_abi_context([item.tool_name for item in (code_spec.api_contracts if code_spec is not None else [])])}\n"
        f"{script_preview}\n\n"
        f"Return JSON: {CODE_REVIEW_JSON_SCHEMA}"
    )
    try:
        payload = llm_client.invoke_json(system_prompt, user_prompt)
    except Exception as exc:
        return [f"LLM code review failed: {exc}"]

    diagnostics = []
    if payload.get("ok") is False:
        diagnostics.append("LLM marked generated artifacts as needing review.")
    values = payload.get("diagnostics", [])
    if isinstance(values, str):
        values = [values]
    if isinstance(values, list):
        diagnostics.extend(str(value) for value in values if str(value).strip())
    return diagnostics


def validate_semantic_api_bindings(api_script: str, executable_steps: list[PlanStep]) -> list[str]:
    """Ensure semantic read operations retain their documented official API."""
    expected = {
        "ReadRunStatus": "cnc_statinfo",
        "ReadPosition": "cnc_rdposition",
        "ReadFeedSpeed": "cnc_actf",
        "ReadSpindleSpeed": "cnc_acts",
        "ReadParameter": "cnc_rdparam",
        "ReadAlarm": "cnc_alarm2",
    }
    tool_data_apis = {
        "cnc_rdtool",
        "cnc_rdtofs",
        "cnc_rdtofsr",
        "cnc_rdtofsinfo",
        "cnc_rdtofsinfo2",
    }
    diagnostics: list[str] = []
    import re
    # Ignore comments so a commented-out symbol cannot satisfy the gate.
    executable_source = re.sub(r"//[^\n]*|/\*.*?\*/", "", api_script, flags=re.DOTALL)
    for step in executable_steps:
        required = expected.get(step.interface_name)
        if step.interface_name == "ToolDataAccess":
            if not any(
                re.search(rf"(?:\b\w+\.)?{re.escape(name)}(?:_p|_fn)?\s*\(", executable_source, re.IGNORECASE)
                for name in tool_data_apis
            ):
                diagnostics.append(
                    "BLOCKING: Missing required semantic operation: tool_data_access. "
                    "Expected one of: cnc_rdtool, cnc_rdtofs, cnc_rdtofsr, cnc_rdtofsinfo, cnc_rdtofsinfo2."
                )
            elif "cnc_rdtool" in executable_source and "IODBTLMNG" not in executable_source:
                diagnostics.append(
                    "BLOCKING: cnc_rdtool tool_data_access must use the official IODBTLMNG output structure."
                )
            continue
        if not required:
            continue
        if step.protocol_function and step.protocol_function != required:
            diagnostics.append(
                f"BLOCKING: semantic interface {step.interface_name} must bind to official API {required}, "
                f"not {step.protocol_function}."
            )
        if step.protocol_function and not re.search(rf"(?:\b\w+\.)?{re.escape(required)}(?:_p|_fn)?\s*\(", executable_source):
            diagnostics.append(
                f"BLOCKING: semantic interface {step.interface_name} is missing an actual call to official API {required}."
            )
    return diagnostics


def validate_code_spec_coverage(api_script: str, code_spec: CodeSpec | None) -> list[str]:
    """Check the generated source against the exact typed Planner hand-off."""
    if code_spec is None:
        return ["CodeSpec missing: CodeEngineer did not receive the typed Planner contract."]
    diagnostics: list[str] = []
    for contract in code_spec.api_contracts:
        if not contract.prototype:
            diagnostics.append(
                f"BLOCKING: CodeSpec has no official prototype evidence for Planner API {contract.tool_name}."
            )
        if contract.tool_name not in api_script:
            diagnostics.append(
                f"BLOCKING: CodeSpec API coverage missing from generated source: {contract.tool_name}."
            )
    return diagnostics


def validate_programmatic_abi_binding(api_script: str, code_spec: CodeSpec | None) -> list[str]:
    """Reject handwritten FOCAS signatures; ABI types must come from the header."""
    if code_spec is None:
        return ["BLOCKING: CodeSpec missing for programmatic ABI validation."]
    import re

    handwritten = re.findall(
        r"using\s+\w+\s*=\s*(?:short|int|long|void|unsigned\s+\w+)\s*\([^;]*\)\s*;",
        api_script,
        re.IGNORECASE,
    )
    if handwritten:
        return [
            "BLOCKING: handwritten FOCAS function-pointer signatures are forbidden; "
            "derive every API type with decltype(&::api_name)."
        ]
    return []


def validate_official_output_usage(api_script: str, code_spec: CodeSpec | None) -> list[str]:
    """Require field-level use of official output structures in code-only RQ1."""
    if code_spec is None:
        return []
    names = {contract.tool_name for contract in code_spec.api_contracts}
    import re
    diagnostics: list[str] = []
    if "cnc_rdposition" in names:
        has_odbpos_output = re.search(
            r"\b(?:\w+\.)?(?:cnc_rdposition|rdposition)\w*\s*\([^;\n]*,\s*(?:&\s*\w+|\w+(?:\.data\(\))?)\s*\)",
            api_script,
            re.IGNORECASE,
        )
        if "ODBPOS" not in api_script or not has_odbpos_output:
            diagnostics.append(
                "BLOCKING: cnc_rdposition must use an official ODBPOS object as its output argument."
            )
    if "cnc_rdtool" in names:
        if "IODBTLMNG" not in api_script:
            diagnostics.append(
                "BLOCKING: cnc_rdtool must use the official IODBTLMNG output structure."
            )
        if not re.search(r"(?:\b\w+\.)?cnc_rdtool(?:_p|_fn)?\s*\(", api_script, re.IGNORECASE):
            diagnostics.append(
                "BLOCKING: cnc_rdtool must be invoked as a real tool-management data read."
            )
    if "fwprintf" in api_script and re.search(r"fwprintf\s*\([^;]*%s", api_script) and "%ls" not in api_script:
        diagnostics.append(
            "BLOCKING: fwprintf wchar_t output must use %ls rather than %s."
        )
    return diagnostics


def validate_download_payload(api_script: str, code_spec: CodeSpec | None) -> list[str]:
    """Check the documented cnc_download3 byte framing when upload is planned."""
    if code_spec is None or "cnc_download3" not in {item.tool_name for item in code_spec.api_contracts}:
        return []
    import re

    diagnostics: list[str] = []
    has_leading_lf_program = re.search(r"\\nO[A-Za-z0-9_]+", api_script) or re.search(
        r'\\n"\s*"\s*O[A-Za-z0-9_]+', api_script
    )
    if not has_leading_lf_program:
        diagnostics.append("BLOCKING: cnc_download3 payload must begin with LF before the O program number.")
    has_trailing_lf_percent = re.search(r"\\n%", api_script) or re.search(
        r'\\n"\s*"\s*%', api_script
    )
    if not has_trailing_lf_percent:
        diagnostics.append("BLOCKING: cnc_download3 payload must end with LF followed by a single percent character.")
    return diagnostics
