from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..integrations.ncguide import default_focas_header_dir, default_focas_runtime_dir
from ..orchestration.roles import ActionSpec, AgentTemplate, message_to_dict
from ..agent_roles import CodeArtifactSpec, CodeSpec, ReviewReport
from ..knowledge import KnowledgeBase
from ..llm import LlmClient
from ..models import GeneratedArtifacts, NcProgramSpec, PlanStep, WorkflowState
from ..tool_protocol import ToolCall, tool_calls_from_step, validate_tool_calls
from ..utils import ensure_dir
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
        self.review_role = CodeReviewRole()

    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None:
            raise ValueError("Cannot generate code before a plan exists.")

        repair_context = summarize_codegen_repair_context(state)
        if repair_context:
            state.plan.rag_context["code_generation_repair_context"] = repair_context
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
        if self.environment is not None:
            state.plan.rag_context["environment_shared_rules"] = self.environment.shared_rules
        codegen_knowledge = retrieve_codegen_knowledge_context(self.knowledge_base, state, executable_steps)
        if codegen_knowledge:
            state.plan.rag_context["code_generation_knowledge"] = codegen_knowledge
        generated_dir = ensure_dir(output_dir / "generated")
        nc_program, nc_generation_diagnostics = generate_nc_program(
            state.request.description,
            state.plan.scenario_type,
            state.plan.nc_program_spec,
            state.plan.rag_context,
            self.llm_client,
        )
        api_script, cpp_generation_diagnostics = generate_cpp_api_script(
            state,
            executable_steps,
            nc_program,
            self.llm_client,
        )

        api_script_path = generated_dir / "api_script.py"
        legacy_cpp_path = generated_dir / "focas_test.cpp"
        if api_script_path.exists():
            api_script_path.unlink()
        if legacy_cpp_path.exists():
            legacy_cpp_path.unlink()
        api_script_path = generated_dir / "api_script.cpp"
        nc_program_path = generated_dir / "program.nc"
        nc_program_path.write_text(nc_program, encoding="utf-8")

        diagnostics: list[str] = []
        max_internal_repair_attempts = 2
        for compile_attempt in range(max_internal_repair_attempts + 1):
            api_script_path.write_text(api_script, encoding="utf-8")
            diagnostics = validate_generated(
                api_script,
                nc_program,
                executable_steps,
                allow_delete_all_programs=bool(state.request.permissions.get("allow_delete_all_programs")),
                require_official_focas_header=True,
                # RQ1 uses --no-execute: packet capture is an execution-time
                # concern and must not block code-only compilation evaluation.
                require_real_pcap_capture=bool(getattr(state.request, "execute", False)),
                code_only_evaluation=bool(state.request.permissions.get("code_only_evaluation")),
            )
            if state.request.permissions.get("code_only_evaluation"):
                diagnostics.extend(validate_code_only_source(api_script))
                diagnostics.extend(validate_code_spec_coverage(api_script, code_spec))
                diagnostics.extend(validate_programmatic_abi_binding(api_script, code_spec))
                diagnostics.extend(validate_official_output_usage(api_script, code_spec))
                diagnostics.extend(validate_download_payload(api_script, code_spec))
            diagnostics.extend(nc_generation_diagnostics)
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
            )
            blocking_diagnostics = hard_blocking_codegen_diagnostics(diagnostics)
            if blocking_diagnostics:
                if compile_attempt < max_internal_repair_attempts and self.llm_client.enabled:
                    api_script = repair_cpp_api_script_after_contract_error(
                        state,
                        executable_steps,
                        nc_program,
                        api_script,
                        "; ".join(blocking_diagnostics),
                        self.llm_client,
                        code_only=bool(state.request.permissions.get("code_only_evaluation")),
                    )
                    continue
                raise ValueError("Generated C++ violates non-negotiable execution safety constraints: " + "; ".join(blocking_diagnostics))
            if state.request.target_environment != "ncguide-generated-cpp":
                break
            compile_error = preflight_compile_generated_cpp(api_script_path, generated_dir)
            if not compile_error:
                if compile_attempt:
                    diagnostics.append(f"CodeGenerationAgent fixed MSVC preflight compilation after {compile_attempt} internal repair attempt(s).")
                break
            diagnostics.append(f"MSVC preflight compilation failed on CodeGenerationAgent internal attempt {compile_attempt + 1}: {compile_error}")
            state.artifacts.diagnostics = diagnostics
            if compile_attempt >= max_internal_repair_attempts or not self.llm_client.enabled:
                raise ValueError("Generated C++ failed MSVC preflight compilation: " + compile_error)
            api_script = repair_cpp_api_script_after_compile_error(
                state,
                executable_steps,
                nc_program,
                api_script,
                compile_error,
                self.llm_client,
                code_only=bool(state.request.permissions.get("code_only_evaluation")),
            )
            cpp_generation_diagnostics.append("CodeGenerationAgent regenerated C++ from MSVC preflight compiler diagnostics.")
        if self.llm_client.enabled:
            review_diagnostics = review_generated_with_llm(
                    self.llm_client,
                    state.request.description,
                    state.plan.scenario_type,
                    nc_program,
                    api_script,
                    planned_api_names=[
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
        state.stage = "execution"
        return state


def preflight_compile_generated_cpp(source_path: Path, work_dir: Path) -> str:
    from ..agent_tools import CompileCppInput, CompileGeneratedCppTool

    exe_path = work_dir / "preflight_api_script.exe"
    result = CompileGeneratedCppTool().invoke(CompileCppInput(source_path, exe_path, work_dir))
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
    return output[-3000:] or f"compiler exit code {result.return_code}"


def select_codegen_executable_steps(steps: list[PlanStep]) -> tuple[list[PlanStep], list[str]]:
    selected = []
    skipped = []
    for step in steps:
        if is_non_executable_codegen_step(step):
            skipped.append(
                f"Skipped {step.step_id} {step.interface_name}/{step.protocol_function}: "
                "Planner step is an analysis/evaluation step, not a direct C++ API/control operation."
            )
            continue
        selected.append(step)
    return selected, skipped


def is_non_executable_codegen_step(step: PlanStep) -> bool:
    text = f"{step.interface_name} {step.protocol_function} {step.action}".lower()
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
    return generated, ["LLM generated NC program from PlannerAgent specification."]


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
        "- Generate concrete FANUC-style NC program blocks, including the given O program name.\n"
        "- Use safe simulator-scale motion only.\n"
        "- If rag_context.coverage_segments contains multiple segments, generate the primary NC payload for the first segment that requires NC motion/program lifecycle. Other non-NC segments should be handled by C++ direct FOCAS calls. If multiple NC-required segments need distinct payloads, include concise segment notes so the C++ generator can embed additional safe O-number payloads if needed.\n"
        "- For coordinate-motion traffic, prefer several observable G01 motion blocks whose main motion lasts about 2-5 seconds each in simulation.\n"
        "- Use moderate low feed and moderate travel: slow enough for repeated sampling, but not so slow that the program takes excessive time. Avoid extremely low feed values or very long travel unless repair context explicitly requires them.\n"
        "- Use PlannerAgent's quality analysis: if feed/position variation is required, balance motion duration, sampling frequency, and total runtime.\n"
        "- If a previous NC program failed or produced poor traffic, revise that program based on the failure context instead of ignoring it.\n"
        "- End with M30.\n"
        "- Return JSON only. Do not include Markdown.\n\n"
        "# Output Schema\n"
        "{\"nc_program\":\"O1234\\nG90 G54\\nG01 X...\\nM30\\n\",\"notes\":[\"short note\"]}"
    )
    user_prompt = (
        f"Task:\n{task_description}\n\n"
        f"Scenario: {scenario}\n"
        f"Planner NC spec:\n"
        f"program_name={nc_program_spec.program_name}\n"
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
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    nc_program = str(payload.get("nc_program", "")).strip()
    if not nc_program:
        blocks = payload.get("blocks", [])
        if isinstance(blocks, list):
            nc_program = "\n".join(str(block).strip() for block in blocks if str(block).strip())
    return normalize_llm_nc_program(nc_program, nc_program_spec.program_name)


def normalize_llm_nc_program(nc_program: str, program_name: str) -> str:
    program_name = safe_nc_program_name(program_name)
    lines = [line.strip() for line in nc_program.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line and not line.startswith("%")]
    if not lines:
        return ""
    if not lines[0].startswith("O"):
        lines.insert(0, program_name)
    else:
        lines[0] = program_name
    if not any(line.upper().startswith("M30") for line in lines):
        lines.append("M30")
    if not is_safe_nc_program(lines):
        return ""
    return "\n".join(lines + [""])


def safe_nc_program_name(program_name: str) -> str:
    return f"O{safe_nc_program_number(nc_program_number(program_name, 5711)):04d}"


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
    fixed_interfaces = {
        "UploadProgram",
        "SelectProgram",
        "ReadProgramNumber",
        "StartProgram",
        "ReadRunStatus",
        "ReadPosition",
        "ReadDistanceToGo",
        "ReadFeedSpeed",
        "ReadSpindleSpeed",
        "ReadAlarm",
    }
    code_only = bool(state.request.permissions.get("code_only_evaluation"))
    if code_only or any(step.interface_name not in fixed_interfaces for step in executable_steps):
        script = generate_cpp_api_script_with_llm(state, executable_steps, nc_program, llm_client)
        return script, [
            "CodeGenerationAgent used the RAG/official-ABI LLM path; no TOOL_REGISTRY admission filter was applied."
        ]
    script = render_fixed_ncguide_cpp_script(state, executable_steps, nc_program)
    if not script:
        raise ValueError("CodeGenerationAgent fixed C++ scaffold renderer returned an empty script.")
    return script, [
        "CodeGenerationAgent used fixed NCGuide C++ scaffold; LLM only supplies fillable NC payload/configuration content."
    ]


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
    allow_delete_all = bool(state.request.permissions.get("allow_delete_all_programs"))
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
        short stRet = cnc_statinfo_fn(handle, &protectStatus);
        LogHost(inputCsv, outputCsv, idx, "S003A_DELETE_ALL_BEFORE_UPLOAD_STATUS", "cnc_statinfo",
                stRet, RetText(stRet),
                "delete_all_failed=true;delete_return=" + std::to_string(preDeleteRet) +
                ";delete_return_text=" + RetText(preDeleteRet) +
                ";aut=" + std::to_string(protectStatus.aut) +
                ";run=" + std::to_string(protectStatus.run) +
                ";motion=" + std::to_string(protectStatus.motion) +
                ";alarm=" + std::to_string(protectStatus.alarm) +
                ";edit=" + std::to_string(protectStatus.edit) +
                ";upload_protection_or_mode_gate=true");
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
    tm local{};
    localtime_s(&local, &t);
    std::ostringstream os;
    os << std::put_time(&local, "%Y-%m-%dT%H:%M:%S");
    return os.str();
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
    if (api == "cnc_allclibhndl3") return 4;
    if (api == "cnc_freelibhndl") return 1;
    if (api == "cnc_statinfo") return 2;
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
    if (api == "cnc_getdtailerr") return 2;
    if (api == "cnc_delall") return 1;
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
              const string& iface, const string& fn, const string& params, const string& ts) {
    csv << index << "," << EscapeCsv(ts) << "," << EscapeCsv(step)
        << "," << EscapeCsv(iface) << "," << EscapeCsv(fn) << "," << EscapeCsv(params)
        << "," << ApiParameterCount(fn) << "\n";
    csv.flush();
}

void LogOutput(std::ofstream& csv, int index, const string& step, const string& api,
               short ret, const string& text, const string& data, const string& ts) {
    csv << index << "," << EscapeCsv(ts) << "," << EscapeCsv(step) << "," << EscapeCsv(api)
        << "," << ret << "," << EscapeCsv(text) << "," << EscapeCsv(data)
        << "," << ApiParameterCount(api) << "\n";
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
    PacketSniffer(const string& networkDevice, const string& pcapFile)
        : device_(networkDevice), file_(pcapFile) {}
    bool Start() {
        dll_ = LoadLibraryA("wpcap.dll");
        if (!dll_) dll_ = LoadLibraryA("Npcap\\wpcap.dll");
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
    void CapturePackets(int packetCount = 12) {
        if (!handle_ || !dumper_ || !dispatch_) return;
        activeDump_ = dump_;
        dispatch_(handle_, packetCount, &PacketSniffer::DumpThunk, reinterpret_cast<u_char*>(dumper_));
        if (dumpFlush_) dumpFlush_(dumper_);
    }
    void Stop() {
        if (dumper_ && dumpClose_) { dumpFlush_(dumper_); dumpClose_(dumper_); dumper_ = nullptr; }
        if (handle_ && close_) { close_(handle_); handle_ = nullptr; }
        if (dll_) { FreeLibrary(dll_); dll_ = nullptr; }
    }
    string Error() const { return error_; }
private:
    static void DumpThunk(u_char* user, const pcap_pkthdr* header, const u_char* packet) {
        if (activeDump_) activeDump_(user, header, packet);
    }
    inline static pcap_dump_t activeDump_ = nullptr;
    string device_;
    string file_;
    string error_;
    HMODULE dll_ = nullptr;
    pcap_t* handle_ = nullptr;
    pcap_dumper_t* dumper_ = nullptr;
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
                    const string& step, const string& api, const string& params, Callable&& fn, OutputBuilder&& buildOutput) {
    string ts = Timestamp();
    PrintApiStart(idx, step, api, params);
    LogInput(inputCsv, idx, step, "during", api, api, params, ts);
    sniffer.CapturePackets();
    short ret = fn();
    sniffer.CapturePackets();
    string outputData = buildOutput();
    PrintApiEnd(idx, api, ret, RetText(ret), outputData);
    LogOutput(outputCsv, idx, step, api, ret, RetText(ret), outputData, ts);
    ++idx;
    return ret;
}

template <typename Callable>
short CallFocas(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx, PacketSniffer& sniffer,
                const string& step, const string& api, const string& params, Callable&& fn, const string& dataPrefix) {
    return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, step, api, params, std::forward<Callable>(fn),
                         [&]() { return dataPrefix; });
}

void LogHost(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx,
             const string& step, const string& api, short ret, const string& text, const string& data) {
    string ts = Timestamp();
    PrintApiStart(idx, step, api, data);
    LogInput(inputCsv, idx, step, "during", api, api, data, ts);
    PrintApiEnd(idx, api, ret, text, data);
    LogOutput(outputCsv, idx, step, api, ret, text, data, ts);
    ++idx;
}

bool ClickScreenPoint(int x, int y) {
    if (x <= 0 || y <= 0) return false;
    SetCursorPos(x, y);
    Sleep(80);
    mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
    Sleep(40);
    mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
    Sleep(120);
    return true;
}

void TriggerNcGuideCycleStart(std::ofstream& inputCsv, std::ofstream& outputCsv, int& idx, int clickCount) {
    int x = EnvInt("NCGUIDE_CYCLE_START_X", 0);
    int y = EnvInt("NCGUIDE_CYCLE_START_Y", 0);
    string mode = EnvString("NCGUIDE_CLICK_MODE", "screen");
    string title = EnvString("NCGUIDE_WINDOW_TITLE", "Main Panel");
    POINT pt{};
    pt.x = x;
    pt.y = y;
    HWND pointWindow = WindowFromPoint(pt);
    char pointTitle[256]{};
    if (pointWindow) GetWindowTextA(pointWindow, pointTitle, 255);
    bool pointWindowLooksRight = (title.empty() || string(pointTitle).find(title) != string::npos);
    bool clicked = ClickScreenPoint(x, y);
    LogHost(inputCsv, outputCsv, idx, "S_CYCLE_START_CLICK", "ncguide_ui_cycle_start", clicked ? 0 : 13,
            clicked ? "EW_OK" : "EW_REJECT",
            "window_title=" + title + ";click_mode=" + mode + ";cycle_start_x=" + std::to_string(x) +
            ";cycle_start_y=" + std::to_string(y) + ";cycle_start_click_count=" + std::to_string(clickCount) +
            ";expected_nc_segment_count=" + std::to_string(kExpectedNcSegmentCount) +
            ";point_window_title=" + string(pointTitle) +
            ";point_window_matches_target=" + string(pointWindowLooksRight ? "true" : "false") +
            ";ui_click_dispatched=" + string(clicked ? "true" : "false"));
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
    std::ofstream inputCsv("data\\focas_api_input.csv", std::ios::binary);
    std::ofstream outputCsv("data\\focas_api_output.csv", std::ios::binary);
    WriteBom(inputCsv);
    WriteBom(outputCsv);
    inputCsv << "index,timestamp,step_id,interface_name,protocol_function,parameters,api_parameter_count\n";
    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n";
    int idx = 1;

    string networkDevice = EnvString("PCAP_NETWORK_DEVICE", "\\Device\\NPF_Loopback");
    string pcapFile = "data\\focas_capture.pcap";
    PacketSniffer sniffer(networkDevice, pcapFile);
    {
        string ts = Timestamp();
        LogInput(inputCsv, idx, "PCAP_START", "before", "PacketCapture", "pcap_open_live", "device=" + networkDevice, ts);
        if (!sniffer.Start()) {
            LogOutput(outputCsv, idx, "PCAP_START", "pcap_open_live", -1, "PCAP_CAPTURE_FAILED",
                      "pcap_capture_enabled=false;error=" + sniffer.Error(), ts);
            return 2;
        }
        LogOutput(outputCsv, idx, "PCAP_START", "pcap_open_live", 0, "PCAP_CAPTURE_STARTED",
                  "pcap_capture_enabled=true;pcap_capture_started=true;file=" + pcapFile, ts);
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
    auto cnc_alarm2_fn = reinterpret_cast<decltype(&::cnc_alarm2)>(GetProcAddress(focasDll, "cnc_alarm2"));
    auto cnc_rdprogdir3_fn = reinterpret_cast<decltype(&::cnc_rdprogdir3)>(GetProcAddress(focasDll, "cnc_rdprogdir3"));
    auto cnc_dwnstart3_fn = reinterpret_cast<decltype(&::cnc_dwnstart3)>(GetProcAddress(focasDll, "cnc_dwnstart3"));
    auto cnc_download3_fn = reinterpret_cast<decltype(&::cnc_download3)>(GetProcAddress(focasDll, "cnc_download3"));
    auto cnc_dwnend3_fn = reinterpret_cast<decltype(&::cnc_dwnend3)>(GetProcAddress(focasDll, "cnc_dwnend3"));
    auto cnc_search_fn = reinterpret_cast<decltype(&::cnc_search)>(GetProcAddress(focasDll, "cnc_search"));
    auto cnc_rdprgnum_fn = reinterpret_cast<decltype(&::cnc_rdprgnum)>(GetProcAddress(focasDll, "cnc_rdprgnum"));
    auto cnc_actf_fn = reinterpret_cast<decltype(&::cnc_actf)>(GetProcAddress(focasDll, "cnc_actf"));
    auto cnc_rdposition_fn = reinterpret_cast<decltype(&::cnc_rdposition)>(GetProcAddress(focasDll, "cnc_rdposition"));
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
    auto cnc_accdecdly_fn = reinterpret_cast<decltype(&::cnc_accdecdly)>(GetProcAddress(focasDll, "cnc_accdecdly"));
    auto cnc_rddynamic_fn = reinterpret_cast<decltype(&::cnc_rddynamic)>(GetProcAddress(focasDll, "cnc_rddynamic"));
    auto cnc_rdaxisdata_fn = reinterpret_cast<decltype(&::cnc_rdaxisdata)>(GetProcAddress(focasDll, "cnc_rdaxisdata"));
    auto cnc_rd3dtooltip_fn = reinterpret_cast<decltype(&::cnc_rd3dtooltip)>(GetProcAddress(focasDll, "cnc_rd3dtooltip"));
    auto cnc_rdmdiprgstat_fn = reinterpret_cast<decltype(&::cnc_rdmdiprgstat)>(GetProcAddress(focasDll, "cnc_rdmdiprgstat"));
    auto cnc_rdmdipntr_fn = reinterpret_cast<decltype(&::cnc_rdmdipntr)>(GetProcAddress(focasDll, "cnc_rdmdipntr"));
__DELETE_ALL_RESOLVE__
    if (!cnc_allclibhndl3_fn || !cnc_freelibhndl_fn || !cnc_statinfo_fn || !cnc_alarm2_fn ||
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
    short ret = CallFocas(inputCsv, outputCsv, idx, sniffer, "S001_CONNECT", "cnc_allclibhndl3",
        "host=127.0.0.1;port=" + std::to_string(focasPort) + ";timeout_seconds=10;coverage_role=support",
        [&]() { return cnc_allclibhndl3_fn("127.0.0.1", static_cast<unsigned short>(focasPort), 10, &handle); },
        "host=127.0.0.1;port=" + std::to_string(focasPort) + ";coverage_role=support");
    if (ret != 0) { cleanup(); return 4; }

    auto freeHandle = [&]() {
        if (handle && cnc_freelibhndl_fn) {
            cnc_freelibhndl_fn(handle);
            handle = 0;
        }
    };

    auto logDetailError = [&](const string& step) {
        ODBERR err{};
        short er = cnc_getdtailerr_fn(handle, &err);
        LogHost(inputCsv, outputCsv, idx, step, "cnc_getdtailerr", er, RetText(er),
                "err_no=" + std::to_string(err.err_no) + ";err_dtno=" + std::to_string(err.err_dtno));
    };

    auto runPlannedApi = [&](const string& stepBase, const string& api, int segment, const string& phase) -> short {
        (void)segment;
        (void)phase;
        string noParams = "";
        if (api == "cnc_allclibhndl3" || api == "cnc_freelibhndl" ||
            api == "cnc_dwnstart3" || api == "cnc_download3" || api == "cnc_dwnend3" ||
            api == "cnc_search" || api == "cnc_rdprgnum" || api == "cnc_rdprogdir3") {
            LogHost(inputCsv, outputCsv, idx, stepBase + "_" + api, api, 0, "STRUCTURAL_API_HANDLED_BY_MAIN_CHAIN",
                    noParams);
            return 0;
        }
        if (api == "cnc_statinfo") {
            ODBST st{};
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_statinfo", "cnc_statinfo",
                noParams, [&]() { return cnc_statinfo_fn(handle, &st); },
                [&]() { return "run=" + std::to_string(st.run) + ";motion=" + std::to_string(st.motion) +
                ";alarm=" + std::to_string(st.alarm); });
        }
        if (api == "cnc_alarm2") {
            long alarmBits = 0;
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_alarm2", "cnc_alarm2",
                noParams, [&]() { return cnc_alarm2_fn(handle, &alarmBits); },
                [&]() { return "alarm_bits=" + std::to_string(alarmBits); });
        }
        if (api == "cnc_rdalmmsg") {
            if (!cnc_rdalmmsg_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdalmmsg", "cnc_rdalmmsg", 1, "UNSUPPORTED_SYMBOL", noParams);
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
        if (api == "cnc_distance") {
            ODBAXIS axis{};
            short len = sizeof(ODBAXIS);
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
                LogHost(inputCsv, outputCsv, idx, stepBase + "_" + axisApi, axisApi, 1, "UNSUPPORTED_SYMBOL", noParams);
                return 1;
            }
            ODBAXIS axis{};
            short len = sizeof(ODBAXIS);
            return CallFocasLazy(inputCsv, outputCsv, idx, sniffer, stepBase + "_" + axisApi, axisApi,
                "axis=-1;length=" + std::to_string(len),
                [&]() { return fn(handle, -1, len, &axis); },
                [&]() { return "axis_data1=" + std::to_string(axis.data[0]) +
                ";axis_data2=" + std::to_string(axis.data[1]) +
                ";axis_data3=" + std::to_string(axis.data[2]); });
        };
        if (api == "cnc_absolute") return runAxisApi("cnc_absolute", cnc_absolute_fn);
        if (api == "cnc_absolute2") return runAxisApi("cnc_absolute2", cnc_absolute2_fn);
        if (api == "cnc_machine") return runAxisApi("cnc_machine", cnc_machine_fn);
        if (api == "cnc_relative") return runAxisApi("cnc_relative", cnc_relative_fn);
        if (api == "cnc_relative2") return runAxisApi("cnc_relative2", cnc_relative2_fn);
        if (api == "cnc_skip") return runAxisApi("cnc_skip", cnc_skip_fn);
        if (api == "cnc_srvdelay") return runAxisApi("cnc_srvdelay", cnc_srvdelay_fn);
        if (api == "cnc_accdecdly") return runAxisApi("cnc_accdecdly", cnc_accdecdly_fn);
        if (api == "cnc_rddynamic") {
            if (!cnc_rddynamic_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rddynamic", "cnc_rddynamic", 1, "UNSUPPORTED_SYMBOL", noParams);
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
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdaxisdata", "cnc_rdaxisdata", 1, "UNSUPPORTED_SYMBOL", noParams);
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
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rd3dtooltip", "cnc_rd3dtooltip", 1, "UNSUPPORTED_SYMBOL", noParams);
                return 1;
            }
            ODB3DHDL tip{};
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rd3dtooltip", "cnc_rd3dtooltip",
                noParams, [&]() { return cnc_rd3dtooltip_fn(handle, &tip); }, "tooltip_read=true");
        }
        if (api == "cnc_rdmdiprgstat") {
            if (!cnc_rdmdiprgstat_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdmdiprgstat", "cnc_rdmdiprgstat", 1, "UNSUPPORTED_SYMBOL", noParams);
                return 1;
            }
            unsigned short mdiStatus = 0;
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdmdiprgstat", "cnc_rdmdiprgstat",
                noParams, [&]() { return cnc_rdmdiprgstat_fn(handle, &mdiStatus); },
                "mdi_program_status=" + std::to_string(mdiStatus));
        }
        if (api == "cnc_rdmdipntr") {
            if (!cnc_rdmdipntr_fn) {
                LogHost(inputCsv, outputCsv, idx, stepBase + "_cnc_rdmdipntr", "cnc_rdmdipntr", 1, "UNSUPPORTED_SYMBOL", noParams);
                return 1;
            }
            ODBMDIP mdi{};
            return CallFocas(inputCsv, outputCsv, idx, sniffer, stepBase + "_cnc_rdmdipntr", "cnc_rdmdipntr",
                noParams, [&]() { return cnc_rdmdipntr_fn(handle, &mdi); }, "mdi_pointer_read=true");
        }
        LogHost(inputCsv, outputCsv, idx, stepBase + "_" + api, api, 1, "SKIPPED_UNSUPPORTED_BY_FIXED_ADAPTER",
                noParams);
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
        [&]() { return "aut=" + std::to_string(initialStatus.aut) + ";run=" + std::to_string(initialStatus.run) +
        ";motion=" + std::to_string(initialStatus.motion) + ";alarm=" + std::to_string(initialStatus.alarm) +
        ";edit=" + std::to_string(initialStatus.edit) + ";coverage_role=support"; });

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
                "host=127.0.0.1;port=" + std::to_string(focasPort) + ";timeout_seconds=10;reason=EW_SOCKET_retry",
                [&]() { return cnc_allclibhndl3_fn("127.0.0.1", static_cast<unsigned short>(focasPort), 10, &handle); },
                "host=127.0.0.1;port=" + std::to_string(focasPort) + ";reason=EW_SOCKET_retry");
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
        short stRet = cnc_statinfo_fn(handle, &uploadEndStatus);
        LogHost(inputCsv, outputCsv, idx, "S008_UPLOAD_END_STATUS_GATE", "cnc_statinfo", stRet, RetText(stRet),
                "upload_end_failed=true;upload_end_return=" + std::to_string(ret) +
                ";upload_end_return_text=" + RetText(ret) +
                ";selected_program_number=" + std::to_string(selectedProgram) +
                ";aut=" + std::to_string(uploadEndStatus.aut) +
                ";run=" + std::to_string(uploadEndStatus.run) +
                ";motion=" + std::to_string(uploadEndStatus.motion) +
                ";alarm=" + std::to_string(uploadEndStatus.alarm) +
                ";edit=" + std::to_string(uploadEndStatus.edit) +
                ";upload_protection_or_mode_gate=" + string(ret == 7 ? "true" : "false"));
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
        short dRet = cnc_rddynamic_fn(handle, -1, len, &dy);
        if (dRet == 0) {
            seq = dy.seqnum;
            prg = dy.prgnum;
            feed = dy.actf;
        }
        return dRet;
    };

    auto verifyCycleStartEffect = [&](int segment, int clickCount, long beforeSeq) {
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
            lastStatRet = cnc_statinfo_fn(handle, &observedStatus);
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
    };

    auto waitUntilCycleStartReady = [&](int segment) -> bool {
        ODBST status{};
        int stableReadyPolls = 0;
        int waited = 0;
        short lastRet = 0;
        while (waited <= 15000) {
            lastRet = cnc_statinfo_fn(handle, &status);
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
        LogHost(inputCsv, outputCsv, idx, "S012_cycle_start_ready_gate", "cnc_statinfo",
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
        ++cycle_start_click_count;
        TriggerNcGuideCycleStart(inputCsv, outputCsv, idx, cycle_start_click_count);
        verifyCycleStartEffect(seg, cycle_start_click_count, beforeSeq);
        for (int sample = 0; sample < 8; ++sample) {
            string sampleStep = "S013_PLANNED_DURING_API_sample" + std::to_string(sample);
            runEarlyAbsoluteAndFeedReads(sampleStep, seg);
            runPlannedApiList(kPlannedDuringApis, sampleStep, seg, "during");
            Sleep(200);
        }
    }

    bool completed = false;
    ODBST last{};
    int waited_ms = 0;
    auto startWait = std::chrono::steady_clock::now();
    while (waited_ms < 20000) {
        CallFocasLazy(inputCsv, outputCsv, idx, sniffer, "S014_program_completion_gate_poll", "cnc_statinfo",
            "program_completion_gate=true;expected_nc_segment_count=" + std::to_string(expected_nc_segment_count) +
            ";cycle_start_click_count=" + std::to_string(cycle_start_click_count),
            [&]() { return cnc_statinfo_fn(handle, &last); },
            [&]() { return "program_completion_gate=true;last_run=" + std::to_string(last.run) +
            ";last_motion=" + std::to_string(last.motion); });
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
            "EW_OK",
            "program_completed=" + string(completed ? "true" : "false") +
            ";position_sample_count=generated_by_fixed_sampler;feed_sample_count=generated_by_fixed_sampler");

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
            "never manually spell a function-pointer typedef or signature. "
            "Use only the official structure type and fields recorded in CodeSpec. "
            "For cnc_rdposition, follow the official header: ODBPOS* points to an array of ODBPOS elements "
            "with official abs/mach/rel/dist POSELM fields; use a positive data_num and parse the requested axis elements. "
            "Never invent idata/ldata members. For cnc_absolute, "
            "use ODBAXIS directly through its official dummy/type/data[] fields; it has no abs member. If MSVC reports that ODBPOS has no dummy or data member, "
            "replace those accesses with the official ODBPOS abs/mach/rel/dist fields; do not define a replacement struct. "
            "The source must compile as C++17; do not use abbreviated function templates such as an auto parameter. "
            "For the active default ODBST definition, use only hdck, tmmode, aut, run, motion, mstb, emergency, alarm, and edit; manual and warning belong to conditional alternate definitions. "
            "Return one complete compilable C++17 source file."
        )
    user_prompt = (
        f"Task description:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"NC program, must preserve unless impossible:\n{nc_program}\n\n"
        f"Executable API steps:\n{steps_for_prompt(executable_steps)}\n\n"
        f"Structured CodeSpec with authoritative ABI contracts:\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
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
        repaired = normalize_official_odbaxis_fields(repaired)
        repaired = repaired.replace("print_axis_data(", "print_axis_positions(")
        repaired = re.sub(
            r"print_rdposition\(\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*\)",
            r"print_rdposition(\1[0])",
            repaired,
        )
        repaired = normalize_official_odbst_field_names(repaired)
        repaired = normalize_cpp17_auto_parameters(repaired)
        repaired = preserve_code_only_abi_preamble(api_script, repaired)
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


def normalize_official_odbst_field_names(api_script: str) -> str:
    """Remove fields unavailable in the active default ODBST definition."""
    import re

    for field in ("manual", "write", "labelskip", "warning", "battery"):
        api_script = re.sub(
            rf"\s*<<\s*\"\s*{field}=\"\s*<<\s*\w+\.{field}\b",
            "",
            api_script,
        )
    return api_script


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
        r"\b(?:program|payload|nc_program|nc_program_text)\b\s*=\s*",
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
    if percent_literals:
        first = percent_literals[0]
        segment = segment[: first.start()] + '""' + segment[first.end() :]
        percent_literals = list(re.finditer(r'"%\\n"', segment))
        if percent_literals:
            last = percent_literals[-1]
            segment = segment[: last.start()] + '"%"' + segment[last.end() :]

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
            "For cnc_rdposition, ODBPOS* points to an array of official objects with abs/mach/rel/dist POSELM fields; "
            "use a positive data_num and do not invent idata/ldata members."
            " For ODBPOS, read the official abs/mach/rel/dist fields; "
            "for Windows wide-character fwprintf diagnostics, use %ls for wchar_t*."
        )
    user_prompt = (
        f"Task:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"NC program:\n{nc_program}\n\n"
        f"Atomic executable steps:\n{steps_for_prompt(executable_steps)}\n\n"
        f"Structured CodeSpec:\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
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
    return repaired


def generate_cpp_api_script_with_llm(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    llm_client: LlmClient,
) -> str:
    assert state.plan is not None
    selected_functions = sorted(
        {
            function_name
            for step in executable_steps
            for function_name in protocol_function_names(step.protocol_function)
        }
    )
    official_abi_context = official_focas_abi_context(selected_functions)
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
        "- For FOCAS connection, follow cpp/focas_connect_demo.cpp exactly: read FOCAS_DLL_DIR with GetEnvironmentVariableW, call SetDllDirectoryW(dll_dir), load dll_dir + '\\\\Fwlib32.dll' with LoadLibraryW, resolve cnc_allclibhndl3/cnc_freelibhndl, then connect to 127.0.0.1:8193 with timeout 10.\n"
        "- Do not hard-code mojibake/garbled Chinese DLL paths inside generated C++; use FOCAS_DLL_DIR and wide-character Windows APIs for DLL loading.\n"
        "- Write data/focas_api_input.csv and data/focas_api_output.csv with UTF-8 BOM.\n"
        "- Every planned executable step must produce one input CSV row and one output CSV row per repeat.\n"
        "- Every input/output CSV row must include a timestamp column recorded at call time. Use ISO-8601 local time, "
        "ISO-8601 UTC time, or epoch milliseconds consistently. Matching input/output rows for one API call should share "
        "the same timestamp when possible.\n"
        "- If a step contains parameters.parameter_generation, use it to generate API input coverage: "
        "fixed parameters use the given value directly; enum parameters must be traversed over every listed value; "
        "range parameters must be sampled at the listed samples or at min/mid/max when samples are absent. "
        "For enum/range expansion, emit a real API call and matching input/output CSV rows for each generated parameter combination. "
        "Do not treat input variation as traffic quality; it is only the call-generation strategy.\n"
        "- The executable steps below are selected by PlannerAgent/LLM from task context and RAG; do not reject them merely because of local registry assumptions.\n"
        "- The listed interface meanings are examples, not the full API universe. If PlannerAgent provides an exact protocol_function that is not in the examples, use the retrieved API/rule knowledge to generate the required dynamic GetProcAddress typedef, argument construction, API call, output parsing, CSV logging, and cleanup.\n"
        "- FOCAS APIs with similar names are not ABI-compatible. Derive each resolved symbol's type from its exact official declaration in Fwlib32.h; never reuse another family member's type.\n"
        "- For cnc_rdposition, allocate a positive-count ODBPOS array and parse each requested element's abs/mach/rel/dist POSELM fields exactly as declared in Fwlib32.h. Never use a negative axis selector as data_num.\n"
        "- For cnc_absolute and cnc_distance, follow the official header declaration and pass the documented length value, not a pointer to a length.\n"
        "- Helper functions used across feed/position/status metric vectors must be type-correct. Prefer a function template such as template<class T> bool HasVariation(const std::vector<T>&) instead of a std::vector<long>-only helper called with std::vector<short>.\n"
        "- Avoid goto statements that jump over initialization of C++ objects such as std::string or std::vector. Prefer early returns, scoped cleanup helpers, or declare all such objects before any possible jump target.\n"
        "- If an API/control step is infeasible, generate a real best-effort implementation plus explicit diagnostics, or fail generation with a clear reason; do not silently skip required steps.\n"
        "- Do not emit SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN, not_executed_by_cpp_generator, placeholder skip rows, or non-fatal diagnostics in place of required executable steps.\n"
        "- The generated C++ must follow the ExecutionAgent CSV contract exactly.\n"
        "- focas_api_input.csv header must be: index,timestamp,step_id,interface_name,protocol_function,parameters,api_parameter_count\n"
        "- focas_api_output.csv header must be: index,timestamp,step_id,api_name,return_code,return_text,data,api_parameter_count\n"
        "- The same monotonically increasing integer index must be written to matching input/output rows.\n"
        "- For operation_kind=focas_sequence, expand api_calls into atomic logging and execution. Each api_calls element is one real FOCAS call: write one input row with its exact single protocol_function, execute it, and write one output row with the same exact single API name. Never write slash, plus, semicolon, arrow, or other delimiters between multiple API names in any api_name or protocol_function field. For example, cnc_dwnstart3, cnc_download3, and cnc_dwnend3 require three separate input/output row pairs.\n"
        "- A high-level interface_name such as UploadProgram may describe a sequence, but it must never collapse several real calls into one CSV record.\n"
        "- Treat tool_calls as the authoritative communication format from PlannerAgent. Each tool_calls element has one tool_name and one arguments object. Preserve call_id, operation_id, phase, tool_name, and arguments when generating code and logs. Do not reconstruct a composite tool_name from multiple elements.\n"
        "- If a previous C++ script failed, revise it based on the previous script preview, diagnostics, compile/runtime errors, and quality assessment.\n"
        "- If repair_context reports LLM timeout, empty cpp_code, or invalid executable steps, simplify: keep only required selected steps, use the scaffold directly, and still return one complete strict JSON object.\n"
        "- UploadProgram must preserve the NC program body below, but the O-number is a runtime allocation detail. Treat PlannerAgent's program_name as a preferred/base O number, then after reading the controller directory choose the actual upload O number in C++.\n"
        "- If PlannerAgent provided coverage_segments, implement them sequentially inside this one generated C++ program. Use clear segment markers in step_id/action/CSV data such as segment_id=01_programmed_coordinate_motion. A segment may have its own NC payload or no NC payload at all.\n"
        "- For coverage_segments with nc_program_required=true, use the provided NC program as the primary payload unless the segment explicitly needs a distinct safe O-number payload. If generating additional payloads, keep them short, bounded, non-destructive, and verify/upload/select/run them independently. For coverage_segments with nc_program_required=false, do not force NC upload or Cycle Start; execute direct FOCAS probe/read/write-readback-restore calls instead.\n"
        "- For each segment, distinguish target functions from support functions in CSV data fields: coverage_role=target or coverage_role=support. Support calls may repeat to create/verify/recover state; target calls are the coverage objectives.\n"
        "- Each CSV output row must represent exactly one real FOCAS/PMC API call or one host/UI helper. Do not put semicolon-separated API names such as cnc_a;cnc_b in one api_name/protocol_function field. Do not create LifecycleManifestCoverage/OffsetManifestCoverage summary rows, and do not mark coverage_role=target unless that specific function pointer was resolved and that specific API was actually called.\n"
        "- Implement real Wireshark/Npcap packet capture with a PacketSniffer class and instantiate it in main with syntax like PacketSniffer sniffer(networkDevice, pcapFile); call sniffer.Start(), sniffer.CapturePackets(...) immediately before and after each FOCAS API call, and sniffer.Stop() during cleanup. PacketSniffer must use Npcap/WinPcap APIs such as pcap_open_live, pcap_dump_open, pcap_dispatch, pcap_dump, and pcap_dump_flush, loaded dynamically if needed. Read networkDevice from the PCAP_NETWORK_DEVICE environment variable, defaulting to the calibrated Npcap loopback device \\\\Device\\\\NPF_Loopback; do not hardcode \\\\Device\\\\NPF_{CHANGE_ME}. Never create fake/minimal pcap files by writing only a pcap header. For this task, packet capture is a hard execution gate: if sniffer.Start() fails, log pcap_capture_enabled=false and PCAP_CAPTURE_FAILED, close resources, and exit nonzero before calling FOCAS APIs. On success, log pcap_capture_enabled=true or pcap_capture_started=true in focas_api_output.csv.\n"
        "- Follow the documented cnc_download3 NC payload format exactly: the first byte must be LF, followed by the runtime-selected O-number and NC blocks separated by LF, with one trailing percent character at the end. Use a payload like \"\\nO1234\\nG90...\\nM30\\n%\". Do not prepend a percent line before the leading LF and do not append data after the final percent.\n"
        "- Before cnc_dwnstart3, implement the exact program-existence step selected by PlannerAgent from RAG using that API's documented prototype and arguments. Set target_program_exists=true only when a successful returned directory/read result contains an entry whose program number exactly equals the candidate O number. Do not replace PlannerAgent's API choice with a locally preferred directory API.\n"
        "- For cnc_rdprogdir-style target existence checks, use bounded windows near the target O number and documented positive entry counts. Do not issue a broad top=1/bottom=9999 request with num=1 as the only check. A nonzero return such as EW_NUMBER/FOCAS_RETURN_2 is existence-check failure evidence, not proof that the target exists; log it clearly and repair the directory-read parameter strategy rather than declaring target_program_exists=true.\n"
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
        "Before every Cycle Start click, poll cnc_statinfo and remaining distance until the previous block is complete or a bounded timeout expires; prefer cnc_distance for remaining distance, or use cnc_rdposition(type=3)/ODBPOS.dist when cnc_distance is not available. Log waited_ms, last run/motion values, distance-to-go values, and ready/timeout in the CSV data. "
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
    user_prompt = (
        f"Task:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"Scenario goal: {state.plan.scenario_goal}\n"
        f"Target environment: {state.request.target_environment}\n"
        f"Explicit task permissions: {state.request.permissions}\n"
        f"FOCAS runtime dir: {default_focas_runtime_dir()}\n"
        f"Official FOCAS header: {default_focas_header_dir() / 'Fwlib32.h'}\n"
        f"Official ABI excerpts for Planner-selected functions:\n{official_abi_context}\n\n"
        f"NC program:\n{nc_program}\n\n"
        f"Coverage segments for this run: {state.plan.rag_context.get('coverage_segments', [])}\n"
        f"Function coverage manifest context: {state.plan.rag_context.get('function_coverage_manifest', {})}\n"
        f"Quality analysis: {state.plan.rag_context.get('planning_quality_analysis', {})}\n"
        f"Quality targets: {state.plan.rag_context.get('quality_targets', {})}\n"
        f"Repair context from previous failed attempt: {summarize_codegen_repair_context(state)}\n"
        f"CodeGenerationAgent direct RAG knowledge: {state.plan.rag_context.get('code_generation_knowledge', [])}\n"
        f"Retrieved API/rule examples for these interfaces: {summarize_codegen_rag_examples(state.plan.rag_context, executable_steps)}\n"
        f"Executable API steps:\n{steps_for_prompt(executable_steps)}\n\n"
        "Parameter generation policy:\n"
        "- fixed: use the provided constant value in every generated call.\n"
        "- enum: call the API once for each listed safe value or value combination.\n"
        "- range: call the API at representative points: min, max, and middle samples from the PlannerAgent strategy.\n"
        "- Log the actual generated input values in focas_api_input.csv parameters for each call.\n"
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
        system_prompt = (
            "You are a C++ engineer generating one complete FANUC FOCAS client for a code-only benchmark.\n"
            "Return JSON only with key cpp_code. The value must be a single compilable C++17 source file containing main().\n"
            "Use the supplied official Fwlib32.h declarations exactly. Do not invent structures, fields, signatures, or symbols.\n"
            "Use FOCAS_DLL_DIR with GetEnvironmentVariableW, SetDllDirectoryW, LoadLibraryW, and GetProcAddress.\n"
            "Connect with cnc_allclibhndl3 to the requested endpoint, execute every Planner tool call exactly once or according to its repeat, check every return code, print concise diagnostics, and release cnc_freelibhndl safely.\n"
            "For every selected API, derive its function-pointer type from the exact official declaration, construct documented arguments, and use only the official output fields.\n"
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
            f"Planner atomic tool calls:\n{steps_for_prompt(executable_steps)}\n\n"
            f"Structured CodeSpec (authoritative contract):\n{asdict(state.plan.code_spec) if state.plan.code_spec is not None else {}}\n\n"
            f"RAG API/rule evidence:\n{state.plan.rag_context.get('code_generation_knowledge', [])}\n"
            f"Retrieved examples:\n{summarize_codegen_rag_examples(state.plan.rag_context, executable_steps)}\n\n"
            "Generate the complete source now. Preserve the exact Planner API names and documented parameters."
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
        cpp_code = normalize_official_odbst_field_names(cpp_code)
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
    if struct_definitions:
        sections.append("[official referenced type definitions]\n" + "\n\n".join(struct_definitions))
    return "\n\n".join(sections)[:16000]


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

    short_t ret = api.cnc_allclibhndl3_fn
        ? api.cnc_allclibhndl3_fn("127.0.0.1", 8193, 10, &api.handle)
        : -1;
    LogInput(inputCsv, csvIndex, "CONNECT", "before", "FOCAS", "cnc_allclibhndl3", "host=127.0.0.1;port=8193;timeout=10");
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
        "struct ODBST { short hdck; short tmmode; short aut; short run; short motion; short mstb; short emergency; short alarm; short edit; };",
        "struct ODBACT { short dummy[2]; long data; };",
        "struct ODBAXIS { short dummy; short type; long data; short dec; short unit; };",
        "struct POSELM { long data; short dec; short unit; short disp; char name; char suff; };",
        "struct ODBPOS { POSELM abs; POSELM mach; POSELM rel; POSELM dist; };",
        "struct ODBPRO { short dummy[2]; short data; short mdata; };",
        "using cnc_allclibhndl3_t = short(__stdcall *)(const char *, unsigned short, long, unsigned short *);",
        "using cnc_freelibhndl_t = short(__stdcall *)(unsigned short);",
        "using cnc_dwnstart3_t = short(__stdcall *)(unsigned short, short);",
        "using cnc_download3_t = short(__stdcall *)(unsigned short, long *, char *);",
        "using cnc_dwnend3_t = short(__stdcall *)(unsigned short);",
        "using cnc_search_t = short(__stdcall *)(unsigned short, short);",
        "using cnc_rdprgnum_t = short(__stdcall *)(unsigned short, ODBPRO *);",
        "using cnc_statinfo_t = short(__stdcall *)(unsigned short, ODBST *);",
        "using cnc_actf_t = short(__stdcall *)(unsigned short, ODBACT *);",
        "using cnc_acts_t = short(__stdcall *)(unsigned short, ODBACT *);",
        "using cnc_rdposition_t = short(__stdcall *)(unsigned short, short, short *, ODBPOS *);",
        "using cnc_distance_t = short(__stdcall *)(unsigned short, short, short *, ODBAXIS *);",
        "using cnc_alarm2_t = short(__stdcall *)(unsigned short, long *);",
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
        '    case 5: return "EW_TYPE";',
        '    case 6: return "EW_DATA";',
        '    case 7: return "EW_NOOPT";',
        '    case 8: return "EW_PROT";',
        '    case 10: return "EW_PARAM";',
        '    case 13: return "EW_MODE";',
        '    case 14: return "EW_REJECT";',
        '    case 16: return "EW_ALARM";',
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
        "    string buttonText = EnvString(\"NCGUIDE_START_BUTTON_TEXT\", \"Run\");",
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
        '    const char* cncIp = "127.0.0.1";',
        "    unsigned short port = 8193;",
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
        button_text = str(step.parameters.get("button_text", "Run"))
        click_mode = str(step.parameters.get("click_mode", "screen"))
        cycle_start_x = str(step.parameters.get("cycle_start_x", "989"))
        cycle_start_y = str(step.parameters.get("cycle_start_y", "914"))
        blocks.extend(
            [
                f'        WriteInput(inputCsv, index, "{step.step_id}", "{step.phase}", "ncguide_ui_cycle_start", "{params}");',
                f'        SetEnvironmentVariableA("NCGUIDE_START_BUTTON_TEXT", "{button_text}");',
                f'        SetEnvironmentVariableA("NCGUIDE_CLICK_MODE", "{click_mode}");',
                f'        SetEnvironmentVariableA("NCGUIDE_CYCLE_START_X", "{cycle_start_x}");',
                f'        SetEnvironmentVariableA("NCGUIDE_CYCLE_START_Y", "{cycle_start_y}");',
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
                '        string data = "aut=" + to_string(status.aut) + ";run=" + to_string(status.run) + ";motion=" + to_string(status.motion) + ";alarm=" + to_string(status.alarm);',
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
        case 5: return "EW_TYPE: Data type error";
        case 6: return "EW_DATA: Data error";
        case 7: return "EW_NOOPT: No option";
        case 8: return "EW_PROT: Write protected";
        case 9: return "EW_OVRFLOW: Memory overflow";
        case 10: return "EW_PARAM: CNC parameter error";
        case 11: return "EW_BUFFER: Buffer full";
        case 12: return "EW_PATH: Path error";
        case 13: return "EW_MODE: CNC mode error";
        case 14: return "EW_REJECT: Execution rejected";
        case 15: return "EW_DTSRVR: Data server error";
        case 16: return "EW_ALARM: Alarm state";
        case 17: return "EW_STOP: Stop state";
        case 18: return "EW_RESET: Reset";
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

    string cncIp = "127.0.0.1";
    unsigned short port = 8193;
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
    diagnostics: list[str] = []
    if "cnc_allclibhndl3" not in api_script:
        diagnostics.append("C++ API script does not connect with cnc_allclibhndl3")
    if "LoadLibraryW" not in api_script or "GetProcAddress" not in api_script:
        diagnostics.append("C++ API script does not dynamically load Fwlib32.dll")
    if "FOCAS_DLL_DIR" not in api_script or "GetEnvironmentVariableW" not in api_script or "SetDllDirectoryW" not in api_script:
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
    if require_official_focas_header:
        diagnostics.extend(validate_official_focas_header_usage(api_script))
    if require_real_pcap_capture:
        diagnostics.extend(validate_real_packet_capture_usage(api_script))
    diagnostics.extend(validate_no_aggregate_or_fake_manifest_coverage(api_script))
    if "M30" not in nc_program:
        diagnostics.append("NC program does not end with M30")
    if not nc_program.startswith("O"):
        diagnostics.append("NC program does not start with a program number")
    if executable_steps:
        planned_interfaces = {step.interface_name for step in executable_steps}
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
        if "UploadProgram" in planned_interfaces and '"%\\n"' in api_script:
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
        diagnostics.extend(validate_metric_helper_types(api_script))
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
    for function_name in ["cnc_absolute", "cnc_distance"]:
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
        if re.search(rf"{function_name}\s*\([^;\n]*,\s*&\s*\w+\s*,", api_script, re.IGNORECASE):
            diagnostics.append(
                f"BLOCKING: {function_name} call passes &length; pass sizeof(ODBAXIS) or another documented length value."
            )
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
        "- For the active default Fwlib32.h, ODBPOS is an array element with POSELM abs/mach/rel/dist fields. Do not substitute the conditional alternate idata/ldata definition.\n"
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
            r"\b\w*cnc_rdposition\w*\s*\([^;\n]*,\s*(?:&\s*\w+|\w+(?:\.data\(\))?)\s*\)",
            api_script,
            re.IGNORECASE,
        )
        if "ODBPOS" not in api_script or not has_odbpos_output:
            diagnostics.append(
                "BLOCKING: cnc_rdposition must use an official ODBPOS object as its output argument."
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
