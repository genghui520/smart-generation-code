from __future__ import annotations

from pathlib import Path
from typing import Any

from ..integrations.ncguide import default_focas_header_dir, default_focas_runtime_dir
from ..knowledge import KnowledgeBase
from ..llm import LlmClient
from ..models import GeneratedArtifacts, NcProgramSpec, PlanStep, WorkflowState
from ..utils import ensure_dir
from .prompts import CODE_REVIEW_JSON_SCHEMA, CODE_REVIEW_SYSTEM_PROMPT, FOCAS_CPP_GENERATION_SYSTEM_PROMPT


class CodeGenerationAgent:
    def __init__(self, llm_client: LlmClient | None = None, knowledge_base: KnowledgeBase | None = None) -> None:
        self.llm_client = llm_client or LlmClient()
        self.knowledge_base = knowledge_base

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
        max_compile_repair_attempts = 1
        for compile_attempt in range(max_compile_repair_attempts + 1):
            api_script_path.write_text(api_script, encoding="utf-8")
            diagnostics = validate_generated(
                api_script,
                nc_program,
                executable_steps,
                allow_delete_all_programs=bool(state.request.permissions.get("allow_delete_all_programs")),
                require_official_focas_header=True,
            )
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
            if compile_attempt >= max_compile_repair_attempts or not self.llm_client.enabled:
                raise ValueError("Generated C++ failed MSVC preflight compilation: " + compile_error)
            api_script = repair_cpp_api_script_after_compile_error(
                state,
                executable_steps,
                nc_program,
                api_script,
                compile_error,
                self.llm_client,
            )
            cpp_generation_diagnostics.append("CodeGenerationAgent regenerated C++ from MSVC preflight compiler diagnostics.")
        if self.llm_client.enabled:
            diagnostics.extend(
                review_generated_with_llm(
                    self.llm_client,
                    state.request.description,
                    state.plan.scenario_type,
                    nc_program,
                    api_script,
                )
            )
        state.artifacts.diagnostics = diagnostics
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
    return [
        {
            "step_id": step.step_id,
            "phase": step.phase,
            "action": step.action,
            "interface_name": step.interface_name,
            "parameters": step.parameters,
            "repeat": step.repeat,
            "interval_seconds": step.interval_seconds,
            "expected_state": step.expected_state,
            "protocol_function": step.protocol_function,
        }
        for step in steps
    ]


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
    if not llm_client.enabled:
        raise RuntimeError("CodeGenerationAgent requires an LLM C++ generation in agent-only mode.")

    script = generate_cpp_api_script_with_llm(state, executable_steps, nc_program, llm_client)
    if not script:
        raise ValueError("CodeGenerationAgent LLM returned an empty C++ API script.")
    return script, ["LLM generated C++ FOCAS API script from PlannerAgent plan and CodeGenerationAgent steps."]


def repair_cpp_api_script_after_compile_error(
    state: WorkflowState,
    executable_steps: list[PlanStep],
    nc_program: str,
    api_script: str,
    compile_error: str,
    llm_client: LlmClient,
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
    user_prompt = (
        f"Task description:\n{state.request.description}\n\n"
        f"Scenario: {state.plan.scenario_type}\n"
        f"NC program, must preserve unless impossible:\n{nc_program}\n\n"
        f"Executable API steps:\n{steps_for_prompt(executable_steps)}\n\n"
        f"MSVC preflight compiler diagnostics:\n{compile_error}\n\n"
        "Previous C++ source to repair:\n"
        f"{api_script}"
    )
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    repaired = str(payload.get("cpp_code", "")).strip()
    if not repaired:
        repaired = str(payload.get("api_script", "")).strip()
    repaired = strip_cpp_fence(repaired)
    if not repaired:
        raise ValueError("CodeGenerationAgent compile-repair LLM returned an empty C++ API script.")
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
        "- For cnc_rdposition, allocate official ODBPOS elements for a positive data_num axis count and parse the official abs/mach/rel/dist fields. Never use a negative axis selector as data_num.\n"
        "- For cnc_absolute and cnc_distance, follow the official header declaration and pass the documented length value, not a pointer to a length.\n"
        "- Helper functions used across feed/position/status metric vectors must be type-correct. Prefer a function template such as template<class T> bool HasVariation(const std::vector<T>&) instead of a std::vector<long>-only helper called with std::vector<short>.\n"
        "- Avoid goto statements that jump over initialization of C++ objects such as std::string or std::vector. Prefer early returns, scoped cleanup helpers, or declare all such objects before any possible jump target.\n"
        "- If an API/control step is infeasible, generate a real best-effort implementation plus explicit diagnostics, or fail generation with a clear reason; do not silently skip required steps.\n"
        "- Do not emit SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN, not_executed_by_cpp_generator, placeholder skip rows, or non-fatal diagnostics in place of required executable steps.\n"
        "- The generated C++ must follow the ExecutionAgent CSV contract exactly.\n"
        "- focas_api_input.csv header must be: index,timestamp,step_id,phase,interface_name,protocol_function,parameters\n"
        "- focas_api_output.csv header must be: index,timestamp,step_id,api_name,return_code,return_text,data\n"
        "- The same monotonically increasing integer index must be written to matching input/output rows.\n"
        "- If a previous C++ script failed, revise it based on the previous script preview, diagnostics, compile/runtime errors, and quality assessment.\n"
        "- If repair_context reports LLM timeout, empty cpp_code, or invalid executable steps, simplify: keep only required selected steps, use the scaffold directly, and still return one complete strict JSON object.\n"
        "- UploadProgram must upload the exact NC program payload provided below.\n"
        "- Follow the documented cnc_download3 NC payload format exactly: the first byte must be LF, followed by O-number and NC blocks separated by LF, with one trailing percent character at the end. Use a payload like \"\\nO1234\\nG90...\\nM30\\n%\". Do not prepend a percent line before the leading LF and do not append data after the final percent.\n"
        "- Before cnc_dwnstart3, implement the exact program-existence step selected by PlannerAgent from RAG using that API's documented prototype and arguments. Set target_program_exists=true only when a successful returned directory/read result contains an entry whose program number exactly equals the expected O number. Do not replace PlannerAgent's API choice with a locally preferred directory API.\n"
        "- For cnc_rdprogdir-style target existence checks, use bounded windows near the target O number and documented positive entry counts. Do not issue a broad top=1/bottom=9999 request with num=1 as the only check. A nonzero return such as EW_NUMBER/FOCAS_RETURN_2 is existence-check failure evidence, not proof that the target exists; log it clearly and repair the directory-read parameter strategy rather than declaring target_program_exists=true.\n"
        "- The default collision policy is non-destructive. If target_program_exists=true, do not call cnc_delete and do not upload over it. Log TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED, program_number_available=false, and exit before cnc_dwnstart3 so RouterAgent can repair_plan with a different O number. If absent, log target_program_exists=false;program_number_available=true and continue upload.\n"
        "- Do not generate cnc_delete as part of the normal UploadProgram collision path. Existing CNC programs must be preserved unless the user task explicitly authorizes deleting a particular program.\n"
        "- cnc_delall is permission-gated. Generate it only when Explicit task permissions contains allow_delete_all_programs=true and PlannerAgent explicitly selected cnc_delall from RAG reasoning. Otherwise its presence is a blocking safety violation. Even when authorized, log delete_all_authorized=true and abort if the call fails.\n"
        "- Any existence-check error other than a documented empty result must abort before cnc_dwnstart3. A confirmed collision must request replanning rather than deletion.\n"
        "- cnc_dwnend3 may report delayed cnc_download3 errors. If cnc_getdtailerr reports err_no=4 despite the verified replacement flow, abort and report that the target O number still exists; do not retry blindly. If err_no=5, the same program is selected and execution must stop for a safe re-selection/repair.\n"
        "- SelectProgram must select the same O program number as the uploaded NC program.\n"
        "- ReadProgramNumber must call cnc_rdprgnum and log the active/main program numbers.\n"
        "- Program lifecycle calls are hard gates. Check cnc_dwnstart3, every cnc_download3 call, cnc_dwnend3, and cnc_search return codes. Then call cnc_rdprgnum and compare the returned current/main program number with the uploaded O number. Log program_verified=true only on an exact match. On any lifecycle failure or mismatch, log PROGRAM_NOT_VERIFIED and exit nonzero before the first Cycle Start. Never log selected_program=<expected> as if selection succeeded when cnc_search returned an error.\n"
        "- StartProgram may use NCGuide UI Cycle Start with screen/client click parameters from the step, but it must not blindly click while the previous single-block segment is still running. "
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
        "- Packet capture support should be best-effort and must not block FOCAS API calls.\n"
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
    payload = llm_client.invoke_json(system_prompt, user_prompt)
    cpp_code = str(payload.get("cpp_code", "")).strip()
    if not cpp_code:
        cpp_code = str(payload.get("api_script", "")).strip()
    return strip_cpp_fence(cpp_code)


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
    names: list[str] = []
    for part in protocol_function.replace(",", "/").split("/"):
        name = part.strip()
        if name and name.startswith("cnc_"):
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
        blocks[alias.upper()] = definition
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
    csv << index << "," << CsvEscape(timestamp) << "," << CsvEscape(stepId) << "," << CsvEscape(phase) << ","
        << CsvEscape(interfaceName) << "," << CsvEscape(protocolFunction) << ","
        << CsvEscape(parameters) << "\n";
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
    inputCsv << "index,timestamp,step_id,phase,interface_name,protocol_function,parameters\n";
    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data\n";
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
        "class PacketCapture {",
        "public:",
        "    bool Start(const string& outputPath) {",
        '        dll_ = LoadLibraryW(L"wpcap.dll");',
        "        if (!dll_) { cout << \"Npcap wpcap.dll not found; pcap capture disabled.\" << endl; return false; }",
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
        "            cout << \"Npcap exports are incomplete; pcap capture disabled.\" << endl;",
        "            return false;",
        "        }",
        "        char errbuf[512] = {};",
        "        pcap_if_t* devices = nullptr;",
        "        if (findalldevs_(&devices, errbuf) != 0 || !devices) { cout << \"pcap_findalldevs failed: \" << errbuf << endl; return false; }",
        "        string selected;",
        "        for (pcap_if_t* dev = devices; dev; dev = dev->next) {",
        "            string name = dev->name ? dev->name : \"\";",
        "            string desc = dev->description ? dev->description : \"\";",
        "            string lower = name + \" \" + desc;",
        "            for (char& c : lower) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));",
        "            if (lower.find(\"loopback\") != string::npos || lower.find(\"npcap\") != string::npos) { selected = name; break; }",
        "        }",
        "        if (selected.empty() && devices->name) selected = devices->name;",
        "        freealldevs_(devices);",
        "        if (selected.empty()) { cout << \"No Npcap adapter found; pcap capture disabled.\" << endl; return false; }",
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
        '    PacketCapture packetCapture;',
        '    bool pcapEnabled = packetCapture.Start("data\\\\focas_api_capture.pcap");',
        '    ofstream inputCsv("data\\\\focas_api_input.csv", ios::binary);',
        '    ofstream outputCsv("data\\\\focas_api_output.csv", ios::binary);',
        "    inputCsv.put(0xEF); inputCsv.put(0xBB); inputCsv.put(0xBF);",
        "    outputCsv.put(0xEF); outputCsv.put(0xBB); outputCsv.put(0xBF);",
        '    inputCsv << "index,timestamp,step_id,phase,api_name,parameters\\n";',
        '    outputCsv << "index,timestamp,step_id,api_name,return_code,return_text,data\\n";',
        "    int index = 0;",
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
    string networkDevice = "\\\\Device\\\\NPF_{{CHANGE_ME}}";
    string methodName = "{primary_method}";
    vector<string> plannedInterfaces = {{{", ".join(f'"{item}"' for item in interfaces)}}};

    CreateDirectoryA("data", NULL);
    string pcapFile = "data\\\\Fanuc_" + methodName + ".pcap";
    string inputCsvFile = "data\\\\Fanuc_" + methodName + "_input.csv";
    string outputCsvFile = "data\\\\Fanuc_" + methodName + "_output.csv";

    PacketSniffer sniffer(networkDevice, pcapFile);
    if (!sniffer.Start()) {{
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
        if "UploadProgram" in planned_interfaces and not uses_delete_all and not has_target_program_availability_flow(api_script):
            diagnostics.append(
                "BLOCKING: NCGuide upload must use the Planner-selected existence API, upload only when "
                "program_number_available=true, and emit TARGET_PROGRAM_EXISTS_REPLAN_REQUIRED without deleting on collision."
            )
        if "UploadProgram" in planned_interfaces and "cnc_delete" in api_script.lower():
            diagnostics.append(
                "BLOCKING: default UploadProgram collision handling must preserve existing programs and replan to a "
                "different O number instead of calling cnc_delete."
            )
        if "StartProgram" in planned_interfaces and not has_cycle_start_ready_gate(api_script):
            diagnostics.append(
                "BLOCKING: planned StartProgram/Cycle Start steps must include a cycle_start_ready_gate or "
                "WaitUntilCycleStartReady helper that polls cnc_statinfo before clicking again."
            )
        if "StartProgram" in planned_interfaces and not has_program_completion_gate(api_script):
            diagnostics.append(
                "BLOCKING: planned StartProgram/Cycle Start workflow must include a final program_completion_gate or "
                "WaitUntilProgramComplete helper before evaluation/disconnect."
            )
        if "StartProgram" in planned_interfaces and not has_program_completion_wait_logic(api_script):
            diagnostics.append(
                "BLOCKING: program_completion_gate must poll cnc_statinfo until idle completion and log completed/timeout, "
                "waited_ms, last_run, and last_motion before evaluation/disconnect."
            )
        if "StartProgram" in planned_interfaces and not has_whole_program_single_block_loop(api_script):
            diagnostics.append(
                "BLOCKING: Single Block execution must count and drive every effective NC segment, including the "
                "O-number line and M30, and log expected_nc_segment_count plus cycle_start_click_count."
            )
        lifecycle_interfaces = {"UploadProgram", "SelectProgram", "ReadProgramNumber", "StartProgram"}
        if lifecycle_interfaces.issubset(planned_interfaces) and not has_program_verification_gate(api_script):
            diagnostics.append(
                "BLOCKING: uploaded-program execution must hard-gate Cycle Start on program_verified and emit "
                "PROGRAM_NOT_VERIFIED when upload/select fails or cnc_rdprgnum does not match the expected O number."
            )
        if ({"ReadPosition", "ReadDistanceToGo"} & planned_interfaces) and not logs_distance_to_go(api_script):
            diagnostics.append(
                "BLOCKING: coordinate sampling must log remaining movement using cnc_distance or "
                "cnc_rdposition(type=3)/ODBPOS.dist as distance-to-go data."
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
            "a reduced POS_ITEM array can overwrite stack memory."
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
        "target_program_exists_replan_required",
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
    return (
        has_existence_read
        and has_exact_match
        and "cnc_delall" not in lowered
        and "cnc_delete" not in lowered
    )


def has_authorized_delete_all_gate(api_script: str) -> bool:
    lowered = api_script.lower()
    if "cnc_delall" not in lowered or "delete_all_authorized=true" not in lowered:
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
) -> list[str]:
    system_prompt = CODE_REVIEW_SYSTEM_PROMPT
    script_preview = (
        "BEGINNING OF C++ SCRIPT:\n"
        f"{api_script[:5000]}\n\n"
        "END OF C++ SCRIPT:\n"
        f"{api_script[-5000:]}"
    )
    user_prompt = (
        f"C++ generation policy:\n{FOCAS_CPP_GENERATION_SYSTEM_PROMPT}\n\n"
        "Verified NCGuide/FOCAS facts for this review:\n"
        "- cnc_download3 NC program data starts with LF and ends with one percent character; it does not start with a percent line.\n"
        "- cnc_getdtailerr err_no=4 after download means the same O number is already registered.\n"
        "- In this confirmed Single Block setup, the O-number line and M30 each consume a Cycle Start, in addition to modal and motion lines. Include them when reviewing expected_nc_segment_count.\n\n"
        f"Task: {task_description}\n"
        f"Scenario: {scenario}\n\n"
        f"NC program:\n{nc_program}\n\n"
        f"C++ FOCAS API script length: {len(api_script)} characters\n"
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
