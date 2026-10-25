"""Strict one-agent implementation for the fair RQ3 ablation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent_tools import (
    CompileCppInput,
    CompileGeneratedCppTool,
    FocasSdkSearchInput,
    FocasSdkSearchTool,
    KnowledgeSearchInput,
    KnowledgeSearchTool,
    RenderFocasCppInput,
    RenderFocasCppTool,
    WriteGeneratedArtifactInput,
    WriteGeneratedArtifactTool,
)
from ..focas_harness import FOCAS_HARNESS_TEMPLATE_PATH, render_focas_harness_template
from ..knowledge import KnowledgeBase
from ..llm import LlmClient, parse_json_object_response
from ..models import GeneratedArtifacts, WorkflowState
from ..orchestration.roles import ActionSpec, AgentTemplate
from ..tool_runtime import ToolRegistry, ToolRuntime
from ..tool_runtime.loop import serialize_tool_output


@dataclass(slots=True)
class SingleAgentRunResult:
    state: WorkflowState
    framework_success: bool
    compilation_success: bool
    failure_category: str
    generated_file_count: int
    repair_attempts: int


class StrictSingleAgent(AgentTemplate):
    """One LLM role that owns research, planning, implementation, and repair."""

    role_name = "SingleAgentRole"
    profile = "strict fair single-agent ablation"
    goal = "Independently research, plan, implement, and compile the requested FOCAS client."
    constraints = ()

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_client: LlmClient,
        *,
        max_repairs: int = 3,
        progress_callback=None,
        tool_runtime: ToolRuntime | None = None,
    ) -> None:
        super().__init__()
        self.register_actions(ActionSpec("produce_solution", "TaskRequest", "GeneratedArtifacts"))
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client
        self.max_repairs = max(0, int(max_repairs))
        self.initial_system_prompt = self._initial_system_prompt()
        self.initial_user_prompt = ""
        self.autonomous_queries: list[dict[str, Any]] = []
        self.research_transcript: list[dict[str, Any]] = []
        self.llm_call_count = 0
        if tool_runtime is None:
            registry = ToolRegistry()
            for tool in (
                KnowledgeSearchTool(knowledge_base),
                FocasSdkSearchTool(),
                WriteGeneratedArtifactTool(),
                RenderFocasCppTool(),
                CompileGeneratedCppTool(),
            ):
                registry.register(tool)
            tool_runtime = ToolRuntime(registry, progress_callback=progress_callback)
        self.tool_runtime = tool_runtime
        self.progress_callback = progress_callback or tool_runtime.progress_callback

    def run(self, state: WorkflowState, output_dir: Path) -> SingleAgentRunResult:
        if state.long_term_memories:
            raise ValueError("Strict Single-Agent must start without historical memory.")
        if not self.llm_client.enabled:
            raise RuntimeError("Strict Single-Agent requires an enabled LLM.")

        self._emit("single_agent_phase", message="starting clean single-agent run")
        self.initial_user_prompt = state.request.description
        payload = self._research_then_generate(self.initial_user_prompt)
        generated_dir = output_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        repair_attempts = 0
        last_compile = None

        while True:
            selected_functions, fragment, nc_program = self._parse_solution(payload)
            self._emit(
                "single_agent_solution",
                selected_function_count=len(selected_functions),
                has_nc_program=bool(nc_program.strip()),
                repair_attempt=repair_attempts,
            )
            harness = render_focas_harness_template(selected_functions)
            harness_path = generated_dir / "focas_harness.cpp"
            fragment_path = generated_dir / "scenario_fragment.inc"
            source_path = generated_dir / "api_script.cpp"
            nc_path = generated_dir / "program.nc"
            executable_path = generated_dir / "api_script.exe"

            self._write(harness_path, harness)
            self._write(fragment_path, fragment)
            if nc_program.strip():
                self._write(nc_path, nc_program)
                persisted_nc_path: Path | None = nc_path
            else:
                if nc_path.exists():
                    nc_path.unlink()
                persisted_nc_path = None
            self.tool_runtime.call(
                self.role_name,
                "render_focas_cpp",
                RenderFocasCppInput(harness_path, fragment_path, source_path),
            )
            last_compile, _record = self.tool_runtime.call(
                self.role_name,
                "compile_generated_cpp",
                CompileCppInput(source_path, executable_path, generated_dir),
            )
            self._emit(
                "single_agent_compile",
                success=last_compile.success,
                repair_attempt=repair_attempts,
                executable_path=str(executable_path),
            )
            diagnostics = []
            if not last_compile.success:
                diagnostics.append(self._raw_compiler_feedback(last_compile.stdout, last_compile.stderr))
            state.artifacts = GeneratedArtifacts(
                api_script=source_path.read_text(encoding="utf-8", errors="replace"),
                nc_program=nc_program,
                api_script_path=source_path,
                nc_program_path=persisted_nc_path,
                diagnostics=diagnostics,
                harness_path=harness_path,
                fragment_path=fragment_path,
            )
            state.tool_call_history = list(self.tool_runtime.history)
            state.repair_attempts = repair_attempts
            state.stage = "execution"
            if last_compile.success:
                self._emit(
                    "single_agent_done",
                    success=True,
                    repair_attempts=repair_attempts,
                    llm_calls=self.llm_call_count,
                )
                return self._result(state, True, "success", repair_attempts)
            if repair_attempts >= self.max_repairs:
                state.errors.append("MSVC compilation failed after the configured Single-Agent repair budget.")
                self._emit(
                    "single_agent_done",
                    success=False,
                    repair_attempts=repair_attempts,
                    llm_calls=self.llm_call_count,
                )
                return self._result(state, False, "compile_error", repair_attempts)

            repair_attempts += 1
            self._emit(
                "single_agent_repair",
                attempt=repair_attempts,
                max_repairs=self.max_repairs,
            )
            payload = self._repair_with_raw_compiler_output(
                original_task=state.request.description,
                current_payload=payload,
                stdout=last_compile.stdout,
                stderr=last_compile.stderr,
                repair_attempt=repair_attempts,
            )

    def _research_then_generate(self, original_task: str) -> dict[str, Any]:
        transcript = original_task
        used_knowledge = False
        used_sdk = False
        for turn in range(8):
            payload = self._invoke_payload(transcript, phase=f"research/generation turn {turn + 1}")
            calls = self._extract_calls(payload)
            if calls:
                results = self._execute_agent_queries(calls)
                used_knowledge = used_knowledge or any(
                    row["tool_name"] == "search_knowledge" and row["success"] for row in results
                )
                used_sdk = used_sdk or any(
                    row["tool_name"] == "search_focas_sdk" and row["success"] for row in results
                )
                self.research_transcript.extend(results)
                transcript = self._transcript(original_task)
                continue
            if used_knowledge and used_sdk:
                return payload
            transcript = (
                self._transcript(original_task)
                + "\n\nYou have not yet used both available read-only research tools. "
                "Choose your own queries and return tool_calls before producing the solution."
            )
        raise RuntimeError("Strict Single-Agent did not finish autonomous research and generation within 8 turns.")

    def _repair_with_raw_compiler_output(
        self,
        *,
        original_task: str,
        current_payload: dict[str, Any],
        stdout: str,
        stderr: str,
        repair_attempt: int,
    ) -> dict[str, Any]:
        prompt = (
            f"Original Scenario Task:\n{original_task}\n\n"
            f"Your own research results from this run:\n{self.research_transcript}\n\n"
            f"Your current generated solution:\n{current_payload}\n\n"
            f"Raw MSVC stdout:\n{stdout}\n\n"
            f"Raw MSVC stderr:\n{stderr}\n\n"
            f"Repair attempt {repair_attempt}/{self.max_repairs}. "
            "Diagnose and repair your own solution. You may issue additional tool_calls using queries you choose, "
            "or return the complete replacement solution JSON. No external diagnosis is provided."
        )
        for turn in range(6):
            payload = self._invoke_payload(prompt, phase=f"repair {repair_attempt} turn {turn + 1}")
            calls = self._extract_calls(payload)
            if not calls:
                return payload
            self.research_transcript.extend(self._execute_agent_queries(calls))
            prompt = (
                f"{prompt}\n\nAdditional research results:\n{self.research_transcript}\n\n"
                "Return more tool_calls or the complete replacement solution JSON."
            )
        raise RuntimeError("Strict Single-Agent repair exceeded its research turn budget.")

    def _execute_agent_queries(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls:
            tool_name = str(call.get("tool_name", call.get("name", ""))).strip()
            arguments = call.get("arguments", call.get("input", {}))
            if not isinstance(arguments, dict):
                results.append(self._rejected_query(tool_name, "Tool arguments must be a JSON object."))
                continue
            query = str(arguments.get("query", ""))
            if not query.strip():
                results.append(self._rejected_query(tool_name, "Research query must be a non-empty string."))
                continue
            if tool_name == "search_knowledge":
                tool_input = KnowledgeSearchInput(
                    query=query,
                    top_k=int(arguments.get("top_k", 8)),
                    filters=dict(arguments.get("filters", {}) or {}),
                )
            elif tool_name == "search_focas_sdk":
                tool_input = FocasSdkSearchInput(
                    query=query,
                    top_k=int(arguments.get("top_k", 8)),
                )
            else:
                results.append(
                    self._rejected_query(tool_name, f"Unsupported research tool: {tool_name or '<missing>'}.")
                )
                continue
            self._emit(
                "single_agent_query",
                tool_name=tool_name,
                query=query[:180],
            )
            output, record = self.tool_runtime.call(self.role_name, tool_name, tool_input)
            self.autonomous_queries.append({"tool_name": tool_name, "query": query})
            results.append({
                "tool_name": tool_name,
                "query": query,
                "success": record.success,
                "output": serialize_tool_output(output),
            })
        return results

    def _rejected_query(self, tool_name: str, error: str) -> dict[str, Any]:
        self._emit(
            "single_agent_query_rejected",
            tool_name=tool_name or "<missing>",
            error=error,
        )
        return {
            "tool_name": tool_name,
            "query": "",
            "success": False,
            "output": {"error": error},
        }

    def _invoke_payload(self, user_prompt: str, *, phase: str) -> dict[str, Any]:
        """Parse JSON without switching to a separate JSON-repair persona."""

        self.llm_call_count += 1
        self._emit("single_agent_llm", status="waiting", phase=phase, call=self.llm_call_count)
        raw = self.llm_client.invoke_text(self.initial_system_prompt, user_prompt)
        self._emit("single_agent_llm", status="received", phase=phase, call=self.llm_call_count)
        try:
            return parse_json_object_response(raw)
        except ValueError:
            correction = (
                f"{user_prompt}\n\nYour previous response was not valid JSON:\n{raw}\n\n"
                "As the same SingleAgent, return the intended tool_calls or complete solution as one valid JSON object."
            )
            self.llm_call_count += 1
            self._emit(
                "single_agent_llm",
                status="waiting for JSON correction",
                phase=phase,
                call=self.llm_call_count,
            )
            repaired = self.llm_client.invoke_text(self.initial_system_prompt, correction)
            self._emit(
                "single_agent_llm",
                status="received JSON correction",
                phase=phase,
                call=self.llm_call_count,
            )
            return parse_json_object_response(repaired)

    def _emit(self, event: str, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, payload)

    def _transcript(self, original_task: str) -> str:
        return (
            f"Original Scenario Task:\n{original_task}\n\n"
            f"Results of the research queries you chose in this run:\n{self.research_transcript}\n\n"
            "Choose more research queries if necessary, or return the complete solution JSON."
        )

    def _write(self, path: Path, content: str) -> None:
        self.tool_runtime.call(
            self.role_name,
            "write_generated_artifact",
            WriteGeneratedArtifactInput(path, content),
        )

    @staticmethod
    def _extract_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
        calls = payload.get("tool_calls")
        if isinstance(calls, list):
            return [row for row in calls if isinstance(row, dict)]
        if payload.get("tool_name") or payload.get("name"):
            return [payload]
        return []

    @staticmethod
    def _parse_solution(payload: dict[str, Any]) -> tuple[list[str], str, str]:
        functions = payload.get("selected_functions")
        fragment = str(payload.get("scenario_fragment_cpp", "")).strip()
        nc_program = str(payload.get("nc_program", "")).strip()
        if not isinstance(functions, list) or not all(isinstance(item, str) for item in functions):
            raise ValueError("Single-Agent solution must contain selected_functions as a string list.")
        if not fragment:
            raise ValueError("Single-Agent solution did not contain scenario_fragment_cpp.")
        return list(dict.fromkeys(functions)), fragment, nc_program

    @staticmethod
    def _raw_compiler_feedback(stdout: str, stderr: str) -> str:
        return f"Raw MSVC stdout:\n{stdout}\nRaw MSVC stderr:\n{stderr}"

    @staticmethod
    def _result(
        state: WorkflowState,
        success: bool,
        failure_category: str,
        repair_attempts: int,
    ) -> SingleAgentRunResult:
        generated_dir = state.artifacts.api_script_path.parent if state.artifacts and state.artifacts.api_script_path else None
        generated_count = len(list(generated_dir.iterdir())) if generated_dir and generated_dir.exists() else 0
        return SingleAgentRunResult(
            state=state,
            framework_success=success,
            compilation_success=success,
            failure_category=failure_category,
            generated_file_count=generated_count,
            repair_attempts=repair_attempts,
        )

    @staticmethod
    def _initial_system_prompt() -> str:
        return (
            "You are the only agent in a fair code-only FANUC FOCAS experiment. Independently understand the "
            "original task, decide what to research, select all task-relevant APIs, decide their order, parameters "
            "and timing, create any required NC content, and implement the solution. No planner, router, CodeSpec, "
            "preselected API list, predefined sequence, task-specific parameter plan, or historical memory is available.\n\n"
            "You have two read-only research tools. While researching return JSON with tool_calls, where each call is "
            "{\"tool_name\":\"search_knowledge|search_focas_sdk\",\"arguments\":{\"query\":\"your query\",\"top_k\":8}}. "
            "You must choose every query yourself and use both tools at least once.\n\n"
            "The shared immutable C++17 harness owns includes, dynamic DLL loading, controller connection, cleanup, "
            "return-code helper, main(), and a call to your scenario fragment. The renderer creates FocasApi pointer "
            "members solely from selected_functions. Do not reproduce or modify the harness. Your fragment must be a "
            "complete function with signature short run_generated_scenario(FocasApi& api, unsigned short handle). "
            "Use official SDK types and call selected members as api.<function>.\n\n"
            "When research is sufficient, return JSON only with exactly these solution fields: "
            "{\"selected_functions\":[\"functions you independently selected\"],"
            "\"scenario_fragment_cpp\":\"complete fragment function\","
            "\"nc_program\":\"self-generated NC text, or empty when unnecessary\"}."
        )


def shared_single_agent_harness_path() -> Path:
    """Testable declaration of the exact shared template source."""

    return FOCAS_HARNESS_TEMPLATE_PATH
