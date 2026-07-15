from __future__ import annotations

import time
import os
from pathlib import Path
from shutil import copy2

from ..agent_tools import (
    CollectExecutionArtifactsInput,
    CollectExecutionArtifactsTool,
    CompileCppInput,
    CompileGeneratedCppTool,
    RunExecutableInput,
    RunGeneratedExecutableTool,
    invoke_tool,
)
from ..agent_tools.base import ToolProgressCallback
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
from ..models import ApiCallLog, CaptureEvent, ExecutionResult, WorkflowState, utc_now
from ..tools import semantic_label_for, supported_ncguide_readonly_interfaces
from ..utils import write_jsonl


SUPPORTED_READONLY_INTERFACES = supported_ncguide_readonly_interfaces()


class ExecutionAgent:
    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        compile_tool: CompileGeneratedCppTool | None = None,
        run_tool: RunGeneratedExecutableTool | None = None,
        collect_tool: CollectExecutionArtifactsTool | None = None,
        tool_progress_callback: ToolProgressCallback | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.compile_tool = compile_tool or CompileGeneratedCppTool()
        self.run_tool = run_tool or RunGeneratedExecutableTool()
        self.collect_tool = collect_tool or CollectExecutionArtifactsTool()
        self.tool_progress_callback = tool_progress_callback

    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None or state.artifacts is None:
            raise ValueError("Cannot execute before plan and generated artifacts exist.")

        target_environment = state.request.target_environment
        if target_environment == "ncguide-generated-cpp":
            return run_generated_cpp_api_script(
                state,
                output_dir,
                self.knowledge_base,
                compile_tool=self.compile_tool,
                run_tool=self.run_tool,
                collect_tool=self.collect_tool,
                tool_progress_callback=self.tool_progress_callback,
            )

        client = make_execution_client(target_environment)
        api_logs: list[ApiCallLog] = []
        capture_events: list[CaptureEvent] = []
        execution_dir = output_dir / "execution"
        skipped_steps: list[str] = []
        skip_is_error = target_environment != "ncguide-bridge-readonly"

        for step in executable_steps(state.plan.steps, target_environment, skipped_steps):
            for _ in range(step.repeat):
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
                response = client.call(step.interface_name, step.parameters)
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
                        phase=step.phase,
                        interface_name=step.interface_name,
                        input_parameters=step.parameters,
                        status_code=status_code,
                        response=response,
                        protocol_function=step.protocol_function,
                        semantic_label=semantic_label(step.interface_name),
                        error=response.get("error"),
                    )
                )
                if step.interval_seconds:
                    time.sleep(min(step.interval_seconds, 0.05))

        errors = [log.error for log in api_logs if log.error]
        if skip_is_error:
            errors.extend(skipped_steps)
        success = not errors and all(log.status_code == 0 for log in api_logs)
        write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
        write_jsonl(execution_dir / "capture_events.jsonl", capture_events)
        if skipped_steps:
            write_jsonl(execution_dir / "skipped_steps.jsonl", [{"message": item} for item in skipped_steps])

        state.result = ExecutionResult(
            task_id=state.request.task_id,
            success=success,
            api_logs=api_logs,
            capture_events=capture_events,
            output_dir=execution_dir,
            errors=[error for error in errors if error],
        )
        state.errors.extend(state.result.errors)
        attach_execution_error_knowledge(state, self.knowledge_base)
        state.stage = "complete"
        return state


def semantic_label(interface_name: str) -> str:
    return semantic_label_for(interface_name)


def run_generated_cpp_api_script(
    state: WorkflowState,
    output_dir: Path,
    knowledge_base: KnowledgeBase | None = None,
    *,
    compile_tool: CompileGeneratedCppTool | None = None,
    run_tool: RunGeneratedExecutableTool | None = None,
    collect_tool: CollectExecutionArtifactsTool | None = None,
    tool_progress_callback: ToolProgressCallback | None = None,
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
    copy2(source_path, local_source)

    compile_tool = compile_tool or CompileGeneratedCppTool()
    run_tool = run_tool or RunGeneratedExecutableTool()
    collect_tool = collect_tool or CollectExecutionArtifactsTool()
    tool_calls = []
    compile_result, compile_trace = invoke_tool(
        compile_tool,
        CompileCppInput(local_source, exe_path, execution_dir),
        tool_progress_callback,
    )
    tool_calls.append(compile_trace)
    errors: list[str] = []
    if not compile_result.success:
        errors.append("C++ API script compilation failed.")
        write_jsonl(execution_dir / "tool_calls.jsonl", tool_calls)
        state.result = ExecutionResult(
            task_id=state.request.task_id,
            success=False,
            api_logs=[],
            capture_events=[],
            output_dir=execution_dir,
            errors=errors,
            tool_calls=tool_calls,
        )
        state.errors.extend(errors)
        attach_execution_error_knowledge(state, knowledge_base)
        state.stage = "complete"
        return state

    timeout_seconds = int(os.environ.get("SMPAGENT_CPP_TIMEOUT_SECONDS", "600"))
    run_result, run_trace = invoke_tool(
        run_tool,
        RunExecutableInput(
            executable_path=exe_path,
            work_dir=execution_dir,
            environment={"FOCAS_DLL_DIR": os.environ.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir()))},
            timeout_seconds=timeout_seconds,
        ),
        tool_progress_callback,
    )
    tool_calls.append(run_trace)
    if run_result.timed_out:
        errors.append(f"C++ API script timed out after {timeout_seconds} seconds.")
    elif run_result.exit_code != 0:
        errors.append(f"C++ API script execution failed with exit code {run_result.exit_code}.")

    collected, collect_trace = invoke_tool(
        collect_tool,
        CollectExecutionArtifactsInput(state.request.task_id, execution_dir),
        tool_progress_callback,
    )
    tool_calls.append(collect_trace)
    api_logs = collected.api_logs
    capture_events = collected.capture_events
    if any(log.status_code != 0 for log in api_logs):
        errors.extend(log.error for log in api_logs if log.error)
    success = not errors and bool(api_logs)
    write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
    write_jsonl(execution_dir / "capture_events.jsonl", capture_events)
    write_jsonl(execution_dir / "tool_calls.jsonl", tool_calls)

    state.result = ExecutionResult(
        task_id=state.request.task_id,
        success=success,
        api_logs=api_logs,
        capture_events=capture_events,
        output_dir=execution_dir,
        errors=[error for error in errors if error],
        tool_calls=tool_calls,
    )
    state.errors.extend(state.result.errors)
    attach_execution_error_knowledge(state, knowledge_base)
    state.stage = "complete"
    return state


def attach_execution_error_knowledge(state: WorkflowState, knowledge_base: KnowledgeBase | None) -> None:
    if knowledge_base is None or state.plan is None or state.result is None or not state.result.errors:
        return
    failed_logs = [
        {
            "step_id": log.step_id,
            "interface_name": log.interface_name,
            "protocol_function": log.protocol_function,
            "status_code": log.status_code,
            "error": log.error,
            "data": str(log.response.get("data", ""))[:220],
        }
        for log in state.result.api_logs
        if log.error or log.status_code != 0
    ][-8:]
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
