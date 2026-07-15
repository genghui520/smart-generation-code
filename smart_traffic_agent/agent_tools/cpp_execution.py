from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..integrations.ncguide import default_focas_header_dir
from ..models import ApiCallLog, CaptureEvent, utc_now
from ..tools import semantic_label_for
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class CompileCppInput:
    source_path: Path
    executable_path: Path
    work_dir: Path
    timeout_seconds: int = 120


@dataclass(slots=True)
class CompileCppOutput:
    return_code: int
    stdout: str
    stderr: str
    executable_path: Path
    command: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.return_code == 0 and self.executable_path.exists()


class CompileGeneratedCppTool(AgentTool[CompileCppInput, CompileCppOutput]):
    name = "compile_generated_cpp"
    description = "Compile a generated C++17 FOCAS program with the 32-bit MSVC toolchain."

    def invoke(self, tool_input: CompileCppInput) -> CompileCppOutput:
        tool_input.work_dir.mkdir(parents=True, exist_ok=True)
        toolchain = find_msvc_x86_toolchain()
        if toolchain is None:
            stderr = "Visual C++ x86 toolchain was not found."
            write_compile_logs(tool_input.work_dir, "", stderr)
            return CompileCppOutput(1, "", stderr, tool_input.executable_path)

        cl_exe, env = toolchain
        command = [
            str(cl_exe),
            "/nologo",
            "/EHsc",
            "/std:c++17",
            "/utf-8",
            "/wd4828",
            tool_input.source_path.name,
            f"/Fe:{tool_input.executable_path.name}",
            "User32.lib",
        ]
        result = subprocess.run(
            command,
            cwd=tool_input.work_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=tool_input.timeout_seconds,
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        write_compile_logs(tool_input.work_dir, stdout, stderr)
        return CompileCppOutput(
            return_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            executable_path=tool_input.executable_path,
            command=command,
        )

    def input_summary(self, tool_input: CompileCppInput) -> dict[str, Any]:
        return {
            "source_path": str(tool_input.source_path),
            "executable_path": str(tool_input.executable_path),
            "work_dir": str(tool_input.work_dir),
        }

    def output_summary(self, tool_output: CompileCppOutput) -> dict[str, Any]:
        return {
            "return_code": tool_output.return_code,
            "executable_path": str(tool_output.executable_path),
            "stdout_chars": len(tool_output.stdout),
            "stderr_chars": len(tool_output.stderr),
        }

    def output_succeeded(self, tool_output: CompileCppOutput) -> bool:
        return tool_output.success

    def output_error(self, tool_output: CompileCppOutput) -> str | None:
        if tool_output.success:
            return None
        details = (tool_output.stderr or tool_output.stdout).strip()
        return details[-1000:] or f"compiler exit code {tool_output.return_code}"


@dataclass(frozen=True, slots=True)
class RunExecutableInput:
    executable_path: Path
    work_dir: Path
    environment: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 600


@dataclass(slots=True)
class RunExecutableOutput:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    timeout_seconds: int

    @property
    def success(self) -> bool:
        return not self.timed_out and self.exit_code == 0


class RunGeneratedExecutableTool(AgentTool[RunExecutableInput, RunExecutableOutput]):
    name = "run_generated_executable"
    description = "Run a compiled generated executable with bounded timeout and captured output."

    def invoke(self, tool_input: RunExecutableInput) -> RunExecutableOutput:
        run_env = os.environ.copy()
        run_env.update(tool_input.environment)
        try:
            result = subprocess.run(
                [str(tool_input.executable_path)],
                cwd=tool_input.work_dir,
                env=run_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=tool_input.timeout_seconds,
            )
            output = RunExecutableOutput(
                exit_code=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                timed_out=False,
                timeout_seconds=tool_input.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output = RunExecutableOutput(
                exit_code=None,
                stdout=decode_process_output(exc.stdout),
                stderr=decode_process_output(exc.stderr),
                timed_out=True,
                timeout_seconds=tool_input.timeout_seconds,
            )
        (tool_input.work_dir / "api_script_stdout.txt").write_text(output.stdout, encoding="utf-8")
        (tool_input.work_dir / "api_script_stderr.txt").write_text(output.stderr, encoding="utf-8")
        return output

    def input_summary(self, tool_input: RunExecutableInput) -> dict[str, Any]:
        return {
            "executable_path": str(tool_input.executable_path),
            "work_dir": str(tool_input.work_dir),
            "timeout_seconds": tool_input.timeout_seconds,
            "environment_keys": sorted(tool_input.environment),
        }

    def output_summary(self, tool_output: RunExecutableOutput) -> dict[str, Any]:
        return {
            "exit_code": tool_output.exit_code,
            "timed_out": tool_output.timed_out,
            "stdout_chars": len(tool_output.stdout),
            "stderr_chars": len(tool_output.stderr),
        }

    def output_succeeded(self, tool_output: RunExecutableOutput) -> bool:
        return tool_output.success

    def output_error(self, tool_output: RunExecutableOutput) -> str | None:
        if tool_output.timed_out:
            return f"execution timed out after {tool_output.timeout_seconds} seconds"
        if tool_output.exit_code != 0:
            return f"execution failed with exit code {tool_output.exit_code}"
        return None


@dataclass(frozen=True, slots=True)
class CollectExecutionArtifactsInput:
    task_id: str
    execution_dir: Path


@dataclass(slots=True)
class CollectExecutionArtifactsOutput:
    api_logs: list[ApiCallLog]
    capture_events: list[CaptureEvent]
    schema_errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return bool(self.api_logs) and not self.schema_errors


class CollectExecutionArtifactsTool(
    AgentTool[CollectExecutionArtifactsInput, CollectExecutionArtifactsOutput]
):
    name = "collect_execution_artifacts"
    description = "Parse generated C++ CSV artifacts into structured API logs and capture events."

    def invoke(self, tool_input: CollectExecutionArtifactsInput) -> CollectExecutionArtifactsOutput:
        return read_cpp_csv_logs(tool_input.task_id, tool_input.execution_dir)

    def input_summary(self, tool_input: CollectExecutionArtifactsInput) -> dict[str, Any]:
        return {"task_id": tool_input.task_id, "execution_dir": str(tool_input.execution_dir)}

    def output_summary(self, tool_output: CollectExecutionArtifactsOutput) -> dict[str, Any]:
        return {
            "api_log_count": len(tool_output.api_logs),
            "capture_event_count": len(tool_output.capture_events),
            "schema_error_count": len(tool_output.schema_errors),
        }

    def output_succeeded(self, tool_output: CollectExecutionArtifactsOutput) -> bool:
        return tool_output.success

    def output_error(self, tool_output: CollectExecutionArtifactsOutput) -> str | None:
        if tool_output.schema_errors:
            return "; ".join(tool_output.schema_errors)
        if not tool_output.api_logs:
            return "generated execution produced no readable API logs"
        return None


def compile_cpp_script(source_path: Path, exe_path: Path, work_dir: Path) -> subprocess.CompletedProcess[str]:
    """Compatibility wrapper; new Agent code should invoke CompileGeneratedCppTool."""

    output = CompileGeneratedCppTool().invoke(CompileCppInput(source_path, exe_path, work_dir))
    return subprocess.CompletedProcess(
        args=output.command,
        returncode=output.return_code,
        stdout=output.stdout,
        stderr=output.stderr,
    )


def find_msvc_x86_toolchain() -> tuple[Path, dict[str, str]] | None:
    msvc_root = latest_existing_dir(
        [
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"),
        ]
    )
    sdk_include_root = latest_existing_dir([Path(r"C:\Program Files (x86)\Windows Kits\10\Include")])
    sdk_lib_root = latest_existing_dir([Path(r"C:\Program Files (x86)\Windows Kits\10\Lib")])
    if msvc_root is None or sdk_include_root is None or sdk_lib_root is None:
        return None
    cl_exe = msvc_root / "bin" / "Hostx64" / "x86" / "cl.exe"
    if not cl_exe.exists():
        cl_exe = msvc_root / "bin" / "Hostx86" / "x86" / "cl.exe"
    if not cl_exe.exists():
        return None
    env = os.environ.copy()
    include_paths = [
        msvc_root / "include",
        sdk_include_root / "ucrt",
        sdk_include_root / "shared",
        sdk_include_root / "um",
        sdk_include_root / "winrt",
        sdk_include_root / "cppwinrt",
        default_focas_header_dir(),
    ]
    lib_paths = [
        msvc_root / "lib" / "x86",
        sdk_lib_root / "ucrt" / "x86",
        sdk_lib_root / "um" / "x86",
    ]
    path_entries = [cl_exe.parent, msvc_root / "bin" / "Hostx64" / "x86"]
    env["INCLUDE"] = ";".join(str(path) for path in include_paths if path.exists())
    env["LIB"] = ";".join(str(path) for path in lib_paths if path.exists())
    env["PATH"] = ";".join(str(path) for path in path_entries if path.exists()) + ";" + env.get("PATH", "")
    return cl_exe, env


def latest_existing_dir(roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            children = [path for path in root.iterdir() if path.is_dir()]
            candidates.extend(children or [root])
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0]


def read_cpp_csv_logs(task_id: str, execution_dir: Path) -> CollectExecutionArtifactsOutput:
    input_path = execution_dir / "data" / "focas_api_input.csv"
    output_path = execution_dir / "data" / "focas_api_output.csv"
    schema_errors = validate_cpp_csv_schema(input_path, output_path)
    inputs = read_csv_by_index(input_path)
    outputs = read_csv_by_index(output_path)
    api_logs: list[ApiCallLog] = []
    capture_events: list[CaptureEvent] = []
    for index, out_row in sorted(outputs.items(), key=lambda item: int(item[0])):
        in_row = inputs.get(index, {})
        api_name = (
            out_row.get("api_name")
            or out_row.get("protocol_function")
            or in_row.get("protocol_function")
            or in_row.get("interface_name")
            or ""
        )
        interface_name = interface_from_focas(api_name)
        step_id = out_row.get("step_id") or in_row.get("step_id") or f"CPP-{index}"
        phase = in_row.get("phase", "during")
        params = {"raw": in_row.get("parameters", "")}
        status_code = safe_status_code(out_row.get("return_code") or out_row.get("status_code"))
        timestamp = out_row.get("timestamp") or in_row.get("timestamp") or utc_now()
        request_payload = {
            "api_name": api_name,
            "step_id": step_id,
            "phase": phase,
            "parameters": params,
        }
        response_payload = {
            "status_code": status_code,
            "function": api_name,
            "return_text": out_row.get("return_text") or out_row.get("error", ""),
            "data": out_row.get("data") or out_row.get("response", ""),
            "executor": "generated_cpp",
        }
        capture_events.append(
            CaptureEvent(
                timestamp=timestamp,
                task_id=task_id,
                interface_name=interface_name,
                direction="request",
                endpoint="generated-cpp://focas",
                payload_summary=request_payload,
            )
        )
        capture_events.append(
            CaptureEvent(
                timestamp=out_row.get("timestamp") or utc_now(),
                task_id=task_id,
                interface_name=interface_name,
                direction="response",
                endpoint="generated-cpp://agent",
                payload_summary=response_payload,
            )
        )
        api_logs.append(
            ApiCallLog(
                timestamp=timestamp,
                task_id=task_id,
                step_id=step_id,
                phase=phase,
                interface_name=interface_name,
                input_parameters=params,
                status_code=status_code,
                response=response_payload,
                protocol_function=api_name,
                semantic_label=semantic_label_for(interface_name),
                error=None if status_code == 0 else response_payload["return_text"],
            )
        )
    if schema_errors:
        api_logs.append(
            ApiCallLog(
                timestamp=utc_now(),
                task_id=task_id,
                step_id="CSV-SCHEMA",
                phase="after",
                interface_name="GeneratedCppCsv",
                input_parameters={"input_path": str(input_path), "output_path": str(output_path)},
                status_code=500,
                response={"status_code": 500, "schema_errors": schema_errors, "executor": "generated_cpp"},
                protocol_function="csv_schema_validation",
                semantic_label="execution_schema",
                error="; ".join(schema_errors),
            )
        )
    return CollectExecutionArtifactsOutput(api_logs, capture_events, schema_errors)


def validate_cpp_csv_schema(input_path: Path, output_path: Path) -> list[str]:
    errors: list[str] = []
    expected_input = ["index", "timestamp", "step_id", "phase", "interface_name", "protocol_function", "parameters"]
    expected_output = ["index", "timestamp", "step_id", "api_name", "return_code", "return_text", "data"]
    legacy_input = ["index", "step_id", "phase", "interface_name", "protocol_function", "parameters"]
    legacy_output = ["index", "step_id", "api_name", "return_code", "return_text", "data"]
    input_header = csv_header(input_path)
    output_header = csv_header(output_path)
    if input_header and input_header[: len(expected_input)] != expected_input and input_header[: len(legacy_input)] != legacy_input:
        errors.append(
            "CSV schema mismatch: focas_api_input.csv must start with "
            f"{expected_input} (legacy without timestamp is still readable), got {input_header}."
        )
    if output_header and output_header[: len(expected_output)] != expected_output and output_header[: len(legacy_output)] != legacy_output:
        errors.append(
            "CSV schema mismatch: focas_api_output.csv must start with "
            f"{expected_output} (legacy without timestamp is still readable), got {output_header}."
        )
    if not input_header:
        errors.append("CSV schema mismatch: focas_api_input.csv is missing or empty.")
    if not output_header:
        errors.append("CSV schema mismatch: focas_api_output.csv is missing or empty.")
    return errors


def csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        try:
            return [cell.strip() for cell in next(reader)]
        except StopIteration:
            return []


def read_csv_by_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(csv.DictReader(file), 1):
            index = row.get("index") or row.get("row_index") or str(row_number)
            rows[str(index)] = row
        return rows


def safe_status_code(value: str | None) -> int:
    try:
        return int(value or 500)
    except ValueError:
        return 500


def interface_from_focas(api_name: str) -> str:
    mapping = {
        "cnc_dwnstart3/cnc_download3/cnc_dwnend3": "UploadProgram",
        "cnc_search": "SelectProgram",
        "cnc_rdprgnum": "ReadProgramNumber",
        "ncguide_ui_cycle_start": "StartProgram",
        "cnc_statinfo": "ReadRunStatus",
        "cnc_rdposition": "ReadPosition",
        "cnc_distance": "ReadDistanceToGo",
        "cnc_actf": "ReadFeedSpeed",
        "cnc_acts": "ReadSpindleSpeed",
        "cnc_alarm2": "ReadAlarm",
    }
    return mapping.get(api_name, api_name or "UnknownApi")


def write_compile_logs(work_dir: Path, stdout: str, stderr: str) -> None:
    (work_dir / "compile_stdout.txt").write_text(stdout, encoding="utf-8")
    (work_dir / "compile_stderr.txt").write_text(stderr, encoding="utf-8")


def decode_process_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
