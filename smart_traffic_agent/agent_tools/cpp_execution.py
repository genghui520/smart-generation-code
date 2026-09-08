from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
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
            "Fwlib32.lib",
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
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def pump_stream(stream: Any, chunks: list[str], target: Any) -> None:
            try:
                for line in iter(stream.readline, ""):
                    chunks.append(line)
                    target.write(line)
                    target.flush()
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        try:
            process = subprocess.Popen(
                [str(tool_input.executable_path)],
                cwd=tool_input.work_dir,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stdout_thread = threading.Thread(
                target=pump_stream,
                args=(process.stdout, stdout_chunks, sys.stdout),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=pump_stream,
                args=(process.stderr, stderr_chunks, sys.stderr),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            timed_out = False
            try:
                exit_code: int | None = process.wait(timeout=tool_input.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                exit_code = None
                process.wait(timeout=10)
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            output = RunExecutableOutput(
                exit_code=exit_code,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
                timed_out=timed_out,
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
    ] + pcap_sdk_include_paths()
    lib_paths = [
        msvc_root / "lib" / "x86",
        sdk_lib_root / "ucrt" / "x86",
        sdk_lib_root / "um" / "x86",
        # FANUC ships Fwlib32.lib at the parent of the series-specific header
        # directory in the SDK layout used by this project.
        default_focas_header_dir().parent,
    ] + pcap_sdk_lib_paths()
    path_entries = [cl_exe.parent, msvc_root / "bin" / "Hostx64" / "x86"]
    env["INCLUDE"] = ";".join(str(path) for path in include_paths if path.exists())
    env["LIB"] = ";".join(str(path) for path in lib_paths if path.exists())
    env["PATH"] = ";".join(str(path) for path in path_entries if path.exists()) + ";" + env.get("PATH", "")
    return cl_exe, env


def pcap_sdk_include_paths() -> list[Path]:
    """Return optional WinPcap/Npcap SDK include directories for generated C++ builds."""

    return existing_paths(
        env_path_list("WINPCAP_SDK_INCLUDE")
        + env_path_list("NPCAP_SDK_INCLUDE")
        + [path / "Include" for path in pcap_sdk_roots()]
        + [
            Path(r"C:\Lib\WinPcapSDK\Include"),
            Path(r"C:\Lib\WpdPack\Include"),
            Path(r"C:\Program Files\Npcap SDK\Include"),
            Path(r"C:\Program Files (x86)\Npcap SDK\Include"),
        ]
    )


def pcap_sdk_lib_paths() -> list[Path]:
    """Return optional WinPcap/Npcap SDK library directories for 32-bit MSVC builds."""

    lib_candidates: list[Path] = []
    for root in pcap_sdk_roots():
        lib_candidates.extend([root / "Lib" / "x86", root / "Lib"])
    lib_candidates.extend(
        [
            Path(r"C:\Lib\WinPcapSDK\Lib\x86"),
            Path(r"C:\Lib\WinPcapSDK\Lib"),
            Path(r"C:\Lib\WpdPack\Lib"),
            Path(r"C:\Program Files\Npcap SDK\Lib\x86"),
            Path(r"C:\Program Files\Npcap SDK\Lib"),
            Path(r"C:\Program Files (x86)\Npcap SDK\Lib\x86"),
            Path(r"C:\Program Files (x86)\Npcap SDK\Lib"),
        ]
    )
    return existing_paths(env_path_list("WINPCAP_SDK_LIB") + env_path_list("NPCAP_SDK_LIB") + lib_candidates)


def pcap_sdk_roots() -> list[Path]:
    roots = env_path_list("WINPCAP_SDK_DIR") + env_path_list("NPCAP_SDK_DIR")
    roots.extend([Path(r"C:\Lib\WinPcapSDK"), Path(r"C:\Lib\WpdPack")])
    return existing_paths(roots)


def env_path_list(name: str) -> list[Path]:
    raw = os.environ.get(name, "")
    return [Path(part.strip()) for part in raw.split(";") if part.strip()]


def existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    existing: list[Path] = []
    for path in paths:
        normalized = str(path)
        key = normalized.casefold()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        existing.append(path)
    return existing


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
    schema_errors.extend(validate_pcap_artifacts(execution_dir / "data", output_path=output_path))
    inputs = read_csv_by_index(input_path)
    outputs = read_csv_by_index(output_path)
    combined_path = execution_dir / "data" / "focas_api_calls.csv"
    per_api_dir = execution_dir / "data" / "by_api"
    export_prefix = descriptive_csv_prefix(task_id, inputs, outputs)
    if schema_errors:
        remove_derived_api_csvs(combined_path, per_api_dir)
    else:
        write_combined_api_csv(inputs, outputs, combined_path)
        remove_stale_descriptive_raw_csv_copies(execution_dir / "data")
        write_descriptive_raw_csv_copies(input_path, output_path, execution_dir / "data", export_prefix)
        write_per_api_csvs(inputs, outputs, per_api_dir, export_prefix=export_prefix)
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
        params = {
            "raw": in_row.get("parameters", ""),
            "api_parameter_count": safe_status_code(
                in_row.get("api_parameter_count")
                or in_row.get("parameter_count")
                or in_row.get("param_count")
                or "0"
            ),
        }
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
            "api_parameter_count": safe_status_code(
                out_row.get("api_parameter_count")
                or out_row.get("return_parameter_count")
                or out_row.get("data_parameter_count")
                or "0"
            ),
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


def write_combined_api_csv(
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
    combined_path: Path,
) -> None:
    """Write one row per API/helper call with input and output side-by-side."""

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "timestamp",
        "step_id",
        "interface_name",
        "protocol_function",
        "api_parameter_count",
        "input_parameters",
        "return_code",
        "return_text",
        "output_data",
    ]
    indexes = sorted(
        set(inputs) | set(outputs),
        key=lambda value: int(value) if str(value).isdigit() else 10**9,
    )
    with combined_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in indexes:
            in_row = inputs.get(index, {})
            out_row = outputs.get(index, {})
            api_name = (
                out_row.get("api_name")
                or out_row.get("protocol_function")
                or in_row.get("protocol_function")
                or in_row.get("interface_name")
                or ""
            )
            writer.writerow(
                {
                    "index": index,
                    "timestamp": out_row.get("timestamp") or in_row.get("timestamp") or "",
                    "step_id": out_row.get("step_id") or in_row.get("step_id") or "",
                    "interface_name": in_row.get("interface_name") or interface_from_focas(api_name),
                    "protocol_function": api_name,
                    "api_parameter_count": (
                        in_row.get("api_parameter_count")
                        or out_row.get("api_parameter_count")
                        or in_row.get("parameter_count")
                        or str(api_parameter_count_for_name(api_name))
                    ),
                    "input_parameters": parameter_payload_only(in_row.get("parameters", "")),
                    "return_code": out_row.get("return_code") or out_row.get("status_code") or "",
                    "return_text": out_row.get("return_text") or out_row.get("error", ""),
                    "output_data": parameter_payload_only(out_row.get("data") or out_row.get("response", "")),
                }
            )


def remove_derived_api_csvs(combined_path: Path, per_api_dir: Path) -> None:
    try:
        if combined_path.exists():
            combined_path.unlink()
    except OSError:
        pass
    if not per_api_dir.exists():
        return
    for path in per_api_dir.glob("*.csv"):
        try:
            path.unlink()
        except OSError:
            pass


def write_per_api_csvs(
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
    output_dir: Path,
    *,
    export_prefix: str = "",
) -> None:
    """Write one CSV per API/helper name, with input and output side-by-side."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in output_dir.glob("*.csv"):
        try:
            stale_csv.unlink()
        except OSError:
            pass
    rows_by_api: dict[str, list[dict[str, str]]] = {}
    indexes = sorted(
        set(inputs) | set(outputs),
        key=lambda value: int(value) if str(value).isdigit() else 10**9,
    )
    for index in indexes:
        in_row = inputs.get(index, {})
        out_row = outputs.get(index, {})
        api_name = (
            out_row.get("api_name")
            or out_row.get("protocol_function")
            or in_row.get("protocol_function")
            or in_row.get("interface_name")
            or "unknown_api"
        )
        row = {
            "index": index,
            "timestamp": out_row.get("timestamp") or in_row.get("timestamp") or "",
            "step_id": out_row.get("step_id") or in_row.get("step_id") or "",
            "interface_name": in_row.get("interface_name") or interface_from_focas(api_name),
            "protocol_function": api_name,
            "api_parameter_count": (
                in_row.get("api_parameter_count")
                or out_row.get("api_parameter_count")
                or in_row.get("parameter_count")
                or str(api_parameter_count_for_name(api_name))
            ),
            "input_parameters": parameter_payload_only(in_row.get("parameters", "")),
            "return_code": out_row.get("return_code") or out_row.get("status_code") or "",
            "return_text": out_row.get("return_text") or out_row.get("error", ""),
            "output_data": parameter_payload_only(out_row.get("data") or out_row.get("response", "")),
        }
        rows_by_api.setdefault(api_name, []).append(row)

    fieldnames = [
        "index",
        "timestamp",
        "step_id",
        "interface_name",
        "protocol_function",
        "api_parameter_count",
        "input_parameters",
        "return_code",
        "return_text",
        "output_data",
    ]
    for api_name, rows in rows_by_api.items():
        filename = f"{safe_csv_filename(export_prefix + '_' if export_prefix else '')}{safe_csv_filename(api_name)}.csv"
        path = output_dir / filename
        try:
            with path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except PermissionError:
            # Windows commonly keeps CSVs locked while they are open in an
            # editor/Excel preview.  Do not let one locked derived file
            # prevent the rest of the per-API dataset from being written.
            continue


CSV_PARAMETER_METADATA_KEYS = {
    "coverage_role",
    "phase",
    "planner_selected_api",
    "sample_role",
    "segment_index",
    "structural_api",
}


def parameter_payload_only(value: str) -> str:
    """Return only API input/output parameters from a semicolon key-value payload.

    Generated C++ uses the same semicolon payload string for both real API
    arguments/results and workflow metadata.  Dataset CSVs should expose only
    the API-facing parameter payload; fields such as phase and segment index
    remain available in internal logs but are not input/output parameters.
    """

    if not value:
        return ""
    parts: list[str] = []
    for raw_part in str(value).split(";"):
        part = raw_part.strip()
        if not part:
            continue
        key = part.split("=", 1)[0].strip()
        if key in CSV_PARAMETER_METADATA_KEYS:
            continue
        parts.append(part)
    return ";".join(parts)


def write_descriptive_raw_csv_copies(
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    export_prefix: str,
) -> None:
    """Copy raw generated input/output CSVs to dataset-friendly descriptive names."""

    if not export_prefix:
        return
    copies = [
        (input_path, output_dir / f"{safe_csv_filename(export_prefix)}_input.csv"),
        (output_path, output_dir / f"{safe_csv_filename(export_prefix)}_output.csv"),
    ]
    for src, dst in copies:
        if not src.exists():
            continue
        try:
            dst.write_bytes(src.read_bytes())
        except OSError:
            pass


def remove_stale_descriptive_raw_csv_copies(output_dir: Path) -> None:
    """Remove previous date-prefixed input/output copies before writing new ones."""

    for pattern in ("????????_*_input.csv", "????????_*_output.csv"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def descriptive_csv_prefix(
    task_id: str,
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
) -> str:
    """Build the compact dataset prefix, e.g. 20260717."""

    date = csv_export_date(inputs, outputs)
    return safe_csv_filename(date)


def csv_export_date(
    inputs: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, str]],
) -> str:
    for rows in (outputs, inputs):
        for row in rows.values():
            timestamp = row.get("timestamp", "")
            digits = "".join(ch for ch in timestamp[:10] if ch.isdigit())
            if len(digits) == 8:
                return digits
    return datetime.now().strftime("%Y%m%d")


def safe_csv_filename(name: str, max_length: int = 96) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in name.strip())
    if len(safe) > max_length:
        digest = hashlib.sha1(safe.encode("utf-8", errors="ignore")).hexdigest()[:10]
        safe = f"{safe[: max_length - 11]}_{digest}"
    return safe or "unknown_api"


def api_parameter_count_for_name(api_name: str) -> int:
    counts = {
        "cnc_allclibhndl3": 4,
        "cnc_freelibhndl": 1,
        "cnc_statinfo": 2,
        "cnc_alarm2": 2,
        "cnc_rdalmmsg": 4,
        "cnc_rdprogdir3": 5,
        "cnc_dwnstart3": 2,
        "cnc_download3": 3,
        "cnc_dwnend3": 1,
        "cnc_search": 2,
        "cnc_rdprgnum": 2,
        "cnc_actf": 2,
        "cnc_rdposition": 4,
        "cnc_distance": 4,
        "cnc_absolute": 4,
        "cnc_absolute2": 4,
        "cnc_machine": 4,
        "cnc_relative": 4,
        "cnc_relative2": 4,
        "cnc_skip": 4,
        "cnc_srvdelay": 4,
        "cnc_accdecdly": 4,
        "cnc_rddynamic": 4,
        "cnc_rdaxisdata": 6,
        "cnc_rd3dtooltip": 2,
        "cnc_rdmdiprgstat": 2,
        "cnc_rdmdipntr": 2,
        "cnc_getdtailerr": 2,
        "cnc_delall": 1,
        "pcap_open_live": 5,
        "LoadLibraryW": 1,
        "ncguide_ui_cycle_start": 2,
    }
    return counts.get(api_name, 0)


def validate_pcap_artifacts(data_dir: Path, output_path: Path | None = None) -> list[str]:
    if not data_dir.exists():
        return ["PCAP capture validation: data directory is missing, so no pcap artifacts were produced."]
    pcap_files = sorted(data_dir.glob("*.pcap"))
    if not pcap_files:
        return ["PCAP capture validation: no .pcap artifacts were produced."]
    nonempty_packet_files = [path for path in pcap_files if path.stat().st_size > 24]
    if not nonempty_packet_files:
        return [
            "PCAP capture validation: all .pcap artifacts are 24 bytes or smaller, "
            "which indicates empty header-only marker files rather than captured packets."
        ]
    if output_path is not None and output_path.exists():
        csv_text = output_path.read_text(encoding="utf-8-sig", errors="replace").casefold()
        if "pcap_capture_enabled=false" in csv_text or "pcap_capture_disabled" in csv_text:
            return [
                "PCAP capture validation: generated execution explicitly logged pcap capture as disabled, "
                "so existing .pcap files cannot be treated as valid traffic from this run."
            ]
        if "pcap_capture_enabled=true" not in csv_text and "pcap_capture_started=true" not in csv_text:
            return [
                "PCAP capture validation: focas_api_output.csv does not log pcap_capture_enabled=true "
                "or pcap_capture_started=true for this execution."
            ]
        output_mtime = output_path.stat().st_mtime
        fresh_packet_files = [
            path for path in nonempty_packet_files if abs(output_mtime - path.stat().st_mtime) <= 180
        ]
        if not fresh_packet_files:
            return [
                "PCAP capture validation: non-empty .pcap files exist, but none were modified near the "
                "current focas_api_output.csv timestamp; refusing to reuse stale capture files."
            ]
    return []


def validate_cpp_csv_schema(input_path: Path, output_path: Path) -> list[str]:
    errors: list[str] = []
    expected_input = [
        "index",
        "timestamp",
        "step_id",
        "interface_name",
        "protocol_function",
        "parameters",
        "api_parameter_count",
    ]
    expected_output = [
        "index",
        "timestamp",
        "step_id",
        "api_name",
        "return_code",
        "return_text",
        "data",
        "api_parameter_count",
    ]
    legacy_input = ["index", "step_id", "phase", "interface_name", "protocol_function", "parameters"]
    legacy_timestamped_input = [
        "index",
        "timestamp",
        "step_id",
        "phase",
        "interface_name",
        "protocol_function",
        "parameters",
        "api_parameter_count",
    ]
    legacy_output = ["index", "step_id", "api_name", "return_code", "return_text", "data"]
    input_header = csv_header(input_path)
    output_header = csv_header(output_path)
    if (
        input_header
        and input_header[: len(expected_input)] != expected_input
        and input_header[: len(legacy_input)] != legacy_input
        and input_header[: len(legacy_timestamped_input)] != legacy_timestamped_input
    ):
        errors.append(
            "CSV schema mismatch: focas_api_input.csv must start with "
            f"{expected_input} (legacy formats are still readable), got {input_header}."
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
    if input_path.exists() and output_path.exists():
        delta = abs(input_path.stat().st_mtime - output_path.stat().st_mtime)
        if delta > 180:
            errors.append(
                "CSV freshness mismatch: focas_api_input.csv and focas_api_output.csv were modified "
                f"{delta:.1f} seconds apart; refusing to pair stale input/output rows by index."
            )
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
