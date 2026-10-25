"""Normalize one framework output for the independent RQ1 Ground Truth evaluator."""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_traffic_agent.agent_tools.cpp_execution import CompileCppInput, CompileGeneratedCppTool


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def select_cpp(generated: Path) -> Path | None:
    sources = [path for path in generated.rglob("*") if path.is_file() and path.suffix.lower() in CPP_SUFFIXES]
    if not sources:
        return None
    preferred = [path for path in sources if path.name.lower() == "api_script.cpp"]
    return max(preferred or sources, key=lambda path: path.stat().st_size)


def extract_nc(source: str) -> str:
    candidates: list[str] = []
    for match in re.finditer(r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\((?P<body>.*?)\)(?P=delimiter)"', source, re.DOTALL):
        candidates.append(match.group("body"))
    for group in re.finditer(r'(?:(?:u8|u|U|L)?"(?:\\.|[^"\\])*"\s*)+', source, re.DOTALL):
        decoded: list[str] = []
        for token in re.findall(r'(?:u8|u|U|L)?"(?:\\.|[^"\\])*"', group.group(0), re.DOTALL):
            literal = re.sub(r"^(?:u8|u|U|L)", "", token)
            try:
                decoded.append(ast.literal_eval(literal))
            except (SyntaxError, ValueError):
                decoded = []
                break
        if decoded:
            candidates.append("".join(decoded))
    valid = [
        value.replace("\r\n", "\n").replace("\r", "\n").strip()
        for value in candidates
        if re.search(r"(?m)^\s*O\d+\b", value, re.IGNORECASE)
        and re.search(r"\bM30\b", value, re.IGNORECASE)
    ]
    return max(valid, key=len) + "\n" if valid else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--scenario", choices=["coordinate_motion", "program_management", "tool_management"], required=True)
    args = parser.parse_args()

    raw = args.raw_output.resolve()
    run = args.run_output.resolve()
    generated = run / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    artifact_root = raw / "generated"
    if not artifact_root.exists() and (raw / "generated_extracted").exists():
        artifact_root = raw / "generated_extracted"
    if not artifact_root.exists() and (raw / "workspace").exists():
        # MetaGPT may finish through native Editor/Terminal tools without
        # returning a project_path. Its byte-for-byte artifacts still live in
        # the isolated run workspace and are valid framework outputs.
        artifact_root = raw / "workspace"
    source_path = select_cpp(artifact_root)
    source = source_path.read_text(encoding="utf-8", errors="replace") if source_path else ""
    api_path = generated / "api_script.cpp"
    api_path.write_text(source, encoding="utf-8")

    nc_sources = sorted(artifact_root.rglob("*.nc")) if artifact_root.exists() else []
    if nc_sources:
        nc = max(nc_sources, key=lambda path: path.stat().st_size).read_text(encoding="utf-8", errors="replace")
        nc_source = str(max(nc_sources, key=lambda path: path.stat().st_size))
    else:
        nc = extract_nc(source)
        nc_source = "extracted_from_generated_cpp" if nc else None
    (generated / "program.nc").write_text(nc, encoding="utf-8")

    if source:
        result = CompileGeneratedCppTool().invoke(
            CompileCppInput(api_path, generated / "api_script.exe", generated)
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if not result.success and "error" not in (stdout + stderr).lower():
            stderr += f"\nerror: compiler returned exit code {result.return_code}\n"
        compile_success = result.success
        compile_command = result.command
    else:
        stdout = ""
        stderr = "error: framework produced no C/C++ source artifact\n"
        compile_success = False
        compile_command = []
    (generated / "compile_stdout.txt").write_text(stdout, encoding="utf-8")
    (generated / "compile_stderr.txt").write_text(stderr, encoding="utf-8")
    for path in (generated / "api_script.exe", generated / "api_script.obj"):
        path.unlink(missing_ok=True)

    usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else None
    # MetaGPT emits an all-zero placeholder when its cost table has no entry for
    # a model. That is not measured token consumption and must remain unavailable.
    if usage and not int(usage.get("total_tokens", 0) or 0):
        usage = None
    # The common evaluator scores the generated artifact. A framework can
    # produce it and later fail in unrelated packaging/logging code; retain
    # that error in rq1_run_metadata while accurately recording artifact
    # generation success here.
    framework_success = source_path is not None
    summary = {
        "scenario_type": args.scenario,
        "framework": metadata.get("framework"),
        "framework_success": framework_success,
        "llm_usage": {**usage, "usage_source": metadata.get("usage_source")} if usage else None,
    }
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run / "rq1_run_metadata.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "framework": metadata.get("framework"),
                "elapsed_seconds": metadata.get("duration_seconds"),
                "command": [metadata.get("framework", "baseline"), "--no-execute"],
                "raw_output": str(raw),
                "source_artifact": str(source_path) if source_path else None,
                "nc_artifact_source": nc_source,
                "framework_success": framework_success,
                "framework_error": metadata.get("error"),
                "compile_success": compile_success,
                "compile_command": compile_command,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if metadata_path.exists():
        shutil.copy2(metadata_path, run / "framework_metadata.json")
    print(json.dumps({"run": str(run), "source": str(source_path) if source_path else None, "nc_source": nc_source, "compile_success": compile_success}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
