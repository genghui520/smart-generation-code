"""Evaluate one RQ1 code-generation run against independent ground truth.

This evaluator is deliberately outside SMPAgent. RQ1 is code-only: simulator
connection, runtime upload success, packet capture, and runtime logs do not
affect the score.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smart_traffic_agent.integrations.ncguide import default_focas_header_dir
from smart_traffic_agent.evaluation_metrics import generation_time_with_source, load_token_usage


GROUND_TRUTH_DIR = ROOT / "evaluation" / "ground_truth"
FOCAS_HEADER = default_focas_header_dir() / "Fwlib32.h"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_dir(path: Path) -> Path:
    if (path / "generated").is_dir():
        return path
    candidates = sorted(path.glob("**/run_*"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run directory found below {path}")
    return candidates[-1]


def read_artifacts(run_dir: Path) -> tuple[str, str, Path, Path]:
    generated = run_dir / "generated"
    cpp_path = generated / "api_script.cpp"
    nc_path = generated / "program.nc"
    cpp = cpp_path.read_text(encoding="utf-8", errors="replace") if cpp_path.exists() else ""
    fragment_path = generated / "scenario_fragment.inc"
    fragment = (
        fragment_path.read_text(encoding="utf-8", errors="replace")
        if fragment_path.exists()
        else ""
    )
    nc = nc_path.read_text(encoding="utf-8", errors="replace") if nc_path.exists() else ""
    # Fragment architecture keeps scenario code in a separate include. Treat
    # the assembled source as the evaluation unit without inlining or mutating
    # the generated artifact on disk.
    return f"{cpp}\n{fragment}", nc, cpp_path, nc_path


def official_api_names() -> set[str]:
    if not FOCAS_HEADER.exists():
        return set()
    header = FOCAS_HEADER.read_text(encoding="latin-1", errors="replace")
    return {
        name.lower()
        for name in re.findall(
            r"FWLIBAPI\s+[^;]*?\b((?:cnc|pmc)_[A-Za-z0-9_]+)\s*\(",
            header,
            re.IGNORECASE | re.DOTALL,
        )
    }


def referenced_api_names(cpp: str) -> set[str]:
    names = set()
    for match in re.finditer(r"\b((?:cnc|pmc)_[A-Za-z0-9_]+)\b", cpp, re.IGNORECASE):
        name = match.group(1)
        tail = cpp[match.end() :]
        canonical = re.sub(r"(?:_p|_fn)$", "", name, flags=re.IGNORECASE)
        if (
            re.match(r"\s*\(", tail)
            or re.search(rf"GetProcAddress\s*\([^,]+,\s*\"{re.escape(canonical)}\"", cpp, re.IGNORECASE)
            or re.search(rf"decltype\s*\(\s*&\s*::?{re.escape(canonical)}", cpp, re.IGNORECASE)
        ):
            names.add(canonical)
    return names


def api_present(cpp: str, api: str) -> bool:
    """Require an actual invocation, not only a declaration or loader string."""
    return bool(re.search(rf"(?:\b\w+\.)?{re.escape(api)}(?:_p|_fn)?\s*\(", cpp, re.IGNORECASE))


def compilation_succeeded(run_dir: Path) -> bool:
    """Detect compiler diagnostics in both stdout and stderr."""
    stdout_path = run_dir / "generated" / "compile_stdout.txt"
    stderr_path = run_dir / "generated" / "compile_stderr.txt"
    if not stdout_path.exists() or not stderr_path.exists():
        return False
    diagnostics = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (stdout_path, stderr_path)
    )
    if re.search(r"(?:fatal\s+)?error\s+[A-Z]?\d+\s*:", diagnostics, re.IGNORECASE):
        return False
    if re.search(r"\b(?:error|fatal error):", diagnostics, re.IGNORECASE):
        return False
    return True


def operation_satisfied(cpp: str, operation: dict[str, Any]) -> bool:
    accepted = [str(api) for api in operation.get("accepted_apis", [])]
    operation_id = str(operation.get("id", ""))
    if operation_id == "upload_nc_program":
        return bool(accepted) and all(api_present(cpp, api) for api in accepted)
    return any(api_present(cpp, api) for api in accepted)


def required_operation_results(cpp: str, truth: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for operation in truth.get("required_operations", []):
        results.append({
            "operation_id": operation.get("id"),
            "required": True,
            "satisfied": operation_satisfied(cpp, operation),
            "accepted_apis": operation.get("accepted_apis", []),
        })
    for group in truth.get("operation_groups", []):
        options = group.get("operation_options", [])
        satisfied = any(operation_satisfied(cpp, option) for option in options)
        results.append({
            "operation_id": group.get("id"),
            "required": True,
            "mode": group.get("mode", "at_least_one"),
            "satisfied": satisfied,
            "options": [option.get("id") for option in options],
        })
    return results


def nc_checks(nc: str, truth: dict[str, Any], cpp: str) -> dict[str, Any]:
    requirements = truth.get("nc_requirements", {})
    numbers = [int(value) for value in re.findall(r"^\s*[Oo](\d+)\b", nc, re.MULTILINE)]
    number_valid = True
    if numbers and requirements.get("program_number_range"):
        bounds = requirements["program_number_range"]
        number_valid = bounds["min"] <= numbers[0] <= bounds["max"]
    required_blocks = requirements.get("required_blocks", [])
    block_results = {block: bool(re.search(rf"\b{re.escape(block)}\b", nc, re.IGNORECASE)) for block in required_blocks}
    feature_results = {}
    for feature in requirements.get("required_features", []):
        lowered = str(feature).lower()
        if "two g01" in lowered:
            feature_results[feature] = len(re.findall(r"\bG01\b", nc, re.IGNORECASE)) >= 2
        elif "two feed" in lowered:
            feature_results[feature] = len(re.findall(r"\bF\s*[-+]?\d+(?:\.\d+)?", nc, re.IGNORECASE)) >= 2
        else:
            feature_results[feature] = True
    end_marker = str(requirements.get("required_end_marker", ""))
    return {
        "file_nonempty": bool(nc.strip()),
        "program_number_present": bool(numbers) or not requirements.get("required", False),
        "program_number_in_range": number_valid,
        "required_blocks": block_results,
        "required_features": feature_results,
        "end_marker_present": bool(re.search(rf"\b{re.escape(end_marker)}\b", nc, re.IGNORECASE)) if end_marker else True,
        "program_number_references_match": not numbers or all(str(number) in cpp for number in numbers),
    }


def evaluate(run_path: Path) -> dict[str, Any]:
    run_dir = find_run_dir(run_path)
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    metrics_path = run_dir / "run_metrics.json"
    run_metrics = load_json(metrics_path) if metrics_path.exists() else {}
    metadata_path = run_dir / "rq1_run_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    generation_time, generation_time_source = generation_time_with_source(run_metrics, metadata)
    scenario = str(summary.get("scenario_type") or run_dir.parent.name)
    truth_path = GROUND_TRUTH_DIR / f"{scenario}.json"
    if not truth_path.exists():
        scenario = run_dir.parent.name
        truth_path = GROUND_TRUTH_DIR / f"{scenario}.json"
    truth = load_json(truth_path)
    cpp, nc, cpp_path, nc_path = read_artifacts(run_dir)
    operations = required_operation_results(cpp, truth)
    satisfied_count = sum(bool(item["satisfied"]) for item in operations)
    operation_coverage = satisfied_count / len(operations) if operations else 0.0
    official = official_api_names()
    referenced = referenced_api_names(cpp)
    unknown = sorted(name for name in referenced if name.lower() not in official)
    resolved = set(re.findall(r"GetProcAddress\s*\([^,]+,\s*\"((?:cnc|pmc)_[A-Za-z0-9_]+)\"", cpp, re.IGNORECASE))
    abi_unbound = sorted(
        name for name in resolved
        if not re.search(rf"decltype\s*\(\s*&\s*::?{re.escape(name)}\b", cpp, re.IGNORECASE)
    )
    compile_stderr = run_dir / "generated" / "compile_stderr.txt"
    compile_stdout = run_dir / "generated" / "compile_stdout.txt"
    compilation_success = compilation_succeeded(run_dir)
    placeholders = sorted(set(re.findall(r"\b(?:TODO|FIXME|CHANGE_ME|SKIPPED_UNSUPPORTED_BY_CPP_CODEGEN|not_executed_by_cpp_generator)\b", cpp, re.IGNORECASE)))
    nc_result = nc_checks(nc, truth, cpp)
    required_complete = operation_coverage == 1.0
    no_placeholders = not placeholders
    cpp_present = cpp_path.exists() and cpp_path.stat().st_size > 0
    nc_required = bool(truth.get("nc_requirements", {}).get("required", False))
    nc_present = nc_path.exists() and nc_path.stat().st_size > 0
    artifacts_present = cpp_present and (nc_present or not nc_required)
    lifecycle_complete = all(api_present(cpp, api) for api in ("cnc_freelibhndl",))
    nc_static_checks = [
        bool(nc_result["file_nonempty"]),
        bool(nc_result["program_number_present"]),
        bool(nc_result["program_number_in_range"]),
        all(nc_result["required_blocks"].values()),
        all(nc_result["required_features"].values()),
        bool(nc_result["end_marker_present"]),
        bool(nc_result["program_number_references_match"]),
    ]
    completeness_checks = {
        "artifacts_present": artifacts_present,
        "required_operations_present": required_complete,
        "no_placeholders_or_skips": no_placeholders,
        "nc_artifact_constraints": all(nc_static_checks) if truth.get("nc_requirements", {}).get("required") else True,
        "error_handling_and_cleanup_present": lifecycle_complete and "return" in cpp,
    }
    api_validity = len(unknown) == 0 and not abi_unbound and bool(official)
    logic_constraints = True
    if scenario == "coordinate_motion":
        logic_constraints = (
            api_present(cpp, "cnc_actf")
            and bool(re.search(r"\b(?:abs|mach|rel|dist)\b", cpp))
            and nc_result["program_number_in_range"]
            and nc_result["program_number_references_match"]
        )
    consistency_components = {
        "required_function_coverage": operation_coverage,
        "additional_api_official_and_abi": 1.0 if api_validity else 0.0,
        "execution_logic_and_task_constraints": 1.0 if logic_constraints else 0.0,
    }
    completeness = sum(completeness_checks.values()) / len(completeness_checks)
    consistency = sum(consistency_components.values()) / len(consistency_components)
    token_usage = load_token_usage(
        (summary, "summary.llm_usage"),
        (run_metrics, "run_metrics.llm_usage"),
    )
    generated_dir = run_dir / "generated"
    generated_file_count = len([
        path for path in generated_dir.rglob("*") if path.is_file()
    ]) if generated_dir.exists() else 0
    framework_success = summary.get("framework_success")
    if framework_success is None:
        return_code = metadata.get("return_code")
        framework_success = (int(return_code) == 0) if return_code is not None else summary.get("success")

    return {
        "run_dir": str(run_dir),
        "scenario": scenario,
        "evaluation_scope": "rq1_code_only",
        "runtime_evidence_used": False,
        "framework_success": framework_success,
        "failure_category": summary.get("failure_category"),
        "generated_file_count": generated_file_count,
        "generation_time_seconds": generation_time,
        "generation_time_source": generation_time_source,
        "compilation_success_rate": 1.0 if compilation_success else 0.0,
        "completeness": completeness,
        "completeness_checks": completeness_checks,
        "consistency": consistency,
        "consistency_components": consistency_components,
        "required_operation_results": operations,
        "referenced_api_names": sorted(referenced),
        "unknown_api_names": unknown,
        "abi_unbound_api_names": abi_unbound,
        "nc_checks": nc_result,
        "overall_task_score": (float(compilation_success) + completeness + consistency) / 3,
        "token_usage": token_usage,
        "metric_definitions": {
            "compilation_success_rate": "For one run, 1 if compilation succeeds and 0 otherwise; across repeated runs, successful compilations divided by total runs.",
            "completeness": "Arithmetic mean of five ground-truth checks: artifact validity, all required operations present, placeholder-free source, NC artifact constraints, and error handling/cleanup.",
            "consistency": "Arithmetic mean of required-operation coverage, official API/ABI validity, and scenario-specific execution/task constraints.",
            "overall_task_score": "Arithmetic mean of compilation success, completeness, and consistency; token usage is reported separately and is not folded into quality scores.",
            "token_usage": "Persisted prompt, completion, and total token counts; total is prompt plus completion when absent. Missing usage is null, never estimated by this evaluator.",
            "generation_time": "PlanningAgent plus CodeGenerationAgent wall-clock duration from run_metrics.json; excludes ExecutionAgent time. For an RQ1 --no-execute wrapper with missing per-agent durations, rq1_run_metadata.elapsed_seconds_code_only is used and labeled as a fallback; otherwise duration is null.",
        },
    }


def csv_row(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten the authoritative RQ1 code-only metrics for tabular reporting."""
    token_usage = result.get("token_usage") or {}
    completeness_checks = result.get("completeness_checks") or {}
    consistency_components = result.get("consistency_components") or {}
    return {
        "run_dir": result.get("run_dir"),
        "scenario": result.get("scenario"),
        "framework_success": result.get("framework_success"),
        "failure_category": result.get("failure_category"),
        "generated_file_count": result.get("generated_file_count"),
        "compilation_success_rate": result.get("compilation_success_rate"),
        "completeness": result.get("completeness"),
        "consistency": result.get("consistency"),
        "overall_task_score": result.get("overall_task_score"),
        "operation_coverage": consistency_components.get("required_function_coverage"),
        "api_and_abi_validity": consistency_components.get("additional_api_official_and_abi"),
        "scenario_logic_constraints": consistency_components.get("execution_logic_and_task_constraints"),
        "artifacts_present": completeness_checks.get("artifacts_present"),
        "required_operations_present": completeness_checks.get("required_operations_present"),
        "no_placeholders_or_skips": completeness_checks.get("no_placeholders_or_skips"),
        "nc_artifact_constraints": completeness_checks.get("nc_artifact_constraints"),
        "error_handling_and_cleanup_present": completeness_checks.get("error_handling_and_cleanup_present"),
        "generation_time_seconds": result.get("generation_time_seconds"),
        "generation_time_source": result.get("generation_time_source"),
        "llm_request_count": token_usage.get("request_count"),
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "token_usage_source": token_usage.get("usage_source"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="A run directory or an output root containing run_*")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.run)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.csv_output:
        row = csv_row(result)
        with args.csv_output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
