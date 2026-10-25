"""Run and independently score the RQ3 code-only ablation variants."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.rq1_scenario_tasks import RQ1_SCENARIO_TASKS
except ModuleNotFoundError:
    from rq1_scenario_tasks import RQ1_SCENARIO_TASKS


VARIANTS = {
    "single-agent": "Single-Agent",
    "without-router": "w_o_RouterAgent",
}


def run_streaming(command: list[str], *, cwd: Path) -> tuple[int, str, str]:
    """Stream child output to the console while retaining separate audit logs."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain(stream, sink: list[str], *, prefix: str = "") -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            sink.append(line)
            print(f"{prefix}{line}", end="", flush=True)
        stream.close()

    stdout_thread = threading.Thread(
        target=drain,
        args=(process.stdout, stdout_lines),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr_lines),
        kwargs={"prefix": "[stderr] "},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return return_code, "".join(stdout_lines), "".join(stderr_lines)


def latest_run(path: Path) -> Path:
    candidates = sorted(path.glob("run_*"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run_* directory was created below {path}")
    return candidates[-1]


def run_one(
    root: Path,
    output_root: Path,
    variant: str,
    scenario: str,
    repetition: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    base_output = output_root / VARIANTS[variant] / scenario
    base_output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "main.py"),
        "--task", RQ1_SCENARIO_TASKS[scenario],
        "--out", str(base_output),
        "--task-id", f"rq3_{variant}_{scenario}_{repetition:03d}",
        "--target", "ncguide-generated-cpp",
        "--no-execute",
        "--code-only-evaluation",
        "--no-quality-gate",
        "--representative-scenario",
        "--no-allow-delete-all-programs",
        "--ablation", variant,
        "--llm-provider", "openai_compatible",
        "--llm-model", args.model,
        "--llm-base-url", args.base_url,
        "--llm-api-key-env", args.api_key_env,
        "--llm-transport", "http",
        "--llm-wire-api", args.wire_api,
        "--llm-timeout-seconds", str(args.timeout_seconds),
        "--temperature", str(args.temperature),
        "--max-repairs", str(args.max_repairs),
        "--no-historical-memory",
    ]
    if args.wire_api == "responses":
        command.extend(["--llm-reasoning-effort", args.reasoning_effort])
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    return_code, captured_stdout, captured_stderr = run_streaming(command, cwd=root)
    elapsed = time.perf_counter() - started
    run_dir = latest_run(base_output)
    (run_dir / "rq3_stdout.txt").write_text(captured_stdout, encoding="utf-8")
    (run_dir / "rq3_stderr.txt").write_text(captured_stderr, encoding="utf-8")
    metadata = {
        "framework": "SMPAgent",
        "experiment": "RQ3_ablation",
        "variant": variant,
        "scenario": scenario,
        "repetition": repetition,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "wire_api": args.wire_api,
        "temperature": args.temperature,
        "max_repairs": args.max_repairs,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "elapsed_seconds_code_only": elapsed,
        "return_code": return_code,
        "run_dir": str(run_dir),
        "command": command,
    }
    (run_dir / "rq1_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    evaluation_path = run_dir / "rq3_ground_truth_evaluation.json"
    evaluation_return_code = 1
    evaluation_stderr = "Generation failed before independent evaluation."
    # A model-generated source that fails validation/compilation is still a
    # valid ablation outcome: the independent evaluator must score it as such.
    if (run_dir / "generated").is_dir():
        evaluation = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "evaluate_rq1_ground_truth.py"),
                str(run_dir),
                "--output", str(evaluation_path),
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        evaluation_return_code = evaluation.returncode
        evaluation_stderr = evaluation.stderr or evaluation.stdout
    if evaluation_return_code != 0:
        (run_dir / "rq3_evaluation_stderr.txt").write_text(
            evaluation_stderr,
            encoding="utf-8",
        )
    result = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else {}
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    usage = result.get("token_usage") or {}
    failure_category = classify_failure(summary, result, evaluation_return_code)
    return {
        **metadata,
        "evaluation_return_code": evaluation_return_code,
        "score": result.get("overall_task_score"),
        "score_percent": (
            round(float(result["overall_task_score"]) * 100, 1)
            if result.get("overall_task_score") is not None
            else None
        ),
        "generation_time_seconds": result.get("generation_time_seconds"),
        "total_tokens": usage.get("total_tokens"),
        "compile_success": result.get("compilation_success_rate"),
        "framework_success": result.get("framework_success", return_code == 0),
        "generated_file_count": result.get("generated_file_count", summary.get("generated_file_count", 0)),
        "failure_category": failure_category,
        "evaluation": str(evaluation_path),
    }


def classify_failure(
    summary: dict[str, object],
    evaluation: dict[str, object],
    evaluation_return_code: int,
) -> str:
    recorded = str(summary.get("failure_category", ""))
    if recorded == "infrastructure_failure":
        return recorded
    if not summary.get("artifacts"):
        return "no_artifact"
    if evaluation_return_code != 0:
        return recorded or "infrastructure_failure"
    if evaluation.get("unknown_api_names") or evaluation.get("abi_unbound_api_names"):
        return "api_abi_error"
    if float(evaluation.get("compilation_success_rate", 0.0) or 0.0) < 1.0:
        return "compile_error"
    components = evaluation.get("consistency_components") or {}
    if isinstance(components, dict) and float(components.get("required_function_coverage", 0.0) or 0.0) < 1.0:
        return "incomplete_operations"
    if isinstance(components, dict) and float(components.get("execution_logic_and_task_constraints", 0.0) or 0.0) < 1.0:
        return "scenario_constraint_failure"
    return "success"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/rq3_ablation_single_agent_fair_20260910"),
    )
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    parser.add_argument("--scenario", choices=["all", *RQ1_SCENARIO_TASKS], default="all")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://fast.smartaipro.cn/v1")
    parser.add_argument("--api-key-env", default="MY_LLM_API_KEY2")
    parser.add_argument("--wire-api", choices=["responses", "chat_completions"], default="responses")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-repairs", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    root = Path(__file__).resolve().parents[1]
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    scenarios = list(RQ1_SCENARIO_TASKS) if args.scenario == "all" else [args.scenario]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for variant in variants:
            for repetition in range(1, args.repetitions + 1):
                print(f"[RQ3] {scenario} | {variant} | {repetition}/{args.repetitions}", flush=True)
                row = run_one(root, args.output_root, variant, scenario, repetition, args)
                rows.append(row)
                print(
                    f"[RQ3] rc={row['return_code']} score={row['score_percent']} "
                    f"time={row['generation_time_seconds']} tokens={row['total_tokens']}",
                    flush=True,
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "rq3_ablation_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "scenario", "variant", "repetition", "score_percent",
        "generation_time_seconds", "total_tokens", "compile_success",
        "framework_success", "generated_file_count", "failure_category",
        "model", "temperature", "run_dir", "evaluation",
    ]
    with (args.output_root / "rq3_ablation_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["evaluation_return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
