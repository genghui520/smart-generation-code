"""Run the SMPAgent code-generation-only RQ1 benchmark."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


SCENARIOS = {
    "coordinate_motion": (
        "Generate a complete FOCAS client for the Coordinate Motion scenario. "
        "The client must connect to 127.0.0.1:8193, read CNC run status, actual feed, "
        "and absolute X/Y/Z position, and include a documented bounded NC program "
        "with G90, G54, G00, multiple G01 blocks with multiple feed values, and M30."
    ),
    "program_management": (
        "Generate a complete FOCAS client for the Program Management scenario. "
        "The client must connect to 127.0.0.1:8193, safely upload or document a bounded "
        "NC program, select and verify its O-number, read CNC run status and program "
        "number, and include complete return-code handling and cleanup. Use only documented "
        "FANUC FOCAS APIs and a valid NC program payload ending in M30."
    ),
    "tool_management": (
        "Generate a complete FOCAS client for the Tool Management scenario. "
        "The client must connect to 127.0.0.1:8193, read relevant CNC status and tool or "
        "offset information using documented FANUC FOCAS APIs, perform only bounded and "
        "explicitly safe operations, check every return code, and cleanly release the "
        "FOCAS handle. Include a complete documented NC context when required by the API."
    ),
}


def run_one(
    root: Path,
    output_root: Path,
    scenario: str,
    repetition: int,
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    timeout_seconds: float,
) -> dict[str, object]:
    scenario_dir = output_root / "SMPAgent" / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    base_output = scenario_dir
    command = [
        sys.executable,
        str(root / "main.py"),
        "--task",
        SCENARIOS[scenario],
        "--out",
        str(base_output),
        "--task-id",
        f"rq1_{scenario}_{repetition:03d}",
        "--target",
        "ncguide-generated-cpp",
        "--no-execute",
        "--code-only-evaluation",
        "--no-quality-gate",
        "--representative-scenario",
        "--no-allow-delete-all-programs",
        "--llm-provider",
        "openai_compatible",
        "--llm-model",
        model,
        "--llm-base-url",
        base_url,
        "--llm-api-key-env",
        api_key_env,
        "--llm-transport",
        "http",
        "--llm-wire-api",
        "responses",
        "--llm-reasoning-effort",
        "low",
        "--llm-timeout-seconds",
        str(timeout_seconds),
    ]
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    run_candidates = sorted(scenario_dir.glob("run_*"), key=lambda path: path.stat().st_mtime)
    run_dir = run_candidates[-1] if run_candidates else scenario_dir / f"run_{repetition:03d}"
    stdout_path = run_dir / "rq1_stdout.txt"
    stderr_path = run_dir / "rq1_stderr.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "scenario": scenario,
        "repetition": repetition,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "return_code": completed.returncode,
        "output_base": str(base_output),
        "command": command,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (run_dir / "rq1_run_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("evaluation/rq1_20260904"))
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default="https://iiegpt.gentelmen.cn")
    parser.add_argument("--api-key-env", default="MY_LLM_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    records = []
    for scenario in scenarios:
        for repetition in range(1, args.repetitions + 1):
            print(f"[RQ1] {scenario} repetition {repetition}/{args.repetitions}", flush=True)
            record = run_one(
                root,
                args.output_root,
                scenario,
                repetition,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                timeout_seconds=args.timeout_seconds,
            )
            records.append(record)
            print(
                f"[RQ1] return_code={record['return_code']} elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "smpagent_run_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if all(record["return_code"] == 0 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
