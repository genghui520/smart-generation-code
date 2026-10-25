"""Finalize an evaluated OpenRelay RQ1 baseline suite without rescoring it."""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rq1_scenario_tasks import baseline_prompt_for_scenario


MODEL = "gpt-5.5"
BASE_URL = "https://openrelayx.cc/v1"
FRAMEWORKS = {
    "GPT-Engineer": "gpt-engineer",
    "MetaGPT": "metagpt",
    "ChatDev 1.0": "chatdev1",
}
SCENARIOS = ("coordinate_motion", "program_management", "tool_management")
FIELDS = [
    "framework", "model", "base_url", "scenario", "framework_success",
    "compilation_success_rate", "completeness", "consistency",
    "overall_task_score", "operation_coverage", "generation_time_seconds",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "token_usage_source", "failure_diagnostics",
]


def console_error(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    matches = re.findall(
        r"(?:openai\.)?([A-Za-z][A-Za-z0-9_]*Error):\s*([^\r\n]+)", text
    )
    return f"{matches[-1][0]}: {matches[-1][1]}" if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    suite = args.suite.resolve()
    report = suite / "report"
    diagnostics: dict[tuple[str, str], str] = {}
    prompt_checks: list[dict[str, object]] = []

    for framework, raw_name in FRAMEWORKS.items():
        for scenario in SCENARIOS:
            raw = suite / "raw" / raw_name / scenario
            run = suite / framework / scenario / "run_001"
            expected = baseline_prompt_for_scenario(scenario)
            actual = (raw / "prompt.txt").read_text(encoding="utf-8")
            exact_match = actual == expected
            prompt_checks.append(
                {"framework": framework, "scenario": scenario, "exact_match": exact_match}
            )
            if not exact_match:
                raise RuntimeError(f"Prompt mismatch: {framework} {scenario}")

            metadata_path = raw / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            # The evaluator's token field identifies the summary location, not
            # the provenance of the counts.  Preserve the provider provenance
            # recorded by each runner in the final CSV.
            if metadata.get("usage") and metadata.get("usage_source") == "metagpt_cost_manager":
                metadata["usage_source"] = "provider_response"
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                shutil.copy2(metadata_path, run / "framework_metadata.json")
            if metadata.get("model") != MODEL or metadata.get("base_url") != BASE_URL:
                raise RuntimeError(f"Model/base URL mismatch: {framework} {scenario}")
            error = metadata.get("error") or console_error(raw / "runner_console.log")
            if not error and not metadata.get("success"):
                error = "Framework returned without a generated C/C++ source artifact."
            diagnostics[(framework, scenario)] = str(error or "")
            if error and not metadata.get("error"):
                metadata["error"] = error
                metadata["error_source"] = "runner_console.log"
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                shutil.copy2(metadata_path, run / "framework_metadata.json")
                rq1_path = run / "rq1_run_metadata.json"
                rq1 = json.loads(rq1_path.read_text(encoding="utf-8"))
                rq1["framework_error"] = error
                rq1_path.write_text(
                    json.dumps(rq1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

    report.mkdir(parents=True, exist_ok=True)
    (report / "prompt_parity_checks.json").write_text(
        json.dumps(prompt_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source = report / "baseline_ground_truth_metrics.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
    final_rows = []
    for row in rows:
        framework, scenario = row["framework"], row["scenario"]
        final_rows.append({
            "framework": framework,
            "model": row["model"],
            "base_url": row["base_url"],
            "scenario": scenario,
            "framework_success": row["framework_success"],
            "compilation_success_rate": row["compilation_success_rate"],
            "completeness": row["completeness"],
            "consistency": row["consistency"],
            "overall_task_score": row["overall_task_score"],
            "operation_coverage": row["operation_coverage"],
            "generation_time_seconds": row["generation_time_seconds"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "token_usage_source": (
                json.loads((suite / "raw" / FRAMEWORKS[framework] / scenario / "metadata.json").read_text(encoding="utf-8")).get("usage_source")
                if row["prompt_tokens"] or row["completion_tokens"] or row["total_tokens"]
                else "unavailable"
            ),
            "failure_diagnostics": diagnostics[(framework, scenario)],
        })
    final_csv = report / "final_baseline_rq1_results.csv"
    with final_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(final_rows)
    print(final_csv)


if __name__ == "__main__":
    main()
