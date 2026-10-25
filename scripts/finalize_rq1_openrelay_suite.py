"""Finalize the one-shot 2026-09-09 OpenRelay RQ1 baseline suite.

This does not alter or recompute Ground Truth metrics.  It records setup
failures that occurred before framework runners could emit metadata and adds
those diagnostics to a presentation CSV derived from the evaluator CSV.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.rq1_scenario_tasks import baseline_prompt_for_scenario


SUITE = ROOT / "evaluation" / "baseline_rq1_gpt55_prompt_parity_openrelay_20260909"
MODEL = "gpt-5.5"
BASE_URL = "https://openrelayx.cc/v1"

FRAMEWORKS = {
    "GPT-Engineer": ("gpt-engineer", "https://github.com/AntonOsika/gpt-engineer.git"),
    "MetaGPT": ("metagpt", "https://github.com/FoundationAgents/MetaGPT.git"),
    "ChatDev 1.0": ("chatdev1", "https://github.com/OpenBMB/ChatDev.git"),
}
SCENARIOS = ("coordinate_motion", "program_management", "tool_management")
FAILURES = {
    ("GPT-Engineer", "coordinate_motion"): "ModuleNotFoundError: No module named 'pyperclip'",
    ("GPT-Engineer", "program_management"): "ModuleNotFoundError: No module named 'langchain.callbacks'",
    ("GPT-Engineer", "tool_management"): "ModuleNotFoundError: No module named 'langchain.callbacks'",
    ("MetaGPT", "coordinate_motion"): "ModuleNotFoundError: No module named 'semantic_kernel'",
    ("MetaGPT", "program_management"): "ModuleNotFoundError: No module named 'semantic_kernel'",
    ("MetaGPT", "tool_management"): "ModuleNotFoundError: No module named 'semantic_kernel'",
    ("ChatDev 1.0", "coordinate_motion"): "ModuleNotFoundError: No module named 'easydict'",
    ("ChatDev 1.0", "program_management"): "ModuleNotFoundError: No module named 'easydict'",
    ("ChatDev 1.0", "tool_management"): "ModuleNotFoundError: No module named 'easydict'",
}


def main() -> None:
    prompt_checks: list[dict[str, object]] = []
    for framework, (raw_name, repository) in FRAMEWORKS.items():
        for scenario in SCENARIOS:
            raw = SUITE / "raw" / raw_name / scenario
            run = SUITE / framework / scenario / "run_001"
            prompt_path = raw / "prompt.txt"
            expected = baseline_prompt_for_scenario(scenario)
            actual = prompt_path.read_text(encoding="utf-8")
            prompt_equal = actual == expected
            prompt_checks.append(
                {"framework": framework, "scenario": scenario, "exact_match": prompt_equal}
            )
            if not prompt_equal:
                raise RuntimeError(f"Prompt mismatch: {framework} {scenario}")

            error = FAILURES[(framework, scenario)]
            metadata = {
                "framework": framework,
                "repository": repository,
                "model": MODEL,
                "base_url": BASE_URL,
                "api_key_env": "MY_LLM_API_KEY2",
                "duration_seconds": None,
                "generation_started": False,
                "usage": None,
                "usage_source": "unavailable",
                "success": False,
                "error": error,
                "metadata_source": "orchestrator_after_pre-generation-import-failure",
            }
            metadata_path = raw / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            (raw / "error.txt").write_text(error + "\n", encoding="utf-8")
            shutil.copy2(metadata_path, run / "framework_metadata.json")
            rq1_metadata_path = run / "rq1_run_metadata.json"
            rq1_metadata = json.loads(rq1_metadata_path.read_text(encoding="utf-8"))
            rq1_metadata["framework_error"] = error
            rq1_metadata["elapsed_seconds"] = None
            rq1_metadata["model"] = MODEL
            rq1_metadata["base_url"] = BASE_URL
            rq1_metadata_path.write_text(
                json.dumps(rq1_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    report = SUITE / "report"
    (report / "prompt_parity_checks.json").write_text(
        json.dumps(prompt_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    source_csv = report / "baseline_ground_truth_metrics.csv"
    rows = list(csv.DictReader(source_csv.open(encoding="utf-8-sig", newline="")))
    required = [
        "framework",
        "model",
        "base_url",
        "scenario",
        "framework_success",
        "compilation_success_rate",
        "completeness",
        "consistency",
        "overall_task_score",
        "operation_coverage",
        "generation_time_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "token_usage_source",
        "failure_diagnostics",
    ]
    final_rows = []
    for row in rows:
        framework = row["framework"]
        scenario = row["scenario"]
        final_rows.append(
            {
                "framework": framework,
                "model": MODEL,
                "base_url": BASE_URL,
                "scenario": scenario,
                "framework_success": row["framework_success"],
                "compilation_success_rate": row["compilation_success_rate"],
                "completeness": row["completeness"],
                "consistency": row["consistency"],
                "overall_task_score": row["overall_task_score"],
                "operation_coverage": row["operation_coverage"],
                "generation_time_seconds": row["generation_time_seconds"] or "",
                "prompt_tokens": row["prompt_tokens"] or "",
                "completion_tokens": row["completion_tokens"] or "",
                "total_tokens": row["total_tokens"] or "",
                "token_usage_source": row["token_usage_source"] or "unavailable",
                "failure_diagnostics": FAILURES[(framework, scenario)],
            }
        )
    final_csv = report / "final_baseline_rq1_results.csv"
    with final_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=required)
        writer.writeheader()
        writer.writerows(final_rows)
    print(final_csv.resolve())


if __name__ == "__main__":
    main()
