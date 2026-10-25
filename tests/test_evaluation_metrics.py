from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_offline_experiments import evaluate
from scripts.evaluate_rq1_ground_truth import csv_row, evaluate as evaluate_ground_truth
from scripts.evaluate_rq1_ground_truth import compilation_succeeded, read_artifacts
from smart_traffic_agent.evaluation_metrics import (
    generation_time_seconds,
    generation_time_with_source,
    load_token_usage,
    normalize_token_usage,
)


def test_normalize_token_usage_derives_total_without_estimation() -> None:
    usage = normalize_token_usage(
        {"prompt_tokens": "10", "completion_tokens": 2, "request_count": 1},
        source="provider_response",
    )
    assert usage == {
        "request_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "usage_source": "provider_response",
    }


def test_load_token_usage_prefers_summary_then_metrics() -> None:
    summary = {"llm_usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    metrics = {"llm_usage": {"prompt_tokens": 30, "completion_tokens": 40}}
    assert load_token_usage((summary, "summary.llm_usage"), (metrics, "run_metrics.llm_usage"))["total_tokens"] == 7
    assert load_token_usage(({}, "summary.llm_usage"), (metrics, "run_metrics.llm_usage"))["usage_source"] == "run_metrics.llm_usage"
    assert load_token_usage(({}, "summary.llm_usage")) is None


def test_generation_time_excludes_execution_agent() -> None:
    assert generation_time_seconds(
        {
            "agent_total_durations_seconds": {
                "PlanningAgent": 1.25,
                "CodeGenerationAgent": 2.75,
                "ExecutionAgent": 99.0,
            }
        }
    ) == 4.0
    assert generation_time_seconds({"agent_total_durations_seconds": {}}) is None
    assert generation_time_with_source(
        {}, {"command": ["main.py", "--no-execute"], "elapsed_seconds": 12.5}
    ) == (12.5, "rq1_run_metadata.elapsed_seconds_code_only")
    assert generation_time_with_source(
        {}, {"command": ["main.py"], "elapsed_seconds": 12.5}
    ) == (None, None)


def test_offline_evaluate_aggregates_run_token_usage(tmp_path: Path) -> None:
    run = tmp_path / "run_001"
    (run / "execution" / "data").mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"steps": []}), encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"llm_usage": {"request_count": 2, "prompt_tokens": 10, "completion_tokens": 5}}),
        encoding="utf-8",
    )
    (run / "run_metrics.json").write_text(json.dumps({"total_elapsed_seconds": 1}), encoding="utf-8")
    (run / "execution" / "data" / "focas_api_output.csv").write_text("api_name,return_code,data\n", encoding="utf-8")
    result = evaluate(tmp_path)
    assert result["runs"][0]["token_usage"]["total_tokens"] == 15
    aggregate = result["aggregate"]["token_usage"]
    assert aggregate["runs_with_usage"] == 1
    assert aggregate["request_count"]["mean"] == 2.0
    assert aggregate["total_tokens"]["mean"] == 15.0
    assert aggregate["total_tokens"]["sd"] is None


def test_ground_truth_csv_row_contains_quality_efficiency_and_cost_metrics() -> None:
    row = csv_row(
        {
            "run_dir": "run_001",
            "scenario": "coordinate_motion",
            "framework_success": False,
            "failure_category": "compile_error",
            "generated_file_count": 3,
            "compilation_success_rate": 1.0,
            "completeness": 0.6,
            "consistency": 0.4,
            "overall_task_score": 2 / 3,
            "generation_time_seconds": 12.5,
            "generation_time_source": "test",
            "completeness_checks": {"artifacts_present": True},
            "consistency_components": {"required_function_coverage": 0.5},
            "token_usage": {
                "request_count": 2,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "usage_source": "provider",
            },
        }
    )
    assert row["compilation_success_rate"] == 1.0
    assert row["framework_success"] is False
    assert row["failure_category"] == "compile_error"
    assert row["generated_file_count"] == 3
    assert row["operation_coverage"] == 0.5
    assert row["generation_time_seconds"] == 12.5
    assert row["total_tokens"] == 15


def test_ground_truth_reads_fragment_as_part_of_evaluation_source(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "api_script.cpp").write_text('#include "scenario_fragment.inc"\n', encoding="utf-8")
    (generated / "scenario_fragment.inc").write_text("cnc_actf(handle, &actual_feed);\n", encoding="utf-8")
    (generated / "program.nc").write_text("O1234\nM30\n", encoding="utf-8")
    cpp, _, _, _ = read_artifacts(tmp_path)
    assert '#include "scenario_fragment.inc"' in cpp
    assert "cnc_actf(handle" in cpp


def test_compilation_success_checks_stdout_diagnostics(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "compile_stderr.txt").write_text("", encoding="utf-8")
    (generated / "compile_stdout.txt").write_text("api_script.cpp\n", encoding="utf-8")
    assert compilation_succeeded(tmp_path)
    (generated / "compile_stdout.txt").write_text(
        "file.cpp(4): error C2065: undeclared identifier\n", encoding="utf-8"
    )
    assert not compilation_succeeded(tmp_path)


def test_tool_management_does_not_require_nc_artifact(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (tmp_path / "summary.json").write_text(
        json.dumps({"scenario_type": "tool_management"}), encoding="utf-8"
    )
    (generated / "api_script.cpp").write_text(
        "\n".join(
            [
                "cnc_allclibhndl3(ip, port, timeout, &handle);",
                "cnc_statinfo(handle, &status);",
                "cnc_rdtofs(handle, number, type, &offset);",
                "cnc_freelibhndl(handle);",
            ]
        ),
        encoding="utf-8",
    )
    (generated / "compile_stdout.txt").write_text("api_script.cpp\n", encoding="utf-8")
    (generated / "compile_stderr.txt").write_text("", encoding="utf-8")

    result = evaluate_ground_truth(tmp_path)

    assert result["completeness_checks"]["artifacts_present"] is True
