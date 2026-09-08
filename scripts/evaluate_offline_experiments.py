"""Compute paper-facing metrics from existing SMPAgent run artifacts.

This script is intentionally offline: it reads files under ``runs/`` and never
connects to a simulator, invokes FOCAS, or captures new traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FOCAS_NAME = re.compile(r"\bcnc_[A-Za-z0-9_]+\b")
STATE_FIELDS = {"aut", "run", "motion", "alarm", "edit"}
FEED_FIELDS = {"feed_speed", "actual_feed"}
POSITION_PREFIXES = ("axis_data", "position_axis", "distance_to_go")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_key_values(value: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in value.split(";"):
        if "=" not in part:
            continue
        key, item = part.split("=", 1)
        if key:
            parsed[key.strip()] = item.strip()
    return parsed


def is_exact_focas_name(name: str) -> bool:
    return bool(re.fullmatch(r"cnc_[A-Za-z0-9_]+", name)) and "_or_" not in name


def planned_functions(plan: dict[str, Any] | None) -> set[str]:
    functions: set[str] = set()
    for step in (plan or {}).get("steps", []):
        for name in FOCAS_NAME.findall(str(step.get("protocol_function", ""))):
            if is_exact_focas_name(name):
                functions.add(name)
    return functions


def successful_functions(rows: list[dict[str, str]]) -> set[str]:
    return {
        row.get("api_name", "")
        for row in rows
        if is_exact_focas_name(row.get("api_name", ""))
        and row.get("return_code") == "0"
    }


def compile_succeeded(run_dir: Path) -> bool:
    executable = run_dir / "execution" / "api_script.exe"
    stderr_paths = [
        run_dir / "execution" / "compile_stderr.txt",
        run_dir / "generated" / "compile_stderr.txt",
    ]
    compiler_errors = any(
        path.exists() and path.read_text(encoding="utf-8", errors="replace").strip()
        for path in stderr_paths
    )
    return executable.exists() and executable.stat().st_size > 0 and not compiler_errors


def program_completed(rows: list[dict[str, str]]) -> bool:
    gates = [row for row in rows if row.get("api_name") == "program_completion_gate"]
    if gates:
        last = gates[-1]
        data = parse_key_values(last.get("data", ""))
        return last.get("return_code") == "0" and data.get("completed") == "true"
    summaries = [row for row in rows if row.get("api_name") == "host_execution_summary"]
    return bool(summaries) and parse_key_values(summaries[-1].get("data", "")).get(
        "program_completed"
    ) == "true"


def collect_dynamic_evidence(rows: list[dict[str, str]]) -> dict[str, Any]:
    states: set[tuple[str, str]] = set()
    positions_by_api: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    feeds: set[str] = set()
    field_values: dict[str, set[str]] = defaultdict(set)
    field_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        if row.get("return_code") != "0":
            continue
        api = row.get("api_name", "")
        values = parse_key_values(row.get("data", ""))
        if api == "cnc_statinfo":
            if "run" in values and "motion" in values:
                states.add((values["run"], values["motion"]))
            selected = {key: value for key, value in values.items() if key in STATE_FIELDS}
        elif api == "cnc_actf":
            feeds.update(values[key] for key in FEED_FIELDS if key in values)
            selected = {key: value for key, value in values.items() if key in FEED_FIELDS}
        elif api in {"cnc_absolute", "cnc_rdposition"}:
            coordinate_keys = sorted(
                key for key in values if key.startswith(POSITION_PREFIXES)
            )
            if coordinate_keys:
                positions_by_api[api].add(tuple(values[key] for key in coordinate_keys))
            selected = {key: values[key] for key in coordinate_keys}
        else:
            selected = {}

        for key, value in selected.items():
            field_id = f"{api}.{key}"
            field_counts[field_id] += 1
            field_values[field_id].add(value)

    observed_fields = {key for key, count in field_counts.items() if count >= 2}
    varying_fields = {key for key in observed_fields if len(field_values[key]) >= 2}
    return {
        "states": sorted([list(value) for value in states]),
        "state_has_idle": ["0", "0"] in [list(value) for value in states],
        "state_has_started_static": ["1", "0"] in [list(value) for value in states],
        "state_has_running_motion": ["3", "1"] in [list(value) for value in states],
        "position_unique_count": max(
            (len(values) for values in positions_by_api.values()), default=0
        ),
        "position_changed": any(len(values) >= 2 for values in positions_by_api.values()),
        "feed_unique_count": len(feeds),
        "feed_changed": len(feeds) >= 2,
        "feed_has_nonzero": any(value not in {"", "0"} for value in feeds),
        "observed_dynamic_field_count": len(observed_fields),
        "varying_dynamic_field_count": len(varying_fields),
        "discriminability_proxy": (
            len(varying_fields) / len(observed_fields) if observed_fields else None
        ),
        "dynamic_field_values": {
            key: sorted(field_values[key]) for key in sorted(observed_fields)
        },
    }


def semantic_completeness(log_path: Path) -> dict[str, int | float | None]:
    total = 0
    complete = 0
    if not log_path.exists():
        return {"total": 0, "complete": 0, "ratio": None}
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            name = str(row.get("protocol_function", ""))
            if not is_exact_focas_name(name):
                continue
            total += 1
            response = row.get("response")
            is_complete = (
                bool(row.get("step_id"))
                and bool(row.get("phase"))
                and isinstance(row.get("input_parameters"), dict)
                and row.get("status_code") is not None
                and isinstance(response, dict)
                and bool(response.get("function"))
                and "data" in response
                and bool(row.get("semantic_label"))
            )
            complete += int(is_complete)
    return {"total": total, "complete": complete, "ratio": complete / total if total else None}


def mean_sd(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) >= 2 else None,
        "median": statistics.median(values) if values else None,
    }


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def seconds_summary(stats: dict[str, float | int | None]) -> str:
    if stats["mean"] is None:
        return "N/A"
    if stats["sd"] is None:
        return f"{stats['mean']:.1f}"
    return f"{stats['mean']:.1f} +/- {stats['sd']:.1f}"


def evaluate(runs_root: Path) -> dict[str, Any]:
    run_results: list[dict[str, Any]] = []
    union_planned: set[str] = set()
    union_successful: set[str] = set()
    pooled_dynamic_values: dict[str, set[str]] = defaultdict(set)
    pooled_dynamic_counts: dict[str, int] = defaultdict(int)

    for run_dir in sorted(path for path in runs_root.glob("run_*") if path.is_dir()):
        plan = load_json(run_dir / "plan.json")
        metrics = load_json(run_dir / "run_metrics.json")
        rows = load_csv(run_dir / "execution" / "data" / "focas_api_output.csv")
        planned = planned_functions(plan)
        successful = successful_functions(rows)
        covered = planned & successful
        dynamic = collect_dynamic_evidence(rows)
        semantic = semantic_completeness(run_dir / "execution" / "api_logs.jsonl")
        compile_ok = compile_succeeded(run_dir)
        completed = program_completed(rows)
        scenario_success = all(
            [
                compile_ok,
                completed,
                dynamic["position_changed"],
                dynamic["feed_changed"],
                dynamic["feed_has_nonzero"],
                dynamic["state_has_running_motion"],
            ]
        )
        workflow_success = metrics.get("success") if metrics is not None else None
        durations = (metrics or {}).get("agent_total_durations_seconds", {})
        generation_time = None
        if metrics is not None:
            generation_time = float(durations.get("PlanningAgent", 0)) + float(
                durations.get("CodeGenerationAgent", 0)
            )
        pcap = run_dir / "execution" / "data" / "focas_capture.pcap"

        for field_id, values in dynamic["dynamic_field_values"].items():
            pooled_dynamic_counts[field_id] += sum(
                1
                for row in rows
                if row.get("return_code") == "0"
                and row.get("api_name") == field_id.split(".", 1)[0]
                and field_id.split(".", 1)[1] in parse_key_values(row.get("data", ""))
            )
            pooled_dynamic_values[field_id].update(values)

        union_planned.update(planned)
        union_successful.update(successful)
        run_results.append(
            {
                "run": run_dir.name,
                "compile_success": compile_ok,
                "program_completed": completed,
                "scenario_task_success": scenario_success,
                "workflow_strict_success": workflow_success,
                "planned_api_count": len(planned),
                "successful_planned_api_count": len(covered),
                "api_coverage": len(covered) / len(planned) if planned else None,
                "successful_api_count": len(successful),
                "api_log_count": len(rows),
                "generation_time_seconds": generation_time,
                "total_elapsed_seconds": (metrics or {}).get("total_elapsed_seconds"),
                "token_usage": None,
                "pcap_exists": pcap.exists(),
                "pcap_bytes": pcap.stat().st_size if pcap.exists() else 0,
                "semantic_annotation_completeness": semantic,
                "dynamic_evidence": dynamic,
                "planned_apis": sorted(planned),
                "successful_planned_apis": sorted(covered),
                "missing_or_unsuccessful_planned_apis": sorted(planned - successful),
            }
        )

    observed_pooled = {
        field_id for field_id, count in pooled_dynamic_counts.items() if count >= 2
    }
    varying_pooled = {
        field_id
        for field_id in observed_pooled
        if len(pooled_dynamic_values[field_id]) >= 2
    }
    coverages = [row["api_coverage"] for row in run_results if row["api_coverage"] is not None]
    generation_times = [
        row["generation_time_seconds"]
        for row in run_results
        if row["generation_time_seconds"] is not None
    ]
    total_times = [
        row["total_elapsed_seconds"]
        for row in run_results
        if row["total_elapsed_seconds"] is not None
    ]
    semantic_total = sum(row["semantic_annotation_completeness"]["total"] for row in run_results)
    semantic_complete = sum(
        row["semantic_annotation_completeness"]["complete"] for row in run_results
    )
    strict_known = [
        row["workflow_strict_success"]
        for row in run_results
        if row["workflow_strict_success"] is not None
    ]

    return {
        "scope": {
            "mode": "offline_existing_artifacts_only",
            "simulator_started": False,
            "new_traffic_generated": False,
            "scenario": "FOCAS coordinate motion",
            "run_count": len(run_results),
        },
        "metric_definitions": {
            "scenario_task_success": "Compiled executable, completed NC program, observed changing position and non-zero changing feed, and captured run=3/motion=1.",
            "workflow_strict_success": "Top-level success value recorded in run_metrics.json; optional API errors can make this stricter than scenario completion.",
            "api_coverage": "Distinct planned exact cnc_* functions with at least one EW_OK return divided by distinct planned exact cnc_* functions, computed per run.",
            "dataset_union_coverage": "Union of successful planned exact cnc_* functions divided by union of planned exact cnc_* functions across all runs.",
            "discriminability_proxy": "Observed state/position/feed response fields with at least two distinct values divided by such fields with at least two samples.",
            "semantic_annotation_completeness_proxy": "FOCAS API log rows containing step, phase, input object, return code, response function/data, and api_call semantic label divided by FOCAS API log rows.",
            "generation_time": "PlanningAgent plus CodeGenerationAgent durations; excludes ExecutionAgent time.",
        },
        "aggregate": {
            "scenario_task_success": {
                "successes": sum(row["scenario_task_success"] for row in run_results),
                "n": len(run_results),
                "rate": sum(row["scenario_task_success"] for row in run_results)
                / len(run_results)
                if run_results
                else None,
            },
            "compile_success": {
                "successes": sum(row["compile_success"] for row in run_results),
                "n": len(run_results),
                "rate": sum(row["compile_success"] for row in run_results) / len(run_results)
                if run_results
                else None,
            },
            "workflow_strict_success": {
                "successes": sum(bool(value) for value in strict_known),
                "n": len(strict_known),
                "rate": sum(bool(value) for value in strict_known) / len(strict_known)
                if strict_known
                else None,
            },
            "per_run_api_coverage": mean_sd(coverages),
            "dataset_union_api_coverage": {
                "successful": len(union_planned & union_successful),
                "planned": len(union_planned),
                "rate": len(union_planned & union_successful) / len(union_planned)
                if union_planned
                else None,
            },
            "generation_time_seconds": mean_sd(generation_times),
            "total_workflow_time_seconds": mean_sd(total_times),
            "discriminability_proxy": {
                "varying_fields": len(varying_pooled),
                "observed_fields": len(observed_pooled),
                "rate": len(varying_pooled) / len(observed_pooled)
                if observed_pooled
                else None,
                "varying_field_names": sorted(varying_pooled),
                "constant_field_names": sorted(observed_pooled - varying_pooled),
            },
            "semantic_annotation_completeness_proxy": {
                "complete": semantic_complete,
                "total": semantic_total,
                "rate": semantic_complete / semantic_total if semantic_total else None,
            },
            "token_usage": None,
        },
        "runs": run_results,
        "unmeasured": {
            "framework_baselines": ["GPT-Engineer", "MetaGPT", "ChatDev"],
            "ablation_variants": ["Single-Agent", "w/o RouterAgent"],
            "protocols": ["EZSocket", "LSV/2", "S7Comm"],
            "metrics": ["token_usage", "packet-level semantic consistency", "RQ4 field recovery accuracy"],
        },
    }


def write_csv_report(path: Path, result: dict[str, Any]) -> None:
    columns = [
        "run",
        "compile_success",
        "program_completed",
        "scenario_task_success",
        "workflow_strict_success",
        "planned_api_count",
        "successful_planned_api_count",
        "api_coverage",
        "api_log_count",
        "generation_time_seconds",
        "total_elapsed_seconds",
        "token_usage",
        "position_unique_count",
        "feed_unique_count",
        "state_has_idle",
        "state_has_started_static",
        "state_has_running_motion",
        "discriminability_proxy",
        "semantic_annotation_completeness_proxy",
        "pcap_bytes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["runs"]:
            dynamic = row["dynamic_evidence"]
            writer.writerow(
                {
                    **{key: row.get(key) for key in columns},
                    "position_unique_count": dynamic["position_unique_count"],
                    "feed_unique_count": dynamic["feed_unique_count"],
                    "state_has_idle": dynamic["state_has_idle"],
                    "state_has_started_static": dynamic["state_has_started_static"],
                    "state_has_running_motion": dynamic["state_has_running_motion"],
                    "discriminability_proxy": dynamic["discriminability_proxy"],
                    "semantic_annotation_completeness_proxy": row[
                        "semantic_annotation_completeness"
                    ]["ratio"],
                }
            )


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    aggregate = result["aggregate"]
    task = aggregate["scenario_task_success"]
    compile_result = aggregate["compile_success"]
    strict = aggregate["workflow_strict_success"]
    coverage = aggregate["per_run_api_coverage"]
    union_coverage = aggregate["dataset_union_api_coverage"]
    discriminability = aggregate["discriminability_proxy"]
    semantic = aggregate["semantic_annotation_completeness_proxy"]
    generation_time = aggregate["generation_time_seconds"]

    lines = [
        "# SMPAgent Offline Experiment Results",
        "",
        "Evaluation date: 2026-09-02",
        "",
        "Scope: existing artifacts from `runs/focas_nc_position_main/run_001` through `run_006`. "
        "No simulator was started and no new traffic was generated.",
        "",
        "## Metric definitions",
        "",
        "- Task Success: compiled executable + completed NC program + changing position + changing, non-zero feed + observed `run=3, motion=1`.",
        "- API Coverage: successful (`EW_OK`) distinct planned exact `cnc_*` functions / distinct planned exact `cnc_*` functions, first computed per run.",
        "- Discriminability: offline response-field proxy over status, coordinate, and feed fields; it is not packet-byte discriminability.",
        "- Semantic Consistency: annotation-completeness proxy; it is not packet-to-operation alignment accuracy.",
        "- Time: PlanningAgent + CodeGenerationAgent time, excluding ExecutionAgent, because RQ1 specifies code-generation-level comparison.",
        "- Each `run_*` directory is one repeated system run. No inferential test is reported for six non-random historical runs.",
        "",
        "## RQ1 - Comparison with existing frameworks",
        "",
        "| Scenario | Method | Task Succ. | Compile Succ. | API Cov. | Time (s) | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| Coordinate Motion | GPT-Engineer | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| Coordinate Motion | MetaGPT | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| Coordinate Motion | ChatDev | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        f"| Coordinate Motion | SMPAgent | {percent(task['rate'])} ({task['successes']}/{task['n']}) | {percent(compile_result['rate'])} ({compile_result['successes']}/{compile_result['n']}) | {percent(coverage['mean'])} +/- {percent(coverage['sd'])} (n={coverage['n']}) | {seconds_summary(generation_time)} (n={generation_time['n']}) | AUTHOR_INPUT_NEEDED |",
        "| Program Management | all methods | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| Tool Management | all methods | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "",
        "## RQ2 - Dataset quality",
        "",
        "| Protocol | Coverage | Discriminability | Semantic Consistency |",
        "|---|---:|---:|---:|",
        f"| FOCAS | {percent(union_coverage['rate'])} ({union_coverage['successful']}/{union_coverage['planned']}) | {percent(discriminability['rate'])} ({discriminability['varying_fields']}/{discriminability['observed_fields']}) proxy | {percent(semantic['rate'])} ({semantic['complete']}/{semantic['total']}) annotation proxy |",
        "| EZSocket | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| LSV/2 | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| S7Comm | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "",
        "## RQ3 - Ablation study",
        "",
        "| Method | Task Succ. | API Cov. | Time (s) | Tokens |",
        "|---|---:|---:|---:|---:|",
        "| Single-Agent | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        "| w/o RouterAgent | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED | AUTHOR_INPUT_NEEDED |",
        f"| SMPAgent | {percent(task['rate'])} ({task['successes']}/{task['n']}) | {percent(coverage['mean'])} +/- {percent(coverage['sd'])} | {seconds_summary(generation_time)} | AUTHOR_INPUT_NEEDED |",
        "",
        "## Per-run audit table",
        "",
        "| Run | Compile | Program complete | Task success | Strict workflow success | API coverage | Position values | Feed values | Idle | Started/static | Running/motion | PCAP bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["runs"]:
        dynamic = row["dynamic_evidence"]
        strict_value = row["workflow_strict_success"]
        lines.append(
            "| {run} | {compile} | {complete} | {task} | {strict} | {coverage} | {position} | {feed} | {idle} | {static} | {motion} | {pcap} |".format(
                run=row["run"],
                compile="yes" if row["compile_success"] else "no",
                complete="yes" if row["program_completed"] else "no",
                task="yes" if row["scenario_task_success"] else "no",
                strict="N/A" if strict_value is None else ("yes" if strict_value else "no"),
                coverage=percent(row["api_coverage"]),
                position=dynamic["position_unique_count"],
                feed=dynamic["feed_unique_count"],
                idle="yes" if dynamic["state_has_idle"] else "no",
                static="yes" if dynamic["state_has_started_static"] else "no",
                motion="yes" if dynamic["state_has_running_motion"] else "no",
                pcap=row["pcap_bytes"],
            )
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        f"The scenario-level criterion is satisfied in {task['successes']} of {task['n']} historical runs. "
        f"The repository's stricter top-level workflow flag is positive in {strict['successes']} of {strict['n']} runs with metrics. "
        "The difference is caused by optional/unsupported API failures after some runs had already completed the NC program and captured dynamic evidence.",
        "",
        "These results are descriptive. The six runs were produced during iterative development, not by a preregistered randomized benchmark. "
        "They therefore provide a current engineering baseline, not a fair comparative claim against other frameworks.",
        "",
        "RQ4 field-recovery accuracy cannot be computed from the current artifacts because no ground-truth packet-field annotations or reverse-engineering predictions are present.",
        "",
        "## AUTHOR_INPUT_NEEDED",
        "",
        "- Run GPT-Engineer, MetaGPT, and ChatDev on the same fixed task set and record compile outcome, task outcome, covered APIs, generation time, and token usage.",
        "- Run Single-Agent and w/o RouterAgent variants using the same prompts, task set, stopping criteria, and repetitions.",
        "- Provide Program Management and Tool Management runs.",
        "- Provide EZSocket, LSV/2, and S7Comm logs and captures.",
        "- Add token accounting to each agent/model call before reporting Tokens.",
        "- Define packet-level ground truth before replacing the RQ2 proxy metrics or reporting RQ4 recovery accuracy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/focas_nc_position_main"),
        help="Directory containing run_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation"),
        help="Directory for JSON, CSV, and Markdown reports.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.runs_root)
    json_path = args.output_dir / "offline_experiment_results.json"
    csv_path = args.output_dir / "offline_experiment_results.csv"
    markdown_path = args.output_dir / "offline_experiment_tables.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv_report(csv_path, result)
    write_markdown_report(markdown_path, result)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
