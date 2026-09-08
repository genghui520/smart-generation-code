"""Build an auditable comparison table from real baseline artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("evaluation/baselines")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    entries = [
        ("GPT-Engineer", ROOT / "gpt-engineer", False),
        ("MetaGPT", ROOT / "MetaGPT", True),
        ("ChatDev 1.0", ROOT / "ChatDev1", False),
    ]
    rows = []
    for method, path, extracted in entries:
        metadata = load(path / "metadata.json")
        score_path = path / ("extracted_score.json" if extracted else "score.json")
        score = load(score_path)
        row = {
            "scenario": "Coordinate Motion",
            "method": method,
            "framework_success": bool(metadata.get("success")),
            "task_success": bool(metadata.get("success")) and bool(score.get("task_success")),
            "compile_success": bool(score.get("compile_success")),
            "api_coverage": score.get("api_coverage"),
            "nc_completeness": score.get("nc_completeness"),
            "time_seconds": metadata.get("duration_seconds"),
            "prompt_tokens": metadata.get("usage", {}).get("prompt_tokens"),
            "completion_tokens": metadata.get("usage", {}).get("completion_tokens"),
            "total_tokens": metadata.get("usage", {}).get("total_tokens"),
            "commit": metadata.get("commit"),
            "artifact_kind": "serialized_response_extraction" if extracted else "framework_output",
            "failure": metadata.get("error"),
        }
        rows.append(row)

    payload = {
        "scope": {
            "task": "evaluation/baseline_task.txt",
            "model": "gpt-5.6-sol",
            "temperature": 0.1,
            "code_generation_only": True,
            "simulator_started": False,
            "new_traffic_generated": False,
            "repetitions_per_method": 1,
        },
        "metric_note": (
            "Task success requires a framework-produced artifact with 100% required API coverage, "
            "100% NC requirement completeness, and successful compilation. MetaGPT's extracted "
            "serialized response is diagnostic only and cannot satisfy framework task success."
        ),
        "rows": rows,
        "excluded": {
            "ChatDev 2.0": "Environment-probe run only; paper baseline uses ChatDev 1.0.",
            "SMPAgent fresh matched run": "Two attempts were blocked before output by upstream HTTP 524; historical simulator runs are not mixed into this code-only table.",
        },
    }
    (ROOT.parent / "baseline_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT.parent / "baseline_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Coordinate-motion code-generation baseline comparison",
        "",
        "Common setting: one run per method, `gpt-5.6-sol`, temperature 0.1, no simulator and no new traffic.",
        "",
        "| Method | Task success | Compile success | API coverage | NC completeness | Time (s) | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {'yes' if row['task_success'] else 'no'} | "
            f"{'yes' if row['compile_success'] else 'no'} | {100*row['api_coverage']:.1f}% | "
            f"{100*row['nc_completeness']:.1f}% | {row['time_seconds']:.1f} | {row['total_tokens']} |"
        )
    lines += [
        "",
        "MetaGPT produced code inside a serialized Engineer2 response, but its command parser rejected it before file creation; the exact extracted response also failed FOCAS compilation and is therefore reported as task failure.",
        "",
        "ChatDev 2.0 is excluded because the paper baseline is ChatDev 1.0. A fresh SMPAgent matched run is not reported because two attempts ended at the provider with HTTP 524 before the planner returned any output.",
    ]
    (ROOT.parent / "baseline_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
