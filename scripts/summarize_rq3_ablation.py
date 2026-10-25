"""Build the paper-facing RQ3 table from independently evaluated runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for evaluation_path in args.root.glob("**/rq3_ground_truth_evaluation.json"):
        run_dir = evaluation_path.parent
        if any((parent / "INVALIDATION.json").exists() for parent in (run_dir, *run_dir.parents)):
            continue
        metadata_path = run_dir / "rq1_run_metadata.json"
        if not metadata_path.exists():
            continue
        evaluation = load_json(evaluation_path)
        metadata = load_json(metadata_path)
        usage = evaluation.get("token_usage") or {}
        # Debug/setup attempts without persisted provider usage are not formal
        # table samples. They remain on disk for auditability.
        if usage.get("total_tokens") is None:
            continue
        variant = str(metadata.get("variant", ""))
        scenario = str(metadata.get("scenario") or evaluation.get("scenario", ""))
        row = {
            "scenario": scenario,
            "method": "Single-Agent" if variant == "single-agent" else "w/o RouterAgent",
            "variant": variant,
            "score_percent": round(float(evaluation["overall_task_score"]) * 100, 1),
            "time_seconds": round(float(evaluation["generation_time_seconds"]), 1),
            "tokens": int(usage["total_tokens"]),
            "compile_success": float(evaluation["compilation_success_rate"]),
            "completeness": float(evaluation["completeness"]),
            "consistency": float(evaluation["consistency"]),
            "generation_return_code": metadata.get("return_code"),
            "run_dir": str(run_dir),
            "evaluation": str(evaluation_path),
        }
        candidates.setdefault((variant, scenario), []).append(row)

    rows: list[dict[str, Any]] = []
    for key, values in candidates.items():
        values.sort(key=lambda item: Path(item["run_dir"]).stat().st_mtime)
        rows.append(values[-1])
    scenario_order = {
        "coordinate_motion": 0,
        "program_management": 1,
        "tool_management": 2,
    }
    method_order = {"Single-Agent": 0, "w/o RouterAgent": 1}
    rows.sort(key=lambda row: (scenario_order.get(row["scenario"], 99), method_order.get(row["method"], 99)))

    json_path = args.root / "rq3_ablation_table.json"
    csv_path = args.root / "rq3_ablation_table.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0]) if rows else ["scenario", "method", "score_percent", "time_seconds", "tokens"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0 if len(rows) == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
