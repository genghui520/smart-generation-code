"""Evaluate normalized GPT-Engineer, MetaGPT, and ChatDev RQ1 runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_rq1_ground_truth import csv_row, evaluate, load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    runs = sorted(path for path in args.suite.glob("*/*/run_*") if (path / "generated").is_dir())
    results = []
    rows = []
    for run in runs:
        result = evaluate(run)
        metadata_path = run / "framework_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        normalized_summary = load_json(run / "summary.json")
        result["framework"] = metadata.get("framework", run.parents[1].name)
        result["model"] = metadata.get("model")
        result["base_url"] = metadata.get("base_url")
        result["framework_success"] = bool(normalized_summary.get("framework_success"))
        results.append(result)
        rows.append({
            "framework": result["framework"],
            "model": result["model"],
            "base_url": result["base_url"],
            "framework_success": result["framework_success"],
            **csv_row(result),
        })
    report = args.suite / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "baseline_ground_truth_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = report / "baseline_ground_truth_metrics.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
