"""Build global and per-API FOCAS return-code knowledge from cached reference pages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smart_traffic_agent.rag.focas_return_codes import (  # noqa: E402
    build_return_code_catalog,
    read_jsonl,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv_outputs(output_dir: Path, catalog: dict, per_api: list[dict]) -> None:
    with (output_dir / "return_codes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "value", "name", "same_value_names", "cnc_meaning", "cnc_handling",
            "pmc_meaning", "pmc_handling", "api_count", "site_url",
        ])
        writer.writeheader()
        for row in catalog["codes"]:
            descriptions = row["global_descriptions"]
            writer.writerow({
                "value": row["value"], "name": row["name"],
                "same_value_names": "; ".join(row["same_value_names"]),
                "cnc_meaning": descriptions.get("cnc", {}).get("meaning", ""),
                "cnc_handling": descriptions.get("cnc", {}).get("handling", ""),
                "pmc_meaning": descriptions.get("pmc", {}).get("meaning", ""),
                "pmc_handling": descriptions.get("pmc", {}).get("handling", ""),
                "api_count": row["api_count"], "site_url": row["site_url"],
            })
    with (output_dir / "api_return_codes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "function", "code_value", "code_name", "api_explanation", "value_source",
            "extraction_method", "source_url", "source_chunk_id",
        ])
        writer.writeheader()
        for row in per_api:
            writer.writerow({
                "function": row["function"], "code_value": row["code_value"],
                "code_name": row["code_name"],
                "api_explanation": "\n\n".join(row["api_explanations"]),
                "value_source": row["value_source"],
                "extraction_method": row["extraction_method"],
                "source_url": row["source_url"],
                "source_chunk_id": row["source_chunk_id"],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "rag_indexes/focas/chunks.jsonl")
    parser.add_argument("--sdk-root", type=Path, default=Path(r"C:\Lib\FOCAS2 Library\Fwlib"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "rag_indexes/focas")
    args = parser.parse_args()
    headers = sorted(args.sdk_root.rglob("Fwlib32.h")) if args.sdk_root.is_dir() else []
    chunks = read_jsonl(args.chunks)
    catalog, per_api, rag_chunks = build_return_code_catalog(chunks, headers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "return_codes.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    write_jsonl(args.output_dir / "api_return_codes.jsonl", per_api)
    write_jsonl(args.output_dir / "return_code_chunks.jsonl", rag_chunks)
    write_csv_outputs(args.output_dir, catalog, per_api)
    print(json.dumps({
        "global_code_names": len(catalog["codes"]),
        "api_error_sections": catalog["api_error_section_count"],
        "per_api_rows": len(per_api),
        "unresolved_names": catalog["unresolved_api_code_names"],
        "sdk_conflicts": catalog["sdk_cross_series_conflicts"],
        "site_conflicts": catalog["site_conflicts"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
