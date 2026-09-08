"""Score code-generation-only baseline artifacts with one shared rubric."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REQUIRED_APIS = ["cnc_allclibhndl3", "cnc_statinfo", "cnc_actf", "cnc_absolute"]
OPTIONAL_LIFECYCLE_APIS = ["cnc_dwnstart3", "cnc_download3", "cnc_dwnend3", "cnc_search"]
NC_PATTERNS = {
    "program_number": r"(?:O|o)\d{4,8}|kPreferredProgramNumber\s*=\s*\d{1,8}",
    "absolute_mode": r"\bG90\b",
    "work_offset": r"\bG54\b",
    # Do not require a leading word boundary because generated C/C++ string
    # literals often contain the source characters ``\\nG00`` / ``\\nG01``.
    "rapid": r"G00\b|G0\b",
    "linear": r"G01\b|G1\b",
    "program_end": r"\bM30\b",
}


def _read_sources(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"} and not path.name.lower().startswith("compile_"):
            try:
                files[str(path.relative_to(root))] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return files


def score(root: Path) -> dict:
    files = _read_sources(root)
    text = "\n".join(files.values())
    api_hits = {api: bool(re.search(rf"\b{re.escape(api)}\b", text)) for api in REQUIRED_APIS}
    nc_hits = {name: bool(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)) for name, pattern in NC_PATTERNS.items()}
    feed_values = set(re.findall(r"\bF\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE))
    nc_hits["multiple_feed_values"] = len(feed_values) >= 2
    lifecycle_hits = {api: bool(re.search(rf"\b{re.escape(api)}\b", text)) for api in OPTIONAL_LIFECYCLE_APIS}
    api_coverage = sum(api_hits.values()) / len(api_hits) if api_hits else 0.0
    nc_completeness = sum(nc_hits.values()) / len(nc_hits) if nc_hits else 0.0
    compile_status = None
    for path in (root / "compile_result.json", root / "generated" / "compile_result.json", root / "metadata.json"):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "compile_success" in data:
                    compile_status = bool(data["compile_success"])
            except (OSError, json.JSONDecodeError):
                pass
    task_success = api_coverage == 1.0 and nc_completeness == 1.0 and compile_status is True
    return {
        "source_file_count": len(files),
        "api_hits": api_hits,
        "api_coverage": api_coverage,
        "nc_requirement_hits": nc_hits,
        "nc_completeness": nc_completeness,
        "lifecycle_api_hits": lifecycle_hits,
        "compile_success": compile_status,
        "task_success": task_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.artifact_root)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
