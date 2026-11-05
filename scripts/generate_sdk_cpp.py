"""Generate statements-only FOCAS C++ from a structured plan and Fwlib32.h.

Usage:
    python scripts/generate_sdk_cpp.py --plan plan.json --output scenario_fragment.inc

The output is intended for the existing focas_harness.cpp pipeline.  It is
executable after that harness supplies ``handle``, ``api``, ``ctx`` and
``ret``; this tool does not invent a second harness or a second ABI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Running ``python scripts/generate_sdk_cpp.py`` places ``scripts/`` rather
# than the repository root on sys.path.  Add the project root explicitly so
# the same entry point works outside an installed package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_traffic_agent.deterministic_cpp_assembler import DeterministicAssemblyError
from smart_traffic_agent.sdk_cpp_generator import generate_core_cpp_from_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="JSON ExecutionPlan subset")
    parser.add_argument("--output", required=True, type=Path, help="statements-only C++ output")
    parser.add_argument("--sdk-header", type=Path, help="override the active Fwlib32.h")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        core = generate_core_cpp_from_plan(plan, sdk_header=args.sdk_header)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(core.rstrip() + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, DeterministicAssemblyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output": str(args.output), "bytes": len(core.encode("utf-8"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
