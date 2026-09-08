"""Compile all generated C/C++ files with the project's x86 MSVC/FOCAS toolchain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.agent_tools.cpp_execution import (
    CompileCppInput,
    CompileGeneratedCppTool,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    sources = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"})
    rows = []
    for source in sources:
        exe = source.with_suffix(".compile_test.exe")
        result = CompileGeneratedCppTool().invoke(CompileCppInput(source, exe, source.parent))
        rows.append({
            "source": str(source.relative_to(root)),
            "success": result.success,
            "return_code": result.return_code,
            "command": result.command,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
        for path in (exe, source.with_suffix(".obj")):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    payload = {
        "compile_success": bool(rows) and all(row["success"] for row in rows),
        "source_count": len(sources),
        "results": rows,
    }
    (root / "compile_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
