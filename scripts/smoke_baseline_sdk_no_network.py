"""Offline/no-network smoke test for the shared baseline SDK staging."""
from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.baseline_sdk import stage_sdk


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="rq1_baseline_sdk_") as raw:
        root = Path(raw)
        manifest = stage_sdk(root / "sdk")
        assert (root / "sdk" / "Fwlib32.h").read_bytes()
        assert (root / "sdk" / "Fwlib32.lib").read_bytes()
        assert manifest["header_sha256"] and manifest["library_sha256"]
        # Parse runners only; no runner main() and no HTTP transport is called.
        for name in ("run_gpt_engineer_baseline.py", "run_metagpt_baseline.py", "run_chatdev1_baseline.py"):
            ast.parse((Path(__file__).parent / name).read_text(encoding="utf-8"))
    print("baseline SDK staging and no-network runner parse smoke: OK")


if __name__ == "__main__":
    main()
