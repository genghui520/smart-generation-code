"""Render the framework-owned NCGuide motion lifecycle (no model UI code)."""
from __future__ import annotations

import json
from pathlib import Path
import re


def render_motion_driver(nc_program: str) -> str:
    lines = [line.strip().rstrip(";").strip() for line in nc_program.replace("\r", "").splitlines()
             if line.strip() and line.strip() != "%"]
    if not lines or not re.fullmatch(r"O\d+(?:\s*\([^)]*\))?", lines[0], re.I):
        raise ValueError("EXECUTION_CONTRACT_ERROR: motion program requires an O-number line")
    if not re.search(r"\bM0?30\b", lines[-1], re.I):
        raise ValueError("EXECUTION_CONTRACT_ERROR: motion program requires a final M30")
    preferred = int(re.match(r"O(\d+)", lines[0], re.I)[1])
    if not 1 <= preferred < 8000:
        preferred = 5700
    body = []
    for index, line in enumerate(lines[1:], 1):
        # Sequence tags let the runtime observe fast blocks without estimating
        # progress from timing alone. Count and execute this exact payload.
        if not re.match(r"N\d+\b", line, re.I) and not line.startswith("("):
            line = f"N{index * 10} {line}"
        body.append(line)
    encoded_body = json.dumps("\n".join(body) + "\n", ensure_ascii=True)
    source = (Path(__file__).parent / "templates" / "ncguide_motion_driver.cpp").read_text(encoding="utf-8")
    return source.replace("__NC_BODY__", encoded_body).replace("__PREFERRED_PROGRAM__", str(preferred))
