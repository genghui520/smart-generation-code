from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.models import ExecutionPlan, NcProgramSpec, PlanStep
from smart_traffic_agent.offline_audit import reaudit_saved_execution


def _construct(model, payload):
    allowed = model.__dataclass_fields__
    return model(**{key: value for key, value in payload.items() if key in allowed})


def load_plan(path: Path) -> ExecutionPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["steps"] = [_construct(PlanStep, row) for row in data.get("steps", [])]
    data["nc_program_spec"] = _construct(NcProgramSpec, data.get("nc_program_spec", {}))
    data.pop("code_spec", None)
    return _construct(ExecutionPlan, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-audit saved FOCAS logs and transactions without CNC execution.")
    parser.add_argument("batch_dir", type=Path, help="Saved batch directory or its execution subdirectory")
    parser.add_argument("--plan", type=Path, help="Plan JSON; defaults to batch_dir/plan.json")
    args = parser.parse_args()
    batch_dir = args.batch_dir.resolve()
    execution_dir = batch_dir / "execution" if (batch_dir / "execution").is_dir() else batch_dir
    plan_path = args.plan or (execution_dir.parent / "plan.json")
    audit = reaudit_saved_execution(execution_dir, load_plan(plan_path))
    print(json.dumps({"execution_dir": str(execution_dir), "audit": audit}, ensure_ascii=False))
    return 0 if audit.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
