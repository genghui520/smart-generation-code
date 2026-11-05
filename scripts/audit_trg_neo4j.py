"""Audit and explicitly migrate the Neo4j TRG graph.

The default mode writes only after creating a JSON backup.  ``--dry-run`` is
read-only and is useful for reviewing the evidence classification first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ``python scripts/...`` sets sys.path to the scripts directory on Windows;
# add the project root explicitly so the command is independent of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.trg_audit import cleanup_confirmed_graph
from smart_traffic_agent.traffic_graph import TrafficRelationshipService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="only audit and create a backup")
    parser.add_argument("--output", type=Path, default=None, help="backup/report directory")
    args = parser.parse_args()
    service = TrafficRelationshipService.from_environment()
    if not service.available or service.store is None:
        raise SystemExit("Neo4j is unavailable; no graph migration was attempted.")
    directory = args.output or Path("runs") / "trg_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        report = cleanup_confirmed_graph(service.store, backup_dir=directory, dry_run=args.dry_run)
        summary = {
            "backup": report["backup"], "dry_run": report["dry_run"],
            "before": report["before"], "after": report["after"],
            "deleted_relations": report["deleted_relations"],
            "proposed_before": report["proposed_before"],
            "unconfirmed_before": report["unconfirmed_before"],
            "deleted_unconfirmed": report["deleted_unconfirmed"],
            "deleted_invalid_confirmed": report["deleted_invalid_confirmed"],
            "deleted_duplicates": report["deleted_duplicates"],
            "retained_filters": report["retained_filters"],
            "retained_prepares": report["retained_prepares"],
            "confirmed_audit_before_cleanup": report["confirmed_audit_before_cleanup"],
            "confirmed_audit_after_cleanup": report["confirmed_audit_after_cleanup"],
        }
        (directory / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        service.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
