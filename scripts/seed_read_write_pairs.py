"""Back up Neo4j and add confirmed FOCAS read/write readback relations."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.read_write_pairs import discover_read_write_pairs, seed_read_write_pair_relations
from smart_traffic_agent.traffic_graph import TrafficRelationshipService
from smart_traffic_agent.trg_audit import audit_graph, read_graph, write_backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rag_indexes/focas/function_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    service = TrafficRelationshipService.from_environment()
    if not service.available or service.store is None:
        raise SystemExit("Neo4j is unavailable; no relation migration was attempted.")
    output_dir = args.output or Path("runs") / "trg_read_write_pairs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes, relations = read_graph(service.store)
    before = audit_graph(service.store)
    backup = write_backup(output_dir, nodes, relations, before)
    pairs, excluded = discover_read_write_pairs(args.manifest)
    result = seed_read_write_pair_relations(service, pairs, dry_run=args.dry_run)
    after = audit_graph(service.store) if not args.dry_run else before
    report = {
        "manifest": str(args.manifest),
        "dry_run": args.dry_run,
        "backup": str(backup),
        "before": before,
        "after": after,
        "pair_count": len(pairs),
        "excluded_pair_count": len(excluded),
        "excluded_pairs": excluded,
        "result": result,
    }
    report_path = output_dir / "read_write_pair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "backup": str(backup),
        "report": str(report_path),
        "dry_run": args.dry_run,
        "before_relations": before.get("relations", 0),
        "after_relations": after.get("relations", 0),
        "pair_count": len(pairs),
        "excluded_pair_count": len(excluded),
        "stored": result.get("counts", {}).get("stored", 0),
        "already_present": result.get("counts", {}).get("already_present", 0),
        "unconfirmed": result.get("counts", {}).get("unconfirmed", 0),
    }, ensure_ascii=False, indent=2))
    service.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
