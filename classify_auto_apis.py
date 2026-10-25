#!/usr/bin/env python3
"""Classify FOCAS functions into capture-first buckets.

Goal: separate read-only functions that need no CNC parameter or mode change
(usable in AUTO) from functions that require parameter writes, mode switching,
or are themselves write/control APIs.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "rag_indexes" / "focas" / "function_manifest.jsonl"
RULES = ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
COVERAGE_STATE = ROOT / "rag_indexes" / "focas" / "function_coverage_state.json"
OUT = ROOT / "inventcom_output" / "api_capture_priority.json"

STATE_WRITE_PATTERNS = re.compile(
    r"^(?:"
    r"cnc_wr|cnc_del|cnc_clear|cnc_clr|cnc_set|cnc_start|cnc_stop|cnc_reset"
    r"|cnc_absolute|cnc_relative|cnc_machine|cnc_distance|cnc_skip|cnc_rewind"
    r"|cnc_download|cnc_upload|cnc_copyprog|cnc_mergeprog|cnc_renameprog"
    r"|cnc_condense|cnc_search|cnc_sendmessage|cnc_register|cnc_reg"
    r"|cnc_inst|cnc_prst|cnc_rstr|cnc_seqsrch|cnc_dnc|cnc_dtsv|cnc_svdt"
    r"|cnc_from|cnc_sram|cnc_cdnc|cnc_cdownload|cnc_cupload|cnc_cverify"
    r"|pmc_wr|pmc_set|pmc_reset|pmc_start|pmc_stop"
    r")"
)

READ_PATTERNS = re.compile(
    r"^(?:cnc_rd|cnc_get|cnc_alarm|cnc_statinfo|cnc_sysinfo|cnc_modal"
    r"|cnc_axisname|cnc_exaxisname|cnc_getlanguage|cnc_getmactype|cnc_getpmactype"
    r"|cnc_gettimer|cnc_getlockstat|cnc_getfrominfo|cnc_getsraminfo|cnc_getfigure"
    r"|cnc_rdprogdir|cnc_rdprgnum|cnc_rdexecprog|cnc_rdexecpt|cnc_rdblkcount"
    r"|pmc_rd|pmc_get|pmc_kpmsiz|pmc_r)"
)

STATE_VERBS = {
    "write", "writes", "set", "setting", "delete", "start", "end", "download",
    "clear", "change", "upload", "stop", "copy", "transfer", "format", "cancel",
    "reset", "move", "restart", "allocate", "make", "input", "output", "preset",
    "rewind", "merge", "condense", "initialization", "new", "select", "search",
    "register", "insert", "remove", "create", "rename", "add", "edit", "modify",
    "update", "enable", "disable", "lock", "unlock", "release", "execute", "run",
    "launch", "abort", "terminate", "kill", "teach", "override",
    "rearrange", "starts", "execution", "list-delete", "list-get", "list-put",
    "mget", "mput", "put",
}
READ_VERBS = {
    "read", "reading", "reads", "get", "check", "verify", "list", "query",
    "retrieve", "display", "monitor", "find", "free",
}

FILE_TRANSFER = re.compile(
    r"\b(?:get|put|mget|mput|list-get|list-put|list-delete)\b.*\bfile\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def any_mode(rule: dict) -> bool:
    for cond in rule.get("conditions", []):
        if cond.get("type") == "cnc_mode":
            text = str(cond.get("requirement", "")).lower()
            if "any cnc mode" in text or "any mode" in text:
                return True
    return False


def first_verb(description: str) -> str:
    parts = (description or "").strip().split()
    return parts[0].lower().strip("[]:.,;") if parts else ""


def classify(name: str, rule: dict | None, description: str = "") -> tuple[str, list[str]]:
    reasons: list[str] = []
    conds = (rule or {}).get("conditions", [])
    types = {c.get("type") for c in conds if isinstance(c, dict)}
    has_param = "cnc_parameter" in types
    has_mode = "cnc_mode" in types
    has_option = "software_option" in types
    mode_ok = any_mode(rule or {})
    write_name = bool(STATE_WRITE_PATTERNS.match(name))
    read_name = bool(READ_PATTERNS.match(name))
    verb = first_verb(description)
    write_desc = verb in STATE_VERBS or bool(FILE_TRANSFER.search(description or ""))
    read_desc = verb in READ_VERBS
    if has_param:
        reasons.append("cnc_parameter_condition")
    if has_mode and not mode_ok:
        reasons.append("mode_specific_condition")
    if has_option:
        reasons.append("software_option_condition")
    if write_name or write_desc:
        reasons.append("write_or_control_name")
        if verb:
            reasons.append(f"verb:{verb}")
    if rule is None:
        reasons.append("no_prerequisite_rule")
    if read_name or read_desc:
        reasons.append("read_name")
    if has_param:
        return "needs_parameter_change", reasons
    if write_name or write_desc:
        return "state_writing_or_control", reasons
    if has_mode and not mode_ok:
        return "needs_mode_switch", reasons
    if read_name or read_desc:
        return "ready_auto_no_parameter", reasons
    if rule is None:
        return "unclassified_no_rule", reasons
    return "ready_auto_no_parameter", reasons


def main() -> None:
    manifest = load_jsonl(MANIFEST)
    unique: dict[str, dict] = {}
    for row in manifest:
        name = str(row.get("function", "")).strip()
        if name:
            unique.setdefault(name, row)
    rules = {
        name: rule
        for rule in load_jsonl(RULES)
        if rule.get("rule_type") == "api_prerequisite"
        for name in (rule.get("applies_to") or [])
    }
    covered: set[str] = set()
    if COVERAGE_STATE.exists():
        state = json.loads(COVERAGE_STATE.read_text(encoding="utf-8"))
        covered = {
            name for name, item in state.get("functions", {}).items()
            if str(item.get("status", "")).startswith("covered")
        }

    buckets: dict[str, list[dict]] = {}
    for name, row in sorted(unique.items()):
        bucket, reasons = classify(name, rules.get(name), str(row.get("description", "")))
        buckets.setdefault(bucket, []).append({
            "function": name,
            "category": row.get("category", ""),
            "description": row.get("description", ""),
            "covered": name in covered,
            "reasons": reasons,
        })

    counts = {bucket: len(rows) for bucket, rows in buckets.items()}
    ready = buckets.get("ready_auto_no_parameter", [])
    ready_uncovered = [item["function"] for item in ready if not item["covered"]]
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    settings_path = ROOT / "runs" / "desktop" / "20261023_152654_fd3f43" / "settings.json"
    if settings_path.is_file():
        from smart_traffic_agent.api_compatibility import unavailable_reason
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        eligibility = dict(settings)
        for name in ready_uncovered:
            reason = unavailable_reason(name, eligibility)
            if reason:
                excluded[name] = reason
            else:
                eligible.append(name)
    payload = {
        "manifest_unique_functions": len(unique),
        "prerequisite_rules": len(rules),
        "already_covered_in_global_state": len(covered),
        "bucket_counts": counts,
        "ready_auto_no_parameter_total": len(ready),
        "ready_auto_no_parameter_uncovered": len(ready_uncovered),
        "ready_auto_no_parameter_functions": ready_uncovered,
        "ready_auto_no_parameter_eligible": len(eligible),
        "ready_auto_no_parameter_eligible_functions": eligible,
        "ready_auto_no_parameter_excluded_by_compatibility": excluded,
        "buckets": buckets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "manifest_unique": len(unique),
        "rules": len(rules),
        "covered_global": len(covered),
        "bucket_counts": counts,
        "ready_uncovered": len(ready_uncovered),
        "ready_eligible": len(eligible),
        "excluded_by_compatibility": len(excluded),
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))
    print("ready sample:", ready_uncovered[:40])


if __name__ == "__main__":
    main()
