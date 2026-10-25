#!/usr/bin/env python3
"""Classify FOCAS functions by CNC mode requirement and CNC parameter
requirement, using the per-function api_prerequisite rules.

The read/write nature of an API is deliberately NOT a bucket here: AUTO/MEM
mode can invoke read, write, and control APIs. The two axes are:
  * CNC mode needed (any, MEM/AUTO, MDI, EDIT, DNC, TAPE, RMT, ...)
  * CNC parameter bits that must be enabled (none / required)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "rag_indexes" / "focas" / "function_manifest.jsonl"
RULES = ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
SETTINGS = ROOT / "runs" / "desktop" / "20261023_152654_fd3f43" / "settings.json"
OUT = ROOT / "inventcom_output" / "api_mode_parameter_classification.json"

MODE_TOKENS = re.compile(r"\b(EDIT|MEM|MDI|DNC|TAPE|RMT|TJOG|THND|JOG)\b")

# Functions whose meaningful traffic is produced only while an NC program is
# executing (motion/feed/spindle/modal/execution-pointer/history families).
# Program-file CRUD APIs are not included: they need a resident program, not a
# running one.
PROGRAM_RUNNING_FUNCTIONS = {
    "cnc_absolute", "cnc_absolute2", "cnc_relative", "cnc_relative2",
    "cnc_machine", "cnc_distance", "cnc_rdposition", "cnc_rddynamic",
    "cnc_rddynamic2", "cnc_actf", "cnc_acts", "cnc_acts2", "cnc_rdspeed",
    "cnc_skip", "cnc_rdaxisdata", "cnc_rdcommand", "cnc_rdcurrent",
    "cnc_rdgcode", "cnc_modal", "cnc_rdexecmcode", "cnc_rdexecprog",
    "cnc_rdexecpt", "cnc_rdactpt", "cnc_rdprgnum", "cnc_exeprgname",
    "cnc_exeprgname2", "cnc_rdseqnum", "cnc_rdblkcount", "cnc_rdspmeter",
    "cnc_rdspload", "cnc_rdspmaxrpm", "cnc_rdspgear", "cnc_rdspcss",
    "cnc_rdspdlname", "cnc_rdophistry", "cnc_rdophistry2", "cnc_rdophistry4",
    "cnc_rdhissgnl", "cnc_rdhissgnl2", "cnc_rdhissgnl3", "cnc_rdopmsg",
    "cnc_rdopmsg2", "cnc_rdopmsg3", "cnc_rdprstrinfo", "cnc_rd3dtooltip",
    "cnc_rd3dpulse", "cnc_rd3dmovrlap", "cnc_rd3dofschg", "cnc_rd5axmandt",
    "cnc_rd5axovrlap", "cnc_rdhndintrpt", "cnc_rdmovrlap", "cnc_rdvolccomp",
    "cnc_rdproctime", "cnc_rdprgdirtime",
}

WRITE_VERBS = {
    "write", "writes", "set", "setting", "delete", "start", "end", "download",
    "clear", "change", "upload", "stop", "copy", "transfer", "format", "cancel",
    "reset", "move", "restart", "allocate", "make", "input", "output", "preset",
    "rewind", "merge", "condense", "initialization", "new", "select", "search",
    "register", "insert", "remove", "create", "rename", "add", "edit", "modify",
    "update", "enable", "disable", "lock", "unlock", "release", "execute", "run",
    "launch", "abort", "terminate", "kill", "teach", "override", "rearrange",
    "starts", "execution", "list-delete", "list-get", "list-put", "mget", "mput", "put",
}
READ_VERBS = {
    "read", "reading", "reads", "get", "check", "verify", "list", "query",
    "retrieve", "display", "monitor", "find", "free",
}
FILE_TRANSFER = re.compile(
    r"\b(?:get|put|mget|mput|list-get|list-put|list-delete)\b.*\bfile\b", re.I,
)

MODE_LABELS = {
    "any_mode": "任意 CNC 模式（AUTO/MEM 可直接调用）",
    "mem_only": "需要 MEM（AUTO）模式",
    "edit_mem": "需要 EDIT 或 MEM（AUTO）",
    "not_mdi": "不能用 MDI（MEM/AUTO 可用）",
    "mdi_only": "需要 MDI 模式",
    "edit_only": "需要 EDIT 模式",
    "mdi_edit": "需要 MDI 或 EDIT",
    "not_mem_mdi": "不能用 MEM/MDI（需 EDIT/JOG 等）",
    "dnc": "需要 DNC 模式",
    "tape": "需要 TAPE 模式",
    "rmt": "需要 RMT 模式",
    "mixed": "多种指定模式",
    "other": "其他模式描述",
    "unknown": "未记录模式限制",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mode_bucket(texts: list[str]) -> str:
    if not texts:
        return "unknown"
    blob = " ".join(str(t) for t in texts).upper().replace("\n", " ")
    if re.search(r"ANY CNC MODE|ANY MODE", blob):
        return "any_mode"
    not_mem_mdi = bool(re.search(r"OTHER THAN\s+MEM\s*,\s*MDI", blob))
    not_mdi = bool(re.search(r"OTHER THAN\s+MDI", blob))
    if not_mem_mdi:
        return "not_mem_mdi"
    if not_mdi:
        return "not_mdi"
    tokens = set(MODE_TOKENS.findall(blob))
    has_mem, has_mdi, has_edit = "MEM" in tokens, "MDI" in tokens, "EDIT" in tokens
    if has_mdi and not has_mem and not has_edit:
        return "mdi_only"
    if has_edit and not has_mem and not has_mdi:
        return "edit_only"
    if has_mem and has_edit and not has_mdi:
        return "edit_mem"
    if has_mdi and has_edit and not has_mem:
        return "mdi_edit"
    if has_mem and not has_edit and not has_mdi:
        return "mem_only"
    if has_mdi:
        return "mixed"
    if "DNC" in tokens:
        return "dnc"
    if "TAPE" in tokens:
        return "tape"
    if "RMT" in tokens:
        return "rmt"
    return "other"


def auto_capable(bucket: str) -> bool | None:
    if bucket in {"any_mode", "mem_only", "edit_mem", "not_mdi"}:
        return True
    if bucket in {"mdi_only", "edit_only", "mdi_edit", "not_mem_mdi",
                  "dnc", "tape", "rmt", "mixed"}:
        return False
    return None


def action_bucket(description: str) -> str:
    parts = (description or "").strip().split()
    verb = parts[0].lower().strip("[]:.,;") if parts else ""
    if verb in WRITE_VERBS or FILE_TRANSFER.search(description or ""):
        return "write_or_control"
    if verb in READ_VERBS:
        return "read"
    return "unknown"


def parameter_bucket(param_conditions: list[dict], has_rule: bool) -> str:
    if not has_rule:
        return "unknown_no_rule"
    return "required" if param_conditions else "none"


def main() -> None:
    manifest = load_jsonl(MANIFEST)
    unique: dict[str, dict] = {}
    for row in manifest:
        name = str(row.get("function", "")).strip()
        if name:
            unique.setdefault(name, row)

    rules_by_function: dict[str, list[dict]] = defaultdict(list)
    for rule in load_jsonl(RULES):
        if rule.get("rule_type") != "api_prerequisite":
            continue
        for name in rule.get("applies_to", []) or []:
            rules_by_function[name].append(rule)

    compatibility = {}
    header_present: dict[str, bool] = {}
    if SETTINGS.is_file():
        from smart_traffic_agent.api_compatibility import unavailable_reason
        settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
        compatibility = {
            name: unavailable_reason(name, settings) == ""
            for name in unique
        }
        header_path = Path(settings["header"]) / "Fwlib32.h"
        if header_path.is_file():
            header_text = header_path.read_text(encoding="latin-1")
            header_present = {
                name: bool(re.search(rf"\b{re.escape(name)}\s*\(", header_text, re.I))
                for name in unique
            }

    by_mode: dict[str, list[str]] = defaultdict(list)
    by_parameter: dict[str, list[str]] = defaultdict(list)
    matrix: dict[str, list[str]] = defaultdict(list)
    cells: dict[str, dict] = {}

    for name, row in sorted(unique.items()):
        rules = rules_by_function.get(name, [])
        conditions = [c for rule in rules for c in rule.get("conditions", []) if isinstance(c, dict)]
        mode_texts = [str(c.get("requirement", "")) for c in conditions if c.get("type") == "cnc_mode"]
        params = [c for c in conditions if c.get("type") == "cnc_parameter"]
        seen = set()
        param_list = []
        for c in params:
            key = (c.get("parameter_number"), c.get("bit"), c.get("expected_value"))
            if key in seen:
                continue
            seen.add(key)
            param_list.append({
                "parameter_number": c.get("parameter_number"),
                "bit": c.get("bit"),
                "expected_value": c.get("expected_value"),
                "series_context": c.get("series_context", ""),
            })

        mode = mode_bucket(mode_texts)
        parameter = parameter_bucket(params, bool(rules))
        cell = f"{mode}__{parameter}"
        by_mode[mode].append(name)
        by_parameter[parameter].append(name)
        matrix[cell].append(name)
        cells[name] = {
            "function": name,
            "category": row.get("category", ""),
            "description": row.get("description", ""),
            "mode_bucket": mode,
            "mode_label": MODE_LABELS.get(mode, mode),
            "mode_text": " | ".join(mode_texts)[:1600],
            "auto_capable": auto_capable(mode),
            "parameter_bucket": parameter,
            "parameters": param_list,
            "has_rule": bool(rules),
            "rule_ids": sorted({r.get("rule_id", "") for r in rules}),
            "compatible_0id_ethernet": compatibility.get(name),
            "header_present": header_present.get(name),
            "action": action_bucket(str(row.get("description", ""))),
            "needs_nc_program": name in PROGRAM_RUNNING_FUNCTIONS,
        }

    def names(predicate):
        return sorted(name for name, item in cells.items() if predicate(item))

    def eligible(item):
        return bool(item.get("compatible_0id_ethernet")) and bool(item.get("header_present"))

    auto_none = names(lambda i: i["auto_capable"] is True and i["parameter_bucket"] == "none")
    auto_required = names(lambda i: i["auto_capable"] is True and i["parameter_bucket"] == "required")
    auto_static = [n for n in auto_none if not cells[n]["needs_nc_program"]]
    needs_program = names(lambda i: i["needs_nc_program"])
    mode_switch = names(lambda i: i["auto_capable"] is False)
    undocumented = names(lambda i: i["auto_capable"] is None)
    action_counts = Counter(cells[n]["action"] for n in auto_none)

    payload = {
        "manifest_unique_functions": len(unique),
        "mode_bucket_counts": {k: len(v) for k, v in sorted(by_mode.items())},
        "parameter_bucket_counts": {k: len(v) for k, v in sorted(by_parameter.items())},
        "matrix_counts": {k: len(v) for k, v in sorted(matrix.items())},
        "by_mode": {k: sorted(v) for k, v in sorted(by_mode.items())},
        "by_parameter": {k: sorted(v) for k, v in sorted(by_parameter.items())},
        "matrix": {k: sorted(v) for k, v in sorted(matrix.items())},
        "functions": cells,
        "mode_labels": MODE_LABELS,
        "summary": {
            "auto_no_parameter_total": len(auto_none),
            "auto_no_parameter_eligible": len([n for n in auto_none if eligible(cells[n])]),
            "auto_no_parameter_static_total": len(auto_static),
            "auto_no_parameter_static_eligible": len([n for n in auto_static if eligible(cells[n])]),
            "needs_nc_program_total": len(needs_program),
            "needs_nc_program_in_auto_no_parameter": len([n for n in auto_none if cells[n]["needs_nc_program"]]),
            "needs_nc_program_in_auto_parameter_required": len([n for n in auto_required if cells[n]["needs_nc_program"]]),
            "auto_no_parameter_by_action": dict(action_counts),
            "auto_parameter_required_total": len(auto_required),
            "auto_parameter_required_eligible": len([n for n in auto_required if eligible(cells[n])]),
            "mode_switch_required_total": len(mode_switch),
            "mode_switch_required_eligible": len([n for n in mode_switch if eligible(cells[n])]),
            "undocumented_mode_total": len(undocumented),
            "undocumented_mode_eligible": len([n for n in undocumented if eligible(cells[n])]),
        },
        "auto_no_parameter": auto_none,
        "auto_no_parameter_eligible_functions": [n for n in auto_none if eligible(cells[n])],
        "auto_no_parameter_static_eligible_functions": [n for n in auto_static if eligible(cells[n])],
        "needs_nc_program": needs_program,
        "auto_parameter_required": auto_required,
        "auto_parameter_required_eligible_functions": [n for n in auto_required if eligible(cells[n])],
        "mode_switch_required": mode_switch,
        "mode_switch_required_eligible_functions": [n for n in mode_switch if eligible(cells[n])],
        "undocumented_mode": undocumented,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "functions": len(unique),
        "mode_bucket_counts": payload["mode_bucket_counts"],
        "parameter_bucket_counts": payload["parameter_bucket_counts"],
        "matrix_counts": payload["matrix_counts"],
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
