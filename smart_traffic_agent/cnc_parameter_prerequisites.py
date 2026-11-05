"""Evidence-backed CNC parameter prerequisite extraction for TRCM.

The graph deliberately contains API nodes only.  CNC parameter identity,
bit/value constraints, controller scope and write safety are relation
attributes.  Concrete values are never selected here; a generated harness
must read the parameter first and make the runtime decision.
"""
from __future__ import annotations

import json
import csv
import re
from pathlib import Path
from typing import Any, Iterable

from .api_compatibility import F0ID_SCOPE_ID, f0id_api_names
from .integrations.ncguide import default_focas_header_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE_CHUNKS = ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
DEFAULT_RETURN_CODES = ROOT / "rag_indexes" / "focas" / "api_return_codes.csv"

_PARAM_REF_RE = re.compile(r"(?<!\d)(\d{1,5})\s*#\s*([0-7])(?!\d)")


def load_api_prerequisite_rules(path: Path = DEFAULT_RULE_CHUNKS) -> list[dict[str, Any]]:
    """Load only structured API prerequisite chunks from the local RAG index."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("rule_type") == "api_prerequisite":
                rows.append(item)
    rows.extend(_load_return_code_parameter_rules(DEFAULT_RETURN_CODES))
    return rows


_RETURN_PARAMETER_RE = re.compile(
    r"(?<!\d)(\d{1,5})\s*#\s*([0-7])\s*(?:=|must\s+be|should\s+be|to)\s*([01])\b",
    re.IGNORECASE,
)


def _return_code_context(text: str, start: int, end: int) -> str:
    """Extract a nearby controller qualifier without inventing one."""
    window = text[max(0, start - 320):min(len(text), end + 240)]
    matches = re.findall(r"Series[^\n.;]{0,180}", window, re.IGNORECASE)
    return str(matches[-1]).strip() if matches else ""


def _load_return_code_parameter_rules(path: Path) -> list[dict[str, Any]]:
    """Convert explicit web return-code parameter requirements into rules.

    This supplements structured prerequisite chunks.  Exception examples such
    as ``except No.3201#2=1`` are intentionally ignored because they are not
    required states.
    """
    if not path.exists():
        return []
    grouped: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    try:
        rows = csv.DictReader(path.open("r", encoding="utf-8-sig", newline=""))
        for row in rows:
            function = str(row.get("function", "")).strip()
            explanation = str(row.get("api_explanation", ""))
            code_name = str(row.get("code_name", "")).strip()
            if not function or not explanation or code_name not in {"EW_PARAM", "EW_PROT", "EW_NOOPT"}:
                continue
            for match in _RETURN_PARAMETER_RE.finditer(explanation):
                context_window = explanation[max(0, match.start() - 40):match.end() + 40].casefold()
                if "except" in context_window or "only" in context_window and "set" not in context_window:
                    continue
                number, bit, value = map(int, match.groups())
                context = _return_code_context(explanation, match.start(), match.end())
                key = (function, code_name, number, bit, value)
                item = grouped.setdefault(key, {
                    "rule_type": "api_prerequisite",
                    "rule_id": f"FOCAS-return-code:{function}:{code_name}:{number}#{bit}={value}",
                    "scenario": "api_return_code_parameter",
                    "rule_text": explanation,
                    "applies_to": [function],
                    "conditions": [{
                        "id": f"return_code_{code_name}_{number}_{bit}_{value}",
                        "type": "cnc_parameter",
                        "parameter_number": number,
                        "bit": bit,
                        "expected_value": value,
                        "requirement": explanation,
                        "check_api": "cnc_rdparam",
                        "series_context": context,
                        "unknown_state": "do_not_write",
                    }],
                    "preparation_operations": [{"operation": "read_parameter", "api": "cnc_rdparam", "parameter_number": number, "bit": bit}],
                    "verification_operations": [{"api": "cnc_rdparam", "check": f"{number}#{bit} equals {value}"}],
                    "source_url": row.get("source_url", ""),
                    "source_file": "rag_indexes/focas/api_return_codes.csv",
                    "source_chunk_id": row.get("source_chunk_id", ""),
                })
                if context and not item["conditions"][0].get("series_context"):
                    item["conditions"][0]["series_context"] = context
    except (OSError, UnicodeError):
        return []
    return list(grouped.values())


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _context_text(rule: dict[str, Any], condition: dict[str, Any]) -> str:
    explicit = str(condition.get("series_context", "")).strip()
    if explicit:
        # A condition-level series qualifier is narrower than the API page's
        # availability table and must not be broadened by that table.
        return explicit + " " + str(condition.get("requirement", ""))
    return " ".join(
        str(rule.get(key, ""))
        for key in ("scenario", "rule_text", "source_file")
    )


def _f0id_scope_decision(rule: dict[str, Any], condition: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a condition is applicable to FANUC 0i-D.

    Explicit exclusions win.  A condition mentioning 0i-D/F or the generic
    0i family applies; a condition limited to A/B/C, F, 15, etc. is retained
    only as a candidate diagnostic.  Rules without a model qualifier inherit
    their exact API page scope and are therefore applicable to the selected
    F0i-D target.
    """
    text = _context_text(rule, condition)
    lowered = text.casefold()
    if "unresolved_series" in lowered:
        return False, "controller series is unresolved"
    if re.search(r"except[^\n]{0,160}0i[- ]?d", lowered):
        return False, "source explicitly excludes 0i-D"
    if re.search(r"0i[- ]?d(?:/f)?", lowered):
        return True, "source explicitly covers 0i-D"
    family_specific = re.search(
        r"0i[- ]?(?:a(?:/b)?(?:/c)?|b/c|f)\b", lowered
    )
    if re.search(r"\b0i\b", lowered) and not family_specific:
        return True, "source covers the generic 0i family"
    if not str(condition.get("series_context", "")).strip():
        return True, "condition inherits the exact API rule scope"
    return False, "condition is limited to another controller series"


def _header_declared(api_name: str, header_path: Path) -> bool:
    try:
        text = header_path.read_text(encoding="latin-1")
    except OSError:
        return False
    return bool(re.search(rf"\b{re.escape(api_name)}\s*\(", text))


def _condition_id(rule: dict[str, Any], condition: dict[str, Any], index: int) -> str:
    return str(condition.get("id") or f"condition_{index}").strip()


def _parameter_ref(number: int, bit: int) -> str:
    return f"{number}#{bit}"


def extract_f0id_parameter_prerequisites(
    api_names: Iterable[str],
    *,
    allowed_api_names: Iterable[str] | None = None,
    device_scope: str = F0ID_SCOPE_ID,
    rule_chunks_path: Path = DEFAULT_RULE_CHUNKS,
    header_path: Path | None = None,
) -> dict[str, Any]:
    """Extract target prerequisites and read/write capability diagnostics.

    ``relations`` contain only conditions with a documented F0i-D decision and
    SDK declarations for every API endpoint.  ``candidates`` preserve rules
    that were seen but cannot safely be applied to this controller scope.
    """
    requested = {str(name).strip() for name in api_names if str(name).strip()}
    scope = {
        str(name).strip()
        for name in (allowed_api_names if allowed_api_names is not None else f0id_api_names())
        if str(name).strip()
    }
    if device_scope != F0ID_SCOPE_ID and allowed_api_names is None:
        scope = set(requested)
    header = header_path or (default_focas_header_dir() / "Fwlib32.h")
    relations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    analyzed: set[str] = set()
    seen: set[tuple[str, str, int, int, int]] = set()

    for rule in load_api_prerequisite_rules(rule_chunks_path):
        applies = {str(name).strip() for name in _as_list(rule.get("applies_to")) if str(name).strip()}
        targets = sorted(applies.intersection(requested).intersection(scope))
        if not targets:
            continue
        analyzed.update(targets)
        conditions = _as_list(rule.get("conditions"))
        preparations = _as_list(rule.get("preparation_operations"))
        verification = _as_list(rule.get("verification_operations"))
        for index, raw_condition in enumerate(conditions, 1):
            if not isinstance(raw_condition, dict) or raw_condition.get("type") != "cnc_parameter":
                continue
            try:
                number = int(raw_condition.get("parameter_number"))
                bit = int(raw_condition.get("bit"))
                expected = int(raw_condition.get("expected_value"))
            except (TypeError, ValueError):
                for target in targets:
                    candidates.append({"target_api": target, "rule_id": rule.get("rule_id", ""), "condition": raw_condition, "reason": "parameter number, bit or expected value is not numeric"})
                continue
            if number < 0 or bit not in range(8) or expected not in (0, 1):
                for target in targets:
                    candidates.append({"target_api": target, "rule_id": rule.get("rule_id", ""), "condition": raw_condition, "reason": "bit prerequisites require an integer parameter, bit 0..7 and value 0/1"})
                continue
            read_api = str(raw_condition.get("check_api") or "cnc_rdparam").strip()
            write_api = str(raw_condition.get("write_api") or "").strip()
            if not write_api:
                for operation in preparations:
                    if isinstance(operation, dict) and operation.get("operation") == "write_parameter_bit":
                        write_api = str(operation.get("api") or "").strip()
                        break
            readback_api = read_api
            for operation in verification:
                if isinstance(operation, dict):
                    readback_api = str(operation.get("api") or readback_api).strip()
                    break
            if not write_api and _header_declared("cnc_wrparam", header):
                # The generic SDK writer is a capability fact, not permission
                # to execute it.  It remains dangerous and manual-only.
                write_api = "cnc_wrparam"
            scope_ok, scope_reason = _f0id_scope_decision(rule, raw_condition)
            endpoint_names = [read_api, readback_api] + ([write_api] if write_api else []) + targets
            missing = [name for name in endpoint_names if name not in scope]
            missing.extend(name for name in endpoint_names if not _header_declared(name, header))
            if not scope_ok:
                missing.append(scope_reason)
            evidence_ids = [
                str(rule.get("rule_id") or ""),
                str(rule.get("source_chunk_id") or ""),
            ]
            evidence_ids.extend(
                str(item) for item in (rule.get("source_url"), rule.get("source_file")) if str(item).strip()
            )
            evidence_ids.extend(
                f"{header}:declaration:{name}"
                for name in endpoint_names
                if name and _header_declared(name, header)
            )
            evidence_ids = list(dict.fromkeys(item for item in evidence_ids if item))
            for target in targets:
                dedupe_key = (read_api, target, number, bit, expected)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                attrs = {
                    "parameter_number": number,
                    "bit": bit,
                    "expected_value": expected,
                    "parameter_reference": _parameter_ref(number, bit),
                    "value_domain": [0, 1],
                    "value_encoding": "single_bit",
                    "whole_parameter_requires_read_modify_write": True,
                    "read_api": read_api,
                    "readback_api": readback_api,
                    "write_api": write_api,
                    "write_supported_by_sdk": bool(write_api and _header_declared(write_api, header)),
                    "write_policy": raw_condition.get("write_policy", "knowledge_only_manual_approval"),
                    "requires_read_snapshot": True,
                    "requires_readback": True,
                    "requires_restore_or_explicit_snapshot": True,
                    "restart_may_be_required": bool(raw_condition.get("restart_may_be_required", rule.get("restart_required", False))),
                    "dangerous_operation": bool(write_api),
                    "auto_execute": False,
                    "controller_scope": device_scope,
                    "scope_reason": scope_reason,
                    "source_rule_id": rule.get("rule_id", ""),
                    "condition_id": _condition_id(rule, raw_condition, index),
                    "requirement": raw_condition.get("requirement", ""),
                }
                payload = {
                    "source_api": read_api,
                    "target_api": target,
                    "relation_type": "REQUIRES",
                    "rule_id": f"parameter_prerequisite:{rule.get('rule_id', '')}:{_condition_id(rule, raw_condition, index)}:{number}#{bit}",
                    "confidence_status": "sdk_validated" if not missing else "proposed",
                    "evidence_ids": evidence_ids,
                    "applicability": {
                        "device_scope": device_scope,
                        "controller_model": "0i-D",
                        "series_context": raw_condition.get("series_context", "") or "rule_scope",
                        "required": True,
                    },
                    "attributes": attrs,
                }
                if missing:
                    candidates.append({"target_api": target, "rule_id": payload["rule_id"], "condition": raw_condition, "reason": missing})
                    continue
                relations.append(payload)
                # Expose the runtime state transfer into a possible writer.
                if write_api:
                    relations.append({
                        "source_api": read_api,
                        "target_api": write_api,
                        "relation_type": "PROVIDES_INPUT",
                        "rule_id": payload["rule_id"] + f":write:{target}",
                        "confidence_status": "sdk_validated",
                        "evidence_ids": evidence_ids,
                        "applicability": {**payload["applicability"], "required": False},
                        "attributes": {**attrs, "parameter_role": "read_original_then_write_one_bit", "target_api": target},
                    })
                    relations.append({
                        "source_api": write_api,
                        "target_api": readback_api,
                        "relation_type": "READS_AFTER_WRITE",
                        "rule_id": payload["rule_id"] + f":readback:{target}",
                        "confidence_status": "sdk_validated",
                        "evidence_ids": evidence_ids,
                        "applicability": {**payload["applicability"], "required": False},
                        "attributes": {**attrs, "parameter_role": "write_then_readback", "target_api": target},
                    })
    return {
        "relations": relations,
        "candidates": candidates,
        "scanned_api_count": len(requested.intersection(scope)),
        "analyzed_api_count": len(analyzed),
        "condition_count": len(relations),
        "candidate_count": len(candidates),
        "scope_api_count": len(scope),
        "controller_model": "0i-D",
        "device_scope": device_scope,
    }


def parameter_prerequisites_for_target(api_name: str, **kwargs: Any) -> dict[str, Any]:
    """Exact lookup used by Planner and offline tests."""
    return extract_f0id_parameter_prerequisites([api_name], **kwargs)
