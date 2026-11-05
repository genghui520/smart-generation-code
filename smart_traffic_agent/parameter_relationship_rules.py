"""Evidence-backed parameter discovery rules for the F0i-D TRCM.

The graph stores API nodes only.  This module describes how an observed
field from one API can constrain a later API argument.  It intentionally
stores constraints rather than concrete device values: values are selected
by the generated harness after the provider API returns.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .api_compatibility import F0ID_SCOPE_ID, f0id_api_names
from .focas_contracts import documented_parameter_names
from .integrations.ncguide import default_focas_header_dir


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "rag_indexes" / "focas" / "chunks.jsonl"
DEFAULT_HEADER = default_focas_header_dir() / "Fwlib32.h"


@dataclass(frozen=True, slots=True)
class ParameterDiscoveryRule:
    rule_id: str
    source_api: str
    target_api: str
    relation_type: str
    source_fields: tuple[str, ...]
    target_parameters: tuple[str, ...]
    transform: str
    filter_strategy: str
    applicability: dict[str, Any]
    evidence_ids: tuple[str, ...] = ()
    dangerous_operation: bool = False
    auto_execute: bool = True
    verification_requirements: tuple[str, ...] = ()
    parameter_kind: str = "formal_argument"

    def as_relation_attributes(self) -> dict[str, Any]:
        parameter_rule = {
            "rule_id": self.rule_id,
            "source_api": self.source_api,
            "source_fields": list(self.source_fields),
            "target_parameters": list(self.target_parameters),
            "parameter": self.target_parameters[0] if self.target_parameters else "",
            "transform": self.transform,
            "filter_strategy": self.filter_strategy,
            "constraints": list(self.verification_requirements),
            "require_runtime_discovery": True,
            "applicability": dict(self.applicability),
            "dangerous_operation": self.dangerous_operation,
            "auto_execute": self.auto_execute,
        }
        return {
            "discovery_family": self.rule_id.split(":", 1)[0],
            "source_fields": list(self.source_fields),
            "target_parameters": list(self.target_parameters),
            "transform": self.transform,
            "filter_strategy": self.filter_strategy,
            "dangerous_operation": self.dangerous_operation,
            "auto_execute": self.auto_execute,
            "verification_requirements": list(self.verification_requirements),
            "parameter_rules": [parameter_rule],
        }


def _rule(
    rule_id: str,
    source_api: str,
    target_api: str,
    relation_type: str,
    source_fields: Iterable[str],
    target_parameters: Iterable[str],
    transform: str,
    filter_strategy: str,
    *,
    verification_requirements: Iterable[str] = (),
    dangerous_operation: bool = False,
    auto_execute: bool = True,
    applicability: dict[str, Any] | None = None,
    parameter_kind: str = "formal_argument",
) -> ParameterDiscoveryRule:
    return ParameterDiscoveryRule(
        rule_id=rule_id,
        source_api=source_api,
        target_api=target_api,
        relation_type=relation_type,
        source_fields=tuple(source_fields),
        target_parameters=tuple(target_parameters),
        transform=transform,
        filter_strategy=filter_strategy,
        applicability=dict(applicability or {}),
        verification_requirements=tuple(verification_requirements),
        dangerous_operation=dangerous_operation,
        auto_execute=auto_execute,
        parameter_kind=parameter_kind,
    )


_AXIS_COUNT_TARGETS = (
    "cnc_absolute", "cnc_absolute2", "cnc_machine", "cnc_relative",
    "cnc_relative2", "cnc_distance", "cnc_rddynamic", "cnc_rddynamic2",
    "cnc_skip",
)
_AXIS_NAME_TARGETS = _AXIS_COUNT_TARGETS
_TOOL_OFFSET_TARGETS = ("cnc_rdtofs", "cnc_rdtofsr", "cnc_wrtofs", "cnc_wrtofsr")
_MACRO_TARGETS = ("cnc_rdmacro", "cnc_wrmacro", "cnc_rdmacror")
_PARAMETER_TARGETS = (
    "cnc_rdparam", "cnc_rdparam3", "cnc_rdparam_ext", "cnc_rdparar",
    "cnc_wrparam", "cnc_wrparas",
)
_ZERO_OFFSET_TARGETS = ("cnc_rdzofs", "cnc_rdzofsr", "cnc_wrzofs", "cnc_wrzofsr")
_PMC_TARGETS = ("pmc_rdpmcrng",)
_PATH_SENSITIVE_TARGETS = (
    "cnc_rdsyssoft", "cnc_rdsyssoft2", "cnc_rdprogdir3", "cnc_rdparam",
    "cnc_rdparam3", "cnc_rdset", "cnc_rdsetr", "cnc_rdposition",
)


def _build_rules() -> tuple[ParameterDiscoveryRule, ...]:
    rules: list[ParameterDiscoveryRule] = [
        _rule(
            "axis_count:sysinfo->axisname", "cnc_sysinfo", "cnc_rdaxisname", "PROVIDES_INPUT",
            ("axes", "max_axis"), ("data_num",), "parse_axes_ascii_and_bound_to_max_axis",
            "active_axis_count", verification_requirements=(
                "parse ODBSYS.axes as the current controlled-axis count",
                "clamp data_num to 1..max_axis; do not invent an axis count",
            ),
        ),
        _rule(
            "axis_count:sysinfo->axisdata", "cnc_sysinfo", "cnc_rdaxisdata", "PROVIDES_INPUT",
            ("axes", "max_axis"), ("len",), "active_axis_count_to_len",
            "active_axis_count", verification_requirements=(
                "use current controlled axes for cls 1/2/3; selected spindle classes remain one axis",
            ),
        ),
        _rule(
            "axis_count:sysinfo->axis_api", "cnc_sysinfo", "__axis_count_targets__", "PROVIDES_INPUT",
            ("axes", "max_axis"), ("axis", "length"), "active_axis_count_to_axis_domain_and_odbaxis_length",
            "axis_1_to_ctrl_axis_or_all_axes", verification_requirements=(
                "axis is -1 (ALL_AXES) or 1..current controlled axes",
                "length is the SDK structure size required by the selected axis mode",
            ),
        ),
        _rule(
            "axis_count:sysinfo->parameter_range_read", "cnc_sysinfo", "cnc_rdparar", "PROVIDES_INPUT",
            ("axes", "max_axis"), ("axis", "length"),
            "active_axis_count_to_axis_domain_and_odbaxis_length",
            "axis_1_to_ctrl_axis_or_all_axes", verification_requirements=(
                "axis is -1 (ALL_AXES) or 1..current controlled axes",
                "length is the SDK structure size required by the selected axis mode",
            ),
        ),
        _rule(
            "axis_name:axisname->axis_api", "cnc_rdaxisname", "__axis_name_targets__", "PROVIDES_INPUT",
            ("name", "suff", "data_num"), ("axis",), "map_name_and_suffix_to_one_based_axis_index",
            "active_axis_name_to_index", verification_requirements=(
                "preserve returned order; do not treat the character name as an axis number",
                "select only an index present in the preceding cnc_rdaxisname result",
            ),
        ),
        _rule(
            "program:directory->search", "cnc_rdprogdir3", "cnc_search", "PROVIDES_INPUT",
            ("number",), ("number",), "select_registered_program_number",
            "registered_program", verification_requirements=("number must be present in the confirmed directory",),
        ),
        _rule(
            "program:directory->copy", "cnc_rdprogdir3", "cnc_copyprog", "PROVIDES_INPUT",
            ("number",), ("src_no", "dst_no"), "select_registered_source_and_unused_destination",
            "source_registered_destination_unused", verification_requirements=(
                "src_no must be registered and readable",
                "dst_no must be legal, unused and not protected",
            ),
        ),
        _rule(
            "program:directory->rename", "cnc_rdprogdir3", "cnc_renameprog", "PROVIDES_INPUT",
            ("number",), ("old_no", "new_no"), "select_registered_source_and_unused_destination",
            "source_registered_destination_unused", verification_requirements=(
                "old_no must be registered; new_no must be legal, unused and not protected",
            ),
        ),
        _rule(
            "program:directory->delete", "cnc_rdprogdir3", "cnc_delete", "PROVIDES_INPUT",
            ("number",), ("number",), "select_registered_non_current_program",
            "registered_not_current_or_protected", verification_requirements=(
                "delete is knowledge-only and requires an explicit human-approved execution step",
            ), dangerous_operation=True, auto_execute=False,
        ),
        _rule(
            "program:directory->upload", "cnc_rdprogdir3", "cnc_upstart3", "PROVIDES_INPUT",
            ("number",), ("s_number", "e_number"), "select_one_registered_program_for_upload",
            "registered_readable_program", verification_requirements=(
                "s_number and e_number identify a confirmed readable program",
                "type=0 is the NC program mode",
            ),
        ),
        _rule(
            "program:directory->download", "cnc_rdprogdir3", "cnc_download3", "PROVIDES_INPUT",
            ("number",), ("data",), "render_nc_payload_with_selected_o_number",
            "unused_unprotected_program", verification_requirements=(
                "selected program number is legal, unused and not protected",
                "leading NC O-number must equal the selected number",
                "download is a request segment inside a successful dwnstart3/dwnend3 session",
            ),
        ),
        _rule(
            "tool_offset:info->tool_read", "cnc_rdtofsinfo", "__tool_offset_targets__", "PROVIDES_INPUT",
            ("use_no", "ofs_type"), ("number", "s_number", "e_number"), "bound_tool_offset_number_to_use_no",
            "available_tool_offset_range", verification_requirements=("use_no is an upper bound; select only available offset numbers",),
        ),
        _rule(
            "macro:info->macro_access", "cnc_rdmacroinfo", "__macro_targets__", "PROVIDES_INPUT",
            ("use_no1", "use_no2"), ("number", "s_number", "e_number"), "expand_0iD_macro_range_indicator",
            "available_macro_range", verification_requirements=(
                "use_no1 identifies local variable capacity",
                "use_no2 is interpreted with the 0i-D documented common-variable range table",
            ),
        ),
        _rule(
            "parameter:range->info", "cnc_rdparanum", "cnc_rdparainfo", "PROVIDES_INPUT",
            ("para_min", "para_max", "total_no"), ("s_number", "read_no"), "walk_parameter_range_then_use_sparse_info",
            "legal_parameter_range", verification_requirements=("read_no is bounded and actual prm_no values come from paraif.info",),
        ),
        _rule(
            "parameter:range->read", "cnc_rdparanum", "__parameter_read_targets__", "PROVIDES_INPUT",
            ("para_min", "para_max", "total_no"), ("number", "prm_no"), "select_parameter_within_reported_range",
            "legal_parameter_range", verification_requirements=("do not assume every number in the min/max range exists",),
        ),
        _rule(
            "parameter:info->read", "cnc_rdparainfo", "__parameter_read_targets__", "PROVIDES_INPUT",
            ("info[].prm_no", "info[].prm_type", "prev_no", "next_no"), ("number", "prm_no", "axis", "length"), "use_sparse_parameter_info_and_axis_attribute",
            "existing_parameter_with_matching_type", verification_requirements=(
                "select only prm_no returned in paraif.info",
                "respect axis/type/read-protection attributes before reading",
            ),
        ),
        _rule(
            "work_offset:info->offset_access", "cnc_rdzofsinfo", "__zero_offset_targets__", "PROVIDES_INPUT",
            ("use_no",), ("number", "s_number", "e_number"), "bound_work_offset_number_to_use_no",
            "available_work_offset_range", verification_requirements=("select only an offset index supported by the returned use_no",),
        ),
        _rule(
            "pmc:info->range", "pmc_rdpmcinfo", "pmc_rdpmcrng", "PROVIDES_INPUT",
            ("info[].pmc_adr", "info[].top_num", "info[].last_num", "info[].adr_attr"),
            ("adr_type", "data_type", "s_number", "e_number"), "map_address_kind_and_bound_address_range",
            "supported_pmc_address_and_data_type", verification_requirements=(
                "adr_type must come from pmc_adr; s/e are bounded by top_num/last_num",
                "data_type requires the documented address-type table; do not infer it from C types",
                "write-protected address metadata is read-only evidence",
            ),
        ),
        _rule(
            "sram:info->start_name", "cnc_getsraminfo", "cnc_sramgetstart", "PROVIDES_INPUT",
            ("info[].sramname",), ("name",), "identity", "available_sram_name",
            verification_requirements=(
                "name must be one of the SRAM names returned by the current cnc_getsraminfo call",
                "do not substitute a historical or guessed SRAM name",
            ),
        ),
        _rule(
            "path:get->set", "cnc_getpath", "cnc_setpath", "PROVIDES_INPUT",
            ("path_no", "maxpath_no"), ("path_no",), "select_path_in_0_to_maxpath_range",
            "valid_path_number", verification_requirements=("do not call setpath unless path_no is within returned range",),
        ),
        _rule(
            "path:set->data_window", "cnc_setpath", "__path_sensitive_targets__", "REQUIRES",
            ("path_no",), ("selected_path_context",), "bind_data_window_to_selected_path",
            "active_path_context", verification_requirements=(
                "selected path is session state, not a guessed formal argument",
                "restore the previously observed path after a path-sensitive read",
            ), parameter_kind="session_context",
        ),
        _rule(
            "program:current->guarded_operation", "cnc_rdprgnum", "cnc_delete", "PROVIDES_INPUT",
            ("data", "mdata"), ("number",), "exclude_current_and_main_program_from_destructive_selection",
            "exclude_current_program", verification_requirements=("current and main program numbers are exclusion evidence, not a deletion target",), dangerous_operation=True, auto_execute=False,
        ),
        _rule(
            "program:sequence->execution_read", "cnc_rdseqnum", "cnc_rdexecprog", "PROVIDES_INPUT",
            ("data",), ("blknum",), "use_sequence_as_execution_block_cursor",
            "current_execution_context", verification_requirements=("only use after confirming an executing program context",),
        ),
        _rule(
            "write_verify:param", "cnc_wrparam", "cnc_rdparam", "READS_AFTER_WRITE",
            ("param",), ("number", "axis"), "read_back_written_parameter_using_snapshot_metadata",
            "write_then_readback", verification_requirements=("snapshot and restore are required; this is knowledge-only",), dangerous_operation=True, auto_execute=False,
        ),
        _rule(
            "write_verify:macro", "cnc_wrmacro", "cnc_rdmacro", "READS_AFTER_WRITE",
            ("number", "mcr_val", "dec_val"), ("number",), "read_back_written_macro_value",
            "write_then_readback", verification_requirements=("snapshot and restore are required; this is knowledge-only",), dangerous_operation=True, auto_execute=False,
        ),
        _rule(
            "write_verify:tool_offset", "cnc_wrtofs", "cnc_rdtofs", "READS_AFTER_WRITE",
            ("number", "type", "data"), ("number", "type"), "read_back_written_tool_offset",
            "write_then_readback", verification_requirements=("snapshot and restore are required; this is knowledge-only",), dangerous_operation=True, auto_execute=False,
        ),
        _rule(
            "write_verify:work_offset", "cnc_wrzofs", "cnc_rdzofs", "READS_AFTER_WRITE",
            ("zofs",), ("number", "axis"), "read_back_written_work_offset",
            "write_then_readback", verification_requirements=("snapshot and restore are required; this is knowledge-only",), dangerous_operation=True, auto_execute=False,
        ),
    ]
    return tuple(rules)


PARAMETER_DISCOVERY_RULES = _build_rules()


_TARGET_GROUPS = {
    "__axis_count_targets__": _AXIS_COUNT_TARGETS,
    "__axis_name_targets__": _AXIS_NAME_TARGETS,
    "__tool_offset_targets__": _TOOL_OFFSET_TARGETS,
    "__macro_targets__": _MACRO_TARGETS,
    "__parameter_read_targets__": _PARAMETER_TARGETS,
    "__zero_offset_targets__": _ZERO_OFFSET_TARGETS,
    "__path_sensitive_targets__": _PATH_SENSITIVE_TARGETS,
}

_TARGET_PARAMETER_OVERRIDES = {
    "cnc_rdtofs": ("number",),
    "cnc_rdtofsr": ("s_number", "e_number"),
    "cnc_wrtofs": ("number",),
    "cnc_wrtofsr": ("tofsr",),
    "cnc_rdmacro": ("number",),
    "cnc_wrmacro": ("number",),
    "cnc_rdmacror": ("s_number", "e_number"),
    "cnc_rdparam": ("number", "axis", "length"),
    "cnc_rdparam3": ("number", "axis", "length"),
    "cnc_rdparam_ext": ("prm_no", "num"),
    "cnc_rdparar": ("s_number", "e_number", "axis", "length"),
    "cnc_wrparam": ("param",),
    "cnc_wrparas": ("param",),
    "cnc_rdzofs": ("number", "axis"),
    "cnc_rdzofsr": ("s_number", "e_number", "axis"),
    "cnc_wrzofs": ("zofs",),
    "cnc_wrzofsr": ("zofsr",),
}


def expanded_rule_targets(rule: ParameterDiscoveryRule) -> tuple[str, ...]:
    return tuple(_TARGET_GROUPS.get(rule.target_api, (rule.target_api,)))


def target_parameter_names(rule: ParameterDiscoveryRule, target_api: str) -> tuple[str, ...]:
    return tuple(_TARGET_PARAMETER_OVERRIDES.get(target_api, rule.target_parameters))


@lru_cache(maxsize=1)
def _chunk_index() -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    try:
        rows = DEFAULT_CHUNKS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return index
    for line in rows:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        function = str(row.get("function", "")).strip()
        kind = str(row.get("chunk_type", "")).strip()
        chunk_id = str(row.get("chunk_id", "")).strip()
        if function and kind and chunk_id:
            index.setdefault(function, {}).setdefault(kind, {"chunk_id": chunk_id, "text": str(row.get("text", ""))})
    return index


def _documentation_evidence(api_name: str) -> tuple[str, ...]:
    kinds = _chunk_index().get(api_name, {})
    return tuple(kinds[kind]["chunk_id"] for kind in ("prototype", "overview", "arguments") if kinds.get(kind))


def _header_has_declaration(api_name: str, header_path: Path) -> bool:
    try:
        text = header_path.read_text(encoding="latin-1")
    except OSError:
        return False
    return bool(re.search(rf"\b{re.escape(api_name)}\s*\(", text))


def _field_is_documented(api_name: str, field: str) -> bool:
    argument_text = _chunk_index().get(api_name, {}).get("arguments", {}).get("text", "")
    if not argument_text:
        return False
    token = str(field).replace("[]", "")
    aliases = {
        "program_number": "number",
        "registered_programs": "number",
        "protected_programs": "number",
    }
    token = aliases.get(token, token)
    return token in argument_text or token.split(".")[-1] in argument_text


def _formal_parameter_is_valid(api_name: str, parameter: str, kind: str) -> bool:
    if kind == "session_context":
        return True
    return parameter in documented_parameter_names(api_name)


def resolve_rules(
    api_names: Iterable[str],
    *,
    allowed_api_names: Iterable[str] | None = None,
    device_scope: str = F0ID_SCOPE_ID,
    header_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve rules against the manifest scope and local SDK/RAG evidence.

    The returned ``relations`` are safe to persist.  ``candidates`` are kept
    as diagnostics and are never promoted to verified graph edges.
    """
    requested = {str(name).strip() for name in api_names if str(name).strip()}
    if allowed_api_names is not None:
        scope: set[str] | None = {str(name).strip() for name in allowed_api_names if str(name).strip()}
    elif device_scope == F0ID_SCOPE_ID:
        scope = set(f0id_api_names())
    else:
        scope = None
    header = header_path or (default_focas_header_dir() / "Fwlib32.h")
    relations: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    analyzed: set[str] = set()
    for rule in PARAMETER_DISCOVERY_RULES:
        targets = expanded_rule_targets(rule)
        matched_targets = [target for target in targets if target in requested and (scope is None or target in scope)]
        if not matched_targets:
            continue
        analyzed.update(matched_targets)
        for target in matched_targets:
            target_parameters = target_parameter_names(rule, target)
            source_ok = scope is None or rule.source_api in scope
            doc_ids = tuple(dict.fromkeys((*_documentation_evidence(rule.source_api), *_documentation_evidence(target))))
            missing: list[str] = []
            if not source_ok:
                missing.append("source API outside selected device scope")
            if not _documentation_evidence(rule.source_api):
                missing.append(f"missing RAG prototype/arguments for {rule.source_api}")
            if not _documentation_evidence(target):
                missing.append(f"missing RAG prototype/arguments for {target}")
            for field in rule.source_fields:
                if not _field_is_documented(rule.source_api, field):
                    missing.append(f"source field not documented: {rule.source_api}.{field}")
            for parameter in target_parameters:
                if not _formal_parameter_is_valid(target, parameter, rule.parameter_kind):
                    missing.append(f"target parameter not in SDK prototype: {target}.{parameter}")
            if not _header_has_declaration(rule.source_api, header):
                missing.append(f"SDK declaration missing: {rule.source_api}")
            if not _header_has_declaration(target, header):
                missing.append(f"SDK declaration missing: {target}")
            payload = asdict(rule)
            payload["target_api"] = target
            payload["target_parameters"] = target_parameters
            payload["evidence_ids"] = list(doc_ids) + [f"{header}:declaration:{rule.source_api}", f"{header}:declaration:{target}"]
            payload["applicability"] = {**rule.applicability, "device_scope": device_scope} if device_scope else dict(rule.applicability)
            if missing:
                candidates.append({"rule": payload, "reason": missing})
                continue
            payload["confidence_status"] = "sdk_validated"
            payload["evidence_ids"] = list(dict.fromkeys(payload["evidence_ids"]))
            relations.append(payload)
    return {
        "relations": relations,
        "candidates": candidates,
        "analyzed_api_count": len(analyzed),
        "scanned_api_count": len(requested & scope) if scope is not None else len(requested),
        "rule_count": len(relations),
        "candidate_count": len(candidates),
        "scope_api_count": len(scope or ()),
    }


def rules_for_target(api_name: str, *, device_scope: str = F0ID_SCOPE_ID) -> dict[str, Any]:
    """Convenience exact lookup used by offline Planner tests."""
    return resolve_rules([api_name], allowed_api_names=f0id_api_names(), device_scope=device_scope)
