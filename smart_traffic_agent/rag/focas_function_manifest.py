from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET


DEFAULT_FOCAS_FUNCTION_WORKBOOK = Path(r"e:\2025-06\codenew\focas_funcs.xlsx")
DEFAULT_FOCAS_FUNCTION_MANIFEST = Path("rag_indexes/focas/function_manifest.jsonl")
DEFAULT_COVERAGE_BATCH_SIZE = 40
DEFAULT_SUPPORT_BATCH_SIZE = 16
DEFAULT_COVERAGE_SEGMENT_COUNT = 3

SHEET_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
FUNCTION_SPLIT_RE = re.compile(r"\s*(?:/|,|，|、|\n|\r)\s*")
FUNCTION_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEADER_ALIASES = {
    "function": {"function", "函数", "对应函数", "api", "api function", "对应api", "功能函数"},
    "meaning": {"meaning", "description", "含义", "说明", "英文含义", "description/function"},
    "category": {"category", "分类", "函数类型", "type", "group"},
}

SCENARIO_BATCH_SPECS: list[dict[str, Any]] = [
    {
        "scenario_batch_id": "programmed_coordinate_motion",
        "scenario_types": {"coordinate_motion", "feed_speed_change", "general_status_collection", "comprehensive_focas_traffic"},
        "main_state_driver": "upload_select_run_single_block_nc_program",
        "quality_target": "program lifecycle evidence plus position/feed/spindle/status variation during NC motion",
        "target_keywords": [
            "absolute",
            "machine",
            "relative",
            "rdposition",
            "distance",
            "position",
            "dynamic",
            "actf",
            "acts",
            "speed",
            "spindle",
            "servo delay",
            "skip",
        ],
        "support_functions": [
            "cnc_allclibhndl3",
            "cnc_freelibhndl",
            "cnc_statinfo",
            "cnc_rdprogdir",
            "cnc_rdprogdir2",
            "cnc_rdprogdir3",
            "cnc_dwnstart3",
            "cnc_download3",
            "cnc_dwnend3",
            "cnc_search",
            "cnc_rdprgnum",
            "cnc_alarm2",
            "cnc_rdalmmsg",
        ],
    },
    {
        "scenario_batch_id": "program_lifecycle",
        "scenario_types": {"program_lifecycle"},
        "main_state_driver": "create_select_read_upload_download_program_artifacts",
        "quality_target": "program directory/transfer/selection/readback state transitions",
        "target_keywords": ["program", "prog", "download", "upload", "dwn", "search", "seq", "mdi"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "tool_offset_work_coordinate_restore",
        "scenario_types": {"tool_offset_setting", "work_coordinate_setting"},
        "main_state_driver": "read_write_readback_restore_offsets_with_bounded_values",
        "quality_target": "tool/work-offset read ranges plus write-readback-restore evidence when safe",
        "target_keywords": ["tofs", "zofs", "offset", "work zero", "tool offset", "radofs", "lenofs"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2", "cnc_rdalmmsg"],
    },
    {
        "scenario_batch_id": "tool_life_management",
        "scenario_types": {"tool_life_management", "tool_management"},
        "main_state_driver": "read_tool_life_tables_then_bounded_write_restore_if_supported",
        "quality_target": "tool life/tool management data readback and safe restore evidence",
        "target_keywords": ["tlife", "tool life", "tool management", "tooldata", "toolinfo", "tool group", "grpid"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "macro_parameter_state_restore",
        "scenario_types": {"parameter_read", "parameter_write_simulated", "macro_variable_read_write"},
        "main_state_driver": "read_ranges_then_bounded_parameter_or_macro_write_readback_restore",
        "quality_target": "parameter/macro read coverage plus safe write-readback-restore where authorized",
        "target_keywords": ["param", "macro", "setting", "prm", "paras"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "pmc_signal_state",
        "scenario_types": {"pmc_signal_read", "pmc_signal_monitoring"},
        "main_state_driver": "read_pmc_metadata_ranges_then_bounded_pmc_write_restore_if_safe",
        "quality_target": "PMC metadata/range traffic and optional write-readback-restore evidence",
        "target_category_keywords": ["pmc"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "data_server_option_probe",
        "scenario_types": {"data_server", "file_server"},
        "main_state_driver": "probe_data_server_directory_mode_transfer_status_without_destructive_file_ops",
        "quality_target": "Data Server option support/unsupported evidence from real return codes",
        "target_category_keywords": ["data server", "dnc"],
        "target_keywords": ["dtsv", "hdd", "host", "file", "ftp", "ds"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "profibus_option_probe",
        "scenario_types": {"profibus", "profibus_dp"},
        "main_state_driver": "probe_profibus_configuration_and_status_options",
        "quality_target": "PROFIBUS option support/unsupported evidence from real return codes",
        "target_category_keywords": ["profibus"],
        "target_keywords": ["prf", "slave", "bus"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo", "cnc_alarm2"],
    },
    {
        "scenario_batch_id": "foundational_system_probe",
        "scenario_types": set(),
        "main_state_driver": "connect_and_read_system_node_status_configuration",
        "quality_target": "low-risk system/library/status baseline evidence",
        "target_keywords": ["node", "lib", "system", "config", "status", "alarm", "diagnos"],
        "support_functions": ["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_statinfo"],
    },
]


def build_focas_function_manifest(
    workbook_path: Path = DEFAULT_FOCAS_FUNCTION_WORKBOOK,
    output_path: Path = DEFAULT_FOCAS_FUNCTION_MANIFEST,
) -> dict[str, Any]:
    rows = load_focas_function_rows(workbook_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summarize_manifest(rows, workbook_path=workbook_path, output_path=output_path)


def load_function_manifest(path: Path = DEFAULT_FOCAS_FUNCTION_MANIFEST) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("function"):
            rows.append(row)
    return rows


def build_function_coverage_context(
    manifest_path: Path = DEFAULT_FOCAS_FUNCTION_MANIFEST,
    *,
    covered_functions: set[str] | None = None,
    batch_size: int = DEFAULT_COVERAGE_BATCH_SIZE,
    scenario_type: str = "comprehensive_focas_traffic",
    support_batch_size: int = DEFAULT_SUPPORT_BATCH_SIZE,
    max_segments: int = DEFAULT_COVERAGE_SEGMENT_COUNT,
) -> dict[str, Any]:
    rows = dedupe_manifest_rows(load_function_manifest(manifest_path))
    covered = set(covered_functions or set())
    remaining = [row for row in rows if str(row.get("function", "")) not in covered]
    segments = select_scenario_segments(
        rows,
        remaining,
        scenario_type=scenario_type,
        target_batch_size=max(1, batch_size),
        support_batch_size=max(0, support_batch_size),
        max_segments=max(1, max_segments),
    )
    primary_segment = segments[0] if segments else empty_segment()
    target_batch = [row for segment in segments for row in segment["target_batch"]]
    support_batch = dedupe_batch_rows([row for segment in segments for row in segment["support_batch"]])
    selected_batch = target_batch + support_batch
    family_counts = Counter(str(row.get("function_family", "")) for row in rows)
    remaining_family_counts = Counter(str(row.get("function_family", "")) for row in remaining)
    return {
        "enabled": bool(rows),
        "source_manifest": str(manifest_path),
        "scenario_batch_id": primary_segment["scenario_batch_id"],
        "main_state_driver": primary_segment["main_state_driver"],
        "quality_target": primary_segment["quality_target"],
        "selection_strategy": "multi_segment_scenario_cluster_target_plus_support",
        "segments": segments,
        "segment_count": len(segments),
        "target_function_count": len(rows),
        "covered_function_count": len(covered.intersection({str(row.get("function", "")) for row in rows})),
        "remaining_function_count": len(remaining),
        "selected_batch_size": len(selected_batch),
        "target_batch_size": len(target_batch),
        "support_batch_size": len(support_batch),
        "batch_size_limit": max(1, batch_size),
        "function_family_counts": dict(sorted(family_counts.items())),
        "remaining_family_counts": dict(sorted(remaining_family_counts.items())),
        "target_batch": target_batch,
        "support_batch": support_batch,
        "selected_batch": selected_batch,
}


def select_scenario_segments(
    all_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    *,
    scenario_type: str,
    target_batch_size: int,
    support_batch_size: int,
    max_segments: int,
) -> list[dict[str, Any]]:
    specs = scenario_specs_for(scenario_type)
    remaining_by_function = {str(row.get("function", "")): row for row in remaining_rows}
    selected_targets: set[str] = set()
    segments: list[dict[str, Any]] = []
    per_segment_limit = max(1, target_batch_size // max_segments)
    for spec in specs:
        if len(selected_targets) >= target_batch_size or len(segments) >= max_segments:
            break
        available_rows = [
            row
            for row in remaining_by_function.values()
            if str(row.get("function", "")) not in selected_targets and row_matches_spec(row, spec)
        ]
        if not available_rows:
            continue
        remaining_slots = target_batch_size - len(selected_targets)
        segment_limit = min(per_segment_limit, remaining_slots)
        target_rows = [
            annotate_batch_row(row, role="target", spec=spec, reason="matched scenario segment target keywords/category")
            for row in available_rows[:segment_limit]
        ]
        selected_targets.update(str(row.get("function", "")) for row in target_rows)
        segment_id = f"{len(segments) + 1:02d}_{spec['scenario_batch_id']}"
        support_rows = support_rows_for_spec(all_rows, spec, target_rows, support_batch_size)
        assign_segment_id(target_rows, segment_id)
        assign_segment_id(support_rows, segment_id)
        segments.append(
            {
                "segment_id": segment_id,
                "scenario_batch_id": spec["scenario_batch_id"],
                "main_state_driver": spec["main_state_driver"],
                "quality_target": spec["quality_target"],
                "selection_strategy": "multi_segment_scenario_cluster_target_plus_support",
                "nc_program_required": nc_program_required_for_spec(spec),
                "target_batch": target_rows,
                "support_batch": support_rows,
            }
        )

    if segments:
        return segments
    return [select_scenario_batch(
        all_rows,
        remaining_rows,
        scenario_type=scenario_type,
        target_batch_size=target_batch_size,
        support_batch_size=support_batch_size,
    )]


def select_scenario_batch(
    all_rows: list[dict[str, Any]],
    remaining_rows: list[dict[str, Any]],
    *,
    scenario_type: str,
    target_batch_size: int,
    support_batch_size: int,
) -> dict[str, Any]:
    candidate_specs = scenario_specs_for(scenario_type)
    for spec in candidate_specs:
        target_rows = [
            annotate_batch_row(row, role="target", spec=spec, reason="matched scenario target keywords/category")
            for row in remaining_rows
            if row_matches_spec(row, spec)
        ][:target_batch_size]
        if target_rows:
            support_rows = support_rows_for_spec(all_rows, spec, target_rows, support_batch_size)
            segment_id = f"01_{spec['scenario_batch_id']}"
            assign_segment_id(target_rows, segment_id)
            assign_segment_id(support_rows, segment_id)
            return {
                "segment_id": segment_id,
                "scenario_batch_id": spec["scenario_batch_id"],
                "main_state_driver": spec["main_state_driver"],
                "quality_target": spec["quality_target"],
                "selection_strategy": "scenario_cluster_target_plus_support",
                "nc_program_required": nc_program_required_for_spec(spec),
                "target_batch": target_rows,
                "support_batch": support_rows,
            }

    fallback_spec = SCENARIO_BATCH_SPECS[-1]
    target_rows = [
        annotate_batch_row(row, role="target", spec=fallback_spec, reason="fallback remaining manifest function")
        for row in remaining_rows[:target_batch_size]
    ]
    support_rows = support_rows_for_spec(all_rows, fallback_spec, target_rows, support_batch_size)
    assign_segment_id(target_rows, "01_fallback_remaining_manifest")
    assign_segment_id(support_rows, "01_fallback_remaining_manifest")
    return {
        "segment_id": "01_fallback_remaining_manifest",
        "scenario_batch_id": "fallback_remaining_manifest",
        "main_state_driver": "cover_remaining_manifest_functions_with_minimal_safe_context",
        "quality_target": "attempt remaining manifest functions with bounded arguments and support evidence",
        "selection_strategy": "fallback_remaining_order_after_scenario_specs_empty",
        "nc_program_required": False,
        "target_batch": target_rows,
        "support_batch": support_rows,
    }


def empty_segment() -> dict[str, Any]:
    return {
        "segment_id": "",
        "scenario_batch_id": "",
        "main_state_driver": "",
        "quality_target": "",
        "selection_strategy": "",
        "nc_program_required": False,
        "target_batch": [],
        "support_batch": [],
    }


def dedupe_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        function = str(row.get("function", "")).strip()
        role = str(row.get("coverage_role", "")).strip()
        key = f"{role}:{function}"
        if not function or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def assign_segment_id(rows: list[dict[str, Any]], segment_id: str) -> None:
    for row in rows:
        row["segment_id"] = segment_id


def nc_program_required_for_spec(spec: dict[str, Any]) -> bool:
    driver = str(spec.get("main_state_driver", "")).lower()
    scenario_id = str(spec.get("scenario_batch_id", "")).lower()
    return "nc_program" in driver or "programmed_coordinate_motion" in scenario_id or "program_lifecycle" in scenario_id


def scenario_specs_for(scenario_type: str) -> list[dict[str, Any]]:
    scenario = scenario_type.strip().lower()
    matched = [
        spec
        for spec in SCENARIO_BATCH_SPECS
        if scenario in {str(item).lower() for item in spec.get("scenario_types", set())}
    ]
    return matched + [spec for spec in SCENARIO_BATCH_SPECS if spec not in matched]


def support_rows_for_spec(
    all_rows: list[dict[str, Any]],
    spec: dict[str, Any],
    target_rows: list[dict[str, Any]],
    support_batch_size: int,
) -> list[dict[str, Any]]:
    if support_batch_size <= 0:
        return []
    target_functions = {str(row.get("function", "")) for row in target_rows}
    by_function = {str(row.get("function", "")): row for row in all_rows}
    support_rows: list[dict[str, Any]] = []
    for function in spec.get("support_functions", []):
        row = by_function.get(function)
        if row is None or function in target_functions:
            continue
        support_rows.append(
            annotate_batch_row(row, role="support", spec=spec, reason="required to create/verify/recover scenario state")
        )
        if len(support_rows) >= support_batch_size:
            break
    return support_rows


def annotate_batch_row(row: dict[str, Any], *, role: str, spec: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "function": row.get("function"),
        "raw_function": row.get("raw_function"),
        "symbol_candidates": row.get("symbol_candidates", []),
        "category": row.get("category", ""),
        "description": row.get("description", ""),
        "source_row": row.get("source_row"),
        "coverage_status": row.get("coverage_status", "unplanned"),
        "coverage_role": role,
        "scenario_batch_id": spec.get("scenario_batch_id"),
        "main_state_driver": spec.get("main_state_driver"),
        "selection_reason": reason,
    }


def row_matches_spec(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    category = str(row.get("category", "")).lower()
    category_keywords = [str(item).lower() for item in spec.get("target_category_keywords", [])]
    if category_keywords and any(keyword in category for keyword in category_keywords):
        return True
    text = " ".join(
        [
            str(row.get("function", "")),
            str(row.get("raw_function", "")),
            str(row.get("description", "")),
            str(row.get("category", "")),
        ]
    ).lower()
    return any(str(keyword).lower() in text for keyword in spec.get("target_keywords", []))


def dedupe_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        function = str(row.get("function", "")).strip()
        if not function or function in seen:
            continue
        seen.add(function)
        deduped.append(row)
    return deduped


def load_focas_function_rows(workbook_path: Path) -> list[dict[str, Any]]:
    sheets = read_xlsx_sheets(workbook_path)
    manifest_rows: list[dict[str, Any]] = []
    for sheet_name, rows in sheets.items():
        if not rows:
            continue
        manifest_rows.extend(parse_sheet_rows(rows, workbook_path=workbook_path, sheet_name=sheet_name))
    return manifest_rows


def parse_sheet_rows(
    rows: list[dict[str, str]],
    *,
    workbook_path: Path,
    sheet_name: str,
) -> list[dict[str, Any]]:
    header = detect_header(rows)
    if header:
        return parse_headered_rows(rows, header, workbook_path=workbook_path, sheet_name=sheet_name)
    return parse_focas_list_rows(rows, workbook_path=workbook_path, sheet_name=sheet_name)


def parse_focas_list_rows(
    rows: list[dict[str, str]],
    *,
    workbook_path: Path,
    sheet_name: str,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    current_category = ""
    for row in rows:
        category = row.get("A", "").strip()
        if category:
            current_category = category
        function_cell = row.get("B", "").strip()
        meaning = row.get("C", "").strip()
        for raw_function in split_function_cell(function_cell):
            manifest_rows.append(
                make_manifest_row(
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    source_row=int(row["_row"]),
                    source_columns={"A": category, "B": function_cell, "C": meaning},
                    category=current_category,
                    raw_function=raw_function,
                    description=meaning,
                    function_cell=function_cell,
                    layout="focas_funcs_list",
                )
            )
    return manifest_rows


def parse_headered_rows(
    rows: list[dict[str, str]],
    header: dict[str, str],
    *,
    workbook_path: Path,
    sheet_name: str,
) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    header_row = header["_row"]
    function_col = header["function"]
    meaning_col = header.get("meaning", "")
    category_col = header.get("category", "")
    current_category = ""
    for row in rows:
        if row["_row"] <= header_row:
            continue
        category = row.get(category_col, "").strip() if category_col else ""
        if category:
            current_category = category
        function_cell = row.get(function_col, "").strip()
        description = row.get(meaning_col, "").strip() if meaning_col else ""
        for raw_function in split_function_cell(function_cell):
            manifest_rows.append(
                make_manifest_row(
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    source_row=int(row["_row"]),
                    source_columns={col: row.get(col, "") for col in sorted(row) if col != "_row"},
                    category=current_category,
                    raw_function=raw_function,
                    description=description,
                    function_cell=function_cell,
                    layout="headered_table",
                )
            )
    return manifest_rows


def make_manifest_row(
    *,
    workbook_path: Path,
    sheet_name: str,
    source_row: int,
    source_columns: dict[str, str],
    category: str,
    raw_function: str,
    description: str,
    function_cell: str,
    layout: str,
) -> dict[str, Any]:
    symbol_candidates = symbol_candidates_for(raw_function, category)
    function = symbol_candidates[0] if symbol_candidates else raw_function
    return {
        "manifest_id": manifest_id(sheet_name, source_row, raw_function),
        "source_workbook": str(workbook_path),
        "sheet": sheet_name,
        "source_row": source_row,
        "source_layout": layout,
        "category": category,
        "raw_function": raw_function,
        "function_cell": function_cell,
        "function": function,
        "symbol_candidates": symbol_candidates,
        "function_family": function.split("_", 1)[0] if "_" in function else "",
        "description": description,
        "source_columns": source_columns,
        "coverage_status": "unplanned",
    }


def symbol_candidates_for(raw_function: str, category: str = "") -> list[str]:
    cleaned = raw_function.strip()
    if not cleaned:
        return []
    if cleaned.startswith(("cnc_", "pmc_", "eth_", "dtsv_")):
        return [cleaned]

    category_lower = category.lower()
    candidates: list[str] = []
    if category_lower in {"pmc", "profibus-dp"}:
        candidates.append(f"pmc_{cleaned}")
    else:
        candidates.append(f"cnc_{cleaned}")
    fallback = f"cnc_{cleaned}"
    if fallback not in candidates:
        candidates.append(fallback)
    raw_candidate = cleaned
    if raw_candidate not in candidates:
        candidates.append(raw_candidate)
    return candidates


def split_function_cell(function_cell: str) -> list[str]:
    if not function_cell:
        return []
    tokens = [token.strip() for token in FUNCTION_SPLIT_RE.split(function_cell) if token.strip()]
    return [token for token in tokens if FUNCTION_TOKEN_RE.match(token)]


def detect_header(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows[:10]:
        normalized = {col: normalize_header(value) for col, value in row.items() if col != "_row"}
        found: dict[str, str] = {"_row": row["_row"]}
        for role, aliases in HEADER_ALIASES.items():
            for col, value in normalized.items():
                if value in aliases:
                    found[role] = col
                    break
        if "function" in found:
            return found
    return {}


def normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def manifest_id(sheet_name: str, source_row: int, raw_function: str) -> str:
    safe_sheet = re.sub(r"[^a-z0-9]+", "-", sheet_name.lower()).strip("-") or "sheet"
    safe_function = re.sub(r"[^a-z0-9_]+", "-", raw_function.lower()).strip("-") or "function"
    return f"{safe_sheet}-r{source_row:04d}-{safe_function}"


def summarize_manifest(rows: list[dict[str, Any]], *, workbook_path: Path, output_path: Path) -> dict[str, Any]:
    functions = [str(row["function"]) for row in rows]
    raw_functions = [str(row["raw_function"]) for row in rows]
    categories = [str(row.get("category", "")) for row in rows if row.get("category")]
    family_counts = Counter(str(row.get("function_family", "")) for row in rows)
    return {
        "source_workbook": str(workbook_path),
        "output_path": str(output_path),
        "row_count": len(rows),
        "unique_function_count": len(set(functions)),
        "unique_raw_function_count": len(set(raw_functions)),
        "category_count": len(set(categories)),
        "categories": sorted(set(categories)),
        "duplicate_raw_functions": sorted(name for name, count in Counter(raw_functions).items() if count > 1),
        "function_family_counts": dict(sorted(family_counts.items())),
    }


def read_xlsx_sheets(workbook_path: Path) -> dict[str, list[dict[str, str]]]:
    if not workbook_path.exists():
        raise FileNotFoundError(f"FOCAS function workbook not found: {workbook_path}")
    with ZipFile(workbook_path) as archive:
        shared_strings = read_shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall("rel:Relationship", REL_NS)
        }
        sheets: dict[str, list[dict[str, str]]] = {}
        sheets_element = workbook.find("a:sheets", SHEET_NS)
        if sheets_element is None:
            return sheets
        for sheet in sheets_element:
            sheet_name = sheet.attrib["name"]
            rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rel_targets[rel_id].lstrip("/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            sheets[sheet_name] = read_sheet_rows(archive, sheet_path, shared_strings)
        return sheets


def read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", SHEET_NS):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", SHEET_NS)))
    return strings


def read_sheet_rows(archive: ZipFile, sheet_path: str, shared_strings: list[str]) -> list[dict[str, str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[dict[str, str]] = []
    for xml_row in root.findall(".//a:sheetData/a:row", SHEET_NS):
        row: dict[str, str] = {"_row": xml_row.attrib["r"]}
        for cell in xml_row.findall("a:c", SHEET_NS):
            cell_ref = cell.attrib.get("r", "")
            column = re.sub(r"\d+", "", cell_ref)
            row[column] = read_cell_value(cell, shared_strings)
        rows.append(row)
    return rows


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", SHEET_NS))
    value = cell.find("a:v", SHEET_NS)
    if value is None:
        return ""
    raw = value.text or ""
    if cell.attrib.get("t") == "s" and raw:
        return shared_strings[int(raw)]
    return raw
