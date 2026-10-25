"""Extract factual FOCAS return-code mappings from indexed references and SDKs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


_DEFINE = re.compile(r"(?m)^\s*#\s*define\s+(EW_[A-Z0-9_]+)\s+\(?\s*(-?\d+)\s*\)?\b")
_SITE_VALUE = re.compile(r"(?m)^(EW_[A-Z0-9_]+)\s*\n\s*\((-?\d+)\)\s*$")
_API_NAME = re.compile(r"(?m)^-\s+(EW_[A-Z0-9_]+)(?:\(([^)]*)\))?\s*:")


def _normalize_explanation(text: str) -> str:
    # The legacy HTML/XML-to-text cleaner sometimes joined adjacent elements
    # without whitespace, e.g. "errorSize". Preserve wording but restore the
    # obvious boundary so both the CSV and RAG text remain readable.
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sdk_code_values(header_paths: list[Path]) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Merge installed SDK definitions, retaining cross-series conflicts."""
    values: dict[str, set[int]] = defaultdict(set)
    for path in header_paths:
        for name, value in _DEFINE.findall(path.read_text(encoding="latin-1")):
            values[name].add(int(value))
    stable = {name: next(iter(found)) for name, found in values.items() if len(found) == 1}
    conflicts = {name: sorted(found) for name, found in values.items() if len(found) > 1}
    return stable, conflicts


def site_code_values(chunks: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[int]]]:
    values: dict[str, set[int]] = defaultdict(set)
    # The original errcode.htm was split into multiple reference_page chunks;
    # each table entry retains adjacent name and numeric value lines.
    for chunk in chunks:
        if chunk.get("source_path") != "errcode.htm":
            continue
        for name, value in _SITE_VALUE.findall(str(chunk.get("text", ""))):
            values[name].add(int(value))
    stable = {name: next(iter(found)) for name, found in values.items() if len(found) == 1}
    conflicts = {name: sorted(found) for name, found in values.items() if len(found) > 1}
    return stable, conflicts


def site_code_descriptions(chunks: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, str]]]:
    """Keep the global CNC/PMC meaning and handling text separately."""
    pages = sorted(
        (chunk for chunk in chunks if chunk.get("source_path") == "errcode.htm"),
        key=lambda chunk: int(chunk.get("chunk_index", 0)),
    )
    text = "\n".join(str(chunk.get("text", "")) for chunk in pages)
    matches = list(_SITE_VALUE.finditer(text))
    descriptions: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for index, match in enumerate(matches):
        previous = text[:match.start()]
        section = "pmc" if previous.rfind("PMC Data Window") > previous.rfind("CNC Data Window") else "cnc"
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        detail = text[match.end():end]
        detail = re.split(r"(?m)^(?:PMC Data Window|Ethernet Version Only)\s*$", detail, maxsplit=1)[0]
        lines = [line.strip() for line in detail.splitlines() if line.strip()]
        if lines:
            descriptions[match.group(1)][section] = {
                "meaning": _normalize_explanation(lines[0]),
                "handling": _normalize_explanation("\n".join(lines[1:])),
            }
    return dict(descriptions)


def extract_api_return_codes(
    chunks: list[dict[str, Any]],
    sdk_values: dict[str, int],
    site_values: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for chunk in chunks:
        if chunk.get("chunk_type") != "error_codes":
            continue
        function = str(chunk.get("function", ""))
        if not function:
            continue
        error_text = str(chunk.get("text", ""))
        matches = list(_API_NAME.finditer(error_text))
        entries: list[tuple[str, str, str]] = []
        for index, match in enumerate(matches):
            detail_end = matches[index + 1].start() if index + 1 < len(matches) else len(error_text)
            explanation = _normalize_explanation(error_text[match.end():detail_end])
            if match.group(2):
                explanation = f"[{match.group(2)}] {explanation}"
            entries.append((match.group(1), explanation, "listed"))
        if not matches:
            # Some pages use prose rather than a list, e.g. cnc_freelibhndl
            # mentions EW_HANDLE as the only exceptional condition.
            prose = error_text.partition("Return/error codes:")[2].strip()
            entries.extend(
                (name, _normalize_explanation(prose), "prose_mention")
                for name in dict.fromkeys(re.findall(r"\bEW_[A-Z0-9_]+\b", prose))
            )
        for name, explanation, extraction_method in entries:
            key = (function, name)
            if key in by_key:
                if explanation and explanation not in by_key[key]["api_explanations"]:
                    by_key[key]["api_explanations"].append(explanation)
                continue
            site_value = site_values.get(name)
            sdk_value = sdk_values.get(name)
            conflict = site_value is not None and sdk_value is not None and site_value != sdk_value
            value = None if conflict else (sdk_value if sdk_value is not None else site_value)
            row = {
                "function": function,
                "code_name": name,
                "code_value": value,
                "value_source": "sdk_and_site" if sdk_value is not None and site_value is not None and not conflict
                else "sdk" if sdk_value is not None and not conflict
                else "site" if site_value is not None and not conflict
                else "conflict" if conflict else "unresolved",
                "api_explanations": [explanation] if explanation else [],
                "extraction_method": extraction_method,
                "source_url": str(chunk.get("source_url", "")),
                "source_path": str(chunk.get("source_path", "")),
                "source_chunk_id": str(chunk.get("chunk_id", "")),
            }
            rows.append(row)
            by_key[key] = row
    return sorted(rows, key=lambda row: (row["function"], row["code_value"] is None, row["code_value"] or 0, row["code_name"]))


def build_return_code_catalog(
    chunks: list[dict[str, Any]], header_paths: list[Path]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sdk, sdk_conflicts = sdk_code_values(header_paths)
    site, site_conflicts = site_code_values(chunks)
    site_descriptions = site_code_descriptions(chunks)
    api_rows = extract_api_return_codes(chunks, sdk, site)
    all_names = sorted(set(sdk) | set(site) | {row["code_name"] for row in api_rows})
    by_name: dict[str, set[str]] = defaultdict(set)
    for row in api_rows:
        by_name[row["code_name"]].add(row["function"])
    codes: list[dict[str, Any]] = []
    for name in all_names:
        site_value = site.get(name)
        sdk_value = sdk.get(name)
        conflict = site_value is not None and sdk_value is not None and site_value != sdk_value
        value = None if conflict else (sdk_value if sdk_value is not None else site_value)
        codes.append({
            "name": name,
            "value": value,
            "sdk_value": sdk_value,
            "site_value": site_value,
            "api_count": len(by_name[name]),
            "global_descriptions": site_descriptions.get(name, {}),
            "site_url": "https://www.woody.vip/fanuc/errcode.htm" if site_value is not None else None,
        })
    aliases: dict[int, list[str]] = defaultdict(list)
    for row in codes:
        if row["value"] is not None:
            aliases[row["value"]].append(row["name"])
    for row in codes:
        row["same_value_names"] = sorted(aliases[row["value"]]) if row["value"] is not None else []
    unresolved = sorted({row["code_name"] for row in api_rows if row["code_value"] is None})
    catalog = {
        "source_index": "rag_indexes/focas/chunks.jsonl",
        "global_reference_url": "https://www.woody.vip/fanuc/errcode.htm",
        "sdk_headers": [str(path) for path in header_paths],
        "api_error_section_count": sum(chunk.get("chunk_type") == "error_codes" for chunk in chunks),
        "api_return_code_row_count": len(api_rows),
        "api_with_error_section_count": len({row["function"] for row in api_rows}),
        "unresolved_api_code_names": unresolved,
        "sdk_cross_series_conflicts": sdk_conflicts,
        "site_conflicts": site_conflicts,
        "codes": sorted(codes, key=lambda row: (row["value"] is None, row["value"] or 0, row["name"])),
    }
    rag_chunks = [{
        "chunk_id": f"focas-{row['function']}-{row['code_name']}-numeric",
        "protocol": "focas",
        "source_type": "focas_return_code_catalog",
        "source_url": row["source_url"],
        "source_path": row["source_path"],
        "function": row["function"],
        "chunk_type": "return_code_numeric",
        "code_name": row["code_name"],
        "code_value": row["code_value"],
        "text": (
            f"Function: {row['function']}\n"
            f"Return code: {row['code_name']} = {row['code_value'] if row['code_value'] is not None else 'unresolved'}\n"
            f"API-specific explanations: {' | '.join(row['api_explanations'])}\n"
            f"Value source: {row['value_source']}"
        ),
    } for row in api_rows]
    return catalog, api_rows, rag_chunks
