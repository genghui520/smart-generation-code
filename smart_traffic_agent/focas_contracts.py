"""Named FOCAS argument and return-code evidence for code generation.

Fwlib32.h is authoritative for ABI types, but many of its declarations omit
parameter names. The indexed API documentation supplies those names and their
meaning; both pieces of evidence are needed before generating a call.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re


_CHUNKS = Path(__file__).resolve().parents[1] / "rag_indexes" / "focas" / "chunks.jsonl"
_RETURN_CODES = _CHUNKS.with_name("api_return_codes.jsonl")
_GLOBAL_RETURN_CODES = _CHUNKS.with_name("return_codes.json")


@lru_cache(maxsize=1)
def _api_docs() -> dict[str, dict[str, object]]:
    docs: dict[str, dict[str, object]] = {}
    if not _CHUNKS.is_file():
        return docs
    with _CHUNKS.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(row.get("function", ""))
            kind = str(row.get("chunk_type", ""))
            if name.startswith(("cnc_", "pmc_")) and kind in {"prototype", "arguments", "error_codes", "purpose", "overview", "options"}:
                docs.setdefault(name, {})[kind] = str(row.get("text", ""))
    if _RETURN_CODES.is_file():
        with _RETURN_CODES.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = str(row.get("function", ""))
                if name.startswith(("cnc_", "pmc_")):
                    docs.setdefault(name, {}).setdefault("numeric_return_codes", []).append(row)
    return docs


def named_prototype(function_name: str) -> str:
    text = str(_api_docs().get(function_name, {}).get("prototype", ""))
    match = re.search(
        rf"FWLIBAPI\s+[^;]*?\b{re.escape(function_name)}\s*\([^;]*?\)\s*;",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return " ".join(match.group(0).split()) if match else ""


def documented_parameter_names(function_name: str) -> list[str]:
    prototype = named_prototype(function_name)
    if not prototype:
        return []
    raw_args = prototype.partition("(")[2].rsplit(")", 1)[0]
    names: list[str] = []
    for raw in raw_args.split(","):
        match = re.search(r"\b([A-Za-z_]\w*)\s*(?:\[[^]]*\])?\s*$", raw.strip())
        if not match:
            return []
        names.append(match.group(1))
    return names


def undocumented_argument_names(function_name: str, planner_arguments: dict[str, object]) -> list[str]:
    """Planner keys absent from a named API declaration (excluding metadata)."""
    formal_names = documented_parameter_names(function_name)
    if not formal_names:
        return []
    return sorted(set(planner_arguments) - set(formal_names) - {"parameter_generation"})


def named_contract_context(function_names: list[str], *, max_chars: int = 12000, full: bool = False) -> str:
    """Compact, exact named declarations and relevant argument/error semantics."""
    sections: list[str] = []
    for name in dict.fromkeys(function_names):
        docs = _api_docs().get(name, {})
        prototype = named_prototype(name)
        if not prototype:
            continue
        lines = [prototype]
        arguments = str(docs.get("arguments", ""))
        if full:
            lines.extend(str(docs.get(kind, "")) for kind in ("purpose", "overview", "arguments", "error_codes", "options"))
        # Keep the leading explanation for each formal argument, not long
        # controller-specific value tables that can crowd out other APIs.
        for match in re.finditer(r"(?m)^-\s+([A-Za-z_]\w*)\s+\((?:in|out|in/out)\):\s*", "" if full else arguments):
            end = arguments.find("\n- ", match.end())
            detail = arguments[match.start():end if end >= 0 else len(arguments)]
            first_paragraph = detail.split("\n\n", 1)[0]
            lines.append(" ".join(first_paragraph.split())[:260])
        numeric_rows = docs.get("numeric_return_codes", [])
        if isinstance(numeric_rows, list) and numeric_rows:
            for row in numeric_rows:
                value = row.get("code_value")
                explanation = " | ".join(row.get("api_explanations", []))
                lines.append(f"- {row['code_name']} ({value if value is not None else 'unresolved'}): {explanation}"[:340])
        else:
            errors = str(docs.get("error_codes", ""))
            for line in errors.splitlines():
                if line.startswith("- EW_"):
                    lines.append(line[:220])
        section = "\n".join(lines)
        if sum(map(len, sections)) + len(section) > max_chars:
            break
        sections.append(section)
    return "\n\n".join(sections)


@lru_cache(maxsize=1)
def _global_code_rows() -> list[dict[str, object]]:
    if not _GLOBAL_RETURN_CODES.is_file():
        return []
    try:
        catalog = json.loads(_GLOBAL_RETURN_CODES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in catalog.get("codes", []) if isinstance(row, dict)]


def resolve_api_return_code(function_name: str, code: int) -> dict[str, object]:
    """Explain an actual result by API and numeric code; never infer a cause from the number alone."""
    api_rows = _api_docs().get(function_name, {}).get("numeric_return_codes", [])
    matches = [
        row for row in api_rows
        if isinstance(row, dict) and row.get("code_value") == code
    ] if isinstance(api_rows, list) else []
    global_rows = [row for row in _global_code_rows() if row.get("value") == code]
    explanations = list(dict.fromkeys(
        str(explanation).strip()
        for row in matches
        for explanation in row.get("api_explanations", [])
        if str(explanation).strip()
    ))
    return {
        "function": function_name,
        "code_value": code,
        "api_code_names": [str(row["code_name"]) for row in matches],
        "api_explanations": explanations,
        "global_code_names": [str(row["name"]) for row in global_rows],
        "global_meanings": list(dict.fromkeys(
            str(section.get("meaning", "")).strip()
            for row in global_rows
            for section in row.get("global_descriptions", {}).values()
            if isinstance(section, dict) and str(section.get("meaning", "")).strip()
        )),
        "documented_for_api": bool(matches),
        "known_globally": bool(global_rows),
    }


def header_error_codes(header: str) -> dict[int, set[str]]:
    """Return-code labels from the configured SDK, not model-written tables."""
    result: dict[int, set[str]] = {}
    for name, value in re.findall(
        r"(?m)^\s*#\s*define\s+(EW_[A-Z0-9_]+)\s+\(?\s*(-?\d+)\s*\)?\b",
        header,
    ):
        result.setdefault(int(value), set()).add(name)
    return result


def validate_return_code_labels(source: str, header: str) -> list[str]:
    """Flag model-written numeric EW_* switch mappings that contradict SDK macros."""
    expected = header_error_codes(header)
    diagnostics: list[str] = []
    for function_name in ("RetText", "ReturnText", "GetReturnCodeMeaning", "ReturnCodeToString"):
        for match in re.finditer(rf"\b{function_name}\s*\([^;{{}}]*\)\s*\{{", source):
            start = match.end() - 1
            depth = 0
            end = start
            for end in range(start, len(source)):
                if source[end] == "{":
                    depth += 1
                elif source[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            body = source[start:end + 1]
            for number, reported in re.findall(
                r'\bcase\s+(-?\d+)\s*:\s*return\s+"(EW_[A-Z0-9_]+)', body
            ):
                actual = expected.get(int(number))
                if actual is not None and reported not in actual:
                    diagnostics.append(
                        f"BLOCKING: {function_name} maps return code {number} to {reported}; "
                        f"configured Fwlib32.h defines {', '.join(sorted(actual))}."
                    )
    return list(dict.fromkeys(diagnostics))
