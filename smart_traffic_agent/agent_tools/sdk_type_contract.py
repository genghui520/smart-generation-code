"""Deterministic SDK type contract lookup for code generation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..integrations.ncguide import default_focas_header_dir
from ..sdk_type_index import load_or_build_sdk_type_index, sdk_index_prompt_context
from .base import AgentTool


@dataclass(frozen=True, slots=True)
class SdkTypeContractInput:
    api_name: str
    parameter: str = ""


def _split_arguments(prototype: str) -> list[str]:
    body = prototype[prototype.find("(") + 1:prototype.rfind(")")]
    return [part.strip() for part in body.split(",") if part.strip()]


def _argument_names(api_name: str) -> tuple[str, ...]:
    """Read documented names when the C prototype uses anonymous arguments."""
    path = Path(__file__).resolve().parents[1] / ".." / "rag_indexes" / "focas" / "vector_db_pdf_rebuilt" / "stable_records.jsonl"
    try:
        # This file is several megabytes. Stream it and stop at the first
        # matching prototype instead of materializing and decoding every row.
        with path.resolve().open("r", encoding="utf-8") as handle:
            for line in handle:
                # Avoid JSON decoding the entire vector export. Only rows
                # carrying this API name can contain its documented argument
                # names; the rest are irrelevant to this contract lookup.
                if api_name not in line:
                    continue
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
                if metadata.get("function") != api_name or metadata.get("chunk_type") != "prototype":
                    continue
                text = str(row.get("text", ""))
                # FOCAS prototype records normally include parameter names.
                match = re.search(r"\((.*)\)", text, re.DOTALL)
                if not match:
                    return ()
                names: list[str] = []
                for item in _split_arguments(match.group(0)):
                    token = re.search(r"([A-Za-z_]\w*)\s*(?:\[\])?\s*$", item)
                    names.append(token.group(1) if token else "")
                return tuple(names)
    except OSError:
        return ()
    return ()


def _declared_sdk_types(arguments: list[str], index) -> list[str]:
    """Extract SDK-defined struct/union names directly from formal arguments.

    The old lookup only added a type for a small void-pointer allowlist. Most
    FOCAS APIs declare their payload types directly, such as ``IDBWRA *`` or
    ``ODBAXISNAME *``; those were incorrectly reported with no candidates.
    """
    known = set(index.structs) | set(index.unions)
    found: list[str] = []
    for declared in arguments:
        for token in re.findall(r"\b[A-Za-z_]\w*\b", declared):
            if token in known and token not in found:
                found.append(token)
    return found


def _related_sdk_types(initial: list[str], index) -> list[str]:
    """Return the complete SDK type closure reachable from prototype types."""
    known = set(index.structs) | set(index.unions)
    result: list[str] = []
    pending = list(initial)
    while pending:
        name = pending.pop(0)
        if name in result or name not in known:
            continue
        result.append(name)
        definition = index.structs.get(name) or index.unions.get(name) or {}
        for field in (definition.get("fields", {}) or {}).values():
            field_type = str(field.get("type", ""))
            for token in re.findall(r"\b[A-Za-z_]\w*\b", field_type):
                token = token.upper()
                if token in known and token not in result:
                    pending.append(token)
    return result


class GetSdkTypeContractTool(AgentTool[SdkTypeContractInput, dict[str, Any]]):
    name = "get_sdk_type_contract"
    description = "Read the active Fwlib32.h prototype and existing SDK struct/union definitions for one API parameter."

    def __init__(self, header_path: Path | None = None, index_path: Path | None = None) -> None:
        self.header_path = header_path or (default_focas_header_dir() / "Fwlib32.h")
        self.index_path = index_path or (Path("rag_indexes/focas") / "sdk_type_index_0iD.json")

    def invoke(self, tool_input: SdkTypeContractInput) -> dict[str, Any]:
        api_name = str(tool_input.api_name or "").strip()
        parameter = str(tool_input.parameter or "").strip()
        if not api_name:
            return {"ok": False, "error": "api_name is required"}
        if not self.header_path.is_file():
            return {"ok": False, "error": f"active SDK header not found: {self.header_path}"}
        index = load_or_build_sdk_type_index(self.header_path, self.index_path)
        prototype = index.functions.get(api_name, "")
        if not prototype:
            return {"ok": False, "error": f"SDK prototype not found for {api_name}", "api_name": api_name}
        args = _split_arguments(prototype)
        names = _argument_names(api_name)
        entries = []
        for position, declared in enumerate(args):
            name = names[position] if position < len(names) else ""
            entries.append({"position": position, "name": name, "declared_type": declared})
        selected = [item for item in entries if not parameter or item["name"] == parameter]
        if parameter and not selected:
            return {"ok": False, "error": f"parameter not found in SDK contract: {api_name}.{parameter}", "prototype": prototype}

        # Resolve every SDK-defined payload type present in the formal
        # prototype, including direct pointers such as IDBWRA * and
        # ODBAXISNAME *. Keep the void-pointer compatibility mappings for
        # legacy declarations whose concrete type is not in the prototype.
        candidate_types: list[str] = _declared_sdk_types(args, index)
        if any("void" in str(item["declared_type"]).lower() for item in selected):
            if api_name in {"cnc_rdparam", "cnc_rdparar", "cnc_wrparam", "cnc_wrparas"}:
                if "IODBPSD" not in candidate_types:
                    candidate_types.append("IODBPSD")
        candidate_types = _related_sdk_types(candidate_types, index)
        type_definitions = {
            name: index.structs.get(name) or index.unions.get(name)
            for name in candidate_types
            if index.structs.get(name) or index.unions.get(name)
        }
        return {
            "ok": True,
            "api_name": api_name,
            "parameter": parameter,
            "header_path": index.header_path,
            "header_sha256": index.header_sha256,
            "prototype": prototype,
            "arguments": entries,
            "selected_arguments": selected,
            "candidate_sdk_types": candidate_types,
            "type_definitions": type_definitions,
            "requires_runtime_shape": bool(candidate_types),
            "evidence": [f"{index.header_path}:prototype:{api_name}", f"{index.header_path}:types"],
            "sdk_context": sdk_index_prompt_context(index, set(candidate_types)) if candidate_types else "",
        }

    def input_summary(self, tool_input: SdkTypeContractInput) -> dict[str, Any]:
        return {"api_name": tool_input.api_name, "parameter": tool_input.parameter}

    def output_summary(self, tool_output: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": bool(tool_output.get("ok")),
            "api_name": tool_output.get("api_name", ""),
            "parameter": tool_output.get("parameter", ""),
            "candidate_sdk_types": tool_output.get("candidate_sdk_types", []),
        }
