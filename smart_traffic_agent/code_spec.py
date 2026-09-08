"""Build the structured code contract from Planner output and project sources."""
from __future__ import annotations

import re
from typing import Any

from .agent_roles import ApiContract, CodeSpec
from .integrations.ncguide import default_focas_header_dir
from .tool_protocol import tool_calls_from_step


def build_code_spec(plan: Any, retrieved_chunks: list[Any] | None = None) -> CodeSpec:
    header_path = default_focas_header_dir() / "Fwlib32.h"
    header = header_path.read_text(encoding="latin-1") if header_path.exists() else ""
    chunks = retrieved_chunks or []
    calls = [call for step in plan.steps for call in tool_calls_from_step(step)]
    contracts: list[ApiContract] = []
    for index, call in enumerate(calls):
        prototype = _prototype_for(header, call.tool_name)
        return_type, parameter_types, output_fields = _parse_prototype(prototype)
        evidence = [
            item.chunk.chunk_id for item in chunks
            if str(item.chunk.metadata.get("function", "")) == call.tool_name
            or call.tool_name in item.chunk.text
        ][:8]
        contracts.append(ApiContract(
            call_id=call.call_id,
            tool_name=call.tool_name,
            phase=call.phase,
            operation_id=call.operation_id,
            arguments=dict(call.arguments),
            prototype=prototype,
            return_type=return_type,
            parameter_types=parameter_types,
            output_fields=output_fields,
            required_calls=[item.tool_name for item in calls[index + 1:index + 4]],
            evidence_chunk_ids=evidence,
        ))
    return CodeSpec(
        scenario=plan.scenario_type,
        api_contracts=contracts,
        nc_program_name=plan.nc_program_spec.program_name,
        source_requirements=[
            "Implement every api_contract from the Planner contract.",
            "Use each contract's official prototype and parameter types.",
            "Check every return code and release resources on every exit path.",
        ],
        official_abi_context=build_official_abi_context(
            [contract.tool_name for contract in contracts], header_path, header
        ),
    )


def build_official_abi_context(
    function_names: list[str], header_path: Any | None = None, header: str | None = None
) -> str:
    header_path = header_path or (default_focas_header_dir() / "Fwlib32.h")
    header = header if header is not None else (
        header_path.read_text(encoding="latin-1") if header_path.exists() else ""
    )
    declarations = []
    seen = set()
    for name in function_names:
        if name in seen:
            continue
        seen.add(name)
        prototype = _prototype_for(header, name)
        if prototype:
            declarations.append(prototype)
    referenced = set()
    for declaration in declarations:
        referenced.update(
            token for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", declaration)
            if token not in {"FWLIBAPI", "WINAPI"}
        )
    typedefs = _extract_typedefs(header)
    structures = [typedefs[name] for name in sorted(referenced) if name in typedefs]
    sections = ["header=" + str(header_path), "[official function declarations]", "\n".join(declarations)]
    if structures:
        sections.extend(["[official referenced type definitions]", "\n\n".join(structures)])
    return "\n".join(sections)


def _extract_typedefs(header: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pattern = re.compile(r"typedef\s+(?:struct|union)\s+\w+\s*\{", re.IGNORECASE)
    for match in pattern.finditer(header):
        open_brace = header.find("{", match.start())
        depth = 0
        close_brace = -1
        for index in range(open_brace, len(header)):
            if header[index] == "{":
                depth += 1
            elif header[index] == "}":
                depth -= 1
                if depth == 0:
                    close_brace = index
                    break
        if close_brace < 0:
            continue
        alias = re.match(r"\s*([A-Za-z_]\w*)\s*;", header[close_brace + 1:])
        if alias:
            # FANUC headers may contain conditional alternate definitions of
            # the same typedef. Keep the first definition, which is the one
            # selected by the default preprocessor configuration used by the
            # configured compiler/header pair.
            blocks.setdefault(
                alias.group(1).upper(),
                header[match.start():close_brace + 1] + " " + alias.group(0).strip(),
            )
    return blocks


def _prototype_for(header: str, function_name: str) -> str:
    if not header:
        return ""
    match = re.search(rf"FWLIBAPI\s+[^;]*?\b{re.escape(function_name)}\s*\([^;]*?\)\s*;", header, re.IGNORECASE | re.DOTALL)
    return " ".join(match.group(0).split()) if match else ""


def _parse_prototype(prototype: str) -> tuple[str, list[str], list[str]]:
    if not prototype:
        return "", [], []
    before, _, args = prototype.partition("(")
    return_type = before.replace("FWLIBAPI", "", 1).replace("WINAPI", "", 1).strip()
    args = args.rsplit(")", 1)[0]
    types: list[str] = []
    outputs: list[str] = []
    for raw in args.split(","):
        item = " ".join(raw.split()).strip()
        if not item or item.lower() == "void":
            continue
        types.append(item)
        if "*" in item or "[" in item:
            outputs.append(item)
    return return_type, types, outputs
