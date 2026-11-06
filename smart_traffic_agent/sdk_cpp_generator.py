"""SDK-backed C++ generation helpers.

Production CodeGenerationAgent responses contain only a bare SDK call and
parameter expressions.  This module validates that small expression and owns
the surrounding declarations, input/output logging, and ``ctx.call`` wrapper.
The older recipe facade remains available for offline compatibility callers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from types import SimpleNamespace
from pathlib import Path
from typing import Any

from .api_log_templates import ApiMetadata, ParameterDefinition, api_metadata
from .deterministic_cpp_assembler import DeterministicAssemblyError, assemble_core_cpp


class SdkCallAssemblyError(ValueError):
    """Raised when a model-authored bare call cannot be safely assembled."""


@dataclass(frozen=True)
class BareApiCall:
    step_id: str
    api_name: str
    call_expr: str
    parameter_values: dict[str, Any]


def _split_cpp_arguments(source: str, opening: int) -> tuple[list[str], int] | None:
    """Split a C++ call while preserving nested lambdas/initializer lists."""
    args: list[str] = []
    start = opening + 1
    paren = brace = bracket = 0
    state = "code"
    index = start
    while index < len(source):
        char = source[index]
        nxt = source[index + 1] if index + 1 < len(source) else ""
        if state == "line":
            if char in "\r\n":
                state = "code"
            index += 1
            continue
        if state == "block":
            if char == "*" and nxt == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state in {"string", "char"}:
            if char == "\\":
                index += 2
                continue
            if (state == "string" and char == '"') or (state == "char" and char == "'"):
                state = "code"
            index += 1
            continue
        if char == "/" and nxt == "/":
            state = "line"
            index += 2
            continue
        if char == "/" and nxt == "*":
            state = "block"
            index += 2
            continue
        if char == '"':
            state = "string"
        elif char == "'":
            state = "char"
        elif char == "(":
            paren += 1
        elif char == ")":
            if paren == 0 and brace == 0 and bracket == 0:
                args.append(source[start:index].strip())
                return args, index
            paren = max(paren - 1, 0)
        elif char == "{":
            brace += 1
        elif char == "}":
            brace = max(brace - 1, 0)
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket = max(bracket - 1, 0)
        elif char == "," and paren == 0 and brace == 0 and bracket == 0:
            args.append(source[start:index].strip())
            start = index + 1
        index += 1
    return None


def parse_bare_api_call(call_expr: str, expected_api: str) -> list[str]:
    """Return ordered arguments from exactly one ``api.<expected_api>`` call."""
    source = str(call_expr or "").strip()
    if not source:
        raise SdkCallAssemblyError(f"SDK_CALL_ASSEMBLY_UNRESOLVED: {expected_api}; call_expr is empty")
    if "```" in source or ";" in source or "ctx.call" in source:
        raise SdkCallAssemblyError(
            f"SDK_CALL_EXPRESSION_INVALID: {expected_api}; only one bare api.<function>(...) expression is allowed"
        )
    match = re.match(r"api\s*\.\s*((?:cnc|pmc)_\w+)\s*\(", source)
    if not match:
        raise SdkCallAssemblyError(
            f"SDK_CALL_EXPRESSION_INVALID: {expected_api}; expected api.{expected_api}(...), got {source[:180]!r}"
        )
    actual = match.group(1)
    if actual != expected_api:
        raise SdkCallAssemblyError(
            f"SDK_CALL_API_MISMATCH: expected={expected_api}; actual={actual}"
        )
    parsed = _split_cpp_arguments(source, match.end() - 1)
    if parsed is None:
        raise SdkCallAssemblyError(f"SDK_CALL_EXPRESSION_INVALID: {expected_api}; unclosed argument list")
    args, close = parsed
    if source[close + 1:].strip():
        raise SdkCallAssemblyError(f"SDK_CALL_EXPRESSION_INVALID: {expected_api}; trailing C++ is forbidden")
    if any(re.search(pattern, source, re.IGNORECASE) for pattern in (r"\bmain\s*\(", r"\b(?:struct|class|typedef)\b", r"\breturn\b")):
        raise SdkCallAssemblyError(f"SDK_CALL_EXPRESSION_INVALID: {expected_api}; declaration/wrapper syntax is forbidden")
    return args


def _placeholder(formal: str) -> str:
    return f"__out_{formal}"


def _symbol(step_id: str, formal: str) -> str:
    safe_step = re.sub(r"\W+", "_", str(step_id)).strip("_") or "step"
    safe_formal = re.sub(r"\W+", "_", formal).strip("_") or "out"
    return f"{safe_step}_{safe_formal}"


def output_symbol_name(step_id: str, formal: str) -> str:
    """Return the stable local name used for an API output object."""
    return _symbol(step_id, formal)


def _leaf_fields(fields: dict[str, Any], prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    for name, details in fields.items():
        details = details if isinstance(details, dict) else {}
        path = f"{prefix}.{name}" if prefix else name
        nested = details.get("fields")
        if isinstance(nested, dict) and nested:
            result.extend(_leaf_fields(nested, path))
        else:
            result.append((path, details))
    return result


def _cpp_value_text(expression: str, type_name: str) -> str:
    """Render a scalar expression for the deterministic input parameter log."""
    clean = str(type_name or "").lower()
    if "char *" in clean or "char*" in clean or "wchar" in clean:
        return f"(({expression}) ? std::string({expression}) : std::string(\"NULL\"))"
    if "bool" in clean or any(token in clean for token in ("short", "long", "int", "float", "double")):
        return f"std::to_string({expression})"
    return f"std::string(\"<\") + std::string(\"{expression}\") + std::string(\">\")"


def _render_parameter_text(metadata: ApiMetadata, args: list[str]) -> str:
    pieces: list[str] = []
    for parameter, expression in zip(metadata.parameters, args):
        if parameter.direction not in {"in", "inout"}:
            continue
        pieces.append(
            f"std::string(\"{parameter.name}=\") + {_cpp_value_text(expression, parameter.type)}"
        )
    if not pieces:
        return 'std::string("")'
    result = pieces[0]
    for piece in pieces[1:]:
        result += f" + std::string(\";\") + {piece}"
    return result


def _normalize_expression_for_match(value: Any) -> str:
    """Normalize simple model parameter values for order diagnostics."""
    text = re.sub(r"\s+", "", str(value))
    # Numeric values are commonly emitted as static_cast<short>(N) while the
    # structured parameter contract stores N.
    text = re.sub(r"(?:static_cast|const_cast)\s*<[^>]+>\(([^()]*)\)", r"\1", text)
    return text


def _validate_parameter_value_alignment(
    metadata: ApiMetadata,
    args: list[str],
    parameter_values: dict[str, Any],
) -> None:
    """Catch swapped adjacent arguments when the model supplied named values."""
    if not parameter_values:
        return
    normalized_args = [_normalize_expression_for_match(item) for item in args]
    for index, parameter in enumerate(metadata.parameters):
        if parameter.name not in parameter_values or parameter.direction not in {"in", "inout"}:
            continue
        expected = _normalize_expression_for_match(parameter_values[parameter.name])
        if not expected or expected == normalized_args[index]:
            continue
        if expected in normalized_args:
            actual_index = normalized_args.index(expected)
            raise SdkCallAssemblyError(
                f"SDK_CALL_PARAMETER_ORDER_MISMATCH: {metadata.api_name}; "
                f"parameter={parameter.name}; expected_position={index + 1}; actual_position={actual_index + 1}"
            )


def _render_output_lambda(metadata: ApiMetadata, output_symbols: dict[str, str]) -> str:
    lines = ["[&](short) -> std::string {", "    std::string data;", "    bool first = true;"]
    leaves: list[tuple[str, str, dict[str, Any]]] = []
    for parameter in metadata.output_parameters:
        symbol = output_symbols.get(parameter.name)
        if not symbol:
            continue
        for path, details in _leaf_fields(parameter.fields):
            leaves.append((symbol, path, details))
    if not leaves:
        raise SdkCallAssemblyError(
            f"SDK_CALL_ASSEMBLY_UNRESOLVED: {metadata.api_name}; no deterministic output field rule"
        )
    for symbol, path, details in leaves:
        expression = f"{symbol}.{path}"
        dimensions = details.get("array_dimensions") if isinstance(details, dict) else None
        if dimensions:
            bound = str(dimensions[0])
            lines.extend([
                f"    for (int i = 0; i < {bound}; ++i) {{",
                "        if (!first) data += \";\"; first = false;",
                f"        data += \"{path}[\" + std::to_string(i) + \"]=\" + std::to_string({symbol}.{path}[i]);",
                "    }",
            ])
        else:
            lines.extend([
                "    if (!first) data += \";\"; first = false;",
                f"    data += \"{path}=\" + std::to_string({expression});",
            ])
    lines.extend(["    return data;", "}"])
    return "\n".join(lines)


def assemble_capture_call(
    *,
    step_id: str,
    api_name: str,
    call_expr: str,
    parameter_values: dict[str, Any] | None = None,
    available_symbols: dict[str, str] | None = None,
    symbol_step_id: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Build one fixed ``ctx.call`` from a model-authored bare SDK call."""
    metadata = api_metadata(api_name)
    if not metadata.complete:
        raise SdkCallAssemblyError(
            f"SDK_CALL_ASSEMBLY_UNRESOLVED: {api_name}; " + "; ".join(metadata.diagnostics)
        )
    args = parse_bare_api_call(call_expr, api_name)
    if len(args) != len(metadata.parameters):
        raise SdkCallAssemblyError(
            f"SDK_CALL_ARGUMENT_COUNT_MISMATCH: {api_name}; expected={len(metadata.parameters)}; actual={len(args)}"
        )
    supplied_values = parameter_values or {}
    formal_names = {parameter.name for parameter in metadata.parameters}
    unknown_values = sorted(set(str(key) for key in supplied_values) - formal_names)
    if unknown_values:
        raise SdkCallAssemblyError(
            f"SDK_CALL_PARAMETER_NAME_UNKNOWN: {api_name}; names={unknown_values}"
        )
    _validate_parameter_value_alignment(metadata, args, supplied_values)
    available = {"handle", "std", "MAX_AXIS", "EW_OK", "EW_FUNC", "nullptr", "nc_payload", "nc_buffer", "nc_length"}
    available.update(str(key) for key in (available_symbols or {}))
    available.update(str(value) for value in (available_symbols or {}).values())
    for symbol in list(available):
        # A qualified runtime binding such as ``runtime.axis`` or
        # ``S001.statinfo.run`` is valid when its base is supplied by the
        # execution context.  Keep the base names available for lexical
        # validation while member names are handled below.
        available.add(str(symbol).split(".", 1)[0].split("->", 1)[0])
    output_symbols: dict[str, str] = {}
    output_name_step = symbol_step_id or step_id
    rendered_args = list(args)
    declarations: list[str] = []
    for index, parameter in enumerate(metadata.parameters):
        expression = rendered_args[index]
        if parameter.direction in {"out", "inout"} and "*" in parameter.type:
            marker = _placeholder(parameter.name)
            if marker not in expression:
                raise SdkCallAssemblyError(
                    f"SDK_CALL_OUTPUT_BINDING_MISSING: {api_name}.{parameter.name}; use &{marker} in call_expr"
                )
            symbol = _symbol(output_name_step, parameter.name)
            output_symbols[parameter.name] = symbol
            rendered_args[index] = expression.replace(marker, symbol)
            available.add(symbol)
            pointed = (parameter.pointed_type or "").strip()
            if not pointed or pointed.lower() in {"char", "wchar_t"}:
                raise SdkCallAssemblyError(
                    f"SDK_CALL_ASSEMBLY_UNRESOLVED: {api_name}.{parameter.name}; output buffer rule is missing"
                )
            declarations.append(f"{pointed} {symbol}{{}};")
        else:
            rendered_args[index] = expression
        # Runtime variable names are supplied by the local symbol table. A
        # dotted member expression is checked by its base identifier.
        identifiers = re.findall(r"\b[A-Za-z_]\w*\b", rendered_args[index])
        # Member names after '.'/'->' are fields supplied by the SDK type
        # definition, not free runtime symbols.  Only the base expression
        # must be present in the local symbol table.
        member_names = set(re.findall(r"(?:\.|->)\s*([A-Za-z_]\w*)", rendered_args[index]))
        keywords = {"static_cast", "const_cast", "reinterpret_cast", "true", "false"}
        unknown = [name for name in identifiers if name not in member_names and name not in available and name not in keywords and name not in {
            "short", "long", "int", "unsigned", "char", "sizeof", "std", "to_string"
        } and not name.startswith("api")]
        if unknown:
            raise SdkCallAssemblyError(
                f"SDK_RUNTIME_SYMBOL_UNRESOLVED: {api_name}; symbols={sorted(set(unknown))}"
            )
    parameter_text = _render_parameter_text(metadata, rendered_args)
    output_lambda = _render_output_lambda(metadata, output_symbols)
    call = (
        "\n".join(declarations)
        + ("\n" if declarations else "")
        + f"ret = ctx.call(\"{step_id}\", \"{api_name}\", {parameter_text}, api.{api_name}, "
        + f"[&]() -> short {{ return api.{api_name}({', '.join(rendered_args)}); }},\n"
        + output_lambda
        + ");\n"
    )
    return call, output_symbols


def _plan_steps(raw_steps: Any) -> list[SimpleNamespace]:
    if not isinstance(raw_steps, list):
        return []
    result: list[SimpleNamespace] = []
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}] must be an object")
        protocol = str(raw.get("protocol_function") or raw.get("api") or "").strip()
        if not protocol:
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}] lacks protocol_function")
        params = raw.get("parameters", {})
        if not isinstance(params, dict):
            raise DeterministicAssemblyError(f"SDK_CPP_PLAN_INVALID: steps[{index - 1}].parameters must be an object")
        result.append(SimpleNamespace(protocol_function=protocol, parameters=params))
    return result


def generate_core_cpp_from_plan(plan: dict[str, Any], *, sdk_header: str | Path | None = None) -> str:
    """Generate statements-only C++ from a JSON-compatible plan object."""
    if not isinstance(plan, dict):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: root must be an object")
    targets = plan.get("targets") or plan.get("target_apis")
    if not isinstance(targets, list) or not targets:
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: targets must be a non-empty array")
    planned = plan.get("planned_functions") or targets
    if not isinstance(planned, list):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: planned_functions must be an array")
    contract = plan.get("parameter_contract", {"functions": {}})
    if not isinstance(contract, dict):
        raise DeterministicAssemblyError("SDK_CPP_PLAN_INVALID: parameter_contract must be an object")
    return assemble_core_cpp(
        targets=[str(item) for item in targets],
        planned_functions=[str(item) for item in planned],
        steps=_plan_steps(plan.get("steps", [])),
        parameter_contract=contract,
        sdk_header=sdk_header,
    )


__all__ = [
    "BareApiCall",
    "SdkCallAssemblyError",
    "assemble_capture_call",
    "generate_core_cpp_from_plan",
    "output_symbol_name",
    "parse_bare_api_call",
]
