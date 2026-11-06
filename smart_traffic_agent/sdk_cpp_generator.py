"""SDK-backed C++ generation helpers.

Production CodeGenerationAgent responses contain only a bare SDK call and
parameter expressions.  ``parameter_values`` contains only formal SDK
arguments; ``struct_values`` contains verified fields for SDK struct/union
pointers.  This module validates those values and owns the surrounding
declarations, input/output logging, and ``ctx.call`` wrapper.
The older recipe facade remains available for offline compatibility callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    struct_values: dict[str, Any] = field(default_factory=dict)


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


def _render_parameter_text(
    metadata: ApiMetadata,
    args: list[str],
    struct_logs: list[tuple[str, str, str]] | None = None,
) -> str:
    pieces: list[str] = []
    for parameter, expression in zip(metadata.parameters, args):
        if parameter.direction not in {"in", "inout"}:
            continue
        pieces.append(
            f"std::string(\"{parameter.name}=\") + {_cpp_value_text(expression, parameter.type)}"
        )
    for field_name, expression, field_type in struct_logs or []:
        pieces.append(
            f"std::string(\"{field_name}=\") + {_cpp_value_text(expression, field_type)}"
        )
    if not pieces:
        return 'std::string("")'
    result = pieces[0]
    for piece in pieces[1:]:
        result += f" + std::string(\";\") + {piece}"
    return result


def _struct_pointer_parameters(metadata: ApiMetadata) -> dict[str, ParameterDefinition]:
    """Return SDK struct/union pointer parameters that can carry field data."""
    return {
        parameter.name: parameter
        for parameter in metadata.parameters
        if "*" in parameter.type and parameter.pointed_type and parameter.fields
    }


# These are the SDK scalar types for which ``std::to_string`` is a
# deterministic and lossless-enough display representation.  Character and
# void pointers are deliberately excluded: in FOCAS they normally denote a
# caller-provided buffer whose size and termination rules must come from the
# API contract.
_NUMERIC_SCALAR_TYPES = frozenset({
    "bool",
    "char16_t", "char32_t",
    "signed char", "unsigned char",
    "short", "signed short", "short int", "signed short int",
    "unsigned short", "unsigned short int",
    "int", "signed int", "unsigned int",
    "long", "signed long", "long int", "signed long int",
    "unsigned long", "unsigned long int",
    "long long", "signed long long", "long long int", "signed long long int",
    "unsigned long long", "unsigned long long int",
    "float", "double", "long double",
    "int8_t", "uint8_t", "int16_t", "uint16_t",
    "int32_t", "uint32_t", "int64_t", "uint64_t",
    "intptr_t", "uintptr_t", "size_t", "ssize_t", "ptrdiff_t",
    "__int8", "unsigned __int8", "__int16", "unsigned __int16",
    "__int32", "unsigned __int32", "__int64", "unsigned __int64",
    # Common Windows/FOCAS typedef spellings used for integer scalars.
    "bool32", "byte", "word", "dword", "uint", "uint8", "uint16", "uint32", "uint64",
    "int8", "int16", "int32", "int64", "uchar", "ushort", "ulong", "ullong",
})


def _normalized_pointed_type(parameter: ParameterDefinition) -> str:
    pointed = str(parameter.pointed_type or "")
    pointed = re.sub(r"\b(?:const|volatile)\b", "", pointed, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", pointed).strip().lower()


def _output_serialization_kind(parameter: ParameterDefinition) -> str:
    """Classify one output pointer for deterministic serialization.

    Numeric pointers without an explicit array dimension are scalar outputs.
    FOCAS uses ``short*``/``long*`` for both scalar values and count/length
    values whose relationship to another argument is documented at runtime;
    the pointer type itself is the deterministic ABI evidence available to the
    assembler.  Fixed arrays remain buffers and require a dedicated serializer.
    """
    if parameter.fields:
        return "struct"
    if parameter.array_dimensions:
        return "buffer"
    pointed = _normalized_pointed_type(parameter)
    if pointed in _NUMERIC_SCALAR_TYPES:
        return "scalar"
    return "buffer"


def _buffer_output_diagnostic(api_name: str, parameter: ParameterDefinition) -> str:
    if parameter.array_dimensions or parameter.valid_length_rule:
        return (
            f"SDK_OUTPUT_SERIALIZATION_UNRESOLVED: {api_name}.{parameter.name}; "
            "buffer output has a documented size rule but no deterministic buffer serializer"
        )
    return (
        f"SDK_OUTPUT_SERIALIZATION_UNRESOLVED: {api_name}.{parameter.name}; "
        "buffer output requires a documented array or length rule"
    )


def _field_value_schema(parameter: ParameterDefinition, field_path: str) -> dict[str, Any] | None:
    current: dict[str, Any] = parameter.fields
    parts = [part for part in str(field_path).split(".") if part]
    for index, part in enumerate(parts):
        match = re.fullmatch(r"([A-Za-z_]\w*)(?:\[(\d+)\])?", part)
        if not match:
            return None
        name = match.group(1)
        details = current.get(name)
        if not isinstance(details, dict):
            return None
        if match.group(2) is not None and not details.get("array_dimensions"):
            return None
        if index == len(parts) - 1:
            return details
        nested = details.get("fields")
        if not nested:
            return None
        current = nested
    return None


def _merge_nested_value(root: dict[str, Any], path: str, value: Any) -> None:
    """Merge a dotted legacy field path into a nested struct_values object."""
    parts = re.findall(r"[A-Za-z_]\w*|\[\d+\]", str(path))
    current: Any = root
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part.startswith("["):
            if not isinstance(current, list):
                raise SdkCallAssemblyError(
                    f"SDK_STRUCT_VALUE_INVALID: cannot index non-array field at {path}"
                )
            position = int(part[1:-1])
            while len(current) <= position:
                current.append({})
            if last:
                current[position] = value
            else:
                current = current[position]
            continue
        if last:
            if isinstance(current, dict) and part in current and isinstance(current[part], dict) and isinstance(value, dict):
                current[part].update(value)
            else:
                current[part] = value
            continue
        next_part = parts[index + 1]
        if part not in current:
            current[part] = [] if next_part.startswith("[") else {}
        current = current[part]


def _normalize_struct_values(
    metadata: ApiMetadata,
    parameter_values: dict[str, Any],
    struct_values: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate formal parameters from nested SDK struct field values.

    Older model responses flattened fields into ``parameter_values``.  They
    are migrated only when the root formal parameter and SDK field path are
    both verified by the active metadata.
    """
    normalized_parameters = dict(parameter_values or {})
    normalized_structs: dict[str, Any] = {}
    if struct_values is not None:
        if not isinstance(struct_values, dict):
            raise SdkCallAssemblyError("SDK_STRUCT_VALUE_INVALID: struct_values must be an object")
        normalized_structs.update(struct_values)
    struct_parameters = _struct_pointer_parameters(metadata)
    for key in list(normalized_parameters):
        key_text = str(key)
        root, separator, path = key_text.partition(".")
        if key_text in struct_parameters and isinstance(normalized_parameters[key], dict):
            normalized_structs.setdefault(key_text, {}).update(normalized_parameters.pop(key_text))
            continue
        if not separator or root not in struct_parameters:
            continue
        if _field_value_schema(struct_parameters[root], path) is None:
            raise SdkCallAssemblyError(f"SDK_STRUCT_FIELD_UNKNOWN: {root}.{path}")
        target = normalized_structs.setdefault(root, {})
        if not isinstance(target, dict):
            raise SdkCallAssemblyError(f"SDK_STRUCT_VALUE_INVALID: {root} must be an object")
        _merge_nested_value(target, path, normalized_parameters.pop(key_text))
    unknown_roots = sorted(set(normalized_structs) - set(struct_parameters))
    if unknown_roots:
        raise SdkCallAssemblyError(
            f"SDK_STRUCT_ROOT_UNKNOWN: {metadata.api_name}; roots={unknown_roots}"
        )
    return normalized_parameters, normalized_structs


def _cpp_assignment_value(value: Any, field_type: str) -> str:
    if isinstance(value, bool):
        raw = "true" if value else "false"
    elif isinstance(value, (int, float)):
        raw = repr(value)
    elif isinstance(value, str) and value.strip():
        raw = value.strip()
    else:
        raise SdkCallAssemblyError(
            f"SDK_STRUCT_VALUE_INVALID: scalar value for type {field_type!r} is missing"
        )
    clean_type = re.sub(r"\s+", " ", str(field_type or "").strip())
    if not clean_type:
        raise SdkCallAssemblyError("SDK_STRUCT_VALUE_UNRESOLVED: field type is unavailable")
    if isinstance(value, (int, float)):
        lowered = clean_type.lower()
        bounds: tuple[float, float] | None = None
        if "unsigned char" in lowered:
            bounds = (0, 255)
        elif lowered == "char":
            bounds = (-128, 127)
        elif "unsigned short" in lowered:
            bounds = (0, 65535)
        elif lowered == "short":
            bounds = (-32768, 32767)
        elif "unsigned long" in lowered or lowered in {"unsigned int", "uint32_t"}:
            bounds = (0, 4294967295)
        elif lowered in {"long", "int", "long int", "int32_t"}:
            bounds = (-2147483648, 2147483647)
        if bounds is not None and not (bounds[0] <= value <= bounds[1]):
            raise SdkCallAssemblyError(
                f"SDK_STRUCT_VALUE_TYPE_MISMATCH: value={value} is outside {clean_type} range"
            )
    if isinstance(value, str) and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value.strip()):
        return raw
    return f"static_cast<{clean_type}>({raw})"


def _render_struct_assignments(
    parameter: ParameterDefinition,
    struct_value: Any,
    symbol: str,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Validate a nested field tree and render assignments/log fields."""
    if not isinstance(struct_value, dict):
        raise SdkCallAssemblyError(
            f"SDK_STRUCT_VALUE_INVALID: {parameter.name} must contain an object"
        )
    assignments: list[str] = []
    logs: list[tuple[str, str, str]] = []

    def visit(schema: dict[str, Any], values: dict[str, Any], prefix: str) -> None:
        for field_name, value in values.items():
            if field_name in {"active_member", "active_branch"}:
                continue
            details = schema.get(field_name)
            if not isinstance(details, dict):
                raise SdkCallAssemblyError(
                    f"SDK_STRUCT_FIELD_UNKNOWN: {parameter.name}.{prefix + field_name}"
                )
            path = f"{prefix}{field_name}"
            nested = details.get("fields")
            dimensions = details.get("array_dimensions") or []
            if nested and dimensions:
                if not isinstance(value, (list, tuple)):
                    raise SdkCallAssemblyError(f"SDK_STRUCT_VALUE_INVALID: {parameter.name}.{path} must be an array")
                for index, item in enumerate(value):
                    if str(dimensions[0]).isdigit() and index >= int(dimensions[0]):
                        raise SdkCallAssemblyError(f"SDK_STRUCT_ARRAY_BOUNDS: {parameter.name}.{path}[{index}]")
                    if not isinstance(item, dict):
                        raise SdkCallAssemblyError(f"SDK_STRUCT_VALUE_INVALID: {parameter.name}.{path}[{index}] must be an object")
                    visit(nested, item, f"{path}[{index}].")
                continue
            if nested:
                if not isinstance(value, dict):
                    raise SdkCallAssemblyError(f"SDK_STRUCT_VALUE_INVALID: {parameter.name}.{path} must be an object")
                nested_values = value
                if str(details.get("type", "")).lstrip().lower().startswith("union"):
                    active = value.get("active_member", value.get("active_branch"))
                    branch_values = {
                        key: item for key, item in value.items()
                        if key not in {"active_member", "active_branch"}
                    }
                    if active is not None:
                        active = str(active)
                        if active not in nested:
                            raise SdkCallAssemblyError(
                                f"SDK_UNION_BRANCH_UNKNOWN: {parameter.name}.{path}.{active}"
                            )
                        if "value" in branch_values and active not in branch_values:
                            branch_values = {active: branch_values["value"]}
                    if len(branch_values) != 1:
                        raise SdkCallAssemblyError(
                            f"SDK_UNION_BRANCH_REQUIRED: {parameter.name}.{path}; provide exactly one active member"
                        )
                    if active is not None and next(iter(branch_values)) != active:
                        raise SdkCallAssemblyError(
                            f"SDK_UNION_BRANCH_MISMATCH: {parameter.name}.{path}; active_member={active}"
                        )
                    nested_values = branch_values
                visit(nested, nested_values, path + ".")
                continue
            if dimensions:
                if not isinstance(value, (list, tuple)):
                    raise SdkCallAssemblyError(f"SDK_STRUCT_VALUE_INVALID: {parameter.name}.{path} must be an array")
                for index, item in enumerate(value):
                    if index >= int(dimensions[0]) if str(dimensions[0]).isdigit() else False:
                        raise SdkCallAssemblyError(f"SDK_STRUCT_ARRAY_BOUNDS: {parameter.name}.{path}[{index}]")
                    expression = f"{symbol}.{path}[{index}]"
                    field_type = str(details.get("type") or "")
                    assignments.append(f"{expression} = {_cpp_assignment_value(item, field_type)};")
                    logs.append((f"{path}[{index}]", expression, field_type))
                continue
            expression = f"{symbol}.{path}"
            field_type = str(details.get("type") or "")
            assignments.append(f"{expression} = {_cpp_assignment_value(value, field_type)};")
            logs.append((path, expression, field_type))

    visit(parameter.fields, struct_value, "")
    return assignments, logs


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
    outputs: list[tuple[str, str, str, dict[str, Any] | None]] = []
    for parameter in metadata.output_parameters:
        symbol = output_symbols.get(parameter.name)
        if not symbol:
            continue
        kind = _output_serialization_kind(parameter)
        if kind == "struct":
            for path, details in _leaf_fields(parameter.fields):
                outputs.append(("field", symbol, path, details))
        elif kind == "scalar":
            outputs.append(("scalar", symbol, parameter.name, None))
        else:
            raise SdkCallAssemblyError(_buffer_output_diagnostic(metadata.api_name, parameter))
    if not outputs:
        raise SdkCallAssemblyError(
            f"SDK_CALL_ASSEMBLY_UNRESOLVED: {metadata.api_name}; no deterministic output field rule"
        )
    for kind, symbol, path, details in outputs:
        if kind == "scalar":
            lines.extend([
                "    if (!first) data += \";\"; first = false;",
                f"    data += \"{path}=\" + std::to_string({symbol});",
            ])
            continue
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
    struct_values: dict[str, Any] | None = None,
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
    supplied_values, supplied_structs = _normalize_struct_values(
        metadata, parameter_values or {}, struct_values
    )
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
    struct_parameters = _struct_pointer_parameters(metadata)
    output_symbols: dict[str, str] = {}
    struct_logs: list[tuple[str, str, str]] = []
    output_name_step = symbol_step_id or step_id
    rendered_args = list(args)
    declarations: list[str] = []
    for index, parameter in enumerate(metadata.parameters):
        expression = rendered_args[index]
        is_struct_pointer = parameter.name in struct_parameters
        if is_struct_pointer:
            marker = _placeholder(parameter.name)
            if marker not in expression:
                raise SdkCallAssemblyError(
                    f"SDK_CALL_OUTPUT_BINDING_MISSING: {api_name}.{parameter.name}; use &{marker} in call_expr"
                )
            symbol = _symbol(output_name_step, parameter.name)
            if parameter.direction in {"out", "inout"} or parameter.name in supplied_structs:
                output_symbols[parameter.name] = symbol
            rendered_args[index] = expression.replace(marker, symbol)
            available.add(symbol)
            pointed = (parameter.pointed_type or "").strip()
            if not pointed:
                raise SdkCallAssemblyError(
                    f"SDK_CALL_ASSEMBLY_UNRESOLVED: {api_name}.{parameter.name}; output buffer rule is missing"
                )
            declarations.append(f"{pointed} {symbol}{{}};")
            if parameter.name in supplied_structs:
                assignments, logs = _render_struct_assignments(
                    parameter, supplied_structs[parameter.name], symbol
                )
                declarations.extend(assignments)
                struct_logs.extend(
                    (f"{parameter.name}.{name}", expression_text, field_type)
                    for name, expression_text, field_type in logs
                )
        elif parameter.direction in {"out", "inout"} and "*" in parameter.type:
            marker = _placeholder(parameter.name)
            if marker not in expression:
                raise SdkCallAssemblyError(
                    f"SDK_CALL_OUTPUT_BINDING_MISSING: {api_name}.{parameter.name}; use &{marker} in call_expr"
                )
            if _output_serialization_kind(parameter) == "buffer":
                raise SdkCallAssemblyError(_buffer_output_diagnostic(api_name, parameter))
            symbol = _symbol(output_name_step, parameter.name)
            output_symbols[parameter.name] = symbol
            rendered_args[index] = expression.replace(marker, symbol)
            available.add(symbol)
            pointed = (parameter.pointed_type or "").strip()
            if not pointed:
                raise SdkCallAssemblyError(
                    f"SDK_OUTPUT_SERIALIZATION_UNRESOLVED: {api_name}.{parameter.name}; "
                    "output type is unavailable"
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
    parameter_text = _render_parameter_text(metadata, rendered_args, struct_logs)
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
