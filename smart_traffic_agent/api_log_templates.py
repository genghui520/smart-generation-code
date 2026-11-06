"""Deterministic API input/output log templates.

This module is deliberately independent of the language model and of the
execution workflow.  It derives the parameter contract from the configured
FOCAS header and the local argument documentation, then renders the stable
display tables from raw runtime rows.  The raw rows remain the source of
truth; the tables are a projection for people and CSV/Excel consumers.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .focas_contracts import _api_docs, named_prototype, resolve_api_return_code
from .integrations.ncguide import default_focas_header_dir
from .sdk_type_index import SdkIndex, load_or_build_sdk_type_index

INPUT_LOG_FIELDS = ["timestamp", "api_name", "params"]
OUTPUT_LOG_FIELDS = ["timestamp", "api_name", "return_code", "return_desc", "success_data"]
_NON_API_PAYLOAD_KEYS = {
    "coverage_role", "pcap_capture_enabled", "pcap_capture_started",
    "pcap_capture_disabled", "pcap_capture_failed", "phase",
    "planner_selected_api", "sample_role", "segment_index",
    "structural_api", "target", "endpoint", "device_context",
    "output_valid", "incomplete_call",
}
# These values identify a live FOCAS session rather than a CNC operation
# input. They remain in the SDK prototype and raw log payload for traceability.
_SESSION_CONTEXT_PARAMETERS = frozenset({"FlibHndl"})
_OUTPUT_WRAPPER_NAMES = frozenset({"u", "alm1", "alm2"})


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    type: str
    direction: str
    pointed_type: str | None = None
    array_dimensions: tuple[str, ...] = ()
    valid_length_rule: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    metadata_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["array_dimensions"] = list(self.array_dimensions)
        return value


@dataclass(frozen=True, slots=True)
class ApiMetadata:
    api_name: str
    prototype: str
    parameters: tuple[ParameterDefinition, ...]
    diagnostics: tuple[str, ...] = ()

    @property
    def input_parameters(self) -> tuple[ParameterDefinition, ...]:
        return tuple(item for item in self.parameters if item.direction in {"in", "inout"})

    @property
    def output_parameters(self) -> tuple[ParameterDefinition, ...]:
        return tuple(item for item in self.parameters if item.direction in {"out", "inout"})

    @property
    def parameter_count(self) -> int | None:
        """Legacy SDK input count retained for CSV/dataset compatibility.

        This count includes the session handle because older datasets used the
        ABI-level argument count. Use ``business_input_count`` when describing
        operation inputs to a user.
        """
        if any(item.direction not in {"in", "inout", "out"} for item in self.parameters):
            return None
        return len(self.input_parameters)

    @property
    def business_input_parameters(self) -> tuple[ParameterDefinition, ...]:
        """CNC operation inputs, excluding the live session handle."""
        return tuple(
            item for item in self.input_parameters
            if item.name not in _SESSION_CONTEXT_PARAMETERS
        )

    @property
    def business_input_count(self) -> int | None:
        if self.parameter_count is None:
            return None
        return len(self.business_input_parameters)

    @property
    def complete(self) -> bool:
        return not self.diagnostics and all(item.metadata_complete for item in self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_name": self.api_name,
            "prototype": self.prototype,
            "parameters": [item.to_dict() for item in self.parameters],
            "parameter_count": self.parameter_count,
            "business_input_count": self.business_input_count,
            "metadata_complete": self.complete,
            "diagnostics": list(self.diagnostics),
        }


@lru_cache(maxsize=1)
def _sdk_index() -> SdkIndex | None:
    header = default_focas_header_dir() / "Fwlib32.h"
    if not header.is_file():
        return None
    index_path = Path("rag_indexes/focas/sdk_type_index_0iD.json")
    try:
        return load_or_build_sdk_type_index(header, index_path)
    except (OSError, ValueError, TypeError):
        return None


def _split_arguments(raw: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(raw):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(raw[start:index].strip())
            start = index + 1
    tail = raw[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _prototype_parameter_count(prototype: str) -> int | None:
    """Return the ABI argument count from one C/C++ function declaration."""
    text = str(prototype or "")
    if "(" not in text or ")" not in text:
        return None
    raw = text.partition("(")[2].rsplit(")", 1)[0].strip()
    if not raw or raw.lower() == "void":
        return 0
    return len(_split_arguments(raw))


_C_TYPE_WORDS = frozenset({
    "bool", "char", "double", "float", "int", "long", "short", "signed",
    "unsigned", "void", "wchar_t", "size_t", "hwnd", "uint", "dword",
})


def _normalize_abi_type(value: str) -> str:
    value = re.sub(r"\b(?:const|volatile)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*\*\s*", " *", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def _header_argument_type(argument: str) -> str:
    """Remove an optional ABI parameter name from an unnamed header argument."""
    value = re.sub(r"\s+", " ", str(argument).strip())
    if "*" in value:
        value = re.sub(r"\*\s*[A-Za-z_]\w*\s*(?=\[|$)", "*", value)
    elif "[" in value:
        value = re.sub(r"\b[A-Za-z_]\w*\s*(?=\[)", "", value)
    else:
        tokens = value.split()
        if len(tokens) > 1 and tokens[-1].lower() not in _C_TYPE_WORDS:
            value = " ".join(tokens[:-1])
    return _normalize_abi_type(value)


def _prototype_parameter_types(prototype: str, *, named: bool) -> list[str] | None:
    """Return normalized parameter types from a named or ABI-only prototype."""
    text = str(prototype or "")
    if "(" not in text or ")" not in text:
        return None
    raw = text.partition("(")[2].rsplit(")", 1)[0].strip()
    if not raw or raw.lower() == "void":
        return []
    result: list[str] = []
    for argument in _split_arguments(raw):
        if named:
            parsed = _type_and_name(argument)
            if parsed is None:
                return None
            type_name, _name, arrays = parsed
            value = type_name + "".join(f"[{item}]" for item in arrays)
        else:
            result.append(_header_argument_type(argument))
            continue
        result.append(_normalize_abi_type(value))
    return result


def _argument_directions(api_name: str) -> dict[str, str]:
    text = str(_api_docs().get(api_name, {}).get("arguments", ""))
    result: dict[str, str] = {}
    for match in re.finditer(r"(?m)^-\s+([A-Za-z_]\w*)\s+\((in\s*/\s*out|inout|in\s+out|in|out)\)", text, re.I):
        direction = re.sub(r"\s+", "", match.group(2).lower()).replace("/", "")
        result[match.group(1)] = "inout" if direction in {"inout", "inout"} else direction
    # The FOCAS page describes ``path_no`` as an address supplied by the
    # caller, while its SDK example passes an uninitialised ``short`` and
    # reads the value after the call.  The local deterministic harness and
    # existing contract tests confirm both values are outputs.  Preserve that
    # verified rule instead of counting an output as an input dimension.
    if api_name == "cnc_getpath":
        result.update({"path_no": "out", "maxpath_no": "out"})
    return result


def _type_and_name(argument: str) -> tuple[str, str, tuple[str, ...]] | None:
    argument = re.sub(r"\s+", " ", argument.strip())
    if not argument or argument == "void":
        return None
    match = re.search(r"([A-Za-z_]\w*)\s*(?:(\[[^]]*\])\s*)*$", argument)
    if not match:
        return None
    name = match.group(1)
    prefix = argument[: match.start(1)].strip()
    arrays = tuple(re.findall(r"\[\s*([^]]+)\s*\]", argument[match.start(1) :]))
    return prefix, name, arrays


def _field_definition(
    type_name: str,
    index: SdkIndex | None,
    _seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if index is None:
        return {}
    inline_match = re.search(r"\b(?:union|struct)\s*\{(?P<body>.*)\}", type_name, re.IGNORECASE | re.DOTALL)
    if inline_match:
        inline_fields: dict[str, Any] = {}
        for declaration in inline_match.group("body").split(";"):
            parsed = _type_and_name(declaration)
            if parsed is None:
                continue
            nested_type, name, arrays = parsed
            details: dict[str, Any] = {
                "type": nested_type,
                "array_dimensions": list(arrays),
            }
            nested = _field_definition(nested_type, index, _seen)
            if nested:
                details["fields"] = nested
            inline_fields[name] = details
        return inline_fields
    clean = type_name.replace("const", "").replace("*", "").strip().upper()
    item = index.structs.get(clean) or index.unions.get(clean)
    if not item:
        return {}
    if clean in _seen:
        return {}
    result: dict[str, Any] = {}
    for name, details in item.get("fields", {}).items():
        field_value: dict[str, Any] = {
            "type": details.get("type"),
            "array_dimensions": list(details.get("array_dimensions", [])),
        }
        nested = _field_definition(
            str(details.get("type", "")), index, _seen | {clean}
        )
        if nested:
            field_value["fields"] = nested
        result[name] = field_value
    return result


@lru_cache(maxsize=512)
def api_metadata(api_name: str) -> ApiMetadata:
    prototype = named_prototype(api_name)
    if not prototype:
        return ApiMetadata(api_name, "", (), (f"SDK prototype unavailable for {api_name}",))
    raw = prototype.partition("(")[2].rsplit(")", 1)[0]
    directions = _argument_directions(api_name)
    index = _sdk_index()
    parameters: list[ParameterDefinition] = []
    diagnostics: list[str] = []
    header_prototype = str(index.functions.get(api_name, "")) if index is not None else ""
    documented_count = _prototype_parameter_count(prototype)
    header_count = _prototype_parameter_count(header_prototype)
    if index is not None and not header_prototype:
        diagnostics.append(
            f"SDK_PROTOTYPE_MISMATCH: {api_name}; API is absent from active Fwlib32.h"
        )
    elif documented_count is not None and header_count is not None and documented_count != header_count:
        diagnostics.append(
            f"SDK_PROTOTYPE_MISMATCH: {api_name}; documented_parameters={documented_count}; "
            f"sdk_header_parameters={header_count}"
        )
    documented_types = _prototype_parameter_types(prototype, named=True)
    header_types = _prototype_parameter_types(header_prototype, named=False)
    if documented_types is not None and header_types is not None and documented_types != header_types:
        diagnostics.append(
            f"SDK_PROTOTYPE_MISMATCH: {api_name}; documented_types={documented_types}; "
            f"sdk_header_types={header_types}"
        )
    parsed_arguments: list[tuple[str, str, tuple[str, ...], str]] = []
    for argument in _split_arguments(raw):
        parsed = _type_and_name(argument)
        if parsed is None:
            diagnostics.append(f"unparsed SDK argument: {argument}")
            continue
        type_name, name, arrays = parsed
        direction = directions.get(name, "unknown")
        parsed_arguments.append((type_name, name, arrays, direction))
    names = {name for _, name, _, _ in parsed_arguments}
    for type_name, name, arrays, direction in parsed_arguments:
        fields = _field_definition(type_name, index)
        complete = direction != "unknown"
        if direction == "unknown":
            diagnostics.append(f"direction unknown for {api_name}.{name}")
        if "*" in type_name and not fields and type_name.replace("*", "").strip().upper() not in {"CHAR", "WCHAR"}:
            # A pointer is still valid metadata even when it points to a
            # scalar; this diagnostic only marks direction uncertainty above.
            pass
        length_rule = None
        if arrays:
            length_rule = "array_dimensions=" + ",".join(arrays)
        elif "*" in type_name:
            length_sources = [candidate for candidate in ("length", "len", "num", "data_num", "msg_len") if candidate in names]
            if length_sources:
                length_rule = "runtime_bound_by=" + ",".join(length_sources)
            elif name.lower() in {"data", "buf", "msg", "messages", "position", "macro"}:
                length_rule = "runtime_length_unconfirmed"
        parameters.append(ParameterDefinition(
            name=name,
            type=type_name,
            direction=direction,
            pointed_type=type_name.replace("const", "").replace("*", "").strip() if "*" in type_name else None,
            array_dimensions=arrays,
            valid_length_rule=length_rule,
            fields=fields,
            metadata_complete=complete,
        ))
    return ApiMetadata(api_name, prototype, tuple(parameters), tuple(dict.fromkeys(diagnostics)))


def parse_payload(value: str | None) -> dict[str, Any]:
    """Parse the runtime display payload while preserving repeated keys."""
    result: dict[str, Any] = {}
    if not value:
        return result
    parts: list[str] = []
    current: list[str] = []
    escaped = False
    for char in str(value):
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ";":
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    for part in parts:
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value_obj: Any = raw
        try:
            value_obj = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            pass
        if key in result:
            previous = result[key]
            result[key] = [previous, value_obj] if not isinstance(previous, list) else [*previous, value_obj]
        else:
            result[key] = value_obj
    return result


def _select_iodbtimer_branch(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only the active ``IODBTIMER`` union branch in a payload.

    Some controller adapters serialize both views of a C union.  ``type`` is
    the SDK selector, so retaining the inactive view creates impossible logs
    (for example a date read appearing to have an hour equal to its year).
    The raw API log remains available; this only normalizes the structured
    display/coverage projection.
    """
    selector = values.get("buf.type", values.get("type"))
    try:
        selector = int(selector)
    except (TypeError, ValueError):
        return values
    if selector not in (0, 1):
        return values
    active = "date" if selector == 0 else "time"
    inactive = "time" if selector == 0 else "date"
    result = dict(values)
    for key in list(result):
        normalized = str(key).removeprefix("buf.")
        if normalized.startswith(f"data.{inactive}.") or normalized.startswith(f"{inactive}."):
            result.pop(key, None)
    return result


def format_payload(values: dict[str, Any]) -> str:
    return ";".join(f"{key}={_format_value(value)}" for key, value in values.items())


def _qualified_field_map(api_name: str, direction: str) -> dict[str, str]:
    metadata = api_metadata(api_name)
    result: dict[str, str] = {}
    collisions: set[str] = set()

    def walk(prefix: str, fields: dict[str, Any]) -> None:
        for name, details in fields.items():
            path = f"{prefix}.{name}"
            nested = details.get("fields") if isinstance(details, dict) else None
            if nested:
                walk(path, nested)
            else:
                if name in result and result[name] != path:
                    collisions.add(name)
                else:
                    result[name] = path

    for parameter in metadata.parameters:
        if parameter.direction not in {direction, "inout"}:
            continue
        walk(parameter.name, parameter.fields)
    for key in collisions:
        result.pop(key, None)
    return result


def api_payload(value: str | None, api_name: str = "", direction: str = "in") -> str:
    """Remove wrapper metadata without losing runtime API values."""
    values = parse_payload(value)
    if api_name in {"cnc_gettimer", "cnc_settimer"}:
        values = _select_iodbtimer_branch(values)
    metadata = api_metadata(api_name) if api_name else None
    field_map = _qualified_field_map(api_name, direction) if api_name else {}
    output_parameter_names = {
        item.name for item in (metadata.parameters if metadata else ())
        if item.direction in {"out", "inout"}
    }
    qualified: dict[str, Any] = {}
    for key, item in values.items():
        if "." in key or key in field_map.values():
            display_key = key
        else:
            display_key = field_map.get(key, key)
        # Output logs are for reading returned values, so omit the enclosing
        # output parameter name (for example ``statinfo.run`` -> ``run`` and
        # ``position.abs.data`` -> ``abs.data``). Keep nested member paths.
        if direction == "out":
            while "." in display_key:
                first, remainder = display_key.split(".", 1)
                bare_first = first.split("[", 1)[0]
                if bare_first not in output_parameter_names and bare_first not in _OUTPUT_WRAPPER_NAMES:
                    break
                display_key = remainder
        qualified[display_key] = item
    values = qualified
    return format_payload({key: item for key, item in values.items() if key not in _NON_API_PAYLOAD_KEYS})


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def return_description(api_name: str, code: int, provided: str | None = None) -> str:
    if provided and re.fullmatch(r"EW_[A-Z0-9_]+", str(provided).strip()):
        return str(provided).strip()
    resolved = resolve_api_return_code(api_name, code)
    names = resolved.get("api_code_names") or resolved.get("global_code_names") or []
    if names:
        return str(names[0])
    return "UNKNOWN" if code != 0 else "EW_OK"


def fixed_input_row(row: dict[str, Any]) -> dict[str, Any]:
    api_name = str(row.get("protocol_function") or row.get("api_name") or "")
    raw = api_payload(str(row.get("parameters") or row.get("params") or ""), api_name, "in")
    return {
        "timestamp": row.get("timestamp", ""),
        "api_name": api_name,
        "params": raw,
    }


def fixed_output_row(row: dict[str, Any], input_row: dict[str, Any] | None = None) -> dict[str, Any]:
    api_name = str(row.get("api_name") or row.get("protocol_function") or (input_row or {}).get("api_name") or "")
    code_raw = row["return_code"] if "return_code" in row else row.get("status_code", "")
    missing_code = code_raw in {None, ""}
    try:
        code = "" if missing_code else int(code_raw)
    except (TypeError, ValueError):
        code = 500
    data = api_payload(str(row.get("data") or row.get("response") or ""), api_name, "out")
    valid = code == 0 and "output_valid=false" not in data.lower()
    return {
        "timestamp": row.get("timestamp", ""),
        "api_name": api_name,
        "return_code": code,
        "return_desc": "UNAVAILABLE" if missing_code else return_description(api_name, int(code), row.get("return_text")),
        "success_data": data if valid else "",
    }


def structured_call_record(input_row: dict[str, Any], output_row: dict[str, Any] | None) -> dict[str, Any]:
    api_name = str(output_row and (output_row.get("api_name") or output_row.get("protocol_function")) or input_row.get("protocol_function") or input_row.get("api_name") or "")
    metadata = api_metadata(api_name)
    fixed_in = fixed_input_row({**input_row, "api_name": api_name})
    fixed_out = fixed_output_row(output_row or {"api_name": api_name, "return_code": 500, "data": "output_valid=false"}, fixed_in)
    return {
        "invocation_id": input_row.get("invocation_id") or (output_row or {}).get("invocation_id"),
        "step_id": input_row.get("step_id") or (output_row or {}).get("step_id"),
        "api_name": api_name,
        "input": {
            **fixed_in,
            "values": parse_payload(fixed_in["params"]),
            "metadata": [item.to_dict() for item in metadata.parameters],
        },
        "output": {
            **fixed_out,
            "values": parse_payload(fixed_out["success_data"]),
        },
        "return_code": fixed_out["return_code"],
        "return_desc": fixed_out["return_desc"],
        "start_timestamp_ns": input_row.get("start_timestamp_ns"),
        "end_timestamp_ns": (output_row or {}).get("end_timestamp_ns"),
        "transaction_id": input_row.get("transaction_id") or (output_row or {}).get("transaction_id") or "",
        "traffic_status": input_row.get("traffic_status") or "pending_transaction_mapping",
        "metadata_diagnostics": list(metadata.diagnostics),
    }


def write_fixed_logs(
    data_dir: Path,
    input_rows: Iterable[dict[str, Any]],
    output_rows: Iterable[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write fixed display tables and lossless structured call records."""
    data_dir.mkdir(parents=True, exist_ok=True)
    inputs = list(input_rows)
    outputs = list(output_rows)
    output_keys = {
        str(row.get("invocation_id") or row.get("index") or "") for row in outputs
    }
    # A crash before the return row is still a call-level observation.  Emit a
    # fixed output row with an empty return code rather than inventing an FOCAS
    # status; the structured record carries the missing-output diagnostic.
    for input_row in inputs:
        key = str(input_row.get("invocation_id") or input_row.get("index") or "")
        if key and key not in output_keys:
            outputs.append({
                "index": input_row.get("index"),
                "timestamp": input_row.get("timestamp", ""),
                "step_id": input_row.get("step_id", ""),
                "api_name": input_row.get("protocol_function") or input_row.get("api_name") or "",
                "return_code": "",
                "return_text": "UNAVAILABLE",
                "data": "output_valid=false",
                "invocation_id": input_row.get("invocation_id"),
                "_missing_output": True,
            })
    output_by_key = {
        str(row.get("invocation_id")): row
        for row in outputs
        if row.get("invocation_id")
    }
    output_by_index = {
        str(row.get("index")): row
        for row in outputs
        if row.get("index") is not None
    }
    input_log = data_dir / "input_log.csv"
    output_log = data_dir / "output_log.csv"
    calls_path = data_dir / "api_calls.jsonl"
    with input_log.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INPUT_LOG_FIELDS)
        writer.writeheader()
        for row in inputs:
            writer.writerow(fixed_input_row(row))
    with output_log.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_LOG_FIELDS)
        writer.writeheader()
        for row in outputs:
            writer.writerow(fixed_output_row(row))
    seen_output_keys: set[str] = set()
    with calls_path.open("w", encoding="utf-8") as stream:
        for input_row in inputs:
            key = str(input_row.get("invocation_id") or "")
            output_row = output_by_key.get(key) or output_by_index.get(str(input_row.get("index")))
            if output_row is not None:
                seen_output_keys.add(str(output_row.get("invocation_id") or output_row.get("index") or ""))
            record = structured_call_record(input_row, output_row)
            if output_row and output_row.get("_missing_output"):
                record["metadata_diagnostics"] = [
                    *record.get("metadata_diagnostics", []),
                    "API return row missing; output is unavailable",
                ]
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        # A process can write a return row after its input row has been lost
        # only in a damaged/interrupted artifact. Keep that evidence in the
        # structured stream instead of silently dropping it.
        for output_row in outputs:
            key = str(output_row.get("invocation_id") or output_row.get("index") or "")
            if key in seen_output_keys:
                continue
            synthetic_input = {
                "index": output_row.get("index"),
                "timestamp": output_row.get("timestamp", ""),
                "step_id": output_row.get("step_id", ""),
                "protocol_function": output_row.get("api_name") or output_row.get("protocol_function") or "",
                "parameters": "",
                "invocation_id": output_row.get("invocation_id"),
                "transaction_id": output_row.get("transaction_id", ""),
            }
            record = structured_call_record(synthetic_input, output_row)
            record["metadata_diagnostics"] = [
                *record.get("metadata_diagnostics", []),
                "input row missing for output invocation",
            ]
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return input_log, output_log, calls_path
