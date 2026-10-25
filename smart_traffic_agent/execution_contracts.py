"""Execution capabilities are derived from operations, never scenario names.

The model owns target arguments. The runtime owns connection/capture and, when
requested, the complete NC lifecycle. Targets and prerequisites stay distinct.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

from .models import ExecutionPlan, PlanStep

SESSION_APIS = frozenset({"cnc_allclibhndl3", "cnc_freelibhndl"})
ALARM_DIAGNOSTIC_APIS = frozenset({
    "cnc_alarm", "cnc_alarm2", "cnc_getdtailerr", "cnc_rdalmmsg",
    "cnc_rdalmmsg2", "cnc_rdalmhisno", "cnc_rdalmhistry5",
    "cnc_startophis", "cnc_stopophis",
})
# APIs whose useful output depends on an NC program changing machine state.
# They must run through the NCGuide program/Cycle Start driver when selected as
# targets; a static idle read is still allowed only as a baseline sample.
MOTION_TARGET_APIS = frozenset({
    "cnc_rdaxisdata", "cnc_rdposition", "cnc_absolute", "cnc_absolute2",
    "cnc_machine", "cnc_relative", "cnc_relative2", "cnc_distance",
    "cnc_actf", "cnc_acts", "cnc_rdspeed", "cnc_rddynamic", "cnc_rddynamic2",
    "cnc_rdsvmeter", "cnc_srvdelay", "cnc_accdecdly",
})
MOTION_RUNTIME_APIS = frozenset({
    "cnc_statinfo", "pmc_rdpmcrng", "cnc_rdprogdir3", "cnc_dwnstart3",
    "cnc_download3", "cnc_dwnend3", "cnc_search", "cnc_rdprgnum", "cnc_rdseqnum",
    "cnc_actf", "cnc_absolute",
})
# These are documented session dependencies, not interchangeable API versions.
API_DEPENDENCIES = {
    # Program-copy inputs must come from a verified directory read; the
    # disposable destination is deleted after read-back verification.
    "cnc_copyprog": ("cnc_rdprogdir3", "cnc_rdprgnum", "cnc_getdtailerr", "cnc_delete"),
    # Rename only a verified resident program into a verified unused number;
    # read the selected number so a running/selected program is not changed blindly.
    "cnc_renameprog": ("cnc_rdprogdir3", "cnc_rdprgnum", "cnc_getdtailerr"),
    "cnc_verify": ("cnc_vrfstart", "cnc_vrfend"),
    "cnc_vrfstart": ("cnc_rdprogdir3", "cnc_upstart3", "cnc_upload3", "cnc_upend3", "cnc_verify", "cnc_vrfend"),
    "cnc_vrfend": ("cnc_rdprogdir3", "cnc_upstart3", "cnc_upload3", "cnc_upend3", "cnc_vrfstart", "cnc_verify"),
    "cnc_upload3": ("cnc_upstart3", "cnc_upend3"),
    "cnc_upend3": ("cnc_upstart3", "cnc_upload3"),
    "cnc_download3": ("cnc_dwnstart3", "cnc_dwnend3"),
    "cnc_dwnend3": ("cnc_dwnstart3", "cnc_download3"),
    "cnc_dwnstart4": ("cnc_download4", "cnc_dwnend4"),
    "cnc_download4": ("cnc_dwnstart4", "cnc_dwnend4"),
    "cnc_dwnend4": ("cnc_dwnstart4", "cnc_download4"),
    "cnc_dncstart2": ("cnc_dnc2", "cnc_dncend2"),
    "cnc_dnc2": ("cnc_dncstart2", "cnc_dncend2"),
    "cnc_dncend2": ("cnc_dncstart2", "cnc_dnc2"),
    "cnc_verify4": ("cnc_vrfstart4", "cnc_vrfend4"),
    "cnc_vrfend4": ("cnc_vrfstart4", "cnc_verify4"),
    "cnc_upstart4": ("cnc_upload4", "cnc_upend4"),
    "cnc_upload4": ("cnc_upstart4", "cnc_upend4"),
    "cnc_upend4": ("cnc_upstart4", "cnc_upload4"),
    "cnc_rdpdf_line": ("cnc_rdprogdir3",),
    "cnc_wrparam": ("cnc_rdparam",),
    "cnc_wrmdipntr": ("cnc_rdmdipntr",),
}

# Session recipes are kept as data so planners and validators can use the same
# source of truth.  The recipe is intentionally per API family: two versions
# may occur in one batch, but each target gets its own independent session.
API_SEQUENCE_RULES = {
    "cnc_download3": ("cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"),
    "cnc_dwnstart3": ("cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"),
    "cnc_dwnend3": ("cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"),
    "cnc_download4": ("cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"),
    "cnc_dwnstart4": ("cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"),
    "cnc_dwnend4": ("cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"),
    "cnc_upload3": ("cnc_upstart3", "cnc_upload3", "cnc_upend3"),
    "cnc_upstart3": ("cnc_upstart3", "cnc_upload3", "cnc_upend3"),
    "cnc_upend3": ("cnc_upstart3", "cnc_upload3", "cnc_upend3"),
    "cnc_upstart4": ("cnc_upstart4", "cnc_upload4", "cnc_upend4"),
    "cnc_upload4": ("cnc_upstart4", "cnc_upload4", "cnc_upend4"),
    "cnc_upend4": ("cnc_upstart4", "cnc_upload4", "cnc_upend4"),
    "cnc_verify4": ("cnc_vrfstart4", "cnc_verify4", "cnc_vrfend4"),
    "cnc_vrfend4": ("cnc_vrfstart4", "cnc_verify4", "cnc_vrfend4"),
    "cnc_dnc2": ("cnc_dncstart2", "cnc_dnc2", "cnc_dncend2"),
}

# Confirmed wire semantics for APIs whose data calls belong to a larger
# transfer session. These declarations are consumed by the PCAP mapper and
# coverage audit; names are explicit because suffixes do not define protocol
# behavior.
TRAFFIC_SESSION_TYPES = frozenset({
    "ordinary_request_response",
    "request_segment_session",
    "response_segment_session",
    "mixed_transaction_session",
})
API_TRAFFIC_SESSION_RULES = {
    "cnc_download3": {
        "session_type": "request_segment_session",
        "start_api": "cnc_dwnstart3", "segment_api": "cnc_download3", "end_api": "cnc_dwnend3",
        "wire_direction": "request",
    },
    "cnc_download4": {
        "session_type": "request_segment_session",
        "start_api": "cnc_dwnstart4", "segment_api": "cnc_download4", "end_api": "cnc_dwnend4",
        "wire_direction": "request",
    },
    "cnc_verify4": {
        "session_type": "request_segment_session",
        "start_api": "cnc_vrfstart4", "segment_api": "cnc_verify4", "end_api": "cnc_vrfend4",
        "wire_direction": "request",
    },
    "cnc_verify": {
        "session_type": "request_segment_session",
        "start_api": "cnc_vrfstart", "segment_api": "cnc_verify", "end_api": "cnc_vrfend",
        "wire_direction": "request",
    },
    "cnc_upload3": {
        "session_type": "response_segment_session",
        "start_api": "cnc_upstart3", "segment_api": "cnc_upload3", "end_api": "cnc_upend3",
        "wire_direction": "response",
    },
    "cnc_upload4": {
        "session_type": "response_segment_session",
        "start_api": "cnc_upstart4", "segment_api": "cnc_upload4", "end_api": "cnc_upend4",
        "wire_direction": "response",
    },
    "cnc_dnc2": {
        "session_type": "mixed_transaction_session",
        "start_api": "cnc_dncstart2", "segment_api": "cnc_dnc2", "end_api": "cnc_dncend2",
        "wire_direction": "both",
    },
}

def requires_isolated_api_session(functions: Iterable[str]) -> bool:
    """Whether targets must be dispatched one target per workflow run."""
    return bool(set(functions) & set(API_SEQUENCE_RULES))

def api_sequence_rules(functions: Iterable[str]) -> list[dict[str, object]]:
    """Return independent documented sessions for the selected APIs.

    This is deliberately data-driven; adding a RAG-extracted recipe only
    requires extending API_SEQUENCE_RULES, rather than adding generator code.
    """
    selected = {str(name).strip() for name in functions if str(name).strip()}
    return [
        {"target": target, "calls": list(recipe), "helpers": [recipe[0], recipe[2]]}
        for target, recipe in API_SEQUENCE_RULES.items() if target in selected
    ]


# Cross-API interaction rules used by PlanningAgent and CodeGenerationAgent.
# The target API remains the coverage objective; listed APIs are state/discovery
# helpers that must be planned before/after it when their evidence is available.
API_INTERACTION_RULES = {
    "cnc_copyprog": {"before": ("cnc_rdprogdir3", "cnc_rdprgnum"), "after": ("cnc_delete",), "on_error": ("cnc_getdtailerr",), "reason": "discover a resident source and disposable destinations; restore each destination"},
    "cnc_renameprog": {"before": ("cnc_rdprogdir3", "cnc_rdprgnum"), "after": ("cnc_rdprogdir3", "cnc_rdprgnum"), "on_error": ("cnc_getdtailerr",), "reason": "discover a resident source and unused destination, rename only a verified pair, then rename it back and verify the original directory/current program state"},
    "cnc_search": {"before": ("cnc_rdprogdir3", "cnc_rdprgnum"), "after": ("cnc_rdprgnum",), "on_error": ("cnc_getdtailerr",), "reason": "select only discovered programs and verify selection/restoration"},
    "cnc_rdpdf_line": {"before": ("cnc_rdprogdir3",), "reason": "derive a real program/file name from directory discovery"},
    "cnc_download3": {"before": ("cnc_dwnstart3",), "after": ("cnc_dwnend3",), "reason": "download is a start/write/end transaction"},
    "cnc_dwnstart4": {"before": (), "after": ("cnc_download4", "cnc_dwnend4"), "reason": "download4 is a version-4 start/write/end transaction; all calls require a successful start"},
    "cnc_download4": {"before": ("cnc_dwnstart4",), "after": ("cnc_dwnend4",), "reason": "download4 must use the matching version-4 start and end functions"},
    "cnc_dwnend4": {"before": ("cnc_dwnstart4", "cnc_download4"), "after": (), "reason": "download4 end is valid only for an active version-4 download session"},
    "cnc_dncstart2": {"before": (), "after": ("cnc_dnc2", "cnc_dncend2"), "reason": "cnc_dncstart2 reports whether CNC-side DNC/M198 is already active; it does not start it"},
    "cnc_dnc2": {"before": ("cnc_dncstart2",), "after": ("cnc_dncend2",), "reason": "DNC data transfer requires an already active CNC-side DNC/M198 operation, confirmed by cnc_dncstart2 returning EW_OK"},
    "cnc_dncend2": {"before": ("cnc_dncstart2", "cnc_dnc2"), "after": (), "reason": "DNC end requires an active CNC-side DNC/M198 operation; cnc_dncstart2 only reports that state"},
    "cnc_upload3": {"before": ("cnc_upstart3",), "after": ("cnc_upend3",), "reason": "upload is a start/read/end transaction"},
    "cnc_wrparam": {"before": ("cnc_rdparam",), "after": ("cnc_rdparam",), "reason": "snapshot, write, read back, and restore parameter value"},
    "cnc_wrmdipntr": {"before": ("cnc_rdmdipntr",), "after": ("cnc_rdmdipntr",), "reason": "snapshot and verify the MDI pointer before restoration"},
}

def analyze_api_interactions(functions: Iterable[str], api_context: Iterable[dict] | None = None) -> dict[str, dict]:
    """Return explicit cross-API discovery/lifecycle/restore requirements.

    This is deliberately separate from the target set: helper calls explain how
    a target is made safe and never count as target coverage. RAG descriptions
    may add evidence, but cannot remove an official rule.
    """
    selected = {str(name).strip() for name in functions if str(name).strip()}
    result = {}
    for target in sorted(selected):
        rule = API_INTERACTION_RULES.get(target)
        if rule:
            result[target] = {
                "target": target,
                "before": list(rule.get("before", ())),
                "after": list(rule.get("after", ())),
                "on_error": list(rule.get("on_error", ())),
                "reason": rule.get("reason", ""),
                "helper_functions": sorted(set(rule.get("before", ())) | set(rule.get("after", ())) | set(rule.get("on_error", ()))),
            }
    # Keep only helpers that have ABI/RAG evidence when context is supplied.
    if api_context is not None:
        known = {str(row.get("function", row.get("protocol_function", ""))).strip() for row in api_context if isinstance(row, dict)}
        for item in result.values():
            for key in ("before", "after", "on_error", "helper_functions"):
                item[key] = [name for name in item[key] if name in known or name in selected]
    return result


# Normalization and plan review must agree on equivalent lifecycle operations.
LIFECYCLE_API_ALIASES = {
    "UploadProgram": frozenset({"cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"}),
    "DownloadProgram4": frozenset({"cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"}),
    "SelectProgram": frozenset({"cnc_search"}),
    "ReadProgramNumber": frozenset({"cnc_rdprgnum"}),
}


def validate_api_lifecycle_fragment(fragment: str) -> list[str]:
    """Reject mismatched download versions and unguarded session API calls."""
    errors: list[str] = []

    def calls(name: str) -> list[int]:
        return [match.start() for match in re.finditer(rf"\bapi\.{re.escape(name)}\s*\(", fragment)]

    for version in ("3", "4"):
        start, transfer, end = (f"cnc_dwnstart{version}", f"cnc_download{version}", f"cnc_dwnend{version}")
        if calls(transfer):
            for required in (start, end):
                if not calls(required):
                    errors.append(f"BLOCKING: {transfer} requires matching {start} and {end} calls.")
            starts, transfers, ends = calls(start), calls(transfer), calls(end)
            if starts and ends and not (starts[0] < transfers[0] < ends[-1]):
                errors.append(f"BLOCKING: {start} must precede {transfer}, which must precede {end}.")
            # A version-3 and a version-4 session may both be present, but a
            # transfer must be enclosed by its own matching start/end pair.
            for pos in transfers:
                matching_starts = [item for item in starts if item < pos]
                matching_ends = [item for item in ends if item > pos]
                if not matching_starts or not matching_ends:
                    errors.append(f"BLOCKING: {transfer} is outside its matching {start}/{end} session.")
                other = "4" if version == "3" else "3"
                other_positions = calls(f"cnc_dwnstart{other}") + calls(f"cnc_download{other}") + calls(f"cnc_dwnend{other}")
                if other_positions and (not matching_starts or not matching_ends):
                    errors.append(f"BLOCKING: {transfer} cannot be paired with version-{other} download session APIs.")
                elif matching_starts and matching_ends and any(
                    matching_starts[-1] < other_pos < matching_ends[0] for other_pos in other_positions
                ):
                    errors.append(f"BLOCKING: {transfer} cannot be paired with version-{other} download session APIs.")
            if starts and transfers:
                if version == "4":
                    for arguments in _cpp_api_call_arguments(fragment, "cnc_dwnstart4"):
                        if len(arguments) < 2:
                            errors.append(
                                "BLOCKING: cnc_dwnstart4 call must provide handle, type, and dir_name arguments."
                            )
                            continue
                        type_argument = re.sub(r"\s+", "", arguments[1])
                        if type_argument not in {"0", "static_cast<short>(0)"}:
                            errors.append(
                                "BLOCKING: cnc_dwnstart4 NC-program sessions must pass type=0; "
                                f"found type argument {arguments[1].strip()}."
                            )
                start_pos = starts[0]
                assignment = re.search(
                    rf"(?:short|auto)\s+(\w+)\s*=\s*ctx\.call\s*"
                    rf"\((?:(?!\bctx\.call\s*\().)*?[\"']{re.escape(start)}[\"']",
                    fragment[:transfers[0]], re.S,
                )
                if assignment:
                    result_name = assignment.group(1)
                    transfer_pos = transfers[0]
                    failure_guard = _cpp_has_exit_guard(
                        fragment, result_name, start_pos, transfer_pos
                    )
                    success_guard = _cpp_has_success_guard(
                        fragment, result_name, transfer_pos
                    )
                    if not failure_guard and not success_guard:
                        errors.append(f"BLOCKING: {transfer} must be skipped unless {start} returns EW_OK.")

    dnc_calls = {name: calls(name) for name in ("cnc_dnc2", "cnc_dncend2")}
    if any(dnc_calls.values()):
        start_calls = calls("cnc_dncstart2")
        start_assignment = re.search(r"(?:short|auto)\s+(\w+)\s*=\s*ctx\.call\s*\([^;]{0,500}$", fragment[:start_calls[0]], re.S) if start_calls else None
        start_match = (
            re.search(rf"if\s*\(\s*{re.escape(start_assignment.group(1))}\s*==\s*EW_OK\s*\)\s*\{{", fragment[start_calls[0]:])
            if start_calls and start_assignment else None
        )
        if not start_calls or not start_match:
            errors.append("BLOCKING: cnc_dnc2/cnc_dncend2 require a successful cnc_dncstart2 gate.")
        else:
            depth, close = 1, start_match.end()
            gate_offset = start_calls[0]
            while gate_offset + close < len(fragment) and depth:
                if fragment[gate_offset + close] == "{": depth += 1
                elif fragment[gate_offset + close] == "}": depth -= 1
                close += 1
            body_start, body_end = gate_offset + start_match.end(), gate_offset + close - 1
            for name, positions in dnc_calls.items():
                if any(not (body_start <= pos < body_end) for pos in positions):
                    errors.append(f"BLOCKING: {name} must only be called after cnc_dncstart2 returns EW_OK.")
    return errors


def _cpp_has_exit_guard(source: str, variable: str, start: int, before: int) -> bool:
    pattern = re.compile(
        rf"if\s*\(\s*{re.escape(variable)}\s*!=\s*EW_OK\s*\)\s*(\{{)?"
    )
    for match in pattern.finditer(source, start, before):
        if match.group(1):
            end = _cpp_matching_brace(source, match.end() - 1)
            if end is not None and end <= before and re.search(
                r"\b(?:return|break|continue|throw)\b", source[match.end():end]
            ):
                return True
        else:
            statement = re.match(
                r"\s*(?:return\b[^;]*|break\b|continue\b|throw\b[^;]*)\s*;",
                source[match.end():before],
            )
            if statement:
                return True
    return False


def _cpp_has_success_guard(source: str, variable: str, call_position: int) -> bool:
    pattern = re.compile(
        rf"if\s*\(\s*{re.escape(variable)}\s*==\s*EW_OK\s*\)\s*\{{"
    )
    for match in pattern.finditer(source, 0, call_position):
        end = _cpp_matching_brace(source, match.end() - 1)
        if end is not None and match.end() <= call_position < end:
            return True
    return False


def _cpp_matching_brace(source: str, open_position: int) -> int | None:
    depth = 0
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    index = open_position
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
        elif block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 1
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char == "/" and next_char == "/":
            line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            block_comment = True
            index += 1
        elif char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _cpp_api_call_arguments(source: str, function_name: str) -> list[list[str]]:
    """Extract complete arguments from api.<name>(...), including nested casts."""
    calls: list[list[str]] = []
    pattern = re.compile(rf"\bapi\.{re.escape(function_name)}\s*\(")
    for match in pattern.finditer(source):
        args: list[str] = []
        start = match.end()
        token_start = start
        parens = braces = brackets = 0
        quote = ""
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = ""
                continue
            if char in {"'", '"'}:
                quote = char
            elif char == "(":
                parens += 1
            elif char == ")":
                if parens == braces == brackets == 0:
                    args.append(source[token_start:index].strip())
                    break
                parens -= 1
            elif char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
            elif char == "[":
                brackets += 1
            elif char == "]":
                brackets -= 1
            elif char == "," and parens == braces == brackets == 0:
                args.append(source[token_start:index].strip())
                token_start = index + 1
        calls.append(args)
    return calls


def lifecycle_interfaces(steps: Iterable[PlanStep]) -> set[str]:
    steps = list(steps)
    interfaces = {step.interface_name for step in steps}
    functions = set().union(*(step_functions(step) for step in steps))
    interfaces.update(name for name, required in LIFECYCLE_API_ALIASES.items()
                      if required <= functions)
    if any(step.operation_kind == "ncguide_ui" and "cycle_start" in step.protocol_function
           for step in steps):
        interfaces.add("StartProgram")
    return interfaces


def step_functions(step: PlanStep) -> set[str]:
    text = step.protocol_function + " " + step.interface_name
    for call in step.api_calls:
        text += " " + str(call.get("function", call.get("protocol_function", call.get("tool_name", ""))))
    return set(re.findall(r"\b(?:cnc|pmc)_\w+\b", text))


def dependency_closure(functions: Iterable[str]) -> set[str]:
    result = set(functions)
    pending = list(result)
    while pending:
        for dependency in API_DEPENDENCIES.get(pending.pop(), ()):
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return result


def motion_required_for_targets(functions: Iterable[str]) -> bool:
    return bool(set(functions) & MOTION_TARGET_APIS)


@dataclass(frozen=True)
class ExecutionContract:
    targets: tuple[str, ...]
    helpers: tuple[str, ...]
    motion: bool = False
    version: int = 1

    @property
    def functions(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.targets) | set(self.helpers)))

    def to_dict(self) -> dict:
        return asdict(self)


def execution_contract(plan: ExecutionPlan, targets: Iterable[str] = ()) -> ExecutionContract:
    selected = set(targets)
    manifest = plan.rag_context.get("function_coverage_manifest", {})
    if not selected:
        selected = {str(row.get("function", "")) for row in manifest.get("selected_batch", [])
                    if row.get("coverage_role") == "target"}
    # A plan may describe Cycle Start through a generic NCGuide UI step rather
    # than the semantic StartProgram interface.  Axis/position sampling APIs
    # need the same motion driver when the plan asks for a running NC program.
    motion = (
        motion_required_for_targets(selected)
        or
        (not selected and plan.scenario_type in {"coordinate_motion", "spindle_speed_change", "feed_speed_change"})
        or any(
            step.interface_name == "StartProgram"
            or (
                step.operation_kind == "ncguide_ui"
                and (
                    "cycle_start" in str(step.protocol_function).lower()
                    or "cycle start" in str(step.action).lower()
                    or "start" in str(step.action).lower()
                )
            )
            for step in plan.steps
        )
    )
    if not selected:
        selected = set().union(*(step_functions(s) for s in plan.steps)) - SESSION_APIS
        if motion:
            selected -= MOTION_RUNTIME_APIS - {"cnc_statinfo", "cnc_actf", "cnc_absolute"}
    selected.discard("")
    functions = dependency_closure(selected) | SESSION_APIS | {"cnc_statinfo"} | ALARM_DIAGNOSTIC_APIS
    # Prerequisite checks and preparation calls are support traffic. They are
    # included in the capture/runtime contract but remain distinct from target
    # coverage metrics.
    prerequisite_plan = plan.rag_context.get("api_prerequisites", {})
    if isinstance(prerequisite_plan, dict):
        for operation in prerequisite_plan.get("preparation_operations", []):
            for row in operation.get("operations", []) if isinstance(operation, dict) else []:
                if isinstance(row, dict) and row.get("api"):
                    functions.add(str(row["api"]))
            for row in operation.get("verification", []) if isinstance(operation, dict) else []:
                if isinstance(row, dict) and row.get("api"):
                    functions.add(str(row["api"]))
    if motion:
        functions |= MOTION_RUNTIME_APIS
    plan.rag_context["api_sequence"] = api_sequence_rules(selected)
    return ExecutionContract(tuple(sorted(selected)), tuple(sorted(functions - selected)), motion)


def normalize_lifecycle_steps(steps: list[PlanStep]) -> None:
    """Remove duplicate semantic wrappers and order setup after connection.

    Explicit directory API versions remain separate operations. Repeated reads
    and target sampling are retained; only lifecycle duplicates are removed.
    """
    aliases = LIFECYCLE_API_ALIASES
    explicit = set().union(*(step_functions(s) for s in steps if s.interface_name not in aliases))
    kept = [s for s in steps if s.interface_name not in aliases or not aliases[s.interface_name] <= explicit]
    connect = [s for s in kept if "cnc_allclibhndl3" in step_functions(s) or s.interface_name == "AllocateHandle"]
    disconnect = [s for s in kept if "cnc_freelibhndl" in step_functions(s) or s.interface_name == "FreeHandle"]
    middle = [s for s in kept if s not in connect and s not in disconnect]
    before = [s for s in middle if s.phase == "before"]
    during = [s for s in middle if s.phase == "during"]
    after = [s for s in middle if s.phase == "after"]
    steps[:] = connect[:1] + before + during + after + disconnect[:1]
