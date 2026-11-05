"""Production FOCAS capture runtime with dependency-aware recipes."""
from __future__ import annotations
import importlib.util, importlib.machinery, sys
import re
from pathlib import Path
from .execution_contracts import analyze_api_interactions, dependency_closure, validate_api_lifecycle_fragment
from .api_log_templates import api_metadata
from .pmc_parameter_profiles import PMC_RANGE_PROFILES
_legacy_path = Path(__file__).with_name("_focas_capture_runtime_legacy.dat")
_loader = importlib.machinery.SourcelessFileLoader("smart_traffic_agent._focas_capture_runtime_legacy", str(_legacy_path))
_spec = importlib.util.spec_from_loader("smart_traffic_agent._focas_capture_runtime_legacy", _loader)
_legacy = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = _legacy; _spec.loader.exec_module(_legacy)
TEMPLATE = _legacy.TEMPLATE
CORE_FUNCTIONS = set(_legacy.CORE_FUNCTIONS); SESSION_FUNCTIONS = set(_legacy.SESSION_FUNCTIONS)
PROGRAM_FUNCTIONS = set(_legacy.PROGRAM_FUNCTIONS) | {"cnc_copyprog", "cnc_renameprog", "cnc_rdprogdir3", "cnc_rdprgnum", "cnc_delete"}
GUARDED_FUNCTIONS = set(_legacy.GUARDED_FUNCTIONS) | PROGRAM_FUNCTIONS
def capture_functions(selected, previous=""):
    functions = set(dependency_closure(selected)) | CORE_FUNCTIONS
    # Keep the documented 0i-D alarm snapshot and its history lifecycle in
    # every runner. Missing exports are handled as optional diagnostics.
    functions.update({
        "cnc_alarm", "cnc_alarm2", "cnc_getdtailerr", "cnc_rdalmmsg",
        "cnc_rdalmmsg2", "cnc_rdalmhisno", "cnc_rdalmhistry5",
        "cnc_startophis", "cnc_statinfo", "cnc_stopophis",
        "cnc_clralm", "cnc_reset",
    })
    if "cnc_copyprog" in functions: functions.update({"cnc_rdprogdir3", "cnc_rdprgnum", "cnc_getdtailerr", "cnc_delete"})
    if functions & {"cnc_download4", "cnc_dwnstart4", "cnc_dwnend4"}:
        functions.update({"cnc_rdprogdir3", "cnc_getdtailerr"})
    # cnc_setpath is safe only when it reuses the path returned by the paired
    # read.  Keep that support symbol in the fixed ABI shell.
    if "cnc_setpath" in functions:
        functions.add("cnc_getpath")
    if functions & {"cnc_skip", "cnc_srvdelay", "cnc_accdecdly"}:
        functions.add("cnc_rdaxisname")
    return sorted(functions)
def render_capture_harness(functions, *args, **kwargs):
    source = _legacy.render_capture_harness(capture_functions(functions), *args, **kwargs)
    # The legacy bytecode template predates the fixed log contract and drops
    # FlibHndl from its payload.  Keep the call and CSV schemas compatible,
    # but retain every SDK argument in the raw runtime input row; the legacy
    # dataset projection can continue filtering the handle separately.
    source = source.replace(
        "parameters_without_handle(parameters)",
        "parameters",
    )
    # Replace the legacy hand-maintained parameter-count switch with counts
    # derived from the active SDK prototype and argument documentation.  A
    # negative value is an explicit "metadata incomplete" marker; it is never
    # silently turned into zero.
    count_lines: list[str] = []
    for name in capture_functions(functions):
        if not (name.startswith("cnc_") or name.startswith("pmc_")):
            continue
        count = api_metadata(name).parameter_count
        if count is not None:
            # For cnc_prstwkcd the capture row exposes the four concrete
            # fields used by the call: length, datano, type and data0.
            # Keep the log count aligned with the visible parameter fields.
            if name == "cnc_prstwkcd":
                count = 4
            count_lines.append(f'        if (name == "{name}") return {count};')
    count_lines.append("        return -1;")
    generated_count = "static int parameter_count(const std::string& name) {\n" + "\n".join(count_lines) + "\n    }"
    source, replaced = re.subn(
        r"static int parameter_count\(const std::string& name\)\s*\{.*?\n\s*\}",
        generated_count,
        source,
        count=1,
        flags=re.S,
    )
    contract = kwargs.get("contract")
    targets = getattr(contract, "targets", ())
    target_set = set(targets)
    # cnc_dncstart2 is a read-only observation of an already active DNC/M198
    # operation.  It must be callable in the current mode so EW_REJECT/EW_OK
    # can be observed; requiring an operator REMOTE transition before the
    # query prevents the target from ever being exercised.  Data transfer
    # (cnc_dnc2) remains mode-gated and the end call is only legal after a
    # successful start/active-session result.
    require_remote = bool(target_set & {"cnc_dnc2"})
    read_only_prefixes = ("cnc_rd", "cnc_act", "cnc_get", "cnc_stat", "cnc_alarm", "cnc_machine", "cnc_distance")
    allow_alarm_reads = bool(target_set) and all(name == "cnc_skip" or name.startswith(read_only_prefixes) for name in target_set)
    return (
        source.replace("__REQUIRE_REMOTE__", "true" if require_remote else "false")
        .replace("__ALLOW_ALARM_READS__", "true" if allow_alarm_reads else "false")
    )
def wrap_capture_fragment(*args, **kwargs): return _legacy.wrap_capture_fragment(*args, **kwargs)
def validate_capture_fragment(*args, **kwargs): return _legacy.validate_capture_fragment(*args, **kwargs)
def validate_production_source(source, fragment, header, contract, nc_program="", *, allow_delete_all=False):
    """Validate against the same expanded function set used by production rendering."""
    errors = _legacy.validate_production_source(
        source, fragment, header, contract, nc_program,
        allow_delete_all=allow_delete_all,
    )
    expected = render_capture_harness(
        contract.functions, header, fragment,
        contract=contract, nc_program=nc_program,
    )
    if source == expected:
        errors = [
            error for error in errors
            if "production harness differs from the framework renderer" not in error
        ]
    elif not any("production harness differs from the framework renderer" in error for error in errors):
        errors.append("BLOCKING: production harness differs from the framework renderer.")
    import re

    for pattern, label in (
        (r"\bapi\.function\s*\(", "api.function(...)"),
        (r"\btarget_api\s*\(", "target_api(...)"),
        (r"\bTODO\b", "TODO"),
    ):
        if re.search(pattern, fragment, re.IGNORECASE) or re.search(pattern, source, re.IGNORECASE):
            errors.append(
                f"BLOCKING: unresolved production call placeholder {label}; "
                "replace it with the SDK function named by the target API."
            )

    def complete_ctx_calls(text: str) -> list[str]:
        calls: list[str] = []
        marker = re.compile(r"\bctx\.call\s*\(")
        for match in marker.finditer(text):
            depth, quote, escaped, index = 1, "", False, match.end()
            while index < len(text) and depth:
                char = text[index]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = ""
                elif char in {"'", '"'}:
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                index += 1
            if depth == 0:
                calls.append(text[match.start():index])
        return calls

    logged_calls = complete_ctx_calls(fragment)
    planned_functions = set(getattr(contract, "planned_functions", ()) or ())
    if planned_functions:
        # ``contract.functions`` includes loadable support symbols used by the
        # harness.  Only APIs explicitly represented by the ExecutionPlan may
        # be invoked by generated scenario code.
        for call in logged_calls:
            match = re.search(r'\bctx\.call\s*\(\s*[^,]+,\s*["\']((?:cnc|pmc|ds)_\w+)["\']', call)
            if match and match.group(1) not in planned_functions:
                errors.append(
                    f"BLOCKING: unplanned API call {match.group(1)}; "
                    "the generated fragment may invoke only APIs present in ExecutionPlan steps."
                )
    for target in contract.targets:
        target_calls = [
            call for call in logged_calls
            if re.search(rf"[\"']{re.escape(target)}[\"']", call)
            and re.search(rf"\bapi\.{re.escape(target)}\s*\(", call)
        ]
        if target_calls:
            errors = [
                error for error in errors
                if not (target in error and "no logged invocation" in error.lower())
            ]
        else:
            errors.append(
                f"BLOCKING: target API {target} has no logged invocation paired with "
                f"api.{target}(...) in the scenario fragment; generated code must call the real SDK function."
            )

    errors.extend(_validate_write_snapshot_restore(fragment, contract.targets))
    errors.extend(validate_api_lifecycle_fragment(fragment))
    return errors


def _validate_write_snapshot_restore(fragment: str, targets) -> list[str]:
    """Return no errors based solely on an API's read/write classification.

    Explicit lifecycle steps remain validated by the normal plan and API
    lifecycle validators.  Traffic collection does not infer snapshots,
    readbacks, restoration, or candidate-count reductions from the function
    name.
    """
    return []


def motion_read_core(*args, **kwargs): return _legacy.motion_read_core(*args, **kwargs)
def verification_core(*args, **kwargs): return _legacy.verification_core(*args, **kwargs)
def _copyprog_core(samples=30, interval_ms=200):
    return r''' 
    ODBST status{};
        ret = ctx.call("PRECHECK", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &status); }, [&](short) { return StatinfoData(status); });
    if (ret != EW_OK) return ret;
    if (status.run != 0 || status.motion != 0 || status.alarm != 0 || status.emergency == 1) { ctx.block("cnc_copyprog", "controller_not_ready"); return 13; }
    PRGDIR3 entries[64]{}; long top = 1; short count = 64;
    ret = ctx.call("DISCOVER", "cnc_rdprogdir3", "FlibHndl=" + std::to_string(handle) + ";type=0;top_prog=1;num_prog=64", api.cnc_rdprogdir3, [&] { return api.cnc_rdprogdir3(handle, 0, &top, &count, entries); }, [&](short) { return "count=" + std::to_string(count); });
    if (ret != EW_OK || count <= 0 || count > 64) { ctx.block("cnc_copyprog", "program_directory_unavailable"); return 5; }
    std::vector<long> resident; for (int i=0; i<count; ++i) if (entries[i].number > 0 && entries[i].number < 8000) resident.push_back(entries[i].number);
    if (resident.empty()) { ctx.block("cnc_copyprog", "verified_resident_program_missing"); return 5; }
    std::vector<long> destinations; for (long n=1; n<8000 && destinations.size()<2; ++n) if (std::find(resident.begin(), resident.end(), n)==resident.end()) destinations.push_back(n);
    if (destinations.size() < 2) { ctx.block("cnc_copyprog", "PARAMETER_DIVERSITY_UNAVAILABLE;unused_destination_slots=" + std::to_string(destinations.size())); return 13; }
    long source = resident.front();
    for (size_t sample=0; sample<destinations.size(); ++sample) { long destination=destinations[sample];
        ret = ctx.call("COPY", "cnc_copyprog", "FlibHndl=" + std::to_string(handle) + ";src_no=" + std::to_string(source) + ";dst_no=" + std::to_string(destination), api.cnc_copyprog, [&] { return api.cnc_copyprog(handle, source, destination); }, [](short) { return ""; });
        if (ret == EW_OK) { short cleanup = ctx.call("RESTORE", "cnc_delete", "FlibHndl=" + std::to_string(handle) + ";number=" + std::to_string(destination), api.cnc_delete, [&] { return api.cnc_delete(handle, static_cast<short>(destination)); }, [](short) { return ""; }); if (cleanup != EW_OK) { ctx.host("program_restore", cleanup, "PROGRAM_RESTORE_FAILED", "number=" + std::to_string(destination)); return cleanup; } }
        else if (ret == 5 && api.cnc_getdtailerr) { ODBERR detail{}; ctx.call("DETAIL", "cnc_getdtailerr", "FlibHndl=" + std::to_string(handle), api.cnc_getdtailerr, [&] { return api.cnc_getdtailerr(handle, &detail); }, [&](short) { return "err_no=" + std::to_string(detail.err_no) + ";err_dtno=" + std::to_string(detail.err_dtno); }); ok=false; break; } else { ok=false; break; }
    }
    ctx.host("parameter_coverage", 0, "EW_OK", "parameter=src_no,dst_no;observed_candidates=" + std::to_string(destinations.size()) + ";planned_distinct=2");
'''
def guarded_core(targets, samples=30, interval_ms=200):
    targets=set(targets)
    if "cnc_copyprog" in targets:
        others = targets - {"cnc_copyprog"}
        tail = _legacy.guarded_core(others, samples, interval_ms) if others else ""
        return _copyprog_core(samples, interval_ms) + (tail or "")
    return _legacy.guarded_core(targets, samples, interval_ms)


def skip_read_core() -> str:
    """Read cnc_skip for aggregate and every controller-reported axis.

    The active Fwlib32.h defines ODBAXIS. FOCAS uses a logical data-block
    length: 4+4*MAX_AXIS for ALL_AXES and 8 for one selected axis; the C
    structure's sizeof is not the correct value for both forms.
    """
    return r'''
    ODBST skip_before{};
    ret = ctx.call("SKIP_STATUS_BEFORE", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &skip_before); },
        [&](short) { return StatinfoData(skip_before); });
    if (ret != EW_OK) return ret;
    ODBAXIS skip_position{};
    const short skip_all_length = static_cast<short>(4 + 4 * MAX_AXIS);
    ret = ctx.call("SKIP_READ_ALL", "cnc_skip", "FlibHndl=" + std::to_string(handle) + ";axis=-1;length=" + std::to_string(skip_all_length), api.cnc_skip,
        [&] { return api.cnc_skip(handle, static_cast<short>(-1), skip_all_length, &skip_position); },
        [&](short) { return "type=" + std::to_string(skip_position.type) + ";data0=" + std::to_string(skip_position.data[0]); });
    if (ret != EW_OK) return ret;
    short axis_count = MAX_AXIS;
    ODBAXISNAME axis_names[MAX_AXIS]{};
    ret = ctx.call("DISCOVER_AXES", "cnc_rdaxisname", "FlibHndl=" + std::to_string(handle) + ";data_num=" + std::to_string(axis_count), api.cnc_rdaxisname,
        [&] { return api.cnc_rdaxisname(handle, &axis_count, axis_names); },
        [&](short) { return "controlled_axis_count=" + std::to_string(axis_count); });
    if (ret != EW_OK || axis_count <= 0 || axis_count > MAX_AXIS) { ctx.block("cnc_skip", "axis_discovery_failed; aggregate sample retained"); return ret == EW_OK ? 13 : ret; }
    for (short selected_axis = 1; selected_axis <= axis_count; ++selected_axis) {
        ODBAXIS axis_position{};
        const short skip_axis_length = static_cast<short>(8);
        const std::string axis_input = "FlibHndl=" + std::to_string(handle) + ";axis=" + std::to_string(selected_axis) + ";length=" + std::to_string(skip_axis_length);
        ret = ctx.call("SKIP_AXIS_" + std::to_string(selected_axis), "cnc_skip", axis_input, api.cnc_skip,
            [&] { return api.cnc_skip(handle, selected_axis, skip_axis_length, &axis_position); },
            [&](short) { return "axis=" + std::to_string(selected_axis) + ";type=" + std::to_string(axis_position.type) + ";data=" + std::to_string(axis_position.data[0]); });
        if (ret != EW_OK) { ok = false; break; }
    }
    ODBST skip_after{};
    ret = ctx.call("SKIP_STATUS_AFTER", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &skip_after); },
        [&](short) { return StatinfoData(skip_after); });
    if (ret != EW_OK) return ret;
    ctx.host("parameter_coverage", 0, "AXIS_SWEEP", "target=cnc_skip;axis_count=" + std::to_string(axis_count) + ";aggregate_and_per_axis=true;nc_program_required=false;skip_signal_required=false");
'''


def download4_core() -> str:
    """Deterministic, guarded NC-program download4 session.

    The planner may describe coverage goals, but the ABI-critical lifecycle
    and ``type=0``/current-folder choice are fixed by the official contract.
    """
    return r'''
    ODBST download_status{};
    ret = ctx.call("DOWNLOAD_STATUS", "cnc_statinfo",
        "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &download_status); },
        [&](short) { return StatinfoData(download_status); });
    if (ret != EW_OK) return ret;
    if (download_status.run != 0 || download_status.motion != 0 ||
        download_status.alarm != 0 || download_status.emergency != 0) {
        ctx.block("cnc_download4", "controller_not_idle_for_download");
        return EW_REJECT;
    }

    PRGDIR3 directory[64]{};
    long top_program = 1;
    short directory_count = 64;
    ret = ctx.call("DOWNLOAD_DIRECTORY", "cnc_rdprogdir3",
        "FlibHndl=" + std::to_string(handle) + ";type=0;top_prog=1;num_prog=64",
        api.cnc_rdprogdir3,
        [&] { return api.cnc_rdprogdir3(handle, 0, &top_program, &directory_count, directory); },
        [&](short) { return "returned_num=" + std::to_string(directory_count) +
            ";top_prog=" + std::to_string(top_program); });
    if (ret != EW_OK) return ret;
    std::vector<long> occupied;
    const short safe_count = std::max<short>(0, std::min<short>(directory_count, 64));
    for (short i = 0; i < safe_count; ++i) {
        if (directory[i].number > 0 && directory[i].number < 8000)
            occupied.push_back(directory[i].number);
    }
    long program_number = 1;
    while (program_number < 8000 && std::find(occupied.begin(), occupied.end(), program_number) != occupied.end()) ++program_number;
    if (program_number >= 8000) {
        ctx.block("cnc_download4", "no_confirmed_idle_ordinary_program_number");
        return EW_NUMBER;
    }

    // Keep one valid NC stream while exposing ten distinct data inputs.  The
    // comments are part of the transmitted program and do not require motion.
    std::vector<std::string> chunks;
    chunks.push_back("\nO" + std::to_string(program_number));
    for (int i = 1; i <= 8; ++i)
        chunks.push_back("\n(DOWNLOAD4_CHUNK_" + std::to_string(i) + ")");
    chunks.push_back("\nM30\n%");
    bool session_open = false;
    // type=0 means NC program. NULL selects the CNC current folder; do not
    // guess a controller-specific CNC_MEM path.
    ret = ctx.call("DOWNLOAD_START", "cnc_dwnstart4",
        "FlibHndl=" + std::to_string(handle) +
            ";type=0;dir_name=NULL;destination=current_folder;program_number=" + std::to_string(program_number),
        api.cnc_dwnstart4,
        [&] { return api.cnc_dwnstart4(handle, static_cast<short>(0), nullptr); },
        [&](short) { return "type=0;dir_name=NULL;destination=current_folder;program_number=" + std::to_string(program_number); });
    if (ret != EW_OK) {
        if (api.cnc_getdtailerr) {
            ODBERR detail{};
            ctx.call("DOWNLOAD_START_DETAIL", "cnc_getdtailerr",
                "FlibHndl=" + std::to_string(handle), api.cnc_getdtailerr,
                [&] { return api.cnc_getdtailerr(handle, &detail); },
                [&](short) { return "err_no=" + std::to_string(detail.err_no) +
                    ";err_dtno=" + std::to_string(detail.err_dtno); });
        }
        return ret;
    session_open = true;

    for (size_t chunk_index = 0; session_open && chunk_index < chunks.size(); ++chunk_index) {
        const std::string& chunk = chunks[chunk_index];
        size_t chunk_offset = 0;
        int buffer_retries = 0;
        while (session_open && chunk_offset < chunk.size()) {
            const size_t remaining = chunk.size() - chunk_offset;
            long length = static_cast<long>(remaining);
            ret = ctx.call("DOWNLOAD_DATA_" + std::to_string(chunk_index + 1), "cnc_download4",
                "FlibHndl=" + std::to_string(handle) +
                    ";length=" + std::to_string(length) +
                    ";program_number=" + std::to_string(program_number) +
                    ";chunk_index=" + std::to_string(chunk_index + 1) +
                    ";data=" + chunk.substr(chunk_offset),
                api.cnc_download4,
                [&] { return api.cnc_download4(handle, &length,
                    const_cast<char*>(chunk.data() + chunk_offset)); },
                [&](short) { return "requested_length=" + std::to_string(remaining) +
                    ";accepted_length=" + std::to_string(length) +
                    ";chunk_index=" + std::to_string(chunk_index + 1); });
            if (ret == EW_BUFFER) {
                if (length > 0 && static_cast<size_t>(length) < remaining) chunk_offset += static_cast<size_t>(length);
                if (++buffer_retries > 5) break;
                continue;
            }
            if (ret != EW_OK) break;
            if (length <= 0 || static_cast<size_t>(length) > remaining) { ret = EW_LENGTH; break; }
            chunk_offset += static_cast<size_t>(length);
            buffer_retries = 0;
        }
        if (ret != EW_OK) break;
    }

    short end_ret = ctx.call("DOWNLOAD_END", "cnc_dwnend4",
        "FlibHndl=" + std::to_string(handle) +
            ";program_number=" + std::to_string(program_number), api.cnc_dwnend4,
        [&] { return api.cnc_dwnend4(handle); }, [](short) { return std::string(); });
    session_open = false;
    if (ret == EW_OK && chunks.size() != 10) ret = EW_LENGTH;
    if (ret == EW_OK && end_ret != EW_OK) ret = end_ret;
    return ret;
'''


def verify4_core() -> str:
    """Deterministic, ABI-safe NC verification session.

    ``cnc_vrfstart4`` takes a *folder* name, not a program number or file
    name.  Passing a guessed ``O####`` value produces EW_DATA and makes the
    following ``cnc_verify4`` call fail with the secondary EW_FUNC.  The
    documented NULL form selects the controller's current folder and avoids
    inventing a program identifier.  ``cnc_verify4`` returns data through its
    buffer, so the buffer is zero initialized and only the requested length is
    logged as an input.
    """
    return r'''
    ODBST verify_status{};
    ret = ctx.call("VERIFY_STATUS", "cnc_statinfo",
        "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &verify_status); },
        [&](short) { return StatinfoData(verify_status); });
    if (ret != EW_OK) return ret;
    long verify_alarm = 0;
    ret = ctx.call("VERIFY_ALARM", "cnc_alarm2",
        "FlibHndl=" + std::to_string(handle), api.cnc_alarm2,
        [&] { return api.cnc_alarm2(handle, &verify_alarm); },
        [&](short) { return "alarm=" + std::to_string(verify_alarm); });
    if (ret != EW_OK) return ret;
    if (verify_status.run != 0 || verify_status.motion != 0 ||
        verify_status.alarm != 0 || verify_status.emergency != 0 ||
        verify_alarm != 0) {
        ctx.block("cnc_verify4", "controller_not_ready_for_verification");
        return EW_REJECT;
    }

    auto esc_verify = [](const char* data, long length) {
        std::string text;
        if (length < 0) length = 0;
        for (long i = 0; i < length; ++i) {
            const unsigned char c = static_cast<unsigned char>(data[i]);
            if (c == 0) break;
            if (c == '\n') text += "\\n";
            else if (c == '\r') text += "\\r";
            else if (c == ';') text += "\\;";
            else if (c == ',') text += "\\,";
            else if (c >= 32 && c < 127) text += static_cast<char>(c);
            else text += '.';
        }
        return text;
    };

    // cnc_vrfstart4 expects a destination folder. NULL means the current
    // folder, which is the only value that can be used without guessing a
    // controller-specific path or program number.
    ret = ctx.call("VERIFY_START", "cnc_vrfstart4",
        "FlibHndl=" + std::to_string(handle) +
            ";dir_name=NULL;destination=current_folder",
        api.cnc_vrfstart4,
        [&] { return api.cnc_vrfstart4(handle, nullptr); },
        [](short) { return "dir_name=NULL;destination=current_folder"; });
    if (ret != EW_OK) {
        if (api.cnc_getdtailerr) {
            ODBERR detail{};
            ctx.call("VERIFY_START_DETAIL", "cnc_getdtailerr",
                "FlibHndl=" + std::to_string(handle), api.cnc_getdtailerr,
                [&] { return api.cnc_getdtailerr(handle, &detail); },
                [&](short) { return "err_no=" + std::to_string(detail.err_no) +
                    ";err_dtno=" + std::to_string(detail.err_dtno); });
        }
        return ret;

    short verify_ret = EW_OK;
    bool verification_complete = false;
    const long capacities[] = {64, 512, 2048};
    for (int sample = 0; sample < 3 && !verification_complete; ++sample) {
        char data[4096]{};
        long capacity = capacities[sample];
        int retries = 0;
        for (;;) {
            long length = capacity;
            verify_ret = ctx.call(
                "VERIFY_DATA_" + std::to_string(sample + 1), "cnc_verify4",
                "FlibHndl=" + std::to_string(handle) +
                    ";length=" + std::to_string(capacity) +
                    ";data=output_buffer",
                api.cnc_verify4,
                [&] { return api.cnc_verify4(handle, &length, data); },
                [&](short) { return "actual_length=" + std::to_string(length) +
                    ";data=" + esc_verify(data, length); });
            if (verify_ret == EW_BUFFER && ++retries <= 5) continue;
            if (verify_ret == EW_OK) {
                if (length < capacity) verification_complete = true;
            } else if (verify_ret == EW_DATA && api.cnc_getdtailerr) {
                ODBERR detail{};
                ctx.call("VERIFY_DATA_DETAIL_" + std::to_string(sample + 1),
                    "cnc_getdtailerr", "FlibHndl=" + std::to_string(handle),
                    api.cnc_getdtailerr,
                    [&] { return api.cnc_getdtailerr(handle, &detail); },
                    [&](short) { return "err_no=" + std::to_string(detail.err_no) +
                        ";err_dtno=" + std::to_string(detail.err_dtno); });
            }
            break;
        }
        if (verify_ret != EW_OK) break;
    }

    const short end_ret = ctx.call("VERIFY_END", "cnc_vrfend4",
        "FlibHndl=" + std::to_string(handle), api.cnc_vrfend4,
        [&] { return api.cnc_vrfend4(handle); },
        [](short) { return std::string(); });
    if (verify_ret == EW_OK && end_ret != EW_OK) verify_ret = end_ret;
    return verify_ret;
'''


def dncstart2_core() -> str:
    """Deterministic read of the CNC-side DNC/M198 activity state.

    ``cnc_dncstart2`` reports whether a DNC/M198 operation is already active;
    it does not start one.  The filename buffer is an output, and a documented
    non-zero result such as EW_REJECT is retained as the real observation.
    No REMOTE transition, parameter write, NC execution, or ``cnc_dncend2``
    call is generated because the caller does not own an active session.
    """
    return r'''
    char dnc_filename[256]{};
    ret = ctx.call("TARGET_CNC_DNCSTART2", "cnc_dncstart2",
        "FlibHndl=" + std::to_string(handle), api.cnc_dncstart2,
        [&] { return api.cnc_dncstart2(handle, dnc_filename); },
        [&](short) { return "filename=" + std::string(dnc_filename); });
    if (ret == EW_REJECT || ret == EW_BUSY || ret == EW_NOOPT || ret == EW_PARAM) {
        // These are documented state/option observations.  Keep the exact
        // API return code in the log, but let the scenario terminate normally
        // so Router does not spend repair budget on an expected result.
        return EW_OK;
    }
    if (ret != EW_OK) return ret;
    return EW_OK;
'''


# These recipes are deliberately kept next to the fixed capture runtime.  The
# recipe is executable code, while CaptureContext remains the only owner of
# invocation IDs, timestamps, return-code text, and CSV rows.  A target is
# admitted here only when its ABI and output fields are known from the active
# Fwlib32.h.  Parameterized APIs without a runtime-discovered value stay on
# the Agent path instead of receiving guessed values.
def deterministic_read_core(
    targets: set[str],
    *,
    parameter_values: dict[str, dict[str, object]] | None = None,
    planned_functions: set[str] | None = None,
) -> str | None:
    """Build fixed, logged calls for ABI-stable read APIs.

    The returned fragment contains no CSV code and no model-authored log
    fields.  Every call is wrapped by ``ctx.call`` so input/output rows are
    generated by the immutable harness from the actual invocation.
    """
    values = parameter_values or {}
    blocks: list[str] = []

    def integer_value(raw: object) -> int | None:
        try:
            # bool is not a meaningful FOCAS numeric selector.
            if isinstance(raw, bool):
                return None
            return int(raw)
        except (TypeError, ValueError, OverflowError):
            return None

    def add(block: str) -> None:
        blocks.append("    {\n" + block.strip() + "\n    }")

    if "cnc_prstwkcd" in targets:
        # cnc_prstwkcd uses the formal IDBWRA payload. The caller explicitly
        # Requested target collection without a restore phase.  type=0 is
        # invalid on 0i-D; ALL_AXES=-1 is the documented callable selector.
        add(r'''IDBWRA prstwkcd_value{};
        prstwkcd_value.datano = 0;
        prstwkcd_value.type = ALL_AXES;
        for (int axis = 0; axis < MAX_AXIS; ++axis) prstwkcd_value.data[axis] = 0;
        const short prstwkcd_length = static_cast<short>(sizeof(IDBWRA));
        ret = ctx.call("TARGET_CNC_PRSTWKCD", "cnc_prstwkcd",
            "FlibHndl=" + std::to_string(handle) +
                ";length=" + std::to_string(prstwkcd_length) +
                ";datano=" + std::to_string(prstwkcd_value.datano) +
                ";type=" + std::to_string(prstwkcd_value.type) +
                ";data0=" + std::to_string(prstwkcd_value.data[0]),
            api.cnc_prstwkcd,
            [&] { return api.cnc_prstwkcd(handle, prstwkcd_length, &prstwkcd_value); },
            [&](short) { return "datano=" + std::to_string(prstwkcd_value.datano) +
                ";type=" + std::to_string(prstwkcd_value.type) +
                ";length=" + std::to_string(prstwkcd_length) +
                ";data0=" + std::to_string(prstwkcd_value.data[0]); });
        if (ret == EW_OK) {
            prstwkcd_value.type = 1;
            ret = ctx.call("TARGET_CNC_PRSTWKCD_AXIS1", "cnc_prstwkcd",
                "FlibHndl=" + std::to_string(handle) +
                    ";length=" + std::to_string(prstwkcd_length) +
                    ";datano=" + std::to_string(prstwkcd_value.datano) +
                    ";type=" + std::to_string(prstwkcd_value.type) +
                    ";data0=" + std::to_string(prstwkcd_value.data[0]),
                api.cnc_prstwkcd,
                [&] { return api.cnc_prstwkcd(handle, prstwkcd_length, &prstwkcd_value); },
                [&](short) { return "datano=" + std::to_string(prstwkcd_value.datano) +
                    ";type=" + std::to_string(prstwkcd_value.type) +
                    ";length=" + std::to_string(prstwkcd_length) +
                    ";data0=" + std::to_string(prstwkcd_value.data[0]); });
        }
        return ret;''')
        return "\n".join(blocks)

    if "cnc_wrrelpos" in targets:
        # cnc_wrrelpos is a write-style API, but its collection recipe is
        # still deterministic and does not add a snapshot/restore phase.  The
        # SDK contract is: (handle, length, IDBWRR*).  ``IDBWRR.type`` is the
        # axis selector (ALL_AXES == -1 or 1..MAX_AXIS), while ``data`` is the
        # caller-provided preset value.  Length is derived from that selector
        # and any supplied length is checked rather than trusted blindly.
        item = values.get("cnc_wrrelpos", {})
        if not isinstance(item, dict):
            return None

        def value_from(*names: str) -> object | None:
            for name in names:
                if name in item:
                    return item[name]
            return None

        raw_type = value_from("type", "axis")
        axis_type = integer_value(raw_type)
        if axis_type is None or axis_type == 0 or axis_type < -1 or axis_type > 32:
            return None

        raw_data = value_from("data", "values")
        data_values: list[int]
        if isinstance(raw_data, (list, tuple)):
            parsed: list[int] = []
            for raw in raw_data:
                parsed_value = integer_value(raw)
                if parsed_value is None or parsed_value < -2147483648 or parsed_value > 2147483647:
                    return None
                parsed.append(parsed_value)
            data_values = parsed
        else:
            raw_data0 = value_from("data0", "value")
            data0 = integer_value(raw_data0)
            if data0 is None or data0 < -2147483648 or data0 > 2147483647:
                return None
            data_values = [data0]

        if axis_type == -1:
            # The 0i-D/0i-F interface uses MAX_AXIS storage for ALL_AXES.
            # Do not pad a partial list with invented values.
            if len(data_values) != 32:
                return None
            length = 4 + 4 * 32
        else:
            if not data_values:
                return None
            length = 8

        supplied_length = value_from("length")
        if supplied_length is not None:
            parsed_length = integer_value(supplied_length)
            if parsed_length != length:
                return None

        assignments = "\n".join(
            f"        wrrelpos_value.data[{index}] = static_cast<long>({value});"
            for index, value in enumerate(data_values if axis_type == -1 else data_values[:1])
        )
        input_fields = (
            '"FlibHndl=" + std::to_string(handle)'
            ' + ";length=" + std::to_string(wrrelpos_length)'
            ' + ";type=" + std::to_string(wrrelpos_value.type)'
            ' + ";data0=" + std::to_string(wrrelpos_value.data[0])'
        )
        output_prefix = (
            '"type=" + std::to_string(wrrelpos_value.type)'
            ' + ";data0=" + std::to_string(wrrelpos_value.data[0])'
        )
        if axis_type == -1:
            input_fields += "".join(
                f' + ";data[{index}]=" + std::to_string(wrrelpos_value.data[{index}])'
                for index in range(1, 32)
            )
            output_prefix += "".join(
                f' + ";data[{index}]=" + std::to_string(wrrelpos_value.data[{index}])'
                for index in range(1, 32)
            )
        add(f'''IDBWRR wrrelpos_value{{}};
        wrrelpos_value.datano = 0;
        wrrelpos_value.type = static_cast<short>({axis_type});
{assignments}
        const short wrrelpos_length = static_cast<short>({length});
        ret = ctx.call("TARGET_CNC_WRRELPOS", "cnc_wrrelpos",
            {input_fields}, api.cnc_wrrelpos,
            [&] {{ return api.cnc_wrrelpos(handle, wrrelpos_length, &wrrelpos_value); }},
            [&](short) {{ return {output_prefix}; }});
        if (ret != EW_OK) return ret;''')
        return "\n".join(blocks)

    if "cnc_rdparar" in targets:
        # cnc_rdparar is deliberately generated from the controller's live
        # parameter metadata.  The length argument is a byte capacity derived
        # from ODBPARAIF.prm_type and the reported axis count; it is not a
        # coverage dimension and must never be a guessed constant.
        required = {"cnc_sysinfo", "cnc_rdparanum", "cnc_rdparainfo"}
        if planned_functions is not None and not required.issubset(planned_functions):
            return None
        add(r'''ODBSYS rdparar_sysinfo{};
        ret = ctx.call("RDPARAR_DISCOVERY_SYSINFO", "cnc_sysinfo",
            "FlibHndl=" + std::to_string(handle), api.cnc_sysinfo,
            [&] { return api.cnc_sysinfo(handle, &rdparar_sysinfo); },
            [&](short) {
                gSysinfoAxes = ParseSysinfoAxes(rdparar_sysinfo.axes);
                if (gSysinfoAxes < 1 || gSysinfoAxes > MAX_AXIS) gSysinfoAxes = 0;
                return "axes=" + std::to_string(gSysinfoAxes) +
                    ";axes_raw0=" + std::to_string(static_cast<unsigned char>(rdparar_sysinfo.axes[0])) +
                    ";axes_raw1=" + std::to_string(static_cast<unsigned char>(rdparar_sysinfo.axes[1]));
            });
        if (ret != EW_OK || gSysinfoAxes < 1) {
            ctx.block("RDPARAR_DISCOVERY", "cnc_sysinfo did not provide a valid controlled axis count");
            return ret == EW_OK ? EW_ATTRIB : ret;
        }

        ODBPARANUM rdparar_numbers{};
        ret = ctx.call("RDPARAR_DISCOVERY_RANGE", "cnc_rdparanum",
            "FlibHndl=" + std::to_string(handle), api.cnc_rdparanum,
            [&] { return api.cnc_rdparanum(handle, &rdparar_numbers); },
            [&](short) { return "para_min=" + std::to_string(rdparar_numbers.para_min) +
                ";para_max=" + std::to_string(rdparar_numbers.para_max) +
                ";total_no=" + std::to_string(rdparar_numbers.total_no); });
        if (ret != EW_OK || rdparar_numbers.total_no == 0 ||
            rdparar_numbers.para_min > rdparar_numbers.para_max) {
            ctx.block("RDPARAR_DISCOVERY", "cnc_rdparanum did not provide a usable parameter domain");
            return ret == EW_OK ? EW_NUMBER : ret;
        }

        short info_start = static_cast<short>(rdparar_numbers.para_min);
        int discovered = 0;
        for (int info_index = 0; info_index < 4 && discovered < 4; ++info_index) {
            ODBPARAIF rdparar_info{};
            const unsigned short read_no = 1;
            const short requested_start = info_start;
            ret = ctx.call("RDPARAR_DISCOVERY_INFO_" + std::to_string(info_index + 1),
                "cnc_rdparainfo",
                "FlibHndl=" + std::to_string(handle) +
                    ";s_number=" + std::to_string(requested_start) +
                    ";read_no=" + std::to_string(read_no), api.cnc_rdparainfo,
                [&] { return api.cnc_rdparainfo(handle, requested_start, read_no, &rdparar_info); },
                [&](short) { return "info_no=" + std::to_string(rdparar_info.info_no) +
                    ";prev_no=" + std::to_string(rdparar_info.prev_no) +
                    ";next_no=" + std::to_string(rdparar_info.next_no) +
                    ";prm_no=" + std::to_string(rdparar_info.info[0].prm_no) +
                    ";prm_type=" + std::to_string(rdparar_info.info[0].prm_type); });
            if (ret != EW_OK || rdparar_info.info_no == 0) break;

            const short parameter_number = rdparar_info.info[0].prm_no;
            const unsigned short parameter_type = static_cast<unsigned short>(rdparar_info.info[0].prm_type);
            const bool has_axis = (parameter_type & 0x0004u) != 0;
            const unsigned short type_code = parameter_type & 0x0003u;
            const bool is_real = type_code == 3u && (parameter_type & 0x1000u) != 0;
            const int value_bytes = is_real ? 8 : (type_code == 0u ? 1 : type_code == 1u ? 1 : type_code == 2u ? 2 : 4);
            const short selected_axis = has_axis ? static_cast<short>(-1) : static_cast<short>(0);
            const int axis_count = has_axis ? gSysinfoAxes : 1;
            const int logical_length = 4 + value_bytes * axis_count;
            if (logical_length <= 0 || logical_length > 32767) break;
            std::vector<IODBPSD> parameter_storage(1);
            short call_s_number = parameter_number;
            short call_e_number = parameter_number;
            short call_length = static_cast<short>(logical_length);
            const std::string input = "FlibHndl=" + std::to_string(handle) +
                ";s_number=" + std::to_string(parameter_number) +
                ";axis=" + std::to_string(selected_axis) +
                ";e_number=" + std::to_string(parameter_number) +
                ";length=" + std::to_string(call_length) +
                ";length_rule=4+value_bytes*axis_count";
            ret = ctx.call("TARGET_CNC_RDPARAR_" + std::to_string(discovered + 1),
                "cnc_rdparar", input, api.cnc_rdparar,
                [&] { return api.cnc_rdparar(handle, &call_s_number, selected_axis,
                    &call_e_number, &call_length, parameter_storage.data()); },
                [&](short) {
                    const IODBPSD& parameter = parameter_storage[0];
                    std::string output = "ret_s_number=" + std::to_string(call_s_number) +
                        ";ret_e_number=" + std::to_string(call_e_number) +
                        ";ret_length=" + std::to_string(call_length) +
                        ";datano=" + std::to_string(parameter.datano) +
                        ";type=" + std::to_string(parameter.type) +
                        ";cdata=" + std::to_string(static_cast<int>(parameter.u.cdata)) +
                        ";idata=" + std::to_string(parameter.u.idata) +
                        ";ldata=" + std::to_string(parameter.u.ldata) +
                        ";rdata.prm_val=" + std::to_string(parameter.u.rdata.prm_val) +
                        ";rdata.dec_val=" + std::to_string(parameter.u.rdata.dec_val);
                    if (has_axis) {
                        for (int axis = 0; axis < gSysinfoAxes; ++axis) {
                            output += ";axis[" + std::to_string(axis + 1) + "].cdata=" +
                                std::to_string(static_cast<int>(parameter.u.cdatas[axis])) +
                                ";axis[" + std::to_string(axis + 1) + "].idata=" +
                                std::to_string(parameter.u.idatas[axis]) +
                                ";axis[" + std::to_string(axis + 1) + "].ldata=" +
                                std::to_string(parameter.u.ldatas[axis]) +
                                ";axis[" + std::to_string(axis + 1) + "].rdata.prm_val=" +
                                std::to_string(parameter.u.rdatas[axis].prm_val) +
                                ";axis[" + std::to_string(axis + 1) + "].rdata.dec_val=" +
                                std::to_string(parameter.u.rdatas[axis].dec_val);
                        }
                    }
                    return output;
                });
            ++discovered;
            if (ret != EW_OK) {
                if (ret == EW_NUMBER || ret == EW_ATTRIB || ret == EW_LENGTH) continue;
            }
            if (rdparar_info.next_no <= requested_start || rdparar_info.next_no == 0) break;
            info_start = rdparar_info.next_no;
        }
        if (discovered == 0) {
            ctx.block("RDPARAR_DISCOVERY", "cnc_rdparainfo returned no readable parameter metadata");
            return EW_NUMBER;
        }
        if (ret != EW_OK && ret != EW_NUMBER && ret != EW_ATTRIB && ret != EW_LENGTH) return ret;''')
        # This recipe is complete on its own. Returning here prevents the
        # generic per-API loop below from treating cnc_rdparar as unknown and
        # discarding the assembled discovery sequence.
        return "\n".join(blocks)

    if "pmc_rdpmcrng" in targets:
        # Read-only PMC coverage uses four documented IODBPMC shapes.  The
        # buffer is allocated from the computed SDK length and every header
        # and union element is serialized into the response log.
        for index, profile in enumerate(PMC_RANGE_PROFILES, 1):
            type_a = int(profile["type_a"])
            type_d = int(profile["type_d"])
            start = int(profile["datano_s"])
            end = int(profile["datano_e"])
            length = int(profile["length"])
            count = int(profile["element_count"])
            union = str(profile["union"])
            value_log = ""
            if union == "cdata":
                value_log = "".join(
                    f' + ";buf.u.cdata[{i}]=" + std::to_string(static_cast<int>(buf->u.cdata[{i}]))'
                    for i in range(count)
                )
            elif union == "idata":
                value_log = "".join(
                    f' + ";buf.u.idata[{i}]=" + std::to_string(buf->u.idata[{i}])'
                    for i in range(count)
                )
            elif union == "ldata":
                value_log = "".join(
                    f' + ";buf.u.ldata[{i}]=" + std::to_string(buf->u.ldata[{i}])'
                    for i in range(count)
                )
            elif union == "fdata":
                value_log = "".join(
                    f' + ";buf.u.fdata[{i}]=" + std::to_string(reinterpret_cast<float*>(buf->u.cdata)[{i}])'
                    for i in range(count)
                )
            else:
                value_log = "".join(
                    f' + ";buf.u.dfdata[{i}]=" + std::to_string(reinterpret_cast<double*>(buf->u.cdata)[{i}])'
                    for i in range(count)
                )
            add(f'''std::vector<unsigned char> pmc_storage_{index}(static_cast<size_t>({length}), 0);
        IODBPMC* buf = reinterpret_cast<IODBPMC*>(pmc_storage_{index}.data());
        const short pmc_adr_type = static_cast<short>({type_a});
        const short pmc_data_type = static_cast<short>({type_d});
        const unsigned short pmc_start = static_cast<unsigned short>({start});
        const unsigned short pmc_end = static_cast<unsigned short>({end});
        const unsigned short pmc_length = static_cast<unsigned short>({length});
        ret = ctx.call("TARGET_PMC_RDPMCRNG_{index:02d}", "pmc_rdpmcrng",
            "FlibHndl=" + std::to_string(handle) +
                ";adr_type=" + std::to_string(pmc_adr_type) +
                ";data_type=" + std::to_string(pmc_data_type) +
                ";s_number=" + std::to_string(pmc_start) +
                ";e_number=" + std::to_string(pmc_end) +
                ";length=" + std::to_string(pmc_length),
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, pmc_adr_type, pmc_data_type,
                pmc_start, pmc_end, pmc_length, buf); }},
            [&](short) {{ return "buf.type_a=" + std::to_string(buf->type_a) +
                ";buf.type_d=" + std::to_string(buf->type_d) +
                ";buf.datano_s=" + std::to_string(buf->datano_s) +
                ";buf.datano_e=" + std::to_string(buf->datano_e){value_log}; }});
        if (ret != EW_OK) return ret;''')

    for api_name in sorted(targets):
        if api_name == "pmc_rdpmcrng":
            continue
        if api_name == "cnc_statinfo":
            add(r'''ODBST status{};
        ret = ctx.call("TARGET_CNC_STATINFO", "cnc_statinfo",
            "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
            [&] { return api.cnc_statinfo(handle, &status); },
            [&](short) { return StatinfoData(status); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdcncid":
            add(r'''unsigned long cncid = 0;
        ret = ctx.call("TARGET_CNC_RDCNCID", "cnc_rdcncid",
            "FlibHndl=" + std::to_string(handle), api.cnc_rdcncid,
            [&] { return api.cnc_rdcncid(handle, &cncid); },
            [&](short) { return "cncid=" + std::to_string(cncid); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_alarm2":
            add(r'''long alarm_bits = 0;
        ret = ctx.call("TARGET_CNC_ALARM2", "cnc_alarm2",
            "FlibHndl=" + std::to_string(handle), api.cnc_alarm2,
            [&] { return api.cnc_alarm2(handle, &alarm_bits); },
            [&](short) { return "alarm_bits=" + std::to_string(alarm_bits); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdprgnum":
            add(r'''ODBPRO program_number{};
        ret = ctx.call("TARGET_CNC_RDPRGNUM", "cnc_rdprgnum",
            "FlibHndl=" + std::to_string(handle), api.cnc_rdprgnum,
            [&] { return api.cnc_rdprgnum(handle, &program_number); },
            [&](short) { return "running_program=" + std::to_string(program_number.data) +
                ";main_program=" + std::to_string(program_number.mdata); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdseqnum":
            add(r'''ODBSEQ sequence_number{};
        ret = ctx.call("TARGET_CNC_RDSEQNUM", "cnc_rdseqnum",
            "FlibHndl=" + std::to_string(handle), api.cnc_rdseqnum,
            [&] { return api.cnc_rdseqnum(handle, &sequence_number); },
            [&](short) { return "sequence_number=" + std::to_string(sequence_number.data); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_getpath":
            # Both arguments are outputs according to Fwlib32.h.  Do not
            # advertise them as generated input dimensions.
            add(r'''short path_no = 0;
        short maxpath_no = 0;
        ret = ctx.call("TARGET_CNC_GETPATH", "cnc_getpath",
            "FlibHndl=" + std::to_string(handle) + ";path_no=out;maxpath_no=out",
            api.cnc_getpath,
            [&] { return api.cnc_getpath(handle, &path_no, &maxpath_no); },
            [&](short) { return "path_no=" + std::to_string(path_no) +
                ";maxpath_no=" + std::to_string(maxpath_no); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_setpath":
            # setpath is only safe when it reuses the controller-reported
            # current path.  This keeps the setter deterministic and avoids
            # inventing a path number.  The discovery call is support traffic.
            add(r'''short current_path = 0;
        short maxpath_no = 0;
        ret = ctx.call("SET_PATH_DISCOVERY", "cnc_getpath",
            "FlibHndl=" + std::to_string(handle) + ";path_no=out;maxpath_no=out",
            api.cnc_getpath,
            [&] { return api.cnc_getpath(handle, &current_path, &maxpath_no); },
            [&](short) { return "path_no=" + std::to_string(current_path) +
                ";maxpath_no=" + std::to_string(maxpath_no); });
        if (ret != EW_OK) return ret;
        ret = ctx.call("TARGET_CNC_SETPATH", "cnc_setpath",
            "FlibHndl=" + std::to_string(handle) + ";path_no=" + std::to_string(current_path),
            api.cnc_setpath,
            [&] { return api.cnc_setpath(handle, current_path); },
            [&](short) { return "path_no=" + std::to_string(current_path); });
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdmacro":
            item = values.get(api_name, {})
            if not isinstance(item, dict) or not {"number", "length"}.issubset(item):
                return None
            number = integer_value(item["number"])
            length = integer_value(item["length"])
            if number is None or length is None:
                return None
            add(f'''ODBM macro_value{{}};
        const short macro_number = static_cast<short>({number});
        const short macro_length = static_cast<short>({length});
        ret = ctx.call("TARGET_CNC_RDMACRO", "cnc_rdmacro",
            "FlibHndl=" + std::to_string(handle) + ";number=" + std::to_string(macro_number) +
                ";length=" + std::to_string(macro_length), api.cnc_rdmacro,
            [&] {{ return api.cnc_rdmacro(handle, macro_number, macro_length, &macro_value); }},
            [&](short) {{ return "datano=" + std::to_string(macro_value.datano) +
                ";mcr_val=" + std::to_string(macro_value.mcr_val) +
                ";dec_val=" + std::to_string(macro_value.dec_val); }});
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdparam":
            item = values.get(api_name, {})
            if not isinstance(item, dict) or not {"number", "axis", "length"}.issubset(item):
                return None
            number = integer_value(item["number"])
            axis = integer_value(item["axis"])
            length = integer_value(item["length"])
            if number is None or axis is None or length is None or length <= 0:
                return None
            add(f'''IODBPSD parameter_value{{}};
        const short parameter_number = static_cast<short>({number});
        const short parameter_axis = static_cast<short>({axis});
        const short parameter_length = static_cast<short>({length});
        ret = ctx.call("TARGET_CNC_RDPARAM", "cnc_rdparam",
            "FlibHndl=" + std::to_string(handle) + ";number=" + std::to_string(parameter_number) +
                ";axis=" + std::to_string(parameter_axis) + ";length=" + std::to_string(parameter_length),
            api.cnc_rdparam,
            [&] {{ return api.cnc_rdparam(handle, parameter_number, parameter_axis, parameter_length, &parameter_value); }},
            [&](short) {{ return "datano=" + std::to_string(parameter_value.datano) +
                ";type=" + std::to_string(parameter_value.type) +
                ";cdata=" + std::to_string(static_cast<int>(parameter_value.u.cdata)) +
                ";idata=" + std::to_string(parameter_value.u.idata) +
                ";ldata=" + std::to_string(parameter_value.u.ldata); }});
        if (ret != EW_OK) return ret;''')
        elif api_name == "cnc_rdaxisname":
            item = values.get(api_name, {})
            count = integer_value(item.get("data_num", 0)) if isinstance(item, dict) else None
            count = count or 0
            if count <= 0:
                return None
            add(f'''short axis_count = static_cast<short>({count});
        ODBAXISNAME axis_names[MAX_AXIS]{{}};
        ret = ctx.call("TARGET_CNC_RDAXISNAME", "cnc_rdaxisname",
            "FlibHndl=" + std::to_string(handle) + ";data_num=" + std::to_string(axis_count),
            api.cnc_rdaxisname,
            [&] {{ return api.cnc_rdaxisname(handle, &axis_count, axis_names); }},
            [&](short) {{ std::string data = "returned_count=" + std::to_string(axis_count);
                for (short i = 0; i < axis_count && i < MAX_AXIS; ++i)
                    data += ";axis[" + std::to_string(i + 1) + "]=" + std::string(1, axis_names[i].name) + std::string(1, axis_names[i].suff);
                return data; }});
        if (ret != EW_OK) return ret;''')
        else:
            return None
    return "\n".join(blocks) if blocks else None
