"""Production FOCAS capture runtime with dependency-aware recipes."""
from __future__ import annotations
import importlib.util, importlib.machinery, sys
from pathlib import Path
from .execution_contracts import dependency_closure, validate_api_lifecycle_fragment
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
    if functions & {"cnc_skip", "cnc_srvdelay", "cnc_accdecdly"}:
        functions.add("cnc_rdaxisname")
    return sorted(functions)
def render_capture_harness(functions, *args, **kwargs):
    source = _legacy.render_capture_harness(capture_functions(functions), *args, **kwargs)
    contract = kwargs.get("contract")
    targets = getattr(contract, "targets", ())
    target_set = set(targets)
    require_remote = bool(target_set & {"cnc_dnc2", "cnc_dncstart2", "cnc_dncend2"})
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
    for target in contract.targets:
        # Require an actual ctx.call wrapper and the exact API label used by
        # the CSV logger; a declaration or bare function-pointer reference is
        # not target coverage.
        # Parameter fields are intentionally semicolon-delimited strings
        # (for example ``"FlibHndl=...;axis=..."``).  Do not use ``[^;]*``
        # here: it truncates a legitimate ctx.call before the API pointer and
        # falsely reports target_not_called.  Bound the search by the next
        # ctx.call wrapper instead.
        invocation = re.search(
            rf'ctx\.call\s*\((?:(?!\bctx\.call\s*\().)*?["\']{re.escape(target)}["\']'
            rf'(?:(?!\bctx\.call\s*\().)*?\bapi\.{re.escape(target)}\s*\(',
            fragment,
            re.S,
        )
        if invocation:
            # The legacy checker only recognizes literal ctx.call labels and
            # rejects equivalent std::string/dynamic-step wrappers. Keep the
            # error only when this source-level check cannot pair the API log
            # label with its FOCAS function pointer.
            errors = [
                error for error in errors
                if not (
                    target in error
                    and "no logged invocation" in error.lower()
                )
            ]
        else:
            errors.append(f"BLOCKING: target API {target} has no logged invocation in the scenario fragment.")
    errors.extend(validate_api_lifecycle_fragment(fragment))
    return errors
def motion_read_core(*args, **kwargs): return _legacy.motion_read_core(*args, **kwargs)
def verification_core(*args, **kwargs): return _legacy.verification_core(*args, **kwargs)
def _copyprog_core(samples=30, interval_ms=200):
    return r''' 
    ODBST status{};
    ret = ctx.call("PRECHECK", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &status); }, [&](short) { return "aut=" + std::to_string(status.aut) + ";run=" + std::to_string(status.run) + ";alarm=" + std::to_string(status.alarm) + ";emergency=" + std::to_string(status.emergency); });
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

    The active Fwlib32.h defines ODBAXIS, and the FOCAS API documents
    ALL_AXES=-1 plus a byte length equal to sizeof(ODBAXIS).
    """
    return r'''
    ODBST skip_before{};
    ret = ctx.call("SKIP_STATUS_BEFORE", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &skip_before); },
        [&](short) { return "aut=" + std::to_string(skip_before.aut) + ";run=" + std::to_string(skip_before.run) + ";motion=" + std::to_string(skip_before.motion); });
    if (ret != EW_OK) return ret;
    ODBAXIS skip_position{};
    const short skip_length = static_cast<short>(sizeof(ODBAXIS));
    ret = ctx.call("SKIP_READ_ALL", "cnc_skip", "FlibHndl=" + std::to_string(handle) + ";axis=-1;length=" + std::to_string(skip_length), api.cnc_skip,
        [&] { return api.cnc_skip(handle, static_cast<short>(-1), skip_length, &skip_position); },
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
        const std::string axis_input = "FlibHndl=" + std::to_string(handle) + ";axis=" + std::to_string(selected_axis) + ";length=" + std::to_string(skip_length);
        ret = ctx.call("SKIP_AXIS_" + std::to_string(selected_axis), "cnc_skip", axis_input, api.cnc_skip,
            [&] { return api.cnc_skip(handle, selected_axis, skip_length, &axis_position); },
            [&](short) { return "axis=" + std::to_string(selected_axis) + ";type=" + std::to_string(axis_position.type) + ";data=" + std::to_string(axis_position.data[0]); });
        if (ret != EW_OK) { ok = false; break; }
    }
    ODBST skip_after{};
    ret = ctx.call("SKIP_STATUS_AFTER", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &skip_after); },
        [&](short) { return "aut=" + std::to_string(skip_after.aut) + ";run=" + std::to_string(skip_after.run) + ";motion=" + std::to_string(skip_after.motion); });
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
        [&](short) { return "run=" + std::to_string(download_status.run) +
            ";motion=" + std::to_string(download_status.motion) +
            ";alarm=" + std::to_string(download_status.alarm) +
            ";emergency=" + std::to_string(download_status.emergency); });
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
    }
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
