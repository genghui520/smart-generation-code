// Fixed motion driver. The fragment is a single sampling pass, never a UI driver.
static short read_runtime_status(FocasApi& api, unsigned short handle, CaptureContext& ctx, ODBST& status) {
    return ctx.call("STATUS", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &status); }, [&](short) {
            return "aut=" + std::to_string(status.aut) + ";run=" + std::to_string(status.run) +
                ";motion=" + std::to_string(status.motion) + ";alarm=" + std::to_string(status.alarm) +
                ";emergency=" + std::to_string(status.emergency);
        });
}
static std::string classify_runtime_alarm_bits(long alarm_bits) {
    std::string classification;
    if (alarm_bits & (1L << 0)) {
        classification += "SW_PARAMETER_WRITE_SWITCH;action=disable_parameter_write_switch_then_reset;automatic_action=not_attempted;";
    }
    if (alarm_bits & (1L << 1)) {
        classification += "PW_POWER_OFF_PARAMETER_SET;action=power_cycle_cnc;automatic_action=not_attempted;";
    }
    if (classification.empty()) {
        classification = "OTHER_OR_MULTIPLE;action=use_alarm_number_and_message_specific_procedure;automatic_action=not_attempted;";
    }
    return classification;
}
static void capture_runtime_alarms(FocasApi& api, unsigned short handle, CaptureContext& ctx) {
    capture_alarm_diagnostics(api, handle, ctx);
}
static bool runtime_ready(const ODBST& status) {
    // For 0i-D/30i ODBST, emergency=2 means ReSET, not an emergency stop.
    // Only 1 is the EMerGency state; preserve the conservative block for
    // undocumented non-zero values.
    bool emergency_stop = status.emergency == 1;
    return status.motion == 0 && (status.run == 0 || status.run == 1) && status.alarm == 0 && !emergency_stop;
}
static short wait_runtime_ready(FocasApi& api, unsigned short handle, CaptureContext& ctx,
                                bool completion = false, bool* recovered = nullptr) {
    if (recovered) *recovered = false;
    ODBST status{};
    int stable = 0, waited_ms = 0;
    for (; waited_ms <= 15000; waited_ms += 100) {
        short rc = read_runtime_status(api, handle, ctx, status);
        if (rc != EW_OK) return rc;
        if (status.alarm || status.emergency == 1) {
            if (status.alarm && recover_alarm(api, handle, ctx)) {
                if (recovered) *recovered = true;
                stable = 0;
                continue;
            }
            ctx.block("CycleStart", "controller_alarm_or_emergency"); return 12;
        }
        bool ready = runtime_ready(status) && (!completion || status.run == 0 || status.run == 1);
        stable = ready ? stable + 1 : 0;
        if (stable == 3) {
            ctx.host(completion ? "program_completion_gate" : "cycle_start_ready_gate", 0, "EW_OK",
                std::string(completion ? "completed=true;" : "ready=true;") + "waited_ms=" + std::to_string(waited_ms) +
                ";last_run=" + std::to_string(status.run) + ";last_motion=" + std::to_string(status.motion));
            return EW_OK;
        }
        Sleep(100);
    }
    ctx.host(completion ? "program_completion_gate" : "cycle_start_ready_gate", 12,
        completion ? "PROGRAM_COMPLETION_TIMEOUT" : "CYCLE_START_READY_TIMEOUT",
        "completed=false;timeout=true;waited_ms=" + std::to_string(waited_ms) +
        ";last_run=" + std::to_string(status.run) + ";last_motion=" + std::to_string(status.motion));
    return 12;
}
static void runtime_cycle_start() {
    std::string title = env("NCGUIDE_WINDOW_TITLE"), mode = env("NCGUIDE_CLICK_MODE", "screen");
    std::string sx = env("NCGUIDE_CYCLE_START_X"), sy = env("NCGUIDE_CYCLE_START_Y");
    if (title.empty() || sx.empty() || sy.empty()) throw std::runtime_error("CYCLE_START_CONFIGURATION_MISSING");
    HWND window = FindWindowA(nullptr, title.c_str());
    if (!window || !IsWindowVisible(window)) throw std::runtime_error("NCGUIDE_WINDOW_NOT_FOUND");
    ShowWindow(window, SW_RESTORE);
    SetForegroundWindow(window);
    Sleep(200);
    POINT point{std::stoi(sx), std::stoi(sy)};
    if (mode == "client") {
        if (!ClientToScreen(window, &point)) throw std::runtime_error("NCGUIDE_CLIENT_COORDINATE_FAILED");
    } else if (mode == "window") {
        RECT origin{};
        if (!GetWindowRect(window, &origin)) throw std::runtime_error("NCGUIDE_WINDOW_COORDINATE_FAILED");
        point.x += origin.left; point.y += origin.top;
    } else if (mode != "screen") throw std::runtime_error("NCGUIDE_CLICK_MODE_INVALID");
    RECT bounds{};
    if (!GetWindowRect(window, &bounds) || !PtInRect(&bounds, point)) throw std::runtime_error("CYCLE_START_OUTSIDE_WINDOW");
    if (GetForegroundWindow() != window) throw std::runtime_error("NCGUIDE_WINDOW_NOT_FOREGROUND");
    bool positioned = false;
    DWORD cursor_error = 0, read_error = 0;
    BOOL moved = FALSE, read_position = FALSE;
    POINT actual{};
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (GetForegroundWindow() != window) throw std::runtime_error("NCGUIDE_WINDOW_NOT_FOREGROUND");
        SetLastError(ERROR_SUCCESS);
        moved = SetCursorPos(point.x, point.y);
        cursor_error = moved ? ERROR_SUCCESS : GetLastError();
        // Read independently even when moving failed; never report initialized zeros as observations.
        SetLastError(ERROR_SUCCESS);
        read_position = GetCursorPos(&actual);
        read_error = read_position ? ERROR_SUCCESS : GetLastError();
        if (moved && read_position && actual.x == point.x && actual.y == point.y) {
            positioned = true; break;
        }
        Sleep(100);
    }
    if (!positioned) throw std::runtime_error("CYCLE_START_CURSOR_FAILED;win32_error=" + std::to_string(cursor_error) +
        ";attempts=3;mode=" + mode + ";requested_x=" + std::to_string(point.x) + ";requested_y=" + std::to_string(point.y) +
        ";set_cursor_ok=" + std::to_string(moved != FALSE) + ";get_cursor_ok=" + std::to_string(read_position != FALSE) +
        ";get_cursor_error=" + std::to_string(read_error) +
        ";actual_x=" + (read_position ? std::to_string(actual.x) : "unknown") +
        ";actual_y=" + (read_position ? std::to_string(actual.y) : "unknown"));
    HWND hit = WindowFromPoint(point);
    if (GetForegroundWindow() != window || (hit != window && !IsChild(window, hit)))
        throw std::runtime_error("CYCLE_START_TARGET_OBSCURED");
    INPUT events[2]{};
    events[0].type = events[1].type = INPUT_MOUSE;
    events[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    events[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SetLastError(ERROR_SUCCESS);
    UINT sent = SendInput(2, events, sizeof(INPUT));
    DWORD click_error = GetLastError();
    if (sent != 2) {
        // A partial send may have pressed the button. Release only; never retry the click.
        if (sent == 1) SendInput(1, &events[1], sizeof(INPUT));
        throw std::runtime_error("CYCLE_START_CLICK_FAILED;sent=" + std::to_string(sent) +
            ";win32_error=" + std::to_string(click_error));
    }
}
static short runtime_sequence(FocasApi& api, unsigned short handle, CaptureContext& ctx, long& sequence) {
    ODBSEQ value{};
    short rc = ctx.call("SEQUENCE", "cnc_rdseqnum", "FlibHndl=" + std::to_string(handle), api.cnc_rdseqnum,
        [&] { return api.cnc_rdseqnum(handle, &value); }, [&](short) { return "seq=" + std::to_string(value.data); });
    if (rc == EW_OK) sequence = value.data;
    return rc;
}
static void runtime_motion_evidence(FocasApi& api, unsigned short handle, CaptureContext& ctx) {
    ODBACT feed{};
    ctx.call("MOTION_EVIDENCE", "cnc_actf", "FlibHndl=" + std::to_string(handle), api.cnc_actf,
        [&] { return api.cnc_actf(handle, &feed); }, [&](short) { return "feed=" + std::to_string(feed.data); });
    ODBAXIS position{};
    // All-axis length is the ABI byte count, not the number of active axes.
    constexpr short axis_length = 4 + 4 * MAX_AXIS;
    static_assert(sizeof(ODBAXIS) >= axis_length, "ODBAXIS buffer is too small");
    ctx.call("MOTION_EVIDENCE", "cnc_absolute", "FlibHndl=" + std::to_string(handle) + ";axis=-1;length=" + std::to_string(axis_length),
        api.cnc_absolute, [&] { return api.cnc_absolute(handle, -1, axis_length, &position); }, [&](short) {
            return "XYZ=[" + std::to_string(position.data[0]) + "," + std::to_string(position.data[1]) + "," + std::to_string(position.data[2]) + "]";
        });
}
// FANUC 0i-MF PMC-to-CNC signals (NCGuide motion adapter only).
// B-64610CM/01: SBK Gn046.1, MLK Gn044.1.
// B-64605CM/01 p.451: *SP G0008.5 is active low.
// cnc_rdopnlsgnl is a software-panel image, not the Main Panel switch state.
static bool runtime_panel_ready(unsigned char g8, unsigned char g44, unsigned char g46) {
    return (g46 & 0x02) != 0 && (g44 & 0x02) == 0 && (g8 & 0x20) != 0;
}
static short check_runtime_panel(FocasApi& api, unsigned short handle, CaptureContext& ctx) {
    unsigned char values[3]{};
    const unsigned short addresses[3] = {8, 44, 46};
    for (int i = 0; i < 3; ++i) {
        IODBPMC buffer{};
        unsigned short address = addresses[i];
        short rc = ctx.call("PANEL_PMC", "pmc_rdpmcrng", "FlibHndl=" + std::to_string(handle) +
            ";adr_type=0;data_type=0;start=" + std::to_string(address) + ";end=" + std::to_string(address) + ";length=9",
            api.pmc_rdpmcrng, [&] { return api.pmc_rdpmcrng(handle, 0, 0, address, address, 9, &buffer); },
            [&](short) { return "G" + std::to_string(address) + "=" + std::to_string(static_cast<unsigned char>(buffer.u.cdata[0])); });
        if (rc != EW_OK) {
            ctx.block("CycleStart", "PANEL_STATE_UNVERIFIED;PMC_read_failed;address=G" + std::to_string(address));
            return rc;
        }
        if (buffer.type_a != 0 || buffer.type_d != 0 || buffer.datano_s != address || buffer.datano_e != address) {
            ctx.block("CycleStart", "PANEL_STATE_UNVERIFIED;PMC_response_range_mismatch"); return 12;
        }
        values[i] = static_cast<unsigned char>(buffer.u.cdata[0]);
    }
    bool ready = runtime_panel_ready(values[0], values[1], values[2]);
    std::string detail = "source=PMC;G8=" + std::to_string(values[0]) + ";G44=" + std::to_string(values[1]) +
        ";G46=" + std::to_string(values[2]) + ";single_block=" + std::to_string((values[2] & 2) != 0) +
        ";machine_lock=" + std::to_string((values[1] & 2) != 0) + ";feed_hold=" + std::to_string((values[0] & 32) == 0);
    if (!ready) { ctx.block("CycleStart", "PANEL_PREREQUISITE_READBACK;" + detail); return 12; }
    ctx.host("panel_ready_gate", 0, "EW_OK", detail);
    return EW_OK;
}
static short run_motion_capture(FocasApi& api, unsigned short handle, CaptureContext& ctx) {
    short rc = wait_runtime_ready(api, handle, ctx);
    if (rc != EW_OK) return rc;
    ODBST status{};
    rc = read_runtime_status(api, handle, ctx, status);
    if (rc != EW_OK) return rc;
    if (status.aut != 1) { ctx.block("CycleStart", "requires_MEM_mode"); return 12; }
    rc = check_runtime_panel(api, handle, ctx);
    if (rc != EW_OK) return rc;
    const long preferred_program = __PREFERRED_PROGRAM__;
    long selected_program = 0;
    for (long candidate = preferred_program; candidate < std::min<long>(8000, preferred_program + 100); ++candidate) {
        PRGDIR3 directory[1]{};
        long top = candidate; short count = 1;
        rc = ctx.call("DIRECTORY", "cnc_rdprogdir3", "FlibHndl=" + std::to_string(handle) +
            ";type=0;top_prog=" + std::to_string(top) + ";num_prog=1", api.cnc_rdprogdir3,
            [&] { return api.cnc_rdprogdir3(handle, 0, &top, &count, directory); },
            [&](short) { return "count=" + std::to_string(count) + ";program=" + std::to_string(directory[0].number); });
        if (rc != EW_OK) return rc;
        if (count < 0 || count > 1) { ctx.block("UploadProgram", "invalid_directory_count"); return 5; }
        bool exists = count == 1 && directory[0].number == candidate;
        if (!exists) { selected_program = candidate; break; }
    }
    if (!selected_program) { ctx.block("UploadProgram", "no_unused_program_number"); return 5; }
    const std::string body = __NC_BODY__;
    std::string payload = "\nO" + std::to_string(selected_program) + "\n" + body + "%";
    std::vector<char> bytes(payload.begin(), payload.end());
    rc = ctx.call("DOWNLOAD_START", "cnc_dwnstart3", "FlibHndl=" + std::to_string(handle) + ";type=0", api.cnc_dwnstart3,
        [&] { return api.cnc_dwnstart3(handle, 0); }, [](short) { return ""; });
    if (rc != EW_OK) return rc;
    size_t offset = 0;
    auto download_started = std::chrono::steady_clock::now();
    while (offset < bytes.size() && std::chrono::steady_clock::now() - download_started < std::chrono::seconds(15)) {
        long sent = static_cast<long>(bytes.size() - offset);
        long requested = sent;
        rc = ctx.call("DOWNLOAD", "cnc_download3", "FlibHndl=" + std::to_string(handle) +
            ";length=" + std::to_string(sent) + ";data=" + std::string(bytes.data() + offset, sent), api.cnc_download3,
            [&] { return api.cnc_download3(handle, &sent, bytes.data() + offset); },
            [&](short) { return "length=" + std::to_string(sent); });
        if (rc != EW_OK && rc != EW_BUFFER) break;
        if (sent < 0 || sent > requested) { rc = 5; break; }
        offset += static_cast<size_t>(sent);
        if (rc == EW_BUFFER || !sent) Sleep(20);
    }
    short end_rc = ctx.call("DOWNLOAD_END", "cnc_dwnend3", "FlibHndl=" + std::to_string(handle), api.cnc_dwnend3,
        [&] { return api.cnc_dwnend3(handle); }, [](short) { return ""; });
    if ((rc != EW_OK && rc != EW_BUFFER) || offset != bytes.size() || end_rc != EW_OK) {
        ctx.host("host_program_verified", 5, "PROGRAM_NOT_VERIFIED", "program_verified=false;download_failed=true"); return 5;
    }
    rc = ctx.call("SELECT", "cnc_search", "FlibHndl=" + std::to_string(handle) + ";prg_no=" + std::to_string(selected_program),
        api.cnc_search, [&] { return api.cnc_search(handle, static_cast<short>(selected_program)); }, [](short) { return ""; });
    if (rc != EW_OK) return rc;
    ODBPRO program{};
    rc = ctx.call("VERIFY", "cnc_rdprgnum", "FlibHndl=" + std::to_string(handle), api.cnc_rdprgnum,
        [&] { return api.cnc_rdprgnum(handle, &program); }, [&](short) { return "main_program=" + std::to_string(program.mdata); });
    if (rc != EW_OK || program.mdata != selected_program) {
        ctx.host("host_program_verified", 5, "PROGRAM_NOT_VERIFIED", "program_verified=false"); return 5;
    }
    ctx.host("host_program_verified", 0, "EW_OK", "program_verified=true;selected_program_number=" + std::to_string(selected_program));
    std::vector<std::string> segments;
    std::istringstream stream(payload); std::string line;
    while (std::getline(stream, line)) if (!line.empty() && line != "%" && line.front() != '(') segments.push_back(line);
    std::string motion_indices, non_motion_indices;
    for (size_t index = 0; index < segments.size(); ++index) {
        std::string& indices = segments[index].find_first_of("XYZ") != std::string::npos ? motion_indices : non_motion_indices;
        if (!indices.empty()) indices += ",";
        indices += std::to_string(index);
    }
    ctx.host("host_nc_segments", 0, "EW_OK", "expected_nc_segment_count=" + std::to_string(segments.size()) +
        ";motion_segment_indices=[" + motion_indices + "];non_motion_segment_indices=[" + non_motion_indices + "]");
    int clicks = 0; short sample_error = run_generated_scenario(api, handle, ctx);
    runtime_motion_evidence(api, handle, ctx);
    bool sample_enabled = sample_error == EW_OK;
    const int interval = std::max(20, std::min(1000, std::stoi(env("SMP_SAMPLE_INTERVAL_MS", "100"))));
    for (size_t segment = 0; segment < segments.size(); ++segment) {
        bool recovered_before_segment = false;
        rc = wait_runtime_ready(api, handle, ctx, false, &recovered_before_segment);
        if (rc != EW_OK) return rc;
        if (recovered_before_segment) {
            if (segment != 0) {
                ctx.block("CycleStart", "program_position_unverified_after_reset");
                return 12;
            }
            rc = ctx.call("RESELECT_AFTER_RESET", "cnc_search",
                "FlibHndl=" + std::to_string(handle) + ";prg_no=" + std::to_string(selected_program),
                api.cnc_search, [&] { return api.cnc_search(handle, static_cast<short>(selected_program)); },
                [](short) { return ""; });
            if (rc != EW_OK) return rc;
            ODBPRO selected{};
            rc = ctx.call("VERIFY_AFTER_RESET", "cnc_rdprgnum", "FlibHndl=" + std::to_string(handle),
                api.cnc_rdprgnum, [&] { return api.cnc_rdprgnum(handle, &selected); },
                [&](short) { return "main_program=" + std::to_string(selected.mdata); });
            if (rc != EW_OK || selected.mdata != selected_program) {
                ctx.block("CycleStart", "program_selection_unverified_after_reset");
                return 12;
            }
        }
        long initial_sequence = 0;
        rc = runtime_sequence(api, handle, ctx, initial_sequence);
        if (rc != EW_OK) return rc;
        rc = check_runtime_panel(api, handle, ctx);
        if (rc != EW_OK) return rc;
        runtime_cycle_start(); ++clicks;
        ctx.host("host_cycle_start", 0, "EW_OK", "segment_index=" + std::to_string(segment) +
            ";expected_nc_segment_count=" + std::to_string(segments.size()) + ";cycle_start_click_count=" + std::to_string(clicks) +
            ";M30_clicked=" + std::string(segment + 1 == segments.size() ? "true" : "false"));
        bool motion_block = segments[segment].find_first_of("XYZ") != std::string::npos;
        bool observed = false; int stable = 0;
        int elapsed = 0;
        auto segment_started = std::chrono::steady_clock::now();
        for (;;) {
            elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - segment_started).count());
            if (elapsed > 30000) break;
            Sleep(interval);
            rc = read_runtime_status(api, handle, ctx, status);
            if (rc != EW_OK) return rc;
            if (status.alarm || status.emergency == 1) {
                if (status.alarm) recover_alarm(api, handle, ctx);
                ctx.block("CycleStart", "program_interrupted_by_alarm_or_emergency"); return 12;
            }
            long sequence = initial_sequence;
            rc = runtime_sequence(api, handle, ctx, sequence);
            if (rc != EW_OK) return rc;
            observed = observed || status.run == 3 || status.motion != 0 || sequence != initial_sequence;
            runtime_motion_evidence(api, handle, ctx);
            if (sample_enabled && (status.motion != 0 || runtime_ready(status))) {
                short target_rc = run_generated_scenario(api, handle, ctx);
                if (target_rc != EW_OK) { sample_error = target_rc; sample_enabled = false; }
            }
            // A failed target stops sampling, but never abandons a running
            // program or clicks another block while axes are still moving.
            stable = runtime_ready(status) ? stable + 1 : 0;
            if (stable >= 3 && elapsed >= 300 && (observed || !motion_block)) break;
            if (motion_block && !observed && elapsed >= 2000 && runtime_ready(status)) {
                ctx.block("CycleStart", "click_had_no_observable_effect"); return 12;
            }
        }
        if (elapsed > 30000) { ctx.block("CycleStart", "segment_completion_timeout"); return 12; }
    }
    bool recovered_during_completion = false;
    rc = wait_runtime_ready(api, handle, ctx, true, &recovered_during_completion);
    if (rc != EW_OK) return rc;
    if (recovered_during_completion) {
        ctx.block("CycleStart", "program_completion_unverified_after_reset");
        return 12;
    }
    if (sample_enabled) sample_error = run_generated_scenario(api, handle, ctx);
    runtime_motion_evidence(api, handle, ctx);
    return sample_error;
}
