    ODBST status{};
    ret = ctx.call("PRECHECK", "cnc_statinfo", "FlibHndl=" + std::to_string(handle), api.cnc_statinfo,
        [&] { return api.cnc_statinfo(handle, &status); }, [&](short) {
            return "run=" + std::to_string(status.run) + ";motion=" + std::to_string(status.motion) +
                ";aut=" + std::to_string(status.aut) + ";alarm=" + std::to_string(status.alarm);
        });
    if (ret != EW_OK) return ret;
    if (status.motion || (status.run != 0 && status.run != 1) || status.alarm || status.emergency == 1 || status.aut == 0) {
        ctx.block("cnc_vrfend", "verification_requires_idle_non_MDI_controller"); return 12;
    }
    PRGDIR3 directory[1]{}; long top=1; short count=1;
    ret = ctx.call("DIRECTORY", "cnc_rdprogdir3", "FlibHndl=" + std::to_string(handle) + ";type=0;top_prog=1;num_prog=1",
        api.cnc_rdprogdir3, [&] { return api.cnc_rdprogdir3(handle, 0, &top, &count, directory); },
        [&](short) { return "count=" + std::to_string(count) + ";program=" + std::to_string(directory[0].number); });
    if (ret != EW_OK) return ret;
    if (count != 1 || directory[0].number < 1) { ctx.block("cnc_vrfend", "resident_program_missing"); return 5; }
    long program = directory[0].number;
    ret = ctx.call("UPLOAD_START", "cnc_upstart3", "FlibHndl=" + std::to_string(handle) +
        ";type=0;s_prgno=" + std::to_string(program) + ";e_prgno=" + std::to_string(program), api.cnc_upstart3,
        [&] { return api.cnc_upstart3(handle, 0, program, program); }, [](short) { return ""; });
    if (ret != EW_OK) return ret;
    std::string resident;
    bool complete = false;
    auto upload_started = std::chrono::steady_clock::now();
    while (resident.size() <= 2 * 1024 * 1024 && std::chrono::steady_clock::now() - upload_started < std::chrono::seconds(15)) {
        char bytes[4096]{}; long length = sizeof(bytes);
        ret = ctx.call("UPLOAD", "cnc_upload3", "FlibHndl=" + std::to_string(handle) + ";length=4096", api.cnc_upload3,
            [&] { return api.cnc_upload3(handle, &length, bytes); }, [&](short) {
                return "length=" + std::to_string(length) + ";data=" + std::string(bytes, std::max<long>(0, std::min<long>(length, sizeof(bytes))));
            });
        if (ret == EW_BUFFER) { Sleep(20); continue; }
        if (ret != EW_OK) break;
        if (length < 0 || length > sizeof(bytes)) { ret=5; break; }
        resident.append(bytes, static_cast<size_t>(length));
        size_t last = resident.find_last_not_of(std::string("\r\n \t\0", 5));
        if (last != std::string::npos && resident[last] == '%' && resident.find('O') != std::string::npos) {
            complete = true; break;
        }
        if (!length) Sleep(20);
    }
    short upload_end = ctx.call("UPLOAD_END", "cnc_upend3", "FlibHndl=" + std::to_string(handle), api.cnc_upend3,
        [&] { return api.cnc_upend3(handle); }, [](short) { return ""; });
    if (!complete || ret != EW_OK || upload_end != EW_OK) {
        ctx.block("cnc_vrfend", "resident_program_read_failed"); return ret != EW_OK ? ret : static_cast<short>(5);
    }
    for (int sample = 0; sample < __SAMPLES__; ++sample) {
        ret = ctx.call("VERIFY_START", "cnc_vrfstart", "FlibHndl=" + std::to_string(handle), api.cnc_vrfstart,
            [&] { return api.cnc_vrfstart(handle); }, [](short) { return ""; });
        if (ret != EW_OK) return ret;
        for (size_t offset=0; offset < resident.size(); offset += 256) {
            short length = static_cast<short>(std::min<size_t>(256, resident.size() - offset));
            ret = ctx.call("VERIFY_DATA", "cnc_verify", "FlibHndl=" + std::to_string(handle) +
                ";number=" + std::to_string(length) + ";data=" + resident.substr(offset, length), api.cnc_verify,
                [&] { return api.cnc_verify(handle, &resident[offset], length); }, [](short) { return ""; });
            if (ret != EW_OK) break;
        }
        // End is executed even if the data call failed; preserve both results.
        short end_ret = ctx.call("VERIFY_END", "cnc_vrfend", "FlibHndl=" + std::to_string(handle), api.cnc_vrfend,
            [&] { return api.cnc_vrfend(handle); }, [](short) { return ""; });
        if (ret != EW_OK) return ret;
        if (end_ret != EW_OK) return end_ret;
        Sleep(__INTERVAL__);
    }
