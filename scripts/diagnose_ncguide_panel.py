"""Read status and software-panel images only; never upload, write or start motion."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_traffic_agent.focas_capture_runtime import render_capture_harness, wrap_capture_fragment, validate_capture_fragment
from smart_traffic_agent.agent_tools.cpp_execution import CompileGeneratedCppTool, CompileCppInput, RunGeneratedExecutableTool, RunExecutableInput

root = Path(__file__).resolve().parents[1]
settings = json.loads((root / "runs/desktop/20261015_163231_b9dc8f/settings.json").read_text(encoding="utf-8"))
out = root / "runs/diagnostics" / ("panel_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
out.mkdir(parents=True)
core = r'''
    for (unsigned short start = 0; start < 128; start += 5) {
        IODBPMC buf{};
        unsigned short end = static_cast<unsigned short>(std::min<int>(127, start + 4));
        unsigned short length = static_cast<unsigned short>(8 + end - start + 1);
        ret = ctx.call("PMC_G", "pmc_rdpmcrng", "adr_type=0;data_type=0;start=" + std::to_string(start) + ";end=" + std::to_string(end), api.pmc_rdpmcrng,
            [&]{ return api.pmc_rdpmcrng(handle, 0, 0, start, end, length, &buf); }, [&](short){
                std::string text;
                for (int i = 0; i <= end - start; ++i) text += "G" + std::to_string(start+i) + "=" + std::to_string(static_cast<unsigned char>(buf.u.cdata[i])) + ";";
                return text;
            });
        if (ret != EW_OK) break;
    }
    for (int sample = 0; sample < 3; ++sample) {
        ODBST status{};
        ret = ctx.call("STATUS", "cnc_statinfo", "handle=" + std::to_string(handle), api.cnc_statinfo,
            [&]{return api.cnc_statinfo(handle, &status);}, [&](short){
                return "aut=" + std::to_string(status.aut) + ";run=" + std::to_string(status.run) +
                    ";motion=" + std::to_string(status.motion) + ";alarm=" + std::to_string(status.alarm) + ";emergency=" + std::to_string(status.emergency);
            });
        if (ret != EW_OK) return ret;
        for (short mask : {32767, 4897, 256}) {
            IODBSGNL panel{};
            ret = ctx.call("PANEL", "cnc_rdopnlsgnl", "handle=" + std::to_string(handle) + ";slct_data=" + std::to_string(mask), api.cnc_rdopnlsgnl,
                [&]{return api.cnc_rdopnlsgnl(handle, mask, &panel);}, [&](short){
                    return "requested_mask=" + std::to_string(mask) + ";returned_mask=" + std::to_string(panel.type) +
                        ";mode=" + std::to_string(panel.mode) + ";single_block=" + std::to_string(panel.sngl_blck) +
                        ";machine_lock=" + std::to_string(panel.machn_lock) + ";feed_hold=" + std::to_string(panel.feed_hold) +
                        ";feed_override=" + std::to_string(panel.feed_ovrd);
                });
            if (ret != EW_OK) return ret;
        }
        Sleep(300);
    }
'''
driver = (root / "smart_traffic_agent/templates/ncguide_motion_driver.cpp").read_text(encoding="utf-8")
panel_helpers = driver[driver.index("static bool runtime_panel_ready("):driver.index("static short run_motion_capture(")]
fragment = panel_helpers + wrap_capture_fragment(core + "\n    return check_runtime_panel(api, handle, ctx);\n")
assert not validate_capture_fragment(fragment)
header = (Path(settings["header"]) / "Fwlib32.h").read_text(encoding="latin-1")
source = render_capture_harness({"cnc_statinfo", "cnc_rdopnlsgnl", "pmc_rdpmcrng"}, header, fragment)
(out / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
cpp = out / "api_script.cpp"
cpp.write_text(source, encoding="utf-8")
exe = out / "api_script.exe"
print("Output:", out, flush=True)
compiled = CompileGeneratedCppTool().invoke(CompileCppInput(cpp, exe, out))
if not compiled.success:
    print(compiled.stdout, compiled.stderr)
    raise SystemExit(1)
env = {"FOCAS_HOST": settings["host"], "FOCAS_PORT": str(settings["port"]), "FOCAS_DLL_DIR": settings["dll"], "PCAP_NETWORK_DEVICE": settings["device"]}
run = RunGeneratedExecutableTool().invoke(RunExecutableInput(exe, out, env, timeout_seconds=30))
print("\n".join(line for line in run.stdout.splitlines() if "pmc_rdpmcrng" not in line), run.stderr)
(out / "diagnosis.json").write_text(json.dumps({"host": settings["host"], "port": settings["port"], "exit_code": run.exit_code, "stdout": run.stdout, "stderr": run.stderr}, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(run.exit_code or 0)
