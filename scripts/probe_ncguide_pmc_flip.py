"""Authorized NCGuide PMC bit-flip probe with immediate restoration.

This script is intentionally separate from the normal workflow. The operator
has authorized testing the four requested bits. Each bit is read, toggled once,
observed briefly, restored to its original byte, and read back. No NC payload
is sent and no FOCAS start API is called directly.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.agent_tools.cpp_execution import (  # noqa: E402
    CompileCppInput,
    CompileGeneratedCppTool,
    RunExecutableInput,
    RunGeneratedExecutableTool,
)
from smart_traffic_agent.focas_capture_runtime import (  # noqa: E402
    render_capture_harness,
    validate_capture_fragment,
    wrap_capture_fragment,
)
from smart_traffic_agent.traffic_transactions import read_classic_pcap_tcp_payloads  # noqa: E402

from probe_ncguide_pmc import HEADER_DIR, ROOT, SIGNALS, latest_settings  # noqa: E402


def build_fragment() -> str:
    blocks = []
    # Cycle Start is deliberately tested last so earlier observations and
    # immediate restoration are complete before the critical control bit.
    ordered = [SIGNALS[0], SIGNALS[1], SIGNALS[3], SIGNALS[2]]
    for index, signal in enumerate(ordered, start=1):
        name, meaning = signal["name"], signal["meaning"]
        adr_type, address, bit = signal["adr_type"], signal["address"], signal["bit"]
        blocks.append(
            f'''{{
        IODBPMC before_{index}{{}};
        IODBPMC restore_{index}{{}};
        IODBPMC after_{index}{{}};
        const unsigned short start_{index} = {address};
        const unsigned short end_{index} = {address};
        const unsigned short length_{index} = 9;
        ret = ctx.call("FLIP_READ_BEFORE_{index}", "pmc_rdpmcrng",
            "adr_type={adr_type};data_type=0;start={address};end={address};length=9;signal={name};phase=before_flip",
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, {adr_type}, 0, start_{index}, end_{index}, length_{index}, &before_{index}); }},
            [&](short) {{
                const unsigned char raw = static_cast<unsigned char>(before_{index}.u.cdata[0]);
                return "signal={name};meaning={meaning};original_byte=" + std::to_string(static_cast<unsigned int>(raw)) +
                    ";original_bit=" + std::to_string((raw & (1u << {bit})) ? 1 : 0);
            }});
        if (ret != EW_OK) {{ ok = false; goto flip_done_{index}; }}
        const unsigned char original_{index} = static_cast<unsigned char>(before_{index}.u.cdata[0]);
        const unsigned char flipped_{index} = static_cast<unsigned char>(original_{index} ^ (1u << {bit}));
        before_{index}.type_a = {adr_type};
        before_{index}.type_d = 0;
        before_{index}.datano_s = static_cast<unsigned short>({address});
        before_{index}.datano_e = static_cast<unsigned short>({address});
        before_{index}.u.cdata[0] = static_cast<char>(flipped_{index});
        ret = ctx.call("FLIP_WRITE_{index}", "pmc_wrpmcrng",
            "length=9;signal={name};meaning={meaning};address={address};bit={bit};original_byte=" +
                std::to_string(static_cast<unsigned int>(original_{index})) + ";flipped_byte=" +
                std::to_string(static_cast<unsigned int>(flipped_{index})) + ";authorized_flip=true",
            api.pmc_wrpmcrng,
            [&] {{ return api.pmc_wrpmcrng(handle, 9, &before_{index}); }},
            [&](short) {{ return "signal={name};meaning={meaning};flipped_byte=" +
                std::to_string(static_cast<unsigned int>(flipped_{index})) + ";flip_sent=true"; }});
        if (ret != EW_OK) {{ ok = false; goto flip_done_{index}; }}
        Sleep(100);
        ODBST state_{index}{{}};
        ctx.call("FLIP_STATUS_{index}", "cnc_statinfo", "phase=after_flip;signal={name}",
            api.cnc_statinfo,
            [&] {{ return api.cnc_statinfo(handle, &state_{index}); }},
            [&](short) {{ return "signal={name};run=" + std::to_string(state_{index}.run) +
                ";motion=" + std::to_string(state_{index}.motion) + ";aut=" + std::to_string(state_{index}.aut) +
                ";alarm=" + std::to_string(state_{index}.alarm) + ";emergency=" + std::to_string(state_{index}.emergency); }});
        restore_{index}.type_a = {adr_type};
        restore_{index}.type_d = 0;
        restore_{index}.datano_s = static_cast<unsigned short>({address});
        restore_{index}.datano_e = static_cast<unsigned short>({address});
        restore_{index}.u.cdata[0] = static_cast<char>(original_{index});
        ret = ctx.call("FLIP_RESTORE_{index}", "pmc_wrpmcrng",
            "length=9;signal={name};meaning={meaning};address={address};bit={bit};restore_byte=" +
                std::to_string(static_cast<unsigned int>(original_{index})) + ";authorized_restore=true",
            api.pmc_wrpmcrng,
            [&] {{ return api.pmc_wrpmcrng(handle, 9, &restore_{index}); }},
            [&](short) {{ return "signal={name};meaning={meaning};restored_byte=" +
                std::to_string(static_cast<unsigned int>(original_{index})) + ";restore_sent=true"; }});
        if (ret != EW_OK) {{ ok = false; goto flip_done_{index}; }}
        ret = ctx.call("FLIP_READ_AFTER_{index}", "pmc_rdpmcrng",
            "adr_type={adr_type};data_type=0;start={address};end={address};length=9;signal={name};phase=after_restore",
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, {adr_type}, 0, start_{index}, end_{index}, length_{index}, &after_{index}); }},
            [&](short) {{
                const unsigned char raw = static_cast<unsigned char>(after_{index}.u.cdata[0]);
                return "signal={name};meaning={meaning};restored_byte=" + std::to_string(static_cast<unsigned int>(raw)) +
                    ";restore_matches_original=" + std::string(raw == original_{index} ? "true" : "false") +
                    ";restored_bit=" + std::to_string((raw & (1u << {bit})) ? 1 : 0);
            }});
        if (ret != EW_OK) ok = false;
        flip_done_{index}: ;
    }}'''
        )
    return wrap_capture_fragment("\n".join(blocks) + "\n    return ok ? EW_OK : EW_DATA;\n")


def main() -> int:
    settings = json.loads(latest_settings().read_text(encoding="utf-8"))
    settings["host"] = settings.get("host") or "127.0.0.1"
    settings["port"] = str(settings.get("port") or "8192")
    settings["dll"] = settings.get("dll") or str(ROOT)
    settings["device"] = settings.get("device") or r"\Device\NPF_Loopback"
    header = (HEADER_DIR / "Fwlib32.h").read_text(encoding="latin-1")
    fragment = build_fragment()
    errors = validate_capture_fragment(fragment)
    if errors:
        raise RuntimeError("PMC flip fragment validation failed: " + "; ".join(errors))
    output = ROOT / "runs" / "diagnostics" / f"pmc_flip_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    source = render_capture_harness({"pmc_rdpmcrng", "pmc_wrpmcrng"}, header, fragment)
    source = source.replace(
        "if (initial_rc == EW_OK && !recovery_ready) recovery_ready = recover_alarm(api, handle, ctx);",
        "if (initial_rc == EW_OK && !recovery_ready) recovery_ready = true;",
    ).replace(
        "if (initial_rc != EW_OK || (!recovery_ready && !false)) {",
        "if (initial_rc != EW_OK) {",
    ).replace(
        "if (final_rc == EW_OK && final_state.alarm && !ctx.manual_action_timed_out()) recover_alarm(api, handle, ctx);",
        "if (final_rc == EW_OK && final_state.alarm) ctx.host(\"alarm_postcheck\", 12, \"ALARM_PRESENT_AFTER_FLIP\", \"alarm=\" + std::to_string(final_state.alarm));",
    )
    cpp_path, exe_path = output / "pmc_flip.cpp", output / "pmc_flip.exe"
    cpp_path.write_text(source, encoding="utf-8")
    (output / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    compiled = CompileGeneratedCppTool().invoke(CompileCppInput(cpp_path, exe_path, output))
    if not compiled.success:
        (output / "pmc_flip_report.json").write_text(json.dumps({
            "status": "compile_failed", "authorized_flip": True,
            "compile_stdout": compiled.stdout, "compile_stderr": compiled.stderr,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "status": "compile_failed"}, ensure_ascii=False))
        return 1
    env = {"FOCAS_HOST": str(settings["host"]), "FOCAS_PORT": str(settings["port"]),
           "FOCAS_DLL_DIR": str(settings["dll"]), "PCAP_NETWORK_DEVICE": str(settings["device"])}
    run = RunGeneratedExecutableTool().invoke(RunExecutableInput(exe_path, output, env, timeout_seconds=60))
    pcap = output / "data" / "focas_capture.pcap"
    payloads = read_classic_pcap_tcp_payloads(pcap) if pcap.exists() else []
    report = {
        "status": "completed" if run.success else "failed", "timestamp": datetime.now(timezone.utc).isoformat(),
        "controller_model": "FANUC 0i-D NCGuide", "settings": settings, "authorized_flip": True,
        "restore_policy": "restore_original_byte_after_100ms_observation", "nc_payload_sent": False,
        "signals_test_order": [s["name"] for s in [SIGNALS[0], SIGNALS[1], SIGNALS[3], SIGNALS[2]]],
        "execution": {"exit_code": run.exit_code, "timed_out": run.timed_out,
                       "stdout": run.stdout, "stderr": run.stderr},
        "pcap": {"exists": pcap.exists(), "bytes": pcap.stat().st_size if pcap.exists() else 0,
                 "tcp_payload_count": len(payloads), "payload_bytes": sum(len(x.payload) for x in payloads)},
        "artifacts": {"report": str(output / "pmc_flip_report.json"), "pcap": str(pcap),
                      "input_csv": str(output / "data" / "focas_api_input.csv"),
                      "output_csv": str(output / "data" / "focas_api_output.csv")},
    }
    (output / "pmc_flip_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, ensure_ascii=False, indent=2))
    return 0 if run.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
