"""Explicit PMC write-path probe that writes each signal's original byte back.

This is an opt-in diagnostic, separate from the read-only probe. It never
flips a bit: every pmc_wrpmcrng call writes the byte just read, then reads it
back. In particular, G0007#2 is never changed from 0 to 1.
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
    for index, signal in enumerate(SIGNALS, start=1):
        name, meaning = signal["name"], signal["meaning"]
        adr_type, address, bit = signal["adr_type"], signal["address"], signal["bit"]
        blocks.append(
            f'''{{
        IODBPMC before_{index}{{}};
        const unsigned short start_{index} = {address};
        const unsigned short end_{index} = {address};
        const unsigned short length_{index} = 9;
        ret = ctx.call("PMC_WRITE_PRE_{index}", "pmc_rdpmcrng",
            "adr_type={adr_type};data_type=0;start=" + std::to_string(start_{index}) +
                ";end=" + std::to_string(end_{index}) + ";length=9;signal={name};phase=before_write",
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, {adr_type}, 0, start_{index}, end_{index}, length_{index}, &before_{index}); }},
            [&](short) {{
                const unsigned char raw = static_cast<unsigned char>(before_{index}.u.cdata[0]);
                return "signal={name};meaning={meaning};raw_byte=" + std::to_string(static_cast<unsigned int>(raw)) +
                    ";bit=" + std::to_string({bit}) + ";bit_value=" + std::to_string((raw & (1u << {bit})) ? 1 : 0) +
                    ";write_mode=same_value_only";
            }});
        if (ret != EW_OK) {{ ctx.host("PMC_WRITE_SKIPPED_{index}", ret, "READ_BEFORE_WRITE_FAILED", "signal={name}"); ok = false; }} else {{
        const unsigned char original_{index} = static_cast<unsigned char>(before_{index}.u.cdata[0]);
        before_{index}.type_a = {adr_type};
        before_{index}.type_d = 0;
        before_{index}.datano_s = static_cast<unsigned short>({address});
        before_{index}.datano_e = static_cast<unsigned short>({address});
        ret = ctx.call("PMC_WRITE_SAME_{index}", "pmc_wrpmcrng",
            "length=9;signal={name};meaning={meaning};address={address};bit={bit};value=" +
                std::to_string(static_cast<unsigned int>(original_{index})) + ";same_value=true;no_bit_flip=true",
            api.pmc_wrpmcrng,
            [&] {{ return api.pmc_wrpmcrng(handle, 9, &before_{index}); }},
            [&](short) {{ return "signal={name};meaning={meaning};written_byte=" +
                std::to_string(static_cast<unsigned int>(original_{index})) +
                ";same_value=true;no_bit_flip=true;write_effect_not_assumed=true"; }});
        IODBPMC after_{index}{{}};
        short readback_{index} = ctx.call("PMC_WRITE_POST_{index}", "pmc_rdpmcrng",
            "adr_type={adr_type};data_type=0;start=" + std::to_string(start_{index}) +
                ";end=" + std::to_string(end_{index}) + ";length=9;signal={name};phase=readback",
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, {adr_type}, 0, start_{index}, end_{index}, length_{index}, &after_{index}); }},
            [&](short) {{
                const unsigned char raw = static_cast<unsigned char>(after_{index}.u.cdata[0]);
                return "signal={name};meaning={meaning};readback_byte=" + std::to_string(static_cast<unsigned int>(raw)) +
                    ";same_as_original=" + std::string(raw == original_{index} ? "true" : "false") +
                    ";bit_value=" + std::to_string((raw & (1u << {bit})) ? 1 : 0);
            }});
        if (ret != EW_OK || readback_{index} != EW_OK) ok = false;
        }}
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
        raise RuntimeError("PMC same-value write fragment validation failed: " + "; ".join(errors))
    output = ROOT / "runs" / "diagnostics" / f"pmc_write_same_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    source = render_capture_harness({"pmc_rdpmcrng", "pmc_wrpmcrng"}, header, fragment)
    # No alarm reset is allowed in this diagnostic. Abort before any write if
    # the status read itself fails; otherwise keep the read-only status record.
    source = source.replace(
        "if (initial_rc == EW_OK && !recovery_ready) recovery_ready = recover_alarm(api, handle, ctx);",
        "if (initial_rc == EW_OK && !recovery_ready) recovery_ready = true;",
    ).replace(
        "if (initial_rc != EW_OK || (!recovery_ready && !false)) {",
        "if (initial_rc != EW_OK) {",
    ).replace(
        "if (final_rc == EW_OK && final_state.alarm && !ctx.manual_action_timed_out()) recover_alarm(api, handle, ctx);",
        "if (final_rc == EW_OK && final_state.alarm) ctx.host(\"alarm_postcheck\", 12, \"ALARM_PRESENT_READ_ONLY\", \"alarm=\" + std::to_string(final_state.alarm));",
    )
    cpp_path, exe_path = output / "pmc_write_same.cpp", output / "pmc_write_same.exe"
    cpp_path.write_text(source, encoding="utf-8")
    (output / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    compiled = CompileGeneratedCppTool().invoke(CompileCppInput(cpp_path, exe_path, output))
    if not compiled.success:
        (output / "pmc_write_same_report.json").write_text(json.dumps({
            "status": "compile_failed", "write_mode": "same_value_only", "write_test": "not_executed",
            "compile_stdout": compiled.stdout, "compile_stderr": compiled.stderr,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    env = {"FOCAS_HOST": str(settings["host"]), "FOCAS_PORT": str(settings["port"]),
           "FOCAS_DLL_DIR": str(settings["dll"]), "PCAP_NETWORK_DEVICE": str(settings["device"])}
    run = RunGeneratedExecutableTool().invoke(RunExecutableInput(exe_path, output, env, timeout_seconds=60))
    pcap = output / "data" / "focas_capture.pcap"
    payloads = read_classic_pcap_tcp_payloads(pcap) if pcap.exists() else []
    report = {
        "status": "completed" if run.success else "failed", "timestamp": datetime.now(timezone.utc).isoformat(),
        "controller_model": "FANUC 0i-D NCGuide", "settings": settings,
        "write_mode": "same_value_only", "bit_flip": False, "nc_execution": False,
        "signals": SIGNALS, "write_api": "pmc_wrpmcrng(unsigned short, unsigned short, IODBPMC*)",
        "execution": {"exit_code": run.exit_code, "timed_out": run.timed_out, "stdout": run.stdout, "stderr": run.stderr},
        "pcap": {"exists": pcap.exists(), "bytes": pcap.stat().st_size if pcap.exists() else 0,
                 "tcp_payload_count": len(payloads), "payload_bytes": sum(len(x.payload) for x in payloads)},
        "artifacts": {"report": str(output / "pmc_write_same_report.json"), "pcap": str(pcap),
                      "input_csv": str(output / "data" / "focas_api_input.csv"),
                      "output_csv": str(output / "data" / "focas_api_output.csv")},
    }
    (output / "pmc_write_same_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"]}, ensure_ascii=False, indent=2))
    return 0 if run.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
