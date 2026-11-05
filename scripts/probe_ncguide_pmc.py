"""Independent read-only PMC feasibility probe for FANUC NCGuide 0i-D.

This script deliberately does not use the four-agent workflow and never calls
pmc_wrpmcrng. It generates a normal 32-bit FOCAS harness so the same Npcap,
CSV and invocation evidence path is used as a real capture.
"""

from __future__ import annotations

import json
import re
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


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD")

SIGNALS = [
    {"name": "G0043#5", "meaning": "DNCI", "adr_type": 0, "address": 43, "bit": 5, "safety": "read_only_mode_signal; no write"},
    {"name": "F0003#4", "meaning": "MRMT", "adr_type": 1, "address": 3, "bit": 4, "safety": "read_only_cnc_to_pmc_signal; no write"},
    {"name": "G0007#2", "meaning": "Cycle Start", "adr_type": 0, "address": 7, "bit": 2, "safety": "critical_control_signal; read only; never write"},
    {"name": "F0000#5", "meaning": "STL", "adr_type": 1, "address": 0, "bit": 5, "safety": "CNC_state_signal; read only; never write"},
]


def latest_settings() -> Path:
    candidates = list((ROOT / "runs" / "desktop").glob("**/settings.json"))
    if not candidates:
        raise FileNotFoundError("runs/desktop 中没有可用 settings.json")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_fragment() -> str:
    blocks = []
    for index, signal in enumerate(SIGNALS, start=1):
        name, meaning = signal["name"], signal["meaning"]
        adr_type, address, bit = signal["adr_type"], signal["address"], signal["bit"]
        blocks.append(
            f'''{{
        IODBPMC pmc_{index}{{}};
        const unsigned short pmc_start_{index} = {address};
        const unsigned short pmc_end_{index} = {address};
        const unsigned short pmc_length_{index} = 9;
        ret = ctx.call("PMC_READ_{index}", "pmc_rdpmcrng",
            "adr_type=" + std::to_string({adr_type}) +
                ";data_type=0;start=" + std::to_string(pmc_start_{index}) +
                ";end=" + std::to_string(pmc_end_{index}) +
                ";length=" + std::to_string(pmc_length_{index}) +
                ";signal={name};meaning={meaning};bit={bit}",
            api.pmc_rdpmcrng,
            [&] {{ return api.pmc_rdpmcrng(handle, {adr_type}, 0,
                pmc_start_{index}, pmc_end_{index}, pmc_length_{index}, &pmc_{index}); }},
            [&](short) {{
                const unsigned char raw = static_cast<unsigned char>(pmc_{index}.u.cdata[0]);
                const unsigned char mask = static_cast<unsigned char>(1u << {bit});
                return "signal={name};meaning={meaning};address={address};bit={bit};raw_byte=" +
                    std::to_string(static_cast<unsigned int>(raw)) +
                    ";bit_value=" + std::to_string((raw & mask) ? 1 : 0) +
                    ";mask=" + std::to_string(static_cast<unsigned int>(mask)) +
                    ";read_only=true";
            }});
        if (ret != EW_OK) return ret;
    }}'''
        )
    return wrap_capture_fragment("\n".join(blocks) + "\n    return 0;\n")


def parse_results(stdout: str) -> list[dict[str, object]]:
    pattern = re.compile(
        r"pmc_rdpmcrng -> ret=(?P<ret>-?\d+) (?P<text>[^;]+); "
        r"signal=(?P<signal>[^;]+);meaning=(?P<meaning>[^;]+);address=(?P<address>\d+);"
        r"bit=(?P<bit>\d+);raw_byte=(?P<raw>\d+);bit_value=(?P<value>\d+)"
    )
    results = []
    for match in pattern.finditer(stdout):
        item = match.groupdict()
        results.append({
            "signal": item["signal"], "meaning": item["meaning"],
            "address": int(item["address"]), "bit": int(item["bit"]),
            "raw_byte": int(item["raw"]), "bit_value": int(item["value"]),
            "return_code": int(item["ret"]), "return_text": item["text"],
        })
    return results


def pcap_stats(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "tcp_payload_count": 0, "payload_bytes": 0}
    payloads = read_classic_pcap_tcp_payloads(path)
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "tcp_payload_count": len(payloads),
        "payload_bytes": sum(len(item.payload) for item in payloads),
    }


def main() -> int:
    settings_path = latest_settings()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["host"] = settings.get("host") or "127.0.0.1"
    settings["port"] = str(settings.get("port") or "8192")
    settings["dll"] = settings.get("dll") or str(ROOT)
    settings["device"] = settings.get("device") or r"\Device\NPF_Loopback"
    header = (HEADER_DIR / "Fwlib32.h").read_text(encoding="latin-1")
    fragment = build_fragment()
    validation_errors = validate_capture_fragment(fragment)
    if validation_errors:
        raise RuntimeError("PMC probe fragment validation failed: " + "; ".join(validation_errors))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "runs" / "diagnostics" / f"pmc_feasibility_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    source = render_capture_harness({"pmc_rdpmcrng"}, header, fragment)
    # The production harness protects state-changing workflows behind alarm
    # recovery. This probe has no state-changing calls, so preserve the alarm
    # evidence and continue with read-only PMC calls without invoking reset.
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
    cpp_path, exe_path = output / "pmc_probe.cpp", output / "pmc_probe.exe"
    cpp_path.write_text(source, encoding="utf-8")
    (output / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    compiled = CompileGeneratedCppTool().invoke(CompileCppInput(cpp_path, exe_path, output))
    if not compiled.success:
        report = {
            "status": "compile_failed", "settings": settings,
            "header": str(HEADER_DIR / "Fwlib32.h"),
            "signature": "pmc_rdpmcrng(unsigned short, short, short, unsigned short, unsigned short, unsigned short, IODBPMC*)",
            "write_test": "not_executed", "compile_stdout": compiled.stdout,
            "compile_stderr": compiled.stderr, "signals": SIGNALS,
        }
        (output / "pmc_feasibility_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    environment = {
        "FOCAS_HOST": str(settings["host"]), "FOCAS_PORT": str(settings["port"]),
        "FOCAS_DLL_DIR": str(settings["dll"]), "PCAP_NETWORK_DEVICE": str(settings["device"]),
    }
    run = RunGeneratedExecutableTool().invoke(RunExecutableInput(exe_path, output, environment, timeout_seconds=60))
    results = parse_results(run.stdout)
    capture_path = output / "data" / "focas_capture.pcap"
    report = {
        "status": "read_completed" if run.success and len(results) == len(SIGNALS) else "read_incomplete",
        "timestamp": datetime.now(timezone.utc).isoformat(), "controller_model": "FANUC 0i-D NCGuide",
        "settings": settings, "header": str(HEADER_DIR / "Fwlib32.h"),
        "signature": "pmc_rdpmcrng(unsigned short, short, short, unsigned short, unsigned short, unsigned short, IODBPMC*)",
        "write_signature": "pmc_wrpmcrng(unsigned short, unsigned short, IODBPMC*)",
        "address_type_evidence": {"G": 0, "F": 1, "Y": 2, "X": 3},
        "byte_length_rule": "length = 8 + N; N=1 => 9", "write_test": "not_executed",
        "write_reason": "指定信号包含模式选择、Cycle Start 和 CNC 状态信号；F/X 地址区也不能据地址本身推定可写，未取得梯形图级安全写入证据。",
        "signals": SIGNALS, "results": results,
        "execution": {"exit_code": run.exit_code, "timed_out": run.timed_out,
                       "stdout": run.stdout, "stderr": run.stderr, "pcap_expected": True},
        "pcap": pcap_stats(capture_path),
        "artifacts": {"pcap_directory": str(output / "data"),
                      "input_csv": str(output / "data" / "focas_api_input.csv"),
                      "output_csv": str(output / "data" / "focas_api_output.csv")},
    }
    (output / "pmc_feasibility_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "results": results}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "read_completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
