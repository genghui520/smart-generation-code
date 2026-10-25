"""Compile production recipes; optionally capture a read-only FOCAS API."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.focas_capture_runtime import guarded_core, wrap_capture_fragment, render_capture_harness, validate_capture_fragment
from smart_traffic_agent.agent_tools.cpp_execution import CompileGeneratedCppTool, CompileCppInput, RunGeneratedExecutableTool, RunExecutableInput, read_cpp_csv_logs
from smart_traffic_agent.integrations.ncguide import default_focas_header_dir, default_focas_runtime_dir
from smart_traffic_agent.agents.code_generator import validate_generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Execute the selected read-only API; never starts axis motion")
    parser.add_argument("--live-api", choices=("cnc_statinfo", "cnc_rdpdf_line"), default="cnc_statinfo", help="Read-only API to capture after prerequisite checks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=Path("runs/diagnostics/production_capture"))
    args = parser.parse_args()
    header = (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1")
    report = {"compile": {}, "live": None}
    for api in ("cnc_statinfo", "cnc_search", "cnc_rdpdf_line", "cnc_wrmdipntr"):
        folder = (args.out / api).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        fragment = wrap_capture_fragment(guarded_core({api}, samples=5, interval_ms=50))
        source = render_capture_harness({api}, header, fragment)
        errors = validate_capture_fragment(fragment) + [x for x in validate_generated(source + "\n" + fragment, "O1234\nM30\n", [], require_official_focas_header=True, require_real_pcap_capture=True) if x.startswith("BLOCKING:")]
        if errors:
            raise RuntimeError(f"{api}: {errors}")
        (folder / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
        path = folder / "api_script.cpp"
        path.write_text(source, encoding="utf-8")
        exe = folder / "api_script.exe"
        compiled = CompileGeneratedCppTool().invoke(CompileCppInput(path, exe, folder))
        report["compile"][api] = compiled.success
        print(api, "compile", compiled.success, flush=True)
        if not compiled.success:
            print(compiled.stdout, compiled.stderr)
            raise RuntimeError(f"Native compile failed: {api}")
        if args.live and api == args.live_api:
            run = RunGeneratedExecutableTool().invoke(RunExecutableInput(exe, folder, {"FOCAS_HOST": args.host, "FOCAS_PORT": str(args.port), "FOCAS_DLL_DIR": str(default_focas_runtime_dir()), "PCAP_NETWORK_DEVICE": r"\Device\NPF_Loopback"}, timeout_seconds=30))
            artifacts = read_cpp_csv_logs("production-capture-readonly", folder)
            calls = [log for log in artifacts.api_logs if log.protocol_function == api]
            report["live"] = {"target_api": api, "exit_code": run.exit_code, "schema_errors": artifacts.schema_errors, "api_calls": len(calls), "matched_api_calls": sum(log.traffic_status == "matched" for log in calls), "distinct_inputs": len({str(log.input_parameters) for log in calls}), "distinct_outputs": len({str(log.response.get("data", "")) for log in calls}), "pcap_bytes": (folder / "data/focas_capture.pcap").stat().st_size if (folder / "data/focas_capture.pcap").exists() else 0}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.live and (report["live"]["exit_code"] != 0 or report["live"]["schema_errors"] or report["live"]["matched_api_calls"] != report["live"]["api_calls"] or not report["live"]["api_calls"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
