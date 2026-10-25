from pathlib import Path
import tempfile


HARNESS_PATH = Path(__file__).resolve().parents[1] / "smart_traffic_agent" / "templates" / "focas_harness.cpp"
CAPTURE_HARNESS_PATH = Path(__file__).resolve().parents[1] / "smart_traffic_agent" / "templates" / "focas_capture_harness.cpp"


def test_harness_uses_configured_focas_dll_directory():
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert 'GetEnvironmentVariableW(L"FOCAS_DLL_DIR"' in source
    assert 'dll_path = directory + L"Fwlib32.dll"' in source
    assert "LoadLibraryW(dll_path.c_str())" in source


def test_harness_falls_back_to_default_dll_name():
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert 'std::wstring dll_path = L"Fwlib32.dll"' in source


def test_capture_runtime_adds_alarm_diagnostics_and_bounded_recovery():
    from smart_traffic_agent.focas_capture_runtime import capture_functions

    names = set(capture_functions({"cnc_rdposition"}))
    assert {
        "cnc_alarm", "cnc_alarm2", "cnc_getdtailerr", "cnc_rdalmmsg2",
        "cnc_rdalmhisno", "cnc_rdalmhistry5", "cnc_stopophis", "cnc_startophis",
        "cnc_clralm", "cnc_reset",
    } <= names

    source = CAPTURE_HARNESS_PATH.read_text(encoding="utf-8")
    assert "if (ret != EW_OK) diagnose(true);" in source
    assert "capture_alarm_diagnostics(api, handle, ctx)" in source
    assert '"cnc_rdalmmsg2"' in source
    assert '"cnc_rdalmhistry5"' in source
    assert '"cnc_startophis"' in source
    assert 'messages[i].alm_no == 100' in source
    assert 'api.cnc_clralm(handle, 0)' in source
    assert 'api.cnc_reset(handle)' in source
    assert '"ALARM_RECOVERY_VERIFY", "cnc_statinfo"' in source
    assert '"ALARM_RECOVERY_VERIFIED"' in source
    assert '"ALARM_POSTCHECK", "cnc_statinfo"' in source


def test_download4_capture_includes_directory_discovery_helper():
    from smart_traffic_agent.focas_capture_runtime import capture_functions

    names = set(capture_functions({"cnc_download4"}))
    assert {"cnc_rdprogdir3", "cnc_getdtailerr"} <= names


def test_alarm_recovery_harness_compiles_with_0id_sdk():
    from smart_traffic_agent.agent_tools.cpp_execution import CompileCppInput, CompileGeneratedCppTool
    from smart_traffic_agent.execution_contracts import ExecutionContract
    from smart_traffic_agent.focas_capture_runtime import render_capture_harness
    from smart_traffic_agent.integrations.ncguide import default_focas_header_dir

    header_path = default_focas_header_dir() / "Fwlib32.h"
    if not header_path.is_file():
        return
    fragment = "short run_generated_scenario(FocasApi&, unsigned short, CaptureContext&) { return EW_OK; }"
    root = Path(__file__).resolve().parents[1] / ".test_tmp"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as folder:
        path = Path(folder)
        (path / "scenario_fragment.inc").write_text(fragment, encoding="utf-8")
        for contract in (None, ExecutionContract(("cnc_acts",), (), motion=True)):
            source = render_capture_harness(
                {"cnc_statinfo"}, header_path.read_text(encoding="latin-1"), fragment,
                contract=contract, nc_program="\nO5711\nM30\n%",
            )
            name = "alarm_recovery_motion" if contract else "alarm_recovery"
            cpp, exe = path / f"{name}.cpp", path / f"{name}.exe"
            cpp.write_text(source, encoding="utf-8")
            result = CompileGeneratedCppTool().invoke(CompileCppInput(cpp, exe, path))
            assert result.success, result.stdout + result.stderr
