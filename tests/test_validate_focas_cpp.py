from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.agent_tools.base import invoke_tool
from smart_traffic_agent.agent_tools.validate_focas_cpp import (
    ValidateFocasCppInput,
    ValidateFocasCppTool,
)


BASE_CPP = """
#include <Fwlib32.h>
void load_and_connect() {
    const char* FOCAS_DLL_DIR = "x";
    GetEnvironmentVariableW(L"FOCAS_DLL_DIR", nullptr, 0);
    SetDllDirectoryW(L"x");
    LoadLibraryW(L"Fwlib32.dll");
    GetProcAddress(nullptr, "cnc_allclibhndl3");
    cnc_allclibhndl3;
}
"""


def run_validator(fragment: str):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cpp = root / "api_script.cpp"
        inc = root / "scenario_fragment.inc"
        cpp.write_text(BASE_CPP, encoding="utf-8")
        inc.write_text(fragment, encoding="utf-8")
        return ValidateFocasCppTool().invoke(
            ValidateFocasCppInput(cpp, inc, nc_program="O1234\nM30\n")
        )


class ValidateFocasCppToolTests(unittest.TestCase):
    def test_poselem_scalar_access_passes(self) -> None:
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBPOS positions[3]{}; auto x = positions[0].abs.data; return EW_OK; }"
        )
        self.assertTrue(result.success)

    def test_poselem_array_access_fails_with_specific_error(self) -> None:
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBPOS positions[3]{}; auto x = positions[0].abs.data[0]; return EW_OK; }"
        )
        self.assertFalse(result.success)
        self.assertTrue(any("POSELM.data is a scalar" in error for error in result.errors))

    def test_odbaxis_array_access_passes(self) -> None:
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBAXIS axis{}; auto x = axis.data[i]; return EW_OK; }"
        )
        self.assertTrue(result.success)

    def test_odbact_scalar_access_passes_and_array_access_fails(self) -> None:
        good = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBACT act{}; auto x = act.data; return EW_OK; }"
        )
        bad = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBACT act{}; auto x = act.data[0]; return EW_OK; }"
        )
        self.assertTrue(good.success)
        self.assertFalse(bad.success)
        self.assertTrue(any("ODBACT.data is a scalar" in error for error in bad.errors))

    def test_forbidden_main_and_loadlibrary_fail(self) -> None:
        main_result = run_validator(
            "int main() { return 0; }\n"
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { return EW_OK; }"
        )
        load_result = run_validator(
            "LoadLibraryW(L'x');\n"
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { return EW_OK; }"
        )
        self.assertFalse(main_result.success)
        self.assertFalse(load_result.success)

    def test_tool_call_record_contains_validation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpp = root / "api_script.cpp"
            inc = root / "scenario_fragment.inc"
            cpp.write_text(BASE_CPP, encoding="utf-8")
            inc.write_text(
                "short run_generated_scenario(FocasApi& api, unsigned short handle) { return EW_OK; }",
                encoding="utf-8",
            )
            output, record = invoke_tool(
                ValidateFocasCppTool(),
                ValidateFocasCppInput(cpp, inc, nc_program="O1234\nM30\n"),
                agent_name="CodeEngineerRole",
            )
            self.assertEqual(record.tool_name, "validate_focas_cpp")
            self.assertTrue(record.success)
            self.assertIn("error_count", record.output_summary)
            self.assertIn("checked_rule_count", record.output_summary)
            self.assertTrue(output.success)

    def test_validation_gate_skips_compile_on_failure(self) -> None:
        compile_called = False
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBPOS positions[3]{}; auto x = positions[0].abs.data[0]; return EW_OK; }"
        )
        if result.success:
            compile_called = True
        self.assertFalse(result.success)
        self.assertFalse(compile_called)

    def test_validation_gate_allows_compile_on_success(self) -> None:
        compile_called = False
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle) { "
            "ODBPOS positions[3]{}; auto x = positions[0].abs.data; return EW_OK; }"
        )
        if result.success:
            compile_called = True
        self.assertTrue(result.success)
        self.assertTrue(compile_called)

    def test_capture_call_arity_is_rejected_before_compile(self) -> None:
        result = run_validator(
            "short run_generated_scenario(FocasApi& api, unsigned short handle, CaptureContext& ctx) { "
            "return ctx.call(\"cnc_statinfo\", [&]() { return api.cnc_statinfo(handle, nullptr); }, "
            "{{\"FlibHndl\", std::to_string(handle)}}, {{\"run\", \"0\"}}); }"
        )
        self.assertFalse(result.success)
        self.assertTrue(any("CAPTURE_CALL_SIGNATURE_MISMATCH" in error for error in result.errors))
        self.assertIn("CaptureContext::call six-argument contract", result.checked_rules)

    def test_raw_model_output_contract_rejects_capture_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpp = root / "api_script.cpp"
            cpp.write_text(BASE_CPP, encoding="utf-8")
            result = ValidateFocasCppTool().invoke(
                ValidateFocasCppInput(
                    cpp,
                    nc_program="O1234\nM30\n",
                    model_output='ctx.call("S001", "cnc_statinfo", "", api.cnc_statinfo, [](){}, [](){})',
                    model_api_name="cnc_statinfo",
                )
            )
        self.assertFalse(result.success)
        self.assertTrue(any("MODEL_BARE_API_CONTRACT" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
