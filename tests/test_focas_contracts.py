from __future__ import annotations

import unittest

from smart_traffic_agent.focas_contracts import (
    documented_parameter_names,
    header_error_codes,
    named_contract_context,
    resolve_api_return_code,
    undocumented_argument_names,
    validate_return_code_labels,
)


HEADER = "#define EW_LENGTH 2\n#define EW_NUMBER 3\n#define EW_ATTRIB 4\n#define EW_TYPE 4\n"


class FocasContractTests(unittest.TestCase):
    def test_named_documentation_corrects_unnamed_header_contract(self) -> None:
        names = documented_parameter_names("cnc_rdtofs")
        self.assertEqual(names, ["FlibHndl", "number", "type", "length", "tofs"])
        context = named_contract_context(["cnc_rdtofs"])
        self.assertIn("short number, short type, short length, ODBTOFS *tofs", context)
        self.assertIn("size of ODBTOFS structure=8", context)
        self.assertIn("EW_LENGTH", context)
        self.assertEqual(
            undocumented_argument_names(
                "cnc_rdtofs",
                {"FlibHndl": "<handle>", "ofs_type": 0, "ofs_no": 1, "data_num": 1, "tofs": "<out>"},
            ),
            ["data_num", "ofs_no", "ofs_type"],
        )

    def test_return_code_validator_rejects_mislabeled_length_error(self) -> None:
        wrong = 'std::string RetText(short code) { switch(code) { case 2: return "EW_NUMBER"; } }'
        correct = 'std::string RetText(short code) { switch(code) { case 2: return "EW_LENGTH"; } }'
        self.assertTrue(any("EW_LENGTH" in item for item in validate_return_code_labels(wrong, HEADER)))
        self.assertEqual(validate_return_code_labels(correct, HEADER), [])

    def test_return_code_aliases_are_accepted(self) -> None:
        mapping = header_error_codes("#define EW_ATTRIB 4\n#define EW_TYPE 4\n")
        self.assertEqual(mapping[4], {"EW_ATTRIB", "EW_TYPE"})
        source = 'std::string RetText(short code) { switch(code) { case 4: return "EW_TYPE"; } }'
        self.assertEqual(validate_return_code_labels(source, HEADER), [])

    def test_api_scoped_return_code_explanation(self) -> None:
        result = resolve_api_return_code("cnc_rdtofs", 2)
        self.assertEqual(result["api_code_names"], ["EW_LENGTH"])
        self.assertTrue(result["documented_for_api"])
        self.assertIn("ODBTOFS", " ".join(result["api_explanations"]))
        self.assertIn("EW_LENGTH", result["global_code_names"])

    def test_return_text_validator_catches_numeric_mismatch(self) -> None:
        source = 'std::string ReturnText(short code) { switch(code) { case 2: return "EW_NUMBER"; } }'
        self.assertTrue(any("EW_LENGTH" in item for item in validate_return_code_labels(source, HEADER)))

    def test_rejects_the_failed_live_run_before_execution(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        source = root / "runs/desktop/20261010_142436_7e9035/batch_001/execution/api_script.cpp"
        sdk = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0i\Fwlib32.h")
        if not source.exists() or not sdk.exists():
            self.skipTest("Historical NCGuide run and local FOCAS SDK are not present")
        diagnostics = validate_return_code_labels(
            source.read_text(encoding="utf-8"), sdk.read_text(encoding="latin-1")
        )
        self.assertTrue(any("return code 2" in item and "EW_LENGTH" in item for item in diagnostics))


if __name__ == "__main__":
    unittest.main()
