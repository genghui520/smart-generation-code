from __future__ import annotations

import unittest

from smart_traffic_agent.rag.focas_return_codes import (
    build_return_code_catalog,
    extract_api_return_codes,
    site_code_descriptions,
    site_code_values,
)


class FocasReturnCodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "source_path": "errcode.htm", "chunk_index": 1,
                "text": "CNC Data Window\nEW_LENGTH\n(2)\nBad length\nCheck buffer size.\n"
                        "EW_NUMBER\n(3)\nBad number\nCheck number.\n"
                        "PMC Data Window\nEW_LENGTH\n(2)\nBad PMC length\nCheck PMC buffer.\n",
            },
            {
                "chunk_type": "error_codes", "function": "cnc_rdtofs",
                "source_url": "https://www.woody.vip/fanuc/Ncdata/cnc_rdtofs.xml",
                "source_path": "Ncdata/cnc_rdtofs.xml",
                "chunk_id": "focas-cnc_rdtofs-error_codes",
                "text": "Function: cnc_rdtofs\nReturn/error codes:\n"
                        "- EW_LENGTH: Data block length errorSize is wrong.\n"
                        "- EW_NUMBER: Offset number is wrong.\n"
                        "- EW_LENGTH: For another series, buffer size is wrong.\n",
            },
        ]

    def test_global_name_value_and_cnc_pmc_descriptions(self) -> None:
        values, conflicts = site_code_values(self.chunks)
        self.assertEqual(values["EW_LENGTH"], 2)
        self.assertEqual(conflicts, {})
        descriptions = site_code_descriptions(self.chunks)
        self.assertEqual(descriptions["EW_LENGTH"]["cnc"]["meaning"], "Bad length")
        self.assertEqual(descriptions["EW_LENGTH"]["pmc"]["handling"], "Check PMC buffer.")

    def test_per_api_keeps_all_distinct_explanations(self) -> None:
        rows = extract_api_return_codes(
            self.chunks, {"EW_LENGTH": 2, "EW_NUMBER": 3}, {"EW_LENGTH": 2, "EW_NUMBER": 3}
        )
        length = next(row for row in rows if row["code_name"] == "EW_LENGTH")
        self.assertEqual(length["code_value"], 2)
        self.assertEqual(length["value_source"], "sdk_and_site")
        self.assertEqual(len(length["api_explanations"]), 2)
        self.assertIn("error Size", length["api_explanations"][0])

    def test_qualified_and_prose_error_mentions_are_not_lost(self) -> None:
        chunks = [
            {"chunk_type": "error_codes", "function": "cnc_actf", "text":
             "Return/error codes:\n- EW_PARAM(Only Series 15): Parameter condition is wrong."},
            {"chunk_type": "error_codes", "function": "cnc_freelibhndl", "text":
             "Return/error codes:\nIn case of except EW_HANDLE error, the handle is already freed."},
        ]
        rows = extract_api_return_codes(chunks, {"EW_PARAM": 9, "EW_HANDLE": -8}, {})
        self.assertEqual(len(rows), 2)
        self.assertIn("Only Series 15", next(row for row in rows if row["function"] == "cnc_actf")["api_explanations"][0])
        self.assertEqual(next(row for row in rows if row["function"] == "cnc_freelibhndl")["extraction_method"], "prose_mention")

    def test_real_cached_catalog_has_numeric_explanations(self) -> None:
        from pathlib import Path
        from smart_traffic_agent.rag.focas_return_codes import read_jsonl

        root = Path(__file__).resolve().parents[1]
        chunks = read_jsonl(root / "rag_indexes/focas/chunks.jsonl")
        catalog, api_rows, rag_chunks = build_return_code_catalog(chunks, [])
        self.assertGreaterEqual(catalog["api_error_section_count"], 680)
        self.assertEqual(catalog["api_with_error_section_count"], catalog["api_error_section_count"])
        self.assertEqual(len(api_rows), len(rag_chunks))
        target = next(
            row for row in api_rows
            if row["function"] == "cnc_rdtofs" and row["code_name"] == "EW_LENGTH"
        )
        self.assertEqual(target["code_value"], 2)
        self.assertIn("ODBTOFS", target["api_explanations"][0])


if __name__ == "__main__":
    unittest.main()
