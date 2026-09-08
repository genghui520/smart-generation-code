from __future__ import annotations

import json
import tempfile
import unittest
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from smart_traffic_agent.rag.focas_function_manifest import (
    build_focas_function_manifest,
    build_function_coverage_context,
    load_focas_function_rows,
    split_function_cell,
    symbol_candidates_for,
)


class FocasFunctionManifestTests(unittest.TestCase):
    def test_split_function_cell_handles_multi_function_cells(self) -> None:
        self.assertEqual(split_function_cell("cnc_dwnstart3/cnc_dwnstart4"), ["cnc_dwnstart3", "cnc_dwnstart4"])
        self.assertEqual(split_function_cell("download3，download4"), ["download3", "download4"])

    def test_symbol_candidates_use_pmc_prefix_for_pmc_and_profibus_categories(self) -> None:
        self.assertEqual(symbol_candidates_for("absolute", "controlled axis/spindle")[0], "cnc_absolute")
        self.assertEqual(symbol_candidates_for("rdpmcrng", "PMC")[0], "pmc_rdpmcrng")
        self.assertEqual(symbol_candidates_for("prfrdconfig", "PROFIBUS-DP")[0], "pmc_prfrdconfig")

    def test_loads_headerless_focas_funcs_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = Path(tmp) / "focas_funcs.xlsx"
            write_minimal_xlsx(
                workbook,
                [
                    ["library handle, node", "allclibhndl3", "Get the library handle(for Ethernet)"],
                    ["", "freelibhndl", "Free library handle"],
                    ["PMC", "rdpmcrng", "Read PMC data(area specified)"],
                ],
            )

            rows = load_focas_function_rows(workbook)

            self.assertEqual([row["function"] for row in rows], ["cnc_allclibhndl3", "cnc_freelibhndl", "pmc_rdpmcrng"])
            self.assertEqual(rows[1]["category"], "library handle, node")
            self.assertIn("cnc_rdpmcrng", rows[2]["symbol_candidates"])

    def test_build_writes_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = Path(tmp) / "focas_funcs.xlsx"
            output = Path(tmp) / "function_manifest.jsonl"
            write_minimal_xlsx(
                workbook,
                [
                    ["controlled axis/spindle", "absolute", "Read absolute position"],
                    ["", "absolute2", "Read absolute position 2"],
                ],
            )

            summary = build_focas_function_manifest(workbook, output)
            lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

            self.assertEqual(summary["row_count"], 2)
            self.assertEqual(summary["unique_function_count"], 2)
            self.assertEqual(lines[0]["manifest_id"], "sheet1-r0001-absolute")
            self.assertEqual(lines[0]["coverage_status"], "unplanned")

    def test_build_function_coverage_context_batches_uncovered_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "function_manifest.jsonl"
            rows = [
                {"function": "cnc_absolute", "raw_function": "absolute", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_absolute", "raw_function": "absolute", "function_family": "cnc", "category": "axis"},
                {"function": "pmc_rdpmcrng", "raw_function": "rdpmcrng", "function_family": "pmc", "category": "PMC"},
                {"function": "cnc_actf", "raw_function": "actf", "function_family": "cnc", "category": "axis"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            context = build_function_coverage_context(
                manifest,
                covered_functions={"cnc_absolute"},
                batch_size=1,
                scenario_type="pmc_signal_read",
            )

            self.assertTrue(context["enabled"])
            self.assertEqual(context["target_function_count"], 3)
            self.assertEqual(context["covered_function_count"], 1)
            self.assertEqual(context["remaining_function_count"], 2)
            self.assertEqual(context["selected_batch"][0]["function"], "pmc_rdpmcrng")
            self.assertEqual(context["selected_batch"][0]["coverage_role"], "target")

    def test_coverage_context_interleaves_scenario_support_apis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "function_manifest.jsonl"
            rows = [
                {"function": "cnc_absolute", "raw_function": "absolute", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_dwnstart3", "raw_function": "dwnstart3", "function_family": "cnc", "category": "CNC program"},
                {"function": "cnc_download3", "raw_function": "download3", "function_family": "cnc", "category": "CNC program"},
                {"function": "cnc_dwnend3", "raw_function": "dwnend3", "function_family": "cnc", "category": "CNC program"},
                {"function": "cnc_search", "raw_function": "search", "function_family": "cnc", "category": "CNC program"},
                {"function": "cnc_rdprgnum", "raw_function": "rdprgnum", "function_family": "cnc", "category": "CNC program"},
                {"function": "cnc_statinfo", "raw_function": "statinfo", "function_family": "cnc", "category": "others"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            context = build_function_coverage_context(
                manifest,
                batch_size=1,
                scenario_type="coordinate_motion",
            )

            self.assertEqual(context["scenario_batch_id"], "programmed_coordinate_motion")
            self.assertEqual(context["target_batch"][0]["function"], "cnc_absolute")
            self.assertEqual(context["target_batch"][0]["coverage_role"], "target")
            self.assertIn("cnc_dwnstart3", [row["function"] for row in context["support_batch"]])
            self.assertTrue(all(row["coverage_role"] == "support" for row in context["support_batch"]))

    def test_coverage_context_skips_terminal_covered_target_functions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "function_manifest.jsonl"
            rows = [
                {"function": "cnc_absolute", "raw_function": "absolute", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_machine", "raw_function": "machine", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_statinfo", "raw_function": "statinfo", "function_family": "cnc", "category": "others"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            context = build_function_coverage_context(
                manifest,
                covered_functions={"cnc_absolute"},
                batch_size=2,
                scenario_type="coordinate_motion",
            )

            target_functions = [row["function"] for row in context["target_batch"]]
            self.assertNotIn("cnc_absolute", target_functions)
            self.assertIn("cnc_machine", target_functions)

    def test_coverage_context_can_select_multiple_scenario_segments_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "function_manifest.jsonl"
            rows = [
                {"function": "cnc_absolute", "raw_function": "absolute", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_machine", "raw_function": "machine", "function_family": "cnc", "category": "axis"},
                {"function": "cnc_rdtofs", "raw_function": "rdtofs", "function_family": "cnc", "category": "tool offset"},
                {"function": "cnc_wrtofs", "raw_function": "wrtofs", "function_family": "cnc", "category": "tool offset"},
                {"function": "pmc_rdpmcrng", "raw_function": "rdpmcrng", "function_family": "pmc", "category": "PMC"},
                {"function": "pmc_wrpmcrng", "raw_function": "wrpmcrng", "function_family": "pmc", "category": "PMC"},
                {"function": "cnc_statinfo", "raw_function": "statinfo", "function_family": "cnc", "category": "others"},
                {"function": "cnc_dwnstart3", "raw_function": "dwnstart3", "function_family": "cnc", "category": "CNC program"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            context = build_function_coverage_context(
                manifest,
                batch_size=6,
                scenario_type="coordinate_motion",
                max_segments=3,
            )

            segment_ids = [segment["scenario_batch_id"] for segment in context["segments"]]
            self.assertGreaterEqual(context["segment_count"], 2)
            self.assertIn("programmed_coordinate_motion", segment_ids)
            self.assertIn("tool_offset_work_coordinate_restore", segment_ids)
            self.assertTrue(all(row.get("segment_id") for row in context["selected_batch"]))


def write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            col = chr(ord("A") + col_index - 1)
            cells.append(
                f'<c r="{col}{row_index}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


if __name__ == "__main__":
    unittest.main()
