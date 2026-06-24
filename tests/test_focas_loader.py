from __future__ import annotations

import unittest

from smart_traffic_agent.rag.focas_loader import (
    function_page_to_xml_path,
    parse_function_refs,
)


class FocasLoaderTests(unittest.TestCase):
    def test_function_page_to_xml_path(self) -> None:
        self.assertEqual(
            function_page_to_xml_path("../Position/cnc_rdposition.htm"),
            "Position/cnc_rdposition.xml",
        )

    def test_parse_function_refs(self) -> None:
        xml = b"""<?xml version="1.0" encoding="Shift_JIS"?>
<root>
  <chapter>
    <title>CNC: Function related to controlled axis/spindle</title>
    <item>
      <fname>cnc_rdposition</fname>
      <fpage>../Position/cnc_rdposition.htm</fpage>
      <explanation>Read the position</explanation>
    </item>
  </chapter>
</root>
"""
        refs = parse_function_refs(xml)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].name, "cnc_rdposition")
        self.assertEqual(refs[0].xml_path, "Position/cnc_rdposition.xml")


if __name__ == "__main__":
    unittest.main()
