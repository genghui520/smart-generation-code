from __future__ import annotations

import unittest

from smart_traffic_agent.rag.fanuc_manual_loader import split_with_overlap


class FanucManualLoaderTests(unittest.TestCase):
    def test_split_with_overlap(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        parts = split_with_overlap(text, max_chars=160, overlap=20)
        self.assertGreater(len(parts), 1)
        self.assertTrue(parts[0])
        self.assertTrue(parts[1])


if __name__ == "__main__":
    unittest.main()
