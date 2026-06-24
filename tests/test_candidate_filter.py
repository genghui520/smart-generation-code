from __future__ import annotations

import unittest

from smart_traffic_agent.rag.candidate_filter import score_candidate


class CandidateFilterTests(unittest.TestCase):
    def test_drops_directory(self) -> None:
        result = score_candidate({"section_title": "目录", "text": "自动运行 10 参数 20"})
        self.assertTrue(result["drop"])

    def test_keeps_traffic_related_section(self) -> None:
        result = score_candidate(
            {
                "section_title": "自动运行",
                "manual_type": "operation_manual",
                "text": "选择程序后启动自动方式，运行状态、坐标和进给速度会变化。",
            }
        )
        self.assertFalse(result["drop"])
        self.assertGreaterEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
