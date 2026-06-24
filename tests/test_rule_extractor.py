from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.rag.rule_extractor import (
    merge_rule_extraction_results,
    prepare_rule_extraction_batches,
)
from smart_traffic_agent.utils import write_jsonl


class RuleExtractorTests(unittest.TestCase):
    def test_prepare_batches_writes_prompts_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.jsonl"
            out_dir = root / "batches"
            write_jsonl(candidates, [sample_candidate("chunk-1"), sample_candidate("chunk-2")])

            manifests = prepare_rule_extraction_batches(candidates, out_dir, batch_size=1)

            self.assertEqual(len(manifests), 2)
            self.assertTrue((out_dir / "batch-0001.md").exists())
            self.assertTrue((out_dir / "manifest.jsonl").exists())
            prompt = (out_dir / "batch-0001.md").read_text(encoding="utf-8")
            self.assertIn("CHUNK_ID: chunk-1", prompt)
            self.assertIn("source_chunk_id", prompt)

    def test_merge_results_adds_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.jsonl"
            results = root / "results"
            output = root / "rule_chunks.jsonl"
            results.mkdir()
            write_jsonl(candidates, [sample_candidate("chunk-1")])
            (results / "batch-0001.json").write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "source_chunk_id": "chunk-1",
                                "rule_type": "nc_rule",
                                "scenario": "coordinate_motion",
                                "rule_text": "Use a safe G01 movement program to create coordinate-changing traffic.",
                                "traffic_value": ["high_distinguishability"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rules = merge_rule_extraction_results(candidates, results, output)

            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0]["source_chunk_id"], "chunk-1")
            self.assertEqual(rules[0]["source_file"], "manual.pdf")
            self.assertEqual(rules[0]["page_start"], 10)
            self.assertEqual(rules[0]["knowledge_type"], "traffic_generation_rule")


def sample_candidate(chunk_id: str) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_file": "manual.pdf",
        "page_start": 10,
        "page_end": 11,
        "section_title": "motion",
        "candidate_score": 2.0,
        "candidate_reason": ["test"],
        "text": "G01 feed movement changes position and feed speed.",
    }


if __name__ == "__main__":
    unittest.main()
