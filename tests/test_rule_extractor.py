from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.llm import LlmClient, LlmConfig
from smart_traffic_agent.rag.rule_extractor import (
    extract_rules_with_llm,
    merge_rule_extraction_merged_json,
    merge_rule_extraction_results,
    prepare_rule_extraction_batches,
)
from smart_traffic_agent.rag.scenario_taxonomy import load_taxonomy
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

    def test_merge_merged_json_repairs_source_chunk_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.jsonl"
            merged_json = root / "merged.json"
            output = root / "rule_chunks.jsonl"
            write_jsonl(
                candidates,
                [
                    sample_candidate(
                        "fanuc-B-64605CM_01-p0034-1-3-2-display-content-005"
                    )
                ],
            )
            merged_json.write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "source_chunk_id": "fanucB-64605CM_01-p0034-wrong-title-005",
                                "rule_type": "collection_rule",
                                "scenario": "diagnostic_query",
                                "rule_text": "Read diagnostic display data before and after the operation.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rules = merge_rule_extraction_merged_json(candidates, merged_json, output)

            self.assertEqual(len(rules), 1)
            self.assertEqual(
                rules[0]["source_chunk_id"],
                "fanuc-B-64605CM_01-p0034-1-3-2-display-content-005",
            )
            self.assertTrue(rules[0]["source_chunk_id_repaired"])

    def test_merge_merged_json_repairs_source_chunk_id_with_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.jsonl"
            merged_json = root / "merged.json"
            output = root / "rule_chunks.jsonl"
            write_jsonl(
                candidates,
                [
                    sample_candidate("fanuc-B-64604CM_01-p0656-7-alarm-001"),
                    sample_candidate("fanuc-B-64604CM_01-p0656-7-1-alarm-display-001"),
                ],
            )
            merged_json.write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "source_chunk_id": "fanuc-B-64604CM_01-p0656-7-1-garbled-001",
                                "rule_type": "operation_rule",
                                "scenario": "alarm_query",
                                "rule_text": "Open the alarm display before collecting alarm traffic.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rules = merge_rule_extraction_merged_json(candidates, merged_json, output)

            self.assertEqual(
                rules[0]["source_chunk_id"],
                "fanuc-B-64604CM_01-p0656-7-1-alarm-display-001",
            )

    def test_extract_rules_with_llm_writes_json_results(self) -> None:
        if os.getenv("RUN_LLM_INTEGRATION_TESTS") != "1":
            self.skipTest("set RUN_LLM_INTEGRATION_TESTS=1 and real LLM credentials to run this integration test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.jsonl"
            results = root / "results"
            write_jsonl(candidates, [sample_candidate("chunk-1")])

            llm_client = LlmClient.from_config(
                LlmConfig(
                    provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
                    model=os.getenv("LLM_MODEL", "gpt-5.6-sol"),
                    base_url=os.getenv("LLM_BASE_URL", "https://fast.smartaipro.cn/v1"),
                    api_key_env=os.getenv("LLM_API_KEY_ENV", "SMARTAIPRO_API_KEY"),
                )
            )
            paths = extract_rules_with_llm(
                candidates,
                results,
                llm_client=llm_client,
                batch_size=1,
            )

            self.assertEqual(len(paths), 1)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["rules"][0]["source_chunk_id"], "chunk-1")

    def test_custom_taxonomy_preserves_protocol_and_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            taxonomy_path = root / "scenario_taxonomy.json"
            candidates = root / "candidates.jsonl"
            results = root / "results"
            output = root / "rule_chunks.jsonl"
            results.mkdir()
            write_jsonl(candidates, [sample_candidate("modbus-chunk-1")])
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "protocol": "modbus",
                        "rule_types": {
                            "api_rule": "Define which Modbus function codes or register operations to call.",
                            "collection_rule": "Define when to capture request and response traffic.",
                        },
                        "scenarios": [
                            {
                                "scenario_id": "register_read",
                                "name": "Register read traffic",
                                "goal": "Generate traffic for reading coils or holding registers.",
                                "traffic_value": ["high_coverage"],
                                "typical_nc_program": [],
                                "operation_phases": ["before", "during", "after"],
                                "recommended_api_functions": ["read_holding_registers"],
                                "distinguishing_signals": ["function_code_03"],
                                "allowed_rule_types": ["api_rule", "collection_rule"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (results / "batch-0001.json").write_text(
                json.dumps(
                    {
                        "rules": [
                            {
                                "source_chunk_id": "modbus-chunk-1",
                                "rule_type": "api_rule",
                                "scenario": "register_read",
                                "rule_text": "Use register read operations to generate Modbus query-response traffic.",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            taxonomy = load_taxonomy(taxonomy_path)
            rules = merge_rule_extraction_results(candidates, results, output, taxonomy=taxonomy)

            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0]["protocol"], "modbus")
            self.assertEqual(rules[0]["scenario"], "register_read")
            self.assertEqual(rules[0]["rule_type"], "api_rule")
            self.assertTrue(rules[0]["rule_id"].startswith("modbus-rule-"))


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
