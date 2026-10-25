from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_dataset_quality import (
    build_protocol_sample,
    evaluate_api_coverage,
    evaluate_api_discriminability,
    evaluate_dataset,
    evaluate_run,
    evaluate_semantic_consistency,
    find_unstable_positions,
    load_ground_truth,
    load_run_invocations,
    mask_unstable_positions,
    samples_differ,
)


def invocation(
    api: str,
    payloads: list[bytes],
    *,
    parameters=None,
    timing: str = "during",
    context=None,
    return_code: int = 0,
) -> dict:
    return {
        "api": api,
        "parameters": parameters or {},
        "timing": timing,
        "context": context or {"device": "same"},
        "return_code": return_code,
        "protocol_sample": tuple(payloads),
        "valid": True,
    }


class DatasetQualityEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coverage_gt = {
            "required_api_groups": [
                {"name": f"group_{index}", "mode": "any", "accepted_apis": [f"cnc_api{index}"]}
                for index in range(4)
            ]
        }

    def test_01_coverage_all_four_groups(self) -> None:
        rows = [invocation(f"cnc_api{index}", [bytes([index + 1])]) for index in range(4)]

        result = evaluate_api_coverage(self.coverage_gt, rows)

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(len(result["covered_groups"]), 4)

    def test_02_coverage_three_of_four_groups(self) -> None:
        rows = [invocation(f"cnc_api{index}", [bytes([index + 1])]) for index in range(3)]

        result = evaluate_api_coverage(self.coverage_gt, rows)

        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["missing_groups"], ["group_3"])

    def test_03_parameter_sensitive_passes_when_two_of_three_samples_differ(self) -> None:
        rows = [
            invocation("cnc_test", [b"X"], parameters={"p": "A"}),
            invocation("cnc_test", [b"Y"], parameters={"p": "B"}),
            invocation("cnc_test", [b"Y"], parameters={"p": "C"}),
        ]

        result = evaluate_api_discriminability("cnc_test", "parameter_sensitive", rows)

        self.assertEqual(result["score"], 1)
        self.assertTrue(result["parameter_variation"]["satisfied"])

    def test_04_parameter_sensitive_fails_when_all_samples_same(self) -> None:
        rows = [
            invocation("cnc_test", [b"X"], parameters={"p": value})
            for value in ("A", "B", "C")
        ]

        result = evaluate_api_discriminability("cnc_test", "parameter_sensitive", rows)

        self.assertEqual(result["score"], 0)

    def test_05_timing_sensitive_passes_when_any_two_timings_differ(self) -> None:
        rows = [
            invocation("cnc_test", [b"A"], timing="before"),
            invocation("cnc_test", [b"B"], timing="during"),
            invocation("cnc_test", [b"B"], timing="after"),
        ]

        result = evaluate_api_discriminability("cnc_test", "timing_sensitive", rows)

        self.assertEqual(result["score"], 1)

    def test_06_timing_sensitive_fails_when_all_samples_same(self) -> None:
        rows = [invocation("cnc_test", [b"A"], timing=value) for value in ("before", "during", "after")]

        result = evaluate_api_discriminability("cnc_test", "timing_sensitive", rows)

        self.assertEqual(result["score"], 0)

    def test_07_fixed_api_is_excluded(self) -> None:
        rows = [invocation("cnc_test", [b"A"], timing=value) for value in ("before", "during", "after")]

        result = evaluate_api_discriminability("cnc_test", "fixed", rows)

        self.assertIsNone(result["score"])
        self.assertTrue(result["excluded"])

    def test_08_unstable_byte_is_masked_and_not_counted_as_parameter_effect(self) -> None:
        rows = [
            invocation("cnc_test", [bytes([1, 2, 3, 4])], parameters={"p": "A"}),
            invocation("cnc_test", [bytes([1, 255, 3, 4])], parameters={"p": "A"}),
            invocation("cnc_test", [bytes([1, 160, 3, 4])], parameters={"p": "A"}),
            invocation("cnc_test", [bytes([1, 99, 3, 4])], parameters={"p": "B"}),
        ]

        mask = find_unstable_positions([row["protocol_sample"] for row in rows[:3]])
        result = evaluate_api_discriminability("cnc_test", "parameter_sensitive", rows)

        self.assertEqual(mask, {0: {1}})
        self.assertEqual(result["unstable_positions"], {"0": [1]})
        self.assertEqual(result["score"], 0)

    def test_09_message_length_difference_is_real_difference(self) -> None:
        self.assertTrue(samples_differ((b"ABC",), (b"ABCD",), {}))

    def test_10_message_count_difference_is_real_difference(self) -> None:
        self.assertTrue(samples_differ((b"A",), (b"A", b"B"), {}))

    def test_11_one_parameter_configuration_is_not_evaluable(self) -> None:
        rows = [
            invocation("cnc_test", [b"A"], parameters={"p": 1}),
            invocation("cnc_test", [b"A"], parameters={"p": 1}),
        ]

        result = evaluate_api_discriminability("cnc_test", "parameter_sensitive", rows)

        self.assertIsNone(result["score"])
        self.assertFalse(result["parameter_variation"]["comparable"])

    def test_12_one_timing_is_not_evaluable(self) -> None:
        rows = [invocation("cnc_test", [b"A"], timing="before") for _ in range(2)]

        result = evaluate_api_discriminability("cnc_test", "timing_sensitive", rows)

        self.assertIsNone(result["score"])
        self.assertFalse(result["timing_variation"]["comparable"])

    def test_13_semantic_consistency_all_aligned(self) -> None:
        rows = [{"valid": True, "aligned": True, "invocation_id": index} for index in range(100)]

        result = evaluate_semantic_consistency(rows)

        self.assertEqual(result["valid_invocations"], 100)
        self.assertEqual(result["aligned_invocations"], 100)
        self.assertEqual(result["score"], 1.0)

    def test_14_semantic_consistency_95_of_100(self) -> None:
        rows = [
            {"valid": True, "aligned": index < 95, "invocation_id": index}
            for index in range(100)
        ]

        result = evaluate_semantic_consistency(rows)

        self.assertEqual(result["aligned_invocations"], 95)
        self.assertEqual(result["misaligned_invocations"], 5)
        self.assertEqual(result["score"], 0.95)

    def test_protocol_sample_rejects_transport_headers_and_unreassembled_tcp(self) -> None:
        self.assertIsNone(build_protocol_sample({"contains_transport_headers": True, "application_messages": ["01"]}))
        self.assertIsNone(build_protocol_sample({"tcp_reassembled": False, "application_messages": ["01"]}))
        self.assertEqual(
            build_protocol_sample({"tcp_reassembled": True, "application_messages": [{"payload_hex": "01 02"}, "0304"]}),
            (b"\x01\x02", b"\x03\x04"),
        )

    def test_dataset_quality_ground_truth_reuses_rq1_api_groups(self) -> None:
        for scenario in ("coordinate_motion", "program_management", "tool_management"):
            ground_truth = load_ground_truth(scenario)
            source_path = Path(__file__).resolve().parents[1] / "evaluation" / "ground_truth" / f"{scenario}.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source_groups = {
                row["id"]: set(row["accepted_apis"])
                for row in source.get("required_operations", [])
            }
            for group in source.get("operation_groups", []):
                source_groups[group["id"]] = {
                    api
                    for option in group.get("operation_options", [])
                    for api in option.get("accepted_apis", [])
                }
            dataset_groups = {
                row["name"]: set(row["accepted_apis"])
                for row in ground_truth["required_api_groups"]
            }
            self.assertEqual(ground_truth["scenario"], scenario)
            self.assertTrue(ground_truth["source_ground_truth"].startswith("../"))
            self.assertEqual(dataset_groups, source_groups)
            self.assertTrue(set(ground_truth["api_variation_expectations"].values()) <= {
                "fixed", "parameter_sensitive", "timing_sensitive", "parameter_and_timing_sensitive"
            })

    def test_mask_preserves_message_shape(self) -> None:
        masked = mask_unstable_positions((b"ABC", b"DE"), {0: {1}})
        self.assertEqual(masked, ((65, None, 67), (68, 69)))

    def test_complete_transaction_sidecar_proves_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "coordinate_motion" / "run_001"
            execution = run_dir / "execution"
            execution.mkdir(parents=True)
            (run_dir / "plan.json").write_text(
                json.dumps({"scenario_type": "coordinate_motion"}), encoding="utf-8"
            )
            log = {
                "timestamp": "2026-09-10T10:00:00.123456",
                "protocol_function": "cnc_allclibhndl3",
                "input_parameters": {"host": "127.0.0.1"},
                "status_code": 0,
                "invocation_id": "call-1",
                "transaction_id": "tx-1",
                "phase": "before",
                "device_context": {"run_state": "idle"},
                "semantic_label": "connect_controller",
            }
            (execution / "api_logs.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")
            transaction = {
                "transaction_id": "tx-1",
                "invocation_id": "call-1",
                "api": "cnc_allclibhndl3",
                "decoded_parameters": {"host": "127.0.0.1"},
                "decoded_return_code": "EW_OK",
                "invocation_order": 0,
                "request_response_associated": True,
                "tcp_reassembled": True,
                "contains_transport_headers": False,
                "application_messages": [{"payload_hex": "0102"}, {"payload_hex": "0304"}],
                "annotation": "connect_controller",
                "traffic_observation_complete": True,
            }
            (execution / "protocol_transactions.jsonl").write_text(
                json.dumps(transaction) + "\n", encoding="utf-8"
            )

            rows, support = load_run_invocations(run_dir)
            semantic = evaluate_semantic_consistency(rows)

            self.assertTrue(support["sufficient_for_semantic_consistency"])
            self.assertEqual(semantic["score"], 1.0)

    def test_legacy_log_without_transaction_is_not_scored_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "coordinate_motion" / "run_001"
            execution = run_dir / "execution"
            execution.mkdir(parents=True)
            (run_dir / "plan.json").write_text(
                json.dumps({"scenario_type": "coordinate_motion"}), encoding="utf-8"
            )
            log = {
                "timestamp": "2026-09-10T10:00:00",
                "protocol_function": "cnc_allclibhndl3",
                "input_parameters": {"host": "127.0.0.1"},
                "status_code": 0,
                "phase": "before",
                "semantic_label": "connect_controller",
            }
            (execution / "api_logs.jsonl").write_text(json.dumps(log) + "\n", encoding="utf-8")

            run_result = evaluate_run(run_dir)
            dataset_result = evaluate_dataset(root)

            self.assertIsNone(run_result["api_coverage"]["score"])
            self.assertIsNone(run_result["discriminability"]["score"])
            self.assertIsNone(run_result["semantic_consistency"]["score"])
            self.assertNotIn("overall_dataset_quality", dataset_result)
            self.assertIsNone(dataset_result["dataset_macro"]["api_coverage"]["score"])


if __name__ == "__main__":
    unittest.main()
