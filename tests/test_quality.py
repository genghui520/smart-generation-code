from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.models import (
    ApiCallLog,
    ExecutionPlan,
    ExecutionResult,
    NcProgramSpec,
    PlanStep,
)
from smart_traffic_agent.quality import build_quality_observation, collect_quality_metrics, output_variation_is_sufficient


def plan_requiring_feed_variation() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan-quality",
        task_id="quality001",
        scenario_type="coordinate_motion",
        scenario_goal="capture feed variation",
        target_environment="simulator",
        nc_program_type="coordinate",
        nc_program_requirements=[],
        nc_program_spec=NcProgramSpec(program_name="O1234"),
        steps=[
            PlanStep("S001", "during", "read feed", "ReadFeedSpeed", {}, repeat=3),
            PlanStep("S002", "during", "read position", "ReadPosition", {}, repeat=3),
        ],
        expected_outputs=[],
        rag_context={
            "quality_targets": {
                "min_feed_samples": 3,
                "expect_feed_variation": True,
                "expect_position_variation": True,
            }
        },
    )


class TrafficQualityTests(unittest.TestCase):
    def test_flags_constant_feed_and_position_as_replan_issue(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id=f"F{index}",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={},
                status_code=0,
                response={"data": "actual_feed=0"},
            )
            for index in range(3)
        ]
        logs.extend(
            [
                ApiCallLog(
                    timestamp="t",
                    task_id="quality001",
                    step_id=f"P{index}",
                    phase="during",
                    interface_name="ReadPosition",
                    input_parameters={},
                    status_code=0,
                    response={"data": "X=10;Y=10;Z=10"},
                )
                for index in range(3)
            ]
        )
        result = ExecutionResult(
            task_id="quality001",
            success=True,
            api_logs=logs,
            capture_events=[],
            output_dir=Path(tempfile.gettempdir()),
        )

        assessment = build_quality_observation(plan_requiring_feed_variation(), result)

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.issues, [])
        self.assertEqual(assessment.metrics["feed_sample_count"], 3)
        self.assertEqual(assessment.metrics["feed_unique_count"], 1)
        self.assertEqual(assessment.metrics["position_sample_count"], 3)
        self.assertEqual(assessment.metrics["position_unique_count"], 1)
        self.assertTrue(any("RouterAgent" in item for item in assessment.recommendations))

    def test_collects_generated_cpp_feed_and_position_fields(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P1",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "axis_count=3;"
                        "abs_X=46.998(raw=46998,dec=3,unit=0);"
                        "abs_Y=10.000(raw=10000,dec=3,unit=0);"
                        "abs_Z=20.000(raw=20000,dec=3,unit=0);"
                        "mach_X=53.002(raw=53002,dec=3,unit=0)"
                    )
                },
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P2",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "axis_count=3;"
                        "abs_X=47.129(raw=47129,dec=3,unit=0);"
                        "abs_Y=10.500(raw=10500,dec=3,unit=0);"
                        "abs_Z=20.000(raw=20000,dec=3,unit=0)"
                    )
                },
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="F1",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={},
                status_code=0,
                response={"data": "raw_shorts=0|0|35|0|0|0|0|0;candidate_feed_long0=0;candidate_feed_long1=35"},
            ),
        ]

        metrics = collect_quality_metrics(logs)

        self.assertEqual(metrics["feed_sample_count"], 1)
        self.assertEqual(metrics["feed_values_preview"], [35.0])
        self.assertEqual(metrics["position_sample_count"], 2)
        self.assertEqual(metrics["position_unique_count"], 2)

    def test_output_variation_requires_program_completion(self) -> None:
        metrics = {
            "changed_output_parameter_count": 6,
            "feed_sample_count": 20,
            "feed_unique_count": 3,
            "position_sample_count": 20,
            "position_unique_count": 5,
            "run_active_count": 10,
            "motion_active_count": 10,
            "program_completed": False,
        }

        self.assertFalse(output_variation_is_sufficient(metrics))
        metrics["program_completed"] = True
        self.assertTrue(output_variation_is_sufficient(metrics))

    def test_collects_program_completion_gate_metric(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="PROGRAM_COMPLETION",
                phase="after",
                interface_name="program_completion_gate",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "program_completion_gate=true;completed=true;timeout=false;"
                        "waited_ms=1200;last_run=0;last_motion=0;distance_to_go=0"
                    )
                },
            )
        ]

        metrics = collect_quality_metrics(logs)

        self.assertEqual(metrics["program_completion_gate_count"], 1)
        self.assertTrue(metrics["program_completed"])

    def test_collects_axis_group_position_and_feed_data_fields(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P1",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "axes=3;"
                        "axis1_name=X;abs=0.0000;mach=0.0000;rel=0.0000;dist=0.0000;"
                        "axis2_name=Y;abs=0.0000;mach=0.0000;rel=0.0000;dist=0.0000;"
                        "axis3_name=Z;abs=0.0000;mach=0.0000;rel=0.0000;dist=0.0000"
                    )
                },
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="F1",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={},
                status_code=0,
                response={"data": "feed_data=0;dummy0=0;dummy1=0"},
            ),
        ]

        metrics = collect_quality_metrics(logs)

        self.assertEqual(metrics["feed_sample_count"], 1)
        self.assertEqual(metrics["feed_values_preview"], [0.0])
        self.assertEqual(metrics["position_sample_count"], 1)
        self.assertEqual(metrics["position_values_preview"], [(0.0, 0.0, 0.0)])

    def test_collects_indexed_axis_position_fields(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P1",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "axes=3;"
                        "axis1_name=\x01\x00;axis1_abs=95;axis1_mach=95;"
                        "axis2_name=\x01\x00;axis2_abs=15;axis2_mach=15;"
                        "axis3_name=\x01\x00;axis3_abs=8;axis3_mach=8"
                    )
                },
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P2",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={},
                status_code=0,
                response={
                    "data": (
                        "axes=3;"
                        "axis1_abs=96;axis1_mach=96;"
                        "axis2_abs=15;axis2_mach=15;"
                        "axis3_abs=8;axis3_mach=8"
                    )
                },
            ),
        ]

        metrics = collect_quality_metrics(logs)

        self.assertEqual(metrics["position_sample_count"], 2)
        self.assertEqual(metrics["position_unique_count"], 2)
        self.assertEqual(metrics["position_values_preview"][0], (95.0, 15.0, 8.0))

    def test_uses_input_and_return_parameter_names_for_metrics(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="P1",
                phase="during",
                interface_name="ReadPosition",
                input_parameters={"raw": "type=-1;axes=X,Y,Z;max_axes=8"},
                status_code=0,
                response={
                    "data": (
                        "axes=3;"
                        "axis1_abs=95000;axis1_abs_dec=3;"
                        "axis2_abs=15000;axis2_abs_dec=3;"
                        "axis3_abs=8000;axis3_abs_dec=3"
                    )
                },
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="F1",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={"raw": "sample_role=during_motion"},
                status_code=0,
                response={"data": "feed_value=120;raw=00"},
            ),
        ]

        metrics = collect_quality_metrics(logs)

        self.assertEqual(metrics["position_values_preview"], [(95.0, 15.0, 8.0)])
        self.assertEqual(metrics["feed_values_preview"], [120.0])
        self.assertIn("axes", metrics["input_parameter_names"]["ReadPosition"])
        self.assertIn("axis1_abs", metrics["return_parameter_names"]["ReadPosition"])
        self.assertIn("feed_value", metrics["return_parameter_names"]["ReadFeedSpeed"])

    def test_tracks_output_parameter_variation(self) -> None:
        logs = [
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="F1",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={"raw": "sample_role=A"},
                status_code=0,
                response={"data": "feed_value=0;constant_flag=1"},
            ),
            ApiCallLog(
                timestamp="t",
                task_id="quality001",
                step_id="F2",
                phase="during",
                interface_name="ReadFeedSpeed",
                input_parameters={"raw": "sample_role=B"},
                status_code=0,
                response={"data": "feed_value=120;constant_flag=1"},
            ),
        ]

        metrics = collect_quality_metrics(logs)
        variation = metrics["output_parameter_variation"]["ReadFeedSpeed"]

        self.assertTrue(variation["feed_value"]["changed"])
        self.assertEqual(variation["feed_value"]["unique_count"], 2)
        self.assertFalse(variation["constant_flag"]["changed"])
        self.assertEqual(metrics["changed_output_parameter_count"], 1)


if __name__ == "__main__":
    unittest.main()
