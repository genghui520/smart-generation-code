from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smart_traffic_agent.agents.planner import (
    attach_protocol_functions,
    build_api_candidate_pool,
    compact_rag_context,
    compact_representative_steps,
    infer_scenario,
    infer_coverage_intent,
    load_scenario_templates,
    plan_steps_from_llm_rows,
    retrieve_planning_context,
    select_scenario_templates,
    summarize_retrieval,
)
from smart_traffic_agent.models import ExecutionPlan, KnowledgeChunk, NcProgramSpec, PlanStep, RetrievedChunk


class InMemoryKnowledgeBase:
    def search_scenario_organization(self, query: str, scenario: str, *, top_k: int = 2) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=KnowledgeChunk(
                    chunk_id=f"scenario-{scenario}",
                    text=f"Scenario-centered knowledge for {scenario}",
                    metadata={"source_type": "scenario", "scenario": scenario},
                ),
                score=0.95,
            )
        ]

    def search_rules_by_type(self, query: str, rule_type: str, *, top_k: int = 3) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=KnowledgeChunk(
                    chunk_id=f"{rule_type}-001",
                    text=f"{rule_type} for spindle speed change",
                    metadata={"source_type": "rule", "rule_type": rule_type, "scenario": "spindle_speed_change"},
                ),
                score=0.9,
            )
        ]

    def search_api(self, query: str, *, top_k: int = 6) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=KnowledgeChunk(
                    chunk_id="cnc-rdspmeter",
                    text="Function: cnc_rdspmeter reads spindle meter data.",
                    metadata={"source_type": "api", "function": "cnc_rdspmeter"},
                ),
                score=0.8,
            )
        ]

    def search(self, query: str, *, top_k: int = 6) -> list[RetrievedChunk]:
        return []


class PlannerTests(unittest.TestCase):
    def test_llm_step_normalizes_compound_protocol_function_to_atomic_calls(self) -> None:
        steps = plan_steps_from_llm_rows(
            [{
                "step_id": "UPLOAD-001",
                "phase": "before",
                "interface_name": "UploadProgram",
                "protocol_function": "cnc_dwnstart3/cnc_download3/cnc_dwnend3",
            }]
        )

        self.assertEqual(
            [call["protocol_function"] for call in steps[0].api_calls],
            ["cnc_dwnstart3", "cnc_download3", "cnc_dwnend3"],
        )

    def test_compact_representative_steps_keeps_supported_unique_phases(self) -> None:
        steps = [
            PlanStep(step_id="S1", phase="before", action="upload", interface_name="UploadProgram"),
            PlanStep(step_id="S2", phase="during", action="read", interface_name="ReadPosition"),
            PlanStep(step_id="S3", phase="during", action="read again", interface_name="ReadPosition"),
            PlanStep(step_id="S4", phase="after", action="stop", interface_name="StopProgram"),
        ]

        compacted = compact_representative_steps(steps, max_steps=3)

        self.assertEqual([(step.interface_name, step.phase) for step in compacted], [
            ("UploadProgram", "before"),
            ("ReadPosition", "during"),
            ("StopProgram", "after"),
        ])

    def test_plan_steps_tolerates_symbolic_repeat_placeholder(self) -> None:
        steps = plan_steps_from_llm_rows(
            [
                {
                    "step_id": "S001",
                    "phase": "during",
                    "action": "guarded cycle start",
                    "interface_name": "StartProgram",
                    "protocol_function": "ncguide_ui_cycle_start",
                    "repeat": "expected_nc_segment_count",
                    "interval_seconds": "poll_interval_seconds",
                }
            ]
        )

        self.assertEqual(steps[0].repeat, 1)
        self.assertEqual(steps[0].interval_seconds, 0.0)

    def test_infer_comprehensive_focas_task(self) -> None:
        self.assertEqual(
            infer_scenario("生成全面的、多样性的 FOCAS 协议流量"),
            "comprehensive_focas_traffic",
        )

    def test_infer_scenario_from_chinese_task(self) -> None:
        self.assertEqual(
            infer_scenario("生成主轴转速变化流量，采集主轴速度和运行状态"),
            "spindle_speed_change",
        )

    def test_retrieve_planning_context_summarizes_rag_sources(self) -> None:
        context = retrieve_planning_context(
            InMemoryKnowledgeBase(),  # type: ignore[arg-type]
            "generate spindle speed change traffic",
            "spindle_speed_change",
        )
        summary = summarize_retrieval(context)

        self.assertIn("scenario_organization", summary)
        self.assertIn("nc_rule", summary)
        self.assertIn("api", summary)
        self.assertEqual(summary["api"][0]["function"], "cnc_rdspmeter")  # type: ignore[index]

    def test_select_scenario_templates_prefers_matching_scene(self) -> None:
        templates = [
            {
                "template_id": "tpl-coordinate",
                "scenario_id": "cluster_coordinate",
                "scenario_name": "coordinate_feed_motion",
                "coverage_priority": "high",
                "cluster_member_count": 3,
                "goal": "template coordinate goal",
                "main_apis": ["cnc_rdposition"],
                "main_objects": ["axis"],
                "expected_signals": ["axis_change"],
            },
            {
                "template_id": "tpl-spindle",
                "scenario_id": "cluster_spindle",
                "scenario_name": "spindle_control",
                "coverage_priority": "medium",
                "cluster_member_count": 1,
                "goal": "template spindle goal",
                "main_apis": ["cnc_acts"],
                "main_objects": ["spindle"],
                "expected_signals": ["spindle_speed"],
            },
        ]

        selected = select_scenario_templates(
            "generate coordinate traffic",
            "coordinate_motion",
            templates,
        )

        self.assertEqual([template["template_id"] for template in selected], ["tpl-coordinate"])

    def test_load_scenario_templates_ignores_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template_path = Path(tmp) / "templates.json"
            template_path.write_text(
                json.dumps({"templates": [{"template_id": "tpl-1"}, "bad-row"]}),
                encoding="utf-8",
            )

            templates = load_scenario_templates(template_path)

            self.assertEqual(templates, [{"template_id": "tpl-1"}])

    def test_llm_steps_preserve_parameter_generation_strategy(self) -> None:
        steps = plan_steps_from_llm_rows(
            [
                {
                    "step_id": "S001",
                    "phase": "during",
                    "action": "read positions for several modes",
                    "interface_name": "ReadPosition",
                    "parameters": {"axes": "XYZ"},
                    "parameter_generation": [
                        {"name": "type", "kind": "enum", "values": [-1, 0, 1], "reason": "finite modes"},
                        {"name": "axis_count", "kind": "range", "min": 1, "max": 8, "samples": [1, 4, 8]},
                        {"name": "host", "kind": "fixed", "value": "127.0.0.1"},
                    ],
                }
            ]
        )

        strategy = steps[0].parameters["parameter_generation"]

        self.assertEqual(strategy[0]["kind"], "enum")
        self.assertEqual(strategy[0]["values"], [-1, 0, 1])
        self.assertEqual(strategy[1]["samples"], [1, 4, 8])
        self.assertEqual(strategy[2]["value"], "127.0.0.1")

    def test_attach_protocol_functions_preserves_llm_selected_api(self) -> None:
        steps = plan_steps_from_llm_rows(
            [
                {
                    "step_id": "S001",
                    "phase": "during",
                    "action": "read remaining movement inferred from API knowledge",
                    "interface_name": "ReadDistanceToGo",
                    "protocol_function": "cnc_distance",
                    "parameters": {},
                    "parameter_generation": [
                        {"name": "axis_count", "kind": "range", "min": 1, "max": 8, "samples": [1, 3, 8]}
                    ],
                }
            ]
        )

        attach_protocol_functions(steps)

        self.assertEqual(steps[0].protocol_function, "cnc_distance")
        self.assertEqual(steps[0].parameters["parameter_generation"][0]["kind"], "range")

    def test_infer_coverage_intent_preserves_full_simulator_goal(self) -> None:
        intent = infer_coverage_intent(
            "我要获得全量的 FOCAS 流量，所以相关 API 都需要囊括，我连的是仿真器",
            "simulator",
        )

        self.assertTrue(intent["full_traffic_coverage"])
        self.assertTrue(intent["simulator_target"])
        self.assertIn("Simulator write/control side effects are acceptable", intent["planner_policy"])

    def test_infer_coverage_intent_detects_full_function_coverage(self) -> None:
        intent = infer_coverage_intent(
            "focas_funcs.xlsx 里面的函数需要全覆盖，连接 NCGuide 仿真器",
            "ncguide-generated-cpp",
        )

        self.assertTrue(intent["full_function_coverage"])
        self.assertTrue(intent["full_traffic_coverage"])
        self.assertTrue(intent["explicit_full_function_coverage_request"])

    def test_infer_coverage_intent_defaults_to_full_function_coverage(self) -> None:
        intent = infer_coverage_intent(
            "generate coordinate motion traffic",
            "ncguide-generated-cpp",
        )

        self.assertTrue(intent["full_function_coverage"])
        self.assertTrue(intent["full_traffic_coverage"])
        self.assertFalse(intent["explicit_full_function_coverage_request"])
        self.assertEqual(intent["coverage_default"], "focas_system_default_full_high_quality_coverage")

    def test_build_api_candidate_pool_uses_rag_templates_and_draft_steps(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-test",
            task_id="task-test",
            scenario_type="coordinate_motion",
            scenario_goal="full coordinate traffic",
            target_environment="simulator",
            nc_program_type="straight_interpolation_motion",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[
                PlanStep(
                    "S001",
                    "during",
                    "read canonical position",
                    "ReadPosition",
                    protocol_function="cnc_rdposition",
                )
            ],
            expected_outputs=[],
            rag_context={
                "api": [
                    {"function": "cnc_absolute", "preview": "Reads absolute position."},
                    {"function": "cnc_wractpt", "preview": "Writes active program pointer."},
                    {"function": "cnc_setpglock", "preview": "Sets program lock."},
                    {"function": "cnc_rdmdiprgstat", "preview": "Reads MDI program status."},
                ],
                "selected_scenario_templates": [
                    {
                        "template_id": "tpl-motion",
                        "scenario_name": "coordinate_feed_motion",
                        "main_apis": ["cnc_rdposition", "cnc_actf"],
                    }
                ],
            },
        )

        pool = build_api_candidate_pool(plan)
        by_function = {item["protocol_function"]: item for item in pool}

        self.assertIn("cnc_rdposition", by_function)
        self.assertIn("cnc_absolute", by_function)
        self.assertIn("cnc_wractpt", by_function)
        self.assertIn("cnc_setpglock", by_function)
        self.assertIn("cnc_rdmdiprgstat", by_function)
        self.assertIn("selected_template:tpl-motion", by_function["cnc_actf"]["sources"])

    def test_build_api_candidate_pool_uses_manifest_batch(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-test",
            task_id="task-test",
            scenario_type="comprehensive_focas_traffic",
            scenario_goal="cover all manifest functions",
            target_environment="ncguide-generated-cpp",
            nc_program_type="coverage",
            nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name="O1234"),
            steps=[],
            expected_outputs=[],
            rag_context={
                "function_coverage_manifest": {
                    "enabled": True,
                    "selected_batch": [
                        {
                            "function": "cnc_rdnodenum",
                            "raw_function": "rdnodenum",
                            "category": "library handle, node",
                            "description": "Read the number of node",
                        },
                        {
                            "function": "pmc_rdpmcrng",
                            "raw_function": "rdpmcrng",
                            "category": "PMC",
                            "description": "Read PMC data",
                        },
                    ],
                }
            },
        )

        pool = build_api_candidate_pool(plan)
        by_function = {item["protocol_function"]: item for item in pool}

        self.assertIn("cnc_rdnodenum", by_function)
        self.assertIn("pmc_rdpmcrng", by_function)
        self.assertIn("function_manifest_target_batch", by_function["pmc_rdpmcrng"]["sources"])

    def test_compact_rag_context_keeps_manifest_batch_summary(self) -> None:
        compact = compact_rag_context(
            {
                "function_coverage_manifest": {
                    "enabled": True,
                    "target_function_count": 704,
                    "scenario_batch_id": "programmed_coordinate_motion",
                    "main_state_driver": "upload_select_run_single_block_nc_program",
                    "selected_batch_size": 1,
                    "segment_count": 1,
                    "segments": [
                        {
                            "segment_id": "01_programmed_coordinate_motion",
                            "scenario_batch_id": "programmed_coordinate_motion",
                            "main_state_driver": "upload_select_run_single_block_nc_program",
                            "quality_target": "position/feed/status variation",
                            "nc_program_required": True,
                            "target_batch": [{"function": "cnc_rdnodenum"}],
                            "support_batch": [{"function": "cnc_statinfo"}],
                        }
                    ],
                    "selected_batch": [
                        {
                            "function": "cnc_rdnodenum",
                            "raw_function": "rdnodenum",
                            "coverage_role": "target",
                            "category": "library handle, node",
                            "description": "Read the number of node",
                            "symbol_candidates": ["cnc_rdnodenum", "rdnodenum"],
                        }
                    ],
                }
            }
        )

        manifest = compact["function_coverage_manifest"]
        self.assertEqual(manifest["target_function_count"], 704)
        self.assertEqual(manifest["scenario_batch_id"], "programmed_coordinate_motion")
        self.assertEqual(manifest["segments"][0]["segment_id"], "01_programmed_coordinate_motion")
        self.assertEqual(manifest["segments"][0]["support_functions"], ["cnc_statinfo"])
        self.assertEqual(manifest["selected_batch"][0]["function"], "cnc_rdnodenum")
        self.assertEqual(manifest["selected_batch"][0]["coverage_role"], "target")


if __name__ == "__main__":
    unittest.main()
