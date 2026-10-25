from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

from smart_traffic_agent.agents.planner import (
    apply_llm_planning_decision,
    attach_protocol_functions,
    build_api_candidate_pool,
    build_plan,
    compact_rag_context,
    compact_representative_steps,
    infer_scenario,
    infer_coverage_intent,
    load_scenario_templates,
    materialize_included_api_candidates,
    plan_steps_from_llm_rows,
    parse_planning_markdown,
    retrieve_planning_context,
    select_scenario_templates,
    summarize_retrieval,
    summarize_long_term_memories,
)
from smart_traffic_agent.models import ExecutionPlan, KnowledgeChunk, NcProgramSpec, PlanStep, RetrievedChunk
from smart_traffic_agent.code_spec import build_code_spec


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


class ErrorExperienceSummaryTests(unittest.TestCase):
    def test_preserves_evidence_and_validation_status(self) -> None:
        summary = summarize_long_term_memories(
            [
                {
                    "memory_id": "experience:axis-length",
                    "record_type": "curated_error_experience",
                    "experience_id": "axis-length",
                    "title": "Axis array length uses axis count",
                    "status": "verified",
                    "confidence": "high",
                    "trigger": "EW_LENGTH from axis APIs",
                    "root_cause": "Used sizeof(ODBAXIS) as length.",
                    "evidence": ["The API length is an axis count."],
                    "fix": "Pass the requested axis count.",
                    "prevention": "Check each API's documented units.",
                    "related_apis": ["cnc_absolute"],
                    "limitations": "Check per-API prototype.",
                }
            ]
        )
        self.assertEqual(summary[0]["status"], "verified")
        self.assertEqual(summary[0]["evidence"], ["The API length is an axis count."])
        self.assertEqual(summary[0]["fix"], "Pass the requested axis count.")

class PlannerTests(unittest.TestCase):
    def test_llm_planning_uses_plan_permissions_and_preserves_motion_lifecycle(self) -> None:
        for target, use_permissions, needs_motion in (
            ("cnc_rdaxisdata", True, True),
            ("cnc_rdaxisdata", False, True),
            ("cnc_download3", True, False),
        ):
            with self.subTest(target=target, use_permissions=use_permissions):
                plan = build_plan("scope-regression", "general_status_collection", "ncguide-generated-cpp", {})
                plan.rag_context["function_coverage_manifest"] = {
                    "selected_batch": [{"function": target, "coverage_role": "target"}],
                    "segments": [],
                }
                if use_permissions:
                    plan.rag_context["task_permissions"] = {
                        "coverage_current_batch_functions": [target],
                    }
                client = Mock()
                client.invoke_text.return_value = json.dumps({
                    "scenario_goal": "capture selected API",
                    "quality_analysis": {"traffic_goal": "capture outputs"},
                    "quality_targets": {"require_target_api_traffic": True},
                    "nc_program_spec": {
                        "purpose": "bounded capture", "block_goals": ["sample"],
                        "constraints": ["bounded"],
                    },
                    "steps": [{
                        "step_id": "S001", "phase": "during", "action": "capture",
                        "interface_name": "ReadValue", "protocol_function": target,
                    }],
                })

                apply_llm_planning_decision(plan, client, f"Capture {target}")

                interfaces = {step.interface_name for step in plan.steps}
                self.assertEqual("StartProgram" in interfaces, needs_motion)
                self.assertIn(target, [step.protocol_function for step in plan.steps])
                if needs_motion:
                    from smart_traffic_agent.execution_contracts import lifecycle_interfaces
                    self.assertTrue({"UploadProgram", "SelectProgram", "ReadProgramNumber"} <= lifecycle_interfaces(plan.steps))
                    self.assertEqual(plan.rag_context["motion_required_by_targets"], [target])

    def test_planner_leaves_program_number_for_codegen(self) -> None:
        plan = build_plan("no-planner-number", "coordinate_motion", "ncguide-generated-cpp", {})

        self.assertEqual(plan.nc_program_spec.program_name, "")
        lifecycle_steps = [
            step for step in plan.steps if step.interface_name in {"UploadProgram", "SelectProgram"}
        ]
        self.assertTrue(lifecycle_steps)
        self.assertTrue(all("program_name" not in step.parameters for step in lifecycle_steps))

    def test_parse_planning_markdown_to_existing_payload_shape(self) -> None:
        payload = parse_planning_markdown("""# Scenario Goal
goal: observable motion

# Quality Analysis
- traffic_goal: position changes
- strategy: ["sample during motion"]

# Quality Targets
- min_feed_samples: 5
- expect_position_variation: true

# NC Program
- program_name: O9001
- purpose: coordinate motion
- block_goals: ["bounded G01 motion"]
- constraints: ["stay within travel"]

# Scene Requirements
- item: {"target_api":"cnc_rdposition","nc_program_required":true,"execution_required":true,"required_mode":"AUTO","operator_action_if_mismatch":true}

# Steps
## S001
- phase: during
- action: read position
- interface_name: ReadPosition
- protocol_function: cnc_rdposition
- parameters: {"axis_count": 3}
- parameter_generation: [{"name":"axis_count","kind":"fixed","value":3}]
- repeat: 5
- interval_seconds: 0.2
""")

        self.assertEqual(payload["scenario_goal"], "observable motion")
        self.assertEqual(payload["nc_program_spec"]["program_name"], "O9001")
        self.assertEqual(payload["steps"][0]["parameters"], {"axis_count": 3})
        self.assertEqual(payload["steps"][0]["repeat"], 5)
        self.assertEqual(payload["scene_requirements"][0]["required_mode"], "AUTO")

    def test_parse_planning_markdown_accepts_json_for_provider_compatibility(self) -> None:
        payload = parse_planning_markdown('{"scenario_goal":"goal","steps":[]}')
        self.assertEqual(payload["scenario_goal"], "goal")

    def test_parse_planning_markdown_accepts_nested_headings_and_bold_keys(self) -> None:
        payload = parse_planning_markdown("""## Scenario Goal
**goal**: read operation history

### NC Program
**purpose**: no motion required
**block_goals** = ["no NC program or Cycle Start is required"]
**constraints**: ["preserve current CNC state"]

## Steps
### S001
- phase: during
- interface_name: ReadOperationHistory
- protocol_function: cnc_stopophis
""")
        self.assertEqual(payload["scenario_goal"], "read operation history")
        self.assertEqual(payload["nc_program_spec"]["block_goals"], ["no NC program or Cycle Start is required"])
        self.assertEqual(payload["steps"][0]["step_id"], "S001")

    def test_missing_nc_program_section_uses_deterministic_non_motion_fallback(self) -> None:
        class MissingNcSpecProvider:
            enabled = True

            def invoke_text(self, *args, **kwargs):
                return """# Scenario Goal
goal: read operation history

# Quality Analysis
- traffic_goal: target call

# Quality Targets
- input_diversity_required: false

# Steps
## S001
- phase: during
- action: stop history
- interface_name: StopHistory
- protocol_function: cnc_stopophis
- parameters: {}
- repeat: 1
- interval_seconds: 0.2
"""

        plan = build_plan("fallback-test", "general_status_collection", "ncguide-generated-cpp", {})
        plan.rag_context["explicit_target_functions"] = ["cnc_stopophis"]
        apply_llm_planning_decision(plan, MissingNcSpecProvider(), "read operation history")
        self.assertEqual(plan.nc_program_spec.block_goals, ["no NC program or Cycle Start is required for this API batch"])
        self.assertTrue(any("deterministic fallback" in note for note in plan.llm_notes))

    def test_empty_nc_block_goals_for_returned_non_motion_target_are_filled(self) -> None:
        class EmptyBlockGoalsProvider:
            enabled = True

            def invoke_text(self, *args, **kwargs):
                return json.dumps({
                    "scenario_goal": "read operation history",
                    "quality_analysis": {"traffic_goal": "capture the target response"},
                    "quality_targets": {"input_diversity_required": False},
                    "nc_program_spec": {"purpose": "read only", "block_goals": [], "constraints": []},
                    "steps": [{
                        "step_id": "S001", "phase": "during", "action": "read history",
                        "interface_name": "ReadOperationHistory", "protocol_function": "cnc_stopophis",
                    }],
                })

        plan = build_plan("empty-goals", "general_status_collection", "ncguide-generated-cpp", {})
        apply_llm_planning_decision(plan, EmptyBlockGoalsProvider(), "read operation history")
        self.assertEqual(plan.nc_program_spec.block_goals, ["no NC program or Cycle Start is required for this API batch"])
        self.assertNotIn("StartProgram", {step.interface_name for step in plan.steps})

    def test_tool_management_scenario_is_classified_and_planned_with_tool_data_access(self) -> None:
        description = (
            "Generate a complete FOCAS client for the Tool Management scenario and read tool information."
        )
        self.assertEqual(infer_scenario(description), "tool_management")
        plan = build_plan("tool-test", "tool_management", "ncguide-generated-cpp", {})
        tool_steps = [step for step in plan.steps if step.interface_name == "ToolDataAccess"]
        self.assertEqual(len(tool_steps), 1)
        self.assertEqual(tool_steps[0].protocol_function, "cnc_rdtool")
        code_spec = build_code_spec(plan, [])
        tool_contract = next(item for item in code_spec.api_contracts if item.tool_name == "cnc_rdtool")
        self.assertIn("IODBTLMNG *", tool_contract.prototype)

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

    def test_llm_steps_keep_parameter_constraints_and_discard_concrete_values(self) -> None:
        steps = plan_steps_from_llm_rows(
            [
                {
                    "step_id": "S001",
                    "phase": "during",
                    "action": "search program by number",
                    "interface_name": "SearchProgram",
                    "protocol_function": "cnc_search",
                    "parameters": {"number": 1234, "axes": "XYZ"},
                    "parameter_generation": [
                        {"name": "type", "kind": "enum", "values": [-1, 0, 1], "constraints": ["documented safe enum"], "reason": "finite modes"},
                        {"name": "axis_count", "kind": "range", "min": 1, "max": 8, "samples": [1, 4, 8], "constraints": ["controlled-axis bounds"]},
                        {"name": "host", "kind": "fixed", "value": "127.0.0.1", "constraints": ["use configured endpoint"]},
                    ],
                    "parameter_constraints": [
                        {"relation": "count must agree with allocated buffer", "inputs": ["count", "buffer"], "evidence": "SDK"}
                    ],
                }
            ]
        )

        strategy = steps[0].parameters["parameter_generation"]

        self.assertEqual(strategy[0]["kind"], "enum")
        self.assertNotIn("values", strategy[0])
        self.assertNotIn("min", strategy[1])
        self.assertNotIn("samples", strategy[1])
        self.assertNotIn("value", strategy[2])
        self.assertNotIn("number", steps[0].parameters)
        self.assertEqual(steps[0].parameters["axes"], "XYZ")
        self.assertEqual(steps[0].parameters["parameter_constraints"][0]["relation"], "count must agree with allocated buffer")

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

    def test_included_related_api_is_materialized_only_with_candidate_and_abi_evidence(self) -> None:
        plan = ExecutionPlan(
            plan_id="plan-test", task_id="task-test", scenario_type="status",
            scenario_goal="capture status and related data", target_environment="simulator",
            nc_program_type="none", nc_program_requirements=[],
            nc_program_spec=NcProgramSpec(program_name=""),
            steps=[PlanStep("S001", "during", "read status", "ReadStatus", protocol_function="cnc_statinfo")],
            expected_outputs=[],
            rag_context={
                "api_candidate_pool": [{"protocol_function": "cnc_absolute"}, {"protocol_function": "cnc_machine"}],
                "official_abi_context": "short cnc_absolute(unsigned short h, short axis, short length, ODBAXIS* data);",
                "api_coverage_analysis": [
                    {"protocol_function": "cnc_absolute", "decision": "include", "relationship": "absolute position"},
                    {"protocol_function": "cnc_machine", "decision": "include"},
                    {"protocol_function": "cnc_unknown", "decision": "include"},
                    {"protocol_function": "cnc_statinfo", "decision": "include"},
                ],
            },
        )

        steps = materialize_included_api_candidates(plan)
        functions = [step.protocol_function for step in steps]
        self.assertEqual(functions, ["cnc_statinfo", "cnc_absolute"])
        self.assertIn("SDK/RAG evidence validation", plan.llm_notes[-1])

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
