from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smart_traffic_agent.agent_soul import AgentSoulStore
from smart_traffic_agent.agents.code_generator import generate_cpp_api_script_with_llm
from smart_traffic_agent.agents.planner import apply_llm_planning_decision, build_plan
from smart_traffic_agent.agents.router import RouterAgent
from smart_traffic_agent.models import (
    GeneratedArtifacts,
    ExecutionResult,
    NcProgramSpec,
    PlanStep,
    TaskRequest,
    WorkflowState,
)


SOUL_ROOT = Path("agent_souls")


class AgentPromptSnapshotTests(unittest.TestCase):
    def test_planner_prompt_contains_soul_and_trg_context(self) -> None:
        plan = build_plan("prompt-planner", "parameter_read", "simulator", {})
        plan.rag_context["agent_soul"] = "planner soul snapshot"
        plan.rag_context["traffic_relationships"] = {
            "apis": {"cnc_rdmacro": {"dependencies": [{"confidence_status": "documented"}]}}
        }
        response = """# Scenario Goal
goal: read a macro variable

# Quality Analysis
- traffic_goal: return a valid macro value
- risk: controller state
- strategy: ["read documented values"]

# Quality Targets
- input_diversity_required: false
- response_state_diversity_required: false
- max_configurations: 1
- response_fields_expected_to_change: []
- response_fields_expected_stable: ["value"]
- response_variation_strategy: "not applicable"

# NC Program
- purpose: read only
- block_goals: ["no NC program or Cycle Start is required"]
- constraints: ["preserve CNC state"]
- generation_notes: []

# Steps
## S001
- phase: during
- action: read macro
- interface_name: ReadMacro
- protocol_function: cnc_rdmacro
- parameters: {}
- parameter_generation: [{"name":"number","kind":"discovered","reason":"documented range","evidence":"SDK"}]
- parameter_constraints: []
- parameter_coverage_goal: {"minimum_distinct":1,"coverage_scope":"documented","evidence":"SDK","restore_strategy":"none","on_insufficient":"report"}
- state_coverage: {}
- repeat: 1
- interval_seconds: 0.1
- expected_state: unchanged
"""
        llm = Mock()
        llm.invoke_text.return_value = response
        apply_llm_planning_decision(plan, llm, "read cnc_rdmacro")
        system_prompt, user_prompt = llm.invoke_text.call_args.args[:2]
        self.assertIn("planner soul snapshot", system_prompt)
        self.assertIn("TRCM context", user_prompt)
        self.assertIn("documented", user_prompt)

    def test_generator_prompt_contains_soul_and_trg_context(self) -> None:
        request = TaskRequest(
            description="generate a macro read client",
            task_id="prompt-generator",
            permissions={"code_only_evaluation": True},
        )
        plan = build_plan(request.task_id, "parameter_read", "simulator", {})
        plan.rag_context["agent_soul"] = "generator soul snapshot"
        plan.rag_context["traffic_relationships"] = {"apis": {"cnc_rdmacro": {"dependencies": []}}}
        plan.steps = [PlanStep("S001", "during", "read macro", "ReadMacro", {}, protocol_function="cnc_rdmacro")]
        state = WorkflowState(request=request, plan=plan)
        llm = Mock()
        llm.invoke_json.return_value = {"api_fragments": [{
            "step_id": "S001",
            "apis": ["cnc_rdmacro"],
            "calls": [{
                "api_name": "cnc_rdmacro",
                "call_expr": "api.cnc_rdmacro(handle, 1, 8, &__out_macro)",
                "parameter_values": {"number": 1, "length": 8},
            }],
        }]}
        with patch("smart_traffic_agent.agents.code_generator.official_focas_abi_context", return_value="ABI"), \
             patch("smart_traffic_agent.agents.code_generator.sdk_type_context_for_abi", return_value="SDK"):
            generate_cpp_api_script_with_llm(state, plan.steps, "", llm)
        system_prompt, user_prompt = llm.invoke_json.call_args.args[:2]
        self.assertIn("generator soul snapshot", system_prompt)
        self.assertIn("Traffic Relationship Graph context", user_prompt)

    def test_router_prompt_contains_own_soul_and_graph_context(self) -> None:
        request = TaskRequest(description="route a failed macro read", task_id="prompt-router")
        plan = build_plan(request.task_id, "parameter_read", "simulator", {})
        plan.rag_context["traffic_relationships"] = {"apis": {"cnc_rdmacro": {"knowledge_status": "unknown"}}}
        plan.steps = [PlanStep("S001", "during", "read macro", "ReadMacro", {}, protocol_function="cnc_rdmacro")]
        state = WorkflowState(request=request, plan=plan, artifacts=GeneratedArtifacts("", ""))
        state.result = ExecutionResult(request.task_id, False, [], [], Path("."), errors=["compile failed"])
        llm = Mock()
        llm.enabled = True
        llm.invoke_json.return_value = {"next_stage": "repair_code", "reason": "compile", "repair_instruction": "repair"}
        router = RouterAgent(llm_client=llm)
        router.soul_store = AgentSoulStore(SOUL_ROOT)
        router.route_with_llm(state)
        system_prompt, user_prompt = llm.invoke_json.call_args.args[:2]
        self.assertIn("Dynamic Agent Soul", system_prompt)
        self.assertIn("traffic_relationships=", user_prompt)
        self.assertIn("unknown", user_prompt)


if __name__ == "__main__":
    unittest.main()
