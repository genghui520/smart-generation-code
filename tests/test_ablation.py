from __future__ import annotations

import inspect
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.rq1_scenario_tasks import RQ1_SCENARIO_TASKS
from smart_traffic_agent import focas_harness
from smart_traffic_agent.agents import code_generator, single_agent
from smart_traffic_agent.agents.single_agent import StrictSingleAgent
from smart_traffic_agent.agent_tools.cpp_execution import find_msvc_x86_toolchain
from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.llm import LlmClient
from smart_traffic_agent.models import TaskRequest, WorkflowState
from smart_traffic_agent.tool_runtime.policy import authorized_tools


REQUIRED_APIS = {
    "cnc_dwnstart3", "cnc_download3", "cnc_dwnend3", "cnc_search",
    "cnc_statinfo", "cnc_rdposition", "cnc_actf", "cnc_rdprgnum",
    "cnc_rdtool", "cnc_rdtofs", "cnc_rdtofsr", "cnc_rdtofsinfo",
    "cnc_rdtofsinfo2",
}


class ScriptedLlm(LlmClient):
    def __init__(self, payloads):
        super().__init__(http_config=object())
        self.payloads = list(payloads)
        self.calls: list[tuple[str, str]] = []

    def invoke_text(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.payloads.pop(0))


def make_agent(llm: LlmClient | None = None) -> StrictSingleAgent:
    return StrictSingleAgent(KnowledgeBase(), llm or ScriptedLlm([]))


class FairSingleAgentAblationTests(unittest.TestCase):
    def test_01_full_and_single_share_exact_harness_renderer_and_path(self) -> None:
        self.assertIs(code_generator.render_focas_harness_template, focas_harness.render_focas_harness_template)
        self.assertIs(single_agent.render_focas_harness_template, focas_harness.render_focas_harness_template)
        self.assertEqual(single_agent.shared_single_agent_harness_path(), focas_harness.FOCAS_HARNESS_TEMPLATE_PATH)

    def test_02_single_agent_source_has_no_planner_or_codespec_preprocessing(self) -> None:
        source = inspect.getsource(single_agent)
        for forbidden in ("build_plan(", "PlanningAgent(", "CodeGenerationAgent(", "build_code_spec("):
            self.assertNotIn(forbidden, source)
        main_source = inspect.getsource(__import__("main").main)
        self.assertLess(
            main_source.index('if args.ablation == "single-agent"'),
            main_source.index("LongTermMemoryStore"),
        )
        self.assertNotIn("prepare_single_agent_state", main_source)

    def test_03_initial_prompt_contains_no_ground_truth_required_api_names(self) -> None:
        agent = make_agent()
        prompt = agent.initial_system_prompt + "\n" + RQ1_SCENARIO_TASKS["coordinate_motion"]
        leaked = sorted(api for api in REQUIRED_APIS if api in prompt)
        self.assertEqual(leaked, [])

    def test_04_initial_prompt_contains_no_predefined_api_order(self) -> None:
        prompt = make_agent().initial_system_prompt
        self.assertNotIn("Step 1", prompt)
        self.assertNotIn("Before / During / After", prompt)
        self.assertNotIn("upload ->", prompt.lower())

    def test_05_initial_prompt_contains_no_task_specific_parameter_plan(self) -> None:
        prompt = make_agent().initial_system_prompt.lower()
        for forbidden in ("o1234", "axis=", "tool number=", "offset number=", "feed="):
            self.assertNotIn(forbidden, prompt)

    def test_06_single_agent_has_no_memory_tools_and_rejects_loaded_memory(self) -> None:
        self.assertNotIn("search_memory", authorized_tools("SingleAgentRole"))
        self.assertNotIn("remember_workflow", authorized_tools("SingleAgentRole"))
        agent = make_agent()
        state = WorkflowState(TaskRequest("task", "id"))
        state.long_term_memories = [{"old": "result"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "without historical memory"):
                agent.run(state, Path(temp_dir))

    def test_07_runner_uses_the_exact_formal_scenario_task_mapping(self) -> None:
        from scripts import run_rq3_ablation

        self.assertIs(run_rq3_ablation.RQ1_SCENARIO_TASKS, RQ1_SCENARIO_TASKS)
        self.assertIn('RQ1_SCENARIO_TASKS[scenario]', inspect.getsource(run_rq3_ablation.run_one))

    def test_08_rag_and_sdk_queries_are_authored_by_the_single_agent(self) -> None:
        llm = ScriptedLlm([
            {"tool_calls": [
                {"tool_name": "search_knowledge", "arguments": {"query": "agent chosen semantic query", "top_k": 2}},
                {"tool_name": "search_focas_sdk", "arguments": {"query": "agent chosen header query", "top_k": 2}},
            ]},
            {"selected_functions": [], "scenario_fragment_cpp": "short run_generated_scenario(FocasApi& api, unsigned short handle) { return EW_OK; }", "nc_program": ""},
        ])
        agent = make_agent(llm)

        payload = agent._research_then_generate(RQ1_SCENARIO_TASKS["tool_management"])

        self.assertIn("scenario_fragment_cpp", payload)
        self.assertEqual(agent.autonomous_queries, [
            {"tool_name": "search_knowledge", "query": "agent chosen semantic query"},
            {"tool_name": "search_focas_sdk", "query": "agent chosen header query"},
        ])
        self.assertEqual(llm.calls[0][1], RQ1_SCENARIO_TASKS["tool_management"])

    def test_09_generation_path_does_not_import_ground_truth_or_evaluator(self) -> None:
        source = inspect.getsource(single_agent) + inspect.getsource(__import__("main"))
        self.assertNotIn("evaluation/ground_truth", source.replace("\\", "/"))
        self.assertNotIn("evaluate_rq1_ground_truth", source)

    def test_10_template_change_is_seen_by_both_variants_without_a_copy(self) -> None:
        original = focas_harness.FOCAS_HARNESS_TEMPLATE_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            temporary_template = Path(temp_dir) / "focas_harness.cpp"
            temporary_template.write_text(original + "\n// SHARED_TEMPLATE_TEST\n", encoding="utf-8")
            with patch.object(focas_harness, "FOCAS_HARNESS_TEMPLATE_PATH", temporary_template):
                full_render = code_generator.render_focas_harness_template([])
                single_render = single_agent.render_focas_harness_template([])
        self.assertIn("SHARED_TEMPLATE_TEST", full_render)
        self.assertEqual(full_render, single_render)

    def test_raw_compiler_feedback_is_returned_without_external_diagnosis(self) -> None:
        llm = ScriptedLlm([{
            "selected_functions": [],
            "scenario_fragment_cpp": "short run_generated_scenario(FocasApi& api, unsigned short handle) { return EW_OK; }",
            "nc_program": "",
        }])
        agent = make_agent(llm)
        current = {"selected_functions": ["model_choice"], "scenario_fragment_cpp": "model code", "nc_program": ""}

        agent._repair_with_raw_compiler_output(
            original_task="formal task",
            current_payload=current,
            stdout="RAW_STDOUT_123",
            stderr="RAW_STDERR_456",
            repair_attempt=1,
        )

        repair_prompt = llm.calls[0][1]
        self.assertIn("RAW_STDOUT_123", repair_prompt)
        self.assertIn("RAW_STDERR_456", repair_prompt)
        self.assertNotIn("you should replace", repair_prompt.lower())

    def test_strict_single_agent_can_render_and_compile_without_a_plan(self) -> None:
        if find_msvc_x86_toolchain() is None:
            self.skipTest("MSVC x86 toolchain is unavailable")
        llm = ScriptedLlm([
            {"tool_calls": [
                {"tool_name": "search_knowledge", "arguments": {"query": "generic controller research"}},
                {"tool_name": "search_focas_sdk", "arguments": {"query": "generic header research"}},
            ]},
            {
                "selected_functions": [],
                "scenario_fragment_cpp": (
                    "short run_generated_scenario(FocasApi& api, unsigned short handle) "
                    "{ (void)api; (void)handle; return EW_OK; }"
                ),
                "nc_program": "",
            },
        ])
        agent = make_agent(llm)
        state = WorkflowState(TaskRequest("formal task", "single-agent-compile"))

        with tempfile.TemporaryDirectory() as temp_dir:
            result = agent.run(state, Path(temp_dir))

        self.assertTrue(result.framework_success)
        self.assertIsNone(result.state.plan)
        self.assertEqual({row.agent_name for row in result.state.tool_call_history}, {"SingleAgentRole"})

    def test_single_agent_emits_live_progress_without_prompt_or_code_dump(self) -> None:
        events: list[tuple[str, dict]] = []
        llm = ScriptedLlm([
            {"tool_calls": [
                {"tool_name": "search_knowledge", "arguments": {"query": "agent query"}},
                {"tool_name": "search_focas_sdk", "arguments": {"query": "header query"}},
            ]},
            {
                "selected_functions": [],
                "scenario_fragment_cpp": (
                    "short run_generated_scenario(FocasApi& api, unsigned short handle) "
                    "{ (void)api; (void)handle; return EW_OK; }"
                ),
                "nc_program": "",
            },
        ])
        agent = StrictSingleAgent(
            KnowledgeBase(),
            llm,
            progress_callback=lambda event, payload: events.append((event, payload)),
        )
        state = WorkflowState(TaskRequest("formal task", "progress-test"))

        with tempfile.TemporaryDirectory() as temp_dir:
            result = agent.run(state, Path(temp_dir))

        self.assertTrue(result.framework_success)
        names = [event for event, _payload in events]
        for required in (
            "single_agent_phase", "single_agent_llm", "single_agent_query",
            "single_agent_solution", "single_agent_compile", "single_agent_done",
        ):
            self.assertIn(required, names)
        event_text = json.dumps(events, ensure_ascii=False)
        self.assertNotIn("formal task", event_text)
        self.assertNotIn("run_generated_scenario", event_text)

    def test_rq3_runner_streams_and_retains_child_output(self) -> None:
        from scripts.run_rq3_ablation import run_streaming

        console = io.StringIO()
        with redirect_stdout(console):
            return_code, stdout, stderr = run_streaming(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('LIVE_OUT', flush=True); print('LIVE_ERR', file=sys.stderr, flush=True)",
                ],
                cwd=Path.cwd(),
            )

        self.assertEqual(return_code, 0)
        self.assertIn("LIVE_OUT", stdout)
        self.assertIn("LIVE_ERR", stderr)
        self.assertIn("LIVE_OUT", console.getvalue())
        self.assertIn("[stderr] LIVE_ERR", console.getvalue())

    def test_malformed_research_call_is_returned_to_same_agent_for_correction(self) -> None:
        llm = ScriptedLlm([
            {"tool_calls": [
                {"tool_name": "search_knowledge", "arguments": {"query": "valid agent query"}},
                {"tool_name": "search_focas_sdk", "arguments": {"query": "valid SDK query"}},
            ]},
            {"tool_calls": [
                {"tool_name": "search_focas_sdk", "arguments": {}},
            ]},
            {
                "selected_functions": [],
                "scenario_fragment_cpp": (
                    "short run_generated_scenario(FocasApi& api, unsigned short handle) "
                    "{ (void)api; (void)handle; return EW_OK; }"
                ),
                "nc_program": "",
            },
        ])
        agent = make_agent(llm)

        payload = agent._research_then_generate("formal task")

        self.assertIn("scenario_fragment_cpp", payload)
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("Research query must be a non-empty string", llm.calls[2][1])
        self.assertEqual(len(agent.autonomous_queries), 2)


if __name__ == "__main__":
    unittest.main()
