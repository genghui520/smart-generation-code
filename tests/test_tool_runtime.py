from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from smart_traffic_agent.agent_tools.base import AgentTool
from smart_traffic_agent.agent_tools.knowledge import KnowledgeSearchInput, KnowledgeSearchTool
from smart_traffic_agent.agent_tools.memory import MemorySearchInput, MemorySearchTool, RememberWorkflowTool
from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.memory import LongTermMemoryStore, MemoryRecord
from smart_traffic_agent.models import KnowledgeChunk, TaskRequest, WorkflowState
from smart_traffic_agent.tool_runtime import (
    ToolCallLoop,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolRegistry,
    ToolRuntime,
)


@dataclass(frozen=True)
class EchoInput:
    value: str


class EchoTool(AgentTool[EchoInput, dict[str, str]]):
    name = "search_knowledge"
    description = "Return the supplied value."

    def invoke(self, tool_input: EchoInput) -> dict[str, str]:
        return {"value": tool_input.value}

    def input_summary(self, tool_input: EchoInput) -> dict[str, str]:
        return {"value": tool_input.value}

    def output_summary(self, tool_output: dict[str, str]) -> dict[str, str]:
        return tool_output


class FailingTool(AgentTool[EchoInput, dict[str, str]]):
    name = "search_knowledge"
    description = "Raise an error."

    def invoke(self, tool_input: EchoInput) -> dict[str, str]:
        raise RuntimeError("expected failure")


class ScriptedLlm:
    enabled = True

    def __init__(self) -> None:
        self.calls = 0

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {"type": "tool_call", "tool_name": "search_knowledge", "arguments": {"value": "from tool"}}
        return {"type": "final", "answer": "done"}


class ToolRuntimeTests(unittest.TestCase):
    def test_registry_register_list_duplicate_and_missing(self) -> None:
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        self.assertEqual(registry.names(), ["search_knowledge"])
        self.assertIs(registry.get("search_knowledge"), tool)
        with self.assertRaises(ValueError):
            registry.register(EchoTool())
        with self.assertRaises(ToolNotFoundError):
            registry.get("missing")

    def test_runtime_success_records_agent_and_result(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        history = []
        runtime = ToolRuntime(registry, history=history)

        output, record = runtime.call("PlanningRole", "search_knowledge", EchoInput("ok"))

        self.assertEqual(output, {"value": "ok"})
        self.assertTrue(record.success)
        self.assertEqual(record.agent_name, "PlanningRole")
        self.assertEqual(history, [record])

    def test_runtime_rejects_unauthorized_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        runtime = ToolRuntime(registry)

        with self.assertRaises(ToolPermissionError):
            runtime.call("RouterRole", "compile_generated_cpp", EchoInput("blocked"))

    def test_runtime_records_tool_exception(self) -> None:
        registry = ToolRegistry()
        registry.register(FailingTool())
        runtime = ToolRuntime(registry)

        with self.assertRaises(ToolExecutionError):
            runtime.call("PlanningRole", "search_knowledge", EchoInput("bad"))

        self.assertEqual(len(runtime.history), 1)
        self.assertFalse(runtime.history[0].success)
        self.assertIn("expected failure", runtime.history[0].error or "")

    def test_tool_call_loop_executes_tool_and_returns_final_payload(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        runtime = ToolRuntime(registry)
        llm = ScriptedLlm()

        result = ToolCallLoop(llm, runtime).run(
            "PlanningRole",
            "system",
            "user",
            input_factory=lambda tool_name, arguments: EchoInput(arguments["value"]),
        )

        self.assertEqual(result, {"type": "final", "answer": "done"})
        self.assertEqual(llm.calls, 2)
        self.assertEqual(len(runtime.history), 1)

    def test_knowledge_tool_searches_configured_knowledge_base(self) -> None:
        knowledge = KnowledgeBase([
            KnowledgeChunk("chunk-1", "cnc_rdposition reads position", {"source_type": "api"}),
        ])
        tool = KnowledgeSearchTool(knowledge)

        output = tool.invoke(KnowledgeSearchInput("cnc_rdposition", top_k=1))

        self.assertEqual([item.chunk.chunk_id for item in output.results], ["chunk-1"])
        self.assertEqual(output.retrieval_method, "term_fallback")

    def test_tool_call_loop_exposes_typed_tool_outputs_to_agent(self) -> None:
        registry = ToolRegistry()
        registry.register(EchoTool())
        runtime = ToolRuntime(registry)
        llm = ScriptedLlm()
        loop = ToolCallLoop(llm, runtime)

        loop.run(
            "PlanningRole",
            "system",
            "user",
            input_factory=lambda tool_name, arguments: EchoInput(arguments["value"]),
        )

        self.assertEqual(len(loop.last_tool_outputs), 1)
        self.assertEqual(loop.last_tool_outputs[0][0], "search_knowledge")
        self.assertEqual(loop.last_tool_outputs[0][1], {"value": "from tool"})

    def test_memory_tools_search_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = LongTermMemoryStore(Path(temp) / "memory.jsonl", vector_db_dir=Path(temp) / "vectors")
            # Keep this unit test independent from Chroma's Windows file handle
            # lifecycle; embedding behavior is covered by the existing memory tests.
            store._ensure_vector_store = lambda: False  # type: ignore[method-assign]
            store.append(
                MemoryRecord(
                    memory_id="m1",
                    timestamp="2026-01-01T00:00:00+00:00",
                    task_description="fix compiler ABI error",
                    task_id="t1",
                    target_environment="ncguide-generated-cpp",
                    scenario_type="coordinate_motion",
                    final_success=True,
                    repair_attempts=1,
                    notes=["fixed ABI"],
                )
            )
            search = MemorySearchTool(store)
            self.assertEqual(search.invoke(MemorySearchInput("compiler ABI error", top_k=1))[0]["memory_id"], "m1")

            state = WorkflowState(
                request=TaskRequest("repair generated code", "t2"),
                repair_attempts=1,
                repair_history=[{"errors": ["compile failure"]}],
            )
            result = RememberWorkflowTool(store).invoke(state)
            self.assertTrue(result["remembered"])
            self.assertEqual(len(store.read_all()), 2)
            store.close()


if __name__ == "__main__":
    unittest.main()
