"""Offline PlannerAgent smoke test; never invokes CodeGenerationAgent or NCGuide."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_traffic_agent.agents.planner import PlanningAgent
from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.llm import LlmClient, LlmConfig
from smart_traffic_agent.models import TaskRequest, WorkflowState


class StubLlm:
    enabled = True
    http_config = None

    def __init__(self) -> None:
        self.calls = 0

    def invoke_text(self, system_prompt: str, user_prompt: str, **kwargs):
        self.calls += 1
        print({
            "prompt_chars": len(system_prompt) + len(user_prompt),
            "diagnostics_context": kwargs.get("diagnostics_context", {}),
        })
        return """# Scenario Goal
goal: observable coordinate motion

# Quality Analysis
- traffic_goal: position/feed/status variation

# Quality Targets
- min_feed_samples: 5
- expect_feed_variation: true
- expect_position_variation: true

# NC Program
- program_name: O9001
- purpose: coordinate motion
- block_goals: ["safe G00/G01 motion"]
- constraints: ["bounded travel"]

# Steps
## S001
- phase: before
- action: read status
- interface_name: ReadRunStatus
- protocol_function: cnc_statinfo
- parameters: {}
- repeat: 1
## S002
- phase: during
- action: read position
- interface_name: ReadPosition
- protocol_function: cnc_rdposition
- parameters: {}
- repeat: 5
- interval_seconds: 0.2
## S003
- phase: during
- action: read feed
- interface_name: ReadFeedSpeed
- protocol_function: cnc_actf
- parameters: {}
- repeat: 5
- interval_seconds: 0.2
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use the configured IIEGPT endpoint")
    args = parser.parse_args()
    llm = (
        LlmClient.from_config(LlmConfig(
            provider="openai_compatible",
            model="gpt-5.5",
            base_url="https://iiegpt.gentelmen.cn/v1",
            api_key_env="MY_LLM_API_KEY",
            transport="http",
            timeout_seconds=240.0,
            max_attempts=1,
        ))
        if args.real else StubLlm()
    )
    vector_db = Path(__file__).resolve().parents[1] / "rag_indexes" / "focas" / "vector_db"
    agent = PlanningAgent(KnowledgeBase(vector_db_dir=vector_db), llm_client=llm)
    state = WorkflowState(request=TaskRequest(
        description="生成坐标运动流量，采集位置、进给速度和运行状态",
        task_id="planner_smoke",
        target_environment="ncguide-generated-cpp",
    ))
    state = agent.run(state)
    calls = getattr(llm, "usage_history", None)
    call_count = len(calls) if calls is not None else llm.calls
    print({"success": state.plan is not None, "scenario": state.plan.scenario_type if state.plan else None,
           "steps": len(state.plan.steps) if state.plan else 0, "llm_requests": call_count})
    return 0 if state.plan is not None and call_count <= 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
