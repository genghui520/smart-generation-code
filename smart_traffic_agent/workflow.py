from __future__ import annotations

from pathlib import Path

from .agents.annotator import AnnotationAgent
from .agents.code_generator import CodeGenerationAgent
from .agents.executor import ExecutionAgent
from .agents.planner import PlanningAgent
from .agents.router import RouterAgent
from .knowledge import KnowledgeBase
from .models import TaskRequest, WorkflowState
from .utils import ensure_dir, write_json


class TrafficGenerationWorkflow:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.router = RouterAgent()
        self.planner = PlanningAgent(knowledge_base)
        self.generator = CodeGenerationAgent()
        self.executor = ExecutionAgent()
        self.annotator = AnnotationAgent()

    def run(self, request: TaskRequest, output_dir: Path) -> WorkflowState:
        output_dir = ensure_dir(output_dir)
        state = WorkflowState(request=request)

        for _ in range(12):
            stage = self.router.route(state)
            state.stage = stage
            if stage == "planning":
                state = self.planner.run(state)
                write_json(output_dir / "plan.json", state.plan)
            elif stage == "code_generation":
                state = self.generator.run(state, output_dir)
            elif stage == "execution":
                state = self.executor.run(state, output_dir)
            elif stage == "annotation":
                state = self.annotator.run(state, output_dir)
            elif stage in {"repair_plan", "repair_code", "repair_execution"}:
                state.repair_attempts += 1
                if state.repair_attempts > 2:
                    break
                state.errors.append(f"{stage} is not implemented for simulator mode")
                break
            elif stage == "complete":
                break

        write_json(output_dir / "summary.json", workflow_summary(state))
        return state


def workflow_summary(state: WorkflowState) -> dict:
    return {
        "task_id": state.request.task_id,
        "stage": state.stage,
        "success": bool(state.result and state.result.success and state.mapping),
        "scenario_type": state.plan.scenario_type if state.plan else None,
        "plan_steps": len(state.plan.steps) if state.plan else 0,
        "api_log_count": len(state.result.api_logs) if state.result else 0,
        "capture_event_count": len(state.result.capture_events) if state.result else 0,
        "mapping_count": len(state.mapping),
        "errors": state.errors,
    }

