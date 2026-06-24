from __future__ import annotations

from ..models import WorkflowStage, WorkflowState


class RouterAgent:
    def route(self, state: WorkflowState) -> WorkflowStage:
        if state.errors and state.result and not state.result.success:
            last_error = " ".join(state.errors[-3:]).lower()
            if any(word in last_error for word in ["plan", "scenario", "order"]):
                return "repair_plan"
            if any(word in last_error for word in ["script", "code", "syntax", "nc"]):
                return "repair_code"
            return "repair_execution"

        if state.plan is None:
            return "planning"
        if state.artifacts is None:
            return "code_generation"
        if state.result is None:
            return "execution"
        if not state.mapping:
            return "annotation"
        return "complete"

