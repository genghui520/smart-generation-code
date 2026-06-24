from __future__ import annotations

import time
from pathlib import Path

from ..integrations.simulator_client import SimulatedCncClient
from ..models import ApiCallLog, CaptureEvent, ExecutionResult, WorkflowState, utc_now
from ..utils import write_jsonl


class ExecutionAgent:
    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None or state.artifacts is None:
            raise ValueError("Cannot execute before plan and generated artifacts exist.")

        client = SimulatedCncClient()
        api_logs: list[ApiCallLog] = []
        capture_events: list[CaptureEvent] = []
        execution_dir = output_dir / "execution"

        for step in state.plan.steps:
            for _ in range(step.repeat):
                timestamp = utc_now()
                capture_events.append(
                    CaptureEvent(
                        timestamp=timestamp,
                        task_id=state.request.task_id,
                        interface_name=step.interface_name,
                        direction="request",
                        endpoint="simulator://cnc",
                        payload_summary=step.parameters,
                    )
                )
                response = client.call(step.interface_name, step.parameters)
                status_code = int(response.get("status_code", 500))
                response_timestamp = utc_now()
                capture_events.append(
                    CaptureEvent(
                        timestamp=response_timestamp,
                        task_id=state.request.task_id,
                        interface_name=step.interface_name,
                        direction="response",
                        endpoint="simulator://agent",
                        payload_summary=response,
                    )
                )
                api_logs.append(
                    ApiCallLog(
                        timestamp=timestamp,
                        task_id=state.request.task_id,
                        step_id=step.step_id,
                        phase=step.phase,
                        interface_name=step.interface_name,
                        input_parameters=step.parameters,
                        status_code=status_code,
                        response=response,
                        semantic_label=semantic_label(step.interface_name),
                        error=response.get("error"),
                    )
                )
                if step.interval_seconds:
                    time.sleep(min(step.interval_seconds, 0.05))

        errors = [log.error for log in api_logs if log.error]
        success = not errors and all(log.status_code == 0 for log in api_logs)
        write_jsonl(execution_dir / "api_logs.jsonl", api_logs)
        write_jsonl(execution_dir / "capture_events.jsonl", capture_events)

        state.result = ExecutionResult(
            task_id=state.request.task_id,
            success=success,
            api_logs=api_logs,
            capture_events=capture_events,
            output_dir=execution_dir,
            errors=[error for error in errors if error],
        )
        state.errors.extend(state.result.errors)
        state.stage = "annotation"
        return state


def semantic_label(interface_name: str) -> str:
    labels = {
        "UploadProgram": "program_upload",
        "SelectProgram": "program_selection",
        "StartProgram": "program_start",
        "StopProgram": "program_stop",
        "ReadRunStatus": "status_query",
        "ReadPosition": "coordinate_read",
        "ReadFeedSpeed": "feed_speed_read",
        "ReadSpindleSpeed": "spindle_speed_read",
        "ReadParameter": "parameter_read",
        "WriteParameter": "parameter_write",
        "ReadAlarm": "alarm_query",
    }
    return labels.get(interface_name, "api_call")

