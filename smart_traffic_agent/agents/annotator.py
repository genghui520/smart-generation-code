from __future__ import annotations

from pathlib import Path

from ..models import MappingRecord, WorkflowState
from ..utils import write_json


class AnnotationAgent:
    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.result is None:
            raise ValueError("Cannot annotate before execution result exists.")

        mapping: list[MappingRecord] = []
        for log in state.result.api_logs:
            related = [
                index
                for index, event in enumerate(state.result.capture_events)
                if event.interface_name == log.interface_name and event.task_id == log.task_id
            ]
            mapping.append(
                MappingRecord(
                    task_id=log.task_id,
                    log_timestamp=log.timestamp,
                    interface_name=log.interface_name,
                    semantic_label=log.semantic_label,
                    related_capture_indexes=related,
                    input_parameters=log.input_parameters,
                    status_code=log.status_code,
                )
            )

        state.mapping = mapping
        write_json(output_dir / "execution" / "mapping.json", mapping)
        state.stage = "complete"
        return state

