from __future__ import annotations

from ..knowledge import KnowledgeBase
from ..models import ExecutionPlan, PlanStep, WorkflowState
from ..utils import tokenize


class PlanningAgent:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def run(self, state: WorkflowState) -> WorkflowState:
        request = state.request
        retrieved = self.knowledge_base.search(request.description, top_k=6)
        scenario = infer_scenario(request.description)
        plan = build_plan(request.task_id, scenario, request.target_environment)
        plan.retrieved_chunk_ids = [item.chunk.chunk_id for item in retrieved]
        state.retrieved_chunks = retrieved
        state.plan = plan
        state.stage = "code_generation"
        return state


def infer_scenario(description: str) -> str:
    terms = set(tokenize(description))
    joined = description.lower()
    if terms.intersection({"坐标", "coordinate", "position", "axis"}) or "x/y/z" in joined:
        return "coordinate_motion"
    if terms.intersection({"主轴", "spindle"}):
        return "spindle_state"
    if terms.intersection({"参数", "parameter", "config"}):
        return "parameter_read_write"
    if terms.intersection({"报警", "alarm", "fault"}):
        return "alarm_query"
    if terms.intersection({"程序", "program", "nc", "gcode"}):
        return "program_lifecycle"
    return "general_status_collection"


def build_plan(task_id: str, scenario: str, target_environment: str) -> ExecutionPlan:
    builders = {
        "coordinate_motion": coordinate_motion_steps,
        "spindle_state": spindle_steps,
        "parameter_read_write": parameter_steps,
        "alarm_query": alarm_steps,
        "program_lifecycle": program_steps,
        "general_status_collection": status_steps,
    }
    steps = builders[scenario]()
    requirements = nc_requirements_for(scenario)
    return ExecutionPlan(
        plan_id=f"plan-{task_id}",
        task_id=task_id,
        scenario_type=scenario,
        scenario_goal=scenario_goal_for(scenario),
        target_environment=target_environment,
        nc_program_type=nc_type_for(scenario),
        nc_program_requirements=requirements,
        steps=steps,
        expected_outputs=[
            "api_call_script",
            "nc_program",
            "capture_events",
            "api_logs",
            "traffic_log_mapping",
        ],
    )


def coordinate_motion_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {"program_name": "O1001"}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {"program_name": "O1001"}),
        PlanStep("S003", "before", "read initial status", "ReadRunStatus", {}),
        PlanStep("S004", "during", "start NC program", "StartProgram", {}),
        PlanStep(
            "S005",
            "during",
            "sample axis positions",
            "ReadPosition",
            {"axes": ["X", "Y", "Z"], "coordinate_system": "machine"},
            repeat=5,
            interval_seconds=0.2,
            expected_state="coordinates change over time",
        ),
        PlanStep("S006", "during", "sample running status", "ReadRunStatus", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S007", "during", "sample feed speed", "ReadFeedSpeed", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S008", "after", "stop or confirm program end", "StopProgram", {}),
        PlanStep("S009", "after", "read final status", "ReadRunStatus", {}),
    ]


def spindle_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload spindle NC program", "UploadProgram", {"program_name": "O2001"}),
        PlanStep("S002", "before", "select spindle NC program", "SelectProgram", {"program_name": "O2001"}),
        PlanStep("S003", "during", "start spindle scene", "StartProgram", {}),
        PlanStep("S004", "during", "read spindle speed", "ReadSpindleSpeed", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S005", "after", "stop spindle scene", "StopProgram", {}),
    ]


def parameter_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read parameter", "ReadParameter", {"parameter_no": 1001}),
        PlanStep("S003", "during", "write safe parameter value", "WriteParameter", {"parameter_no": 1001, "value": 1}),
        PlanStep("S004", "after", "read parameter after write", "ReadParameter", {"parameter_no": 1001}),
    ]


def alarm_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read baseline status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "query active alarms", "ReadAlarm", {}, repeat=3, interval_seconds=0.2),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def program_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "upload NC program", "UploadProgram", {"program_name": "O3001"}),
        PlanStep("S002", "before", "select NC program", "SelectProgram", {"program_name": "O3001"}),
        PlanStep("S003", "during", "start program", "StartProgram", {}),
        PlanStep("S004", "during", "read program status", "ReadRunStatus", {}, repeat=4, interval_seconds=0.2),
        PlanStep("S005", "after", "stop program", "StopProgram", {}),
    ]


def status_steps() -> list[PlanStep]:
    return [
        PlanStep("S001", "before", "read initial status", "ReadRunStatus", {}),
        PlanStep("S002", "during", "read position", "ReadPosition", {"axes": ["X", "Y", "Z"]}),
        PlanStep("S003", "after", "read final status", "ReadRunStatus", {}),
    ]


def scenario_goal_for(scenario: str) -> str:
    goals = {
        "coordinate_motion": "Generate traffic with changing coordinates, run state, and feed speed.",
        "spindle_state": "Generate traffic related to spindle start, stop, and speed changes.",
        "parameter_read_write": "Generate traffic for parameter read/write behavior.",
        "alarm_query": "Generate traffic for alarm and status queries.",
        "program_lifecycle": "Generate traffic for NC program upload, selection, run, and stop.",
        "general_status_collection": "Generate baseline CNC status collection traffic.",
    }
    return goals[scenario]


def nc_type_for(scenario: str) -> str:
    if scenario == "coordinate_motion":
        return "straight_interpolation_motion"
    if scenario == "spindle_state":
        return "spindle_speed_change"
    if scenario == "program_lifecycle":
        return "basic_program_lifecycle"
    return "minimal_safe_program"


def nc_requirements_for(scenario: str) -> list[str]:
    common = ["include a valid program number", "end with M30", "keep execution short"]
    if scenario == "coordinate_motion":
        return common + ["move X/Y/Z through observable points", "use G90 and G01 feed motion"]
    if scenario == "spindle_state":
        return common + ["start spindle with M03", "change spindle speed", "stop spindle with M05"]
    return common

