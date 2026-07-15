from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ToolSafety = Literal["read_only", "control", "write"]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    interface_name: str
    focas_function: str
    semantic_label: str
    safety_level: ToolSafety
    description: str
    parameter_schema: dict[str, object] = field(default_factory=dict)
    supported_by_cpp_codegen: bool = False
    supported_by_ncguide_readonly: bool = False
    supported_by_simulator: bool = True


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "UploadProgram": ToolSpec(
        interface_name="UploadProgram",
        focas_function="cnc_dwnstart3/cnc_download3/cnc_dwnend3",
        semantic_label="program_upload",
        safety_level="control",
        description="Upload an NC program into the CNC controller.",
        parameter_schema={"program_name": "str", "program_text": "str"},
        supported_by_cpp_codegen=True,
        supported_by_simulator=True,
    ),
    "SelectProgram": ToolSpec(
        interface_name="SelectProgram",
        focas_function="cnc_search",
        semantic_label="program_selection",
        safety_level="control",
        description="Select an NC program before execution.",
        parameter_schema={"program_name": "str"},
        supported_by_cpp_codegen=True,
        supported_by_simulator=True,
    ),
    "ReadProgramNumber": ToolSpec(
        interface_name="ReadProgramNumber",
        focas_function="cnc_rdprgnum",
        semantic_label="program_number_read",
        safety_level="read_only",
        description="Read the currently selected/running NC program number and main program number.",
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
        supported_by_simulator=True,
    ),
    "ReadProgramDirectory": ToolSpec(
        interface_name="ReadProgramDirectory",
        focas_function="cnc_rdprogdir3",
        semantic_label="program_existence_check",
        safety_level="read_only",
        description="Check whether the exact target NC program number is already registered before upload.",
        parameter_schema={"program_name": "str", "type": "int|default=0", "num_prog": "int|default=1"},
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
        supported_by_simulator=True,
    ),
    "DeleteProgram": ToolSpec(
        interface_name="DeleteProgram",
        focas_function="cnc_delete",
        semantic_label="target_program_delete",
        safety_level="write",
        description="Delete only the exact generated target NC program when it already exists; never delete all programs.",
        parameter_schema={"program_name": "str", "require_exact_match": "bool|default=true"},
        supported_by_cpp_codegen=True,
        supported_by_simulator=True,
    ),
    "StartProgram": ToolSpec(
        interface_name="StartProgram",
        focas_function="ncguide_ui_cycle_start",
        semantic_label="program_start",
        safety_level="control",
        description="Start the selected NC program by triggering NCGuide operator UI controls.",
        parameter_schema={
            "window_title": "str|optional",
            "click_mode": "client|screen|optional",
            "cycle_start_x": "int|optional",
            "cycle_start_y": "int|optional",
            "mode_x": "int|optional",
            "mode_y": "int|optional",
        },
        supported_by_cpp_codegen=True,
        supported_by_simulator=True,
    ),
    "StopProgram": ToolSpec(
        interface_name="StopProgram",
        focas_function="operator_cycle_stop",
        semantic_label="program_stop",
        safety_level="control",
        description="Stop or end the running NC program. This requires NCGuide/operator control and is not generated as a direct FOCAS API call until a verified stop method is configured.",
        supported_by_simulator=True,
    ),
    "ReadRunStatus": ToolSpec(
        interface_name="ReadRunStatus",
        focas_function="cnc_statinfo",
        semantic_label="status_query",
        safety_level="read_only",
        description="Read CNC automatic operation status, motion state, emergency state, and alarm state.",
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
    "ReadPosition": ToolSpec(
        interface_name="ReadPosition",
        focas_function="cnc_rdposition",
        semantic_label="coordinate_read",
        safety_level="read_only",
        description="Read axis position data during a machining or motion scene.",
        parameter_schema={"axes": "list[str]", "coordinate_system": "machine|absolute|relative|distance"},
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
    "ReadDistanceToGo": ToolSpec(
        interface_name="ReadDistanceToGo",
        focas_function="cnc_distance",
        semantic_label="remaining_distance_read",
        safety_level="read_only",
        description="Read remaining distance to go for controlled axes during motion completion gating.",
        parameter_schema={"axes": "list[str]", "axis_count": "int|optional"},
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
    "ReadFeedSpeed": ToolSpec(
        interface_name="ReadFeedSpeed",
        focas_function="cnc_actf",
        semantic_label="feed_speed_read",
        safety_level="read_only",
        description="Read actual feed speed while the NC program is running.",
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
    "ReadSpindleSpeed": ToolSpec(
        interface_name="ReadSpindleSpeed",
        focas_function="cnc_acts",
        semantic_label="spindle_speed_read",
        safety_level="read_only",
        description="Read actual spindle speed.",
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
    "ReadParameter": ToolSpec(
        interface_name="ReadParameter",
        focas_function="cnc_rdparam",
        semantic_label="parameter_read",
        safety_level="read_only",
        description="Read CNC parameter data.",
        parameter_schema={"parameter_no": "int", "axis": "int|optional"},
        supported_by_simulator=True,
    ),
    "WriteParameter": ToolSpec(
        interface_name="WriteParameter",
        focas_function="cnc_wrparam",
        semantic_label="parameter_write",
        safety_level="write",
        description="Write CNC parameter data. Restricted for safe experiments.",
        parameter_schema={"parameter_no": "int", "value": "int|float|str", "axis": "int|optional"},
        supported_by_simulator=True,
    ),
    "ReadAlarm": ToolSpec(
        interface_name="ReadAlarm",
        focas_function="cnc_alarm2",
        semantic_label="alarm_query",
        safety_level="read_only",
        description="Read CNC alarm bit/status information.",
        supported_by_cpp_codegen=True,
        supported_by_ncguide_readonly=True,
    ),
}


def get_tool(interface_name: str) -> ToolSpec | None:
    return TOOL_REGISTRY.get(interface_name)


def focas_function_for(interface_name: str) -> str:
    tool = get_tool(interface_name)
    return tool.focas_function if tool else ""


def semantic_label_for(interface_name: str) -> str:
    tool = get_tool(interface_name)
    return tool.semantic_label if tool else "api_call"


def supported_cpp_codegen_interfaces() -> set[str]:
    return {
        name
        for name, tool in TOOL_REGISTRY.items()
        if tool.supported_by_cpp_codegen
    }


def supported_ncguide_readonly_interfaces() -> set[str]:
    return {
        name
        for name, tool in TOOL_REGISTRY.items()
        if tool.supported_by_ncguide_readonly
    }


def interface_to_focas_mapping() -> dict[str, str]:
    return {name: tool.focas_function for name, tool in TOOL_REGISTRY.items()}
