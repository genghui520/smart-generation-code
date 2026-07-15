from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ..utils import write_json
from .scenario_clusterer import (
    DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH,
    DEFAULT_SCENARIO_CLUSTERS_PATH,
)


DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH = Path("rag_indexes/focas/final_scenario_templates.json")


SCENE_GOALS = {
    "coordinate_feed_motion": "生成坐标轴运动、进给速度变化与运行状态变化相关的 FOCAS 流量。",
    "alarm_diagnosis": "生成报警、诊断、异常状态查询相关的 FOCAS 流量。",
    "parameter_macro_access": "生成参数、宏变量、刀补等数据访问相关的 FOCAS 流量。",
    "ethernet_connection_exception": "生成连接建立、释放、超时和异常请求相关的 FOCAS 流量。",
    "pmc_signal_monitoring": "生成 PMC/DI/DO 信号、机床状态和 I/O 状态查询相关的 FOCAS 流量。",
    "program_lifecycle": "生成 NC 程序上传、选择、启动、运行、停止和状态查询相关的 FOCAS 流量。",
    "tool_offset_management": "生成刀具信息、刀补、刀具寿命和补偿状态相关的 FOCAS 流量。",
    "work_coordinate_setting": "生成工件坐标系、坐标偏置、主轴/参数块读取相关的 FOCAS 流量。",
    "spindle_control": "生成主轴启动、停止、转速变化与主轴状态采集相关的 FOCAS 流量。",
    "general_status_collection": "生成机床基础状态、坐标、速度、报警的基线采集流量。",
}


SCENE_NC_PROGRAMS = {
    "coordinate_feed_motion": ["valid program number", "G90 G54", "G00/G01/G02/G03 motion", "multiple F values", "M30 end"],
    "alarm_diagnosis": ["minimal safe program", "optional alarm-triggering simulator state", "M30 end"],
    "parameter_macro_access": ["not required or minimal safe program", "read before/write/read-back when safe", "M30 end"],
    "ethernet_connection_exception": ["not required", "connection open/close/timeout sequence"],
    "pmc_signal_monitoring": ["not required or minimal safe program", "optional PMC/DI/DO state changes"],
    "program_lifecycle": ["valid program number", "short executable NC program", "upload/select/start/stop", "M30 end"],
    "tool_offset_management": ["G40/G41/G42 or G43/G44/G49 when supported", "safe tool offset data access", "M30 end"],
    "work_coordinate_setting": ["G54-G59 coordinate system use", "safe coordinate offset read/write when supported", "M30 end"],
    "spindle_control": ["M03/M04 spindle start", "multiple S values", "M05 spindle stop", "M30 end"],
    "general_status_collection": ["minimal safe program", "G04 dwell", "M30 end"],
}


SCENE_INTERFACE_PLAN = {
    "coordinate_feed_motion": ["ReadRunStatus", "ReadPosition", "ReadFeedSpeed", "ReadRunStatus"],
    "alarm_diagnosis": ["ReadRunStatus", "ReadAlarm", "ReadParameter", "ReadRunStatus"],
    "parameter_macro_access": ["ReadRunStatus", "ReadParameter", "ReadParameter"],
    "ethernet_connection_exception": ["ReadRunStatus", "ReadAlarm", "ReadRunStatus"],
    "pmc_signal_monitoring": ["ReadRunStatus", "ReadParameter", "ReadAlarm", "ReadRunStatus"],
    "program_lifecycle": ["UploadProgram", "SelectProgram", "StartProgram", "ReadRunStatus", "StopProgram"],
    "tool_offset_management": ["ReadRunStatus", "ReadParameter", "ReadPosition"],
    "work_coordinate_setting": ["ReadRunStatus", "ReadParameter", "ReadPosition"],
    "spindle_control": ["UploadProgram", "SelectProgram", "StartProgram", "ReadSpindleSpeed", "ReadRunStatus", "StopProgram"],
    "general_status_collection": ["ReadRunStatus", "ReadPosition", "ReadFeedSpeed", "ReadSpindleSpeed", "ReadAlarm"],
}


INTERFACE_TO_FOCAS = {
    "UploadProgram": "cnc_download",
    "SelectProgram": "cnc_search",
    "StartProgram": "cnc_start",
    "StopProgram": "cnc_stop",
    "ReadRunStatus": "cnc_statinfo",
    "ReadPosition": "cnc_rdposition",
    "ReadFeedSpeed": "cnc_actf",
    "ReadSpindleSpeed": "cnc_acts",
    "ReadParameter": "cnc_rdparam",
    "WriteParameter": "cnc_wrparam",
    "ReadAlarm": "cnc_alarm2",
}


def build_final_scenario_templates(
    *,
    review_csv_path: Path = DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH,
    clusters_path: Path = DEFAULT_SCENARIO_CLUSTERS_PATH,
    output_path: Path = DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH,
) -> dict[str, Any]:
    review_rows = load_review_rows(review_csv_path)
    cluster_payload = json.loads(clusters_path.read_text(encoding="utf-8")) if clusters_path.exists() else {}
    clusters_by_id = {cluster.get("cluster_id"): cluster for cluster in cluster_payload.get("clusters", [])}
    templates = [row_to_template(row, clusters_by_id.get(row["cluster_id"], {}), index) for index, row in enumerate(review_rows, 1)]
    payload = {
        "protocol": "focas",
        "knowledge_type": "scenario_execution_templates",
        "description": "Execution templates converted from automatically clustered FOCAS scenario candidates.",
        "source_review_csv": str(review_csv_path),
        "source_clusters": str(clusters_path),
        "template_count": len(templates),
        "templates": templates,
    }
    write_json(output_path, payload)
    return payload


def load_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def row_to_template(row: dict[str, str], cluster: dict[str, Any], index: int) -> dict[str, Any]:
    scene_name = row.get("suggested_scene_name") or "general_status_collection"
    cluster_id = row["cluster_id"]
    main_apis = split_semicolon(row.get("dominant_apis", ""))
    main_nc = split_semicolon(row.get("dominant_nc_or_operation", ""))
    main_signals = split_semicolon(row.get("dominant_signals", ""))
    main_objects = split_semicolon(row.get("dominant_objects", ""))
    interfaces = interface_sequence(scene_name, main_apis)
    return {
        "template_id": f"focas-template-{index:02d}",
        "scenario_id": cluster_id,
        "scenario_name": scene_name,
        "cluster_member_count": int(row.get("member_count") or 0),
        "goal": SCENE_GOALS.get(scene_name, f"生成 {scene_name} 相关 FOCAS 流量。"),
        "coverage_priority": coverage_priority(row),
        "main_objects": main_objects,
        "main_apis": main_apis,
        "main_nc_or_operation_features": main_nc,
        "expected_signals": main_signals,
        "required_rule_types": split_semicolon(row.get("dominant_rule_types", "")) or [
            "nc_rule",
            "operation_rule",
            "collection_rule",
            "safety_rule",
        ],
        "nc_program_requirements": SCENE_NC_PROGRAMS.get(scene_name, SCENE_NC_PROGRAMS["general_status_collection"]),
        "operation_template": build_operation_template(scene_name, interfaces),
        "collection_template": build_collection_template(interfaces),
        "safety_template": build_safety_template(scene_name),
        "agent_task_hint": agent_task_hint(scene_name, row),
        "source_cluster_profile": compact_cluster_profile(cluster),
        "review": {
            "suggested_action": row.get("suggested_action", ""),
            "review_note": row.get("review_note", ""),
            "representative_units": row.get("representative_units", ""),
        },
    }


def split_semicolon(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def coverage_priority(row: dict[str, str]) -> str:
    count = int(row.get("member_count") or 0)
    action = row.get("suggested_action", "")
    if "split" in action or count >= 100:
        return "high"
    if count >= 40:
        return "medium"
    return "normal"


def interface_sequence(scene_name: str, main_apis: list[str]) -> list[str]:
    interfaces = list(SCENE_INTERFACE_PLAN.get(scene_name, SCENE_INTERFACE_PLAN["general_status_collection"]))
    for api in main_apis:
        mapped = api_to_interface(api)
        if mapped and mapped not in interfaces:
            interfaces.append(mapped)
    return interfaces


def api_to_interface(api: str) -> str:
    api = api.lower()
    if "rdposition" in api or "rdpos" in api or "rdactpt" in api:
        return "ReadPosition"
    if "actf" in api or "rdspeed" in api:
        return "ReadFeedSpeed"
    if "acts" in api or "rdsp" in api or "spload" in api:
        return "ReadSpindleSpeed"
    if "alarm" in api or "alm" in api:
        return "ReadAlarm"
    if "param" in api or "parm" in api or "macro" in api or "tofs" in api or "tool" in api:
        return "ReadParameter"
    if "search" in api:
        return "SelectProgram"
    if "download" in api or "dwn" in api:
        return "UploadProgram"
    if "statinfo" in api or "status" in api:
        return "ReadRunStatus"
    return ""


def build_operation_template(scene_name: str, interfaces: list[str]) -> list[dict[str, Any]]:
    steps = []
    for index, interface in enumerate(interfaces, 1):
        phase = "before" if index == 1 else "after" if index == len(interfaces) else "during"
        steps.append(
            {
                "step_id": f"T{index:03d}",
                "phase": phase,
                "action": action_for_interface(interface, scene_name),
                "interface_name": interface,
                "protocol_function": INTERFACE_TO_FOCAS.get(interface, ""),
                "parameters": default_parameters(interface, index),
                "repeat": repeat_for_interface(interface),
                "interval_seconds": 0.2 if repeat_for_interface(interface) > 1 else 0.0,
            }
        )
    return steps


def action_for_interface(interface: str, scene_name: str) -> str:
    actions = {
        "UploadProgram": f"upload NC program for {scene_name}",
        "SelectProgram": f"select NC program for {scene_name}",
        "StartProgram": f"start {scene_name} scene",
        "StopProgram": f"stop {scene_name} scene",
        "ReadRunStatus": "collect CNC run status",
        "ReadPosition": "collect axis position",
        "ReadFeedSpeed": "collect feed speed",
        "ReadSpindleSpeed": "collect spindle speed",
        "ReadParameter": "collect parameter-like data",
        "ReadAlarm": "collect alarm state",
    }
    return actions.get(interface, f"call {interface}")


def default_parameters(interface: str, index: int) -> dict[str, Any]:
    if interface in {"UploadProgram", "SelectProgram"}:
        return {"program_name": f"O{7000 + index}"}
    if interface == "ReadPosition":
        return {"axes": ["X", "Y", "Z"], "coordinate_system": "machine"}
    if interface == "ReadParameter":
        return {"parameter_no": 1000 + index}
    return {}


def repeat_for_interface(interface: str) -> int:
    if interface in {"ReadPosition", "ReadFeedSpeed", "ReadSpindleSpeed", "ReadRunStatus", "ReadAlarm"}:
        return 3
    return 1


def build_collection_template(interfaces: list[str]) -> dict[str, Any]:
    return {
        "before": [item for item in interfaces if item in {"ReadRunStatus", "ReadParameter"}][:2],
        "during": [item for item in interfaces if item.startswith("Read")],
        "after": ["ReadRunStatus"],
        "capture_outputs": ["api_logs", "capture_events", "semantic_mapping"],
    }


def build_safety_template(scene_name: str) -> dict[str, Any]:
    restricted = ["avoid destructive writes on real devices", "prefer simulator or read-only bridge for first run"]
    if scene_name in {"parameter_macro_access", "tool_offset_management", "work_coordinate_setting"}:
        restricted.append("write operations must be simulated or explicitly approved")
    if scene_name == "ethernet_connection_exception":
        restricted.append("limit timeout/retry counts to avoid blocking execution")
    return {
        "allowed_environment": ["simulator", "ncguide-bridge-readonly", "ncguide-bridge"],
        "restricted_operations": restricted,
        "abnormal_traffic_allowed": scene_name.endswith("exception") or "alarm" in scene_name,
    }


def agent_task_hint(scene_name: str, row: dict[str, str]) -> str:
    objects = row.get("dominant_objects", "")
    apis = row.get("dominant_apis", "")
    return (
        f"围绕 {scene_name} 生成可控 FOCAS 流量；优先覆盖对象 [{objects}]，"
        f"优先调用 API [{apis}]；按运行前/中/后采集并输出语义标签。"
    )


def compact_cluster_profile(cluster: dict[str, Any]) -> dict[str, Any]:
    profile = cluster.get("profile", {})
    return {
        "trigger_types": profile.get("trigger_types", [])[:3],
        "semantic_objects": profile.get("semantic_objects", [])[:5],
        "api_features": profile.get("api_features", [])[:5],
        "nc_or_operation_features": profile.get("nc_or_operation_features", [])[:5],
        "expected_signals": profile.get("expected_signals", [])[:5],
    }
