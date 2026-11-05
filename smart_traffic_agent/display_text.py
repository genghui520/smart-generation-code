"""Chinese presentation helpers for the desktop UI and execution reports.

The capture contract, JSON keys, FOCAS names, error constants, and Neo4j
relationship types stay unchanged.  This module only adds a readable display
layer so logs can explain an event while retaining its exact technical data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


AGENT_LABELS = {
    "PlanningAgent": "规划智能体",
    "PlanningRole": "规划智能体",
    "CodeGenerationAgent": "代码生成智能体",
    "CodeEngineerRole": "代码生成智能体",
    "ExecutionAgent": "执行智能体",
    "ExecutionRole": "执行智能体",
    "RouterAgent": "路由智能体",
    "RouterRole": "路由智能体",
    "TrafficGenerationWorkflow": "采集工作流",
}

STAGE_LABELS = {
    "planning": "规划阶段",
    "code_generation": "代码生成阶段",
    "execution": "执行阶段",
    "repair_plan": "规划修复阶段",
    "repair_code": "代码修复阶段",
    "repair_execution": "执行修复阶段",
    "complete": "汇总阶段",
    "router": "路由判断阶段",
}

EVENT_LABELS = {
    "workflow_start": "工作流开始",
    "memory_loaded": "历史记忆已加载",
    "memory_reloaded_for_error": "失败后重新加载历史记忆",
    "agent_start": "智能体开始工作",
    "agent_complete": "智能体完成工作",
    "agent_error": "智能体执行失败",
    "tool_start": "工具调用开始",
    "tool_complete": "工具调用完成",
    "tool_error": "工具调用失败",
    "router_decision": "路由决定",
    "repair_complete": "修复完成",
    "workflow_complete": "工作流结束",
    "coverage_batch": "开始处理目标 API",
    "coverage_progress": "API 采集进度",
    "coverage_batch_failed": "目标 API 工作流失败",
    "coverage_summary": "全局采集汇总",
    "scenario": "场景开始",
    "scenario_done": "场景完成",
    "scenario_partial": "场景部分完成",
    "scenario_failed": "场景失败",
    "status": "运行状态",
    "success": "采集成功",
    "partial": "采集部分完成",
    "failure": "采集失败",
}

RELATION_LABELS = {
    "filters": "参数筛选：前置 API 的真实返回值用于筛选后续输入",
    "prepares": "前置准备：前置 API 为后续调用建立执行条件或会话",
    "REQUIRES": "前置依赖：目标 API 需要先完成该调用",
    "PRECEDES": "顺序关系：该调用必须先于目标调用",
    "PROVIDES_INPUT": "参数供给：该 API 提供目标参数的候选值或筛选依据",
    "SESSION_SEQUENCE": "会话顺序：属于同一 Start → Data → End 会话",
    "READS_AFTER_WRITE": "写后复查：写入后需要重新读取验证",
}

STATUS_LABELS = {
    "known": "规则完整且有证据",
    "partially_known": "部分规则已知，缺少运行时信息",
    "unknown": "暂无可靠规则",
    "documented": "文档已确认",
    "sdk_validated": "SDK 已校验",
    "runtime_verified": "运行时已验证",
    "proposed": "待验证候选",
    "rejected": "已拒绝",
    "superseded": "已被新版本关系替代",
    "collected": "已采集",
    "partial": "部分采集",
    "blocked": "已阻断",
    "failed": "失败",
    "local_success": "本地调用成功",
    "target_not_called": "目标未实际调用",
    "completed": "工作流正常结束",
    "success": "调用成功",
    "matched": "网络事务已匹配",
    "not_required": "无需独立网络流量",
    "not_observed": "未观察到目标流量",
    "complete": "已完整保存",
}


def agent_label(value: Any) -> str:
    text = str(value or "").strip()
    return AGENT_LABELS.get(text, text or "未知智能体")


def stage_label(value: Any) -> str:
    text = str(value or "").strip()
    return STAGE_LABELS.get(text, text or "未知阶段")


def status_label(value: Any) -> str:
    text = str(value or "").strip()
    return STATUS_LABELS.get(text, text or "未记录")


def relation_label(value: Any) -> str:
    text = str(value or "").strip()
    explanation = RELATION_LABELS.get(text)
    return f"{text}（{explanation}）" if explanation else text


def _error_explanation(code: str) -> str:
    explanations = {
        "EW_OK": "调用成功",
        "EW_BUFFER": "缓冲区不足或分段长度需要按 SDK 规则重试",
        "EW_LENGTH": "长度参数或实际传输长度不符合 SDK 要求",
        "EW_BUSY": "控制器或会话仍处于忙状态",
        "EW_ATTRIB": "文件/程序属性或会话类型不合法",
        "EW_FUNC": "当前控制器或功能选件不支持该调用",
        "API_PREREQUISITE_BLOCKED": "适用的强制前置条件未满足",
        "PARAMETER_COVERAGE_INCOMPLETE": "离线参数覆盖不足，仅记录诊断，不代表目标调用失败",
        "target_not_called": "目标 API 没有实际调用记录",
    }
    return explanations.get(code, "需要结合原始返回码、调用参数和前置调用判断")


def localize_technical_text(text: Any) -> str:
    """Translate common display markers while preserving original identifiers."""
    value = str(text or "")
    for source, target in AGENT_LABELS.items():
        if f"[{source}]" not in value:
            value = value.replace(source, f"{target}[{source}]")
    for code in sorted(STATUS_LABELS, key=len, reverse=True):
        if code in RELATION_LABELS or f"[{code}]" in value:
            continue
        value = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])",
            f"{status_label(code)}[{code}]",
            value,
        )
    for code in re.findall(r"\b(?:EW_[A-Z]+|API_[A-Z_]+|PARAMETER_[A-Z_]+|PROGRAM_[A-Z_]+)\b", value):
        if f"[{code}]" not in value:
            value = value.replace(code, f"{_error_explanation(code)}[{code}]")
    for code in RELATION_LABELS:
        if f"[{code}]" not in value:
            value = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(code)}(?![A-Za-z0-9_])",
                relation_label(code),
                value,
            )
    return value


def format_event(event: Any, data: dict[str, Any] | None = None) -> str:
    """Create one concise Chinese log line with the raw event payload intact."""
    name = str(event or "event")
    payload = dict(data or {})
    label = EVENT_LABELS.get(name, name)
    if name == "agent_start":
        detail = f"{agent_label(payload.get('agent'))}开始处理"
    elif name == "router_decision":
        detail = f"下一阶段：{stage_label(payload.get('next_stage'))}"
        reason = str(payload.get("route_reason", "")).strip()
        if reason:
            detail += f"；原因：{localize_technical_text(reason)}"
    elif name in {"tool_start", "tool_complete", "tool_error"}:
        detail = f"工具：{payload.get('tool', payload.get('tool_name', '未命名工具'))}"
    elif name in {"success", "partial", "failure", "status"}:
        detail = localize_technical_text(payload.get("message", "")) or "请查看详细数据"
    elif name in {"agent_error", "scenario_failed", "coverage_batch_failed"}:
        detail = localize_technical_text(payload.get("error", payload.get("message", "执行失败")))
    else:
        detail = localize_technical_text(payload.get("message", ""))
        if not detail:
            detail = "已记录事件"
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"[{label}] {detail}；原始事件={name}；技术数据={raw}"


def human_summary_text(summary: dict[str, Any]) -> str:
    """Render a stable, human-readable companion to summary.json."""
    success = bool(summary.get("success"))
    status = status_label(summary.get("collection_status", "not_run"))
    coverage = summary.get("function_coverage") or {}
    lines = [
        "SMPAgent 执行摘要",
        "=" * 24,
        f"任务：{summary.get('task_id', '未记录')}",
        f"最终状态：{'成功' if success else '未完成'}；在线采集状态：{status}",
        f"工作流：{status_label(summary.get('workflow_status', 'not_run'))}；"
        f"目标 API：{status_label(summary.get('api_execution_status', 'not_run'))}；"
        f"网络关联：{status_label(summary.get('capture_status', 'not_run'))}；"
        f"标注保存：{status_label(summary.get('annotation_status', 'not_run'))}",
        f"当前阶段：{stage_label(summary.get('stage'))}",
        f"API 调用记录：{summary.get('api_log_count', 0)} 条；抓包事件：{summary.get('capture_event_count', 0)} 条",
        f"计划步骤：{summary.get('plan_steps', 0)}；工具调用：{len(summary.get('tool_call_history') or [])} 次；修复次数：{summary.get('repair_attempts', 0)}",
        f"参数/覆盖：已捕获 {coverage.get('captured_function_count', 0)}，已判定 {coverage.get('resolved_function_count', 0)}，不可用 {coverage.get('unavailable_function_count', 0)}",
        f"离线审计：{status_label(summary.get('offline_audit_status', 'not_run'))}",
    ]
    errors = [str(item) for item in (summary.get("errors") or []) if str(item).strip()]
    if errors:
        lines.append("问题与原始技术信息：")
        lines.extend(f"- {localize_technical_text(item)}" for item in errors[:20])
    capture_reason = str(summary.get("capture_reason", "")).strip()
    if capture_reason:
        lines.append(f"网络关联说明：{localize_technical_text(capture_reason)}")
    relation_types = summary.get("relation_types") or summary.get("trcm_relation_types") or {}
    if isinstance(relation_types, dict) and relation_types:
        lines.append("TRCM 关系：")
        lines.extend(f"- {relation_label(name)}：{count}" for name, count in sorted(relation_types.items()))
    lines.append("完整字段、API 名称、返回码、参数和路径请以同目录 summary.json、运行日志及原始 PCAP 为准。")
    return "\n".join(lines) + "\n"


def write_human_summary(path: Path, summary: dict[str, Any]) -> Path:
    output = Path(path)
    output.write_text(human_summary_text(summary), encoding="utf-8")
    return output
