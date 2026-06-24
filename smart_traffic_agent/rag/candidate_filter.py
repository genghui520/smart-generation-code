from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..utils import write_jsonl


DROP_TITLE_PATTERNS = [
    r"^目录$",
    r"^前言$",
    r"^索引$",
    r"^附录$",
    r"^A\.\d+\s*参数的说明$",
    r"^A\.\d+\s*数据类型$",
    r"^A\.\d+\s*标准参数设定表$",
    r"^4\s*参数的说明$",
    r"^4\.4\s*标准参数设定表$",
    r"说明书改版履历",
    r"安全使用须知",
    r"警告、注意和注释",
    r"一般警告和注意",
    r"警告一览表",
    r"^I\.?\s*概述$",
    r"^概述$",
    r"^B-\d+",
]

DROP_TEXT_PATTERNS = [
    r"本说明书的任何内容不得以任何方式复制",
    r"外汇和外国贸易法",
    r"再出口",
    r"请仔细阅读本说明书，并加以妥善保管",
]

KEEP_TITLE_KEYWORDS = [
    "自动运行",
    "手动运行",
    "MDI",
    "程序",
    "编辑",
    "输入/输出",
    "输入／输出",
    "坐标",
    "参考点",
    "插补",
    "进给",
    "主轴",
    "刀具",
    "补偿",
    "偏置",
    "参数",
    "报警",
    "诊断",
    "PMC",
    "以太网",
    "数据",
    "宏",
    "系统变量",
    "运行",
    "状态",
    "历史",
    "测试运行",
    "伺服",
    "I/O",
    "DI/DO",
    "FSSB",
]

STRONG_PARAMETER_TITLE_KEYWORDS = [
    "轴控制",
    "坐标系",
    "存储行程",
    "进给速度",
    "加/减速",
    "伺服",
    "DI/DO",
    "程序",
    "主轴",
    "刀具偏置",
    "固定循环",
    "刚性攻丝",
    "用户宏程序",
    "跳转功能",
    "外部数据输入",
    "手动运行",
    "自动运行",
    "程序再启动",
    "PMC",
    "以太网",
    "数据服务器",
    "故障诊断",
    "波形诊断",
    "FSSB",
    "嵌入式以太网",
    "手动手轮",
    "错误操作",
    "防止错误操作",
    "I/O",
]

KEEP_TEXT_KEYWORDS = [
    "G功能",
    "G代码",
    "M代码",
    "程序段",
    "自动方式",
    "MDI方式",
    "MEM",
    "EDIT",
    "运行状态",
    "启动",
    "停止",
    "复位",
    "进给速度",
    "主轴速度",
    "坐标系",
    "机床坐标",
    "工件坐标",
    "当前位置",
    "参数",
    "报警",
    "诊断",
    "PMC",
    "输入输出",
    "输入/输出",
    "以太网",
    "信号",
    "刀具补偿",
    "宏变量",
    "系统变量",
]

LOW_VALUE_TITLE_KEYWORDS = [
    "版权",
    "出口",
    "说明书列表",
    "相关说明书",
    "规格编号",
]


def filter_candidate_chunks(
    input_path: Path,
    output_path: Path,
    *,
    min_score: float = 1.0,
) -> list[dict[str, Any]]:
    rows = read_jsonl(input_path)
    candidates: list[dict[str, Any]] = []

    for row in rows:
        decision = score_candidate(row)
        if decision["drop"]:
            continue
        if decision["score"] < min_score:
            continue

        enriched = dict(row)
        enriched["knowledge_type"] = "manual_candidate"
        enriched["candidate_score"] = decision["score"]
        enriched["candidate_reason"] = decision["reasons"]
        candidates.append(enriched)

    write_jsonl(output_path, candidates)
    return candidates


def score_candidate(row: dict[str, Any]) -> dict[str, Any]:
    title = str(row.get("section_title", ""))
    text = str(row.get("text", ""))
    haystack = f"{title}\n{text}"
    reasons: list[str] = []
    score = 0.0

    for pattern in DROP_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return {"drop": True, "score": 0.0, "reasons": [f"drop_title:{pattern}"]}

    for pattern in DROP_TEXT_PATTERNS:
        if re.search(pattern, text):
            return {"drop": True, "score": 0.0, "reasons": [f"drop_text:{pattern}"]}

    if row.get("manual_type") == "parameter_manual" and "参数" in title:
        if not any(keyword.lower() in title.lower() for keyword in STRONG_PARAMETER_TITLE_KEYWORDS):
            return {
                "drop": True,
                "score": 0.0,
                "reasons": ["drop_parameter_table_without_traffic_topic"],
            }

    for keyword in LOW_VALUE_TITLE_KEYWORDS:
        if keyword in title:
            score -= 1.0
            reasons.append(f"low_value_title:{keyword}")

    title_hits = [keyword for keyword in KEEP_TITLE_KEYWORDS if keyword.lower() in title.lower()]
    text_hits = [keyword for keyword in KEEP_TEXT_KEYWORDS if keyword.lower() in haystack.lower()]

    if title_hits:
        score += 2.0 + min(len(title_hits), 4) * 0.5
        reasons.extend(f"title:{keyword}" for keyword in title_hits[:6])

    if text_hits:
        score += min(len(text_hits), 8) * 0.35
        reasons.extend(f"text:{keyword}" for keyword in text_hits[:8])

    if row.get("manual_type") == "operation_manual":
        score += 0.4
        reasons.append("manual_type:operation_manual")
    elif row.get("manual_type") == "parameter_manual":
        score += 0.3
        reasons.append("manual_type:parameter_manual")
    elif row.get("manual_type") == "maintenance_manual":
        score += 0.3
        reasons.append("manual_type:maintenance_manual")

    if len(text) < 120:
        score -= 0.8
        reasons.append("short_text")

    return {"drop": False, "score": round(score, 3), "reasons": reasons}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
