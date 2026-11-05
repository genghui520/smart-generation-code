"""Build the Mitsubishi EZSocket knowledge base used by SMPAgent.

The reference manual documents an OLE/COM API rather than a public text
protocol.  This builder keeps the extracted manual pages and the companion
SimCNC C# wrapper in a traceable JSONL index, then creates the stable vector
files consumed by :class:`smart_traffic_agent.knowledge.KnowledgeBase`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUAL = Path(
    r"E:\BaiduNetdiskDownload\三合一数控机床虚拟机\三菱A2\FCSB1224W000 参考手册 ib1501632f.pdf"
)
DEFAULT_SOURCE = Path(
    r"E:\BaiduNetdiskDownload\三合一数控机床虚拟机\三菱A2\SimCNC-master"
)
DEFAULT_OUT = ROOT / "rag_indexes" / "ezsocket"


CORE_APIS: list[dict[str, Any]] = [
    {"function": "SetTCPIPProtocol", "category": "communication", "read_only": False, "description": "为 M700/M800 设置 TCP/IP 通信目标；端口 683。必须在 Open2/Open3 前调用。"},
    {"function": "Open2", "category": "communication", "read_only": False, "description": "打开 NC 通信线路，输入系统类型、NC 控制单元编号、超时和主机名。"},
    {"function": "Close", "category": "communication", "read_only": False, "description": "关闭已打开的 NC 通信线路。"},
    {"function": "System_GetVersion", "category": "system", "read_only": True, "description": "读取 NC 系统编号、名称、PLC 版本或控制单元版本。推荐作为连接后的只读验证 API。"},
    {"function": "Status_GetRunStatus", "category": "run_status", "read_only": True, "description": "读取刀长测量、自动运行、自动运行启动和自动运行暂停状态。lIndex 支持 0 到 3。"},
    {"function": "Position_GetCurrentPosition", "category": "position", "read_only": True, "description": "读取指定轴的当前位置。轴号从 1 开始。"},
    {"function": "Position_GetMachinePosition", "category": "position", "read_only": True, "description": "读取指定轴的机械坐标，可指定跳跃 ON/OFF。"},
    {"function": "Position_GetWorkPosition", "category": "position", "read_only": True, "description": "读取指定轴的工件坐标，可指定跳跃 ON/OFF。"},
    {"function": "Position_GetDistance2", "category": "position", "read_only": True, "description": "读取指定轴的剩余指令距离，可指定跳跃 ON/OFF。"},
    {"function": "Position_GetFeedSpeed", "category": "position", "read_only": True, "description": "读取进给速度。"},
    {"function": "Command_GetCommand2", "category": "command", "read_only": True, "description": "读取 M/S/T/B 功能指令的模态值。"},
    {"function": "Monitor_GetSpindleMonitor", "category": "spindle", "read_only": True, "description": "读取主轴转速、负载、LED 和报警等主轴监视数据。"},
    {"function": "Monitor_GetServoMonitor", "category": "servo", "read_only": True, "description": "读取伺服轴速度、负载、电流或报警监视数据。"},
    {"function": "System_GetAlarm2", "category": "alarm", "read_only": True, "description": "读取 NC、PLC、操作信息或全部报警信息。"},
    {"function": "Time_GetRunTime", "category": "time", "read_only": True, "description": "读取自动运行累计时间。"},
    {"function": "Program_GetProgramNumber2", "category": "program", "read_only": True, "description": "读取主程序或子程序编号。"},
    {"function": "Program_GetSequenceNumber", "category": "program", "read_only": True, "description": "读取当前主程序或子程序的顺序编号。"},
    {"function": "Program_CurrentBlockRead", "category": "program", "read_only": True, "description": "读取当前执行程序段。"},
    {"function": "Parameter_GetData3", "category": "parameter", "read_only": True, "description": "读取参数数据；组号、项目号、数量和轴号必须保持一致。"},
    {"function": "CommonVariable_Read2", "category": "common_variable", "read_only": True, "description": "读取共变量并返回变量类型；M700/M800 的变量编号必须落在手册规定范围内。"},
    {"function": "Tool_GetToolSetSize", "category": "tool", "read_only": True, "description": "读取刀具补偿组数。"},
    {"function": "Tool_GetOffset", "category": "tool", "read_only": True, "description": "读取指定刀具补偿数据。"},
    {"function": "File_GetDriveInformation", "category": "file", "read_only": True, "description": "读取 NC 文件驱动器信息。"},
    {"function": "File_FindDir2", "category": "file", "read_only": True, "description": "检索 NC 文件目录信息。读取目录后应调用 ResetDir 结束检索。"},
    {"function": "File_GetDriveSize", "category": "file", "read_only": True, "description": "读取驱动器可用容量。"},
    {"function": "File_OpenNCFile2", "category": "file", "read_only": True, "description": "以只读模式打开加工程序专用文件。"},
    {"function": "File_ReadNCFile2", "category": "file", "read_only": True, "description": "读取已打开的加工程序专用文件。"},
    {"function": "File_CloseNCFile2", "category": "file", "read_only": True, "description": "关闭加工程序专用文件。"},
    {"function": "CommonVariable_Write2", "category": "common_variable", "read_only": False, "description": "写入共变量；会直接改变 NC 数据，默认禁止自动采集。"},
    {"function": "Parameter_SetData3", "category": "parameter", "read_only": False, "description": "设定参数；必须确认机械参数设定模式和恢复路径。默认禁止自动采集。"},
    {"function": "Operation_Run", "category": "operation", "read_only": False, "description": "启动 PLC 程序，会改变控制器状态。默认禁止自动采集。"},
]


def run_pdftotext(pdf: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        output = Path(temp.name)
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(output)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return output.read_text(encoding="utf-8", errors="replace")
    finally:
        output.unlink(missing_ok=True)


def page_chunks(text: str, *, source: str, max_chars: int = 2200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_number, raw_page in enumerate(text.split("\f"), 1):
        page = re.sub(r"[ \t]+\n", "\n", raw_page).strip()
        if not page:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page) if part.strip()]
        buffer: list[str] = []
        size = 0
        part_number = 1
        for paragraph in paragraphs:
            if buffer and size + len(paragraph) > max_chars:
                rows.append({
                    "chunk_id": f"manual-{page_number:03d}-{part_number:02d}",
                    "text": "\n\n".join(buffer),
                    "source_type": "manual",
                    "protocol": "ezsocket",
                    "source_file": source,
                    "source_page": page_number,
                })
                part_number += 1
                buffer, size = [], 0
            buffer.append(paragraph)
            size += len(paragraph)
        if buffer:
            rows.append({
                "chunk_id": f"manual-{page_number:03d}-{part_number:02d}",
                "text": "\n\n".join(buffer),
                "source_type": "manual",
                "protocol": "ezsocket",
                "source_file": source,
                "source_page": page_number,
            })
    return rows


def source_api_rows(source_root: Path) -> list[dict[str, Any]]:
    candidates = list(source_root.rglob("MitCom.cs"))
    source_file = str(candidates[0]) if candidates else ""
    source_text = candidates[0].read_text(encoding="utf-8", errors="replace") if candidates else ""
    rows: list[dict[str, Any]] = []
    for item in CORE_APIS:
        function = item["function"]
        matches = list(re.finditer(rf"EZNcCom\.{re.escape(function)}\s*\(", source_text))
        wrapper = ""
        if matches:
            before = source_text[: matches[0].start()]
            method_match = list(re.finditer(r"public\s+[^{;]+?\s+(\w+)\s*\([^)]*\)", before))
            wrapper = method_match[-1].group(1) if method_match else ""
        rows.append({
            "chunk_id": f"api-{function}",
            "function": function,
            "library_method": function,
            "wrapper_method": wrapper,
            "category": item["category"],
            "read_only": item["read_only"],
            "source_type": "api",
            "protocol": "ezsocket",
            "source_file": source_file,
            "text": (
                f"EZSocket API {function}. {item['description']} "
                f"调用对象为 FCSB1224W000 的 EZNCAUTLib 自动化接口；"
                f"返回值 0/S_OK 表示成功，非 0 或 S_FALSE 表示失败。"
            ),
        })
    return rows


def error_rules() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "rule-connection-order",
            "source_type": "rule",
            "rule_type": "connection_prerequisite",
            "protocol": "ezsocket",
            "text": "M700/M800 使用 EZSocket TCP/IP 时，必须先调用 SetTCPIPProtocol(host, 683)，再调用 Open2/Open3；NC 控制单元编号应为 1 到 255，且每个并发连接使用不同编号。",
            "applies_to": ["SetTCPIPProtocol", "Open2"],
            "safe_to_write_in_isolated_simulator": True,
        },
        {
            "chunk_id": "rule-read-only-test",
            "source_type": "rule",
            "rule_type": "api_test",
            "protocol": "ezsocket",
            "text": "首次系统测试优先使用 Status_GetRunStatus(lIndex=1)，读取自动运行状态；lIndex=0 到 3 分别对应刀长测量、自动运行、自动运行启动和自动运行暂停。该调用不改变 CNC 状态。",
            "applies_to": ["Status_GetRunStatus"],
            "safe_to_write_in_isolated_simulator": True,
        },
        {
            "chunk_id": "rule-write-safety",
            "source_type": "rule",
            "rule_type": "write_safety",
            "protocol": "ezsocket",
            "text": "写参数、写共变量、设定刀补、启动 PLC 程序、写入 NC 文件和启动运行操作会直接作用于 CNC；在没有明确恢复步骤和用户授权时不得自动执行。",
            "applies_to": ["Parameter_SetData3", "CommonVariable_Write2", "Operation_Run"],
            "safe_to_write_in_isolated_simulator": False,
        },
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(manual: Path, source_root: Path, output: Path) -> dict[str, int]:
    text = run_pdftotext(manual)
    manual_rows = page_chunks(text, source=str(manual))
    api_rows = source_api_rows(source_root)
    rules = error_rules()
    write_jsonl(output / "manual_chunks.jsonl", manual_rows)
    write_jsonl(output / "api_chunks.jsonl", api_rows)
    write_jsonl(output / "rule_chunks.jsonl", rules)
    write_jsonl(output / "function_manifest.jsonl", [
        {key: value for key, value in row.items() if key not in {"text", "source_file"}}
        | {"description": row["text"], "source_file": row["source_file"]}
        for row in api_rows
    ])
    (output / "source_manifest.json").write_text(json.dumps({
        "protocol": "ezsocket",
        "manual": str(manual),
        "simcnc_source": str(source_root),
        "manual_pages": len(text.split("\f")),
        "api_count": len(api_rows),
        "rule_count": len(rules),
        "test_api": "Status_GetRunStatus",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manual_chunks": len(manual_rows), "api_chunks": len(api_rows), "rules": len(rules)}


def build_vectors(output: Path) -> None:
    from smart_traffic_agent.rag.build_rag_index import BGEEmbeddingFunction
    import numpy as np

    rows: list[dict[str, Any]] = []
    for filename in ("manual_chunks.jsonl", "api_chunks.jsonl", "rule_chunks.jsonl"):
        path = output / filename
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    texts = []
    records = []
    for row in rows:
        metadata = {key: value for key, value in row.items() if key not in {"text"}}
        metadata["source_type"] = row.get("source_type", "manual")
        records.append({"id": row["chunk_id"], "text": row["text"], "metadata": metadata})
        texts.append(row["text"])
    embedding = BGEEmbeddingFunction()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):
        vectors.extend(embedding(texts[start:start + 64]))
    vector_dir = output / "vector_db"
    vector_dir.mkdir(parents=True, exist_ok=True)
    np.save(vector_dir / "stable_embeddings.npy", np.asarray(vectors, dtype=np.float32))
    with (vector_dir / "stable_records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, default=DEFAULT_MANUAL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-vectors", action="store_true")
    args = parser.parse_args()
    counts = build(args.manual, args.source, args.out)
    if not args.no_vectors:
        build_vectors(args.out)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
