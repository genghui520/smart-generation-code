"""Build the independent Heidenhain LSV-2 knowledge base.

The vendor headers are the source of truth.  pyLSV2 names are kept only as a
runtime convenience mapping in the generated records; they are never used to
decide which LSV-2 functions exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .build_rag_index import BGEEmbeddingFunction, EMBEDDING_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEADER_DIR = Path(r"C:\Program Files (x86)\HEIDENHAIN\SDK\RemoTools\lsv2lib\includes\lsv2")
DEFAULT_DOC_FILES = [
    Path(r"C:\Program Files (x86)\HEIDENHAIN\SDK\Help\lsv2protocol.chm"),
    Path(r"C:\Program Files (x86)\HEIDENHAIN\SDK\Help\lsv2lib.chm"),
    Path(r"C:\Program Files (x86)\HEIDENHAIN\SDK\RemoTools\Doc\LSV2LIB32.txt"),
    Path(r"C:\Program Files (x86)\HEIDENHAIN\SDK\RemoTools\Doc\LSV2ctrl.txt"),
]
OUTPUT_DIR = PROJECT_ROOT / "rag_indexes" / "heidenhain"

_RUNTIME_MAPPING = {
    "LSV2ReceiveVersions": "versions",
    "LSV2ReceivePara": "parameters",
    "LSV2ReceiveDirInfo": "directory_info",
    "LSV2ReceiveDir": "directory_content",
    "LSV2ReceiveDrives": "drive_info",
    "LSV2ReceiveFileInfo": "file_info",
    "LSV2GetFileList": "get_file_list",
    "LSV2ReceiveAxesData": "axes_location",
    "LSV2ReceiveTime": "get_remote_datetime",
    "LSV2ReceiveScreenDump": "grab_screen_dump",
    "LSV2ReceiveRunInfo": "program_status",
    "LSV2ReceiveDncInfo": "override_state",
    "LSV2ReceiveMachineConstant": "get_machine_parameter",
    "LSV2ReceiveScopeData": "read_scope_signals",
    "LSV2ReceiveDataEntryInfo": "read_data_path",
    "LSV2ReceiveMem": "read_plc_memory",
}


def _header_category(path: Path) -> str:
    return path.stem.removeprefix("lsv2_")


def _clean_comment(block: str) -> str:
    block = re.sub(r"^\s*[/!*]+\s?", "", block, flags=re.M)
    block = re.sub(r"\s*[*]/\s*$", "", block)
    block = re.sub(r"^\s*#.*$", "", block, flags=re.M)
    return re.sub(r"\s+", " ", block).strip()


def _doc_for(text: str, start: int) -> dict[str, str]:
    """Extract the nearest Doxygen function block without trusting examples."""
    end = text.rfind("*/", 0, start)
    begin = text.rfind("/*!", 0, end + 2) if end >= 0 else -1
    if begin < 0 or end < begin or start - end > 1800:
        return {"brief": "", "documentation": ""}
    block = text[begin:end + 2]
    brief = ""
    match = re.search(r"\\brief\s+(.+?)(?=\\(?:param|note|return|retval|see|ingroup|name|$))", block, re.I | re.S)
    if match:
        brief = _clean_comment(match.group(1))
    return {"brief": brief, "documentation": _clean_comment(block)}


def _risk(name: str, header: str, documentation: str) -> str:
    low = name.casefold()
    read_prefix = ("lsv2receive", "lsv2get", "lsv2has", "lsv2is", "lsv2connected", "lsv2calc")
    utility_reads = ("lsv2scantime", "lsv2format", "lsv2getkey", "lsv2splitpath", "lsv2haswildcard")
    if low.startswith(read_prefix) or low.startswith(utility_reads):
        return "read_only"
    if any(token in low for token in ("delete", "transmit", "write", "set", "make", "rename", "copy", "clear", "control", "activate", "lock", "keypress", "install", "register", "extract", "pushdir", "popdir")):
        return "write_or_control"
    if any(token in low for token in ("open", "close", "login", "logout", "logon", "enable", "abort", "break", "manage", "wait", "background", "hook", "server")):
        return "session_or_control"
    if "require lsv2login" in documentation.casefold() or "require" in documentation.casefold() and "password" in documentation.casefold():
        return "permissioned"
    return "read_only"


def _official_compatibility(documentation: str) -> str:
    """Classify only what the vendor text explicitly says.

    The global LSV2 documentation says NCK controls (including TNC640) only
    support a subset.  Therefore an unqualified function is a candidate that
    still needs a real TNC640/version/option check, not a confirmed match.
    """
    text = documentation.casefold().replace(" ", "")
    if "virtualtnc" in text and ("nottargetsystem" in text or "notonatargetsystem" in text):
        return "virtual_tnc_only"
    if "onlyitnc530" in text or "onitnc530" in text or "foritnc530" in text or "itnc530sampling" in text:
        return "itnc530_only"
    if "nckbasedcontrols" in text or "nck-basedcontrols" in text:
        return "nck_based_including_tnc640"
    return "general_library_function_verify_on_tnc640"


def parse_headers(header_dir: Path = DEFAULT_HEADER_DIR) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not header_dir.is_dir():
        raise FileNotFoundError(f"LSV-2 header directory not found: {header_dir}")
    pattern = re.compile(r"\b(LSV2[A-Za-z0-9_]+)\s*\([^;{}]*\);", re.S)
    for path in sorted(header_dir.glob("lsv2_*.h")):
        if path.name.endswith("_def.h") or path.name in {"lsv2_all.h", "lsv2_jh.h"}:
            continue
        text = path.read_text(encoding="latin-1", errors="ignore")
        # Keep offsets stable while blanking comments so examples such as
        # ``::LSV2Login()`` are never mistaken for declarations.
        masked = re.sub(r"/\*.*?\*/|//[^\r\n]*", lambda m: " " * len(m.group(0)), text, flags=re.S)
        category = _header_category(path)
        for match in pattern.finditer(masked):
            name = match.group(1)
            doc = _doc_for(text, match.start())
            # References in comments can match the broad expression.  A real
            # prototype has a return type on the same declaration region.
            declaration_start = text.rfind("\n", 0, match.start()) + 1
            declaration = text[declaration_start:match.end()].strip()
            # A prototype can wrap over several lines; trim any whitespace
            # before the declaration's return type on the first line.
            declaration = re.sub(r"^\s*", "", declaration)
            entry = records.setdefault(name, {
                "function": name,
                "categories": [],
                "headers": [],
                "prototype": declaration,
                "brief": doc["brief"],
                "documentation": doc["documentation"],
            })
            if category not in entry["categories"]:
                entry["categories"].append(category)
            if path.name not in entry["headers"]:
                entry["headers"].append(path.name)
            if len(doc["documentation"]) > len(entry["documentation"]):
                entry["prototype"] = declaration
                entry["brief"] = doc["brief"] or entry["brief"]
                entry["documentation"] = doc["documentation"]
    result = []
    for name in sorted(records, key=str.casefold):
        row = records[name]
        category = row["categories"][0] if row["categories"] else "uncategorized"
        row.update({
            "category": category,
            "controller": "TNC640",
            "runtime_mapping": _RUNTIME_MAPPING.get(name, ""),
            "risk": _risk(name, row["headers"][0] if row["headers"] else "", row["documentation"]),
            "official_compatibility": _official_compatibility(row["documentation"]),
            # The user explicitly enables all non-iTNC530-only functions as
            # current TNC640 collection targets. Runtime success is still
            # determined by the actual controller response.
            "tnc640_collectable": _official_compatibility(row["documentation"]) != "itnc530_only",
            "tnc640_collection_basis": "user_enabled_target" if _official_compatibility(row["documentation"]) == "general_library_function_verify_on_tnc640" else "official_nck_or_exclusion_rule",
            "source_path": str(header_dir),
            "source_type": "official_lsv2_header",
            "protocol": "lsv2",
        })
        result.append(row)
    return result


def api_chunks(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row in manifest:
        name = row["function"]
        common = {"function": name, "category": row["category"], "protocol": "lsv2", "controller": "TNC640"}
        chunks.append({"chunk_id": f"lsv2-{name.casefold()}-prototype", "chunk_type": "prototype", **common,
                       "text": f"{name} ({row['category']}): {row['prototype']}\n官方兼容性：{row['official_compatibility']}；当前 TNC640 采集目标：{'是' if row['tnc640_collectable'] else '否'}；风险等级：{row['risk']}；运行映射：{row['runtime_mapping'] or '无 pyLSV2 便捷映射'}"})
        if row["documentation"]:
            chunks.append({"chunk_id": f"lsv2-{name.casefold()}-documentation", "chunk_type": "documentation", **common,
                           "text": f"{name} 官方头文件说明：{row['documentation']}"})
    return chunks


def rule_chunks() -> list[dict[str, Any]]:
    rules = [
        ("connection", "LSV-2 使用 TCP 19000；先 LSV2Open/LSV2OpenEx，再按权限登录，最后 LSV2ReceivePara 初始化库参数。TNC640 默认地址由界面配置，安全探针必须 safe_mode=True。", ["LSV2Open", "LSV2OpenEx", "LSV2Login", "LSV2ReceivePara"]),
        ("permissions", "权限按官方 LSV-2 访问级别区分：FILETRANSFER、DNC、PLCDEBUG、DATA、INSPECT、SCOPE。权限不足、选件缺失或型号不适用时记录失败原因，不计入成功覆盖。", ["LSV2Login"]),
        ("read_only", "知识库优先使用 LSV2Receive*、LSV2Get*、LSV2Has*、LSV2Is* 等读取函数。只读采集不得发送按键、删除文件、写机床常量或改变控制器状态。", []),
        ("write_control", "LSV2Set*、LSV2Write*、LSV2Transmit*、LSV2Delete*、LSV2Key*、LSV2Control* 和远程控制函数标为高风险；未经明确授权不进入只读采集目标。", []),
        ("runtime_mapping", "pyLSV2 只是运行适配层；例如 versions 映射 LSV2ReceiveVersions、directory_info 映射 LSV2ReceiveDirInfo。规划和检索主键必须使用官方 LSV2 函数名。", list(_RUNTIME_MAPPING)),
        ("tnc640", "TNC640 的适用性由真实设备结果和官方说明共同决定。iTNC530 专用实时读取接口不能因为 Python 包存在就算作 TNC640 可用。", []),
        ("official_compatibility", "官方 lsv2_all.h 明确说明：LSV2 在 iTNC530 上是 fully supported；TNC620/TNC640 等 NCK-based controls 只支持 LSV2 的 subset。官方没有提供 201 个函数逐项 TNC640 全量兼容表，因此未明确标注的函数必须按软件版本、选件、权限和真实调用结果确认。", []),
        ("tnc640_collection_policy", "当前采集策略按用户要求将 201 个官方函数中的 199 个纳入 TNC640 采集目标；排除 2 个明确的 iTNC530 专用函数。纳入目标表示允许进入计划，真实成功、权限错误、选件缺失和控制器不支持必须分别记录。", []),
        ("coverage", "只有真实成功调用并写入结构化 API 日志的函数才计入覆盖；连接成功、目录为空、权限错误和计划名称不能代替目标函数流量。", []),
    ]
    return [{"chunk_id": f"heidenhain-rule-{idx:03d}", "rule_id": f"heidenhain-{kind}", "rule_type": kind,
             "scenario": "heidenhain_lsv2", "protocol": "lsv2", "controller": "TNC640", "applies_to": applies,
             "source_type": "rule", "rule_text": text} for idx, (kind, text, applies) in enumerate(rules, 1)]


def scenario_chunks() -> list[dict[str, Any]]:
    scenarios = [
        ("system_capability", "系统版本与能力", ["LSV2ReceiveVersions", "LSV2ReceivePara", "LSV2ReceiveSystemInfo"]),
        ("program_state", "程序状态与 DNC", ["LSV2ReceiveRunInfo", "LSV2ReceiveDncInfo", "LSV2ReceiveTime"]),
        ("file_system", "文件系统与目录", ["LSV2ReceiveDirInfo", "LSV2GetFileList", "LSV2ReceiveFileInfo", "LSV2ReceiveDir"]),
        ("axes", "轴位置与机床状态", ["LSV2ReceiveAxesData", "LSV2ReceiveStatusFlags"]),
        ("data_plc", "Data Path、PLC 与机床常量读取", ["LSV2ReceiveDataEntryInfo", "LSV2ReceiveMem", "LSV2ReceiveMachineConstant"]),
        ("diagnostics", "诊断、日志和屏幕", ["LSV2ReceiveLogFile", "LSV2ReceiveScreenDump", "LSV2GetErrString"]),
        ("scope", "Remote Scope 实时信号", ["LSV2ReceiveScopeData", "LSV2ReceiveScopeInfo"]),
    ]
    return [{"chunk_id": f"heidenhain-scenario-{key}", "scenario": key, "knowledge_type": "scenario_organization",
             "source_type": "scenario", "protocol": "lsv2", "text": f"{title}（TNC640）：官方 LSV-2 函数 {', '.join(apis)}。先确认权限/选件，再将每个目标函数的真实成功或失败写入日志。"}
            for key, title, apis in scenarios]


def _source_manifest(header_dir: Path) -> dict[str, Any]:
    files = []
    for path in [*sorted(header_dir.glob("lsv2_*.h")), *DEFAULT_DOC_FILES]:
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path), "name": path.name, "sha256": digest, "size": path.stat().st_size})
    return {"protocol": "lsv2", "controller": "TNC640", "source_of_truth": "official Heidenhain LSV-2 Library headers", "files": files}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(header_dir: Path = DEFAULT_HEADER_DIR, output_dir: Path = OUTPUT_DIR, *, reset: bool = True) -> dict[str, int]:
    manifest = parse_headers(header_dir)
    if not manifest:
        raise RuntimeError(f"No LSV-2 functions found in {header_dir}")
    api = api_chunks(manifest)
    rules = rule_chunks()
    scenarios = scenario_chunks()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "library_function_manifest.jsonl", manifest)
    _write_jsonl(output_dir / "function_manifest.jsonl", manifest)
    _write_jsonl(output_dir / "tnc640_collection_manifest.jsonl", [row for row in manifest if row["tnc640_collectable"]])
    _write_jsonl(output_dir / "chunks.jsonl", api)
    _write_jsonl(output_dir / "rule_chunks.jsonl", rules)
    _write_jsonl(output_dir / "scenario_chunks.jsonl", scenarios)
    (output_dir / "source_manifest.json").write_text(json.dumps(_source_manifest(header_dir), ensure_ascii=False, indent=2), encoding="utf-8")

    vector_dir = output_dir / "vector_db"
    if reset and vector_dir.exists():
        shutil.rmtree(vector_dir)
    embed = BGEEmbeddingFunction(EMBEDDING_MODEL)
    docs: list[tuple[str, str, dict[str, Any]]] = []
    for row in api:
        source = next(m for m in manifest if m["function"] == row["function"])
        docs.append((f"api-{row['chunk_id']}", f"Function: {row['function']}\nCategory: {row['category']}\n{row['text']}", {"source_type": "api", "chunk_id": row["chunk_id"], "function": row["function"], "category": row["category"], "chunk_type": row["chunk_type"], "risk": source["risk"], "official_compatibility": source["official_compatibility"], "tnc640_collectable": source["tnc640_collectable"]}))
    for row in rules:
        docs.append((f"rule-{row['rule_id']}", row["rule_text"], {"source_type": "rule", "chunk_id": row["chunk_id"], "rule_id": row["rule_id"], "rule_type": row["rule_type"], "scenario": row["scenario"], "applies_to": json.dumps(row["applies_to"], ensure_ascii=False)}))
    for row in scenarios:
        docs.append((f"scenario-{row['chunk_id']}", row["text"], {"source_type": "scenario", "chunk_id": row["chunk_id"], "scenario": row["scenario"], "rule_type": "scenario_organization"}))
    vector_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_dir))
    collection = client.create_collection("heidenhain_lsv2_knowledge", embedding_function=embed, metadata={"description": "Official Heidenhain LSV-2 Library for TNC640"})
    ids, texts, metas = zip(*docs)
    for start in range(0, len(ids), 100):
        collection.add(ids=list(ids[start:start + 100]), documents=list(texts[start:start + 100]), metadatas=list(metas[start:start + 100]))
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 100):
        vectors.extend(embed(list(texts[start:start + 100])))
    np.save(vector_dir / "stable_embeddings.npy", np.asarray(vectors, dtype=np.float32))
    _write_jsonl(vector_dir / "stable_records.jsonl", [{"id": i, "text": t, "metadata": m} for i, t, m in docs])
    client.close()
    return {"functions": len(manifest), "tnc640_collectable": sum(bool(row["tnc640_collectable"]) for row in manifest), "api_chunks": len(api), "rules": len(rules), "scenarios": len(scenarios), "documents": len(docs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header-dir", type=Path, default=DEFAULT_HEADER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.header_dir, args.output_dir, reset=not args.no_reset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
