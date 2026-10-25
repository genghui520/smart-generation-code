"""Desktop service contract and isolated worker; no benchmark entry points."""
from __future__ import annotations

import json
import os
import sys
import socket
import math
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
EVENT = "@@SMP@@"
# Full-manifest collection dispatches one target API per workflow. This keeps
# failures isolated and lets the coordinator move to the next target directly.
API_CAPTURE_BATCH_SIZE = 1
PROMPT = """使用 SMPAgent 为 FANUC NCGuide 采集真实 FOCAS 流量。
采集范围：$scope_title（$scope_id）。
场景目标：$goal
场景对象：$objects
设备状态（采集条件，不是额外场景）：$states
必须覆盖的目标 API：$apis
可重复使用的辅助 API：$support_apis
场景操作与输入参数：
$operations
NC 程序要求：$nc_requirements
预期可观测信号：$expected_signals
安全约束：$safety
$coverage_instruction
连接地址：$host:$port。连接必须读取 FOCAS_HOST 和 FOCAS_PORT 环境变量。
采集网卡：$device。使用真实 Npcap 抓包，保存 PCAP、API 输入输出 CSV 和运行日志。
采集次数参考值：$samples；采样间隔参考值：$interval 秒。根据 API 和场景规则生成有界流程。
在场景开始、运行和结束阶段采集；每次 API 调用必须记录函数名、全部实际输入参数、返回码和输出，最后释放连接。
禁止扩展到未选择的场景。不要运行论文评估、基线比较或消融实验。
用户补充要求：$requirements
"""

FULL_API_PROMPT = """使用 SMPAgent 为 FANUC NCGuide 执行$coverage_scope FOCAS API 与参数流量采集。
以 $manifest 为函数清单；本次目标共 $function_count 个唯一 API：$selected_functions。必须分批覆盖本次选中的每一个 API，禁止把未选 API 算作目标。
对每个 API：依据官方 Fwlib32.h、RAG 原型和控制器能力构造参数；记录函数名、全部实际输入参数、返回码和全部输出。
参数覆盖规则：固定参数使用有依据的固定值；枚举参数覆盖所有安全且有文档依据的枚举值；范围参数至少覆盖边界值、典型值和安全中间值；指针、缓冲区、长度、轴号、程序号、参数号等关联参数必须保持一致。
写入或控制类 API 只有在环境和恢复路径均已验证时才允许使用有界值，并执行“读取原值→写入/控制→读取验证→恢复原值”；无法安全调用时不得为了覆盖率强行探测。头文件无原型或 DLL 无导出可由静态证据判定为不可用；NCGuide 不支持须有真实证据。所有不可用项都不得计入已采集流量。
根据每个 API 的功能、所需设备状态和前置依赖组织执行上下文；辅助 API 可重复调用，但不能代替目标 API 计数。
只有目标 API 确实需要程序执行/坐标变化时才执行 NC 程序和 Cycle Start；程序下载、验证、目录、文件、参数、刀具、PMC 等 API 应直接采集其真实请求/响应，不得被坐标运动门控阻塞。NC 文本作为传输参数不等于要启动程序。
若 Npcap 启动失败，在任何 FOCAS 目标调用前终止该批并记录具体错误；不得生成只有 CSV 没有 PCAP 的伪成功。
连接地址：$host:$port；抓包网卡：$device。使用真实 Npcap，保存每批 PCAP、API 输入输出 CSV、计划和运行日志。
每批采样次数参考值：$samples；采样间隔参考值：$interval 秒。持续执行后续批次，直到函数清单各项均有明确状态；仅当每项都有匹配的真实流量时才能宣称流量全覆盖，不得在只完成一个批次时宣称全覆盖。
对于有状态的程序传输/验证/上传 API，本批只设置一个目标 API；其余函数只能作为该目标文档明确要求的辅助调用。不得把 cnc_download3、cnc_download4、cnc_verify4、cnc_vrfend4、cnc_upstart3、cnc_upload3、cnc_upend3、cnc_upstart4 一次性放进同一个会话；每个目标完成独立会话后再进入下一个批次。不同版本的 start/data/end 绝不交叉配对。
不要运行论文评估、基线比较或消融实验。用户补充要求：$requirements
"""


from .protocol_profiles import protocol_profile, require_capture_adapter
from .api_compatibility import UNSELECTED, SDK_UNKNOWN, CONTROLLER_MODELS, unavailable_reason, detect_sdk_model


def scenarios():
    from .rag.scenario_templates import consolidate_scenario_templates
    data = json.loads((ROOT / "rag_indexes/focas/final_scenario_templates.json").read_text(encoding="utf-8"))
    return consolidate_scenario_templates(data["templates"])


def atomic_scene_api_names(scene):
    """Keep each requested FOCAS function separate for execution evidence."""
    return sorted({
        name.strip()
        for entry in scene.get("main_apis", [])
        for name in re.split(r"[,/;+]+", str(entry))
        if re.fullmatch(r"(?:cnc|pmc)_[A-Za-z0-9_]+", name.strip())
    })


def filter_scene_for_config(scene, config):
    """Remove APIs unavailable on the configured controller from one scene.

    A scene is a batch of related operations.  Unsupported target/support calls
    and their API operation steps are omitted; the remaining scene is still
    collected as one batch.  The original catalog object is never mutated.
    """
    import copy
    filtered = copy.deepcopy(scene)
    def allowed(name):
        return not unavailable_reason(str(name).strip(), config)
    def filter_entries(entries):
        result = []
        for entry in entries or []:
            names = [name.strip() for name in re.split(r"[,/;+]+", str(entry))
                     if re.fullmatch(r"(?:cnc|pmc)_[A-Za-z0-9_]+", name.strip())]
            kept = [name for name in names if allowed(name)]
            if kept:
                result.append(";".join(kept))
        return result
    filtered["main_apis"] = filter_entries(filtered.get("main_apis", []))
    filtered["target_apis"] = filter_entries(filtered.get("target_apis", []))
    filtered["support_apis"] = filter_entries(filtered.get("support_apis", []))
    variants = []
    for variant in filtered.get("operation_variants", []) or []:
        variant["target_apis"] = [name for name in variant.get("target_apis", []) if allowed(name)]
        steps = []
        for step in variant.get("operation_template", []) or []:
            function = str(step.get("protocol_function", "")).strip()
            if re.fullmatch(r"(?:cnc|pmc)_[A-Za-z0-9_]+", function) and not allowed(function):
                continue
            steps.append(step)
        variant["operation_template"] = steps
        if variant.get("target_apis") or steps:
            variants.append(variant)
    if "operation_variants" in filtered:
        filtered["operation_variants"] = variants
    return filtered


def function_manifest():
    path = ROOT / "rag_indexes/focas/function_manifest.jsonl"
    try:
        stat = path.stat()
    except OSError:
        return []
    return _load_function_manifest(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_function_manifest(path_text: str, modified_ns: int, file_size: int):
    path = Path(path_text)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("function"):
                rows.append(row)
    return list({row["function"]: row for row in rows}.values())


READ_ONLY_PRIORITY_PATH = ROOT / "inventcom_output" / "api_capture_priority.json"
AUTO_NO_PARAMETER_PATH = ROOT / "inventcom_output" / "api_mode_parameter_classification.json"


def read_only_priority_functions(config=None):
    """Read-only, no-parameter-change targets from the offline classifier.

    The stored list was classified against the currently configured 0i-D
    header; re-apply compatibility here so a different controller model does
    not silently collect unsupported functions.
    """
    if not READ_ONLY_PRIORITY_PATH.is_file():
        return []
    payload = json.loads(READ_ONLY_PRIORITY_PATH.read_text(encoding="utf-8"))
    names = list(payload.get("ready_auto_no_parameter_eligible_functions", []))
    if config:
        eligibility = dict(config)
        if "header" in eligibility:
            eligibility["controller_model"] = detect_sdk_model(eligibility.pop("header"))
        names = [name for name in names if not unavailable_reason(name, eligibility)]
    return names


def auto_no_parameter_functions(config=None):
    """AUTO/MEM-capable functions with no documented CNC parameter prerequisite.

    The static variant excludes motion-sampling APIs, which need a running NC
    program even though they need no parameter change.
    """
    if not AUTO_NO_PARAMETER_PATH.is_file():
        return []
    payload = json.loads(AUTO_NO_PARAMETER_PATH.read_text(encoding="utf-8"))
    names = list(payload.get("auto_no_parameter_eligible_functions", []))
    if config:
        eligibility = dict(config)
        if "header" in eligibility:
            eligibility["controller_model"] = detect_sdk_model(eligibility.pop("header"))
        names = [name for name in names if not unavailable_reason(name, eligibility)]
    return names


def defaults():
    # Reuse the project's dotenv semantics without importing the LLM stack in the UI.
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('\"').strip("'")
    values.update(os.environ)
    from .integrations.ncguide import default_focas_runtime_dir
    return dict(protocol="focas", controller_model=detect_sdk_model(values.get("FOCAS_HEADER_DIR", r"C:\Lib\FOCAS2 Library\Fwlib\0iD")), host=values.get("FOCAS_HOST", "127.0.0.1"), port=values.get("FOCAS_PORT", "8193"),
                device=values.get("PCAP_NETWORK_DEVICE", r"\Device\NPF_Loopback"),
                dll=values.get("FOCAS_DLL_DIR", str(default_focas_runtime_dir())),
                header=values.get("FOCAS_HEADER_DIR", r"C:\Lib\FOCAS2 Library\Fwlib\0iD"),
                bridge=str(ROOT / ".tools/focas_bridge_cpp/focas_bridge.exe"),
                vector=str(ROOT / "rag_indexes/focas/vector_db_pdf_rebuilt"),
                base_url="https://api.deepseek.com", model="deepseek-flash",
                api_key=values.get("DEEPSEEK_API_KEY", ""), wire_api="chat_completions",
                reasoning_effort="high",
                samples="30", interval="0.2", timeout="120", requirements="",
                output=str(ROOT / "runs/desktop"), resume_from="", template=PROMPT,
                ui_start=False,
                start_x=values.get("NCGUIDE_CYCLE_START_X", ""),
                start_y=values.get("NCGUIDE_CYCLE_START_Y", ""),
                window_title=values.get("NCGUIDE_WINDOW_TITLE", "Main Panel"))



def chinese_api_description(function: str, description: str = "") -> str:
    """Return a concise Chinese explanation for the single-API selector."""
    known = {
        "cnc_copyprog": "复制指定的 CNC 程序。采集前需要发现真实源程序，成功后删除临时目标程序。",
        "cnc_delete": "删除指定的 CNC 程序。仅允许删除已明确识别的临时程序。",
        "cnc_rdprogdir3": "读取 CNC 程序目录，用于发现当前实际存在的程序号。",
        "cnc_rdprgnum": "读取当前选中的主程序号，用于选择结果和恢复状态校验。",
        "cnc_search": "搜索并选择指定程序号，通常需要读取并恢复原程序选择状态。",
        "cnc_statinfo": "读取 CNC 当前运行、运动、报警和急停状态。",
        "cnc_getdtailerr": "读取最近一次 FOCAS 调用的详细错误号。",
        "cnc_download3": "向 CNC 下载程序内容，需要配合下载开始和结束 API。",
        "cnc_upload3": "从 CNC 上传程序内容，需要配合上传开始和结束 API。",
        "cnc_wrparam": "写入 CNC 参数；必须先读取原值、验证写入并恢复原值。",
        "cnc_wrmdipntr": "写入 MDI 指针；必须先读取原指针并验证恢复。",
    }
    if function in known:
        return known[function]
    text = str(description or "").strip()
    if not text:
        return "暂无中文说明，将依据官方原型和 RAG 资料规划参数。"
    replacements = (
        ("Read the ", "读取"), ("Read ", "读取"), ("Write the ", "写入"),
        ("Write ", "写入"), ("information of", "信息"), ("number of", "数量"),
        ("current ", "当前"), ("program", "程序"), ("position", "位置"),
        ("status", "状态"), ("data", "数据"), ("value", "值"),
        ("the ", ""), (" of ", "的"),
    )
    translated = text
    for source, target in replacements:
        translated = translated.replace(source, target).replace(source.lower(), target)
    return translated.rstrip(".") + "。"

def _lines(values):
    return "；".join(str(value) for value in values if str(value).strip()) or "无"


def _operation_lines(scene):
    rows = []
    variants = scene.get("operation_variants") or [{"operation_template": scene.get("operation_template", [])}]
    for variant in variants:
        if len(variants) > 1:
            rows.append(
                f"- 子操作 {variant.get('source_cluster_id', '')}: "
                f"目标 API={','.join(variant.get('target_apis', []))}"
            )
        for step in variant.get("operation_template", []):
            rows.append(
                f"  {step.get('phase', '')}/{step.get('step_id', '')}: {step.get('protocol_function', '')}; "
                f"parameters={json.dumps(step.get('parameters', {}), ensure_ascii=False)}; "
                f"repeat={step.get('repeat', 1)}; interval_seconds={step.get('interval_seconds', 0)}"
            )
    return "\n".join(rows) or "- 依据官方原型与场景知识生成有界步骤"


def render_prompt(config, scene):
    scene = filter_scene_for_config(scene, config)
    return Template(config["template"]).substitute(
        scope_title=scene["scenario_name"], scope_id=scene["scenario_id"],
        goal=scene.get("goal", ""), objects=_lines(scene.get("main_objects", [])),
        states=_lines(scene.get("device_states", [])),
        apis=_lines(scene.get("target_apis", scene.get("main_apis", []))),
        support_apis=_lines(scene.get("support_apis", [])), operations=_operation_lines(scene),
        nc_requirements=_lines(scene.get("nc_program_requirements", [])),
        expected_signals=_lines(scene.get("expected_signals", [])),
        safety=_lines(scene.get("safety_template", {}).get("restricted_operations", [])),
        coverage_instruction=(
            "仅覆盖本场景列出的 API 及其文档化输入参数。对枚举/范围参数生成安全的代表值；"
            "不得用同类 API 替代清单内 API，也不得把未调用的 API 记为已覆盖。"
        ),
        **{k: config[k] for k in ("host", "port", "device", "samples", "interval", "requirements")})


def is_full_api_mode(config):
    if config.get("capture_mode"):
        return config["capture_mode"] == "all_apis"
    if "selected_apis" in config:
        known = {row["function"] for row in function_manifest()}
        return set(config.get("selected_apis", [])) == known and bool(known)
    known = {scene["scenario_id"] for scene in scenarios()}
    return set(config.get("selected", [])) == known and bool(known)


def selected_api_names(config):
    """Return selected targets in manifest order, with no duplicate counting."""
    eligibility = dict(config)
    if "header" in eligibility:
        eligibility["controller_model"] = detect_sdk_model(eligibility.pop("header"))
    if config.get("capture_mode") == "all_apis" or "selected_apis" not in config:
        return [row["function"] for row in function_manifest() if not unavailable_reason(row["function"], eligibility)]
    selected = set(config.get("selected_apis", []))
    return [row["function"] for row in function_manifest()
            if row["function"] in selected and not unavailable_reason(row["function"], eligibility)]


def render_full_api_prompt(config, *, batch_targets=None):
    selected = list(batch_targets) if batch_targets is not None else selected_api_names(config)
    single_api = config.get("capture_mode") == "single_api"
    all_selected = not single_api and len(selected) == len(function_manifest())
    prompt = Template(config.get("full_template") or FULL_API_PROMPT).substitute(
        manifest="rag_indexes/focas/function_manifest.jsonl",
        function_count=len(selected),
        coverage_scope="全部" if all_selected else ("本批选定的" if batch_targets is not None and not single_api else "选定的"),
        selected_functions="清单中的全部函数" if all_selected else ", ".join(selected),
        scene_names=_lines(dict.fromkeys(scene["scenario_name"] for scene in scenarios())),
        **{k: config[k] for k in ("host", "port", "device", "samples", "interval", "requirements")},
    )
    if config.get("capture_mode") == "single_api":
        prompt = prompt.replace("必须分批覆盖本次选中的每一个 API", "只采集本次选中的 API")
        prompt = prompt.replace("在任何 FOCAS 目标调用前终止该批", "在任何 FOCAS 目标调用前终止本次采集")
        prompt = prompt.replace("保存每批 PCAP", "保存 PCAP")
        prompt = prompt.replace("每批采样次数参考值", "采样次数参考值")
        prompt = prompt.replace("持续执行后续批次，直到函数清单各项均有明确状态；", "本次仅执行一次采集流程，失败后结束；")
        prompt = prompt.replace("不得在只完成一个批次时宣称全覆盖", "不得在缺少真实流量时宣称采集完成")
    return prompt


def render_selected_prompts(config):
    if config.get("capture_mode") == "single_scene":
        catalog = {scene["scenario_id"]: scene for scene in scenarios()}
        return [render_prompt(config, filter_scene_for_config(catalog[sid], config)) for sid in config.get("selected", []) if sid in catalog]
    if config.get("capture_mode") == "all_apis":
        return [render_full_api_prompt(config)]
    if "selected_apis" in config:
        return [render_full_api_prompt(config)] if config.get("selected_apis") else []
    if is_full_api_mode(config):
        return [render_full_api_prompt(config)]
    catalog = {scene["scenario_id"]: scene for scene in scenarios()}
    return [render_prompt(config, catalog[sid]) for sid in config.get("selected", []) if sid in catalog]


def selection_requires_cycle_start(config):
    """Use the same motion family as API planning, plus explicit program starts."""
    if config.get("protocol", "focas") != "focas":
        return False
    from .rag.focas_function_manifest import MOTION_FUNCTION_RE
    mode = config.get("capture_mode")
    if mode == "single_scene" or (not mode and "selected_apis" not in config and not is_full_api_mode(config)):
        selected = set(config.get("selected", []))
        scenes = [filter_scene_for_config(scene, config) for scene in scenarios() if scene["scenario_id"] in selected]
        names = [name for scene in scenes for name in atomic_scene_api_names(scene)]
        for scene in scenes:
            variants = [scene, *scene.get("operation_variants", [])]
            for variant in variants:
                for step in variant.get("operation_template", []):
                    if (step.get("interface_name") == "StartProgram" or
                            step.get("protocol_function") in {"cnc_start", "ncguide_ui_cycle_start"}):
                        return True
    else:
        names = selected_api_names(config)
    return any(MOTION_FUNCTION_RE.fullmatch(name) or name == "cnc_start" for name in names)


def cycle_start_configuration_error(config):
    required = selection_requires_cycle_start(config)
    if not required and not config.get("ui_start"):
        return ""
    try:
        for key in ("start_x", "start_y"):
            int(str(config.get(key, "")).strip())
        valid = bool(str(config.get("window_title", "")).strip())
    except (ValueError, TypeError):
        valid = False
    if not valid or (required and not config.get("ui_start")):
        return ("所选采集内容需要运行 NC 程序。" if required else "") + (
            "请先在“采集模式与提示词”页点击“开始定位 Cycle Start”，"
            "在 NCGuide 上选择 Cycle Start 按钮位置；完成定位并启用 Cycle Start 后才能开始采集。")
    return ""


def validate(config, capture=False):
    errors = []
    for key, label in (("host", "仿真器地址"), ("base_url", "模型地址"), ("model", "模型名称"), ("api_key", "API Key")):
        if not str(config.get(key, "")).strip():
            errors.append(f"请填写{label}")
    from urllib.parse import urlparse
    url = urlparse(config.get("base_url", ""))
    if url.scheme not in ("http", "https") or not url.netloc:
        errors.append("模型地址必须是完整的 HTTP(S) URL")
    for key, low, high, cast in (("port", 1, 65535, int), ("samples", 1, 100000, int),
                                 ("interval", 0.01, 3600, float), ("timeout", 1, 600, float)):
        try:
            if not low <= cast(config[key]) <= high:
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            errors.append(f"{key} 必须在 {low}–{high} 之间")
    if capture:
        if not protocol_profile(config)["available"]:
            errors.append(f"{protocol_profile(config)['label']} 采集适配器尚未接入")
            return errors
        mode = config.get("capture_mode")
        if "header" not in config and config.get("controller_model", UNSELECTED) not in (*CONTROLLER_MODELS, SDK_UNKNOWN):
            errors.append("请选择有效的仿真器控制器型号")
        requested = config.get("selected_apis", []) if mode != "all_apis" else []
        if mode == "single_scene" or (not mode and "selected_apis" not in config):
            requested = [name for scene in scenarios() if scene["scenario_id"] in config.get("selected", [])
                         for name in atomic_scene_api_names(filter_scene_for_config(scene, config))]
            if not requested:
                errors.append("当前型号下该批次没有可采集 API")
        if mode and mode not in ("single_api", "single_scene", "all_apis",
                                 "read_only_priority", "auto_no_parameter"):
            errors.append("未知采集模式")
        if mode == "single_api" and len(config.get("selected_apis", [])) != 1:
            errors.append("单个 API 模式请选择且仅选择一个 API")
        if mode == "single_scene" and len(config.get("selected", [])) != 1:
            errors.append("单个场景模式请选择且仅选择一个场景")
        if mode != "single_scene" and ("selected_apis" in config or mode == "all_apis"):

            ids = selected_api_names(config) if mode == "all_apis" else config.get("selected_apis", [])
            known = {row["function"] for row in function_manifest()}
            if not isinstance(ids, list) or not ids:
                errors.append("请至少选择一个 API")
            elif set(ids) - known:
                errors.append(f"存在未知 API：{sorted(set(ids) - known)}")
            else:
                for function in ids:
                    reason = unavailable_reason(str(function), config)
                    if reason:
                        errors.append(f"API 兼容性检查失败：{reason}")
        else:
            known = {s["scenario_id"] for s in scenarios()}
            ids = config.get("selected", [])
            if not ids:
                errors.append("请至少选择一个场景")
            if set(ids) - known:
                errors.append(f"存在未知场景：{sorted(set(ids) - known)}")
        try:
            render_selected_prompts(config)
        except (KeyError, ValueError) as exc:
            errors.append(f"API 或提示词模板无效：{exc}")
        for key, label in (("dll", "FOCAS DLL 目录"), ("header", "FOCAS SDK 目录"), ("vector", "RAG 向量库")):
            if not Path(config.get(key, "")).is_dir():
                errors.append(f"{label}不存在")
        if not config.get("output", "").strip():
            errors.append("请填写输出目录")
        if not config.get("device", "").strip():
            errors.append("请填写抓包网卡")
        cycle_error = cycle_start_configuration_error(config)
        if cycle_error:
            errors.append(cycle_error)
        resume = str(config.get("resume_from", "")).strip()
        if resume:
            output_root = Path(config.get("output", "")).expanduser().resolve()
            resume_path = Path(resume).expanduser().resolve()
            if not resume_path.is_dir() or not resume_path.is_relative_to(output_root):
                errors.append("续采目录必须是采集输出目录下已存在的一次运行目录")
            elif not (resume_path / "settings.json").is_file():
                errors.append("续采目录缺少 settings.json，无法核对原采集配置")
    return errors


def configure(config):
    for key, env in (("host", "FOCAS_HOST"), ("host", "FOCAS_BRIDGE_HOST"), ("port", "FOCAS_PORT"),
                     ("port", "FOCAS_BRIDGE_PORT"), ("dll", "FOCAS_DLL_DIR"), ("header", "FOCAS_HEADER_DIR"),
                     ("device", "PCAP_NETWORK_DEVICE"), ("api_key", "DEEPSEEK_API_KEY"),
                     ("start_x", "NCGUIDE_CYCLE_START_X"), ("start_y", "NCGUIDE_CYCLE_START_Y"),
                     ("window_title", "NCGUIDE_WINDOW_TITLE")):
        os.environ[env] = str(config[key])
    os.environ["NCGUIDE_ENABLE_UI_START"] = "1" if config.get("ui_start") else "0"
    os.environ["NCGUIDE_CLICK_MODE"] = "screen"
    # The generated FOCAS executable is Win32/x86. Python may be x64, so its
    # successful System32 Npcap check does not prove the child can load wpcap.
    npcap_x86 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "SysWOW64" / "Npcap"
    if npcap_x86.is_dir():
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if str(npcap_x86).lower() not in {entry.lower() for entry in path_entries}:
            os.environ["PATH"] = str(npcap_x86) + os.pathsep + os.environ.get("PATH", "")


def make_llm(config):
    from .llm import LlmClient, LlmConfig
    return LlmClient.from_config(LlmConfig(provider="openai_compatible", model=config["model"],
        base_url=config["base_url"], api_key_env="DEEPSEEK_API_KEY", transport="http",
        wire_api=config["wire_api"], timeout_seconds=float(config["timeout"]),
        reasoning_effort=config.get("reasoning_effort"),
        max_output_tokens=(int(config["max_output_tokens"]) if config.get("max_output_tokens") else None)))


def test_simulator(config):
    require_capture_adapter(config)
    from .integrations.ncguide import FocasCppBridgeClient
    with socket.create_connection((config["host"], int(config["port"])), timeout=5):
        pass
    result = FocasCppBridgeClient(Path(config["bridge"]), Path(config["dll"]),
        host=config["host"], port=int(config["port"]), timeout_seconds=5).call("ReadRunStatus")
    if result.get("status_code") != 0:
        raise RuntimeError(f"TCP 可达，但 FOCAS 状态读取失败：{result}")
    return "FOCAS 连接成功，已读取 CNC 运行状态"


def test_llm(config):
    reply = make_llm(config).invoke_text("This is a connection test. Reply OK.", "OK", agent_name="ConnectionTest")
    if not reply.strip():
        raise RuntimeError("模型返回空响应")
    return "模型调用成功，已收到有效响应"


def test_capture_device(config):
    import ctypes as ct
    # Python and generated x86 client require their respective Npcap runtimes.
    runtime = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/Npcap/wpcap.dll"
    runtime_x86 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "SysWOW64/Npcap/wpcap.dll"
    if not runtime_x86.is_file():
        raise RuntimeError(f"x86 FOCAS 采集程序缺少 Npcap 运行库：{runtime_x86}")
    dll = ct.CDLL(str(runtime))
    dll.pcap_open_live.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.c_char_p]
    dll.pcap_open_live.restype = ct.c_void_p
    dll.pcap_close.argtypes = [ct.c_void_p]
    error = ct.create_string_buffer(256)
    handle = dll.pcap_open_live(config["device"].encode(), 65536, 0, 100, error)
    if not handle:
        raise RuntimeError("无法打开抓包网卡：" + error.value.decode(errors="replace"))
    dll.pcap_close(handle)


def preflight(config):
    errors = validate(config, capture=True)
    if errors:
        raise ValueError("\n".join(errors))
    from .agent_tools.cpp_execution import find_msvc_x86_toolchain
    if find_msvc_x86_toolchain() is None:
        raise RuntimeError("未找到 MSVC x86 编译器或 Windows SDK")
    for path in (Path(config["header"]) / "Fwlib32.h", Path(config["dll"]) / "Fwlib32.dll"):
        if not path.is_file():
            raise RuntimeError(f"缺少必需文件：{path}")
    lib_candidates = [Path(config["header"]) / "Fwlib32.lib", Path(config["header"]).parent / "Fwlib32.lib"]
    if not any(p.is_file() for p in lib_candidates):
        raise RuntimeError("SDK 目录或其父目录中缺少 Fwlib32.lib")
    test_simulator(config)
    test_llm(config)
    test_capture_device(config)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryFile(dir=out):
        pass
    return "采集前检查通过：FOCAS、模型、编译器、SDK、抓包网卡、输出目录"


def capture_output_dir(config):
    """Resume only an existing run with the same capture contract."""
    from datetime import datetime
    from uuid import uuid4

    resume = str(config.get("resume_from", "")).strip()
    if not resume:
        out = Path(config["output"]) / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
        out.mkdir(parents=True)
        return out
    out = Path(resume).expanduser().resolve()
    output_root = Path(config["output"]).expanduser().resolve()
    if not out.is_dir() or not out.is_relative_to(output_root):
        raise ValueError("续采目录不在采集输出目录内")
    old = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    for key in ("protocol", "controller_model", "capture_mode", "selected", "selected_apis", "host", "port", "device", "dll", "header", "ui_start", "start_x", "start_y", "window_title"):
        fallback = "focas" if key == "protocol" else UNSELECTED if key == "controller_model" else None
        if old.get(key, fallback) != config.get(key, fallback):
            raise ValueError(f"续采配置 {key} 与原运行不一致；请新建采集，不要混合两次流量")
    return out


def next_scene_folder(out, index, scene):
    base = f"{index:02d}_{scene['template_id']}"
    folder = out / base
    if not folder.exists():
        return folder
    retry = 1
    while (out / f"{base}_retry_{retry:03d}").exists():
        retry += 1
    return out / f"{base}_retry_{retry:03d}"


def scene_has_verified_success(out, sid):
    path = out / "capture_progress.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8")).get("scenarios", {}).get(sid, {})
        folder = Path(record.get("folder", "")).resolve()
        return record.get("status") == "success" and folder.is_relative_to(out.resolve()) and folder.is_dir() and any(
            p.stat().st_size > 24 for p in folder.rglob("*.pcap")
        )
    except (OSError, ValueError, TypeError):
        return False


def write_capture_progress(path, progress):
    from .utils import write_json
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, progress)
    temporary.replace(path)


def simulator_is_idle_after_failure(config):
    """Never start another scenario over a partially executed Single Block run."""
    from .integrations.ncguide import FocasCppBridgeClient

    response = FocasCppBridgeClient(
        Path(config["bridge"]), Path(config["dll"]),
        host=config["host"], port=int(config["port"]), timeout_seconds=5,
    ).call("ReadRunStatus")
    if response.get("status_code") != 0:
        return False
    data = response.get("data") or response.get("response") or response
    return isinstance(data, dict) and str(data.get("run")) in {"0", "1"} and all(
        str(data.get(key)) == "0" for key in ("motion", "alarm", "emergency")
    )


def capture_unavailable_after_batch(errors: list[str], folder: Path) -> bool:
    """Distinguish sniffer failure from an API with no matched target traffic."""
    fatal_markers = (
        "PCAP_CAPTURE_FAILED", "PCAP_UNAVAILABLE", "PCAP_OPEN_LIVE_FAILED",
        "PCAP_DUMP_OPEN_FAILED", "CAPTURE_NOT_STARTED",
    )
    if any(marker in error.upper() for error in errors for marker in fatal_markers):
        return True
    return not any(path.is_file() for path in folder.rglob("*.pcap"))


def manual_operator_action_timed_out(folder: Path) -> bool:
    stdout_path = folder / "execution" / "api_script_stdout.txt"
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        return "MANUAL_ACTION_TIMEOUT" in text or "MANUAL_ACTION_DEFERRED" in text
    except OSError:
        return False


def worker():
    os.chdir(ROOT)
    config = json.loads(sys.stdin.readline())
    secret = config.get("api_key", "")
    # Redact third-party diagnostics too, before they cross the process boundary.
    class Redacted:
        def __init__(self, stream): self.stream = stream
        def write(self, value): return self.stream.write(value.replace(secret, "[REDACTED]") if secret else value)
        def flush(self): self.stream.flush()
    sys.stdout = Redacted(sys.stdout)
    sys.stderr = Redacted(sys.stderr)
    def emit(event, data):
        print(EVENT + json.dumps({"event": event, "data": data}, ensure_ascii=False, default=str), flush=True)
    try:
        action = config["action"]
        if action != "llm":
            require_capture_adapter(config)
        if protocol_profile(config)["available"]:
            configure(config)
        else:
            os.environ["DEEPSEEK_API_KEY"] = str(config.get("api_key", ""))
        if action != "capture":
            message = {"simulator": test_simulator, "llm": test_llm, "preflight": preflight}[action](config)
            emit("success", {"message": message})
            return 0
        emit("status", {"message": "正在重新检查采集条件"})
        emit("status", {"message": preflight(config)})
        from .knowledge import KnowledgeBase
        from .models import TaskRequest
        from .workflow import TrafficGenerationWorkflow, workflow_success
        from .utils import write_json
        out = capture_output_dir(config)
        emit("output", {"path": str(out)})
        safe = {k: v for k, v in config.items() if k != "api_key"}
        if not (out / "settings.json").exists():
            write_json(out / "settings.json", safe)
        kb = KnowledgeBase(vector_db_dir=Path(config["vector"]))
        catalog = {s["scenario_id"]: s for s in scenarios()}
        if config.get("capture_mode") != "single_scene" and ("selected_apis" in config or is_full_api_mode(config)):
            return run_full_api_capture(config, out, kb, emit)
        total = len(config["selected"])
        progress_path = out / "capture_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"scenarios": {}}
        failures = []
        stop_after_failure = False
        for index, sid in enumerate(config["selected"], 1):
            if stop_after_failure:
                break
            scene = filter_scene_for_config(catalog[sid], config)
            emit("scenario", {"index": index, "total": total, "name": scene["scenario_name"]})
            if scene_has_verified_success(out, sid):
                emit("scenario_done", {"index": index, "total": total, "resumed": True})
                continue
            folder = next_scene_folder(out, index, scene)
            request = TaskRequest(description=render_prompt(config, scene), task_id=folder.name,
                protocol=config.get("protocol", "focas"), target_environment="ncguide-generated-cpp", quality_gate_enabled=True,
                permissions={"representative_scenario": True, "selected_scenario_ids": [sid],
                    "required_api_functions": atomic_scene_api_names(scene),
                    "allow_delete_all_programs": False, "historical_memory_enabled": True})
            try:
                workflow = TrafficGenerationWorkflow(kb, llm_client=make_llm(config), progress_callback=emit)
                state = workflow.run(request, folder)
                if not workflow_success(state):
                    raise RuntimeError("; ".join(state.errors) or "未通过场景质量与流量校验")
                pcaps = [p for p in folder.rglob("*.pcap") if p.is_file()]
                local_only = state.result is not None and state.result.collection_status == "local_success"
                if not pcaps and not local_only:
                    raise RuntimeError("Npcap 未保存 PCAP 文件")
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures.append(f"{scene['scenario_name']}: {message}")
                progress["scenarios"][sid] = {"status": "failed", "folder": str(folder), "error": message}
                emit("scenario_failed", {"index": index, "total": total, "name": scene["scenario_name"], "error": message})
                manual_action_unresolved = manual_operator_action_timed_out(folder)
                if manual_action_unresolved:
                    os.environ["SMP_SKIP_MANUAL_ACTION_PROMPTS"] = "1"
                try:
                    stop_after_failure = not manual_action_unresolved and not simulator_is_idle_after_failure(config)
                except Exception:
                    stop_after_failure = not manual_action_unresolved
                if stop_after_failure:
                    emit("status", {"message": "控制器未证实为空闲；为避免叠加运行，停止后续场景，可在恢复空闲后续采"})
                elif manual_action_unresolved:
                    emit("status", {"message": "人工操作等待结束；当前场景已保留失败记录，本轮继续其他可独立采集的场景"})
            else:
                if state.result is not None and state.result.collection_status == "partial":
                    progress["scenarios"][sid] = {
                        "status": "partial", "folder": str(folder),
                        "collection_evidence": state.result.collection_evidence,
                    }
                    emit("scenario_partial", {"index": index, "total": total, "name": scene["scenario_name"]})
                else:
                    progress["scenarios"][sid] = {"status": "success", "folder": str(folder)}
                    emit("scenario_done", {"index": index, "total": total})
            finally:
                write_capture_progress(progress_path, progress)
        if failures:
            raise RuntimeError(f"{len(failures)}/{total} 个场景未通过；可指定本次输出目录续采。详情：" + " | ".join(failures))
        partial_scenes = [record for record in progress.get("scenarios", {}).values()
                          if record.get("status") == "partial"]
        if partial_scenes:
            emit("partial", {"message": f"执行完成；{len(partial_scenes)} 个场景的目标网络证据不足，原始日志与 PCAP 已保留。"})
            return 0
        emit("success", {"message": f"已完成 {total} 个场景，结果：{out}"})
        return 0
    except Exception as exc:
        emit("failure", {"message": f"{type(exc).__name__}: {exc}"})
        return 1



@contextmanager
def _collection_csv_lock(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def refresh_overall_collection_csv(path: Path) -> dict[str, int]:
    """Serialize incremental ledger merges across desktop worker processes."""
    with _collection_csv_lock(path):
        return _refresh_overall_collection_csv_locked(path)


def _refresh_overall_collection_csv_locked(path: Path) -> dict[str, int]:
    """Merge verified traffic from every desktop run into the 308-API ledger."""
    import csv
    import hashlib
    import shutil
    from collections import defaultdict
    from .function_coverage import is_non_wire_observable
    from .traffic_transactions import annotate_session_transactions, valid_session_segment_evidence
    from .execution_contracts import API_TRAFFIC_SESSION_RULES

    if not path.is_file():
        return {}
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    encoding = None
    for candidate in ("utf-8-sig", "utf-8", "utf-16", "cp936", "gb18030"):
        try:
            text = raw.decode(candidate, errors="strict")
        except UnicodeDecodeError:
            continue
        encoding = candidate
        break
    if encoding is None:
        raise UnicodeError(f"Unsupported or damaged collection CSV encoding: {path}")
    import io
    with io.StringIO(text, newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "function" not in fieldnames:
        raise ValueError(f"Invalid collection progress CSV: {path}")

    matched_success_runs = defaultdict(set)
    local_success_runs = defaultdict(set)
    fully_covered_runs = defaultdict(set)
    for run in sorted(path.parent.iterdir()):
        if not run.is_dir():
            continue
        for log_path in run.rglob("api_logs.jsonl"):
            pcap_paths = [p for p in log_path.parent.rglob("*.pcap") if p.stat().st_size > 24]
            transaction_path = log_path.parent / "protocol_transactions.jsonl"
            transactions = []
            if transaction_path.is_file():
                with transaction_path.open(encoding="utf-8") as transaction_stream:
                    for line in transaction_stream:
                        try:
                            row = json.loads(line)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if isinstance(row, dict):
                            transactions.append(row)
                annotate_session_transactions(transactions)
            by_call = {str(row.get("invocation_id")): row for row in transactions
                       if row.get("invocation_id")}
            with log_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        log = json.loads(line)
                    except ValueError:
                        continue
                    name = log.get("protocol_function")
                    if not (isinstance(name, str) and name.startswith(("cnc_", "pmc_"))
                            and log.get("status_code") == 0):
                        continue
                    if is_non_wire_observable(name):
                        local_success_runs[name].add(run.name)
                        continue
                    if not pcap_paths:
                        continue
                    transaction = by_call.get(str(log.get("invocation_id") or ""), {})
                    evidence = dict(log)
                    if transaction.get("session_traffic"):
                        evidence["session_traffic"] = transaction["session_traffic"]
                    wire_valid = (
                        valid_session_segment_evidence(evidence)
                        if name in API_TRAFFIC_SESSION_RULES
                        else log.get("traffic_status") == "matched"
                    )
                    if wire_valid:
                        matched_success_runs[name].add(run.name)
        state_path = run / "full_api_coverage_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for name, record in state.get("functions", {}).items():
                if (run.name in matched_success_runs[name]
                        and record.get("status") in ("covered_success", "covered_expected_error")
                        and record.get("traffic_matched") is True):
                    fully_covered_runs[name].add(run.name)

    for row in rows:
        name = row["function"]
        evidence = list(dict.fromkeys(
            item.strip() for item in row.get("evidence_runs", "").split(";") if item.strip()
        ))
        evidence.extend(sorted(matched_success_runs[name] - set(evidence)))
        evidence.extend(sorted(local_success_runs[name] - set(evidence)))
        row["evidence_runs"] = ";".join(evidence)
        if row.get("collected") or matched_success_runs[name] or local_success_runs[name]:
            row["collected"] = "✓"
            row["status"] = "已采集"
            if fully_covered_runs[name]:
                row["latest_status"] = "covered_success"
            elif local_success_runs[name] and row.get("latest_status") != "covered_success":
                row["latest_status"] = "local_success"
            elif row.get("latest_status") != "covered_success":
                row["latest_status"] = "traffic_observed"
    backup = path.with_name(path.name + f".bak.{digest[:16]}")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return {
        "function_count": len(rows),
        "collected_count": sum(bool(row.get("collected")) for row in rows),
        "fully_covered_count": sum(row.get("latest_status") == "covered_success" for row in rows),
        "traffic_observed_count": sum(row.get("latest_status") == "traffic_observed" for row in rows),
        "local_success_count": sum(row.get("latest_status") == "local_success" for row in rows),
    }


def write_selected_api_coverage_progress(
    out, coverage_path, selected, batch, status, *, targets=(), errors=(),
):
    """Atomically snapshot the entire collection run after every batch."""
    from .function_coverage import load_function_coverage_state, summarize_function_coverage_state
    from .traffic_transactions import valid_session_segment_evidence
    from .execution_contracts import API_TRAFFIC_SESSION_RULES
    from collections import Counter

    state = load_function_coverage_state(coverage_path)
    scoped = dict(state, functions={
        name: record for name, record in state.get("functions", {}).items() if name in selected
    })
    summary = summarize_function_coverage_state(scoped)
    completed_targets = {
        name for name, record in scoped["functions"].items()
        if (record.get("status") in ("covered_success", "covered_expected_error")
            and record.get("traffic_matched") is True)
        or record.get("status") == "local_success"
    }
    batches = []
    observed_calls = Counter()
    matched_calls = Counter()
    successful_matched_calls = Counter()
    eligible_calls = Counter()
    successful_eligible_calls = Counter()
    total_api_logs = total_capture_events = total_pcap_bytes = total_raw_pcap_bytes = 0
    raw_pcap_count = 0
    for folder in sorted(out.glob("batch_*")):
        if not folder.is_dir():
            continue
        summary_path = folder / "summary.json"
        summary_data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        api_log_path = folder / "execution" / "api_logs.jsonl"
        if api_log_path.is_file():
            for line in api_log_path.open(encoding="utf-8"):
                try:
                    log = json.loads(line)
                except (ValueError, TypeError):
                    continue
                name = str(log.get("protocol_function") or "")
                if not name.startswith(("cnc_", "pmc_")):
                    continue
                observed_calls[name] += 1
                eligible = (
                    valid_session_segment_evidence(log)
                    if name in API_TRAFFIC_SESSION_RULES
                    else log.get("traffic_status") == "matched"
                )
                if eligible:
                    eligible_calls[name] += 1
                    if log.get("status_code") == 0:
                        successful_eligible_calls[name] += 1
                if log.get("traffic_status") == "matched":
                    matched_calls[name] += 1
                    if log.get("status_code") == 0:
                        successful_matched_calls[name] += 1
        pcaps = [path for path in folder.rglob("*.pcap") if path.is_file()]
        pcap_bytes = sum(path.stat().st_size for path in pcaps)
        raw_pcaps = [path for path in pcaps if path.name == "focas_capture.pcap"]
        raw_pcap_bytes = sum(path.stat().st_size for path in raw_pcaps)
        api_logs = int(summary_data.get("api_log_count") or 0)
        capture_events = int(summary_data.get("capture_event_count") or 0)
        total_api_logs += api_logs
        total_capture_events += capture_events
        total_pcap_bytes += pcap_bytes
        total_raw_pcap_bytes += raw_pcap_bytes
        raw_pcap_count += len(raw_pcaps)
        coverage = summary_data.get("function_coverage") or {}
        batches.append({
            "batch": folder.name,
            "status": ("completed" if summary_data.get("success") else "failed") if summary_data else "running",
            "collection_status": summary_data.get("collection_status", "not_recorded"),
            "offline_audit_status": summary_data.get("offline_audit_status", "not_recorded"),
            "target_functions": coverage.get("attempted_target_functions") or sorted((coverage.get("parameter_coverage") or {}).get("functions", {})),
            "api_log_count": api_logs,
            "capture_event_count": capture_events,
            "pcap_count": len(pcaps),
            "pcap_bytes": pcap_bytes,
            "raw_pcap_bytes": raw_pcap_bytes,
            "parameter_coverage_passed": (coverage.get("parameter_coverage") or {}).get("passed"),
            "errors": summary_data.get("errors", []),
            "folder": str(folder),
        })
    api_progress = {}
    for name in sorted(selected):
        record = scoped["functions"].get(name, {})
        api_progress[name] = {
            "target_status": record.get("status") if record.get("coverage_role") == "target" else "pending",
            "target_covered": name in completed_targets,
            "observed_call_count": observed_calls[name],
            "matched_call_count": matched_calls[name],
            "successful_matched_call_count": successful_matched_calls[name],
            "eligible_call_count": eligible_calls[name],
            "successful_eligible_call_count": successful_eligible_calls[name],
            "observed_as_support": bool(observed_calls[name] and record.get("coverage_role") != "target"),
        }
    support_api_progress = {
        name: {
            "observed_call_count": observed_calls[name],
            "matched_call_count": matched_calls[name],
            "successful_matched_call_count": successful_matched_calls[name],
            "eligible_call_count": eligible_calls[name],
            "successful_eligible_call_count": successful_eligible_calls[name],
            "in_selected_manifest": name in selected,
        }
        for name in sorted(observed_calls)
        if scoped["functions"].get(name, {}).get("coverage_role") != "target"
    }
    payload = {
        "mode": "overall_capture_progress",
        "manifest_function_count": len(selected),
        "selected_api_functions": sorted(selected),
        "captured_function_count": int(summary.get("captured_function_count", 0)),
        "resolved_function_count": int(summary.get("resolved_function_count", 0)),
        "unavailable_function_count": int(summary.get("unavailable_function_count", 0)),
        "batch": batch,
        "status_counts": summary.get("status_counts", {}),
        "missing_target_functions_this_batch": sorted(set(targets) - completed_targets),
        "last_batch_status": status,
        "last_batch_targets": sorted(set(targets)),
        "last_batch_errors": list(errors),
        "overall_progress": {
            "batch_count": len(batches),
            "completed_batch_count": sum(item["status"] == "completed" for item in batches),
            "failed_batch_count": sum(item["status"] == "failed" for item in batches),
            "api_log_count": total_api_logs,
            "capture_event_count": total_capture_events,
            "pcap_bytes": total_pcap_bytes,
            "raw_pcap_count": raw_pcap_count,
            "raw_pcap_bytes": total_raw_pcap_bytes,
            "observed_api_function_count": len(observed_calls),
            "matched_api_function_count": len(matched_calls),
            "successful_matched_api_function_count": len(successful_matched_calls),
            "eligible_api_function_count": len(eligible_calls),
            "successful_eligible_api_function_count": len(successful_eligible_calls),
            "support_observed_function_count": len(support_api_progress),
            "support_matched_function_count": sum(bool(row["matched_call_count"]) for row in support_api_progress.values()),
            "pending_target_count": len(selected) - int(summary.get("resolved_function_count", 0)),
        },
        "api_progress": api_progress,
        "support_api_progress": support_api_progress,
        "batches": batches,
    }
    if status in ("completed", "failed"):
        try:
            payload["overall_csv_progress"] = refresh_overall_collection_csv(out.parent / "采集进度.csv")
        except (OSError, ValueError) as exc:
            # A locked or malformed ledger must not interrupt live capture.
            payload["overall_csv_update_error"] = f"{type(exc).__name__}: {exc}"
    write_capture_progress(out / "coverage_progress.json", payload)
    return payload


FULL_API_SCHEDULER_FILE = "full_api_scheduler_state.json"
FULL_API_SCHEDULER_SCHEMA = 1


def load_full_api_scheduler_state(path: Path, selected_order: list[str]) -> dict:
    selected = list(selected_order)
    if not path.is_file():
        return {
            "schema_version": FULL_API_SCHEDULER_SCHEMA,
            "selected_api_functions": selected,
            "attempted_this_run": [],
            "targets": {name: {"status": "pending", "attempt_count": 0} for name in selected},
            "auxiliary_coverage": {},
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"全 API 调度检查点无法读取：{path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != FULL_API_SCHEDULER_SCHEMA:
        raise ValueError(f"全 API 调度检查点版本不受支持：{path}")
    if state.get("selected_api_functions") != selected:
        raise ValueError("续采目标清单与全 API 调度检查点不一致；请使用原目标清单续采")
    state.setdefault("attempted_this_run", [])
    state.setdefault("targets", {})
    state.setdefault("auxiliary_coverage", {})
    for name in selected:
        state["targets"].setdefault(name, {"status": "pending", "attempt_count": 0})
    return state


def save_full_api_scheduler_state(path: Path, state: dict) -> None:
    from datetime import datetime, timezone

    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _api_log_is_successful_coverage(function: str, row: dict) -> bool:
    try:
        return_code = int(row.get("status_code", row.get("api_return_code", 9999)))
    except (TypeError, ValueError):
        return False
    if return_code != 0 or row.get("error"):
        return False
    from .function_coverage import is_non_wire_observable
    if is_non_wire_observable(function):
        return True
    from .execution_contracts import API_TRAFFIC_SESSION_RULES
    if function in API_TRAFFIC_SESSION_RULES:
        from .traffic_transactions import valid_session_segment_evidence
        return valid_session_segment_evidence(row)
    return row.get("traffic_status") == "matched"


def discover_successful_auxiliary_coverage(run_roots: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Find auxiliary calls whose evidence meets the API's independent success rule."""
    evidence: dict[str, list[dict[str, str]]] = {}
    batch_folders: list[Path] = []
    seen_roots: set[Path] = set()
    for candidate in run_roots:
        try:
            resolved_root = candidate.resolve()
        except OSError:
            continue
        if resolved_root in seen_roots or not resolved_root.is_dir():
            continue
        seen_roots.add(resolved_root)
        if re.fullmatch(r"batch_\d+(?:_retry_\d+)?", resolved_root.name):
            batch_folders.append(resolved_root)
        elif any(resolved_root.glob("batch_*")):
            batch_folders.extend(resolved_root.glob("batch_*"))
        else:
            for run_folder in resolved_root.iterdir():
                if run_folder.is_dir():
                    batch_folders.extend(run_folder.glob("batch_*"))
    for batch_folder in batch_folders:
        if not batch_folder.is_dir():
            continue
        summary_path = batch_folder / "summary.json"
        log_path = batch_folder / "execution" / "api_logs.jsonl"
        if not summary_path.is_file() or not log_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        coverage = summary.get("function_coverage") or {}
        support_functions = set(coverage.get("support_functions") or [])
        if not support_functions:
            support_functions = set(coverage.get("attempted_support_functions") or [])
        if not support_functions:
            continue
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            function = str(row.get("protocol_function") or "")
            if function not in support_functions or not _api_log_is_successful_coverage(function, row):
                continue
            evidence.setdefault(function, []).append({
                "run": batch_folder.parent.name,
                "batch": batch_folder.name,
                "invocation_id": str(row.get("invocation_id") or ""),
            })
    return evidence


def full_api_scheduler_summary(state: dict) -> dict:
    targets = state.get("targets", {})
    counts = {key: 0 for key in ("success", "failure", "blocked", "partial", "historical_skip", "unprocessed")}
    execution_counts = {}
    repair_counts = []
    for name in state.get("selected_api_functions", []):
        record = targets.get(name, {})
        status = record.get("status", "pending")
        if status == "history_skipped":
            counts["historical_skip"] += 1
        elif status == "success":
            counts["success"] += 1
        elif status == "failed":
            counts["failure"] += 1
        elif status == "blocked":
            counts["blocked"] += 1
        elif status == "partial":
            counts["partial"] += 1
        else:
            counts["unprocessed"] += 1
        execution_counts[name] = int(record.get("attempt_count", 0))
        if record.get("attempt_count"):
            repair_counts.append(int(record.get("repair_attempts", 0)))
    attempted = len(set(state.get("attempted_this_run", [])))
    return {
        "selected_target_count": len(state.get("selected_api_functions", [])),
        "attempted_target_count": attempted,
        "workflow_attempt_count": sum(execution_counts.values()),
        "duplicate_target_attempt_count": sum(max(0, count - 1) for count in execution_counts.values()),
        "average_repair_attempts": round(sum(repair_counts) / len(repair_counts), 3) if repair_counts else 0.0,
        **counts,
        "target_execution_counts": execution_counts,
    }


def workflow_failure_category(errors: list[str], result=None, coverage_record: dict | None = None) -> str:
    text = " ".join(str(error) for error in errors).lower()
    blocked_markers = (
        "api_prerequisite_blocked", "blocked_permission_required", "ew_noopt",
        "environment_precheck_failed", "environment preflight", "preflight blocked",
        "cycle_start_configuration_missing", "ncguide_window_not_found",
        "controller_not_confirmed_idle", "pcap_capture_failed", "pcap_unavailable",
    )
    blocked_statuses = {
        "unsupported_by_header", "export_missing", "option_unsupported",
        "blocked_permission_required", "blocked_prerequisite",
    }
    if (any(marker in text for marker in blocked_markers)
            or str((coverage_record or {}).get("status", "")) in blocked_statuses):
        return "blocked"
    execution = getattr(result, "result", None)
    if execution is not None and (
        execution.collection_status == "partial"
        or bool(execution.api_logs)
        or bool(execution.capture_events)
    ):
        return "partial"
    return "failed"


def run_full_api_capture(config, out, knowledge_base, emit):
    require_capture_adapter(config)
    """Run one manifest target per workflow; each workflow keeps its repair budget."""
    import re
    from .function_coverage import (
        build_function_coverage_metrics, load_function_coverage_state,
        summarize_function_coverage_state, has_captured_traffic, has_successful_coverage,
        load_resolved_functions,
    )
    from .models import TaskRequest
    from .rag.focas_function_manifest import build_function_coverage_context
    from .workflow import TrafficGenerationWorkflow, workflow_success
    from .utils import write_json
    from .diversity_audit import load_recent_diversity_feedback
    from datetime import datetime, timezone

    selected = set(selected_api_names(config))
    manifest = [row for row in function_manifest() if row["function"] in selected]
    selected_order = [str(row["function"]) for row in manifest]
    function_count = len(manifest)
    single_api = config.get("capture_mode") == "single_api"
    if not function_count:
        raise ValueError("没有选中可采集的 API")
    coverage_path = out / "full_api_coverage_state.json"
    coverage_state = load_function_coverage_state(coverage_path)

    def batch_has_target_call(folder):
        log_path = folder / "execution" / "api_logs.jsonl"
        try:
            with log_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if record.get("protocol_function") in selected:
                        return True
        except OSError:
            pass
        return False

    current_run_batches = [
        path for path in out.iterdir() if path.is_dir()
        and re.fullmatch(r"batch_\d+(?:_retry_\d+)?", path.name)
        and batch_has_target_call(path)
    ]

    # A fresh single-API run must perform a real target call even when its
    # local state was preseeded by an older all-API run. Keep coverage only
    # when this same output directory already contains evidence from a run.
    if single_api and not current_run_batches:
        functions = coverage_state.setdefault("functions", {})
        for function in selected:
            functions.pop(function, None)

    # Import prior state, but skip only records with valid API success evidence.
    historical_paths = [] if single_api else [ROOT / "rag_indexes/focas/function_coverage_state.json"]
    history_root = out.parent
    if not single_api and history_root.is_dir():
        historical_paths.extend(
            path for path in history_root.glob("*/full_api_coverage_state.json")
            if path.resolve() != coverage_path.resolve()
        )
    from .function_coverage import has_captured_traffic, is_resolved
    functions = coverage_state.setdefault("functions", {})
    historical_success: set[str] = set()
    for historical_path in historical_paths:
        historical = load_function_coverage_state(historical_path)
        for function, candidate in historical.get("functions", {}).items():
            if not isinstance(candidate, dict):
                continue
            if has_successful_coverage(candidate):
                historical_success.add(str(function))
            current = functions.get(function)
            if not isinstance(current, dict):
                functions[function] = dict(candidate)
                continue
            # Prefer a terminal record, then a record with matched traffic.
            current_rank = (int(is_resolved(current)), int(has_captured_traffic(current)))
            candidate_rank = (int(is_resolved(candidate)), int(has_captured_traffic(candidate)))
            if candidate_rank > current_rank:
                functions[function] = dict(candidate)
    header_path = Path(config["header"]) / "Fwlib32.h"
    header = header_path.read_text(encoding="latin-1")
    header_blocked: set[str] = set()
    for row in manifest:
        function = str(row["function"])
        if (not re.search(rf"\b{re.escape(function)}\s*\(", header, re.IGNORECASE)
                and not has_successful_coverage(functions.get(function, {}))):
            header_blocked.add(function)
            functions[function] = {
                "function": function,
                "status": "unsupported_by_header",
                "terminal": True,
                "coverage_role": "target",
                "evidence": f"symbol absent from configured header: {header_path}",
            }
    write_json(coverage_path, coverage_state)
    # Each target receives one outer workflow dispatch. Repair attempts stay
    # inside that workflow instance and never consume the next target's budget.
    natural_max_batches = 1 if single_api else max(1, function_count)
    try:
        requested_batch_limit = int(os.environ.get("FOCAS_MAX_CAPTURE_BATCHES", "0"))
    except (TypeError, ValueError):
        requested_batch_limit = 0
    batch_limit = requested_batch_limit if requested_batch_limit > 0 else natural_max_batches
    max_batches = min(natural_max_batches, batch_limit)
    existing_batches = [
        int(match.group(1))
        for path in out.iterdir() if path.is_dir()
        if (match := re.fullmatch(r"batch_(\d+)(?:_retry_\d+)?", path.name))
    ]
    first_batch = max(existing_batches, default=0) + 1
    def selected_summary(state):
        scoped = dict(state, functions={
            name: record for name, record in state.get("functions", {}).items() if name in selected
        })
        return summarize_function_coverage_state(scoped)

    write_selected_api_coverage_progress(
        out, coverage_path, selected, first_batch - 1, "ready",
    )
    scheduler_path = out / FULL_API_SCHEDULER_FILE
    scheduler_state = None if single_api else load_full_api_scheduler_state(scheduler_path, selected_order)
    attempted_this_run: set[str] = set()
    target_outcomes: dict[str, dict] = {}
    historical_auxiliary: dict[str, list[dict[str, str]]] = {}
    if not single_api:
        historical_roots = [out.parent]
        historical_auxiliary = discover_successful_auxiliary_coverage(historical_roots)
        scheduler_state["auxiliary_coverage"].update(historical_auxiliary)
        attempted_this_run = set(str(name) for name in scheduler_state.get("attempted_this_run", []))
        target_outcomes = scheduler_state["targets"]
        current_functions = load_function_coverage_state(coverage_path).get("functions", {})
        for function in selected_order:
            outcome = target_outcomes[function]
            if function in attempted_this_run and outcome.get("status") == "running":
                if has_successful_coverage(current_functions.get(function, {})):
                    outcome.update(status="success", recovered_after_interrupt=True)
                else:
                    outcome.update(status="failed", errors=["INTERRUPTED_DURING_WORKFLOW"])
            elif function not in attempted_this_run and outcome.get("status") == "pending":
                current_record = current_functions.get(function, {})
                if has_successful_coverage(current_record):
                    outcome.update(status="history_skipped", skip_source="run_coverage_state")
                elif function in historical_success:
                    outcome.update(status="history_skipped", skip_source="historical_coverage_state")
                elif function in historical_auxiliary:
                    outcome.update(status="history_skipped", skip_source="verified_auxiliary_traffic")
                elif function in header_blocked:
                    outcome.update(status="blocked", reason="symbol absent from configured SDK header")
            elif outcome.get("status") == "pending" and function in header_blocked:
                outcome.update(status="blocked", reason="symbol absent from configured SDK header")
        scheduler_state["attempted_this_run"] = [name for name in selected_order if name in attempted_this_run]
        save_full_api_scheduler_state(scheduler_path, scheduler_state)
    batch_failures_path = out / "coverage_batch_failures.json"
    if batch_failures_path.is_file():
        try:
            batch_failures = list(json.loads(batch_failures_path.read_text(encoding="utf-8")).get("failures", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            batch_failures = []
    else:
        batch_failures = []
    parameter_path = out / "actual_parameter_coverage.json"
    parameter_records = json.loads(parameter_path.read_text(encoding="utf-8")).get("records", []) if parameter_path.is_file() else []
    planned_any_target = False
    for batch in range(first_batch, first_batch + max_batches):
        if single_api:
            eligible_targets = selected - load_resolved_functions(coverage_path)
            if not eligible_targets:
                break
            covered_for_planner = load_resolved_functions(coverage_path)
        else:
            eligible_targets = {
                name for name in selected_order
                if target_outcomes[name].get("status") == "pending"
                and name not in attempted_this_run
            }
            if not eligible_targets:
                break
            covered_for_planner = selected - eligible_targets
        preview = build_function_coverage_context(
            ROOT / "rag_indexes/focas/function_manifest.jsonl",
            covered_functions=covered_for_planner,
            target_functions=selected,
            batch_size=API_CAPTURE_BATCH_SIZE,
            max_segments=1,
        )
        current_targets = [
            str(row["function"]) for row in preview.get("target_batch", [])
            if str(row.get("function", "")) in eligible_targets
        ]
        # Enforce one target per workflow even if manifest selection later
        # returns more than the configured limit.
        current_targets = (current_targets or [name for name in selected_order if name in eligible_targets])[:1]
        if not current_targets:
            break
        current_target = current_targets[0]
        if not single_api:
            attempted_this_run.add(current_target)
            scheduler_state["attempted_this_run"] = [
                name for name in selected_order if name in attempted_this_run
            ]
            target_outcomes[current_target].update(
                status="running", attempt_count=1, batch=batch,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            save_full_api_scheduler_state(scheduler_path, scheduler_state)
        write_selected_api_coverage_progress(
            out, coverage_path, selected, batch, "running", targets=current_targets,
        )
        state_before = load_function_coverage_state(coverage_path)
        summary_before = selected_summary(state_before)
        resolved_before = int(summary_before.get("resolved_function_count", 0))
        captured_before = int(summary_before.get("captured_function_count", 0))
        if resolved_before >= function_count:
            break
        emit("coverage_batch", {"batch": batch, "single_api": single_api, "covered": captured_before, "resolved": resolved_before, "total": function_count})
        base = out / f"batch_{batch:03d}"
        folder = base
        retry = 1
        while folder.exists():
            folder = out / f"batch_{batch:03d}_retry_{retry:03d}"
            retry += 1
        request = TaskRequest(
            description=render_full_api_prompt(config, batch_targets=current_targets), task_id=folder.name,
            protocol=config.get("protocol", "focas"), target_environment="ncguide-generated-cpp", quality_gate_enabled=True,
            permissions={
                "representative_scenario": False,
                "full_api_parameter_coverage": True,
                "coverage_target_functions": current_targets,
                "coverage_current_batch_functions": current_targets,
                "coverage_skip_functions": [],
                "require_planned_target_traffic": True,
                "allow_delete_all_programs": False,
                "historical_memory_enabled": True,
                "capture_samples": int(config["samples"]),
                "capture_interval_seconds": float(config["interval"]),
                "historical_diversity_feedback": load_recent_diversity_feedback(
                    out.parent, current_targets, exclude_run=out,
                ),
            },
        )
        result = None
        workflow_exception = None
        try:
            workflow = TrafficGenerationWorkflow(
                knowledge_base, llm_client=make_llm(config), progress_callback=emit,
                coverage_state_path=coverage_path,
                coverage_batch_size=API_CAPTURE_BATCH_SIZE,
                coverage_segment_count=1,
            )
            result = workflow.run(request, folder)
        except Exception as exc:
            workflow_exception = exc
        if batch_has_target_call(folder):
            current_run_batches.append(folder)
        planned_any_target = planned_any_target or bool(result and result.plan is not None)
        manual_action_unresolved = manual_operator_action_timed_out(folder)
        if manual_action_unresolved:
            os.environ["SMP_SKIP_MANUAL_ACTION_PROMPTS"] = "1"
        logs = result.result.api_logs if result and result.result else []
        for log in logs:
            parameter_records.append({
                "batch": batch,
                "protocol_function": log.protocol_function,
                "step_id": log.step_id,
                "input_parameters": log.input_parameters,
                "status_code": log.status_code,
                "error": log.error,
            })
        write_json(parameter_path, {
            "description": "Actual input parameter values sent to FOCAS, grouped by batch/call.",
            "record_count": len(parameter_records),
            "records": parameter_records,
        })
        workflow_ok = result is not None and workflow_success(result)
        if workflow_exception is not None or not workflow_ok:
            errors = ([f"{type(workflow_exception).__name__}: {workflow_exception}"]
                      if workflow_exception is not None else
                      [str(error) for error in (result.errors or (result.result.errors if result.result else []))])
            if not errors:
                errors = ["workflow returned without a successful terminal state"]
            failure = {
                "batch": batch, "single_api": single_api,
                "targets": [current_target], "errors": errors, "folder": str(folder),
                "repair_attempts": int(result.repair_attempts) if result else 0,
                "repair_history": list(result.repair_history) if result else [],
                "workflow_attempt_count": 1,
            }
            manual_timeout = manual_action_unresolved
            if manual_timeout:
                failure["deferred_for_manual_action"] = True
                failure["resume_hint"] = "完成仿真器 REMOTE/RESET 操作后重新续采；本轮将先继续其他 API。"
            batch_failures.append(failure)
            write_json(batch_failures_path, {"failures": batch_failures})
            write_selected_api_coverage_progress(
                out, coverage_path, selected, batch, "failed",
                targets=[current_target], errors=errors,
            )
            emit("coverage_batch_failed", failure)
            if single_api:
                raise RuntimeError(f"单 API 采集失败，已结束本次任务：{errors}")
            capture_attempted = bool(result and result.result and (
                result.result.api_logs
                or any(call.tool_name == "run_generated_executable" for call in result.result.tool_calls)
            )) or (folder / "execution/api_script_stdout.txt").is_file()
            capture_failed = capture_attempted and capture_unavailable_after_batch(errors, folder)
            coverage_record = load_function_coverage_state(coverage_path).get("functions", {}).get(current_target, {})
            category = "blocked" if manual_timeout or capture_failed else workflow_failure_category(
                errors, result, coverage_record,
            )
            target_outcomes[current_target].update(
                status=category, errors=errors, folder=str(folder),
                repair_attempts=int(result.repair_attempts) if result else 0,
                repair_history=list(result.repair_history) if result else [],
                collection_status=result.result.collection_status if result and result.result else "not_run",
            )
            successful_aux = discover_successful_auxiliary_coverage([folder])
            for name, evidence in successful_aux.items():
                scheduler_state["auxiliary_coverage"].setdefault(name, []).extend(evidence)
                if name in target_outcomes and name not in attempted_this_run and target_outcomes[name].get("status") == "pending":
                    target_outcomes[name].update(status="history_skipped", skip_source="verified_auxiliary_traffic")
            stop_reason = ""
            if capture_failed:
                stop_reason = "PCAP_CAPTURE_UNAVAILABLE"
            elif capture_attempted and not manual_timeout:
                try:
                    if not simulator_is_idle_after_failure(config):
                        stop_reason = "CONTROLLER_NOT_CONFIRMED_IDLE"
                except Exception as exc:
                    stop_reason = f"CONTROLLER_IDLE_CHECK_FAILED: {exc}"
            if stop_reason:
                target_outcomes[current_target].update(status="blocked", stop_reason=stop_reason)
                scheduler_state["stop_reason"] = stop_reason
                scheduler_state["stopped_at_batch"] = batch
            save_full_api_scheduler_state(scheduler_path, scheduler_state)
            if manual_timeout:
                os.environ["SMP_SKIP_MANUAL_ACTION_PROMPTS"] = "1"
            emit("status", {
                "message": f"目标 {current_target} 工作流结束（{category}），修复 {failure['repair_attempts']} 次；已保留证据，不在本轮全局重调。"
            })
            if stop_reason:
                emit("status", {"message": f"后续目标暂未处理：{stop_reason}"})
                break
            continue
        pcaps = [path for path in folder.rglob("*.pcap") if path.is_file()]
        local_only = result.result is not None and result.result.collection_status == "local_success"
        if not pcaps and not local_only:
            error = "Npcap 未保存 PCAP 文件"
            if single_api:
                raise RuntimeError(f"单 API 采集期间 {error}")
            target_outcomes[current_target].update(
                status="blocked", errors=[error], folder=str(folder),
                repair_attempts=int(result.repair_attempts),
                repair_history=list(result.repair_history),
            )
            scheduler_state["stop_reason"] = "PCAP_OUTPUT_MISSING"
            scheduler_state["stopped_at_batch"] = batch
            save_full_api_scheduler_state(scheduler_path, scheduler_state)
            emit("coverage_batch_failed", {
                "batch": batch, "targets": [current_target], "errors": [error], "folder": str(folder),
            })
            break
        metrics = build_function_coverage_metrics(result.plan, result.result)
        state_after = load_function_coverage_state(coverage_path)
        summary_after = selected_summary(state_after)
        resolved_after = int(summary_after.get("resolved_function_count", 0))
        captured_after = int(summary_after.get("captured_function_count", 0))
        emit("coverage_progress", {"batch": batch, "single_api": single_api, "covered": captured_after, "resolved": resolved_after, "total": function_count})
        target_record = state_after.get("functions", {}).get(current_target, {})
        if not single_api:
            if has_successful_coverage(target_record):
                target_outcomes[current_target].update(status="success")
            else:
                target_outcomes[current_target].update(status="partial")
            target_outcomes[current_target].update(
                folder=str(folder),
                repair_attempts=int(result.repair_attempts),
                repair_history=list(result.repair_history),
                collection_status=result.result.collection_status if result.result else "not_run",
                target_call_count=sum(
                    log.protocol_function == current_target for log in (result.result.api_logs if result.result else [])
                ),
            )
            successful_aux = discover_successful_auxiliary_coverage([folder])
            for name, evidence in successful_aux.items():
                scheduler_state["auxiliary_coverage"].setdefault(name, []).extend(evidence)
                if name in target_outcomes and name not in attempted_this_run and target_outcomes[name].get("status") == "pending":
                    target_outcomes[name].update(status="history_skipped", skip_source="verified_auxiliary_traffic")
            save_full_api_scheduler_state(scheduler_path, scheduler_state)
        write_selected_api_coverage_progress(
            out, coverage_path, selected, batch, "completed",
            targets=current_targets,
        )
    if not single_api:
        scheduler_metrics = full_api_scheduler_summary(scheduler_state)
        scheduler_state["summary"] = scheduler_metrics
        scheduler_state["run_status"] = "completed" if scheduler_metrics["unprocessed"] == 0 else "partial"
        save_full_api_scheduler_state(scheduler_path, scheduler_state)
        progress_path = out / "coverage_progress.json"
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            progress = {}
        progress["last_batch_status"] = (
            "partial"
            if any(scheduler_metrics[key] for key in ("failure", "blocked", "partial", "unprocessed"))
            else "completed"
        )
        progress["last_batch_targets"] = list(scheduler_state.get("selected_api_functions", []))
        progress["last_batch_errors"] = [
            error
            for record in scheduler_state.get("targets", {}).values()
            for error in record.get("errors", [])
            if isinstance(record, dict)
        ]
        progress["scheduler_summary"] = scheduler_metrics
        progress["scheduler_state_path"] = str(scheduler_path)
        write_capture_progress(progress_path, progress)
        emit("coverage_summary", scheduler_metrics)
        if any(scheduler_metrics[key] for key in ("failure", "blocked", "partial", "unprocessed")):
            emit("partial", {
                "message": f"全 API 逐个调度结束：成功 {scheduler_metrics['success']}，失败 {scheduler_metrics['failure']}，阻断 {scheduler_metrics['blocked']}，部分采集 {scheduler_metrics['partial']}，历史跳过 {scheduler_metrics['historical_skip']}，未处理 {scheduler_metrics['unprocessed']}。",
                **scheduler_metrics,
            })
        else:
            emit("success", {
                "message": f"全 API 调度完成：成功 {scheduler_metrics['success']}，失败 0，阻断 0，部分采集 0，历史跳过 {scheduler_metrics['historical_skip']}，未处理 0。",
                **scheduler_metrics,
            })
        return 0
    final_state = load_function_coverage_state(coverage_path)
    final_summary = selected_summary(final_state)
    captured = int(final_summary.get("captured_function_count", 0))
    resolved = int(final_summary.get("resolved_function_count", 0))
    if not planned_any_target and resolved < function_count:
        unresolved = sorted(selected - load_resolved_functions(coverage_path))
        raise RuntimeError(
            f"未判定 API：本轮所有工作流均未生成可验证计划；未完成目标：{unresolved}。"
            f"规划/生成失败证据见 {out / 'coverage_batch_failures.json'}"
        )
    if not current_run_batches:
        message = (
            "本轮没有执行任何目标 API；历史覆盖只用于全局进度，不能计为本轮真实采集成功。"
        )
        write_selected_api_coverage_progress(
            out, coverage_path, selected, first_batch - 1, "partial",
            errors=["NO_CURRENT_RUN_BATCH_EXECUTED"],
        )
        emit("partial", {"message": message})
        return 0
    if resolved < function_count:
        write_selected_api_coverage_progress(
            out, coverage_path, selected, batch, "partial",
            targets=set(selected) - load_resolved_functions(coverage_path),
        )
        emit("partial", {"message": f"已结束在线执行流程；{captured}/{function_count} 个 API 已采集，{function_count - resolved} 个仍需补采或环境确认。原始批次已保留：{out}"})
        return 0
    if captured < function_count:
        write_selected_api_coverage_progress(out, coverage_path, selected, batch, "partial")
        emit("partial", {"message": f"在线执行流程完成；{captured}/{function_count} 个 API 有流量证据，其余已记录为环境限制或不可用。状态见 {coverage_path}"})
        return 0
    label = "单 API" if single_api else "全部 API"
    emit("success", {"message": f"{label} 与参数流量采集完成：{captured}/{function_count}，结果：{out}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(worker())

