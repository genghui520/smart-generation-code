from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils import write_json


RULE_TYPES = {
    "nc_rule": "定义用什么 NC 程序构造加工场景。",
    "operation_rule": "定义如何在仿真器中启动、暂停、恢复、结束该场景。",
    "collection_rule": "定义场景运行前/中/后调用哪些 API 采集流量。",
    "safety_rule": "定义仿真器中哪些操作允许、哪些操作需要限制、哪些异常流量可以生成。",
}


@dataclass(slots=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    goal: str
    traffic_value: list[str]
    typical_nc_program: list[str]
    operation_phases: list[str]
    recommended_api_functions: list[str]
    distinguishing_signals: list[str]
    allowed_rule_types: list[str]


@dataclass(slots=True)
class ProtocolTaxonomy:
    protocol: str
    rule_types: dict[str, str]
    scenarios: list[ScenarioDefinition]
    rule_extraction_schema: dict[str, Any]

    @property
    def allowed_rule_types(self) -> set[str]:
        return set(self.rule_types)

    @property
    def allowed_scenarios(self) -> set[str]:
        return {scenario.scenario_id for scenario in self.scenarios}

    @property
    def fallback_scenario(self) -> str:
        if "general_status_collection" in self.allowed_scenarios:
            return "general_status_collection"
        if self.scenarios:
            return self.scenarios[0].scenario_id
        return ""


SCENARIOS = [
    ScenarioDefinition(
        scenario_id="coordinate_motion",
        name="坐标运动流量",
        goal="生成 X/Y/Z 或指定轴坐标随时间变化的流量。",
        traffic_value=["high_coverage", "high_distinguishability", "high_quality"],
        typical_nc_program=["valid program number", "G90 absolute mode", "G01 feed movement", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdposition", "cnc_rdspeed"],
        distinguishing_signals=["坐标值连续变化", "进给速度非零", "运行状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="feed_speed_change",
        name="进给速度变化流量",
        goal="生成进给速度、进给倍率或加减速相关变化流量。",
        traffic_value=["high_distinguishability", "high_quality"],
        typical_nc_program=["G01/G02/G03 feed movement", "multiple F values", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdspeed", "cnc_rdposition"],
        distinguishing_signals=["进给速度变化", "坐标变化", "运行状态为运行中"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="spindle_start_stop",
        name="主轴启停流量",
        goal="生成主轴启动、停止及状态变化流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["M03 or M04 spindle start", "M05 spindle stop", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdspmeter", "cnc_rdspeed"],
        distinguishing_signals=["主轴从停止到旋转", "主轴停止事件", "主轴状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="spindle_speed_change",
        name="主轴转速变化流量",
        goal="生成不同 S 指令或主轴速度变化引起的流量。",
        traffic_value=["high_distinguishability", "high_quality"],
        typical_nc_program=["M03 spindle start", "multiple S values", "M05 spindle stop", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdspmeter", "cnc_statinfo"],
        distinguishing_signals=["主轴转速变化", "主轴启动和停止阶段可区分"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="program_lifecycle",
        name="程序生命周期流量",
        goal="覆盖程序准备、上传、选择、启动、运行、停止、查询等生命周期行为。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["valid program number", "short executable NC program", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_search", "cnc_rdprogdir", "cnc_upload", "cnc_download"],
        distinguishing_signals=["程序号变化", "程序目录变化", "运行状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="auto_run_pause_resume",
        name="自动运行暂停恢复流量",
        goal="生成自动运行中暂停、恢复、停止等状态转换流量。",
        traffic_value=["high_distinguishability", "high_quality"],
        typical_nc_program=["short executable NC program", "movement or spindle command", "M30 end"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdposition", "cnc_rdspeed"],
        distinguishing_signals=["running/paused/completed 状态变化", "暂停期间运动量停止变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="mdi_execution",
        name="MDI 指令执行流量",
        goal="生成 MDI 方式下单段或短指令执行相关流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["single MDI block", "safe command such as status or simple motion in simulator"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdposition", "cnc_rdspeed"],
        distinguishing_signals=["MDI 方式状态", "单段执行前后状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="manual_jog",
        name="手动/JOG/手轮进给流量",
        goal="生成手动进给、JOG 或手轮相关运动流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["not required or simulator-driven manual movement"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdposition", "cnc_rdspeed"],
        distinguishing_signals=["手动方式", "坐标随手动操作变化", "进给速度变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="reference_return",
        name="参考点返回流量",
        goal="生成参考点返回、回零、位置建立相关流量。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["G28/G30 or simulator reference return operation"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_rdposition"],
        distinguishing_signals=["坐标向参考点变化", "参考点相关状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="work_coordinate_setting",
        name="工件坐标系/偏置流量",
        goal="生成工件坐标系、坐标偏置、坐标设定相关流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["G54-G59 coordinate system use", "safe coordinate offset in simulator"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdzofs", "cnc_wrzofs", "cnc_rdposition"],
        distinguishing_signals=["工件坐标偏置变化", "同一机床位置下工件坐标变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="tool_offset_setting",
        name="刀具补偿/刀具偏置流量",
        goal="生成刀具长度补偿、刀具径补偿、刀具偏置读写相关流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["G43/G44/G49 or G40-G42 use", "tool offset table changes in simulator"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdtofs", "cnc_wrtofs", "cnc_rdposition"],
        distinguishing_signals=["刀具偏置值变化", "补偿模式变化", "加工路径差异"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="parameter_read",
        name="参数读取流量",
        goal="生成 CNC 参数读取和参数范围查询相关流量。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["not required"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdparam", "cnc_rdparainfo"],
        distinguishing_signals=["参数号和数据类型变化", "读取范围变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="parameter_write_simulated",
        name="参数写入仿真流量",
        goal="在仿真器中生成安全参数写入、保护错误或模式错误相关流量。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["not required"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdparam", "cnc_wrparam"],
        distinguishing_signals=["参数写入请求", "写后读回", "保护或模式错误返回"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="alarm_query",
        name="报警查询流量",
        goal="生成当前报警、报警消息、报警状态相关查询流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["optional alarm-triggering simulator state"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_alarm2", "cnc_rdalmmsg", "cnc_rdalminfo"],
        distinguishing_signals=["报警状态变化", "报警号和报警消息变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="diagnostic_query",
        name="诊断数据查询流量",
        goal="生成诊断号、故障诊断画面、维护状态相关流量。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["optional diagnostic-triggering simulator state"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_diagnoss", "cnc_diagnosr", "cnc_rddiaginfo"],
        distinguishing_signals=["诊断号变化", "诊断值变化", "故障状态变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="pmc_signal_read",
        name="PMC/DI/DO 信号读取流量",
        goal="生成 PMC 地址、DI/DO 信号、I/O 状态读取相关流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["optional simulator signal changes"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["pmc_rdpmcrng", "pmc_wrpmcrng", "pmc_getdtailerr"],
        distinguishing_signals=["PMC 地址范围变化", "DI/DO 信号位变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="macro_variable_read_write",
        name="宏变量读写流量",
        goal="生成用户宏变量、系统变量读写和调用相关流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["macro variable use", "G65 or macro call if simulator supports it"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_rdmacro", "cnc_wrmacro", "cnc_rdmacroinfo"],
        distinguishing_signals=["宏变量号变化", "宏变量值变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="operation_history_query",
        name="操作历史/报警历史查询流量",
        goal="生成操作历史、报警历史、运行记录相关查询流量。",
        traffic_value=["high_coverage", "high_quality"],
        typical_nc_program=["optional prior operations to create history"],
        operation_phases=["before", "after"],
        recommended_api_functions=["cnc_rdophistry", "cnc_rdalmhistry"],
        distinguishing_signals=["历史记录条目变化", "报警历史条目变化"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="ethernet_connection",
        name="以太网连接/断开/异常流量",
        goal="生成 FOCAS 连接建立、释放、超时、Socket 错误相关流量。",
        traffic_value=["high_coverage", "high_distinguishability", "high_quality"],
        typical_nc_program=["not required"],
        operation_phases=["before", "after"],
        recommended_api_functions=["cnc_allclibhndl3", "cnc_freelibhndl", "cnc_getdtailerr"],
        distinguishing_signals=["连接建立", "连接释放", "EW_SOCKET 或超时错误"],
        allowed_rule_types=list(RULE_TYPES),
    ),
    ScenarioDefinition(
        scenario_id="abnormal_invalid_request",
        name="非法请求/异常流量",
        goal="在仿真器中生成非法参数号、非法模式、非法句柄、越界地址等异常流量。",
        traffic_value=["high_coverage", "high_distinguishability"],
        typical_nc_program=["not required or optional safe simulator state"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_getdtailerr", "pmc_getdtailerr"],
        distinguishing_signals=["EW_NUMBER/EW_LENGTH/EW_ATTRIB/EW_DATA/EW_HANDLE/EW_MODE 等错误"],
        allowed_rule_types=["safety_rule", "collection_rule", "operation_rule"],
    ),
    ScenarioDefinition(
        scenario_id="general_status_collection",
        name="通用状态采集流量",
        goal="生成不依赖特定加工动作的 CNC 状态、模式、基础数据采集流量。",
        traffic_value=["high_coverage"],
        typical_nc_program=["not required"],
        operation_phases=["before", "during", "after"],
        recommended_api_functions=["cnc_statinfo", "cnc_sysinfo", "cnc_rdposition"],
        distinguishing_signals=["运行模式", "系统信息", "基础状态"],
        allowed_rule_types=["collection_rule", "operation_rule", "safety_rule"],
    ),
]


RULE_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "rules": [
        {
            "rule_type": "nc_rule | operation_rule | collection_rule | safety_rule",
            "scenario": "one scenario_id from the taxonomy",
            "traffic_value": [],
            "applicable_environment": ["simulator"],
            "rule_text": "",
            "nc_program_requirements": [],
            "operation_sequence": [],
            "collection_timing": [],
            "recommended_api_functions": [],
            "allowed_operations": [],
            "restricted_operations": [],
            "abnormal_traffic_allowed": [],
            "distinguishing_signals": [],
            "quality_checks": [],
        }
    ]
}


def taxonomy_dict() -> dict[str, Any]:
    return {
        "protocol": "focas",
        "rule_types": RULE_TYPES,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
        "rule_extraction_schema": RULE_EXTRACTION_JSON_SCHEMA,
    }


def write_taxonomy(path: Path) -> None:
    write_json(path, taxonomy_dict())


def build_rule_extraction_prompt() -> str:
    scenario_lines = "\n".join(
        f"- {scenario.scenario_id}: {scenario.name}。{scenario.goal}" for scenario in SCENARIOS
    )
    rule_type_lines = "\n".join(f"- {name}: {description}" for name, description in RULE_TYPES.items())

    return f"""你是数控机床流量生成规则抽取专家。

目标：
从给定的 FANUC 0i-MF 手册片段中，抽取对生成高覆盖、高区分度、高质量 CNC/FOCAS 流量有用的规则。

只允许输出四类规则：
{rule_type_lines}

scenario 必须从以下列表选择，不允许自造场景名：
{scenario_lines}

不要抽取：
- 目录、前言、版权、出口限制、说明书改版履历。
- 泛泛安全提醒，除非它直接约束仿真器中允许/限制的流量生成行为。
- 与流量生成无关的纯说明。
- 无具体场景、状态、参数、信号、NC 程序、API 采集或异常含义的内容。

不要输出以下来源字段：
- source_file
- source_chunk_id
- page_start
- page_end
- section_title
这些字段由程序根据输入 chunk 自动补充。

只输出合法 JSON，不要 Markdown，不要解释。格式必须是：
{{
  "rules": [
    {{
      "rule_type": "nc_rule | operation_rule | collection_rule | safety_rule",
      "scenario": "",
      "traffic_value": [],
      "applicable_environment": ["simulator"],
      "rule_text": "",
      "nc_program_requirements": [],
      "operation_sequence": [],
      "collection_timing": [],
      "recommended_api_functions": [],
      "allowed_operations": [],
      "restricted_operations": [],
      "abnormal_traffic_allowed": [],
      "distinguishing_signals": [],
      "quality_checks": []
    }}
  ]
}}

字段要求：
- 如果片段没有有用规则，输出 {{"rules": []}}。
- rule_type 只能是 nc_rule、operation_rule、collection_rule、safety_rule。
- scenario 必须从上面的场景列表选择；无法归类时使用 general_status_collection。
- recommended_api_functions 只能填写明确相关的 FOCAS 函数名；不确定则留空。
- rule_text 用中文，简洁说明这条规则如何帮助流量生成。
- 不要编造手册片段没有支持的事实。

输入片段格式：
CHUNK_ID: ...
SOURCE_FILE: ...
PAGES: ...
SECTION: ...
TEXT:
...
"""


def write_rule_extraction_prompt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_rule_extraction_prompt(), encoding="utf-8")


def default_taxonomy() -> ProtocolTaxonomy:
    return ProtocolTaxonomy(
        protocol="focas",
        rule_types=RULE_TYPES,
        scenarios=SCENARIOS,
        rule_extraction_schema=RULE_EXTRACTION_JSON_SCHEMA,
    )


def scenario_from_dict(data: dict[str, Any]) -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario_id=str(data.get("scenario_id", "")).strip(),
        name=str(data.get("name", "")).strip(),
        goal=str(data.get("goal", "")).strip(),
        traffic_value=string_list(data.get("traffic_value")),
        typical_nc_program=string_list(data.get("typical_nc_program")),
        operation_phases=string_list(data.get("operation_phases")),
        recommended_api_functions=string_list(data.get("recommended_api_functions")),
        distinguishing_signals=string_list(data.get("distinguishing_signals")),
        allowed_rule_types=string_list(data.get("allowed_rule_types")) or list(RULE_TYPES),
    )


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def load_taxonomy(path: Path | None = None) -> ProtocolTaxonomy:
    if path is None:
        return default_taxonomy()
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = [scenario_from_dict(row) for row in data.get("scenarios", [])]
    scenarios = [scenario for scenario in scenarios if scenario.scenario_id]
    if not scenarios:
        raise ValueError(f"{path} does not contain any scenarios")
    rule_types = data.get("rule_types") or RULE_TYPES
    if not isinstance(rule_types, dict) or not rule_types:
        raise ValueError(f"{path} rule_types must be a non-empty object")
    return ProtocolTaxonomy(
        protocol=str(data.get("protocol") or path.parent.name or "protocol"),
        rule_types={str(key): str(value) for key, value in rule_types.items()},
        scenarios=scenarios,
        rule_extraction_schema=data.get("rule_extraction_schema") or RULE_EXTRACTION_JSON_SCHEMA,
    )


def taxonomy_dict(taxonomy: ProtocolTaxonomy | None = None) -> dict[str, Any]:
    taxonomy = taxonomy or default_taxonomy()
    return {
        "protocol": taxonomy.protocol,
        "rule_types": taxonomy.rule_types,
        "scenarios": [asdict(scenario) for scenario in taxonomy.scenarios],
        "rule_extraction_schema": taxonomy.rule_extraction_schema,
    }


def write_taxonomy(path: Path, taxonomy: ProtocolTaxonomy | None = None) -> None:
    write_json(path, taxonomy_dict(taxonomy))


def build_rule_extraction_prompt(taxonomy: ProtocolTaxonomy | None = None) -> str:
    taxonomy = taxonomy or default_taxonomy()
    scenario_lines = "\n".join(
        f"- {scenario.scenario_id}: {scenario.name}. {scenario.goal}"
        for scenario in taxonomy.scenarios
    )
    rule_type_lines = "\n".join(
        f"- {name}: {description}" for name, description in taxonomy.rule_types.items()
    )
    rule_type_choices = " | ".join(taxonomy.rule_types)
    fallback = taxonomy.fallback_scenario or "general_status_collection"

    return f"""你是工业控制协议流量生成规则抽取专家。
目标：从给定的 {taxonomy.protocol} 协议或设备资料片段中，抽取对生成高覆盖、高区分度、高质量工业协议流量有用的规则。

只允许输出以下规则类型：
{rule_type_lines}

scenario 必须从以下场景列表选择，不允许自造场景名：
{scenario_lines}

不要抽取目录、前言、版权、泛泛安全提醒、与流量生成无关的纯说明。
不要输出 source_file、source_chunk_id、page_start、page_end、section_title；这些字段由程序根据输入 chunk 自动补充。

只输出合法 JSON，不要 Markdown，不要解释。格式必须是：
{{
  "rules": [
    {{
      "rule_type": "{rule_type_choices}",
      "scenario": "",
      "traffic_value": [],
      "applicable_environment": ["simulator"],
      "rule_text": "",
      "nc_program_requirements": [],
      "operation_sequence": [],
      "collection_timing": [],
      "recommended_api_functions": [],
      "allowed_operations": [],
      "restricted_operations": [],
      "abnormal_traffic_allowed": [],
      "distinguishing_signals": [],
      "quality_checks": []
    }}
  ]
}}

字段要求：
- 如果片段没有有用规则，输出 {{"rules": []}}。
- rule_type 只能从上述 rule_types 选择。
- scenario 必须从上面的场景列表选择；无法归类时使用 {fallback}。
- recommended_api_functions 只填写资料中明确支持的函数、API、服务名或协议操作；不确定则留空。
- rule_text 用中文，简洁说明这条规则如何帮助流量生成。
- 不要编造资料片段没有支持的事实。

输入片段格式：
CHUNK_ID: ...
SOURCE_FILE: ...
PAGES: ...
SECTION: ...
TEXT:
...
"""


def write_rule_extraction_prompt(path: Path, taxonomy: ProtocolTaxonomy | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_rule_extraction_prompt(taxonomy), encoding="utf-8")
