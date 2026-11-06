# Smart Traffic Agent（SMPAgent）项目设计文档

> 文档版本：v1.0（以当前工作区代码和运行产物为基线）
> 编写日期：2026-11-06
> 项目目录：`E:\研二下\smart generation code`

## 1. 文档目的

本文档说明 Smart Traffic Agent（简称 SMPAgent）的目标、边界、模块职责、数据契约、运行流程、采集证据、错误处理和部署方式。文档同时区分两类内容：

1. **当前实现**：可以在代码、测试或运行产物中直接核对的行为。
2. **设计建议**：用于降低失败率、扩大协议覆盖或改善可维护性的后续方案。

当前工作区存在未提交修改和运行临时文件，因此本文档描述的是“当前工作区版本”，不能替代以后发布版本的变更记录。

## 2. 项目背景与目标

### 2.1 背景

CNC 协议流量采集通常需要同时处理以下对象：

- 自然语言任务、场景和目标 API；
- FANUC FOCAS SDK 的 C/C++ ABI、结构体、联合体、指针和数组；
- NCGuide 或真实 CNC 的连接、程序状态和 Cycle Start 操作；
- C++ 编译器、FOCAS DLL、Npcap 网卡和抓包文件；
- API 输入输出日志与 PCAP 请求/响应的调用级关联；
- 失败后的重试、重规划、代码修复和证据保留。

如果只让模型直接生成一份完整 C++ 程序，模型需要在同一次输出中同时解决协议知识、类型系统、运行时生命周期、Windows UI 自动化和日志格式，失败很难定位。因此 SMPAgent 将任务拆分为规划、代码生成、执行和质量判断四个角色，并把 SDK 类型校验、编译、运行和抓包结果纳入状态机。

### 2.2 目标

系统目标是把如下流程自动化：

```text
自然语言任务
  -> 任务路由
  -> 协议/场景规划
  -> 目标 API 与辅助 API 依赖展开
  -> C++/NC 产物生成
  -> 编译与设备执行
  -> API 日志、CSV、PCAP 和事务证据
  -> 质量评估
  -> 完成或按证据修复
```

主要目标包括：

- 采集可复现、可追溯的 CNC 协议数据集；
- 保证目标 API 确实被调用，并记录实际参数和返回值；
- 将成功执行与“存在有效网络流量证据”分开判断；
- 根据当前 SDK 头文件校验函数原型和结构体字段；
- 在失败时保存编译错误、运行错误、返回码、日志和中间产物；
- 支持 FANUC FOCAS，并为海德汉 LSV/2、三菱 EZSocket、西门子 S7Comm 提供可替换适配边界。

### 2.3 非目标

当前版本不把以下内容视为默认目标：

- 自动控制未经授权的真实机床加工动作；
- 把 TCP 建连成功伪装成 CNC API 调用成功；
- 在 SDK 没有原型、运行库没有导出或仿真器不支持时伪造返回值；
- 用空 PCAP 或手写 PCAP 头替代真实 Npcap 抓包；
- 把“规划生成完成”当作“数据采集完成”；
- 在没有实际错误证据时无限重复生成相同计划或相同二进制。

## 3. 系统边界与适用场景

### 3.1 输入边界

系统输入由 `TaskRequest` 表示，核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `description` | 用户自然语言任务 |
| `task_id` | 一次任务或批次的唯一标识 |
| `protocol` | 协议类型，默认 `cnc`，桌面入口使用 `focas`、`lsv2` 等 |
| `target_environment` | `simulator`、`ncguide-generated-cpp`、`ncguide-bridge`、`ncguide-bridge-readonly` 等 |
| `permissions` | 写操作、历史记忆、目标 API、设备范围等约束 |
| `quality_gate_enabled` | 是否启用动态输出质量门控 |

### 3.2 输出边界

一次运行通常产生以下证据：

```text
<output_dir>/
├─ summary.json
├─ summary_zh.txt
├─ plan.json 或 execution_plan.json
├─ generated/
│  ├─ api_script.*
│  ├─ program.nc
│  └─ *.inc / bindings.*
├─ execution/
│  ├─ api_logs.jsonl
│  ├─ capture_events.jsonl
│  ├─ tool_calls.jsonl
│  ├─ protocol_transactions.jsonl
│  ├─ traffic_relationship_feedback.json
│  └─ data/
│     ├─ focas_api_input.csv
│     ├─ focas_api_output.csv
│     ├─ focas_api_capture.pcap
│     └─ by_api/
└─ repair_history、诊断和临时编译文件
```

不同适配器的证据形式不同：FOCAS 重点是 C++ 运行日志、CSV 和 PCAP；LSV/2 重点是每次 pyLSV2 请求/响应的结构化日志；EZSocket 当前主要是 TCP 建连和可选 COM 只读调用结果。

## 4. 总体架构

```text
┌──────────────────────────────┐
│ CLI（main.py）/原生桌面（Tk） │
└──────────────┬───────────────┘
               │ TaskRequest + 配置
               v
┌────────────────────────────────────────────────────────┐
│ TrafficGenerationWorkflow（LangGraph）                 │
│ router -> planning -> code_generation -> execution      │
│    ^          ^              ^             │             │
│    └── repair_plan / repair_code / repair_execution ───┘
└───────┬───────────────┬───────────────┬────────────────┘
        v               v               v
  Knowledge/RAG   SDK/Contract      ToolRuntime
  Memory/Soul     Type Index        compile/run/capture
  TRCM/参数工具  Harness/Assembler  设备适配器
        └───────────────┬───────────────┘
                        v
             计划、代码、运行和证据文件
```

### 4.1 主要分层

| 层 | 主要模块 | 职责 |
| --- | --- | --- |
| 入口层 | `main.py`、`desktop.py`、`desktop_backend.py` | 接收任务、配置设备和模型、启动/停止工作流、展示进度 |
| 编排层 | `workflow.py`、`orchestration/` | 建立 LangGraph、维护状态、控制路由和修复次数 |
| Agent 层 | `agents/router.py`、`planner.py`、`code_generator.py`、`executor.py` | 路由、规划、生成和执行 |
| 知识层 | `knowledge.py`、`rag/`、`rag_indexes/` | 官方手册、函数清单、SDK 类型、协议文档和检索 |
| 工具层 | `agent_tools/`、`tool_runtime.py` | 编译、运行、采集、知识检索、记忆和关系图工具 |
| 适配层 | `integrations/`、`protocol_profiles.py` | FOCAS/NCGuide、LSV/2、EZSocket 等设备接口 |
| 证据层 | `api_log_templates.py`、`dataset_export.py`、事务匹配代码 | 记录 API、CSV、PCAP、调用关联和质量指标 |

## 5. 入口与用户操作流程

### 5.1 CLI 入口

CLI 入口在 `main.py`，用于无界面运行、探测环境、生成知识库和执行任务。典型操作包括：

```powershell
python -m smart_traffic_agent init-knowledge --out examples/knowledge.json
python -m smart_traffic_agent run --knowledge examples/knowledge.json --out runs/demo
python -m smart_traffic_agent probe-ncguide --install-dir "D:\Program Files (x86)\FANUC\NCGuide FS0i-F"
python -m smart_traffic_agent build-ezsocket-knowledge
python -m smart_traffic_agent test-ezsocket --host 192.168.139.130 --port 683 --api Status_GetRunStatus
```

CLI 可以配置协议、目标环境、LLM provider、FOCAS 头文件目录、FOCAS 端口、Npcap 网卡、NCGuide 窗口和 Cycle Start 坐标等参数。

### 5.2 原生桌面入口

桌面入口由 `desktop.py` 提供，使用 Tk 原生窗口，不依赖浏览器。主界面包含：

1. **连接配置**：协议、设备地址、端口、SDK 目录、DLL 目录、Npcap 网卡、模型和输出目录。
2. **采集模式与提示词**：单个 API、场景或全部 API，分别维护场景提示词和全 API 提示词。
3. **采集监控**：工作流事件、工具调用、进度、文件数量、耗时和错误。
4. **流量关系图**：显示 API 节点、依赖关系和执行证据。

启动前会重新检查连接、模型、编译器、SDK、Npcap 设备和输出目录。需要运行 NC 程序的场景必须完成 Cycle Start 定位；定位遮罩会拦截校准点击，避免校准动作本身启动程序。

### 5.3 协议入口状态

| 协议 | 当前状态 | 当前行为 |
| --- | --- | --- |
| FANUC FOCAS | 已接入真实采集 | NCGuide/FOCAS DLL、C++、Npcap、CSV、PCAP |
| 海德汉 LSV/2 | 已接入只读采集 | TNC640 默认 TCP 19000，pyLSV2 安全模式和只读白名单 |
| 三菱 EZSocket | 可连接测试 | 默认 TCP 683，可选 COM 只读调用；完整采集未开放 |
| 西门子 S7Comm | 配置预览 | 当前没有可执行连接和采集适配器 |

## 6. LangGraph 工作流

### 6.1 状态流转

`TrafficGenerationWorkflow` 在 `workflow.py` 中构建图：

```text
planning（初始由 router 决定）
      ↓
code_generation
      ↓
execution
      ↓
router 判断质量和错误
   ├─ complete
   ├─ repair_plan -> planning
   ├─ repair_code -> code_generation
   └─ repair_execution -> execution
```

实际图的入口是 `router`。Router 根据当前 `WorkflowState` 判断下一节点，规划、生成和执行节点完成后都会回到 Router。默认最大修复次数为 3，可通过构造参数调整。

### 6.2 路由优先级

Router 先使用确定性规则，再在存在多个合法转移时调用模型：

- 生成 C++ 或编译失败：优先 `repair_code`；
- SDK 类型不匹配：优先依据头文件类型索引修代码；
- 抓包工具失败：优先 `repair_execution`；
- 目标 API 没有实际调用：修复目标调用路径；
- 重复目标失败：保留失败证据并重新规划独立目标；
- 运动类目标缺乏运行/位置/进给变化：重新规划场景和采样策略；
- 已经满足终止证据：进入 `complete`；
- LLM 瞬时错误且无可安全修复证据：结束并保留错误。

Router 的设计原则是把“代码缺陷”“运行环境缺陷”“计划缺陷”和“离线审计告警”分开。例如事务匹配失败不自动触发代码修复，NCGuide UI 点击失败不被误判为采样计划失败。

### 6.3 工作流状态持久化

`WorkflowState` 包含：

- 请求、当前阶段和检索片段；
- `ExecutionPlan`、`GeneratedArtifacts` 和 `ExecutionResult`；
- 质量评估、映射记录和错误列表；
- 修复次数、修复历史、长期记忆和工具调用历史；
- `ExecutionContext` 中按 API 保存的输入、输出、状态码、调用 ID 和证据。

工作流结束时写入 `summary.json` 和中文摘要，并尝试更新长期错误记忆、Agent Soul 经验和函数覆盖状态。

## 7. Agent 职责设计

### 7.1 RouterAgent

输入是完整工作流状态，输出是一个合法的下一阶段和修复指令。Router 不生成 C++，也不直接执行设备调用。它负责：

- 判断当前状态可到达的阶段；
- 读取近期失败 API 日志、返回码说明、质量指标和运行证据；
- 识别失败属于计划、代码、执行还是可接受的局部告警；
- 防止成功后再次进入无意义的修复循环。

### 7.2 PlanningAgent

PlanningAgent 将自然语言任务和检索资料转成 `ExecutionPlan`。它负责：

- 确定场景目标、目标 API 和辅助 API；
- 按 `before/during/after` 阶段拆分 `PlanStep`；
- 建立参数来源、依赖、输入、输出和失败策略；
- 规划 NC 程序需求、程序生命周期和 Cycle Start 门控；
- 使用 FOCAS 函数清单、SDK 合同、前置条件、TRCM 和历史记忆；
- 对全 API 模式维护覆盖批次，而不是把一批 API 宣称为全覆盖。

### 7.3 CodeGenerationAgent

CodeGenerationAgent 将计划转换为 C++ API 脚本、NC 程序和必要的片段文件。当前生产路径包含：

- 从当前 `Fwlib32.h` 建立或加载 SDK 类型索引；
- 生成结构化 `CodeSpec` 和 API 合同；
- 对 FOCAS 使用固定 Harness 和本地确定性组装；
- 运行 C++ 结构体成员访问、函数参数和输出字段校验；
- 对需要模型决策的部分使用严格 JSON 契约；
- 生成后进行静态验证，失败时把诊断交给 Router。

模型审查可以提供建议，但对于真实 NCGuide 采集，类型合同、Harness 和编译器才拥有最终执行准入权。

### 7.4 ExecutionAgent

ExecutionAgent 负责：

- 编译生成的 C++；
- 设置 `FOCAS_DLL_DIR`、`FOCAS_HOST`、`FOCAS_PORT`、`PCAP_NETWORK_DEVICE` 等环境；
- 运行 FOCAS/NCGuide 或桥接器；
- 收集 CSV、PCAP、API 日志、工具调用和协议事务；
- 将执行结果写入 `ExecutionResult` 和 `ExecutionContext`；
- 把成功运行证据反馈给 Traffic Relationship Graph；
- 将错误附加到长期记忆，供后续路由和规划检索。

## 8. 核心数据模型

### 8.1 `TaskRequest`

表示用户请求和执行边界。`permissions` 不只是权限开关，也承载目标 API、设备范围、是否启用历史记忆、代码只评估等运行约束。

### 8.2 `PlanStep`

每个步骤包括：

- `step_id`、`phase`、`action`、`interface_name`；
- `parameters`、`repeat`、`interval_seconds`；
- `api`、`protocol_function`、`role`；
- `parameter_sources`、`dependencies`、`required_inputs`、`outputs`；
- `failure_policy` 和 `blocks_dependents`。

`role` 用于区分目标 API 和辅助 API。辅助 API 可以负责连接、状态读取、程序上传、句柄释放或参数发现，但不能替代目标 API 的成功调用。

### 8.3 `ExecutionPlan`

包含计划 ID、任务 ID、场景目标、目标环境、NC 程序规格、步骤、预期输出、检索片段、运行时绑定、前置条件、会话和采集配置。`schema_version=2` 表示当前结构化交接格式。

### 8.4 `GeneratedArtifacts`

保存 API 脚本、NC 程序及其路径，另外保存诊断、Harness、片段和绑定文件路径。生成物路径会被复制到独立执行目录，便于保留完整复现材料。

### 8.5 `ApiCallLog` 与 `CaptureEvent`

`ApiCallLog` 记录 API 级输入、返回码、输出、调用 ID、事务 ID、时间戳、设备上下文和 `traffic_status`。`CaptureEvent` 记录请求/响应方向、端点和摘要。两者分别服务于语义日志和运行时事件追踪。

### 8.6 `ExecutionResult`

保存任务是否成功、API 日志、捕获事件、输出目录、错误、工具调用、采集状态和证据。`success` 只描述执行层结果；最终是否可以完成，还要结合目标调用、程序完成门控和有效流量证据。

## 9. 知识、检索和记忆

### 9.1 RAG 知识库

知识库按协议和用途拆分：

- `rag_indexes/focas/`：FOCAS 手册分块、函数清单、返回码、场景模板和 SDK 索引；
- `rag_indexes/ezsocket/`：FCSB1224W000 手册与 `MitCom.cs` 生成的 API 条目和安全规则；
- `rag_indexes/heidenhain/`：LSV/2 相关文档和 API 目录。

检索结果以 `RetrievedChunk` 表示，包含知识块和相关性分数。Planner 和 CodeGenerator 使用检索结果建立计划和 ABI 证据，Router 使用检索结果解释失败。

### 9.2 SDK 类型索引

`sdk_type_index.py` 从当前 `Fwlib32.h` 解析：

- 结构体和联合体名、字段类型、数组维度；
- FOCAS 函数原型；
- 头文件 SHA-256，用于判断缓存是否仍对应当前 SDK；
- 结构体成员访问是否合法。

类型索引是验证辅助，不替代 C/C++ 编译器。头文件缺失或型号无法唯一识别时，桌面层显示“无法唯一识别”，不会猜测型号。

### 9.3 Traffic Relationship Graph（TRCM）

TRCM 用 API 节点和证据限定的关系表示调用依赖、前置步骤和输出到输入的绑定。规划阶段读取关系，执行成功后写入 `traffic_relationship_feedback.json`。未验证的 `proposed` 关系不能自动成为强制依赖；没有 Neo4j 配置时，系统保留 RAG/SDK 路径并把关系状态标为未知或不可用。

### 9.4 Agent Soul 与长期记忆

四个 Agent 分别从 `agent_souls/router`、`planner`、`code_generator`、`executor` 读取角色经验。长期记忆由 `LongTermMemoryStore` 保存错误及修复结果，按任务和 Agent 范围检索。记忆缺失不能直接变成执行规则；所有经验必须结合当前 SDK、当前环境和当前错误证据。

## 10. API 执行合同

### 10.1 目标 API 与辅助 API

计划必须明确目标 API。辅助 API 只用于满足前置条件或提供证据，例如：

| 类型 | 示例 | 作用 |
| --- | --- | --- |
| 会话辅助 | `cnc_allclibhndl3`、`cnc_freelibhndl` | 建立和释放 FOCAS 句柄 |
| 状态辅助 | `cnc_statinfo`、`cnc_rdprgnum` | 读取设备状态、程序号和完成条件 |
| 程序辅助 | `cnc_dwnstart4`、`cnc_dwnend4`、`cnc_search` | 上传、选择和结束程序操作 |
| 目标调用 | 用户选定的 `cnc_alarm2`、`cnc_wrrelpos` 等 | 产生需要采集的数据和流量 |

只有目标调用真实进入 `ctx.call`，并在 API 日志中留下实际结果，目标才算被调用。连接成功、辅助 API 成功或模型输出了 API 名称都不能替代该条件。

### 10.2 参数、输出和运行时绑定

计划中的参数来源可以是固定值、SDK 规则、设备发现、前一 API 输出或用户配置。执行时通过 `ExecutionContext` 保存实际值，后续步骤只能使用已返回且有证据的值，不能为失败或缺失的 API 虚构绑定值。

### 10.3 返回码语义

返回码解释以当前 SDK 和 FOCAS 返回码索引为准。单个数值不能在没有上下文时被直接解释为唯一根因。例如 `EW_DATA` 不自动等同于“程序号冲突”；需要结合目录、详细错误和目标 API 的实际日志判断。

## 11. C++ 代码生成和固定 Harness

### 11.1 生成策略

当前生产模式不把“完整可执行 C++”全部交给模型自由书写。模型交接的核心数据类似：

```json
{
  "api_name": "cnc_wrrelpos",
  "call_expr": "api.cnc_wrrelpos(handle, 8, &__out_wrrelpos)",
  "parameter_values": {"length": 8},
  "struct_values": {
    "wrrelpos": {"datano": 0, "type": 1, "data": [12]}
  }
}
```

模型应提供裸 API 调用、正式参数表达式和结构体字段值；本地组装器负责把它们放进固定 Harness。

### 11.2 固定 Harness 的职责

Harness 统一处理：

- 环境变量读取和 DLL 动态加载；
- `GetProcAddress` 解析 FOCAS 函数；
- 句柄申请、释放和超时；
- Npcap 初始化、按调用窗口抓包和资源清理；
- API 输入/输出 CSV；
- `ctx.call` 调用包装、调用 ID 和时间戳；
- NCGuide 程序上传、选择、Cycle Start 和完成门控；
- 返回码和运行错误输出。

### 11.3 结构体/联合体传递机制

结构体是当前代码生成中最容易出错的部分之一。约束如下：

1. `parameter_values` 只能保存正式函数参数，不把结构体字段塞进普通参数字典。
2. `struct_values` 保存结构体或联合体实例的字段值。
3. 字段名、字段类型、数组维度以当前 `Fwlib32.h` 和 SDK 类型索引为权威。
4. 组装器声明结构体变量、填充字段、生成输出序列化规则，并组装 `ctx.call`。
5. 对联合体只能访问当前声明中存在且有证据的成员。
6. 外层数组和元素字段必须按 SDK 声明区分，不能根据字段名猜测维度。
7. 输出字段规则缺失时必须停止组装并报告错误，不能生成“看起来能编译”的占位代码。

以 `cnc_wrrelpos` 为例，历史错误把 `POSELM.data` 当成数组；实际 SDK 中 `POSELM.data` 是标量，数组维度在 `ODBPOS` 外层。该类问题需要由 SDK 索引和确定性组装器拦截。

### 11.4 生成前后校验

生成前检查：

- 目标 API 是否在当前 SDK 中存在；
- ABI 参数数量、方向和类型是否匹配；
- 结构体字段、数组维度和联合体成员是否合法；
- 目标 API 是否具有确定性输出字段规则；
- 计划中的前置条件是否能由 Harness 执行。

生成后检查：

- C++ 是否包含动态 DLL 加载和必要环境变量；
- 是否包含 Npcap 真抓包路径；
- 是否生成目标 API 的真实 `ctx.call`；
- CSV 字段是否符合 `api_log_templates.py`；
- NCGuide UI 坐标是否来自校准变量，而不是硬编码；
- 编译器是否能通过实际 MSVC x86 编译。

## 12. FOCAS、NCGuide 和真实采集流程

### 12.1 连接和运行时

FOCAS C++ 路径使用 `FOCAS_DLL_DIR` 定位 `Fwlib32.dll`，通过宽字符 Windows API 加载 DLL，并使用 `FOCAS_HOST`、`FOCAS_PORT` 建立连接。默认配置来自桌面后端和环境变量；当前 NCGuide 配置常见端口为 8192，生成 Harness 仍支持通过环境变量覆盖。

### 12.2 Npcap 抓包

Npcap 是真实采集的硬门控：

- 从 `PCAP_NETWORK_DEVICE` 读取网卡；
- 初始化失败时写入 `PCAP_CAPTURE_FAILED` 并在调用 FOCAS API 前退出；
- 成功时记录 `pcap_capture_enabled=true` 或 `pcap_capture_started=true`；
- 在 API 调用前后采集包并刷新文件；
- 不创建只有 PCAP 头、没有真实数据的伪文件。

### 12.3 API 日志和事务关联

每次调用生成输入行和输出行，共享 `index` 与 `invocation_id`。离线事务关联根据调用时间窗、TCP 流和请求/响应顺序，把 API 日志与 PCAP payload 关联。`traffic_status=matched` 表示调用有可靠的请求/响应流量证据；没有匹配流量的 API 日志仍可保留，但不能计入真实流量覆盖。

### 12.4 NC 程序生命周期

需要运动或程序执行的场景通常遵循：

```text
读取状态/程序目录
  -> 生成或上传临时 O 号程序
  -> 选择程序
  -> 通过校准的 Cycle Start 坐标触发单段执行
  -> 读取运行状态、位置和进给
  -> 等待程序完成门控
  -> 读取最终状态
  -> 清理临时程序并释放句柄
```

固定 Harness 使用单段执行约束，避免在前一段仍运行时盲目再次点击 Cycle Start。Cycle Start 定位失败、窗口不在前台或坐标模式无效属于执行/UI 问题，应由 `repair_execution` 或操作员修复。

## 13. 质量、覆盖率和完成判定

### 13.1 质量维度

系统同时记录以下指标：

- 目标 API 调用率；
- API 返回成功率和预期错误比例；
- 参数覆盖和边界值覆盖；
- 输出字段是否实际变化；
- 程序是否完成；
- 运动类 API 的位置、进给、运行状态变化；
- PCAP 请求/响应匹配率；
- API 日志与事务的调用关联完整性；
- 生成、编译、运行和抓包失败原因。

### 13.2 运动与非运动 API 的差异

位置、进给、主轴运动等 API 需要程序完成和状态/位置变化证据。目录、文件、参数、刀具、PMC、报警等 API 不应被强制套用运动门控；它们应根据目标 API 自身的返回字段、参数变化和 PCAP 证据判断。

### 13.3 多样性的定义与审计合同

本项目中的“多样性”定义为：**在同一个目标 API 的语义有效域内，经过真实成功调用并由日志或 PCAP 证据确认的、相互不同的输入配置、返回结果、控制器状态窗口或应用层报文样本的数量与覆盖范围。**

多样性不等于调用次数，也不等于把同一个值换一种字符串格式。重复调用、只改变句柄或指针地址、只改变缓冲区长度/容量、只改变字段排列方式，以及失败调用，都不能自动产生新的语义多样性。

#### 13.3.1 四类多样性

| 类型 | 定义 | 适用条件 | 主要证据 |
| --- | --- | --- | --- |
| 输入多样性（`input_diversity`） | 目标 API 的文档化业务输入维度形成不同的规范化组合 | API 存在安全且有文档依据的输入域 | 成功调用、完整正式参数、匹配 PCAP；`eligible_distinct_configurations` |
| 响应多样性（`response_diversity`） | 成功调用返回的有效字段或数据载荷形成不同的规范化值 | API 的返回内容本身具有可观察变化要求 | `output`/`data` 字段、响应状态变化、响应哈希 |
| 状态多样性（`state_diversity`） | 控制器在不同真实运行/运动/状态窗口产生可区分的观测 | 目标依赖 `runtime_motion_state` 或明确的运行状态场景 | `cnc_statinfo`、目标 API 前/中/后采样、程序完成和状态窗口 |
| 应用流量多样性（`application_traffic`） | FOCAS 应用层请求/响应 payload 的不同 SHA-256 样本 | 有可重组的 TCP/PCAP 证据 | `distinct_request_payloads`、`distinct_response_payloads` |

对于下载、上传等会话型 API，还要单独记录会话多样性：请求段、响应段和双向混合会话分别统计，不能把一个完整会话的辅助开始/结束调用误计成目标 API 的多个输入样本。

#### 13.3.2 API 多样性分类

Planner 根据 `scenario_capabilities.py` 的 `classify_api_variation` 生成变异合同，目标 API 分为：

| `variation_class` | 规则 |
| --- | --- |
| `call_only` | 只要求一次成功且有匹配流量的目标调用；不强行改变输入或设备状态。典型如连接、状态读取、报警读取。 |
| `input_only` | 要求不同的文档化输入配置；响应值可以保持不变。典型如按轴号或选择器读取。 |
| `response_state_only` / `response_observation_only` | 重点记录返回值或观察到的设备状态；不虚构输入变化，也不强制写入设备。 |
| `input_and_response` | 同时要求合法输入组合和返回/状态变化；两项分别审计，不能用一项替代另一项。 |

`applicable` 表示某类多样性在语义上适用，`required` 表示本次任务必须满足。二者分开保存，避免把“可以观察”误判成“必须改变”，或把一个没有安全变化域的 API 强行扩展成参数扫描。

#### 13.3.3 输入组合的规范化和计数

输入组合由 API 合同中的 `dimensions` 决定，使用 `parameter_coverage.input_values()` 提取正式输入，再按稳定键排序序列化为签名。计数规则为：

1. 只纳入文档化业务输入；`FlibHndl`、句柄、地址、端口、超时等传输上下文不属于语义维度。
2. 输出指针、结构体地址和缓冲区指针不属于输入维度。
3. `length`、`size`、`capacity` 等形状字段通常不单独创造多样性；只有 API 合同明确要求时才参与组合。动态结构体长度必须与当前 SDK 的实际字节数一致。
4. 数值按规范化值比较，例如 `1` 与 `1.0` 不形成两个样本；字典/数组按稳定 JSON 表示比较。
5. 一组耦合参数必须作为整体 case 比较，不能把不同 case 的边际字段拼成一个“覆盖完成”的组合。
6. 只有状态码为 0、没有错误、正式输入完整且拥有有效流量证据的调用，才进入 `eligible_distinct_configurations`。
7. 恢复原值、验证选择和运动证据等辅助步骤会从目标 API 多样性计数中排除；恢复动作不能制造多样性。

参数合同通过 `minimum_distinct`、`expected_values`、`cases` 和 `ranges` 表达目标：枚举域应覆盖计划值，范围域至少检查边界、典型或中间值，耦合 case 必须完整匹配。合法域不超过 20 个配置时默认枚举完整域；超过 20 个时使用有文档依据的有界子集，实际预算受 `diversity_configuration_budget`、`capture_samples` 和任务质量目标共同限制。固定输入只调用一次，不作为人工制造的变化维度。

#### 13.3.4 响应、状态和流量的有效性

- 响应多样性只使用成功调用的实际返回字段或 `data`，失败调用的输出缓冲区不能制造变化。
- 状态多样性必须来自真实控制器状态窗口。延时、程序文件存在或计划中写了“运行中”都不能代替 `cnc_statinfo` 和目标采样证据。
- 运动质量门控要求程序完成，并至少有 5 个进给样本、5 个位置样本、进给和位置各自至少 2 个唯一值、至少一个运行或运动激活样本，以及至少 3 个发生变化的输出参数。该门控只用于实际需要 NC 运动的批次。
- 应用层流量多样性按 payload 二进制内容计算 SHA-256；十六进制格式差异不算变化。普通 API 要求 `traffic_status=matched`；会话型 API 必须满足 `valid_session_segment_evidence`，包括 API/调用 ID 对齐、开始/结束返回码均为 0、PCAP TCP 重组、流量方向匹配、会话完整结束和调用关联确认。

#### 13.3.5 多样性审计输出

`diversity_audit.py` 生成离线、证据驱动的审计结果，版本为 `1`，每个函数至少包含：

- `variation_class`；
- `input_diversity`：适用性、是否必需、维度、尝试/有效/历史/新增配置数量和预算；
- `response_diversity`：适用性、是否必需、有效响应值数量和哈希；
- `application_traffic`：请求/响应 payload 数量、历史数量、新增数量和哈希；
- `sessions`：完整会话数量及请求段、响应段、混合段的变体数量；
- `calls`：实际目标调用数和成功调用数；
- `incremental_recommendation`：继续尝试未见文档输入、观察下一状态、记录环境限制或转向下一个目标。

该审计是离线证据和下一轮规划的依据，不会因为参数或流量多样性不足而篡改已经成功的执行结果；它也不能把历史审计当作当前运行证据。只有目标 API 调用、成功条件、PCAP/会话证据和适用的多样性要求同时满足时，才可以在覆盖状态中标记相应目标为完成。

### 13.4 全 API 覆盖

“全部 API”模式使用函数清单和覆盖状态文件分批执行。一次批次只代表一次批次，不能因为计划包含很多 API 就宣称全覆盖。已经有可靠 PCAP 请求/响应记录的 API 可以跳过，失败目标保留证据并在后续批次重试或标记阻塞。

## 14. 错误分类与修复策略

| 错误类别 | 典型证据 | 处理阶段 |
| --- | --- | --- |
| 计划错误 | 依赖缺失、目标 API 不适用、运动场景没有有效 NC 规格 | `repair_plan` |
| 生成/类型错误 | `SDK_TYPE_MISMATCH`、结构体字段不存在、输出规则无法解析 | `repair_code` |
| 编译错误 | MSVC 找不到类型、参数数目不符、未解析符号 | `repair_code` |
| 运行/ABI 错误 | 访问冲突、栈溢出、DLL 导出缺失、超时 | `repair_code` 或 `repair_execution` |
| NCGuide/UI 错误 | 窗口不在前台、坐标无效、Cycle Start 失败 | `repair_execution` |
| 抓包错误 | Npcap 设备不存在、`PCAP_CAPTURE_FAILED` | `repair_execution` |
| 目标调用缺失 | `TARGET_API_NOT_CALLED`、无目标调用日志 | `repair_code` |
| 设备前置条件错误 | 选件、权限、程序状态或参数前置条件不满足 | 证据化后 `repair_plan` |
| 离线审计告警 | 事务未匹配、参数覆盖不足 | 保留诊断，通常不重写代码 |

修复必须保留原始生成物、编译输出、运行日志和修复历史。相同错误重复出现时，不应无限重试未改变的计划或二进制。

## 15. 已观察到的运行证据与问题分析

以下结论来自当前工作区 `runs/desktop` 中的实际 `summary.json` 和相关日志，不是根据代码外推的结果。

### 15.1 `cnc_alarm2`：组装阶段失败后修复成功

`runs/desktop/20261106_170742_d4a4e8/batch_001/summary.json` 显示：

```text
SDK_CALL_ASSEMBLY_UNRESOLVED: cnc_alarm2;
no deterministic output field rule
```

该批次没有生成可执行 C++，`api_log_count=0`，没有真实 FOCAS 调用。随后 `runs/desktop/20261106_181903_6ce505/batch_001/summary.json` 显示同类快速路径修复后成功：0 次修复、9 次 API 调用、`cnc_alarm2` 返回 `EW_OK`、输出 `alarm=0`，并生成 PCAP 和 API CSV。

这说明该失败来自本地组装规则缺口，而不是设备返回码或抓包失败。

### 15.2 `cnc_wrrelpos`：结构体和运行时符号问题

`runs/desktop/20261106_155735_083250/batch_001/summary.json` 记录了两次代码修复后成功，但过程中出现：

```text
SDK_RUNTIME_SYMBOL_UNRESOLVED: symbols=['ODBAXIS']
SDK_RUNTIME_SYMBOL_UNRESOLVED: symbols=['__out_relative']
```

`runs/desktop/20261104_221541_716159/batch_001/summary.json` 还记录了 `POSELM.data` 被错误当作数组的问题。`runs/desktop/20261105_223303_734222/batch_001/summary.json` 在通用结构体组装器实现前报：

```text
DETERMINISTIC_ASSEMBLY_UNSUPPORTED
```

这些记录直接表明：结构体/联合体、输出变量和固定 Harness 之间的边界是当前 C++ 生成成功率的重要影响因素。

### 15.3 `cnc_rdparam`：虚构类型导致编译失败

`runs/desktop/20261105_103047_8b9018/batch_001/summary.json` 显示生成了不存在的 `ODBPARAINFO`，MSVC 编译失败。该问题属于 SDK 类型证据未被正确约束，而不是“调用太复杂”本身。

### 15.4 复杂度风险的可证据化结论

历史成功记录中一次模型请求同时携带完整执行计划、多个 API、ABI、结构体、RAG、关系图、历史示例和运行时变量；某次记录包含 10 个计划步骤、350 个检索片段和 8 个 API fragment。这个输入规模会增加模型在类型、字段和调用边界上的决策负担，但当前证据不能把所有失败都归因于复杂度：

- `cnc_alarm2` 的失败是本地确定性输出规则缺失；
- `cnc_rdparam` 的失败是 SDK 类型错误；
- `cnc_wrrelpos` 同时涉及结构体组装和运行时符号；
- 另有失败来自前置条件、仿真器选件、程序生命周期和质量门控。

因此，当前最合理的设计结论是：**结构体处理和过大的生成上下文都是风险放大器，但需要按错误证据分类，不能用“步骤复杂”解释全部失败。**

## 16. 多协议适配器设计

### 16.1 统一接口

协议配置由 `protocol_profiles.py` 管理，适配器至少应提供：

- 连接测试；
- 安全能力声明；
- 目标 API/命令目录；
- 参数和返回值规范化；
- 结构化日志；
- 采集证据和错误映射。

### 16.2 FOCAS

FOCAS 是当前最完整的适配器，拥有 SDK 头文件、DLL、动态符号解析、C++ Harness、NCGuide UI、Npcap、CSV、PCAP 和事务匹配。

### 16.3 海德汉 LSV/2

LSV/2 适配器位于 `integrations/heidenhain_lsv2.py`，使用 pyLSV2 和只读白名单。连接测试读取 `TNC:` 目录信息；采集时每个 API 默认调用一次，保存请求/响应结构化日志，不启动 FOCAS C++ 后端，也不生成 FOCAS PCAP。需要 DNC、PLCDEBUG、DATA、INSPECT 或 SCOPE 权限的调用会单独记录失败原因。

### 16.4 三菱 EZSocket

EZSocket 适配器位于 `integrations/ezsocket.py`。当前支持 TCP/683 建连探测和可选 COM 只读调用。真实 OLE/COM 调用要求 Windows 安装并注册运行库及 `pywin32`；未满足条件时返回明确错误，不伪造 CNC 状态值。完整 API 采集仍需后续命令目录、执行合同和数据证据实现。

### 16.5 西门子 S7Comm

当前只在桌面连接页保留配置入口，未接入可执行连接和采集适配器。后续实现前应先确定协议库、读写安全边界、抓包方式和设备授权模型。

## 17. 安全约束

- 默认优先只读 API；写参数、删除程序、发按键和远程控制必须有显式计划和权限。
- 写操作应遵循“读取原值 → 有界写入/控制 → 读取验证 → 恢复原值”。
- NCGuide UI 点击必须使用用户校准坐标和窗口校验，不能盲点固定数字坐标。
- 设备值、程序号、文件名和运行时绑定只能来自配置、SDK 规则或实际返回值。
- 真实 CNC 场景必须明确目标环境和允许操作；模拟器、桥接器和真实设备不能混淆。
- API 日志、PCAP 和错误证据应与任务 ID、批次 ID、调用 ID 关联，便于审计和复现。

## 18. 配置、环境变量和部署

### 18.1 Python 依赖

`pyproject.toml` 要求 Python 3.10 及以上，主要依赖包括 `langgraph`、`langchain`、`chromadb`、`sentence-transformers`、`neo4j`、`requests` 和 `pyLSV2`。EZSocket 的 COM 路径额外需要 Windows `pywin32`。

### 18.2 关键环境变量

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | OpenAI-compatible 模型密钥 |
| `FOCAS_HOST` / `FOCAS_PORT` | FOCAS/NCGuide 地址和端口 |
| `FOCAS_HEADER_DIR` | 当前 `Fwlib32.h` 目录 |
| `FOCAS_DLL_DIR` | `Fwlib32.dll` 目录 |
| `FOCAS_CPP_BRIDGE_EXE` | FOCAS C++ bridge 路径 |
| `FOCAS_BRIDGE_PYTHON` / `FOCAS_BRIDGE_PORT` | bridge Python 和端口 |
| `PCAP_NETWORK_DEVICE` | Npcap 设备名 |
| `SMPAGENT_CPP_TIMEOUT_SECONDS` | C++ 进程超时覆盖 |
| `SMP_MAX_NC_RUNTIME_SECONDS` | NC 运行预算 |
| `NCGUIDE_WINDOW_TITLE` | NCGuide 窗口标题 |
| `NCGUIDE_CYCLE_START_X/Y` | Cycle Start 校准坐标 |
| `NCGUIDE_CLICK_MODE` | `client` 或 `screen` |
| `SMP_TRG_NEO4J_URI` 及账号变量 | TRCM Neo4j 后端 |
| `SMPAGENT_SOUL_ROOT` | Agent Soul 根目录 |

### 18.3 Windows 运行要求

真实 FOCAS 采集需要：

- FANUC NCGuide 或可访问的 FOCAS 服务；
- 与当前 SDK 头文件对应的 FOCAS DLL；
- MSVC x86 编译工具链；
- Npcap 和可用抓包网卡；
- 正确的 FOCAS 主机、端口和窗口配置。

桌面入口建议通过 `run_desktop.cmd` 启动，以确保 Python 环境、DLL 搜索路径和工作目录一致。

## 19. 测试策略

测试分为四层：

1. **纯函数和数据契约测试**：模型序列化、计划规范化、返回码映射、CSV schema。
2. **生成与静态校验测试**：SDK 类型索引、结构体成员访问、确定性组装、Harness 必需标记。
3. **工具和模拟执行测试**：编译工具、模拟 CNC、bridge、日志和停止流程。
4. **真实环境验收**：NCGuide 连接、真实 FOCAS 调用、Npcap PCAP、事务匹配和 Cycle Start。

现有测试位于 `tests/`，包括 `test_deterministic_cpp_assembler.py`、`test_execution_tools.py`、`test_focas_capture_runtime.py`、`test_fast_path.py` 和日志/数据集导出测试。

真实环境验收应至少检查：

- 生成文件能被当前 MSVC 编译；
- DLL 加载和 `GetProcAddress` 成功；
- Npcap 开始并产生非空 PCAP；
- 目标 API 有 API 输入/输出日志；
- 目标 API 的调用与 PCAP 请求/响应匹配；
- 返回码和输出字段被正确序列化；
- NC 程序场景满足完成门控；
- 失败时仍保留完整证据。

## 20. 当前限制

- FOCAS SDK 类型索引是轻量解析器，不能替代完整 C/C++ 预处理器和编译器。
- 复杂结构体、匿名联合体、嵌套数组和输出字段规则仍会增加生成风险。
- 全 API 覆盖依赖函数清单、设备能力、仿真器选件和分批状态，不能保证一次运行完成。
- PCAP 事务匹配依赖调用时间窗和 TCP 流；并发调用或额外后台流量会增加关联难度。
- NCGuide UI 自动化依赖窗口标题、显示器布局和操作员校准。
- EZSocket 完整采集和 S7Comm 执行适配器尚未完成。
- Neo4j、长期记忆和 Agent Soul 属于辅助知识路径，不能替代当前 SDK 和设备证据。
- 当前工作区有较多实验产物和未提交修改，发布前需要整理版本、锁定依赖并单独建立回归基线。

## 21. 后续改进建议

### 21.1 降低 C++ 生成失败率

- 按当前目标 API 裁剪 RAG 和计划上下文，只向模型提供相关结构体和函数；
- 复杂结构体 API 采用单 API、单步骤生成；
- 把类型解析、字段合法性、数组维度和输出字段规则全部移到本地工具；
- 为每个可采集 API 建立版本化 `ApiContract`，包含 ABI、输入、输出和序列化规则；
- 将固定 Harness、结构体声明和日志代码与模型生成片段彻底分离；
- 对模型输出实行 JSON schema、字段白名单和编译前拒绝策略；
- 将“辅助 API 生成”和“目标 API 调用”分别校验，避免辅助路径覆盖目标调用缺失。

### 21.2 改善计划质量

- 让 Planner 先生成小型依赖图，再展开为完整 `ExecutionPlan`；
- 为每个目标 API 提前执行前置条件检查；
- 将设备能力、选件和 SDK 适用性作为计划输入，而不是执行后才发现；
- 维护独立的参数覆盖策略，避免全量 API 任务把所有参数规则一次性塞给模型。

### 21.3 改善可观测性

- 为每次模型请求保存上下文摘要、检索数量、响应 schema 校验和 token 统计；
- 为每个生成片段记录源 API、SDK 哈希和组装规则版本；
- 为 PCAP 关联保存更细的窗口、TCP 流和排除包诊断；
- 建立按错误类别统计的成功率，而不是只统计总成功率。

### 21.4 扩展协议

- 为 LSV/2 和 EZSocket 建立与 FOCAS 等价的 `ProtocolExecutionContract`；
- 明确每个协议的只读、写入、权限和抓包证据要求；
- 在 S7Comm 适配器开放前完成安全评审、命令目录和设备验收；
- 将协议级结果统一映射到 `ApiCallLog`，但保留协议专有字段。

## 22. 设计结论

SMPAgent 的核心不是单次生成一段 C++，而是一个带有知识检索、类型约束、固定运行时、真实设备执行和证据审计的闭环工作流。当前代码已经具备 Router、Planner、CodeGenerator、Executor、RAG、SDK 类型索引、TRCM、Agent Soul、FOCAS/NCGuide 采集和多协议边界。

从历史执行内容看，结构体处理和过大的生成上下文确实会提高失败风险，但实际失败还来自确定性组装规则、SDK 类型错误、运行时符号、设备前置条件、UI 门控和抓包条件。后续应继续把模型负责的“决策”压缩到结构化合同，把类型和运行时正确性放回本地工具、编译器和真实证据中。
