# Smart Traffic Agent

## SMPAgent 桌面采集入口

选择需要运行 NC 程序的 API 或场景后，启动前必须完成 Cycle Start 定位并启用。未完成时会提示并切到定位页面，不会启动采集进程；全部 API 模式也执行此检查。仅数据读写类目标不强制要求定位。

连接页提供 FANUC FOCAS、三菱 EZSocket、西门子 S7Comm、海德汉 LSV/2 协议入口。当前 FOCAS 已接入真实采集；其余三种协议首次选择时继承当前 FOCAS 的完整配置、API 清单、场景、提示词及 Cycle Start 设置，全部显示并可编辑；之后各协议在本次会话中独立保留修改。继承内容明确标记为 FOCAS 模板，真实协议采集与连接测试仍待接入，不会误启动 FOCAS 后端。

双击项目根目录的 `run_desktop.cmd` 启动原生 Windows 窗口，无需浏览器。
也可执行 `.\.venv\Scripts\python.exe -m smart_traffic_agent.desktop`。

连接页显示“SDK 自动识别型号”，无需手动选择。读取所配置 SDK 目录中的 Fwlib32.h 型号宏（例如 FS0IDD → 0i-D），按 [FOCAS 函数表](https://www.woody.vip/fanuc/flist_com.htm) 的 Ethernet 适用矩阵筛选：不适用 API 保留在列表中但变灰且不可选，鼠标或键盘尝试选择会提示原因；更换 SDK 目录后自动识别型号，并清除不再适用的已选项。数量、提示词与“全部 API”实际采集范围均只计算可选函数。缺少适用记录的函数单独提示“缺少适用记录，暂不可选”，不等同于文档明确不支持。完整场景含不可选目标时启动校验会提示并阻止执行。头文件缺失或通用 SDK 无法唯一识别型号时，明确显示“SDK 无法唯一识别型号（显示全部）”，不猜测型号。

适用数据离线保存在 `rag_indexes/focas/api_compatibility.json`，包含来源与获取日期；可运行 `python scripts/build_api_compatibility.py` 更新。型号按网站分组展示；当前采集使用 Ethernet，所以 HSSB 专用 API 也不可选。文档适用不代表仿真器已启用相应选件，实际采集仍会验证返回结果。

1. 在“连接配置”填写或选择 NCGuide 地址、端口、抓包网卡及 SDK 路径，点击“测试 FOCAS 连接”。测试会真正读取 CNC 状态，而非仅检查 TCP 端口。
2. 模型地址、模型名沿用 `main.py` 当前默认值；API Key 自动从环境变量或项目 `.env` 的 `DEEPSEEK_API_KEY` 读取，密码形式显示，不保存到运行配置文件。支持 Chat Completions / Responses。
3. 在“采集模式与提示词”选择“单个 API”或“全部 API”。单个 API 模式支持分类、搜索并单选函数；全部 API 模式按函数清单逐个采集，不受搜索筛选影响。已有覆盖成功的 API 会跳过；单个目标失败时保留证据并继续下一个目标，本轮不重复尝试该 API。首次验证可设置 FOCAS_MAX_CAPTURE_BATCHES=1，只执行一个目标并保留结果。场景与 API 提示词分别保存编辑内容，切换模式不会丢失。只有调用与真实 PCAP 请求/响应匹配才计入采集数，不可用项不计入实采覆盖率。
4. 完成两项连接测试后，使用窗口顶部始终可见的“▶ 开始采集”按钮；API 页右下方也有“▶ 开始采集所选内容”。启动时会重新检查连接、模型、编译器、SDK、Npcap 网卡和输出目录；失败会给出原因。修改连接参数后必须重新测试。
5. 所选 API 自动分批执行，每批单独输出。监控页显示后端事件、耗时、API 实采进度、PCAP/CSV 文件数量和大小；详细日志写入本次目录的 `desktop.log`。未产生包含数据的 PCAP 不会报告成功。

每个场景的 `execution/data/by_api/` 按 API 名称整理抓包和参数日志。例如 `cnc_actf` 的重复调用会放在 `20261009_focas_actf.pcap`、`20261009_focas_actf_input.csv`、`20261009_focas_actf_output.csv`。两张 CSV 使用相同的 `index` 和 `invocation_id` 对应一次调用；`traffic_status=matched` 表示该调用有可靠的请求/响应流量关联。没有匹配报文的调用仍保留输入、输出记录，但不会生成空 PCAP。总抓包和总输入/输出日志仍位于 `execution/data/`。

桌面入口直接调用 `TrafficGenerationWorkflow`，关闭流量质量评分门控，不运行评估、基线和消融脚本；仍保留 API 返回码、代码编译及采集失败检查。已有实验文件保留以便追溯。

Cycle Start 坐标不再使用不可靠的固定默认值。在场景页点击“开始定位 Cycle Start”，窗口会隐藏并显示覆盖所有显示器的半透明定位遮罩；点击 NCGuide 的 Cycle Start 位置后，遮罩拦截此次点击（不会实际启动程序），界面自动识别下方窗口、填入屏幕 X/Y 和窗口标题并确认。未定位时不能开始需要自动启动的采集。停止按钮终止本次工作进程和编译/采集子进程，保留部分文件；它不等同于停止仿真器内已启动的 NC 程序，必要时在 NCGuide 内停止程序。

规划完成后直接进入代码生成，不设置独立的规划审批/反复修复关卡。后续生成、编译、执行或结果验证失败时，再依据实际错误判断是否需要修代码、调整执行或重新规划；保留代码校验与运行安全检查。

主要实现：`smart_traffic_agent/desktop.py`（原生界面）、`smart_traffic_agent/desktop_backend.py`（连接测试、模板和采集服务）。

This repository contains a runnable first version of the CNC protocol traffic
generation workflow described in the design documents.

## Agent Soul and Traffic Relationship Graph

The four existing agents load scoped, versioned role guidance from
`agent_souls/{router,planner,code_generator,executor}/`. Runtime experiences
remain in the existing long-term memory store and are retrieved only for the
agent and task that need them. Missing soul files or experience records do not
silently become execution rules.

The Planner queries the shared Traffic Relationship Graph and parameter tools
before asking the model to plan. API nodes and evidence-scoped relations are
stored through `TrafficRelationshipService`; set `SMP_TRG_NEO4J_URI` (and the
matching user/password variables) to use Neo4j. Without a configured backend,
the service reports `unknown` and the original RAG/SDK planning path remains
available. Relations marked `proposed` are hypotheses and cannot add mandatory
dependencies. Complete successful execution evidence is written to
`execution/traffic_relationship_feedback.json` and may promote only relations
that pass explicit verification.

Parameter discovery and filtering are separate from the graph: observed
values, operation-specific exclusions, documented ranges, and bounded
combinations are returned as structured tool results. Upload-existing and
download-new program selection use separate filtering contexts, and empty or
unverified discovery never invents a device value.

It implements:

- task routing across planning, code generation, execution, repair, and quality assessment
- lightweight local knowledge retrieval
- structured execution plans for CNC traffic scenarios
- LLM-generated C++ API scripts and NC programs
- a simulator execution backend
- operation logs, capture events, and traffic-quality metrics

The external systems from the design, such as RAGFlow, real CNC SDKs, and packet
capture tools, are represented by replaceable interfaces. The default backend is
safe to run locally and produces deterministic simulation artifacts.

## Quick Start

```powershell
python -m smart_traffic_agent init-knowledge --out examples/knowledge.json
python -m smart_traffic_agent run --knowledge examples/knowledge.json --out runs/demo
```

Each run writes to a new numbered subdirectory such as
`runs/demo/run_001`, `runs/demo/run_002`. The run directory will contain:

- `plan.json`
- `generated/api_script.py`
- `generated/program.nc`
- `execution/api_logs.jsonl`
- `execution/capture_events.jsonl`
- `summary.json`

The workflow is orchestrated with LangGraph. Router, planning, code generation,
execution, and repair are graph nodes that share a workflow state. LangChain
is used as the chat-model abstraction. The default run uses the official OpenAI-compatible
DeepSeek endpoint with `deepseek-flash`, high reasoning effort, and project-local `.env` configuration. Agent decisions require a configured
LLM. Unit tests cover local validation and state contracts without pretending to
produce agent decisions; LLM integration tests require real provider credentials.

## Test

```powershell
python -m unittest discover -s tests
```

## FOCAS RAG Workflow

The repository keeps the builders and small reference files in Git. Large
generated JSONL indexes and extracted manual text are ignored because they can be
rebuilt locally and may contain copyrighted manual content.

Build the FOCAS API reference chunks:

```powershell
python -m smart_traffic_agent build-focas-rag --out rag_indexes/focas/chunks.jsonl
```

Build the global FOCAS function coverage manifest from the local function list
workbook. This manifest is the source of truth for the default FOCAS goal:
full, high-quality coverage of every listed function across batches.

```powershell
python -m smart_traffic_agent build-focas-function-manifest --workbook "e:\2025-06\codenew\focas_funcs.xlsx" --out rag_indexes/focas/function_manifest.jsonl
```

Coverage progresses cumulatively across scenario batches. Each run records target
function outcomes in a local state file, while support APIs used to create or
verify a scenario are tracked separately and do not count as target progress.
One run may contain multiple ordered coverage segments, such as an NC motion
segment, a program-lifecycle segment, and a tool-offset read/write-restore
segment. The generated C++ executes those segments sequentially; segments that
need NC motion may use bounded NC payloads, while probe/read/write-restore
segments can run as direct FOCAS call blocks.
Inspect the cumulative state with:

```powershell
python -m smart_traffic_agent show-focas-function-coverage --state rag_indexes/focas/function_coverage_state.json
```

Build FANUC manual source chunks from local PDFs:

```powershell
python -m smart_traffic_agent build-fanuc-manual-chunks --manual-dir "E:\研二下\FANUC-0i-MF全套说明书" --out rag_indexes/focas/manual_chunks.jsonl
```

Filter manual chunks into traffic-generation candidates:

```powershell
python -m smart_traffic_agent filter-manual-candidates --input rag_indexes/focas/manual_chunks.jsonl --out rag_indexes/focas/candidate_chunks.jsonl
```

Write the scenario taxonomy and rule extraction prompt:

```powershell
python -m smart_traffic_agent write-focas-taxonomy
```

Prepare batched prompts for model-based rule extraction:

```powershell
python -m smart_traffic_agent prepare-rule-extraction-batches --input rag_indexes/focas/candidate_chunks.jsonl --out-dir rag_indexes/focas/rule_extraction_batches
```

Save model outputs as JSON files under
`rag_indexes/focas/rule_extraction_results/`, then merge them:

```powershell
python -m smart_traffic_agent merge-rule-extraction-results --out rag_indexes/focas/rule_chunks.jsonl
```

If the model has already produced one combined JSON file, merge that file into
traceable rule chunks:

```powershell
python -m smart_traffic_agent merge-rule-extraction-results --merged-json rag_indexes/focas/rule_extraction_results_merged.json --out rag_indexes/focas/rule_chunks.jsonl
```

Build the local Chroma vector database from API chunks and rule chunks:

```powershell
python -m smart_traffic_agent build-focas-vector-db --api-chunks rag_indexes/focas/chunks.jsonl --rule-chunks rag_indexes/focas/rule_chunks.jsonl --out-dir rag_indexes/focas/vector_db
```

Run a RAG-driven LangGraph workflow. The default run configuration is:

- protocol: `focas`
- target: `simulator`
- vector DB: `rag_indexes/focas/vector_db`
- LLM provider: `openai_compatible`
- model: `deepseek-flash`
- base URL: `https://api.deepseek.com`
- transport: direct HTTP (`/chat/completions`)
- thinking: enabled, `reasoning_effort=high`

Put the project-local API key in `.env`:

```env
DEEPSEEK_API_KEY=your_api_key
```

Then run directly. If no task text is provided, the default task is to generate
comprehensive and diverse FOCAS protocol traffic:

```powershell
python -m smart_traffic_agent run
```

Override defaults only when needed:

```powershell
python -m smart_traffic_agent run "生成主轴转速变化流量，采集主轴速度和运行状态"
python -m smart_traffic_agent run "生成坐标运动流量" --out runs/coordinate_demo
python -m smart_traffic_agent run "生成坐标运动流量" --llm-provider openai_compatible --llm-model "deepseek-flash" --llm-base-url "https://api.deepseek.com" --llm-api-key-env DEEPSEEK_API_KEY --llm-transport http --llm-reasoning-effort high
```

The defaults already match the command above; explicit options are only needed to override them.


## FANUC NCGuide Probe

Probe a local FANUC NCGuide FS0i-F installation before enabling real FOCAS
execution:

```powershell
python -m smart_traffic_agent probe-ncguide --install-dir "D:\Program Files (x86)\FANUC\NCGuide FS0i-F" --ports 8193 8194 6002
```

The probe checks:

- whether the NCGuide directory exists
- whether `Fwlib32.dll` / `fwlibNCG.dll` are present
- whether the current Python process can load those DLLs
- whether the configured TCP ports are reachable

The NCGuide installation under `Program Files (x86)` usually ships 32-bit FOCAS
DLLs. A 64-bit Python process cannot load them directly, so real DLL calls go
through a small 32-bit C++ bridge helper. The bridge loads the FOCAS runtime DLLs
from `FOCAS_DLL_DIR` when that variable is set; otherwise it uses the known
working demo runtime directory and falls back to the NCGuide installation
directory.

Build the C++ bridge:

```powershell
.\scripts\build_focas_cpp_bridge.ps1
```

Test the compiled bridge:

```powershell
python -m smart_traffic_agent test-focas-cpp-bridge --action read_run_status
```

## NCGuide UI Start Trigger

The generated C++ execution path can upload/select an NC program through FOCAS
and trigger NCGuide Cycle Start through Windows UI input. First calibrate the
window-relative coordinates:

```powershell
& ".venv\Scripts\python.exe" scripts\calibrate_ncguide_ui.py
```

If Cycle Start is in a separate signal simulator window, list candidate windows
and calibrate that window instead:

```powershell
& ".venv\Scripts\python.exe" scripts\calibrate_ncguide_ui.py --list-windows
& ".venv\Scripts\python.exe" scripts\calibrate_ncguide_ui.py --window-title "Machine Signal Simulator" --interactive
```

Move the mouse to the NCGuide Cycle Start button and note `client_x/client_y`.
If mode switching is also needed, move the mouse to the MEM/AUTO control and
note that coordinate too. Then run:

```powershell
& ".venv\Scripts\python.exe" main.py `
  --trigger-ncguide-ui `
  --ncguide-cycle-start-x <cycle_start_x> `
  --ncguide-cycle-start-y <cycle_start_y>
```

If a mode click is required before Cycle Start:

```powershell
& ".venv\Scripts\python.exe" main.py `
  --trigger-ncguide-ui `
  --ncguide-mode-x <mode_x> `
  --ncguide-mode-y <mode_y> `
  --ncguide-cycle-start-x <cycle_start_x> `
  --ncguide-cycle-start-y <cycle_start_y>
```

If a different NCGuide or FOCAS runtime is needed:

```powershell
$env:FOCAS_DLL_DIR="E:\path\to\working\focas\dlls"
python -m smart_traffic_agent test-focas-cpp-bridge --action read_run_status
```

Generated C++ uses the controller-specific official FANUC header for ABI types while still loading the DLL dynamically. The default FS0i-D SDK header directory is:

```powershell
C:\Lib\FOCAS2 Library\Fwlib\0iD
```

Override it when needed:

```powershell
$env:FOCAS_HEADER_DIR="C:\path\to\controller-specific\Fwlib"
python main.py
```

`Fwlib32.lib` is not required for generated scripts because all FOCAS entry points are resolved with `GetProcAddress`.

## ExecutionAgent Tool Pipeline

`ExecutionAgent` orchestrates three explicit tools instead of invoking the compiler or process APIs directly:

1. `CompileGeneratedCppTool` invokes the real 32-bit MSVC compiler.
2. `RunGeneratedExecutableTool` runs the real generated executable with the FOCAS runtime environment and timeout.
3. `CollectExecutionArtifactsTool` parses CSV output into structured API logs and capture events.

Every invocation is written to `execution/tool_calls.jsonl` with its tool name, duration, success state, input summary, output summary, and error. Unit tests may inject controlled test doubles to verify orchestration, but normal workflow runs always use the real compiler and executable tools.

Test the bridge helper after installing a 32-bit Python runtime:

```powershell
python -m smart_traffic_agent test-focas-bridge `
  --python "C:\Python32\python.exe" `
  --install-dir "D:\Program Files (x86)\FANUC\NCGuide FS0i-F" `
  --action read_run_status
```

The first real FOCAS call is intentionally read-only: `cnc_statinfo`. It checks
the library handle and CNC status path before any NC program upload, start, stop,
or write operation is enabled.

To run the agent through the bridge later:

```powershell
$env:FOCAS_BRIDGE_PYTHON="C:\Python32\python.exe"
python -m smart_traffic_agent run "读取 NCGuide 运行状态" --target ncguide-bridge
```


