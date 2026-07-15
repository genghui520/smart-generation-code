# Smart Traffic Agent

This repository contains a runnable first version of the CNC protocol traffic
generation workflow described in the design documents.

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
is used as the chat-model abstraction. The default run uses the OpenAI-compatible
SmartAIPro endpoint with `gpt-5.6-sol` and project-local `.env` configuration. Agent decisions require a configured
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
- model: `gpt-5.6-sol`
- base URL: `https://fast.smartaipro.cn/v1`

Put the project-local API key in `.env`:

```env
SMARTAIPRO_API_KEY=your_smartaipro_key
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
python -m smart_traffic_agent run "生成坐标运动流量" --llm-provider openai_compatible --llm-model "gpt-5.6-sol" --llm-base-url "https://fast.smartaipro.cn/v1" --llm-api-key-env SMARTAIPRO_API_KEY
```

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
