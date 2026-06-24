# Smart Traffic Agent

This repository contains a runnable first version of the CNC protocol traffic
generation workflow described in the design documents.

It implements:

- task routing across planning, code generation, execution, and annotation
- lightweight local knowledge retrieval
- structured execution plans for CNC traffic scenarios
- generated API scripts and NC programs
- a simulator execution backend
- operation logs, capture events, and traffic-log mappings

The external systems from the design, such as RAGFlow, real CNC SDKs, and packet
capture tools, are represented by replaceable interfaces. The default backend is
safe to run locally and produces deterministic simulation artifacts.

## Quick Start

```powershell
python -m smart_traffic_agent init-knowledge --out examples/knowledge.json
python -m smart_traffic_agent run "generate coordinate change traffic for CNC simulator" --knowledge examples/knowledge.json --out runs/demo
```

The run directory will contain:

- `plan.json`
- `generated/api_script.py`
- `generated/program.nc`
- `execution/api_logs.jsonl`
- `execution/capture_events.jsonl`
- `execution/mapping.json`
- `summary.json`

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
