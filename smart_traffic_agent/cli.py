from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .integrations.ncguide import (
    DEFAULT_FOCAS_PORTS,
    DEFAULT_NCGUIDE_DIR,
    FocasBridgeClient,
    FocasCppBridgeClient,
    default_focas_runtime_dir,
    probe_ncguide,
)
from .knowledge import KnowledgeBase, sample_knowledge
from .llm import LlmClient, LlmConfig
from .models import TaskRequest
from .rag.candidate_filter import filter_candidate_chunks
from .rag.fanuc_manual_loader import DEFAULT_MANUAL_DIR, build_fanuc_manual_chunks
from .rag.focas_loader import DEFAULT_FOCAS_BASE_URL, build_focas_chunks
from .rag.protocol_document_loader import build_protocol_document_chunks
from .rag.rule_extractor import (
    DEFAULT_BATCH_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RULE_CHUNKS_PATH,
    extract_rules_with_llm,
    merge_rule_extraction_merged_json,
    merge_rule_extraction_results,
    prepare_rule_extraction_batches,
)
from .rag.scenario_organizer import (
    DEFAULT_SCENARIO_CHUNKS_PATH,
    DEFAULT_SCENARIO_KNOWLEDGE_PATH,
    build_scenario_knowledge,
)
from .rag.scenario_clusterer import (
    DEFAULT_RULE_CHUNKS_PATH as DEFAULT_CLUSTER_RULE_CHUNKS_PATH,
    DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH,
    DEFAULT_SCENARIO_CLUSTERS_PATH,
    build_scenario_21_clusters,
    write_cluster_review_csv,
)
from .rag.scenario_taxonomy import load_taxonomy, write_rule_extraction_prompt, write_taxonomy
from .rag.scenario_templates import (
    DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH,
    build_final_scenario_templates,
)
from .workflow import TrafficGenerationWorkflow, workflow_success


DEFAULT_RUN_TASK = "生成全面的、多样性的 FOCAS 协议流量，覆盖程序生命周期、坐标运动、主轴转速、运行状态、参数读取和报警查询。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart-traffic-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-knowledge", help="write a sample knowledge base")
    init_parser.add_argument("--out", type=Path, default=Path("examples/knowledge.json"))

    run_parser = subparsers.add_parser("run", help="run a traffic generation task")
    run_parser.add_argument("task", nargs="?", default=DEFAULT_RUN_TASK, help="natural language traffic generation task")
    run_parser.add_argument("--knowledge", type=Path, default=Path("examples/knowledge.json"))
    run_parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    run_parser.add_argument("--task-id", default="")
    run_parser.add_argument("--target", default="simulator")
    run_parser.add_argument("--protocol", default="focas")
    run_parser.add_argument("--vector-db", type=Path, default=Path("rag_indexes/focas/vector_db"), help="optional Chroma vector DB for RAG retrieval")
    run_parser.add_argument("--llm-provider", default="openai_compatible", choices=["disabled", "openai_compatible", "openai", "tokenhub"])
    run_parser.add_argument("--llm-model", default="gpt-5.6-sol")
    run_parser.add_argument("--llm-base-url", default="https://fast.smartaipro.cn/v1")
    run_parser.add_argument("--llm-api-key-env", default="SMARTAIPRO_API_KEY")
    run_parser.add_argument(
        "--allow-delete-all-programs",
        action="store_true",
        help="Authorize PlannerAgent to consider cnc_delall for this run. Disabled by default.",
    )

    focas_parser = subparsers.add_parser("build-focas-rag", help="build FOCAS RAG chunks from online reference")
    focas_parser.add_argument("--base-url", default=DEFAULT_FOCAS_BASE_URL)
    focas_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/chunks.jsonl"))
    focas_parser.add_argument("--limit", type=int, default=0, help="limit function XML downloads for testing")

    manual_parser = subparsers.add_parser("build-fanuc-manual-chunks", help="build manual source chunks from FANUC PDFs")
    manual_parser.add_argument("--manual-dir", type=Path, default=DEFAULT_MANUAL_DIR)
    manual_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/manual_chunks.jsonl"))
    manual_parser.add_argument("--files", nargs="*", default=None, help="PDF file names to include")
    manual_parser.add_argument("--max-chars", type=int, default=6000)

    document_parser = subparsers.add_parser("build-protocol-document-chunks", help="build generic source chunks from PDF/HTML/DOCX/TXT/MD files")
    document_parser.add_argument("inputs", type=Path, nargs="+", help="source files or directories")
    document_parser.add_argument("--protocol", required=True, help="protocol name, for example modbus/opcua/s7/focas")
    document_parser.add_argument("--out", type=Path, default=None, help="output JSONL path; defaults to rag_indexes/<protocol>/document_chunks.jsonl")
    document_parser.add_argument("--max-chars", type=int, default=6000)
    document_parser.add_argument("--overlap", type=int, default=500)

    candidate_parser = subparsers.add_parser("filter-manual-candidates", help="filter manual chunks for traffic-rule extraction")
    candidate_parser.add_argument("--input", type=Path, default=Path("rag_indexes/focas/manual_chunks.jsonl"))
    candidate_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/candidate_chunks.jsonl"))
    candidate_parser.add_argument("--min-score", type=float, default=1.0)

    taxonomy_parser = subparsers.add_parser("write-focas-taxonomy", help="write FOCAS traffic scenario taxonomy and rule prompt")
    taxonomy_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/scenario_taxonomy.json"))
    taxonomy_parser.add_argument("--prompt-out", type=Path, default=Path("rag_indexes/focas/rule_extraction_prompt.md"))

    batch_parser = subparsers.add_parser("prepare-rule-extraction-batches", help="write batched prompts for manual rule extraction")
    batch_parser.add_argument("--input", type=Path, default=Path("rag_indexes/focas/candidate_chunks.jsonl"))
    batch_parser.add_argument("--out-dir", type=Path, default=DEFAULT_BATCH_DIR)
    batch_parser.add_argument("--batch-size", type=int, default=8)
    batch_parser.add_argument("--max-chars-per-chunk", type=int, default=4500)
    batch_parser.add_argument("--taxonomy", type=Path, default=None, help="protocol taxonomy JSON; defaults to built-in FOCAS taxonomy")

    extract_parser = subparsers.add_parser("extract-rules-with-llm", help="extract traffic-generation rules by calling the configured LLM")
    extract_parser.add_argument("--input", type=Path, default=Path("rag_indexes/focas/candidate_chunks.jsonl"))
    extract_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    extract_parser.add_argument("--batch-size", type=int, default=4)
    extract_parser.add_argument("--max-chars-per-chunk", type=int, default=4500)
    extract_parser.add_argument("--limit-batches", type=int, default=0, help="limit batches for a small test run")
    extract_parser.add_argument("--llm-provider", default="openai_compatible", choices=["openai_compatible", "openai", "tokenhub"])
    extract_parser.add_argument("--llm-model", default="gpt-5.6-sol")
    extract_parser.add_argument("--llm-base-url", default="https://fast.smartaipro.cn/v1")
    extract_parser.add_argument("--llm-api-key-env", default="SMARTAIPRO_API_KEY")
    extract_parser.add_argument("--taxonomy", type=Path, default=None, help="protocol taxonomy JSON; defaults to built-in FOCAS taxonomy")

    merge_parser = subparsers.add_parser("merge-rule-extraction-results", help="merge model JSON outputs into rule chunks")
    merge_parser.add_argument("--candidates", type=Path, default=Path("rag_indexes/focas/candidate_chunks.jsonl"))
    merge_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    merge_parser.add_argument("--merged-json", type=Path, default=None, help="merge one already-combined JSON result file")
    merge_parser.add_argument("--out", type=Path, default=DEFAULT_RULE_CHUNKS_PATH)
    merge_parser.add_argument("--taxonomy", type=Path, default=None, help="protocol taxonomy JSON; defaults to built-in FOCAS taxonomy")

    vector_parser = subparsers.add_parser("build-focas-vector-db", help="build Chroma vector DB for FOCAS RAG")
    vector_parser.add_argument("--api-chunks", type=Path, default=Path("rag_indexes/focas/chunks.jsonl"))
    vector_parser.add_argument("--rule-chunks", type=Path, default=Path("rag_indexes/focas/rule_chunks.jsonl"))
    vector_parser.add_argument("--scenario-chunks", type=Path, default=DEFAULT_SCENARIO_CHUNKS_PATH)
    vector_parser.add_argument("--out-dir", type=Path, default=Path("rag_indexes/focas/vector_db"))
    vector_parser.add_argument("--no-reset", action="store_true", help="do not delete the existing vector DB first")

    organize_parser = subparsers.add_parser("organize-focas-scenarios", help="build scenario-centered FOCAS knowledge")
    organize_parser.add_argument("--api-chunks", type=Path, default=Path("rag_indexes/focas/chunks.jsonl"))
    organize_parser.add_argument("--rule-chunks", type=Path, default=Path("rag_indexes/focas/rule_chunks.jsonl"))
    organize_parser.add_argument("--out", type=Path, default=DEFAULT_SCENARIO_KNOWLEDGE_PATH)
    organize_parser.add_argument("--scenario-chunks", type=Path, default=DEFAULT_SCENARIO_CHUNKS_PATH)

    cluster_parser = subparsers.add_parser("cluster-focas-scenarios", help="cluster fine-grained FOCAS traffic scenarios")
    cluster_parser.add_argument("--taxonomy", type=Path, default=None, help="scenario taxonomy JSON; defaults to built-in FOCAS taxonomy")
    cluster_parser.add_argument("--rules", type=Path, default=DEFAULT_CLUSTER_RULE_CHUNKS_PATH)
    cluster_parser.add_argument("--scenario-knowledge", type=Path, default=None, help="optional organized scenario knowledge; omitted by default to keep clustering data-driven")
    cluster_parser.add_argument("--cluster-count", type=int, default=0, help="optional fixed k; default 0 lets the algorithm select k")
    cluster_parser.add_argument("--min-clusters", type=int, default=6)
    cluster_parser.add_argument("--max-clusters", type=int, default=36)
    cluster_parser.add_argument("--out", type=Path, default=DEFAULT_SCENARIO_CLUSTERS_PATH)

    review_parser = subparsers.add_parser("export-scenario-cluster-review", help="export scenario cluster review CSV")
    review_parser.add_argument("--clusters", type=Path, default=DEFAULT_SCENARIO_CLUSTERS_PATH)
    review_parser.add_argument("--out", type=Path, default=DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH)

    template_parser = subparsers.add_parser("build-scenario-templates", help="build executable scenario templates from reviewed clusters")
    template_parser.add_argument("--review", type=Path, default=DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH)
    template_parser.add_argument("--clusters", type=Path, default=DEFAULT_SCENARIO_CLUSTERS_PATH)
    template_parser.add_argument("--out", type=Path, default=DEFAULT_FINAL_SCENARIO_TEMPLATES_PATH)

    probe_parser = subparsers.add_parser("probe-ncguide", help="probe a FANUC NCGuide installation")
    probe_parser.add_argument("--install-dir", type=Path, default=DEFAULT_NCGUIDE_DIR)
    probe_parser.add_argument("--host", default="127.0.0.1")
    probe_parser.add_argument("--ports", type=int, nargs="*", default=DEFAULT_FOCAS_PORTS)
    probe_parser.add_argument("--json", action="store_true", help="print the full probe result as JSON")

    bridge_parser = subparsers.add_parser("test-focas-bridge", help="call FOCAS through a 32-bit bridge helper")
    bridge_parser.add_argument("--python", type=Path, default=Path("C:/Python32/python.exe"), help="32-bit Python executable")
    bridge_parser.add_argument("--install-dir", type=Path, default=default_focas_runtime_dir())
    bridge_parser.add_argument("--host", default="127.0.0.1")
    bridge_parser.add_argument("--port", type=int, default=DEFAULT_FOCAS_PORTS[0])
    bridge_parser.add_argument(
        "--action",
        choices=["connect", "read_run_status", "read_position", "read_feed_speed", "read_spindle_speed", "read_alarm"],
        default="read_run_status",
    )

    cpp_bridge_parser = subparsers.add_parser("test-focas-cpp-bridge", help="call FOCAS through the compiled C++ bridge helper")
    cpp_bridge_parser.add_argument("--bridge-exe", type=Path, default=Path(".tools/focas_bridge_cpp/focas_bridge.exe"))
    cpp_bridge_parser.add_argument("--install-dir", type=Path, default=default_focas_runtime_dir())
    cpp_bridge_parser.add_argument("--host", default="127.0.0.1")
    cpp_bridge_parser.add_argument("--port", type=int, default=DEFAULT_FOCAS_PORTS[0])
    cpp_bridge_parser.add_argument(
        "--action",
        choices=["probe", "connect", "read_run_status", "read_position", "read_feed_speed", "read_spindle_speed", "read_alarm"],
        default="read_run_status",
    )

    args = parser.parse_args(argv)

    if args.command == "init-knowledge":
        kb = sample_knowledge()
        kb.to_json(args.out)
        print(f"Knowledge base written to {args.out}")
        return 0

    if args.command == "run":
        output_dir = next_run_output_dir(args.out)
        if args.vector_db:
            kb = KnowledgeBase(vector_db_dir=args.vector_db)
            if not kb.vector_search_available:
                kb = KnowledgeBase.from_json(args.knowledge)
        else:
            kb = KnowledgeBase.from_json(args.knowledge)
        if not kb.chunks and not kb.vector_search_available:
            kb = sample_knowledge()
        request = TaskRequest(
            description=args.task,
            task_id=args.task_id or uuid.uuid4().hex[:12],
            protocol=args.protocol,
            target_environment=args.target,
            permissions={"allow_delete_all_programs": args.allow_delete_all_programs},
        )
        llm_api_key_env = args.llm_api_key_env
        if args.llm_provider == "tokenhub" and llm_api_key_env == "LLM_API_KEY":
            llm_api_key_env = "TOKENHUB_API_KEY"
        llm_client = LlmClient.from_config(
            LlmConfig(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=llm_api_key_env,
            )
        )
        state = TrafficGenerationWorkflow(kb, llm_client=llm_client).run(request, output_dir)
        success = workflow_success(state)
        print(f"Task: {request.task_id}")
        print(f"Scenario: {state.plan.scenario_type if state.plan else 'none'}")
        print(f"Success: {success}")
        print(f"Output: {output_dir}")
        return 0 if success else 1

    if args.command == "build-focas-rag":
        chunks = build_focas_chunks(
            args.out,
            base_url=args.base_url,
            limit=args.limit or None,
        )
        print(f"FOCAS chunks: {len(chunks)}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "build-fanuc-manual-chunks":
        chunks = build_fanuc_manual_chunks(
            args.out,
            manual_dir=args.manual_dir,
            files=args.files,
            max_chars=args.max_chars,
        )
        print(f"FANUC manual chunks: {len(chunks)}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "build-protocol-document-chunks":
        output_path = args.out or Path("rag_indexes") / args.protocol / "document_chunks.jsonl"
        chunks = build_protocol_document_chunks(
            output_path,
            protocol=args.protocol,
            inputs=args.inputs,
            max_chars=args.max_chars,
            overlap=args.overlap,
        )
        print(f"Protocol document chunks: {len(chunks)}")
        print(f"Protocol: {args.protocol}")
        print(f"Output: {output_path}")
        return 0

    if args.command == "filter-manual-candidates":
        candidates = filter_candidate_chunks(args.input, args.out, min_score=args.min_score)
        print(f"Candidate chunks: {len(candidates)}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "write-focas-taxonomy":
        write_taxonomy(args.out)
        write_rule_extraction_prompt(args.prompt_out)
        print(f"Taxonomy: {args.out}")
        print(f"Prompt: {args.prompt_out}")
        return 0

    if args.command == "prepare-rule-extraction-batches":
        taxonomy = load_taxonomy(args.taxonomy)
        manifests = prepare_rule_extraction_batches(
            args.input,
            args.out_dir,
            batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
            taxonomy=taxonomy,
        )
        chunk_count = sum(manifest["chunk_count"] for manifest in manifests)
        print(f"Rule extraction batches: {len(manifests)}")
        print(f"Candidate chunks: {chunk_count}")
        print(f"Output dir: {args.out_dir}")
        print(f"Manifest: {args.out_dir / 'manifest.jsonl'}")
        return 0

    if args.command == "extract-rules-with-llm":
        taxonomy = load_taxonomy(args.taxonomy)
        llm_client = LlmClient.from_config(
            LlmConfig(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                api_key_env=args.llm_api_key_env,
            )
        )
        paths = extract_rules_with_llm(
            args.input,
            args.results_dir,
            llm_client=llm_client,
            batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
            limit_batches=args.limit_batches or None,
            taxonomy=taxonomy,
        )
        print(f"LLM result files: {len(paths)}")
        print(f"Model: {args.llm_model}")
        print(f"Output dir: {args.results_dir}")
        return 0

    if args.command == "merge-rule-extraction-results":
        taxonomy = load_taxonomy(args.taxonomy)
        if args.merged_json:
            rules = merge_rule_extraction_merged_json(args.candidates, args.merged_json, args.out, taxonomy=taxonomy)
        else:
            rules = merge_rule_extraction_results(args.candidates, args.results_dir, args.out, taxonomy=taxonomy)
        print(f"Rule chunks: {len(rules)}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "build-focas-vector-db":
        from .rag.build_rag_index import build as build_focas_vector_db

        count = build_focas_vector_db(
            api_chunks_path=args.api_chunks,
            rule_chunks_path=args.rule_chunks,
            scenario_chunks_path=args.scenario_chunks,
            vector_db_dir=args.out_dir,
            reset=not args.no_reset,
        )
        print(f"Vector documents: {count}")
        print(f"Output: {args.out_dir}")
        return 0

    if args.command == "organize-focas-scenarios":
        payload = build_scenario_knowledge(
            api_chunks_path=args.api_chunks,
            rule_chunks_path=args.rule_chunks,
            output_path=args.out,
            scenario_chunks_path=args.scenario_chunks,
        )
        print(f"Scenario knowledge: {payload['statistics']['scenario_count']}")
        print(f"Source rules: {payload['statistics']['source_rule_count']}")
        print(f"Output: {args.out}")
        print(f"Scenario chunks: {args.scenario_chunks}")
        return 0

    if args.command == "cluster-focas-scenarios":
        taxonomy = load_taxonomy(args.taxonomy)
        payload = build_scenario_21_clusters(
            args.out,
            taxonomy=taxonomy,
            rule_chunks_path=args.rules,
            scenario_knowledge_path=args.scenario_knowledge,
            cluster_count=args.cluster_count or None,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
        )
        print(f"Scenario clusters: {payload['statistics']['scenario_cluster_count']}")
        print(f"Cluster selection: {payload['cluster_count_selection']['mode']}")
        print(f"Unique clusters: {payload['statistics']['unique_cluster_count']}")
        print(f"Knowledge units: {payload['statistics']['knowledge_unit_count']}")
        print(f"Nonempty clusters: {payload['statistics']['nonempty_cluster_count']}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "export-scenario-cluster-review":
        rows = write_cluster_review_csv(args.clusters, args.out)
        print(f"Review rows: {len(rows)}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "build-scenario-templates":
        payload = build_final_scenario_templates(
            review_csv_path=args.review,
            clusters_path=args.clusters,
            output_path=args.out,
        )
        print(f"Scenario templates: {payload['template_count']}")
        print(f"Output: {args.out}")
        return 0

    if args.command == "probe-ncguide":
        result = probe_ncguide(
            install_dir=args.install_dir,
            host=args.host,
            ports=args.ports,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0
        print(f"Install dir: {result.install_dir}")
        print(f"Install dir exists: {result.install_dir_exists}")
        print(f"Python bits: {result.python_bits}")
        print("FOCAS DLLs:")
        for name, info in result.dlls.items():
            print(f"  {name}: exists={info['exists']} bits={info['bits']} path={info['path']}")
        print("TCP ports:")
        for port, ok in result.ports.items():
            print(f"  {result.host}:{port}: {'open' if ok else 'closed'}")
        print(f"Can load DLL in current Python: {result.can_load_focas_dll_in_current_python}")
        print("Notes:")
        for note in result.notes:
            print(f"  - {note}")
        return 0 if result.install_dir_exists else 1

    if args.command == "test-focas-bridge":
        client = FocasBridgeClient(
            python_exe=args.python,
            install_dir=args.install_dir,
            host=args.host,
            port=args.port,
        )
        result = client.run_bridge(args.action)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status_code") == 0 else 1

    if args.command == "test-focas-cpp-bridge":
        client = FocasCppBridgeClient(
            bridge_exe=args.bridge_exe,
            install_dir=args.install_dir,
            host=args.host,
            port=args.port,
        )
        result = client.run_bridge(args.action)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status_code") == 0 else 1

    return 1


def next_run_output_dir(base_out: Path) -> Path:
    if not base_out.exists():
        return base_out / "run_001"
    for index in range(1, 10000):
        candidate = base_out / f"run_{index:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a new run directory under {base_out}")
