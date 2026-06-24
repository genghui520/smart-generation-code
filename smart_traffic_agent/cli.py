from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from .knowledge import KnowledgeBase, sample_knowledge
from .models import TaskRequest
from .rag.candidate_filter import filter_candidate_chunks
from .rag.fanuc_manual_loader import DEFAULT_MANUAL_DIR, build_fanuc_manual_chunks
from .rag.focas_loader import DEFAULT_FOCAS_BASE_URL, build_focas_chunks
from .rag.rule_extractor import (
    DEFAULT_BATCH_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RULE_CHUNKS_PATH,
    merge_rule_extraction_results,
    prepare_rule_extraction_batches,
)
from .rag.scenario_taxonomy import write_rule_extraction_prompt, write_taxonomy
from .workflow import TrafficGenerationWorkflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="smart-traffic-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-knowledge", help="write a sample knowledge base")
    init_parser.add_argument("--out", type=Path, default=Path("examples/knowledge.json"))

    run_parser = subparsers.add_parser("run", help="run a traffic generation task")
    run_parser.add_argument("task", help="natural language traffic generation task")
    run_parser.add_argument("--knowledge", type=Path, default=Path("examples/knowledge.json"))
    run_parser.add_argument("--out", type=Path, default=Path("runs/latest"))
    run_parser.add_argument("--task-id", default="")
    run_parser.add_argument("--target", default="simulator")
    run_parser.add_argument("--protocol", default="cnc")

    focas_parser = subparsers.add_parser("build-focas-rag", help="build FOCAS RAG chunks from online reference")
    focas_parser.add_argument("--base-url", default=DEFAULT_FOCAS_BASE_URL)
    focas_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/chunks.jsonl"))
    focas_parser.add_argument("--limit", type=int, default=0, help="limit function XML downloads for testing")

    manual_parser = subparsers.add_parser("build-fanuc-manual-chunks", help="build manual source chunks from FANUC PDFs")
    manual_parser.add_argument("--manual-dir", type=Path, default=DEFAULT_MANUAL_DIR)
    manual_parser.add_argument("--out", type=Path, default=Path("rag_indexes/focas/manual_chunks.jsonl"))
    manual_parser.add_argument("--files", nargs="*", default=None, help="PDF file names to include")
    manual_parser.add_argument("--max-chars", type=int, default=6000)

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

    merge_parser = subparsers.add_parser("merge-rule-extraction-results", help="merge model JSON outputs into rule chunks")
    merge_parser.add_argument("--candidates", type=Path, default=Path("rag_indexes/focas/candidate_chunks.jsonl"))
    merge_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    merge_parser.add_argument("--out", type=Path, default=DEFAULT_RULE_CHUNKS_PATH)

    args = parser.parse_args(argv)

    if args.command == "init-knowledge":
        kb = sample_knowledge()
        kb.to_json(args.out)
        print(f"Knowledge base written to {args.out}")
        return 0

    if args.command == "run":
        kb = KnowledgeBase.from_json(args.knowledge)
        if not kb.chunks:
            kb = sample_knowledge()
        request = TaskRequest(
            description=args.task,
            task_id=args.task_id or uuid.uuid4().hex[:12],
            protocol=args.protocol,
            target_environment=args.target,
        )
        state = TrafficGenerationWorkflow(kb).run(request, args.out)
        success = bool(state.result and state.result.success and state.mapping)
        print(f"Task: {request.task_id}")
        print(f"Scenario: {state.plan.scenario_type if state.plan else 'none'}")
        print(f"Success: {success}")
        print(f"Output: {args.out}")
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
        manifests = prepare_rule_extraction_batches(
            args.input,
            args.out_dir,
            batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
        chunk_count = sum(manifest["chunk_count"] for manifest in manifests)
        print(f"Rule extraction batches: {len(manifests)}")
        print(f"Candidate chunks: {chunk_count}")
        print(f"Output dir: {args.out_dir}")
        print(f"Manifest: {args.out_dir / 'manifest.jsonl'}")
        return 0

    if args.command == "merge-rule-extraction-results":
        rules = merge_rule_extraction_results(args.candidates, args.results_dir, args.out)
        print(f"Rule chunks: {len(rules)}")
        print(f"Output: {args.out}")
        return 0

    return 1
