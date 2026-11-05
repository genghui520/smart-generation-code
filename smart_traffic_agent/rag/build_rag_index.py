"""
Build FOCAS RAG vector knowledge base.

Input:
  - rag_indexes/focas/chunks.jsonl        (API function reference)
  - rag_indexes/focas/rule_chunks.jsonl   (traffic generation rules)
  - rag_indexes/focas/scenario_chunks.jsonl (scenario-centered organization)

Output:
  - rag_indexes/focas/vector_db/           (Chroma persistent DB)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# -- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "chunks.jsonl"
RETURN_CODE_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "return_code_chunks.jsonl"
RULE_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
SCENARIO_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "scenario_chunks.jsonl"
ALARM_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "alarm_chunks.jsonl"
VECTOR_DB_DIR = PROJECT_ROOT / "rag_indexes" / "focas" / "vector_db_pdf"

# -- Embedding model -----------------------------------------------------
# Chinese + English compatible lightweight model (33 MB)
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


def resolve_embedding_model(model_name: str) -> str:
    """Use an already cached model when the runtime has no network access."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots = cache_root / "hub" / safe_name / "snapshots"
    if snapshots.is_dir():
        candidates = [p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").is_file()]
        if candidates:
            return str(max(candidates, key=lambda p: p.stat().st_mtime))
    return model_name


class BGEEmbeddingFunction(EmbeddingFunction):
    """Wrap sentence-transformers as a Chroma EmbeddingFunction."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        resolved_model = resolve_embedding_model(model_name)
        self.model = SentenceTransformer(
            resolved_model,
            trust_remote_code=True,
            local_files_only=resolved_model != model_name,
        )

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        return embeddings.tolist()


# -- Data loading --------------------------------------------------------


def load_api_chunks(path: Path) -> list[dict[str, Any]]:
    """Load API function reference chunks (JSONL)."""
    return load_jsonl(path)


def load_rule_chunks(path: Path) -> list[dict[str, Any]]:
    """Load final traffic generation rules (JSONL)."""
    rules = load_jsonl(path)
    # Keep source-specific manual conversions separate from the large generated
    # rule file while including them in every rebuilt knowledge index.
    companion = path.with_name("f0id_manual_rules.jsonl")
    if companion.is_file() and companion.resolve() != path.resolve():
        rules.extend(load_jsonl(companion))
    return rules


def load_scenario_chunks(path: Path) -> list[dict[str, Any]]:
    """Load scenario-centered organization chunks (JSONL)."""
    if not path.exists():
        return []
    return load_jsonl(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -- Convert to Chroma documents -----------------------------------------


def api_chunk_to_docs(
    chunks: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """API function chunks -> Chroma (ids, texts, metadatas)."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for ch in chunks:
        cid = ch.get("chunk_id", "")
        text = ch.get("text", "")
        if not text:
            continue

        function = ch.get("function", "")
        category = ch.get("category", "")
        chunk_type = ch.get("chunk_type", "")
        source_url = ch.get("source_url", "")

        search_text = text
        if function:
            search_text = f"Function: {function}\nCategory: {category}\n{text}"

        ids.append(f"api-{cid}")
        texts.append(search_text)
        metadatas.append(
            {
                "source_type": "api",
                "chunk_id": cid,
                "function": function,
                "category": category,
                "chunk_type": chunk_type,
                "source_url": source_url,
            }
        )

    return ids, texts, metadatas


def rule_to_docs(
    rules: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """Rules -> Chroma (ids, texts, metadatas)."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for idx, rule in enumerate(rules):
        source_chunk_id = rule.get("source_chunk_id", f"rule-{idx:05d}")
        rule_id = rule.get("rule_id", f"rule-{idx:05d}")
        rule_type = rule.get("rule_type", "")
        scenario = rule.get("scenario", "")
        rule_text = rule.get("rule_text", "")
        nc_req = rule.get("nc_program_requirements", [])
        ops = rule.get("operation_sequence", [])
        timing = rule.get("collection_timing", [])
        apis = rule.get("recommended_api_functions", [])
        signals = rule.get("distinguishing_signals", [])
        checks = rule.get("quality_checks", [])
        applies_to = rule.get("applies_to", [])
        conditions = rule.get("conditions", [])
        preparations = rule.get("preparation_operations", [])

        if not rule_text:
            continue

        parts = [
            f"Scenario: {scenario}",
            f"Type: {rule_type}",
            rule_text,
        ]
        if nc_req:
            parts.append("Requirements: " + "; ".join(nc_req))
        if ops:
            parts.append("Operation: " + "; ".join(ops[:3]))
        if timing:
            parts.append("Timing: " + "; ".join(timing[:3]))
        if apis:
            parts.append("APIs: " + "; ".join(apis[:8]))
        if signals:
            parts.append("Signals: " + "; ".join(signals[:3]))
        if checks:
            parts.append("Quality checks: " + "; ".join(checks[:3]))
        if applies_to:
            parts.append("Applies to APIs: " + "; ".join(str(item) for item in applies_to))
        if conditions:
            parts.append("Prerequisite conditions: " + json.dumps(conditions, ensure_ascii=False))
        if preparations:
            parts.append("Preparation operations: " + json.dumps(preparations, ensure_ascii=False))
        search_text = "\n".join(parts)

        uid = f"rule-{rule_id}"
        ids.append(uid)
        texts.append(search_text)
        metadatas.append(
            {
                "source_type": "rule",
                "rule_id": rule_id,
                "rule_type": rule_type,
                "scenario": scenario,
                "source_chunk_id": source_chunk_id,
                "source_file": rule.get("source_file", ""),
                "source_url": rule.get("source_url", ""),
                "page_start": rule.get("page_start", -1) or -1,
                "page_end": rule.get("page_end", -1) or -1,
                "applies_to": json.dumps(rule.get("applies_to", []), ensure_ascii=False),
                "conditions": json.dumps(rule.get("conditions", []), ensure_ascii=False),
                "preparation_operations": json.dumps(rule.get("preparation_operations", []), ensure_ascii=False),
                "verification_operations": json.dumps(rule.get("verification_operations", []), ensure_ascii=False),
                "restart_required": bool(rule.get("restart_required", False)),
                "safe_to_write_in_isolated_simulator": bool(rule.get("safe_to_write_in_isolated_simulator", False)),
            }
        )

    return ids, texts, metadatas


def scenario_chunk_to_docs(
    chunks: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """Scenario-centered chunks -> Chroma (ids, texts, metadatas)."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        scenario = chunk.get("scenario", "")
        if not chunk_id or not text:
            continue
        ids.append(f"scenario-{chunk_id}")
        texts.append(text)
        metadatas.append(
            {
                "source_type": "scenario",
                "chunk_id": chunk_id,
                "knowledge_type": chunk.get("knowledge_type", "scenario_centered_knowledge"),
                "scenario": scenario,
                "rule_type": "scenario_organization",
            }
        )

    return ids, texts, metadatas


def alarm_chunk_to_docs(
    chunks: list[dict],
) -> tuple[list[str], list[str], list[dict]]:
    """Alarm knowledge chunks -> Chroma documents."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for index, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"alarm-{index:05d}")
        text = chunk.get("text", "").strip()
        if not text:
            continue
        ids.append(f"alarm-{chunk_id}")
        texts.append(text)
        metadatas.append({
            "source_type": "alarm",
            "chunk_id": chunk_id,
            "alarm_code": str(chunk.get("alarm_code", "")),
            "alarm_prefix": (re.match(r"[A-Za-z]+", str(chunk.get("alarm_code", ""))) or [""])[0].upper(),
            "alarm_bit": int(chunk.get("alarm_bit", -1)),
            "alarm_category": chunk.get("alarm_category", ""),
            "controller": chunk.get("controller", ""),
            "action_class": chunk.get("action_class", "manual_required"),
            "automatic_action": chunk.get("automatic_action", "none"),
            "source_url": chunk.get("source_url", ""),
            "source_file": chunk.get("source_file", ""),
            "page": int(chunk.get("page", chunk.get("source_page", -1)) or -1),
            "source_page": int(chunk.get("source_page", chunk.get("page", -1)) or -1),
        })
    return ids, texts, metadatas


# -- Main builder --------------------------------------------------------


def reset_vector_db(path: Path) -> None:
    """Remove the old persistent DB before rebuilding it."""
    resolved = path.resolve()
    allowed_parent = (PROJECT_ROOT / "rag_indexes" / "focas").resolve()
    if allowed_parent not in resolved.parents:
        raise ValueError(f"Refusing to delete unexpected vector DB path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def build(
    *,
    api_chunks_path: Path = API_CHUNKS_PATH,
    return_code_chunks_path: Path = RETURN_CODE_CHUNKS_PATH,
    rule_chunks_path: Path = RULE_CHUNKS_PATH,
    scenario_chunks_path: Path = SCENARIO_CHUNKS_PATH,
    alarm_chunks_path: Path = ALARM_CHUNKS_PATH,
    vector_db_dir: Path = VECTOR_DB_DIR,
    reset: bool = True,
) -> int:
    title = "Build FOCAS RAG Knowledge Base"
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading API function chunks...")
    api_chunks = load_api_chunks(api_chunks_path)
    if return_code_chunks_path.is_file():
        api_chunks.extend(load_jsonl(return_code_chunks_path))
    print(f"  -> {len(api_chunks)} items")

    print("\n[2/4] Loading rule chunks...")
    rules = load_rule_chunks(rule_chunks_path)
    print(f"  -> {len(rules)} items")

    # 2. Convert to documents
    print("\n[3/4] Loading scenario-centered knowledge...")
    scenario_chunks = load_scenario_chunks(scenario_chunks_path)
    print(f"  -> {len(scenario_chunks)} items")
    alarm_chunks = load_jsonl(alarm_chunks_path) if alarm_chunks_path.is_file() else []
    print(f"  -> Alarm knowledge: {len(alarm_chunks)} items")

    print("\n[4/5] Converting to Chroma documents...")
    api_ids, api_texts, api_metas = api_chunk_to_docs(api_chunks)
    rule_ids, rule_texts, rule_metas = rule_to_docs(rules)
    scenario_ids, scenario_texts, scenario_metas = scenario_chunk_to_docs(scenario_chunks)
    alarm_ids, alarm_texts, alarm_metas = alarm_chunk_to_docs(alarm_chunks)
    n_api = len(api_ids)
    n_rule = len(rule_ids)
    n_scenario = len(scenario_ids)
    n_alarm = len(alarm_ids)
    print(f"  -> API functions: {n_api} docs")
    print(f"  -> Rules: {n_rule} docs")
    print(f"  -> Scenario organizations: {n_scenario} docs")
    print(f"  -> Alarm knowledge: {n_alarm} docs")
    print(f"  -> Total: {n_api + n_rule + n_scenario + n_alarm} docs")

    # 3. Write to Chroma
    print("\n[5/5] Writing to Chroma vector store...")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dir: {vector_db_dir}")
    if reset:
        reset_vector_db(vector_db_dir)

    embedding_fn = BGEEmbeddingFunction(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(vector_db_dir))

    collection = client.create_collection(
        name="focas_knowledge",
        embedding_function=embedding_fn,
        metadata={"description": "FOCAS API + Traffic Generation Rules + Scenario Organization"},
    )

    BATCH = 100
    all_ids = api_ids + rule_ids + scenario_ids + alarm_ids
    all_texts = api_texts + rule_texts + scenario_texts + alarm_texts
    all_metas = api_metas + rule_metas + scenario_metas + alarm_metas

    for start in range(0, len(all_ids), BATCH):
        end = min(start + BATCH, len(all_ids))
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            metadatas=all_metas[start:end],
        )
        print(f"  [OK] {end}/{len(all_ids)}")

    # Stable read-only companion index. Planner queries this matrix directly,
    # avoiding Chroma's background HNSW compactor on Windows.
    stable_vectors: list[list[float]] = []
    for start in range(0, len(all_texts), BATCH):
        end = min(start + BATCH, len(all_texts))
        stable_vectors.extend(embedding_fn(all_texts[start:end]))
    np.save(vector_db_dir / "stable_embeddings.npy", np.asarray(stable_vectors, dtype=np.float32))
    with (vector_db_dir / "stable_records.jsonl").open("w", encoding="utf-8") as file:
        for doc_id, text, metadata in zip(all_ids, all_texts, all_metas):
            file.write(json.dumps({"id": doc_id, "text": text, "metadata": metadata}, ensure_ascii=False) + "\n")
    print(f"  Stable index: {len(stable_vectors)} vectors")

    print("\n" + "=" * 60)
    print("Build complete!")
    print(f"  Collection: focas_knowledge")
    print(f"  Documents: {collection.count()}")
    print(f"  Location: {vector_db_dir}")

    # Test query
    print("\n" + "-" * 60)
    q = "read spindle speed and control with G96 constant surface speed"
    print(f"Test query: '{q}'")
    results = collection.query(query_texts=[q], n_results=3)
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        print(f"\n  #{i + 1} (distance: {dist:.4f})")
        print(f"  type: {meta['source_type']}")
        if meta["source_type"] == "rule":
            print(
                f"  scenario: {meta['scenario']} | "
                f"rule_type: {meta['rule_type']}"
            )
        else:
            print(f"  function: {meta.get('function', '')}")
        preview = doc[:120].replace("\n", " ")
        print(f"  text: {preview}...")
    print("-" * 60)

    # Chroma 1.5 runs the local HNSW writer/compactor in the background.
    # Close it explicitly so every segment is flushed before another process
    # opens the persistent directory (especially important on Windows).
    client.close()

    return len(all_ids)


if __name__ == "__main__":
    build()
