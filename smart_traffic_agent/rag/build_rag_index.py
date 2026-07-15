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
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# -- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "chunks.jsonl"
RULE_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "rule_chunks.jsonl"
SCENARIO_CHUNKS_PATH = PROJECT_ROOT / "rag_indexes" / "focas" / "scenario_chunks.jsonl"
VECTOR_DB_DIR = PROJECT_ROOT / "rag_indexes" / "focas" / "vector_db"

# -- Embedding model -----------------------------------------------------
# Chinese + English compatible lightweight model (33 MB)
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class BGEEmbeddingFunction(EmbeddingFunction):
    """Wrap sentence-transformers as a Chroma EmbeddingFunction."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

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
    return load_jsonl(path)


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
                "page_start": rule.get("page_start", -1) or -1,
                "page_end": rule.get("page_end", -1) or -1,
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
    rule_chunks_path: Path = RULE_CHUNKS_PATH,
    scenario_chunks_path: Path = SCENARIO_CHUNKS_PATH,
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
    print(f"  -> {len(api_chunks)} items")

    print("\n[2/4] Loading rule chunks...")
    rules = load_rule_chunks(rule_chunks_path)
    print(f"  -> {len(rules)} items")

    # 2. Convert to documents
    print("\n[3/4] Loading scenario-centered knowledge...")
    scenario_chunks = load_scenario_chunks(scenario_chunks_path)
    print(f"  -> {len(scenario_chunks)} items")

    print("\n[4/5] Converting to Chroma documents...")
    api_ids, api_texts, api_metas = api_chunk_to_docs(api_chunks)
    rule_ids, rule_texts, rule_metas = rule_to_docs(rules)
    scenario_ids, scenario_texts, scenario_metas = scenario_chunk_to_docs(scenario_chunks)
    n_api = len(api_ids)
    n_rule = len(rule_ids)
    n_scenario = len(scenario_ids)
    print(f"  -> API functions: {n_api} docs")
    print(f"  -> Rules: {n_rule} docs")
    print(f"  -> Scenario organizations: {n_scenario} docs")
    print(f"  -> Total: {n_api + n_rule + n_scenario} docs")

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
    all_ids = api_ids + rule_ids + scenario_ids
    all_texts = api_texts + rule_texts + scenario_texts
    all_metas = api_metas + rule_metas + scenario_metas

    for start in range(0, len(all_ids), BATCH):
        end = min(start + BATCH, len(all_ids))
        collection.add(
            ids=all_ids[start:end],
            documents=all_texts[start:end],
            metadatas=all_metas[start:end],
        )
        print(f"  [OK] {end}/{len(all_ids)}")

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

    return len(all_ids)


if __name__ == "__main__":
    build()
