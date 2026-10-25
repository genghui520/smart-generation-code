import json
from pathlib import Path

from smart_traffic_agent.rag.build_rag_index import alarm_chunk_to_docs
from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.models import KnowledgeChunk


def test_alarm_chunks_are_structured_and_filterable():
    path = Path("rag_indexes/focas/alarm_chunks.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, texts, metadata = alarm_chunk_to_docs(rows)
    assert len(ids) == len(rows) >= 7
    assert all(item["source_type"] == "alarm" for item in metadata)
    assert any(item["alarm_code"] == "SW0100" for item in metadata)
    assert any(item["alarm_prefix"] == "SW" for item in metadata)
    assert all(item["source_page"] > 0 for item in metadata)
    assert all("F0-iD参数说明手册.pdf" in item["source_file"] for item in metadata)


def test_alarm_search_filters_term_fallback():
    rows = [json.loads(line) for line in Path("rag_indexes/focas/alarm_chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _, texts, metadata = alarm_chunk_to_docs(rows)
    chunks = [
        {"chunk_id": meta["chunk_id"], "text": text, "metadata": meta}
        for text, meta in zip(texts, metadata)
    ]
    kb = KnowledgeBase([KnowledgeChunk(**item) for item in chunks])
    results = kb.search_alarms("SW 参数写入开关", top_k=3)
    assert results
    assert all(item.chunk.metadata["source_type"] == "alarm" for item in results)
