from pathlib import Path

from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.models import KnowledgeChunk
from smart_traffic_agent.rag.build_heidenhain_rag_index import api_chunks, parse_headers, rule_chunks


def test_official_lsv2_headers_are_the_manifest_source():
    rows = parse_headers()
    names = {row["function"] for row in rows}
    assert len(rows) > 50
    assert "LSV2ReceiveVersions" in names
    versions = next(row for row in rows if row["function"] == "LSV2ReceiveVersions")
    assert versions["category"] == "basic"
    assert versions["runtime_mapping"] == "versions"
    assert versions["official_compatibility"] == "general_library_function_verify_on_tnc640"

    scope_itnc = next(row for row in rows if row["function"] == "LSV2SelectChannelsITNC")
    assert scope_itnc["official_compatibility"] == "itnc530_only"

    scope_nck = next(row for row in rows if row["function"] == "LSV2SelectChannel")
    assert scope_nck["official_compatibility"] == "nck_based_including_tnc640"
    assert sum(bool(row["tnc640_collectable"]) for row in rows) == 199
    assert not next(row for row in rows if row["function"] == "LSV2SelectChannelsITNC")["tnc640_collectable"]


def test_api_chunks_support_exact_official_function_lookup():
    chunks = api_chunks(parse_headers())
    kb = KnowledgeBase([KnowledgeChunk(row["chunk_id"], row["text"], {
        "source_type": "api", "function": row["function"], "chunk_type": row["chunk_type"],
    }) for row in chunks])
    results = kb.lookup_api_exact(["LSV2ReceiveDirInfo"])
    assert results
    assert all(item.chunk.metadata["function"] == "LSV2ReceiveDirInfo" for item in results)


def test_write_and_control_rules_are_explicitly_separate_from_read_only_rules():
    rules = rule_chunks()
    write_rule = next(row for row in rules if row["rule_type"] == "write_control")
    assert "LSV2Set" in write_rule["rule_text"]
    assert any(row["rule_type"] == "read_only" for row in rules)


def test_heidenhain_store_does_not_default_to_focas_collection(tmp_path: Path):
    kb = KnowledgeBase(vector_db_dir=tmp_path / "missing", collection_name="heidenhain_lsv2_knowledge")
    assert kb.collection_name == "heidenhain_lsv2_knowledge"
    assert kb.status["vector_db"]["status"] == "missing"
