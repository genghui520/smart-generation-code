from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.models import KnowledgeChunk, RetrievedChunk
from smart_traffic_agent.agents.planner import compact_rag_context


def test_exact_api_lookup_returns_all_chunks_for_only_exact_function_names():
    kb = KnowledgeBase(chunks=[
        KnowledgeChunk("abs-prototype", "prototype", {"function": "cnc_absolute", "chunk_type": "prototype"}),
        KnowledgeChunk("abs-arguments", "arguments", {"function": "cnc_absolute", "chunk_type": "arguments"}),
        KnowledgeChunk("abs2-prototype", "different API", {"function": "cnc_absolute2", "chunk_type": "prototype"}),
    ])

    results = kb.lookup_api_exact(["CNC_ABSOLUTE"])

    assert {row.chunk.chunk_id for row in results} == {"abs-prototype", "abs-arguments"}
    assert all(row.score == 1.0 for row in results)
    assert {row.chunk.metadata["chunk_type"] for row in results} == {"prototype", "arguments"}


def test_exact_api_lookup_can_select_specific_document_types():
    kb = KnowledgeBase(chunks=[
        KnowledgeChunk("proto", "prototype", {"function": "cnc_skip", "chunk_type": "prototype"}),
        KnowledgeChunk("options", "options", {"function": "cnc_skip", "chunk_type": "options"}),
    ])

    results = kb.lookup_api_exact(["cnc_skip"], chunk_types={"options"})

    assert [row.chunk.chunk_id for row in results] == ["options"]


def test_prerequisite_lookup_matches_exact_applies_to_names():
    kb = KnowledgeBase(chunks=[
        KnowledgeChunk("rule-a", "rule A", {
            "rule_type": "api_prerequisite", "applies_to": '["cnc_absolute"]',
        }),
        KnowledgeChunk("rule-b", "rule B", {
            "rule_type": "api_prerequisite", "applies_to": '["cnc_absolute2"]',
        }),
        KnowledgeChunk("other", "not a prerequisite", {
            "rule_type": "operation_rule", "applies_to": '["cnc_absolute"]',
        }),
    ])

    results = kb.lookup_api_prerequisites_exact(["cnc_absolute"])

    assert [row.chunk.chunk_id for row in results] == ["rule-a"]


def test_exact_api_contract_text_is_preserved_for_planner_prompt():
    contract_text = "long exact API contract " + ("argument detail " * 80)
    compact = compact_rag_context({
        "exact_api_contracts": [{
            "function": "cnc_absolute", "chunk_type": "arguments",
            "chunk_id": "exact-doc", "text": contract_text,
        }]
    })

    assert compact["exact_api_contracts"][0]["text"] == contract_text


def test_planner_context_compacts_raw_exact_retrieval_rows():
    chunk = RetrievedChunk(
        KnowledgeChunk("download3", "cnc_download3 prototype and arguments", {
            "function": "cnc_download3", "chunk_type": "arguments", "source_type": "api",
        }),
        score=1.0,
    )

    compact = compact_rag_context({"api": [chunk]})

    assert compact["api"] == [{
        "rule_type": None,
        "scenario": None,
        "function": "cnc_download3",
        "preview": "cnc_download3 prototype and arguments",
    }]
