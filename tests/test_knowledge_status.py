from smart_traffic_agent.knowledge import KnowledgeBase
from smart_traffic_agent.traffic_graph import TrafficGraphError, TrafficRelationshipService


def test_vector_db_missing_is_explicit_and_local_fallback_is_distinguished():
    missing = "__definitely_missing_vector_db__"
    kb = KnowledgeBase(vector_db_dir=missing)
    assert kb.status["status"] == "unavailable"
    assert kb.status["vector_db"]["status"] == "missing"
    assert kb.status["fallback_available"] is False

    fallback = KnowledgeBase([])
    fallback.chunks = [object()]  # type: ignore[list-item]
    fallback.vector_db_path = missing
    fallback.vector_db_status = "missing"
    assert fallback.status["status"] == "fallback"
    assert fallback.status["fallback_available"] is True


def test_graph_unconfigured_and_connection_error_are_reported():
    service = TrafficRelationshipService(None)
    assert service.status["status"] == "unconfigured"
    assert service.call("get_api_info", "cnc_statinfo")["available"] is False

    class BrokenStore:
        def get_api_info(self, *_args, **_kwargs):
            raise TrafficGraphError("connection refused")

    broken = TrafficRelationshipService(BrokenStore())
    result = broken.call("get_api_info", "cnc_statinfo")
    assert result["available"] is False
    assert broken.status["status"] == "error"
    assert "connection refused" in broken.status["error"]
