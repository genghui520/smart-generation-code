from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from smart_traffic_agent.traffic_graph import (
    ApiNode,
    InMemoryTrafficGraphStore,
    TrafficRelation,
    TrafficRelationshipService,
)
from smart_traffic_agent.traffic_relationship_mining import seed_documented_relations
from smart_traffic_agent.agents.executor import ExecutionAgent
from smart_traffic_agent.models import ApiCallLog, ExecutionPlan, ExecutionResult, NcProgramSpec, PlanStep, TaskRequest, WorkflowState


class TrafficGraphTests(unittest.TestCase):
    def test_proposed_relation_never_validates_as_mandatory(self) -> None:
        store = InMemoryTrafficGraphStore()
        store.upsert_api(ApiNode("focas:source", "source"))
        store.upsert_api(ApiNode("focas:target", "target"))
        relation = TrafficRelation(
            "rel:test", "source", "target", "REQUIRES", confidence_status="proposed"
        )
        store.upsert_relation(relation)

        self.assertFalse(store.validate_traffic_chain("target")["valid"])
        verified = store.verify_relation("rel:test", [{"evidence_id": "run-1"}])
        self.assertEqual(verified["relation"]["confidence_status"], "runtime_verified")
        self.assertTrue(store.validate_traffic_chain("target")["valid"])

    def test_download_target_seeds_documented_start_and_end_chain(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_download3"])
        chain = service.call("get_traffic_chain", "cnc_download3", 3)

        names = {row["name"] for row in chain["nodes"]}
        pairs = {(row["source_api"], row["target_api"]) for row in chain["relations"]}
        self.assertTrue({"cnc_download3", "cnc_dwnstart3", "cnc_dwnend3"} <= names)
        self.assertIn(("cnc_dwnstart3", "cnc_download3"), pairs)
        self.assertIn(("cnc_download3", "cnc_dwnend3"), pairs)

    def test_upload_target_seeds_its_matching_session_only(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        seed_documented_relations(service, ["cnc_upload3"])
        chain = service.call("get_traffic_chain", "cnc_upload3", 3)
        names = {row["name"] for row in chain["nodes"]}
        pairs = {(row["source_api"], row["target_api"]) for row in chain["relations"]}
        self.assertTrue({"cnc_upload3", "cnc_upstart3", "cnc_upend3"} <= names)
        self.assertIn(("cnc_upstart3", "cnc_upload3"), pairs)
        self.assertIn(("cnc_upload3", "cnc_upend3"), pairs)

    def test_unconfigured_service_is_explicitly_unknown(self) -> None:
        result = TrafficRelationshipService(None).call("get_api_dependencies", "cnc_statinfo")
        self.assertFalse(result["available"])
        self.assertEqual(result["knowledge_status"], "unknown")

    def test_executor_persists_relationship_feedback_without_changing_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_id = "trg-feedback"
            state = WorkflowState(request=TaskRequest("read two APIs", task_id))
            state.plan = ExecutionPlan(
                "plan-trg", task_id, "parameter_read", "read", "simulator", "", [],
                NcProgramSpec(""),
                [
                    PlanStep("S1", "before", "read", "ReadStatus", {}, protocol_function="cnc_statinfo"),
                    PlanStep("S2", "during", "read", "ReadMacro", {}, protocol_function="cnc_rdmacro"),
                ], [],
            )
            state.result = ExecutionResult(
                task_id, True,
                [
                    ApiCallLog("t", task_id, "S1", "before", "ReadStatus", {}, 0, {}, protocol_function="cnc_statinfo"),
                    ApiCallLog("t", task_id, "S2", "during", "ReadMacro", {}, 0, {}, protocol_function="cnc_rdmacro"),
                ], [], Path(tmp),
            )
            store = InMemoryTrafficGraphStore()
            store.upsert_api(ApiNode("focas:cnc_statinfo", "cnc_statinfo"))
            store.upsert_api(ApiNode("focas:cnc_rdmacro", "cnc_rdmacro"))
            agent = ExecutionAgent()
            agent.traffic_relationships = TrafficRelationshipService(store)
            agent._record_relationship_feedback(state)

            self.assertTrue(state.result.success)
            self.assertEqual(state.plan.rag_context["traffic_relationship_feedback"]["status"], "recorded")
            self.assertTrue((Path(tmp) / "traffic_relationship_feedback.json").is_file())


if __name__ == "__main__":
    unittest.main()
