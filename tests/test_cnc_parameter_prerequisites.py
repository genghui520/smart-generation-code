from __future__ import annotations

import unittest

from smart_traffic_agent.api_compatibility import f0id_api_names
from smart_traffic_agent.cnc_parameter_prerequisites import extract_f0id_parameter_prerequisites
from smart_traffic_agent.traffic_graph import InMemoryTrafficGraphStore, TrafficRelationshipService
from smart_traffic_agent.traffic_relationship_mining import seed_cnc_parameter_prerequisite_relations


class CncParameterPrerequisiteTests(unittest.TestCase):
    def test_f0id_8132_condition_exposes_read_write_readback_without_authorizing_write(self) -> None:
        scope = f0id_api_names()
        report = extract_f0id_parameter_prerequisites(
            ["cnc_rdgrpid"], allowed_api_names=scope, device_scope="fanuc:0i-D"
        )
        requires = [
            item for item in report["relations"]
            if item["relation_type"] == "REQUIRES"
            and item["target_api"] == "cnc_rdgrpid"
            and item["attributes"]["parameter_number"] == 8132
        ]
        self.assertEqual(len(requires), 1)
        attrs = requires[0]["attributes"]
        self.assertEqual(attrs["bit"], 0)
        self.assertEqual(attrs["expected_value"], 1)
        self.assertEqual(attrs["value_domain"], [0, 1])
        self.assertEqual(attrs["read_api"], "cnc_rdparam")
        self.assertEqual(attrs["write_api"], "cnc_wrparam")
        self.assertFalse(attrs["auto_execute"])
        self.assertTrue(attrs["requires_read_snapshot"])
        self.assertTrue(any(item["relation_type"] == "PROVIDES_INPUT" for item in report["relations"]))
        self.assertTrue(any(item["relation_type"] == "READS_AFTER_WRITE" for item in report["relations"]))

    def test_controller_specific_condition_is_not_promoted_when_scope_is_unresolved(self) -> None:
        scope = f0id_api_names()
        report = extract_f0id_parameter_prerequisites(
            ["cnc_rdomhisno"], allowed_api_names=scope, device_scope="fanuc:0i-D"
        )
        self.assertFalse(any(
            item["relation_type"] == "REQUIRES"
            and item["target_api"] == "cnc_rdomhisno"
            and item["attributes"]["parameter_number"] == 3196
            for item in report["relations"]
        ))
        self.assertTrue(any(item["target_api"] == "cnc_rdomhisno" for item in report["candidates"]))

    def test_seed_and_exact_graph_query_are_idempotent(self) -> None:
        service = TrafficRelationshipService(InMemoryTrafficGraphStore())
        allowed = ("cnc_rdgrpid", "cnc_rdparam", "cnc_wrparam")
        seed_cnc_parameter_prerequisite_relations(
            service, ["cnc_rdgrpid"], allowed_api_names=allowed, device_scope="fanuc:0i-D"
        )
        first = len(service.store.relations)
        seed_cnc_parameter_prerequisite_relations(
            service, ["cnc_rdgrpid"], allowed_api_names=allowed, device_scope="fanuc:0i-D"
        )
        self.assertEqual(first, len(service.store.relations))
        query = service.call(
            "get_parameter_prerequisites", "cnc_rdgrpid",
            device_scope="fanuc:0i-D", allowed_api_names=allowed,
        )
        self.assertEqual(query["knowledge_status"], "known")
        self.assertEqual(len(query["prerequisites"]), 1)
        self.assertEqual(query["prerequisites"][0]["parameter"]["parameter_reference"], "8132#0")

    def test_return_code_parameter_evidence_adds_missing_page_condition(self) -> None:
        scope = f0id_api_names()
        report = extract_f0id_parameter_prerequisites(
            ["cnc_wrhissgnl3"], allowed_api_names=scope, device_scope="fanuc:0i-D"
        )
        refs = {
            item["attributes"]["parameter_reference"]
            for item in report["relations"]
            if item["relation_type"] == "REQUIRES"
        }
        self.assertIn("3195#6", refs)


if __name__ == "__main__":
    unittest.main()
