from __future__ import annotations

import unittest

from smart_traffic_agent.agent_tools.traffic_graph import (
    ProposeRelationInput,
    ProposeTrafficRelationTool,
)
from smart_traffic_agent.traffic_graph import InMemoryTrafficGraphStore, TrafficRelationshipService
from smart_traffic_agent.traffic_graph import ApiNode


class TrafficGraphToolTests(unittest.TestCase):
    def test_proposal_tool_downgrades_unverified_status(self) -> None:
        store = InMemoryTrafficGraphStore()
        store.upsert_api(ApiNode("focas:a", "a"))
        store.upsert_api(ApiNode("focas:b", "b"))
        output = ProposeTrafficRelationTool(TrafficRelationshipService(store)).invoke(ProposeRelationInput({
            "source_api": "a", "target_api": "b", "relation_type": "REQUIRES",
            "confidence_status": "runtime_verified",
        }))
        self.assertEqual(output["relation"]["confidence_status"], "proposed")


if __name__ == "__main__":
    unittest.main()
