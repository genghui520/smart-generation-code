from __future__ import annotations

import unittest

from smart_traffic_agent.agent_roles import PlanSpec
from smart_traffic_agent.orchestration.roles import ActionSpec, AgentTemplate, Role, message_to_dict


class AgentRoleTests(unittest.TestCase):
    def test_role_publishes_typed_artifact_message(self) -> None:
        role = Role()
        role.set_actions(ActionSpec("write_plan", "KnowledgeSpec", "PlanSpec"))

        message = role.publish("write_plan", "PlanSpec", plan_id="p1")
        serialized = message_to_dict(message, PlanSpec("p1", 2, 3))

        self.assertEqual(serialized["artifact_type"], "PlanSpec")
        self.assertEqual(serialized["artifact"]["step_count"], 2)

    def test_role_rejects_unknown_action(self) -> None:
        role = Role()
        with self.assertRaises(ValueError):
            role.publish("unknown", "Artifact")

    def test_agent_template_exposes_role_contract_and_registered_actions(self) -> None:
        agent = AgentTemplate(
            profile="Planner",
            goal="Build a plan",
            constraints=("Use retrieved evidence",),
        )
        agent.role_name = "PlannerRole"
        agent.register_actions(ActionSpec("write_plan", "KnowledgeSpec", "PlanSpec"))

        description = agent.describe()

        self.assertEqual(description["role"], "PlannerRole")
        self.assertEqual(description["goal"], "Build a plan")
        self.assertEqual(description["constraints"], ["Use retrieved evidence"])
        self.assertEqual(agent.action("write_plan").output_artifact, "PlanSpec")

    def test_agent_template_publishes_only_registered_actions(self) -> None:
        agent = AgentTemplate()
        agent.role_name = "TestRole"
        agent.register_actions(ActionSpec("act", "Input", "Output"))

        message = agent.publish("act", "Output", value=1)

        self.assertEqual(message.sender, "TestRole")
        self.assertEqual(message.content["value"], 1)


if __name__ == "__main__":
    unittest.main()
