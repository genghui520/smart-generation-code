"""Domain-independent Role/Action/Message contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    input_artifact: str
    output_artifact: str


@dataclass(slots=True)
class AgentMessage:
    sender: str
    action: str
    artifact_type: str
    content: dict[str, Any] = field(default_factory=dict)


class Role:
    role_name = "Role"

    def __init__(self) -> None:
        self.actions: dict[str, ActionSpec] = {}

    def set_actions(self, *actions: ActionSpec) -> None:
        self.actions = {action.name: action for action in actions}

    def publish(self, action: str, artifact_type: str, **content: Any) -> AgentMessage:
        if action not in self.actions:
            raise ValueError(f"{self.role_name} cannot publish unknown action: {action}")
        contract = self.actions[action]
        if artifact_type != contract.output_artifact:
            raise ValueError(f"{self.role_name}.{action} must publish {contract.output_artifact}, not {artifact_type}")
        return AgentMessage(self.role_name, action, artifact_type, content)


class AgentTemplate(Role):
    """Reusable MetaGPT-style Agent base independent of any domain."""

    profile = ""
    goal = ""
    constraints: tuple[str, ...] = ()

    def __init__(self, *, profile: str | None = None, goal: str | None = None, constraints: tuple[str, ...] | None = None) -> None:
        super().__init__()
        if profile is not None:
            self.profile = profile
        if goal is not None:
            self.goal = goal
        if constraints is not None:
            self.constraints = tuple(constraints)
        self.environment: Any | None = None

    def register_actions(self, *actions: ActionSpec) -> None:
        self.set_actions(*actions)

    def action(self, name: str) -> ActionSpec:
        try:
            return self.actions[name]
        except KeyError as exc:
            raise ValueError(f"{self.role_name} has no registered action: {name}") from exc

    def describe(self) -> dict[str, Any]:
        return {"role": self.role_name, "profile": self.profile, "goal": self.goal,
                "constraints": list(self.constraints), "actions": [asdict(item) for item in self.actions.values()]}


def message_to_dict(message: AgentMessage, artifact: Any | None = None) -> dict[str, Any]:
    result = {"sender": message.sender, "action": message.action, "artifact_type": message.artifact_type, "content": message.content}
    if artifact is not None:
        result["artifact"] = asdict(artifact) if hasattr(artifact, "__dataclass_fields__") else artifact
    return result
