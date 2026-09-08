"""Explicit agent graph builder used by the workflow orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Type

from langgraph.graph import StateGraph

from .agent_roles import AgentTemplate


@dataclass(frozen=True, slots=True)
class GraphNodeSpec:
    name: str
    agent: str


@dataclass
class AgentGraph:
    """Small declarative layer between AgentTemplate roles and LangGraph."""

    state_type: Type[Any]
    _builder: StateGraph = field(init=False, repr=False)
    _nodes: dict[str, GraphNodeSpec] = field(default_factory=dict, init=False)
    _edges: list[tuple[str, str]] = field(default_factory=list, init=False)
    _conditional_edges: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._builder = StateGraph(self.state_type)

    def add_agent_node(
        self,
        name: str,
        agent: AgentTemplate,
        handler: Callable[[Any], Any],
    ) -> None:
        if name in self._nodes:
            raise ValueError(f"Agent graph node already registered: {name}")
        self._nodes[name] = GraphNodeSpec(name=name, agent=agent.role_name)
        self._builder.add_node(name, handler)

    def set_entry_point(self, name: str) -> None:
        self._builder.set_entry_point(name)

    def add_edge(self, source: str, target: str) -> None:
        self._edges.append((source, target))
        self._builder.add_edge(source, target)

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[Any], str],
        destinations: Mapping[str, str],
    ) -> None:
        self._conditional_edges.append(
            {"source": source, "destinations": dict(destinations)}
        )
        self._builder.add_conditional_edges(source, router, dict(destinations))

    def compile(self):
        return self._builder.compile()

    def describe(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"name": node.name, "agent": node.agent}
                for node in self._nodes.values()
            ],
            "edges": [
                {"source": source, "target": target}
                for source, target in self._edges
            ],
            "conditional_edges": list(self._conditional_edges),
        }
