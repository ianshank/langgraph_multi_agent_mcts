"""Hierarchical topology: nested topologies for multi-level decomposition.

Composes a parent topology over groups of agents, where each group itself
runs through a child topology (e.g. parent=pipeline of fan_out_in groups).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.framework.harness.topology.base import (
    AgentLike,
    AgentOutcome,
    BaseTopology,
    aggregate,
)
from src.framework.harness.topology.fan_out_in import FanOutInTopology
from src.framework.harness.topology.pipeline import PipelineTopology


@dataclass
class HierarchicalTopology(BaseTopology):
    """Nest a child topology under a parent topology.

    Default: parent=pipeline, child=fan_out_in. Both are configurable so
    callers can compose ``pipeline(producer_reviewer)``,
    ``fan_out_in(expert_pool)``, etc.
    """

    name: str = "hierarchical"
    parent: BaseTopology = field(default_factory=PipelineTopology)
    child: BaseTopology = field(default_factory=FanOutInTopology)
    group_size: int = 2

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.VERIFIER_RANKED,
    ) -> AgentOutcome:
        if not agents:
            return aggregate([], policy)
        groups = [agents[i : i + self.group_size] for i in range(0, len(agents), self.group_size)]
        # Wrap each group as a synthetic agent that delegates to the child topology.
        wrapped: list[AgentLike] = [_GroupAgent(group=g, child=self.child) for g in groups]
        return await self.parent.run(task, wrapped, policy=policy)


@dataclass
class _GroupAgent:
    """Adapter: a sequence of agents driven by a child topology, exposed as one agent."""

    group: Sequence[AgentLike]
    child: BaseTopology
    name: str = "group"

    async def run(self, task: Task) -> AgentOutcome:
        outcome = await self.child.run(task, self.group)
        return outcome


__all__ = ["HierarchicalTopology"]
