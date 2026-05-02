"""Expert pool topology: a router selects one agent per task."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.framework.harness.topology.base import (
    AgentLike,
    AgentOutcome,
    BaseTopology,
    aggregate,
)


@dataclass
class ExpertPoolTopology(BaseTopology):
    """Route the task to a single specialised agent.

    The ``router`` callable inspects the task and returns the *index* of the
    chosen agent. A default round-robin router is provided for testing.
    """

    name: str = "expert_pool"
    router: Callable[[Task, Sequence[AgentLike]], int] = field(
        default=lambda task, agents: hash(task.id) % len(agents) if agents else 0
    )

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.FIRST_SUCCESS,
    ) -> AgentOutcome:
        if not agents:
            return aggregate([], policy)
        index = self.router(task, agents) % len(agents)
        chosen = agents[index]
        outcome = await self._run_one(chosen, task)
        return aggregate([outcome], policy)


__all__ = ["ExpertPoolTopology"]
