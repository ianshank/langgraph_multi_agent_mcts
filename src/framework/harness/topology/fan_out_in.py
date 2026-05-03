"""Fan-out / fan-in topology: parallel branches with aggregated result."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.framework.harness.topology.base import (
    AgentLike,
    AgentOutcome,
    BaseTopology,
    aggregate,
)


@dataclass
class FanOutInTopology(BaseTopology):
    """Run every agent against the same task in parallel, then aggregate."""

    name: str = "fan_out_in"

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.CONFIDENCE_WEIGHTED,
    ) -> AgentOutcome:
        outcomes = await self._run_all_parallel(agents, task)
        return aggregate(outcomes, policy)


__all__ = ["FanOutInTopology"]
