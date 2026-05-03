"""Pipeline topology: each agent's response feeds the next as task metadata."""

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
class PipelineTopology(BaseTopology):
    """Sequential execution; output of N flows into N+1 via task metadata."""

    name: str = "pipeline"

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.ALL_MUST_PASS,
    ) -> AgentOutcome:
        outcomes: list[AgentOutcome] = []
        current_task = task
        for agent in agents:
            outcome = await self._run_one(agent, current_task)
            outcomes.append(outcome)
            if not outcome.success:
                # Halt on first failure; aggregator will reflect "not all passed".
                break
            current_task = Task(
                id=current_task.id,
                goal=current_task.goal,
                acceptance_criteria=current_task.acceptance_criteria,
                constraints=current_task.constraints,
                metadata={**current_task.metadata, "previous_response": outcome.response},
                raw=current_task.raw,
            )
        return aggregate(outcomes, policy)


__all__ = ["PipelineTopology"]
