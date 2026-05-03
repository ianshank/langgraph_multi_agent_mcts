"""Producer-Reviewer topology: producer drafts, reviewer critiques, loop.

Implements the brief's "sequential thinking and deep reasoning" pattern. The
first agent produces a draft against the task; the second reviews using its
own ``run`` method (the reviewer's response is interpreted as accept/reject
text). On rejection, the producer is given a refined task containing the
reviewer's feedback in metadata, and the loop continues until acceptance or
the iteration cap is hit.
"""

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

_ACCEPT_MARKER = "ACCEPT"
_REJECT_MARKER = "REJECT"


def _is_accepted(text: str) -> bool:
    """A reviewer's response is an acceptance iff it contains ``ACCEPT``."""
    return _ACCEPT_MARKER in text.upper()


@dataclass
class ProducerReviewerTopology(BaseTopology):
    """Producer ↔ reviewer loop with a hard iteration cap."""

    name: str = "producer_reviewer"
    max_rounds: int = 3

    async def run(
        self,
        task: Task,
        agents: Sequence[AgentLike],
        *,
        policy: AggregationPolicy = AggregationPolicy.VERIFIER_RANKED,
    ) -> AgentOutcome:
        outcomes: list[AgentOutcome]
        if len(agents) < 2:
            # Degenerate case: not enough agents → run as fan-out.
            outcomes = await self._run_all_parallel(agents, task)
            return aggregate(outcomes, policy)
        producer, reviewer, *rest = agents
        outcomes = []
        current_task = task
        for round_index in range(self.max_rounds):
            draft = await self._run_one(producer, current_task)
            outcomes.append(draft)
            if not draft.success:
                break
            review_task = Task(
                id=current_task.id,
                goal=f"Review the following draft against the task. Reply with ACCEPT or REJECT plus feedback.\n\nDRAFT:\n{draft.response}",
                acceptance_criteria=current_task.acceptance_criteria,
                constraints=current_task.constraints,
                metadata={**current_task.metadata, "round": round_index, "draft": draft.response},
                raw=current_task.raw,
            )
            review = await self._run_one(reviewer, review_task)
            outcomes.append(review)
            if not review.success:
                break
            if _is_accepted(review.response):
                accepted_score = max(draft.confidence, review.confidence)
                return aggregate(outcomes, policy, verifier_score=accepted_score)
            # Rejected → feed feedback back into the producer's next attempt.
            current_task = Task(
                id=current_task.id,
                goal=current_task.goal,
                acceptance_criteria=current_task.acceptance_criteria,
                constraints=current_task.constraints,
                metadata={
                    **current_task.metadata,
                    "previous_draft": draft.response,
                    "review_feedback": review.response,
                    "round": round_index + 1,
                },
                raw=current_task.raw,
            )
        # Optional rest agents are unused in this baseline; they could be
        # consulted in a future enhancement but are present so callers can
        # supply a richer pool without breaking the signature.
        del rest
        return aggregate(outcomes, policy)


__all__ = ["ProducerReviewerTopology"]
