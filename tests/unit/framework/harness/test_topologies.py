"""Unit tests for topology runners and aggregation policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.framework.harness.settings import AggregationPolicy
from src.framework.harness.state import Task
from src.framework.harness.topology import (
    AgentOutcome,
    ExpertPoolTopology,
    FanOutInTopology,
    HierarchicalTopology,
    PipelineTopology,
    ProducerReviewerTopology,
    SupervisorTopology,
    aggregate,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeAgent:
    name: str
    response: str = "ok"
    confidence: float = 0.5
    success: bool = True
    raises: bool = False
    inspector: list[Task] = field(default_factory=list)
    response_fn: Any = None  # optional callable: Task -> str

    async def run(self, task: Task) -> AgentOutcome:
        if self.raises:
            raise RuntimeError(f"{self.name} boom")
        self.inspector.append(task)
        text = self.response_fn(task) if self.response_fn else self.response
        return AgentOutcome(agent_name=self.name, response=text, confidence=self.confidence, success=self.success)


def _task(**overrides: Any) -> Task:
    base = {"id": "T", "goal": "G"}
    base.update(overrides)
    return Task(**base)


# ---------------------------------------------------------------------
# aggregate()
# ---------------------------------------------------------------------


def test_aggregate_first_success_picks_earliest_success() -> None:
    """``FIRST_SUCCESS`` returns the first successful outcome by order."""
    outcomes = [
        AgentOutcome(agent_name="a", response="x", success=False, error="e"),
        AgentOutcome(agent_name="b", response="y", success=True, confidence=0.4),
        AgentOutcome(agent_name="c", response="z", success=True, confidence=0.9),
    ]
    out = aggregate(outcomes, AggregationPolicy.FIRST_SUCCESS)
    assert out.agent_name == "b"


def test_aggregate_all_must_pass_fails_on_partial() -> None:
    """``ALL_MUST_PASS`` flags ``success=False`` if any contributor failed."""
    outcomes = [
        AgentOutcome(agent_name="a", response="x", success=True, confidence=0.5),
        AgentOutcome(agent_name="b", response="y", success=False, error="e"),
    ]
    out = aggregate(outcomes, AggregationPolicy.ALL_MUST_PASS)
    assert out.success is False


def test_aggregate_confidence_weighted_picks_highest() -> None:
    """``CONFIDENCE_WEIGHTED`` picks the highest-confidence success."""
    outcomes = [
        AgentOutcome(agent_name="a", response="x", success=True, confidence=0.3),
        AgentOutcome(agent_name="b", response="y", success=True, confidence=0.9),
        AgentOutcome(agent_name="c", response="z", success=True, confidence=0.5),
    ]
    out = aggregate(outcomes, AggregationPolicy.CONFIDENCE_WEIGHTED)
    assert out.agent_name == "b"


def test_aggregate_carries_intermediate_steps() -> None:
    """Aggregation surfaces every contributor in ``metadata['intermediate']``."""
    outcomes = [AgentOutcome(agent_name=f"a{i}", response=str(i), success=True, confidence=0.1 * i) for i in range(3)]
    out = aggregate(outcomes, AggregationPolicy.VERIFIER_RANKED)
    assert len(out.metadata["intermediate"]) == 3


def test_aggregate_empty_returns_failure() -> None:
    """No outcomes → a single failing aggregate."""
    out = aggregate([], AggregationPolicy.VERIFIER_RANKED)
    assert out.success is False


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_threads_response_into_next_task() -> None:
    """Each agent's response shows up in the next task's ``previous_response``."""
    a = FakeAgent(name="a", response="from-a")
    b = FakeAgent(name="b", response="from-b")
    topo = PipelineTopology()
    out = await topo.run(_task(), [a, b])
    assert b.inspector[0].metadata.get("previous_response") == "from-a"
    assert out.success


@pytest.mark.asyncio
async def test_pipeline_halts_on_first_failure() -> None:
    """A failing agent prevents the rest of the pipeline from running."""
    a = FakeAgent(name="a", success=False)
    b = FakeAgent(name="b")
    topo = PipelineTopology()
    out = await topo.run(_task(), [a, b])
    assert out.success is False
    assert b.inspector == []


# ---------------------------------------------------------------------
# Fan-out / Fan-in
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_in_runs_in_parallel() -> None:
    """Every agent receives the original task in fan-out."""
    agents = [FakeAgent(name=f"a{i}", confidence=0.1 * i, response=str(i)) for i in range(3)]
    topo = FanOutInTopology()
    out = await topo.run(_task(), agents)
    # Confidence-weighted default → highest confidence wins.
    assert out.response == "2"
    assert all(a.inspector for a in agents)


@pytest.mark.asyncio
async def test_fan_out_in_handles_exception() -> None:
    """An agent that raises becomes a failed outcome, not a crash."""
    agents = [FakeAgent(name="ok", confidence=0.4), FakeAgent(name="boom", raises=True)]
    topo = FanOutInTopology()
    out = await topo.run(_task(), agents)
    intermediate = out.metadata["intermediate"]
    assert len(intermediate) == 2


# ---------------------------------------------------------------------
# Expert pool
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expert_pool_routes_to_chosen_agent() -> None:
    """The router determines which agent runs."""
    a = FakeAgent(name="a")
    b = FakeAgent(name="b")
    topo = ExpertPoolTopology(router=lambda task, agents: 1)
    out = await topo.run(_task(), [a, b])
    assert b.inspector and not a.inspector
    assert out.agent_name == "b"


# ---------------------------------------------------------------------
# Producer-reviewer
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_producer_reviewer_accepts_immediately() -> None:
    """A reviewer that says ACCEPT halts after one round."""
    producer = FakeAgent(name="producer", response="draft v1")
    reviewer = FakeAgent(name="reviewer", response="ACCEPT looks good")
    topo = ProducerReviewerTopology(max_rounds=3)
    out = await topo.run(_task(), [producer, reviewer])
    assert out.success
    # One draft, one review → two intermediate entries.
    assert len(out.metadata["intermediate"]) == 2


@pytest.mark.asyncio
async def test_producer_reviewer_loops_until_cap() -> None:
    """A reviewer that always rejects → loop runs ``max_rounds`` times."""
    producer = FakeAgent(name="producer", response="draft")
    reviewer = FakeAgent(name="reviewer", response="REJECT not good enough")
    topo = ProducerReviewerTopology(max_rounds=2)
    await topo.run(_task(), [producer, reviewer])
    assert len(producer.inspector) == 2  # one per round


# ---------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_dispatches_then_finishes() -> None:
    """The supervisor delegates to a worker and then declares DONE."""
    supervisor_responses = ["DELEGATE: 0 do the thing", "DONE"]

    counter = {"i": 0}

    def supervisor_fn(task: Task) -> str:
        idx = counter["i"]
        counter["i"] += 1
        return supervisor_responses[idx]

    sup = FakeAgent(name="sup", response_fn=supervisor_fn)
    worker = FakeAgent(name="worker", response="worker did it")
    topo = SupervisorTopology(max_rounds=3)
    out = await topo.run(_task(), [sup, worker])
    assert worker.inspector  # delegated to
    assert out.success


# ---------------------------------------------------------------------
# Hierarchical
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hierarchical_pipeline_of_fan_out_in() -> None:
    """Default hierarchical topology composes pipeline-of-fan-out-in."""
    agents = [FakeAgent(name=f"a{i}", confidence=0.1 * (i + 1)) for i in range(4)]
    topo = HierarchicalTopology(group_size=2)
    out = await topo.run(_task(), agents)
    # Each agent runs at least once (within one of the two groups).
    assert all(a.inspector for a in agents)
    assert "intermediate" in out.metadata
