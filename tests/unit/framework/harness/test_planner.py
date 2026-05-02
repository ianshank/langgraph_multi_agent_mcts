"""Unit tests for the planner implementations."""

from __future__ import annotations

from typing import Any

import pytest

from src.adapters.llm.base import LLMResponse
from src.framework.harness.planner import HeuristicPlanner, LLMPlanner
from src.framework.harness.state import AcceptanceCriterion, Task

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_heuristic_planner_one_step_per_criterion() -> None:
    """Each acceptance criterion becomes a plan step."""
    task = Task(
        id="t1",
        goal="g",
        acceptance_criteria=(
            AcceptanceCriterion(id="c1", description="lints clean"),
            AcceptanceCriterion(id="c2", description="tests pass"),
        ),
    )
    plan = await HeuristicPlanner().plan(task)
    assert len(plan.steps) == 2


@pytest.mark.asyncio
async def test_heuristic_planner_falls_back_when_no_criteria() -> None:
    """No criteria → a single 'achieve goal' step."""
    plan = await HeuristicPlanner().plan(Task(id="t", goal="ship"))
    assert len(plan.steps) == 1


class _ScriptedClient:
    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text=self._response, model="m", finish_reason="stop")


@pytest.mark.asyncio
async def test_llm_planner_parses_valid_json() -> None:
    """Valid JSON plans are parsed into structured steps."""
    payload = (
        "```json\n"
        "{\n"
        '  "summary": "two-step plan",\n'
        '  "rationale": "decompose",\n'
        '  "steps": [\n'
        '    {"id": "s1", "description": "read"},\n'
        '    {"id": "s2", "description": "patch"}\n'
        "  ]\n"
        "}\n"
        "```"
    )
    planner = LLMPlanner(_ScriptedClient(payload), max_tokens=200)  # type: ignore[arg-type]
    plan = await planner.plan(Task(id="t", goal="g"))
    assert plan.summary == "two-step plan"
    assert [s.id for s in plan.steps] == ["s1", "s2"]


@pytest.mark.asyncio
async def test_llm_planner_falls_back_on_garbage() -> None:
    """Malformed responses fall back to the heuristic planner."""
    planner = LLMPlanner(_ScriptedClient("not json"), max_tokens=200)  # type: ignore[arg-type]
    plan = await planner.plan(
        Task(
            id="t",
            goal="g",
            acceptance_criteria=(AcceptanceCriterion(id="c1", description="d"),),
        )
    )
    # Heuristic plan with one step per criterion.
    assert len(plan.steps) == 1
