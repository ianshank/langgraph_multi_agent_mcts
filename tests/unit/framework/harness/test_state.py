"""Unit tests for harness state, task, plan, and observation dataclasses."""

from __future__ import annotations

import pytest

from src.framework.harness.state import (
    AcceptanceCriterion,
    ContextPayload,
    HarnessState,
    Observation,
    Plan,
    PlanStep,
    Task,
    ToolInvocation,
    VerificationResult,
)

pytestmark = pytest.mark.unit


def test_task_is_immutable() -> None:
    """``Task`` is a frozen dataclass — runtime fields cannot be reassigned."""
    task = Task(id="t1", goal="build")
    # Frozen dataclasses raise FrozenInstanceError (a subclass of AttributeError)
    # when fields are reassigned.
    with pytest.raises((AttributeError, TypeError)):
        task.goal = "change"  # type: ignore[misc]


def test_plan_holds_ordered_steps() -> None:
    """Plan steps preserve insertion order."""
    plan = Plan(
        task_id="t1",
        summary="three-step plan",
        steps=(
            PlanStep(id="s1", description="read"),
            PlanStep(id="s2", description="patch"),
            PlanStep(id="s3", description="verify"),
        ),
    )
    assert [s.id for s in plan.steps] == ["s1", "s2", "s3"]


def test_context_payload_render_omits_empty_sections() -> None:
    """Empty optional sections must not appear in the rendered system prompt."""
    payload = ContextPayload(
        system_prompt="You are a careful agent.",
        task_brief="Make tests pass.",
    )
    msgs = payload.render()
    assert msgs[0]["role"] == "system"
    text = msgs[0]["content"]
    assert "# Task" in text
    assert "# Plan" not in text
    assert "# Memory" not in text
    assert "# Spec" not in text


def test_context_payload_render_includes_all_sections() -> None:
    """When every section is populated, every header appears."""
    payload = ContextPayload(
        system_prompt="sys",
        task_brief="t",
        plan_brief="p",
        memory_excerpt="m",
        spec_excerpt="s",
        rag_snippets=("a", "b"),
    )
    text = payload.render()[0]["content"]
    for header in ("# Task", "# Plan", "# Memory", "# Spec", "# Retrieved"):
        assert header in text


def test_acceptance_criterion_carries_check_text() -> None:
    """``AcceptanceCriterion`` round-trips human-readable check details."""
    crit = AcceptanceCriterion(id="c1", description="pytest passes", check="pytest -q")
    assert crit.check == "pytest -q"


def test_observation_records_spillover_path() -> None:
    """Spillover path is preserved when output is truncated."""
    obs = Observation(
        invocation_id="i1",
        tool_name="shell",
        success=True,
        payload="head ... tail",
        spillover_path="/tmp/full.log",
        duration_ms=1.0,
    )
    assert obs.spillover_path == "/tmp/full.log"


def test_harness_state_record_appends_history() -> None:
    """``record`` appends a structured history entry without losing prior ones."""
    state = HarnessState(iteration=2)
    state.record("intent", {"task_id": "t1"})
    state.record("plan", {"steps": 3})
    assert len(state.history) == 2
    assert state.history[0]["phase"] == "intent"
    assert state.history[1]["iteration"] == 2


def test_tool_invocation_arguments_default_empty() -> None:
    """``ToolInvocation`` provides a default-empty arguments dict."""
    inv = ToolInvocation(id="i1", tool_name="shell")
    assert inv.arguments == {}


def test_verification_result_failure_paths() -> None:
    """Failed verifications carry their reasons."""
    v = VerificationResult(
        passed=False,
        score=0.4,
        failed_criteria=("c1", "c2"),
        notes="tests failed",
    )
    assert not v.passed
    assert v.failed_criteria == ("c1", "c2")
