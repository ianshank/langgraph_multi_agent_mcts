"""Unit tests for the harness budget primitives."""

from __future__ import annotations

import pytest

from src.framework.harness.loop.budget import (
    Budget,
    BudgetBundle,
    IterationBudget,
    TimeBudget,
)

pytestmark = pytest.mark.unit


def test_budget_consume_and_remaining() -> None:
    """Token-style budget tracks consumed and remaining amounts."""
    b = Budget(limit=100.0, name="tokens")
    b.consume(40)
    assert b.remaining() == 60.0
    assert not b.exhausted()
    b.consume(60)
    assert b.exhausted()
    assert b.remaining() == 0.0


def test_budget_rejects_negative_consumption() -> None:
    """Negative consumption is a programming error."""
    with pytest.raises(ValueError):
        Budget(limit=10).consume(-1)


def test_iteration_budget_increment_default_one() -> None:
    """``IterationBudget.consume`` defaults to one tick."""
    ib = IterationBudget(limit=3)
    ib.consume()
    ib.consume()
    assert ib.consumed == 2
    assert not ib.exhausted()
    ib.consume()
    assert ib.exhausted()
    assert ib.remaining() == 0


def test_time_budget_uses_injected_clock() -> None:
    """The clock is injectable so tests can advance time deterministically."""
    fake_now = [0.0]

    def fake_clock() -> float:
        return fake_now[0]

    tb = TimeBudget(limit_seconds=10.0, started_at=0.0, clock=fake_clock, name="time")
    fake_now[0] = 5.0
    assert tb.elapsed() == pytest.approx(5.0)
    assert not tb.exhausted()
    fake_now[0] = 11.0
    assert tb.exhausted()
    assert tb.remaining() == 0.0


def test_time_budget_reset() -> None:
    """``reset`` re-anchors the start time so a fresh window begins."""
    fake_now = [100.0]
    tb = TimeBudget(
        limit_seconds=5.0,
        started_at=fake_now[0],
        clock=lambda: fake_now[0],
        name="time",
    )
    fake_now[0] = 110.0
    assert tb.exhausted()
    tb.reset()
    assert not tb.exhausted()
    assert tb.elapsed() == pytest.approx(0.0)


def test_budget_bundle_first_exhausted() -> None:
    """``BudgetBundle.first_exhausted`` returns whichever sub-budget tripped first."""
    bundle = BudgetBundle(
        tokens=Budget(limit=10, name="tokens"),
        time=TimeBudget(limit_seconds=10.0, started_at=0.0, clock=lambda: 5.0, name="time"),
        iterations=IterationBudget(limit=3, name="iterations"),
    )
    assert bundle.first_exhausted() is None
    bundle.iterations.consume(3)
    assert bundle.first_exhausted() == "iterations"
