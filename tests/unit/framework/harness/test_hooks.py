"""Unit tests for the hook chain."""

from __future__ import annotations

import pytest

from src.framework.harness.hooks import BaseHook, HookChain
from src.framework.harness.outcomes import HookViolation
from src.framework.harness.protocols import HookCost
from src.framework.harness.state import HarnessState

pytestmark = pytest.mark.unit


class _AlwaysPass(BaseHook):
    async def check(self, state: HarnessState) -> None:
        return None


class _AlwaysFail(BaseHook):
    async def check(self, state: HarnessState) -> HookViolation:
        return HookViolation(hook_name=self.name, detail="fail")


class _Raises(BaseHook):
    async def check(self, state: HarnessState) -> HookViolation:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_empty_chain_passes() -> None:
    """An empty chain returns a passing outcome."""
    out = await HookChain().run(HarnessState())
    assert out.passed is True


@pytest.mark.asyncio
async def test_short_circuit_stops_chain() -> None:
    """A short-circuit failure prevents later hooks from running."""
    later = _AlwaysPass(name="later", cost_class=HookCost.EXPENSIVE)
    chain = HookChain.of(
        _AlwaysFail(name="early", cost_class=HookCost.CHEAP, short_circuit=True),
        later,
    )
    out = await chain.run(HarnessState())
    assert out.passed is False
    assert out.short_circuited is True
    assert [v.hook_name for v in out.violations] == ["early"]


@pytest.mark.asyncio
async def test_non_short_circuit_accumulates() -> None:
    """``short_circuit=False`` lets the chain continue past failures."""
    chain = HookChain.of(
        _AlwaysFail(name="a", cost_class=HookCost.CHEAP, short_circuit=False),
        _AlwaysFail(name="b", cost_class=HookCost.CHEAP, short_circuit=False),
    )
    out = await chain.run(HarnessState())
    assert out.passed is False
    assert {v.hook_name for v in out.violations} == {"a", "b"}
    assert out.short_circuited is False


@pytest.mark.asyncio
async def test_chain_orders_cheap_first() -> None:
    """Hooks are sorted by cost class with stable tie-breaks."""
    chain = HookChain.of(
        _AlwaysPass(name="exp", cost_class=HookCost.EXPENSIVE),
        _AlwaysPass(name="cheap1", cost_class=HookCost.CHEAP),
        _AlwaysPass(name="med", cost_class=HookCost.MEDIUM),
        _AlwaysPass(name="cheap2", cost_class=HookCost.CHEAP),
    )
    assert chain.names() == ["cheap1", "cheap2", "med", "exp"]


@pytest.mark.asyncio
async def test_exception_becomes_violation() -> None:
    """A hook raising should become a structured violation, not a crash."""
    chain = HookChain.of(_Raises(name="boom", cost_class=HookCost.MEDIUM))
    out = await chain.run(HarnessState())
    assert out.passed is False
    assert "RuntimeError" in out.violations[0].detail


@pytest.mark.asyncio
async def test_add_re_sorts_in_place() -> None:
    """``HookChain.add`` preserves stable cost ordering."""
    chain = HookChain.of(_AlwaysPass(name="a", cost_class=HookCost.EXPENSIVE))
    chain.add(_AlwaysPass(name="b", cost_class=HookCost.CHEAP))
    chain.add(_AlwaysPass(name="c", cost_class=HookCost.MEDIUM))
    assert chain.names() == ["b", "c", "a"]
