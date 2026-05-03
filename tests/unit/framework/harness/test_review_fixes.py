"""Tests for changes added in response to Copilot review.

Covers:

* ``ITERATION_TIMEOUT_SECONDS`` is enforced via ``asyncio.wait_for`` in the
  per-iteration phase chain.
* Out-of-range ``anchor_line`` and negative ``window`` produce
  :class:`HashAnchorMismatch`, not raw ``IndexError``/``ValueError``.
* ``ToolExecutionTimeout`` is no longer importable from ``tools.executor``.
* ``VERIFIER_FAIL_FAST`` setting was removed and is no longer accessible.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.context import DefaultContextInjector
from src.framework.harness.hooks import HookChain
from src.framework.harness.intent import DefaultIntentNormalizer
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.outcomes import BudgetExhausted
from src.framework.harness.planner import HeuristicPlanner
from src.framework.harness.replay.clock import DeterministicClock
from src.framework.harness.tools import AsyncToolExecutor, ToolRegistry
from src.framework.harness.tools.hashed_edit import (
    HashAnchorMismatch,
    HashedEdit,
    apply_edit,
    file_sha256,
)
from src.framework.harness.verifier import AlwaysReject

pytestmark = pytest.mark.unit


class _HangingLLM:
    """LLM client that sleeps forever — used to trip the per-iteration timeout."""

    async def generate(self, **kwargs: Any) -> Any:
        await asyncio.sleep(60.0)
        raise AssertionError("should never reach")


def _runner(tmp_path: Path, llm: Any, *, iteration_timeout: float) -> HarnessRunner:
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MAX_ITERATIONS=3,
        ITERATION_TIMEOUT_SECONDS=iteration_timeout,
        TOTAL_BUDGET_SECONDS=30,
        PLANNER_ENABLED=False,
    )
    return HarnessRunner(
        settings=settings,
        llm=llm,
        intent=DefaultIntentNormalizer(),
        planner=HeuristicPlanner(),
        context_injector=DefaultContextInjector(),
        tool_executor=AsyncToolExecutor(ToolRegistry(), settings),
        verifier=AlwaysReject(),
        hooks=HookChain(),
        clock=DeterministicClock(seed=1),
    )


@pytest.mark.asyncio
async def test_iteration_timeout_terminates_runner(tmp_path: Path) -> None:
    """A hung Reason phase trips ``ITERATION_TIMEOUT_SECONDS`` and exits cleanly."""
    runner = _runner(tmp_path, _HangingLLM(), iteration_timeout=0.1)
    result = await runner.run("hang the reason phase")
    assert isinstance(result.outcome, BudgetExhausted)
    assert result.outcome.budget == "time"
    assert result.iterations == 1


def test_apply_edit_rejects_out_of_range_anchor(tmp_path: Path) -> None:
    """An anchor beyond the file length surfaces as ``HashAnchorMismatch``."""
    target = tmp_path / "f.txt"
    target.write_text("a\nb\nc\n")
    edit = HashedEdit(
        path=target,
        expected_file_hash=file_sha256(target),
        anchor_line=99,
        expected_window_hash="any",
        new_content="x",
    )
    with pytest.raises(HashAnchorMismatch) as exc_info:
        apply_edit(edit)
    assert "out of range" in str(exc_info.value)


def test_apply_edit_rejects_negative_window(tmp_path: Path) -> None:
    """A negative window is rejected before any disk read."""
    target = tmp_path / "f.txt"
    target.write_text("a\n")
    edit = HashedEdit(
        path=target,
        expected_file_hash=file_sha256(target),
        anchor_line=0,
        expected_window_hash="any",
        new_content="x",
        window=-1,
    )
    with pytest.raises(HashAnchorMismatch):
        apply_edit(edit)


def test_tool_execution_timeout_is_no_longer_exported() -> None:
    """The unused ``ToolExecutionTimeout`` exception has been removed."""
    import src.framework.harness.tools.executor as ex_mod

    assert not hasattr(ex_mod, "ToolExecutionTimeout")


def test_verifier_fail_fast_setting_removed() -> None:
    """The unused ``VERIFIER_FAIL_FAST`` setting is no longer present."""
    s = HarnessSettings()
    assert not hasattr(s, "VERIFIER_FAIL_FAST")


def test_harness_module_loggers_use_mcts_prefix() -> None:
    """Per-module loggers carry the ``mcts.`` namespace from ``get_logger``."""
    from src.framework.harness.loop.runner import HarnessRunner as _HR

    # A freshly-constructed runner uses the convention.
    settings = HarnessSettings()
    registry = ToolRegistry()

    class _StubLLM:
        async def generate(self, **kwargs: Any) -> Any:  # pragma: no cover - not invoked
            return None

    runner = _HR(
        settings=settings,
        llm=_StubLLM(),
        intent=DefaultIntentNormalizer(),
        planner=HeuristicPlanner(),
        context_injector=DefaultContextInjector(),
        tool_executor=AsyncToolExecutor(registry, settings),
        verifier=AlwaysReject(),
    )
    assert runner.logger.name.startswith("mcts.")
