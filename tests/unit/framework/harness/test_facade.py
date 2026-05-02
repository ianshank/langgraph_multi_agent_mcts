"""Tests for the AsyncAgentBase facade over HarnessRunner."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.agents.base import AgentContext
from src.framework.harness import HarnessSettings
from src.framework.harness.context import DefaultContextInjector
from src.framework.harness.hooks import HookChain
from src.framework.harness.intent import DefaultIntentNormalizer
from src.framework.harness.loop.facade import HarnessAgentAdapter
from src.framework.harness.loop.runner import HarnessRunner
from src.framework.harness.planner import HeuristicPlanner
from src.framework.harness.replay.clock import DeterministicClock
from src.framework.harness.tools import AsyncToolExecutor, ToolRegistry
from src.framework.harness.verifier import AcceptanceCriteriaVerifier
from tests.fixtures.harness_fixtures import ScriptedLLM, echo_tool, llm_tool_call

pytestmark = pytest.mark.unit


def _make_runner(tmp_path: Path, llm: ScriptedLLM) -> HarnessRunner:
    settings = HarnessSettings(
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MAX_ITERATIONS=2,
    )
    registry = ToolRegistry()
    schema, handler = echo_tool()
    registry.register(schema, handler)
    return HarnessRunner(
        settings=settings,
        llm=llm,
        intent=DefaultIntentNormalizer(),
        planner=HeuristicPlanner(),
        context_injector=DefaultContextInjector(),
        tool_executor=AsyncToolExecutor(registry, settings),
        verifier=AcceptanceCriteriaVerifier(),
        hooks=HookChain(),
        clock=DeterministicClock(seed=7),
    )


@pytest.mark.asyncio
async def test_facade_emits_graph_builder_compatible_keys(tmp_path: Path) -> None:
    """``HarnessAgentAdapter`` populates the keys ``GraphBuilder`` reads."""
    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "passes"})])
    runner = _make_runner(tmp_path, llm)
    adapter = HarnessAgentAdapter(runner)

    raw = await adapter.process(
        query="do the thing",
        rag_context=None,
    )
    metadata = raw["metadata"]
    assert "decomposition_quality_score" in metadata
    assert "confidence" in metadata
    assert "outcome" in metadata
    assert metadata["outcome"] in {"terminal", "budget_exhausted", "continue", "retryable", "hook_violation"}
    assert metadata["agent_name"] == "harness"


@pytest.mark.asyncio
async def test_facade_carries_intermediate_steps(tmp_path: Path) -> None:
    """History entries flow into ``intermediate_steps`` for downstream use."""
    llm = ScriptedLLM(responses=[llm_tool_call("echo", {"value": "passes"})])
    runner = _make_runner(tmp_path, llm)
    adapter = HarnessAgentAdapter(runner)
    ctx = AgentContext(
        query="do",
        metadata={"k": "v"},
        rag_context="r",
    )
    result = await adapter._process_impl(ctx)
    assert result.intermediate_steps  # non-empty
    phases = {step["phase"] for step in result.intermediate_steps}
    assert {"intent", "plan"}.issubset(phases)
