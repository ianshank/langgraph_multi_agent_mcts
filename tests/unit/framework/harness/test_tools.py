"""Unit tests for tool registry, executor, and truncation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.state import ToolInvocation
from src.framework.harness.tools import (
    AsyncToolExecutor,
    ToolRegistry,
    ToolSchema,
    truncate_with_spillover,
)

pytestmark = pytest.mark.unit


def test_truncate_no_spillover_when_short(tmp_path: Path) -> None:
    """If payload fits within head+tail, no spillover file is written."""
    out, spillover = truncate_with_spillover(
        "hello",
        head_chars=10,
        tail_chars=10,
        spillover_dir=tmp_path,
        correlation_id="cid",
        step_id="s1",
        marker_template="{path}",
    )
    assert out == "hello"
    assert spillover is None
    assert not list(tmp_path.iterdir())


def test_truncate_writes_spillover_for_long(tmp_path: Path) -> None:
    """Long payloads are truncated and the full text is preserved on disk."""
    payload = "abcdefghij" * 100  # 1000 chars
    out, spillover = truncate_with_spillover(
        payload,
        head_chars=20,
        tail_chars=20,
        spillover_dir=tmp_path,
        correlation_id="cid",
        step_id="s1",
        marker_template="\n…[{path}]…\n",
    )
    assert spillover is not None
    spill_file = Path(spillover)
    assert spill_file.exists()
    assert spill_file.read_text() == payload
    assert out.startswith(payload[:20])
    assert out.endswith(payload[-20:])
    assert spillover in out


def test_truncate_rejects_negative_chars(tmp_path: Path) -> None:
    """Negative head/tail counts are a programming error."""
    with pytest.raises(ValueError):
        truncate_with_spillover(
            "x",
            head_chars=-1,
            tail_chars=10,
            spillover_dir=tmp_path,
            correlation_id="c",
            step_id="s",
            marker_template="",
        )


def test_registry_register_and_lookup() -> None:
    """Registry maps name → handler with schema metadata."""
    registry = ToolRegistry()

    async def handler(args: dict[str, Any]) -> str:
        return "ok"

    schema = ToolSchema(name="t1", description="desc", parameters={"type": "object"})
    registry.register(schema, handler)
    assert registry.has("t1")
    assert registry.get_handler("t1") is handler
    assert registry.get_schema("t1").description == "desc"
    assert registry.list_names() == ["t1"]


def test_registry_rejects_empty_name() -> None:
    """Empty tool names are forbidden."""
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(ToolSchema(name="", description=""), lambda _a: None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_executor_returns_observation_for_unknown_tool(tmp_path: Path) -> None:
    """Calling an unknown tool produces a failed observation, not an exception."""
    settings = HarnessSettings(
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill", MEMORY_ROOT=tmp_path / "mem", OUTPUT_DIR=tmp_path / "runs"
    )
    executor = AsyncToolExecutor(ToolRegistry(), settings)
    obs = await executor.execute(
        ToolInvocation(id="i", tool_name="missing", arguments={}),
        correlation_id="cid",
    )
    assert obs.success is False
    assert "unknown tool" in obs.payload


@pytest.mark.asyncio
async def test_executor_truncates_long_payloads(tmp_path: Path) -> None:
    """Output that exceeds head+tail is truncated with a spillover pointer."""
    settings = HarnessSettings(
        TOOL_OUTPUT_HEAD_CHARS=10,
        TOOL_OUTPUT_TAIL_CHARS=10,
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
    )
    registry = ToolRegistry()

    async def handler(args: dict[str, Any]) -> str:
        return "x" * 1000

    registry.register(ToolSchema(name="big", description="big"), handler)
    executor = AsyncToolExecutor(registry, settings)
    obs = await executor.execute(
        ToolInvocation(id="i1", tool_name="big", arguments={}),
        correlation_id="cid",
    )
    assert obs.spillover_path is not None
    assert Path(obs.spillover_path).exists()
    assert obs.success is True


@pytest.mark.asyncio
async def test_executor_enforces_timeout(tmp_path: Path) -> None:
    """A handler that hangs is cancelled after the configured timeout."""
    import asyncio

    settings = HarnessSettings(
        TOOL_DEFAULT_TIMEOUT_SECONDS=0.1,
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill",
        MEMORY_ROOT=tmp_path / "mem",
        OUTPUT_DIR=tmp_path / "runs",
    )
    registry = ToolRegistry()

    async def slow(args: dict[str, Any]) -> str:
        await asyncio.sleep(2.0)
        return "never"

    registry.register(ToolSchema(name="slow", description="slow"), slow)
    executor = AsyncToolExecutor(registry, settings)
    obs = await executor.execute(
        ToolInvocation(id="i1", tool_name="slow", arguments={}),
        correlation_id="cid",
    )
    assert obs.success is False
    assert obs.metadata.get("reason") == "timeout"


def test_executor_tool_schemas_shape(tmp_path: Path) -> None:
    """``tool_schemas`` returns the OpenAI/Anthropic ``tools=`` shape."""
    settings = HarnessSettings(
        TOOL_OUTPUT_SPILLOVER_DIR=tmp_path / "spill", MEMORY_ROOT=tmp_path / "mem", OUTPUT_DIR=tmp_path / "runs"
    )
    registry = ToolRegistry()
    registry.register(
        ToolSchema(name="t", description="d", parameters={"type": "object", "properties": {}}),
        lambda _a: None,  # type: ignore[arg-type]
    )
    executor = AsyncToolExecutor(registry, settings)
    schemas = executor.tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    fn = schemas[0]["function"]
    assert isinstance(fn, dict)
    assert fn["name"] == "t"
