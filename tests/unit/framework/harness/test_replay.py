"""Unit tests for the replay subsystem."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.adapters.llm.base import LLMResponse, LLMToolResponse, ToolCall
from src.framework.harness.replay import (
    Cassette,
    CassetteMiss,
    ReplayLLMClient,
    hash_request,
    make_replay_client,
)

pytestmark = pytest.mark.unit


def test_hash_request_is_order_invariant() -> None:
    """Identical payloads with different key insertion orders hash equal."""
    a = {"x": 1, "y": [1, 2], "z": {"a": 1, "b": 2}}
    b = {"z": {"b": 2, "a": 1}, "y": [1, 2], "x": 1}
    assert hash_request(a) == hash_request(b)


def test_cassette_append_and_lookup(tmp_path: Path) -> None:
    """Appending an entry should make it retrievable via ``lookup``."""
    cassette = Cassette(tmp_path / "test.jsonl")
    request = {"messages": [{"role": "user", "content": "hi"}]}
    response = {"type": "response", "text": "hello", "usage": {}, "model": "m", "finish_reason": "stop"}
    entry = cassette.append(request, response)
    assert cassette.lookup(entry.request_hash) is not None
    assert len(cassette) == 1


def test_cassette_persists_across_instances(tmp_path: Path) -> None:
    """A fresh cassette over the same path should see prior recordings."""
    path = tmp_path / "shared.jsonl"
    Cassette(path).append({"prompt": "a"}, {"type": "response", "text": "1"})
    second = Cassette(path)
    assert len(second) == 1


@pytest.mark.asyncio
async def test_record_mode_writes_then_replay_can_read(tmp_path: Path) -> None:
    """Record then replay round-trips successfully."""
    inner = AsyncMock()
    inner.generate.return_value = LLMResponse(text="recorded", model="m", finish_reason="stop")

    recorder = make_replay_client(inner=inner, cassette_dir=tmp_path, mode="record")
    rec_response = await recorder.generate(prompt="hi", temperature=0.0)
    assert isinstance(rec_response, LLMResponse)
    assert rec_response.text == "recorded"
    assert len(recorder.cassette) == 1

    # Now replay — inner client should not be invoked.
    replay_inner = AsyncMock()
    replayer = make_replay_client(inner=replay_inner, cassette_dir=tmp_path, mode="replay")
    rp_response = await replayer.generate(prompt="hi", temperature=0.0)
    assert isinstance(rp_response, LLMResponse)
    assert rp_response.text == "recorded"
    replay_inner.generate.assert_not_called()


@pytest.mark.asyncio
async def test_replay_miss_raises(tmp_path: Path) -> None:
    """Unknown requests in replay mode must raise :class:`CassetteMiss`."""
    replayer = make_replay_client(inner=None, cassette_dir=tmp_path, mode="replay")
    with pytest.raises(CassetteMiss):
        await replayer.generate(prompt="never recorded")


@pytest.mark.asyncio
async def test_replay_preserves_tool_calls(tmp_path: Path) -> None:
    """Tool-call responses must round-trip through the cassette."""
    tool_calls = [ToolCall(id="t1", name="shell", arguments={"cmd": "ls"})]
    inner = AsyncMock()
    inner.generate.return_value = LLMToolResponse(
        text="",
        usage={"total_tokens": 10},
        model="m",
        finish_reason="tool_use",
        tool_calls=tool_calls,
    )
    recorder = make_replay_client(inner=inner, cassette_dir=tmp_path, mode="record")
    await recorder.generate(prompt="run ls", tools=[{"name": "shell"}])

    replayer = make_replay_client(inner=None, cassette_dir=tmp_path, mode="replay")
    rp = await replayer.generate(prompt="run ls", tools=[{"name": "shell"}])
    assert isinstance(rp, LLMToolResponse)
    assert len(rp.tool_calls) == 1
    assert rp.tool_calls[0].name == "shell"
    assert rp.tool_calls[0].arguments == {"cmd": "ls"}


@pytest.mark.asyncio
async def test_passthrough_does_not_record(tmp_path: Path) -> None:
    """Passthrough mode forwards but does not write the cassette."""
    inner = AsyncMock()
    inner.generate.return_value = LLMResponse(text="x")
    client = make_replay_client(inner=inner, cassette_dir=tmp_path, mode="passthrough")
    await client.generate(prompt="ignore me")
    assert len(client.cassette) == 0


def test_record_mode_requires_inner_client(tmp_path: Path) -> None:
    """Record mode without an inner client is a configuration error."""
    with pytest.raises(ValueError):
        ReplayLLMClient(inner=None, cassette=Cassette(tmp_path / "c.jsonl"), mode="record")


@pytest.mark.asyncio
async def test_streaming_passes_through_to_inner(tmp_path: Path) -> None:
    """Streaming requests bypass the cassette entirely."""

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield "chunk"

        return _gen()

    inner = AsyncMock()
    inner.generate = fake_stream  # type: ignore[assignment]
    client = make_replay_client(inner=inner, cassette_dir=tmp_path, mode="record")
    result = await client.generate(prompt="stream", stream=True)
    chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        chunks.append(chunk)
    assert chunks == ["chunk"]
    assert len(client.cassette) == 0
