"""Helper builders for harness tests.

These are *not* pytest fixtures — they are plain factory functions and small
classes that test conftests use to set up scenarios. Pytest fixtures live in
``tests/integration/harness/conftest.py`` and
``tests/unit/framework/harness/conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.adapters.llm.base import LLMResponse, LLMToolResponse, ToolCall
from src.framework.harness.tools import ToolSchema


@dataclass
class ScriptedLLM:
    """Tiny fake :class:`LLMClient` driven by a pre-written response queue.

    Each call dequeues the next response. If the queue is empty, an empty
    string response is returned so tests don't deadlock — callers can detect
    over-consumption via :attr:`call_count`.
    """

    responses: list[LLMResponse] = field(default_factory=list)
    call_count: int = 0
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_kwargs = {
            "messages": messages,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "stream": stream,
            "stop": stop,
            **kwargs,
        }
        if not self.responses:
            return LLMResponse(text="", model="scripted", finish_reason="stop")
        return self.responses.pop(0)


def llm_text(text: str, *, tokens: int = 10) -> LLMResponse:
    """Build a plain :class:`LLMResponse`."""
    return LLMResponse(
        text=text,
        usage={"total_tokens": tokens, "prompt_tokens": tokens // 2, "completion_tokens": tokens // 2},
        model="scripted",
        finish_reason="stop",
    )


def llm_tool_call(name: str, args: dict[str, Any], *, tokens: int = 10) -> LLMToolResponse:
    """Build an :class:`LLMToolResponse` with a single tool call."""
    return LLMToolResponse(
        text="",
        usage={"total_tokens": tokens, "prompt_tokens": tokens // 2, "completion_tokens": tokens // 2},
        model="scripted",
        finish_reason="tool_use",
        tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=args)],
    )


def echo_tool() -> tuple[ToolSchema, Callable[[dict[str, Any]], Any]]:
    """A trivial tool: returns ``arguments['value']``."""

    async def handler(args: dict[str, Any]) -> str:
        return str(args.get("value", ""))

    schema = ToolSchema(
        name="echo",
        description="Echo back the 'value' argument.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
    )
    return schema, handler


__all__ = ["ScriptedLLM", "echo_tool", "llm_text", "llm_tool_call"]
