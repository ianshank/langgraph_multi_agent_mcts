"""Tool registry — name → callable + schema."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class ToolSchema:
    """Public description of a tool, suitable for LLM ``tools=`` payloads."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RegisteredTool:
    schema: ToolSchema
    handler: ToolHandler


class ToolRegistry:
    """Mapping of tool name → handler/schema. Registration is mutation-only;
    handlers are looked up by exact name."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(self, schema: ToolSchema, handler: ToolHandler) -> None:
        """Register or replace a tool by name."""
        if not schema.name:
            raise ValueError("tool name cannot be empty")
        self._tools[schema.name] = _RegisteredTool(schema=schema, handler=handler)

    def unregister(self, name: str) -> None:
        """Remove a tool by name (no-op if absent)."""
        self._tools.pop(name, None)

    def has(self, name: str) -> bool:
        """Membership check."""
        return name in self._tools

    def get_handler(self, name: str) -> ToolHandler:
        """Return the handler or raise ``KeyError``."""
        return self._tools[name].handler

    def get_schema(self, name: str) -> ToolSchema:
        """Return the schema or raise ``KeyError``."""
        return self._tools[name].schema

    def list_names(self) -> list[str]:
        """Names of all registered tools."""
        return sorted(self._tools.keys())

    def schemas(self) -> list[ToolSchema]:
        """Schemas of all registered tools, in name order."""
        return [self._tools[name].schema for name in self.list_names()]


__all__ = ["ToolHandler", "ToolRegistry", "ToolSchema"]
