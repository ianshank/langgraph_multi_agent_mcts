"""Memory tools registered with the :class:`ToolRegistry`.

Each tool is a small wrapper that delegates to the :class:`MarkdownMemoryStore`
methods. Tools accept and return strings; argument parsing is lenient.
"""

from __future__ import annotations

from typing import Any

from src.framework.harness.memory.markdown import MarkdownMemoryStore
from src.framework.harness.tools.registry import ToolRegistry, ToolSchema


def register_memory_tools(registry: ToolRegistry, store: MarkdownMemoryStore) -> None:
    """Register the standard memory tools onto ``registry``."""

    async def memory_view(args: dict[str, Any]) -> str:
        path = str(args.get("path") or store.settings.MEMORY_INDEX_FILENAME)
        # The index file is a derived view — materialise it on demand so the
        # caller never sees a stale or missing copy.
        if path == store.settings.MEMORY_INDEX_FILENAME:
            return await store.read_index()
        return await store.view(path)

    async def memory_query(args: dict[str, Any]) -> str:
        since = args.get("since_iso")
        events = await store.query_episodic(since_iso=str(since) if since else None)
        # Render compactly so the model gets a survey, not a fire-hose.
        return "\n".join(f"{e['at']} [{e['phase']}] {e['correlation_id'][:8]}" for e in events)

    async def memory_compact(_args: dict[str, Any]) -> str:
        return await store.compact()

    registry.register(
        ToolSchema(
            name="memory_view",
            description="Return the contents of a file under the memory root.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        ),
        memory_view,
    )
    registry.register(
        ToolSchema(
            name="memory_query",
            description="List recent memory events as 'timestamp [phase] correlation' lines.",
            parameters={
                "type": "object",
                "properties": {"since_iso": {"type": "string"}},
                "required": [],
            },
        ),
        memory_query,
    )
    registry.register(
        ToolSchema(
            name="memory_compact",
            description="Force the compactor to materialise MEMORY.md from the event log.",
            parameters={"type": "object", "properties": {}},
        ),
        memory_compact,
    )


__all__ = ["register_memory_tools"]
