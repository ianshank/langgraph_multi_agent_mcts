"""Single-writer compactor that materialises ``MEMORY.md`` from events.

Compaction is deterministic and idempotent: given the same set of events,
the resulting ``MEMORY.md`` is byte-identical regardless of input ordering
within commutative phases (intent / plan / verify groupings).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from src.framework.harness.memory.events import MemoryEvent, MemoryEventLog


@dataclass
class MemoryCompactor:
    """Materialise the ``MEMORY.md`` index from the event log."""

    log: MemoryEventLog
    index_path: Path
    lock_path: Path = field(init=False)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    _async_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self.lock_path = self.index_path.parent / ".compactor.lock"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    async def compact_once(self) -> str:
        """Rebuild ``MEMORY.md`` from the current event log; returns its content."""
        async with self._async_lock:
            content = await asyncio.to_thread(self._compact_blocking)
            return content

    def _compact_blocking(self) -> str:
        events = self.log.collect_all()
        rendered = render_index(events)
        # Atomic write via temp file + rename.
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp.write_text(rendered, encoding="utf-8")
        tmp.replace(self.index_path)
        return rendered


def render_index(events: Iterable[MemoryEvent]) -> str:
    """Pure function: turn events into the canonical ``MEMORY.md`` content.

    The output is grouped by ``correlation_id`` (newest first, by max-event
    timestamp), then by phase within each correlation. This grouping is
    commutative within a phase, which is what makes the operation idempotent
    over reorderings inside a single phase block.
    """
    by_corr: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in events:
        by_corr[event.correlation_id].append(event)

    sections: list[str] = ["# Harness Memory Index", ""]
    if not by_corr:
        sections.append("_No events yet._")
        return "\n".join(sections) + "\n"

    # Sort correlations by their newest event timestamp (descending).
    ordered = sorted(
        by_corr.items(),
        key=lambda kv: max(e.at for e in kv[1]),
        reverse=True,
    )
    for corr_id, group in ordered:
        sections.append(f"## Correlation `{corr_id}`")
        # Group by phase, stable-sorted by phase name for deterministic output.
        phases: dict[str, list[MemoryEvent]] = defaultdict(list)
        for event in group:
            phases[event.phase].append(event)
        for phase in sorted(phases.keys()):
            sections.append(f"### Phase `{phase}`")
            for event in sorted(phases[phase], key=lambda e: e.at):
                rendered = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
                sections.append(f"- {event.at.isoformat()} — {rendered}" if rendered else f"- {event.at.isoformat()}")
            sections.append("")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


__all__ = ["MemoryCompactor", "render_index"]
