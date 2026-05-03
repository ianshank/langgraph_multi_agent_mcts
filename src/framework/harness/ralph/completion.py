"""Completion-marker detection for the Ralph loop."""

from __future__ import annotations

from pathlib import Path


def is_complete(spec_path: Path | None, marker: str, *, content: str | None = None) -> bool:
    """Return ``True`` if ``marker`` appears in the spec or supplied ``content``.

    The Ralph loop polls this on every outer cycle; a positive result halts
    the loop. Either the spec file's content or an inline ``content`` string
    may be checked.
    """
    if not marker:
        return False
    if content is not None and marker in content:
        return True
    if spec_path is None or not spec_path.exists():
        return False
    return marker in spec_path.read_text(encoding="utf-8")


__all__ = ["is_complete"]
