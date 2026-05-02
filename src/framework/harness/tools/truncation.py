"""Output truncation with spillover-to-disk for full logs."""

from __future__ import annotations

from pathlib import Path


def truncate_with_spillover(
    payload: str,
    *,
    head_chars: int,
    tail_chars: int,
    spillover_dir: Path,
    correlation_id: str,
    step_id: str,
    marker_template: str,
) -> tuple[str, str | None]:
    """Truncate ``payload`` and persist the full text on disk.

    Returns ``(truncated_payload, spillover_path_or_none)``. If the payload is
    short enough, no spillover file is written and the second tuple element
    is ``None``.

    The marker template may contain ``{path}`` which is substituted with the
    spillover path so the model sees a deterministic pointer to the full log.
    """
    if head_chars < 0 or tail_chars < 0:
        raise ValueError("head_chars and tail_chars must be non-negative")
    if len(payload) <= head_chars + tail_chars:
        return payload, None

    spillover_dir = spillover_dir / correlation_id
    spillover_dir.mkdir(parents=True, exist_ok=True)
    path = spillover_dir / f"{step_id}.log"
    path.write_text(payload, encoding="utf-8")

    head = payload[:head_chars]
    tail = payload[-tail_chars:] if tail_chars > 0 else ""
    marker = marker_template.format(path=str(path))
    return f"{head}{marker}{tail}", str(path)


__all__ = ["truncate_with_spillover"]
