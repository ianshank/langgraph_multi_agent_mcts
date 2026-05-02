"""Unit tests for the recording / deterministic clocks."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.framework.harness.replay import (
    DeterministicClock,
    RecordingClock,
    SystemClock,
)

pytestmark = pytest.mark.unit


def test_system_clock_satisfies_protocol() -> None:
    """``SystemClock`` satisfies the runtime-checkable protocol."""
    assert isinstance(SystemClock(), RecordingClock)


def test_deterministic_clock_is_reproducible() -> None:
    """Two clocks seeded identically must yield identical sequences."""
    c1 = DeterministicClock(seed=42)
    c2 = DeterministicClock(seed=42)
    seq1 = [c1.now(), c1.uuid(), c1.random(), c1.monotonic()]
    seq2 = [c2.now(), c2.uuid(), c2.random(), c2.monotonic()]
    assert seq1 == seq2


def test_deterministic_clock_starts_at_configured_instant() -> None:
    """Anchor time is honored."""
    anchor = datetime(2030, 6, 1, tzinfo=UTC)
    c = DeterministicClock(seed=0, start=anchor, tick_seconds=1.0)
    first = c.now()
    second = c.now()
    assert first >= anchor
    assert second > first


def test_deterministic_clock_uuids_unique_per_call() -> None:
    """Each call must produce a distinct UUID even with the same seed."""
    c = DeterministicClock(seed=1)
    uuids = {c.uuid() for _ in range(50)}
    assert len(uuids) == 50


def test_deterministic_clock_monotonic_increasing() -> None:
    """Monotonic readings never decrease."""
    c = DeterministicClock(seed=0, tick_seconds=0.5)
    samples = [c.monotonic() for _ in range(10)]
    assert samples == sorted(samples)
    assert samples[-1] > samples[0]


def test_system_clock_monotonic_increasing() -> None:
    """Real monotonic clock never goes backwards across two reads."""
    c = SystemClock()
    a = c.monotonic()
    b = c.monotonic()
    assert b >= a
