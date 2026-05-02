"""Record/replay primitives — clock, RNG, UUID, and LLM cassettes."""

from src.framework.harness.replay.cassette import Cassette, CassetteEntry, hash_request
from src.framework.harness.replay.client import (
    CassetteMiss,
    ReplayLLMClient,
    ReplayMode,
    make_replay_client,
)
from src.framework.harness.replay.clock import (
    DeterministicClock,
    RecordingClock,
    SystemClock,
)

__all__ = [
    "Cassette",
    "CassetteEntry",
    "CassetteMiss",
    "DeterministicClock",
    "RecordingClock",
    "ReplayLLMClient",
    "ReplayMode",
    "SystemClock",
    "hash_request",
    "make_replay_client",
]
