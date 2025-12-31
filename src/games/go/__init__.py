"""
Go Game Environment.

Implements the game of Go for AlphaZero-style training.
Supports configurable board sizes (9x9, 13x13, 19x19).

Based on:
- michaelnny/alpha_zero Go implementation
- Standard Go rules with Chinese scoring
"""

from .config import GoConfig
from .engine import GoEngine
from .state import GoGameState, create_go_state

__all__ = [
    "GoConfig",
    "GoEngine",
    "GoGameState",
    "create_go_state",
]
