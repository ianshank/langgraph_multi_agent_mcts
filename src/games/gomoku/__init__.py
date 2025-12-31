"""
Gomoku (Five in a Row) Game Environment.

Implements Gomoku for AlphaZero-style training.
Supports freestyle rules on configurable board sizes.

Gomoku is simpler than Go, making it ideal for:
- Testing AlphaZero implementations
- Quick training experiments
- Educational purposes
"""

from .config import GomokuConfig
from .state import GomokuGameState, create_gomoku_state

__all__ = [
    "GomokuConfig",
    "GomokuGameState",
    "create_gomoku_state",
]
