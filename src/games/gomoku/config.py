"""
Gomoku Game Configuration.

Provides configurable parameters for the Gomoku game environment.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import GameConfig


@dataclass
class GomokuConfig(GameConfig):
    """
    Configuration for Gomoku game environment.

    Supports various board sizes and rule variants.
    """

    # Board size (typically 13x13 or 15x15)
    board_size: int = 13

    # Number of stones in a row to win
    win_length: int = 5

    # History planes for state representation
    history_length: int = 4

    # Variant rules
    # "freestyle": any 5+ in a row wins
    # "standard": exactly 5 wins, overlines don't count
    # "renju": professional rules with restrictions for black
    variant: str = "freestyle"

    # Maximum game length (as factor of board area)
    max_game_length_factor: float = 1.0

    def __post_init__(self):
        """Validate and compute derived values."""
        if self.board_size < self.win_length:
            raise ValueError(f"board_size ({self.board_size}) must be >= win_length ({self.win_length})")

        if self.variant not in ("freestyle", "standard", "renju"):
            raise ValueError(f"Unknown variant: {self.variant}")

        # Compute action size: board_size^2 (no pass in Gomoku)
        self.action_size = self.board_size * self.board_size

        # Compute input planes:
        # - history_length planes for black stones
        # - history_length planes for white stones
        # - 1 plane for current player
        self.num_input_planes = 2 * self.history_length + 1

        # Compute max game length
        self.max_game_length = int(self.board_size * self.board_size * self.max_game_length_factor)

        # Gomoku doesn't have pass
        self.allow_pass = False

    def position_to_index(self, row: int, col: int) -> int:
        """Convert board position to action index."""
        return row * self.board_size + col

    def index_to_position(self, index: int) -> tuple[int, int]:
        """Convert action index to board position."""
        return index // self.board_size, index % self.board_size


# Preset configurations
def get_small_config() -> GomokuConfig:
    """Configuration for small board (fast games)."""
    return GomokuConfig(board_size=9, win_length=5)


def get_standard_config() -> GomokuConfig:
    """Configuration for standard 13x13 board."""
    return GomokuConfig(board_size=13, win_length=5)


def get_large_config() -> GomokuConfig:
    """Configuration for 15x15 board."""
    return GomokuConfig(board_size=15, win_length=5)
