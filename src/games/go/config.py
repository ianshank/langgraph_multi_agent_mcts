"""
Go Game Configuration.

Provides configurable parameters for the Go game environment.
All values have sensible defaults that can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import GameConfig


@dataclass
class GoConfig(GameConfig):
    """
    Configuration for Go game environment.

    Supports standard board sizes: 9x9, 13x13, 19x19
    """

    # Board size (9, 13, or 19 are standard)
    board_size: int = 9

    # Komi (compensation for black's first-move advantage)
    # Standard komi: 5.5 for 9x9, 6.5 for 13x13 and 19x19
    komi: float = 5.5

    # History planes for state representation
    # AlphaGo Zero used 8 planes of history
    history_length: int = 8

    # Superko rule: prevent any repeated board position
    enable_superko: bool = True

    # Allow suicide moves (self-capture)
    allow_suicide: bool = False

    # Maximum game length (proportional to board area)
    max_game_length_factor: float = 2.0

    # Scoring method: "chinese" (area) or "japanese" (territory)
    scoring_method: str = "chinese"

    def __post_init__(self):
        """Validate and compute derived values."""
        if self.board_size not in (5, 7, 9, 11, 13, 15, 17, 19):
            raise ValueError(f"board_size must be odd between 5-19, got {self.board_size}")

        # Compute action size: board_size^2 + 1 for pass
        self.action_size = self.board_size * self.board_size + 1

        # Compute input planes:
        # - history_length planes for black stones history
        # - history_length planes for white stones history
        # - 1 plane for current player (all 1s if black, all 0s if white)
        self.num_input_planes = 2 * self.history_length + 1

        # Set appropriate komi based on board size if not explicitly set
        if self.komi == 5.5 and self.board_size > 9:
            self.komi = 6.5 if self.board_size >= 13 else 5.5

        # Compute max game length
        self.max_game_length = int(self.board_size * self.board_size * self.max_game_length_factor)

        self.allow_pass = True

    @property
    def pass_action(self) -> int:
        """Get the pass action index."""
        return self.board_size * self.board_size

    def position_to_index(self, row: int, col: int) -> int:
        """Convert board position to action index."""
        return row * self.board_size + col

    def index_to_position(self, index: int) -> tuple[int, int]:
        """Convert action index to board position."""
        if index == self.pass_action:
            return -1, -1  # Pass
        return index // self.board_size, index % self.board_size


# Preset configurations for common Go variants
def get_9x9_config() -> GoConfig:
    """Configuration for 9x9 Go (fast games)."""
    return GoConfig(board_size=9, komi=5.5)


def get_13x13_config() -> GoConfig:
    """Configuration for 13x13 Go (intermediate)."""
    return GoConfig(board_size=13, komi=6.5)


def get_19x19_config() -> GoConfig:
    """Configuration for 19x19 Go (full game)."""
    return GoConfig(board_size=19, komi=6.5)
