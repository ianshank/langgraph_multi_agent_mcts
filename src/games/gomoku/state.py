"""
Gomoku Game State.

Implements the GameEnvironment interface for Gomoku,
providing all methods required by the Neural MCTS framework.

Gomoku (Five in a Row) is simpler than Go:
- No captures
- No ko rules
- First to get 5 in a row wins
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..base import GameConfig, GameEnvironment, GameMetadata, GameRegistry, GameResult, PlayerColor
from .config import GomokuConfig

if TYPE_CHECKING:
    pass


class Stone(IntEnum):
    """Stone colors."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> Stone:
        """Get opposing color."""
        if self == Stone.BLACK:
            return Stone.WHITE
        elif self == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


@dataclass(frozen=False)
class GomokuGameState(GameEnvironment[int]):
    """
    Gomoku game state implementing GameEnvironment interface.

    Simple and fast implementation suitable for testing AlphaZero.
    """

    # Board state
    _board: np.ndarray = field(default=None, repr=False)  # type: ignore
    _config: GomokuConfig = field(default_factory=GomokuConfig)

    # Current player
    _current_player: Stone = Stone.BLACK

    # Move history
    _history: list[np.ndarray] = field(default_factory=list, repr=False)
    _move_count: int = 0
    _last_move: int | None = None

    # Game state
    _is_terminal: bool = False
    _winner: Stone | None = None

    # Caches
    _tensor_cache: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _hash_cache: str | None = field(default=None, repr=False, compare=False)
    _legal_actions_cache: list[int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize board."""
        if self._board is None:
            size = self._config.board_size
            object.__setattr__(self, "_board", np.zeros((size, size), dtype=np.int8))

    @classmethod
    def initial_state(cls, config: GameConfig | None = None) -> GomokuGameState:
        """Create initial game state."""
        if config is None:
            config = GomokuConfig()
        elif not isinstance(config, GomokuConfig):
            config = GomokuConfig(board_size=config.board_size)

        size = config.board_size
        return cls(
            _board=np.zeros((size, size), dtype=np.int8),
            _config=config,
        )

    @property
    def config(self) -> GomokuConfig:
        """Get game configuration."""
        return self._config

    @property
    def current_player(self) -> PlayerColor:
        """Get current player."""
        return PlayerColor.BLACK if self._current_player == Stone.BLACK else PlayerColor.WHITE

    @property
    def metadata(self) -> GameMetadata:
        """Get game metadata."""
        return GameMetadata(
            move_number=self._move_count,
            player_to_move=self.current_player,
            last_action=self._last_move,
        )

    @property
    def board_size(self) -> int:
        """Get board size."""
        return self._config.board_size

    def get_legal_actions(self) -> list[int]:
        """Get all legal actions (empty positions)."""
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache

        if self._is_terminal:
            self._legal_actions_cache = []
            return []

        # All empty positions are legal
        legal = []
        size = self.board_size
        for row in range(size):
            for col in range(size):
                if self._board[row, col] == Stone.EMPTY:
                    legal.append(self._config.position_to_index(row, col))

        self._legal_actions_cache = legal
        return legal

    def apply_action(self, action: int) -> GomokuGameState:
        """Apply action and return new state."""
        if action not in self.get_legal_actions():
            raise ValueError(f"Illegal action: {action}")

        row, col = self._config.index_to_position(action)

        # Copy board
        new_board = self._board.copy()
        new_board[row, col] = self._current_player.value

        # Update history
        new_history = self._history.copy()
        new_history.append(self._board.copy())
        if len(new_history) > self._config.history_length:
            new_history = new_history[-self._config.history_length :]

        # Check for win
        winner = self._check_winner(new_board, row, col, self._current_player)

        # Check for draw (board full)
        is_terminal = winner is not None or np.all(new_board != Stone.EMPTY)

        return GomokuGameState(
            _board=new_board,
            _config=self._config,
            _current_player=self._current_player.opponent,
            _history=new_history,
            _move_count=self._move_count + 1,
            _last_move=action,
            _is_terminal=is_terminal,
            _winner=winner,
        )

    def _check_winner(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        player: Stone,
    ) -> Stone | None:
        """
        Check if the last move resulted in a win.

        Args:
            board: Current board state
            row: Row of last move
            col: Column of last move
            player: Player who made the move

        Returns:
            Winning player or None
        """
        win_length = self._config.win_length
        size = self.board_size

        # Directions to check: horizontal, vertical, diagonal, anti-diagonal
        directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal
            (1, -1),  # anti-diagonal
        ]

        for dr, dc in directions:
            count = 1  # Count the placed stone

            # Count in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Count in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            # Check if we have enough
            if self._config.variant == "freestyle":
                if count >= win_length:
                    return player
            else:
                # Standard: exactly win_length (overlines don't count)
                if count == win_length:
                    return player

        return None

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self._is_terminal

    def get_result(self) -> GameResult:
        """Get game result."""
        if not self._is_terminal:
            return GameResult.IN_PROGRESS

        if self._winner == Stone.BLACK:
            return GameResult.BLACK_WIN
        elif self._winner == Stone.WHITE:
            return GameResult.WHITE_WIN
        return GameResult.DRAW

    def get_reward(self, player: PlayerColor) -> float:
        """Get reward for specified player."""
        if not self._is_terminal:
            return 0.0

        if self._winner is None:
            return 0.0  # Draw

        if player == PlayerColor.BLACK:
            return 1.0 if self._winner == Stone.BLACK else -1.0
        else:
            return 1.0 if self._winner == Stone.WHITE else -1.0

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor."""
        if self._tensor_cache is not None:
            return self._tensor_cache

        planes = []
        size = self.board_size

        # Get historical boards
        boards = self._history + [self._board]
        # Pad if needed
        while len(boards) < self._config.history_length:
            boards.insert(0, np.zeros((size, size), dtype=np.int8))
        # Use most recent
        boards = boards[-self._config.history_length :]

        # Black stone planes
        for board in boards:
            planes.append((board == Stone.BLACK).astype(np.float32))

        # White stone planes
        for board in boards:
            planes.append((board == Stone.WHITE).astype(np.float32))

        # Current player plane
        player_plane = np.ones((size, size), dtype=np.float32)
        if self._current_player == Stone.WHITE:
            player_plane = np.zeros((size, size), dtype=np.float32)
        planes.append(player_plane)

        tensor = torch.from_numpy(np.stack(planes, axis=0))
        self._tensor_cache = tensor
        return tensor

    def get_canonical_state(self) -> GomokuGameState:
        """Get state from current player's perspective."""
        return self

    def get_state_hash(self) -> str:
        """Get unique hash for this state."""
        if self._hash_cache is not None:
            return self._hash_cache

        board_hash = self._board.tobytes().hex()
        player = "B" if self._current_player == Stone.BLACK else "W"
        self._hash_cache = f"{board_hash}_{player}"
        return self._hash_cache

    def action_to_index(self, action: int) -> int:
        """Convert action to index."""
        return action

    def index_to_action(self, index: int) -> int:
        """Convert index to action."""
        return index

    def get_action_mask(self) -> np.ndarray:
        """Get boolean mask of legal actions."""
        mask = np.zeros(self._config.action_size, dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask

    def copy(self) -> GomokuGameState:
        """Create a deep copy."""
        return GomokuGameState(
            _board=self._board.copy(),
            _config=self._config,
            _current_player=self._current_player,
            _history=self._history.copy(),
            _move_count=self._move_count,
            _last_move=self._last_move,
            _is_terminal=self._is_terminal,
            _winner=self._winner,
        )

    def get_symmetries(
        self,
        policy: np.ndarray,
    ) -> list[tuple[GomokuGameState, np.ndarray]]:
        """
        Get symmetrically equivalent states.

        Gomoku has 8 symmetries like Go.
        """
        symmetries = []
        size = self.board_size
        policy_board = policy.reshape(size, size)

        for k in range(4):
            for flip in [False, True]:
                new_board = np.rot90(self._board, k)
                new_policy_board = np.rot90(policy_board, k)

                if flip:
                    new_board = np.fliplr(new_board)
                    new_policy_board = np.fliplr(new_policy_board)

                new_state = GomokuGameState(
                    _board=new_board.copy(),
                    _config=self._config,
                    _current_player=self._current_player,
                    _history=[np.rot90(h, k) for h in self._history],
                    _move_count=self._move_count,
                    _is_terminal=self._is_terminal,
                    _winner=self._winner,
                )

                symmetries.append((new_state, new_policy_board.flatten()))

        return symmetries

    def render(self) -> str:
        """Get string representation."""
        symbols = {Stone.EMPTY: ".", Stone.BLACK: "X", Stone.WHITE: "O"}
        lines = []

        # Column labels
        col_labels = "   " + " ".join(chr(ord("A") + i) for i in range(self.board_size))
        lines.append(col_labels)

        for row in range(self.board_size):
            row_str = f"{self.board_size - row:2d} "
            for col in range(self.board_size):
                stone = Stone(self._board[row, col])
                row_str += symbols[stone] + " "
            row_str += f"{self.board_size - row:2d}"
            lines.append(row_str)

        lines.append(col_labels)
        lines.append(f"Move: {self._move_count}, To play: {self.current_player.name}")

        if self._is_terminal:
            lines.append(f"Result: {self.get_result().name}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"GomokuGameState(size={self.board_size}, "
            f"move={self._move_count}, "
            f"player={self.current_player.name})"
        )


# Factory functions
def create_gomoku_state(
    board_size: int = 13,
    win_length: int = 5,
) -> GomokuGameState:
    """
    Create a new Gomoku game state.

    Args:
        board_size: Board size
        win_length: Stones in a row to win

    Returns:
        Initial Gomoku game state
    """
    config = GomokuConfig(board_size=board_size, win_length=win_length)
    return GomokuGameState.initial_state(config)


# Register with GameRegistry
GameRegistry.register("gomoku", GomokuGameState, GomokuConfig)
