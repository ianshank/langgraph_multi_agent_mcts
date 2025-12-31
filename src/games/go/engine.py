"""
Go Game Engine.

Implements Go game logic including:
- Stone placement and capture
- Liberty counting
- Ko and superko detection
- Scoring (Chinese rules)

This is a pure Python implementation optimized for clarity.
For production, consider using cython or numpy-based implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Set

from .config import GoConfig


class Stone(IntEnum):
    """Stone colors on the board."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> Stone:
        """Get the opposing stone color."""
        if self == Stone.BLACK:
            return Stone.WHITE
        elif self == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


@dataclass
class Group:
    """
    A connected group of stones.

    Tracks stones in the group and their liberties.
    """

    color: Stone
    stones: set[tuple[int, int]] = field(default_factory=set)
    liberties: set[tuple[int, int]] = field(default_factory=set)

    def add_stone(self, pos: tuple[int, int]) -> None:
        """Add a stone to the group."""
        self.stones.add(pos)

    def add_liberty(self, pos: tuple[int, int]) -> None:
        """Add a liberty to the group."""
        self.liberties.add(pos)

    def remove_liberty(self, pos: tuple[int, int]) -> None:
        """Remove a liberty from the group."""
        self.liberties.discard(pos)

    @property
    def num_liberties(self) -> int:
        """Number of liberties."""
        return len(self.liberties)

    @property
    def is_captured(self) -> bool:
        """Check if the group has no liberties."""
        return len(self.liberties) == 0

    def merge(self, other: Group) -> None:
        """Merge another group into this one."""
        self.stones.update(other.stones)
        self.liberties.update(other.liberties)
        # Remove stones that are now occupied
        self.liberties -= self.stones


class GoEngine:
    """
    Go game engine implementing core game logic.

    Handles:
    - Move validation and execution
    - Capture detection
    - Ko rule enforcement
    - Game ending and scoring
    """

    def __init__(self, config: GoConfig | None = None):
        """
        Initialize Go engine.

        Args:
            config: Game configuration
        """
        self.config = config or GoConfig()
        self.size = self.config.board_size

        # Board state: 2D array of Stone values
        self.board: np.ndarray = np.zeros((self.size, self.size), dtype=np.int8)

        # Current player (BLACK moves first)
        self.current_player: Stone = Stone.BLACK

        # Move history for superko detection
        self.position_history: list[str] = []

        # Ko point (position that cannot be immediately recaptured)
        self.ko_point: tuple[int, int] | None = None

        # Game state
        self.move_count: int = 0
        self.consecutive_passes: int = 0
        self.is_game_over: bool = False
        self.winner: Stone | None = None

        # Capture counts
        self.captures: dict[Stone, int] = {Stone.BLACK: 0, Stone.WHITE: 0}

        # Group tracking for efficient liberty computation
        self._group_map: dict[tuple[int, int], Group] = {}

    def copy(self) -> GoEngine:
        """Create a deep copy of the engine state."""
        new_engine = GoEngine(self.config)
        new_engine.board = self.board.copy()
        new_engine.current_player = self.current_player
        new_engine.position_history = self.position_history.copy()
        new_engine.ko_point = self.ko_point
        new_engine.move_count = self.move_count
        new_engine.consecutive_passes = self.consecutive_passes
        new_engine.is_game_over = self.is_game_over
        new_engine.winner = self.winner
        new_engine.captures = self.captures.copy()
        # Groups will be rebuilt on demand
        return new_engine

    def get_neighbors(self, row: int, col: int) -> Iterator[tuple[int, int]]:
        """Get valid neighboring positions."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def get_stone(self, row: int, col: int) -> Stone:
        """Get stone at position."""
        return Stone(self.board[row, col])

    def set_stone(self, row: int, col: int, stone: Stone) -> None:
        """Set stone at position."""
        self.board[row, col] = stone.value

    def find_group(self, row: int, col: int) -> Group | None:
        """
        Find the group containing the stone at the given position.

        Uses flood-fill to find all connected stones.
        """
        stone = self.get_stone(row, col)
        if stone == Stone.EMPTY:
            return None

        group = Group(color=stone)
        visited: set[tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            current = self.get_stone(r, c)
            if current == stone:
                group.add_stone((r, c))
                for nr, nc in self.get_neighbors(r, c):
                    if (nr, nc) not in visited:
                        neighbor = self.get_stone(nr, nc)
                        if neighbor == stone:
                            stack.append((nr, nc))
                        elif neighbor == Stone.EMPTY:
                            group.add_liberty((nr, nc))

        return group

    def count_liberties(self, row: int, col: int) -> int:
        """Count liberties of the group at the given position."""
        group = self.find_group(row, col)
        return group.num_liberties if group else 0

    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if a move is valid.

        A move is valid if:
        1. The position is empty
        2. It doesn't violate ko rule
        3. It doesn't result in self-capture (unless allowed)
        4. It doesn't violate superko (if enabled)
        """
        # Check bounds
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False

        # Must be empty
        if self.get_stone(row, col) != Stone.EMPTY:
            return False

        # Check ko rule
        if self.ko_point == (row, col):
            return False

        # Simulate move to check for self-capture
        if not self._is_move_legal(row, col, self.current_player):
            return False

        return True

    def _is_move_legal(self, row: int, col: int, player: Stone) -> bool:
        """
        Check if a move is legal (doesn't result in self-capture).

        A move is legal if:
        - It has at least one liberty, OR
        - It captures at least one opponent group, OR
        - It connects to a friendly group with more than one liberty
        """
        opponent = player.opponent

        # Check if move has immediate liberty
        for nr, nc in self.get_neighbors(row, col):
            if self.get_stone(nr, nc) == Stone.EMPTY:
                return True

        # Check if move captures opponent stones
        for nr, nc in self.get_neighbors(row, col):
            if self.get_stone(nr, nc) == opponent:
                group = self.find_group(nr, nc)
                if group and group.num_liberties == 1:
                    # This neighbor is the only liberty, move captures
                    return True

        # Check if move connects to friendly group with >1 liberty
        for nr, nc in self.get_neighbors(row, col):
            if self.get_stone(nr, nc) == player:
                group = self.find_group(nr, nc)
                if group and group.num_liberties > 1:
                    return True

        # No liberties, no captures, and no friendly groups with spare liberties
        # This would be suicide
        return self.config.allow_suicide

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves for the current player."""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col):
                    moves.append((row, col))
        return moves

    def get_legal_action_indices(self) -> list[int]:
        """Get legal moves as action indices (including pass)."""
        indices = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_move(row, col):
                    indices.append(self.config.position_to_index(row, col))
        # Pass is always legal
        indices.append(self.config.pass_action)
        return indices

    def play_move(self, row: int, col: int) -> tuple[bool, int]:
        """
        Play a stone at the given position.

        Args:
            row: Row index
            col: Column index

        Returns:
            Tuple of (success, captures_made)
        """
        if not self.is_valid_move(row, col):
            return False, 0

        # Place stone
        self.set_stone(row, col, self.current_player)
        captures = 0
        captured_single: tuple[int, int] | None = None

        # Check for captures
        opponent = self.current_player.opponent
        for nr, nc in self.get_neighbors(row, col):
            if self.get_stone(nr, nc) == opponent:
                group = self.find_group(nr, nc)
                if group and group.is_captured:
                    # Capture the group
                    if len(group.stones) == 1:
                        captured_single = next(iter(group.stones))
                    for sr, sc in group.stones:
                        self.set_stone(sr, sc, Stone.EMPTY)
                        captures += 1

        # Update capture count
        self.captures[self.current_player] += captures

        # Set ko point if applicable
        if captures == 1 and captured_single:
            # Check if the placed stone now has exactly one liberty
            new_group = self.find_group(row, col)
            if new_group and new_group.num_liberties == 1:
                self.ko_point = captured_single
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Record position for superko
        if self.config.enable_superko:
            self.position_history.append(self._get_position_hash())

        # Update game state
        self.move_count += 1
        self.consecutive_passes = 0
        self.current_player = opponent

        return True, captures

    def play_pass(self) -> None:
        """Play a pass move."""
        self.consecutive_passes += 1
        self.move_count += 1
        self.ko_point = None
        self.current_player = self.current_player.opponent

        # Game ends after two consecutive passes
        if self.consecutive_passes >= 2:
            self.is_game_over = True
            self._determine_winner()

    def play_action(self, action_index: int) -> tuple[bool, int]:
        """
        Play a move by action index.

        Args:
            action_index: Action index (board position or pass)

        Returns:
            Tuple of (success, captures_made)
        """
        if action_index == self.config.pass_action:
            self.play_pass()
            return True, 0

        row, col = self.config.index_to_position(action_index)
        return self.play_move(row, col)

    def _get_position_hash(self) -> str:
        """Get hash of current board position."""
        return self.board.tobytes().hex()

    def _determine_winner(self) -> None:
        """Determine the winner using Chinese scoring."""
        black_score, white_score = self.score()

        if black_score > white_score:
            self.winner = Stone.BLACK
        elif white_score > black_score:
            self.winner = Stone.WHITE
        else:
            self.winner = None  # Tie

    def score(self) -> tuple[float, float]:
        """
        Calculate scores using Chinese rules (area scoring).

        Returns:
            Tuple of (black_score, white_score)
        """
        # Count stones
        black_stones = np.sum(self.board == Stone.BLACK.value)
        white_stones = np.sum(self.board == Stone.WHITE.value)

        # Count territory (empty points surrounded by one color)
        black_territory = 0
        white_territory = 0
        visited = np.zeros_like(self.board, dtype=bool)

        for row in range(self.size):
            for col in range(self.size):
                if visited[row, col] or self.get_stone(row, col) != Stone.EMPTY:
                    continue

                # Flood fill to find connected empty region
                region: set[tuple[int, int]] = set()
                borders: set[Stone] = set()
                stack = [(row, col)]

                while stack:
                    r, c = stack.pop()
                    if visited[r, c]:
                        continue
                    visited[r, c] = True

                    stone = self.get_stone(r, c)
                    if stone == Stone.EMPTY:
                        region.add((r, c))
                        for nr, nc in self.get_neighbors(r, c):
                            if not visited[nr, nc]:
                                stack.append((nr, nc))
                    else:
                        borders.add(stone)

                # Territory belongs to a color only if bordered by just that color
                if borders == {Stone.BLACK}:
                    black_territory += len(region)
                elif borders == {Stone.WHITE}:
                    white_territory += len(region)

        # Chinese scoring: stones + territory
        black_score = float(black_stones + black_territory)
        white_score = float(white_stones + white_territory + self.config.komi)

        return black_score, white_score

    def get_result(self) -> float:
        """
        Get game result from black's perspective.

        Returns:
            1.0 for black win, -1.0 for white win, 0.0 for tie or in progress
        """
        if not self.is_game_over:
            return 0.0

        if self.winner == Stone.BLACK:
            return 1.0
        elif self.winner == Stone.WHITE:
            return -1.0
        return 0.0

    def to_numpy(self) -> np.ndarray:
        """Convert board to numpy array."""
        return self.board.copy()

    def get_state_tensor(self, history_boards: list[np.ndarray] | None = None) -> np.ndarray:
        """
        Get state tensor for neural network input.

        Shape: (num_input_planes, board_size, board_size)

        Planes:
        - history_length planes for black stone positions (current to oldest)
        - history_length planes for white stone positions (current to oldest)
        - 1 plane for current player (1 if black, 0 if white)
        """
        planes = []

        # Get historical boards (or use empty boards)
        if history_boards is None:
            history_boards = []

        # Pad history if needed
        while len(history_boards) < self.config.history_length - 1:
            history_boards.insert(0, np.zeros((self.size, self.size), dtype=np.int8))

        # Use most recent boards (including current)
        boards = history_boards[-(self.config.history_length - 1) :] + [self.board]

        # Black stone planes
        for board in boards:
            planes.append((board == Stone.BLACK.value).astype(np.float32))

        # White stone planes
        for board in boards:
            planes.append((board == Stone.WHITE.value).astype(np.float32))

        # Current player plane
        player_plane = np.ones((self.size, self.size), dtype=np.float32)
        if self.current_player == Stone.WHITE:
            player_plane = np.zeros((self.size, self.size), dtype=np.float32)
        planes.append(player_plane)

        return np.stack(planes, axis=0)

    def __str__(self) -> str:
        """Get string representation of the board."""
        symbols = {Stone.EMPTY: ".", Stone.BLACK: "X", Stone.WHITE: "O"}
        lines = []

        # Column labels
        col_labels = "   " + " ".join(chr(ord("A") + i) for i in range(self.size))
        lines.append(col_labels)

        for row in range(self.size):
            row_str = f"{self.size - row:2d} "
            for col in range(self.size):
                stone = self.get_stone(row, col)
                row_str += symbols[stone] + " "
            row_str += f"{self.size - row:2d}"
            lines.append(row_str)

        lines.append(col_labels)
        return "\n".join(lines)
