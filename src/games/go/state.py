"""
Go Game State.

Implements the GameEnvironment interface for Go,
providing all methods required by the Neural MCTS framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..base import GameConfig, GameEnvironment, GameMetadata, GameRegistry, GameResult, PlayerColor
from .config import GoConfig
from .engine import GoEngine, Stone

if TYPE_CHECKING:
    pass


@dataclass(frozen=False)
class GoGameState(GameEnvironment[int]):
    """
    Go game state implementing the GameEnvironment interface.

    This class wraps the GoEngine and provides all methods required
    by the Neural MCTS framework for self-play training and inference.

    Action type is int (action indices).
    """

    # Internal engine
    _engine: GoEngine = field(default=None, repr=False)  # type: ignore
    _config: GoConfig = field(default_factory=GoConfig)

    # History for state representation
    _history: list[np.ndarray] = field(default_factory=list, repr=False)

    # Cached values
    _tensor_cache: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _hash_cache: str | None = field(default=None, repr=False, compare=False)
    _legal_actions_cache: list[int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize internal engine."""
        if self._engine is None:
            object.__setattr__(self, "_engine", GoEngine(self._config))

    @classmethod
    def initial_state(cls, config: GameConfig | None = None) -> GoGameState:
        """
        Create the initial Go game state.

        Args:
            config: Optional GoConfig

        Returns:
            Initial game state
        """
        if config is None:
            config = GoConfig()
        elif not isinstance(config, GoConfig):
            # Convert generic GameConfig to GoConfig
            config = GoConfig(board_size=config.board_size)

        engine = GoEngine(config)
        return cls(_engine=engine, _config=config)

    @classmethod
    def from_engine(cls, engine: GoEngine) -> GoGameState:
        """
        Create state from existing engine.

        Args:
            engine: GoEngine instance

        Returns:
            GoGameState wrapping the engine
        """
        return cls(_engine=engine, _config=engine.config)

    @property
    def config(self) -> GoConfig:
        """Get game configuration."""
        return self._config

    @property
    def current_player(self) -> PlayerColor:
        """Get current player to move."""
        if self._engine.current_player == Stone.BLACK:
            return PlayerColor.BLACK
        return PlayerColor.WHITE

    @property
    def metadata(self) -> GameMetadata:
        """Get game metadata."""
        return GameMetadata(
            move_number=self._engine.move_count,
            player_to_move=self.current_player,
            extra={
                "captures_black": self._engine.captures[Stone.BLACK],
                "captures_white": self._engine.captures[Stone.WHITE],
                "consecutive_passes": self._engine.consecutive_passes,
            },
        )

    @property
    def engine(self) -> GoEngine:
        """Get the underlying engine."""
        return self._engine

    @property
    def board_size(self) -> int:
        """Get board size."""
        return self._config.board_size

    def get_legal_actions(self) -> list[int]:
        """Get all legal action indices."""
        if self._legal_actions_cache is None:
            self._legal_actions_cache = self._engine.get_legal_action_indices()
        return self._legal_actions_cache

    def apply_action(self, action: int) -> GoGameState:
        """
        Apply an action and return a new state.

        Args:
            action: Action index (board position or pass)

        Returns:
            New game state after applying the action
        """
        # Create a copy of the engine
        new_engine = self._engine.copy()

        # Apply the action
        success, _ = new_engine.play_action(action)
        if not success:
            raise ValueError(f"Invalid action: {action}")

        # Update history
        new_history = self._history.copy()
        new_history.append(self._engine.board.copy())

        # Keep only recent history
        if len(new_history) > self._config.history_length:
            new_history = new_history[-self._config.history_length :]

        return GoGameState(
            _engine=new_engine,
            _config=self._config,
            _history=new_history,
        )

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        if self._engine.is_game_over:
            return True

        # Also check for maximum game length
        if self._engine.move_count >= self._config.max_game_length:
            return True

        return False

    def get_result(self) -> GameResult:
        """Get the game result."""
        if not self.is_terminal():
            return GameResult.IN_PROGRESS

        # Determine winner
        result = self._engine.get_result()
        if result > 0:
            return GameResult.BLACK_WIN
        elif result < 0:
            return GameResult.WHITE_WIN
        return GameResult.DRAW

    def get_reward(self, player: PlayerColor) -> float:
        """
        Get reward for specified player.

        Args:
            player: Player to get reward for

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or in-progress
        """
        if not self.is_terminal():
            return 0.0

        result = self._engine.get_result()

        if player == PlayerColor.BLACK:
            return result
        else:
            return -result

    def to_tensor(self) -> torch.Tensor:
        """
        Convert state to tensor for neural network input.

        Returns:
            Tensor of shape (num_planes, board_size, board_size)
        """
        if self._tensor_cache is not None:
            return self._tensor_cache

        state_array = self._engine.get_state_tensor(self._history)
        tensor = torch.from_numpy(state_array)
        self._tensor_cache = tensor
        return tensor

    def get_canonical_state(self) -> GoGameState:
        """
        Get state from current player's perspective.

        For Go, we don't flip the board as both players
        see the same perspective. However, we track who
        is to move in the state representation.

        Returns:
            Self (Go doesn't require perspective flipping)
        """
        return self

    def get_state_hash(self) -> str:
        """
        Get unique hash for this state.

        Returns:
            Hash string
        """
        if self._hash_cache is not None:
            return self._hash_cache

        # Include board state and current player
        board_hash = self._engine.board.tobytes().hex()
        player = "B" if self._engine.current_player == Stone.BLACK else "W"
        self._hash_cache = f"{board_hash}_{player}"
        return self._hash_cache

    def action_to_index(self, action: int) -> int:
        """
        Convert action to neural network output index.

        For Go, actions are already indices.
        """
        return action

    def index_to_action(self, index: int) -> int:
        """
        Convert neural network output index to action.

        For Go, indices are already actions.
        """
        return index

    def get_action_mask(self) -> np.ndarray:
        """
        Get boolean mask of legal actions.

        Returns:
            Boolean array of shape (action_size,)
        """
        mask = np.zeros(self._config.action_size, dtype=bool)
        for action in self.get_legal_actions():
            mask[action] = True
        return mask

    def copy(self) -> GoGameState:
        """Create a deep copy of this state."""
        return GoGameState(
            _engine=self._engine.copy(),
            _config=self._config,
            _history=self._history.copy(),
        )

    def get_symmetries(
        self,
        policy: np.ndarray,
    ) -> list[tuple[GoGameState, np.ndarray]]:
        """
        Get symmetrically equivalent states and policies.

        Go has 8 symmetries (4 rotations x 2 reflections).

        Args:
            policy: Policy probabilities

        Returns:
            List of (symmetric_state, symmetric_policy) tuples
        """
        symmetries = []
        board = self._engine.board
        size = self.board_size

        # Reshape policy to board + pass
        policy_board = policy[:-1].reshape(size, size)
        policy_pass = policy[-1]

        for k in range(4):  # 4 rotations
            for flip in [False, True]:  # 2 reflections
                # Transform board
                new_board = np.rot90(board, k)
                new_policy_board = np.rot90(policy_board, k)

                if flip:
                    new_board = np.fliplr(new_board)
                    new_policy_board = np.fliplr(new_policy_board)

                # Create new state with transformed board
                new_engine = self._engine.copy()
                new_engine.board = new_board.copy()

                # Flatten policy and add pass action
                new_policy = np.append(new_policy_board.flatten(), policy_pass)

                new_state = GoGameState(
                    _engine=new_engine,
                    _config=self._config,
                    _history=[np.rot90(h, k) for h in self._history],
                )

                symmetries.append((new_state, new_policy))

        return symmetries

    def render(self) -> str:
        """Get human-readable string representation."""
        lines = [str(self._engine)]
        lines.append(f"Move: {self._engine.move_count}")
        lines.append(f"To play: {'Black' if self.current_player == PlayerColor.BLACK else 'White'}")

        if self.is_terminal():
            black_score, white_score = self._engine.score()
            lines.append(f"Score: Black {black_score} - White {white_score}")
            lines.append(f"Result: {self.get_result().name}")

        return "\n".join(lines)

    def validate_action(self, action: int) -> bool:
        """Check if an action is valid."""
        return action in self.get_legal_actions()

    def get_score(self) -> tuple[float, float]:
        """
        Get current score.

        Returns:
            Tuple of (black_score, white_score)
        """
        return self._engine.score()

    def __str__(self) -> str:
        """String representation."""
        return self.render()

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"GoGameState(size={self.board_size}, "
            f"move={self._engine.move_count}, "
            f"player={self.current_player.name})"
        )


# -------------------- Factory Functions --------------------


def create_go_state(
    board_size: int = 9,
    komi: float | None = None,
) -> GoGameState:
    """
    Create a new Go game state.

    Args:
        board_size: Board size (9, 13, or 19)
        komi: Komi compensation (default based on board size)

    Returns:
        Initial Go game state
    """
    config = GoConfig(board_size=board_size)
    if komi is not None:
        config.komi = komi
    return GoGameState.initial_state(config)


def create_go_state_from_sgf(sgf_content: str) -> GoGameState:
    """
    Create Go state from SGF string.

    Args:
        sgf_content: SGF format game record

    Returns:
        Go game state at the end of the SGF

    Note:
        This is a simplified SGF parser. For production use,
        consider using a dedicated SGF library.
    """
    # Parse board size
    import re

    size_match = re.search(r"SZ\[(\d+)\]", sgf_content)
    board_size = int(size_match.group(1)) if size_match else 19

    # Parse komi
    komi_match = re.search(r"KM\[([0-9.]+)\]", sgf_content)
    komi = float(komi_match.group(1)) if komi_match else 6.5

    # Create initial state
    state = create_go_state(board_size=board_size, komi=komi)

    # Parse moves
    move_pattern = re.compile(r";([BW])\[([a-s]{2})?\]")
    for match in move_pattern.finditer(sgf_content):
        _color = match.group(1)
        pos = match.group(2)

        if pos:
            # Convert SGF coordinates to indices
            col = ord(pos[0]) - ord("a")
            row = ord(pos[1]) - ord("a")
            action = state.config.position_to_index(row, col)
        else:
            # Pass
            action = state.config.pass_action

        if action in state.get_legal_actions():
            state = state.apply_action(action)

    return state


# -------------------- Register with GameRegistry --------------------

# Register Go game
GameRegistry.register("go", GoGameState, GoConfig)
