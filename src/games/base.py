"""
Abstract Game Environment Interface.

Provides a game-agnostic interface for AlphaZero-style self-play training.
All game implementations (Chess, Go, Gomoku, etc.) should inherit from this base.

This design follows the michaelnny/alpha_zero architecture while maintaining
backwards compatibility with the existing GameState interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


class GameResult(Enum):
    """Possible game outcomes."""

    WHITE_WIN = auto()
    BLACK_WIN = auto()
    DRAW = auto()
    IN_PROGRESS = auto()


class PlayerColor(Enum):
    """Player colors/sides."""

    WHITE = 1
    BLACK = -1

    @property
    def opponent(self) -> PlayerColor:
        """Get the opposing player."""
        return PlayerColor.BLACK if self == PlayerColor.WHITE else PlayerColor.WHITE


# Type variable for action types (can be int, str, tuple, etc.)
ActionT = TypeVar("ActionT")


@dataclass
class GameConfig:
    """
    Base configuration for game environments.

    Subclass this for game-specific configurations.
    """

    # Board dimensions
    board_size: int = 19

    # Action space
    action_size: int = 362  # board_size^2 + 1 for pass

    # State representation
    num_input_planes: int = 17  # Number of input channels for neural network

    # History length for state representation
    history_length: int = 8

    # Game-specific rules
    allow_pass: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, got {self.board_size}")
        if self.action_size <= 0:
            raise ValueError(f"action_size must be positive, got {self.action_size}")


@dataclass
class GameMetadata:
    """
    Metadata for a game state.

    Useful for tracking game history, debugging, and analysis.
    """

    move_number: int = 0
    game_id: str = ""
    player_to_move: PlayerColor = PlayerColor.WHITE
    last_action: Any | None = None
    is_resignation: bool = False
    extra: dict = field(default_factory=dict)


class GameEnvironment(ABC, Generic[ActionT]):
    """
    Abstract base class for game environments.

    Provides a unified interface for AlphaZero-style self-play training.
    All game implementations should inherit from this class.

    This interface is designed to be:
    - Game-agnostic: Works with any two-player zero-sum game
    - Immutable: apply_action returns a new state, original unchanged
    - Efficient: Supports caching and lazy evaluation
    - Compatible: Works with existing GameState interface via adapter

    Type Parameters:
        ActionT: The type used to represent actions (int, str, tuple, etc.)
    """

    @property
    @abstractmethod
    def config(self) -> GameConfig:
        """Get the game configuration."""
        ...

    @property
    @abstractmethod
    def current_player(self) -> PlayerColor:
        """Get the current player to move."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> GameMetadata:
        """Get game metadata."""
        ...

    @abstractmethod
    def get_legal_actions(self) -> list[ActionT]:
        """
        Get all legal actions from current state.

        Returns:
            List of legal actions
        """
        ...

    @abstractmethod
    def apply_action(self, action: ActionT) -> GameEnvironment[ActionT]:
        """
        Apply an action and return a new state.

        The original state should remain unchanged (immutability).

        Args:
            action: Action to apply

        Returns:
            New game state after applying the action

        Raises:
            ValueError: If action is illegal
        """
        ...

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the game is over.

        Returns:
            True if game is over (win, loss, or draw)
        """
        ...

    @abstractmethod
    def get_result(self) -> GameResult:
        """
        Get the game result.

        Returns:
            GameResult enum value
        """
        ...

    @abstractmethod
    def get_reward(self, player: PlayerColor) -> float:
        """
        Get reward for specified player.

        Args:
            player: Player to get reward for

        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw or in-progress
        """
        ...

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """
        Convert state to tensor for neural network input.

        Returns:
            Tensor of shape (num_planes, board_size, board_size)
        """
        ...

    @abstractmethod
    def get_canonical_state(self) -> GameEnvironment[ActionT]:
        """
        Get state from current player's perspective.

        For many games, this flips the board when it's the second player's turn.

        Returns:
            Canonical state representation
        """
        ...

    @abstractmethod
    def get_state_hash(self) -> str:
        """
        Get unique hash for this state.

        Used for transposition detection and MCTS node caching.

        Returns:
            Unique hash string
        """
        ...

    @abstractmethod
    def action_to_index(self, action: ActionT) -> int:
        """
        Convert action to neural network output index.

        Args:
            action: Action to convert

        Returns:
            Index in range [0, action_size)
        """
        ...

    @abstractmethod
    def index_to_action(self, index: int) -> ActionT:
        """
        Convert neural network output index to action.

        Args:
            index: Index in range [0, action_size)

        Returns:
            Corresponding action
        """
        ...

    @abstractmethod
    def get_action_mask(self) -> np.ndarray:
        """
        Get boolean mask of legal actions.

        Returns:
            Boolean array of shape (action_size,) where True = legal
        """
        ...

    @abstractmethod
    def copy(self) -> GameEnvironment[ActionT]:
        """
        Create a deep copy of this state.

        Returns:
            New GameEnvironment with identical state
        """
        ...

    @classmethod
    @abstractmethod
    def initial_state(cls, config: GameConfig | None = None) -> GameEnvironment[ActionT]:
        """
        Create the initial game state.

        Args:
            config: Optional game configuration

        Returns:
            Initial game state
        """
        ...

    # -------------------- Optional Methods (with default implementations) --------------------

    def get_symmetries(self, policy: np.ndarray) -> list[tuple[GameEnvironment[ActionT], np.ndarray]]:
        """
        Get symmetrically equivalent states and policies.

        Used for data augmentation during training.
        Default implementation returns only the original state.

        Args:
            policy: Policy probabilities for current state

        Returns:
            List of (symmetric_state, symmetric_policy) tuples
        """
        return [(self, policy)]

    def render(self) -> str:
        """
        Get human-readable string representation.

        Returns:
            String representation of the game state
        """
        return str(self)

    def get_observation(self) -> np.ndarray:
        """
        Get observation as numpy array.

        Default implementation converts tensor to numpy.

        Returns:
            Observation array
        """
        return self.to_tensor().numpy()

    def validate_action(self, action: ActionT) -> bool:
        """
        Check if an action is valid.

        Args:
            action: Action to validate

        Returns:
            True if action is legal
        """
        return action in self.get_legal_actions()

    # -------------------- Compatibility Methods --------------------

    def get_legal_actions_as_indices(self) -> list[int]:
        """
        Get legal actions as neural network indices.

        Returns:
            List of action indices
        """
        return [self.action_to_index(a) for a in self.get_legal_actions()]

    def apply_action_by_index(self, index: int) -> GameEnvironment[ActionT]:
        """
        Apply action using neural network index.

        Args:
            index: Action index

        Returns:
            New game state
        """
        action = self.index_to_action(index)
        return self.apply_action(action)

    # -------------------- GameState Compatibility Layer --------------------

    def to_game_state(self) -> "GameStateAdapter":
        """
        Convert to legacy GameState interface.

        Returns:
            GameStateAdapter wrapping this environment
        """
        return GameStateAdapter(self)


class GameStateAdapter:
    """
    Adapter to make GameEnvironment compatible with legacy GameState interface.

    This enables backwards compatibility with existing MCTS and training code.
    """

    def __init__(self, env: GameEnvironment):
        self._env = env

    @property
    def env(self) -> GameEnvironment:
        """Get underlying game environment."""
        return self._env

    def get_legal_actions(self) -> list[Any]:
        """Return list of legal actions from this state."""
        return self._env.get_legal_actions()

    def apply_action(self, action: Any) -> "GameStateAdapter":
        """Apply action and return new state."""
        new_env = self._env.apply_action(action)
        return GameStateAdapter(new_env)

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self._env.is_terminal()

    def get_reward(self, player: int = 1) -> float:
        """Get reward for the player (1 or -1)."""
        player_color = PlayerColor.WHITE if player == 1 else PlayerColor.BLACK
        return self._env.get_reward(player_color)

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        return self._env.to_tensor()

    def get_canonical_form(self, player: int) -> "GameStateAdapter":
        """Get state from perspective of given player."""
        if player == self._env.current_player.value:
            return self
        return GameStateAdapter(self._env.get_canonical_state())

    def get_hash(self) -> str:
        """Get unique hash for this state (for caching)."""
        return self._env.get_state_hash()

    def action_to_index(self, action: Any) -> int:
        """Map action to its index in the neural network's action space."""
        return self._env.action_to_index(action)


# -------------------- Factory Pattern for Game Creation --------------------


class GameRegistry:
    """
    Registry for game environments.

    Allows dynamic game creation by name, supporting extensibility.
    """

    _games: dict[str, type[GameEnvironment]] = {}
    _configs: dict[str, type[GameConfig]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        game_class: type[GameEnvironment],
        config_class: type[GameConfig] | None = None,
    ) -> None:
        """
        Register a game environment.

        Args:
            name: Game identifier (e.g., "go", "chess", "gomoku")
            game_class: GameEnvironment subclass
            config_class: Optional GameConfig subclass
        """
        cls._games[name.lower()] = game_class
        if config_class is not None:
            cls._configs[name.lower()] = config_class

    @classmethod
    def create(cls, name: str, config: GameConfig | None = None) -> GameEnvironment:
        """
        Create a game environment by name.

        Args:
            name: Game identifier
            config: Optional game configuration

        Returns:
            New game environment at initial state

        Raises:
            ValueError: If game name is not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._games:
            available = ", ".join(cls._games.keys())
            raise ValueError(f"Unknown game: {name}. Available: {available}")

        game_class = cls._games[name_lower]
        return game_class.initial_state(config)

    @classmethod
    def get_default_config(cls, name: str) -> GameConfig:
        """
        Get default configuration for a game.

        Args:
            name: Game identifier

        Returns:
            Default GameConfig for the game
        """
        name_lower = name.lower()
        if name_lower in cls._configs:
            return cls._configs[name_lower]()
        return GameConfig()

    @classmethod
    def list_games(cls) -> list[str]:
        """
        List all registered games.

        Returns:
            List of game names
        """
        return list(cls._games.keys())


# -------------------- Utility Functions --------------------


def create_initial_state_fn(
    game_name: str,
    config: GameConfig | None = None,
) -> Callable[[], GameEnvironment]:
    """
    Create a factory function for initial game states.

    Useful for passing to training pipelines.

    Args:
        game_name: Name of the game
        config: Optional game configuration

    Returns:
        Function that creates initial game states
    """

    def factory() -> GameEnvironment:
        return GameRegistry.create(game_name, config)

    return factory


def validate_game_implementation(game_class: type[GameEnvironment]) -> list[str]:
    """
    Validate that a game implementation is correct.

    Performs basic sanity checks on the game implementation.

    Args:
        game_class: GameEnvironment subclass to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    try:
        # Create initial state
        state = game_class.initial_state()

        # Check basic properties
        if state.is_terminal():
            errors.append("Initial state should not be terminal")

        legal_actions = state.get_legal_actions()
        if not legal_actions:
            errors.append("Initial state should have legal actions")

        # Check action application
        if legal_actions:
            action = legal_actions[0]
            try:
                new_state = state.apply_action(action)
                if new_state is state:
                    errors.append("apply_action should return a new state, not modify in-place")
            except Exception as e:
                errors.append(f"apply_action failed: {e}")

        # Check tensor conversion
        try:
            tensor = state.to_tensor()
            if tensor.dim() != 3:
                errors.append(f"to_tensor should return 3D tensor, got {tensor.dim()}D")
        except Exception as e:
            errors.append(f"to_tensor failed: {e}")

        # Check action mask
        try:
            mask = state.get_action_mask()
            if mask.sum() != len(legal_actions):
                errors.append(
                    f"Action mask has {mask.sum()} legal actions, " f"but get_legal_actions returned {len(legal_actions)}"
                )
        except Exception as e:
            errors.append(f"get_action_mask failed: {e}")

        # Check action conversion roundtrip
        if legal_actions:
            action = legal_actions[0]
            try:
                index = state.action_to_index(action)
                recovered = state.index_to_action(index)
                if recovered != action:
                    errors.append(f"Action conversion roundtrip failed: {action} -> {index} -> {recovered}")
            except Exception as e:
                errors.append(f"Action conversion failed: {e}")

    except Exception as e:
        errors.append(f"Failed to create initial state: {e}")

    return errors
