"""
Chess GameEnvironment Adapter.

Adapts the existing ChessGameState to the new GameEnvironment interface
for unified game-agnostic training.

This provides backwards compatibility while enabling the chess implementation
to be used with the new AlphaZero training infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..base import GameConfig, GameEnvironment, GameMetadata, GameRegistry, GameResult, PlayerColor
from .config import ChessActionSpaceConfig, ChessBoardConfig, ChessConfig, GamePhase
from .state import ChessGameState

if TYPE_CHECKING:
    pass


@dataclass
class ChessGameConfig(GameConfig):
    """
    GameConfig adapter for chess.

    Wraps ChessConfig to provide GameConfig interface.
    """

    # Reference to full chess config
    chess_config: ChessConfig = field(default_factory=ChessConfig)

    def __post_init__(self):
        """Initialize from chess config."""
        self.board_size = self.chess_config.board.board_size
        self.action_size = self.chess_config.action_size
        self.num_input_planes = self.chess_config.input_channels
        self.history_length = self.chess_config.board.history_length
        self.allow_pass = False  # Chess doesn't have pass


class ChessEnvironment(GameEnvironment[str]):
    """
    Chess game environment implementing GameEnvironment interface.

    Wraps ChessGameState to provide unified interface for training.
    Action type is str (UCI move notation like "e2e4").
    """

    def __init__(
        self,
        state: ChessGameState | None = None,
        config: ChessGameConfig | None = None,
    ):
        """
        Initialize chess environment.

        Args:
            state: Existing ChessGameState or None for initial position
            config: Game configuration
        """
        if config is None:
            config = ChessGameConfig()

        self._config = config
        self._state = state or ChessGameState.initial(
            board_config=config.chess_config.board,
            action_config=config.chess_config.action_space,
        )

        # Cache for expensive operations
        self._tensor_cache: torch.Tensor | None = None
        self._hash_cache: str | None = None
        self._legal_actions_cache: list[str] | None = None

    @classmethod
    def initial_state(cls, config: GameConfig | None = None) -> ChessEnvironment:
        """Create initial chess position."""
        if config is None:
            game_config = ChessGameConfig()
        elif isinstance(config, ChessGameConfig):
            game_config = config
        else:
            # Convert generic GameConfig
            game_config = ChessGameConfig()

        return cls(config=game_config)

    @classmethod
    def from_fen(
        cls,
        fen: str,
        config: ChessGameConfig | None = None,
    ) -> ChessEnvironment:
        """
        Create chess environment from FEN string.

        Args:
            fen: FEN string representation
            config: Optional configuration

        Returns:
            ChessEnvironment at specified position
        """
        if config is None:
            config = ChessGameConfig()

        state = ChessGameState.from_fen(
            fen=fen,
            board_config=config.chess_config.board,
            action_config=config.chess_config.action_space,
        )
        return cls(state=state, config=config)

    @property
    def config(self) -> ChessGameConfig:
        """Get game configuration."""
        return self._config

    @property
    def current_player(self) -> PlayerColor:
        """Get current player to move."""
        return PlayerColor.WHITE if self._state.current_player == 1 else PlayerColor.BLACK

    @property
    def metadata(self) -> GameMetadata:
        """Get game metadata."""
        return GameMetadata(
            move_number=self._state.move_number,
            player_to_move=self.current_player,
            extra={
                "fen": self._state.fen,
                "phase": self._state.get_game_phase().value,
                "material_balance": self._state.get_material_balance(),
                "halfmove_clock": self._state.halfmove_clock,
            },
        )

    @property
    def state(self) -> ChessGameState:
        """Get underlying ChessGameState."""
        return self._state

    @property
    def fen(self) -> str:
        """Get FEN string for current position."""
        return self._state.fen

    @property
    def game_phase(self) -> GamePhase:
        """Get current game phase."""
        return self._state.get_game_phase()

    def get_legal_actions(self) -> list[str]:
        """Get all legal moves in UCI format."""
        if self._legal_actions_cache is None:
            self._legal_actions_cache = self._state.get_legal_actions()
        return self._legal_actions_cache

    def apply_action(self, action: str) -> ChessEnvironment:
        """
        Apply a move and return new environment.

        Args:
            action: Move in UCI format (e.g., "e2e4")

        Returns:
            New ChessEnvironment after the move
        """
        new_state = self._state.apply_action(action)
        return ChessEnvironment(state=new_state, config=self._config)

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self._state.is_terminal()

    def get_result(self) -> GameResult:
        """Get game result."""
        if not self.is_terminal():
            return GameResult.IN_PROGRESS

        reward = self._state.get_reward(player=1)  # From white's perspective
        if reward > 0:
            return GameResult.WHITE_WIN
        elif reward < 0:
            return GameResult.BLACK_WIN
        return GameResult.DRAW

    def get_reward(self, player: PlayerColor) -> float:
        """Get reward for specified player."""
        player_int = 1 if player == PlayerColor.WHITE else -1
        return self._state.get_reward(player=player_int)

    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        if self._tensor_cache is None:
            self._tensor_cache = self._state.to_tensor()
        return self._tensor_cache

    def get_canonical_state(self) -> ChessEnvironment:
        """
        Get state from current player's perspective.

        For chess, the board representation already accounts
        for the current player in the encoding.
        """
        return self

    def get_state_hash(self) -> str:
        """Get unique hash for this state."""
        if self._hash_cache is None:
            self._hash_cache = self._state.get_hash()
        return self._hash_cache

    def action_to_index(self, action: str) -> int:
        """Convert UCI move to neural network output index."""
        return self._state.action_to_index(action)

    def index_to_action(self, index: int) -> str:
        """Convert neural network output index to UCI move."""
        return self._state.index_to_action(index)

    def get_action_mask(self) -> np.ndarray:
        """Get boolean mask of legal actions."""
        return self._state.get_action_mask()

    def copy(self) -> ChessEnvironment:
        """Create a deep copy."""
        return ChessEnvironment(
            state=self._state.copy(),
            config=self._config,
        )

    def get_symmetries(
        self,
        policy: np.ndarray,
    ) -> list[tuple[ChessEnvironment, np.ndarray]]:
        """
        Get symmetrically equivalent states.

        Chess has limited symmetry compared to Go:
        - Board can be flipped horizontally for color swap training
        - This is primarily used for data augmentation

        For simplicity, we only return the original state here.
        Color swap augmentation is handled during training.
        """
        return [(self, policy)]

    def render(self) -> str:
        """Get human-readable string representation."""
        lines = [str(self._state)]
        lines.append(f"Move: {self._state.move_number}")
        lines.append(f"Phase: {self.game_phase.value}")
        lines.append(f"To play: {self.current_player.name}")
        lines.append(f"FEN: {self.fen}")

        if self.is_terminal():
            lines.append(f"Result: {self.get_result().name}")

        return "\n".join(lines)

    def validate_action(self, action: str) -> bool:
        """Check if an action is valid."""
        return action in self.get_legal_actions()

    # -------------------- Chess-specific methods --------------------

    def is_check(self) -> bool:
        """Check if current player is in check."""
        return self._state.is_check()

    def is_checkmate(self) -> bool:
        """Check if current player is checkmated."""
        return self._state.is_checkmate()

    def is_stalemate(self) -> bool:
        """Check if position is stalemate."""
        return self._state.is_stalemate()

    def is_insufficient_material(self) -> bool:
        """Check for insufficient material draw."""
        return self._state.is_insufficient_material()

    def is_fifty_moves(self) -> bool:
        """Check if fifty-move rule applies."""
        return self._state.is_fifty_moves()

    def is_repetition(self, count: int = 3) -> bool:
        """Check for position repetition."""
        return self._state.is_repetition(count)

    def get_material_balance(self) -> int:
        """Get material balance (positive = white advantage)."""
        return self._state.get_material_balance()

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"ChessEnvironment(fen='{self.fen}', "
            f"move={self._state.move_number}, "
            f"phase={self.game_phase.value})"
        )


# -------------------- Factory Functions --------------------


def create_chess_environment(
    fen: str | None = None,
    config: ChessConfig | None = None,
) -> ChessEnvironment:
    """
    Create a chess environment.

    Args:
        fen: Optional FEN string (uses starting position if None)
        config: Optional chess configuration

    Returns:
        ChessEnvironment at specified position
    """
    game_config = ChessGameConfig(chess_config=config or ChessConfig())

    if fen:
        return ChessEnvironment.from_fen(fen, game_config)
    return ChessEnvironment.initial_state(game_config)


def create_chess_initial_state_fn(
    config: ChessConfig | None = None,
):
    """
    Create factory function for initial chess states.

    Args:
        config: Optional chess configuration

    Returns:
        Function that creates initial ChessEnvironment
    """
    game_config = ChessGameConfig(chess_config=config or ChessConfig())

    def factory() -> ChessEnvironment:
        return ChessEnvironment.initial_state(game_config)

    return factory


# -------------------- Register with GameRegistry --------------------

GameRegistry.register("chess", ChessEnvironment, ChessGameConfig)
