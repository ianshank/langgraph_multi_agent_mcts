"""
Unit tests for the abstract GameEnvironment interface.

Tests the base game abstraction layer that all game implementations
must conform to.
"""

import pytest
import numpy as np
import torch

from src.games.base import (
    GameConfig,
    GameEnvironment,
    GameMetadata,
    GameRegistry,
    GameResult,
    GameStateAdapter,
    PlayerColor,
    create_initial_state_fn,
    validate_game_implementation,
)


class TestGameConfig:
    """Tests for GameConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GameConfig()
        assert config.board_size == 19
        assert config.action_size == 362
        assert config.num_input_planes == 17
        assert config.history_length == 8
        assert config.allow_pass is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GameConfig(
            board_size=9,
            action_size=82,
            num_input_planes=10,
            history_length=4,
            allow_pass=False,
        )
        assert config.board_size == 9
        assert config.action_size == 82
        assert config.num_input_planes == 10
        assert config.history_length == 4
        assert config.allow_pass is False

    def test_invalid_board_size(self):
        """Test that invalid board size raises error."""
        with pytest.raises(ValueError, match="board_size must be positive"):
            GameConfig(board_size=0)

    def test_invalid_action_size(self):
        """Test that invalid action size raises error."""
        with pytest.raises(ValueError, match="action_size must be positive"):
            GameConfig(action_size=-1)


class TestPlayerColor:
    """Tests for PlayerColor enum."""

    def test_player_values(self):
        """Test player color values."""
        assert PlayerColor.WHITE.value == 1
        assert PlayerColor.BLACK.value == -1

    def test_opponent(self):
        """Test opponent property."""
        assert PlayerColor.WHITE.opponent == PlayerColor.BLACK
        assert PlayerColor.BLACK.opponent == PlayerColor.WHITE


class TestGameResult:
    """Tests for GameResult enum."""

    def test_result_values(self):
        """Test all result types exist."""
        assert GameResult.WHITE_WIN is not None
        assert GameResult.BLACK_WIN is not None
        assert GameResult.DRAW is not None
        assert GameResult.IN_PROGRESS is not None


class TestGameMetadata:
    """Tests for GameMetadata dataclass."""

    def test_default_metadata(self):
        """Test default metadata values."""
        metadata = GameMetadata()
        assert metadata.move_number == 0
        assert metadata.game_id == ""
        assert metadata.player_to_move == PlayerColor.WHITE
        assert metadata.last_action is None
        assert metadata.is_resignation is False
        assert metadata.extra == {}

    def test_custom_metadata(self):
        """Test custom metadata values."""
        metadata = GameMetadata(
            move_number=10,
            game_id="game123",
            player_to_move=PlayerColor.BLACK,
            last_action="e2e4",
            extra={"key": "value"},
        )
        assert metadata.move_number == 10
        assert metadata.game_id == "game123"
        assert metadata.player_to_move == PlayerColor.BLACK
        assert metadata.last_action == "e2e4"
        assert metadata.extra == {"key": "value"}


class TestGameRegistry:
    """Tests for GameRegistry."""

    def test_list_games(self):
        """Test listing registered games."""
        games = GameRegistry.list_games()
        assert isinstance(games, list)
        # Should have at least the games we registered
        # (go, gomoku are registered in their modules)

    def test_unknown_game_raises_error(self):
        """Test that unknown game name raises error."""
        with pytest.raises(ValueError, match="Unknown game"):
            GameRegistry.create("nonexistent_game")


class TestGameStateAdapter:
    """Tests for GameStateAdapter compatibility layer."""

    def test_adapter_interface(self):
        """Test that adapter has required methods."""
        # Create a mock game environment to test adapter
        from src.games.gomoku import create_gomoku_state

        env = create_gomoku_state(board_size=5, win_length=3)
        adapter = env.to_game_state()

        # Test all required methods exist
        assert hasattr(adapter, "get_legal_actions")
        assert hasattr(adapter, "apply_action")
        assert hasattr(adapter, "is_terminal")
        assert hasattr(adapter, "get_reward")
        assert hasattr(adapter, "to_tensor")
        assert hasattr(adapter, "get_canonical_form")
        assert hasattr(adapter, "get_hash")
        assert hasattr(adapter, "action_to_index")

    def test_adapter_functionality(self):
        """Test adapter methods work correctly."""
        from src.games.gomoku import create_gomoku_state

        env = create_gomoku_state(board_size=5, win_length=3)
        adapter = env.to_game_state()

        # Test get_legal_actions
        actions = adapter.get_legal_actions()
        assert len(actions) == 25  # 5x5 board, all empty

        # Test apply_action
        new_adapter = adapter.apply_action(actions[0])
        assert new_adapter is not adapter  # Should return new state

        # Test is_terminal
        assert adapter.is_terminal() is False

        # Test to_tensor
        tensor = adapter.to_tensor()
        assert isinstance(tensor, torch.Tensor)

        # Test get_hash
        hash_val = adapter.get_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0


class TestCreateInitialStateFn:
    """Tests for factory function creation."""

    def test_create_factory(self):
        """Test creating factory function for game states."""
        # This requires games to be registered
        from src.games.gomoku import GomokuGameState

        # Register if not already
        GameRegistry.register("test_gomoku", GomokuGameState)

        factory = create_initial_state_fn("test_gomoku")
        assert callable(factory)

        # Factory should create valid state
        state = factory()
        assert state is not None
        assert not state.is_terminal()


class TestValidateGameImplementation:
    """Tests for game implementation validation."""

    def test_validate_gomoku(self):
        """Test validation of Gomoku implementation."""
        from src.games.gomoku import GomokuGameState

        errors = validate_game_implementation(GomokuGameState)
        assert errors == [], f"Validation errors: {errors}"

    def test_validate_go(self):
        """Test validation of Go implementation."""
        from src.games.go import GoGameState

        errors = validate_game_implementation(GoGameState)
        assert errors == [], f"Validation errors: {errors}"
