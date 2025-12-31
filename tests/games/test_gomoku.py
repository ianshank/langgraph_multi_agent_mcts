"""
Unit tests for Gomoku game environment.

Tests Gomoku rules, win detection, and GameEnvironment interface compliance.
"""

import numpy as np
import pytest
import torch

from src.games.gomoku import GomokuConfig, GomokuGameState, create_gomoku_state
from src.games.base import GameResult, PlayerColor


class TestGomokuConfig:
    """Tests for GomokuConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GomokuConfig()
        assert config.board_size == 13
        assert config.win_length == 5
        assert config.history_length == 4
        assert config.variant == "freestyle"
        assert config.action_size == 169  # 13*13
        assert config.allow_pass is False

    def test_small_config(self):
        """Test small board configuration."""
        config = GomokuConfig(board_size=9, win_length=5)
        assert config.action_size == 81

    def test_position_conversion(self):
        """Test position to index conversion."""
        config = GomokuConfig(board_size=9)
        assert config.position_to_index(0, 0) == 0
        assert config.position_to_index(0, 8) == 8
        assert config.position_to_index(8, 8) == 80

        row, col = config.index_to_position(0)
        assert row == 0 and col == 0

    def test_invalid_board_size(self):
        """Test that board smaller than win length raises error."""
        with pytest.raises(ValueError, match="board_size.*must be >= win_length"):
            GomokuConfig(board_size=3, win_length=5)


class TestGomokuGameState:
    """Tests for GomokuGameState."""

    def test_initial_state(self):
        """Test initial state creation."""
        state = create_gomoku_state(board_size=9, win_length=5)
        assert state.current_player == PlayerColor.BLACK
        assert not state.is_terminal()
        assert state.metadata.move_number == 0

    def test_get_legal_actions(self):
        """Test getting legal actions."""
        state = create_gomoku_state(board_size=5, win_length=3)
        actions = state.get_legal_actions()
        assert len(actions) == 25  # All positions on 5x5 board

    def test_apply_action(self):
        """Test applying action."""
        state = create_gomoku_state(board_size=5, win_length=3)
        action = 12  # Center of 5x5 board

        new_state = state.apply_action(action)
        assert new_state is not state
        assert new_state.current_player == PlayerColor.WHITE
        assert action not in new_state.get_legal_actions()
        assert new_state.metadata.move_number == 1

    def test_invalid_action_raises_error(self):
        """Test that invalid action raises error."""
        state = create_gomoku_state(board_size=5, win_length=3)
        state = state.apply_action(12)

        with pytest.raises(ValueError):
            state.apply_action(12)  # Already occupied

    def test_horizontal_win(self):
        """Test horizontal win detection."""
        state = create_gomoku_state(board_size=9, win_length=5)

        # Place 5 black stones in a row horizontally
        moves = [(2, 0), (8, 0), (2, 1), (8, 1), (2, 2), (8, 2), (2, 3), (8, 3), (2, 4)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_vertical_win(self):
        """Test vertical win detection."""
        state = create_gomoku_state(board_size=9, win_length=5)

        # Place 5 black stones in a column
        moves = [(0, 4), (0, 0), (1, 4), (0, 1), (2, 4), (0, 2), (3, 4), (0, 3), (4, 4)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_diagonal_win(self):
        """Test diagonal win detection."""
        state = create_gomoku_state(board_size=9, win_length=5)

        # Place 5 black stones diagonally
        moves = [(0, 0), (8, 0), (1, 1), (8, 1), (2, 2), (8, 2), (3, 3), (8, 3), (4, 4)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_anti_diagonal_win(self):
        """Test anti-diagonal win detection."""
        state = create_gomoku_state(board_size=9, win_length=5)

        # Place 5 black stones anti-diagonally
        moves = [(0, 4), (8, 0), (1, 3), (8, 1), (2, 2), (8, 2), (3, 1), (8, 3), (4, 0)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_draw(self):
        """Test draw when board is full."""
        # Use 3x3 board with win_length 3 for simpler test
        state = create_gomoku_state(board_size=3, win_length=4)  # Impossible to win

        # Fill the board
        for i in range(9):
            if not state.is_terminal():
                actions = state.get_legal_actions()
                if actions:
                    state = state.apply_action(actions[0])

        assert state.is_terminal()
        assert state.get_result() == GameResult.DRAW

    def test_get_reward(self):
        """Test reward calculation."""
        state = create_gomoku_state(board_size=5, win_length=3)
        assert state.get_reward(PlayerColor.BLACK) == 0.0  # In progress

        # Create winning position for black
        moves = [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.get_reward(PlayerColor.BLACK) == 1.0
        assert state.get_reward(PlayerColor.WHITE) == -1.0

    def test_to_tensor(self):
        """Test tensor conversion."""
        state = create_gomoku_state(board_size=9, win_length=5)
        tensor = state.to_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dim() == 3
        # 4 history * 2 colors + 1 player plane = 9 planes
        assert tensor.shape == (9, 9, 9)

    def test_get_state_hash(self):
        """Test state hashing."""
        state1 = create_gomoku_state(board_size=5, win_length=3)
        state2 = create_gomoku_state(board_size=5, win_length=3)

        # Same state should have same hash
        assert state1.get_state_hash() == state2.get_state_hash()

        # Different state should have different hash
        state2 = state2.apply_action(12)
        assert state1.get_state_hash() != state2.get_state_hash()

    def test_action_conversion(self):
        """Test action index conversion."""
        state = create_gomoku_state(board_size=5, win_length=3)

        for action in state.get_legal_actions():
            index = state.action_to_index(action)
            recovered = state.index_to_action(index)
            assert action == recovered

    def test_action_mask(self):
        """Test action mask."""
        state = create_gomoku_state(board_size=5, win_length=3)
        mask = state.get_action_mask()

        assert mask.dtype == bool
        assert mask.shape == (25,)
        assert mask.sum() == 25  # All positions legal

        # After one move
        state = state.apply_action(12)
        mask = state.get_action_mask()
        assert mask.sum() == 24
        assert mask[12] is False or mask[12] == False  # Handle numpy bool

    def test_copy(self):
        """Test state copy."""
        state = create_gomoku_state(board_size=5, win_length=3)
        state = state.apply_action(12)

        copy = state.copy()
        assert copy is not state
        assert copy.get_state_hash() == state.get_state_hash()

    def test_symmetries(self):
        """Test symmetry generation."""
        state = create_gomoku_state(board_size=5, win_length=3)
        policy = np.ones(25) / 25

        symmetries = state.get_symmetries(policy)
        assert len(symmetries) == 8  # 4 rotations * 2 flips

        # All policies should sum to 1
        for sym_state, sym_policy in symmetries:
            assert abs(sym_policy.sum() - 1.0) < 0.001

    def test_render(self):
        """Test rendering."""
        state = create_gomoku_state(board_size=5, win_length=3)
        state = state.apply_action(12)

        rendered = state.render()
        assert isinstance(rendered, str)
        assert "Move:" in rendered
        assert "To play:" in rendered
        assert "X" in rendered  # Black stone symbol

    def test_validate_action(self):
        """Test action validation."""
        state = create_gomoku_state(board_size=5, win_length=3)

        assert state.validate_action(0) is True
        assert state.validate_action(24) is True
        assert state.validate_action(1000) is False


class TestGomokuEdgeCases:
    """Tests for edge cases in Gomoku."""

    def test_win_at_edge(self):
        """Test win detection at board edge."""
        state = create_gomoku_state(board_size=5, win_length=3)

        # Win along top edge
        moves = [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_win_at_corner(self):
        """Test win detection from corner."""
        state = create_gomoku_state(board_size=5, win_length=3)

        # Diagonal win from corner
        moves = [(0, 0), (4, 4), (1, 1), (4, 3), (2, 2)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.BLACK_WIN

    def test_four_in_a_row_not_win(self):
        """Test that 4 in a row doesn't win (when win_length=5)."""
        state = create_gomoku_state(board_size=9, win_length=5)

        # Place 4 black stones in a row
        moves = [(2, 0), (8, 0), (2, 1), (8, 1), (2, 2), (8, 2), (2, 3)]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert not state.is_terminal()

    def test_white_win(self):
        """Test white can win."""
        state = create_gomoku_state(board_size=5, win_length=3)

        # White wins
        moves = [
            (0, 0),  # Black
            (1, 0),  # White
            (0, 4),  # Black
            (1, 1),  # White
            (4, 4),  # Black
            (1, 2),  # White wins
        ]
        for row, col in moves:
            action = state.config.position_to_index(row, col)
            if action in state.get_legal_actions():
                state = state.apply_action(action)

        assert state.is_terminal()
        assert state.get_result() == GameResult.WHITE_WIN
