"""
Unit tests for the Elo rating system.

Tests rating calculations, match recording, and tournament functionality.
"""

import math
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.training.elo_rating import (
    BayesianEloConfig,
    BayesianEloSystem,
    EloConfig,
    EloRatingSystem,
    MatchResult,
    PlayerStats,
    calculate_rating_from_performance,
    rating_diff_to_win_probability,
    win_probability_to_rating_diff,
)


class TestEloConfig:
    """Tests for EloConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EloConfig()
        assert config.initial_rating == 1500.0
        assert config.k_factor_default == 32.0
        assert config.k_factor_new_player == 40.0
        assert config.k_factor_high_rated == 16.0
        assert config.new_player_threshold == 30
        assert config.high_rating_threshold == 2400.0
        assert config.min_rating == 100.0
        assert config.max_rating == 4000.0
        assert config.draw_value == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = EloConfig(
            initial_rating=1200.0,
            k_factor_default=24.0,
        )
        assert config.initial_rating == 1200.0
        assert config.k_factor_default == 24.0

    def test_invalid_k_factor(self):
        """Test that invalid k_factor raises error."""
        with pytest.raises(ValueError, match="k_factor_default must be positive"):
            EloConfig(k_factor_default=0)

    def test_invalid_rating_range(self):
        """Test that invalid rating range raises error."""
        with pytest.raises(ValueError, match="min_rating.*must be less than max_rating"):
            EloConfig(min_rating=2000, max_rating=1000)


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_basic_match_result(self):
        """Test basic match result creation."""
        result = MatchResult(
            player_a="alice",
            player_b="bob",
            score_a=1.0,
        )
        assert result.player_a == "alice"
        assert result.player_b == "bob"
        assert result.score_a == 1.0
        assert result.score_b == 0.0
        assert result.winner == "alice"

    def test_draw_result(self):
        """Test draw match result."""
        result = MatchResult(
            player_a="alice",
            player_b="bob",
            score_a=0.5,
        )
        assert result.score_b == 0.5
        assert result.winner is None

    def test_from_winner(self):
        """Test creating result from winner."""
        # Alice wins
        result = MatchResult.from_winner("alice", "bob", "alice")
        assert result.score_a == 1.0
        assert result.winner == "alice"

        # Bob wins
        result = MatchResult.from_winner("alice", "bob", "bob")
        assert result.score_a == 0.0
        assert result.winner == "bob"

        # Draw
        result = MatchResult.from_winner("alice", "bob", None)
        assert result.score_a == 0.5
        assert result.winner is None

    def test_invalid_winner(self):
        """Test that invalid winner raises error."""
        with pytest.raises(ValueError, match="Winner.*must be one of"):
            MatchResult.from_winner("alice", "bob", "charlie")


class TestPlayerStats:
    """Tests for PlayerStats dataclass."""

    def test_initial_stats(self):
        """Test initial player stats."""
        stats = PlayerStats(player_id="alice", rating=1500.0)
        assert stats.player_id == "alice"
        assert stats.rating == 1500.0
        assert stats.games_played == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        assert stats.win_rate == 0.0
        assert stats.performance == 0.0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        stats = PlayerStats(
            player_id="alice",
            rating=1500.0,
            games_played=10,
            wins=6,
            losses=2,
            draws=2,
        )
        assert stats.win_rate == 0.6

    def test_performance_calculation(self):
        """Test performance calculation."""
        stats = PlayerStats(
            player_id="alice",
            rating=1500.0,
            games_played=10,
            wins=6,
            losses=2,
            draws=2,
        )
        # Performance = (wins + 0.5 * draws) / games = (6 + 1) / 10 = 0.7
        assert stats.performance == 0.7

    def test_record_rating(self):
        """Test rating history recording."""
        stats = PlayerStats(player_id="alice", rating=1500.0)
        stats.rating = 1550.0
        stats.record_rating()

        assert len(stats.rating_history) == 1
        assert stats.rating_history[0][1] == 1550.0
        assert stats.peak_rating == 1550.0


class TestEloRatingSystem:
    """Tests for EloRatingSystem."""

    def test_create_player(self):
        """Test player creation."""
        system = EloRatingSystem()
        stats = system.get_or_create_player("alice")

        assert stats.player_id == "alice"
        assert stats.rating == 1500.0
        assert "alice" in system.players

    def test_get_rating(self):
        """Test rating retrieval."""
        system = EloRatingSystem()
        rating = system.get_rating("alice")
        assert rating == 1500.0

    def test_set_rating(self):
        """Test setting rating."""
        system = EloRatingSystem()
        system.set_rating("alice", 1600.0)
        assert system.get_rating("alice") == 1600.0

    def test_rating_clamped_to_bounds(self):
        """Test that ratings are clamped to valid range."""
        config = EloConfig(min_rating=100, max_rating=3000)
        system = EloRatingSystem(config)

        system.set_rating("alice", 50.0)  # Below min
        assert system.get_rating("alice") == 100.0

        system.set_rating("bob", 5000.0)  # Above max
        assert system.get_rating("bob") == 3000.0

    def test_expected_score(self):
        """Test expected score calculation."""
        system = EloRatingSystem()

        # Equal ratings: expected score = 0.5
        expected = system.expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001

        # Higher rating: expected score > 0.5
        expected = system.expected_score(1600, 1400)
        assert expected > 0.5

        # 400 point difference: ~10:1 odds
        expected = system.expected_score(1900, 1500)
        assert abs(expected - 0.909) < 0.01

    def test_update_ratings_win(self):
        """Test rating update after a win."""
        system = EloRatingSystem()
        system.set_rating("alice", 1500)
        system.set_rating("bob", 1500)

        new_a, new_b = system.update_ratings("alice", "bob", 1.0)

        assert new_a > 1500  # Winner gains rating
        assert new_b < 1500  # Loser loses rating
        assert abs((new_a - 1500) + (new_b - 1500)) < 1  # Approximately zero-sum

    def test_update_ratings_draw(self):
        """Test rating update after a draw."""
        system = EloRatingSystem()
        system.set_rating("alice", 1600)
        system.set_rating("bob", 1400)

        new_a, new_b = system.update_ratings("alice", "bob", 0.5)

        # Higher rated player should lose rating in a draw
        assert new_a < 1600
        # Lower rated player should gain rating in a draw
        assert new_b > 1400

    def test_k_factor_new_player(self):
        """Test K-factor for new players."""
        config = EloConfig(
            k_factor_new_player=40.0,
            k_factor_default=32.0,
            new_player_threshold=30,
        )
        system = EloRatingSystem(config)

        player = system.get_or_create_player("alice")
        k = system.get_k_factor(player)
        assert k == 40.0

    def test_k_factor_experienced_player(self):
        """Test K-factor for experienced players."""
        config = EloConfig(
            k_factor_new_player=40.0,
            k_factor_default=32.0,
            new_player_threshold=30,
        )
        system = EloRatingSystem(config)

        player = system.get_or_create_player("alice")
        player.games_played = 50
        k = system.get_k_factor(player)
        assert k == 32.0

    def test_k_factor_high_rated(self):
        """Test K-factor for high-rated players."""
        config = EloConfig(
            k_factor_high_rated=16.0,
            high_rating_threshold=2400.0,
            new_player_threshold=30,
        )
        system = EloRatingSystem(config)

        player = system.get_or_create_player("alice")
        player.games_played = 100
        player.rating = 2500.0
        k = system.get_k_factor(player)
        assert k == 16.0

    def test_match_history(self):
        """Test match history recording."""
        system = EloRatingSystem()
        system.update_ratings("alice", "bob", 1.0)
        system.update_ratings("alice", "charlie", 0.5)

        assert len(system.match_history) == 2
        assert system.match_history[0].player_a == "alice"
        assert system.match_history[0].score_a == 1.0

    def test_get_win_probability(self):
        """Test win probability calculation."""
        system = EloRatingSystem()
        system.set_rating("alice", 1600)
        system.set_rating("bob", 1400)

        prob = system.get_win_probability("alice", "bob")
        assert prob > 0.5

    def test_get_leaderboard(self):
        """Test leaderboard generation."""
        system = EloRatingSystem()
        system.set_rating("alice", 1800)
        system.set_rating("bob", 1600)
        system.set_rating("charlie", 1700)

        leaderboard = system.get_leaderboard()
        assert len(leaderboard) == 3
        assert leaderboard[0].player_id == "alice"
        assert leaderboard[1].player_id == "charlie"
        assert leaderboard[2].player_id == "bob"

    def test_get_head_to_head(self):
        """Test head-to-head record."""
        system = EloRatingSystem()
        system.update_ratings("alice", "bob", 1.0)
        system.update_ratings("alice", "bob", 1.0)
        system.update_ratings("alice", "bob", 0.5)
        system.update_ratings("bob", "alice", 1.0)  # Bob wins (alice is player_b)

        h2h = system.get_head_to_head("alice", "bob")
        assert h2h["total_matches"] == 4
        assert h2h["wins_a"] == 2  # Alice won 2
        assert h2h["wins_b"] == 1  # Bob won 1
        assert h2h["draws"] == 1

    def test_rating_uncertainty(self):
        """Test rating uncertainty estimation."""
        system = EloRatingSystem()

        # New player has high uncertainty
        uncertainty_new = system.get_rating_uncertainty("alice")
        assert uncertainty_new > 100

        # Experienced player has lower uncertainty
        player = system.get_or_create_player("bob")
        player.games_played = 100
        uncertainty_exp = system.get_rating_uncertainty("bob")
        assert uncertainty_exp < uncertainty_new

    def test_save_and_load(self):
        """Test saving and loading system state."""
        system = EloRatingSystem()
        system.update_ratings("alice", "bob", 1.0)
        system.update_ratings("alice", "charlie", 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elo.json"
            system.save(path)

            loaded = EloRatingSystem.load(path)

            assert loaded.get_rating("alice") == system.get_rating("alice")
            assert len(loaded.match_history) == 2


class TestBayesianEloSystem:
    """Tests for BayesianEloSystem."""

    def test_expected_score_with_draw(self):
        """Test expected score with draw probability."""
        system = BayesianEloSystem()
        win, draw, loss = system.expected_score_with_draw(1500, 1500)

        assert abs(win - loss) < 0.01  # Should be symmetric
        assert draw > 0  # Should have some draw probability
        assert abs(win + draw + loss - 1.0) < 0.001  # Should sum to 1


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_rating_diff_to_win_probability(self):
        """Test rating difference to probability conversion."""
        # Equal ratings
        prob = rating_diff_to_win_probability(0)
        assert abs(prob - 0.5) < 0.001

        # 400 point advantage
        prob = rating_diff_to_win_probability(400)
        assert abs(prob - 0.909) < 0.01

        # 400 point disadvantage
        prob = rating_diff_to_win_probability(-400)
        assert abs(prob - 0.091) < 0.01

    def test_win_probability_to_rating_diff(self):
        """Test probability to rating difference conversion."""
        # 50% win probability = 0 rating diff
        diff = win_probability_to_rating_diff(0.5)
        assert abs(diff) < 1

        # Higher probability = positive diff
        diff = win_probability_to_rating_diff(0.9)
        assert diff > 0

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion."""
        original_diff = 200
        prob = rating_diff_to_win_probability(original_diff)
        recovered_diff = win_probability_to_rating_diff(prob)
        assert abs(original_diff - recovered_diff) < 1

    def test_calculate_rating_from_performance(self):
        """Test performance rating calculation."""
        # Perfect score against 1500-rated opponents
        opponents = [(1500, 1.0), (1500, 1.0), (1500, 1.0)]
        rating = calculate_rating_from_performance(opponents)
        assert rating > 1500  # Should be higher than opponents

        # Zero score
        opponents = [(1500, 0.0), (1500, 0.0), (1500, 0.0)]
        rating = calculate_rating_from_performance(opponents)
        assert rating < 1500  # Should be lower than opponents

        # 50% score
        opponents = [(1500, 1.0), (1500, 0.0)]
        rating = calculate_rating_from_performance(opponents)
        assert abs(rating - 1500) < 50  # Should be close to opponent average
