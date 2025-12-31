"""
Elo Rating System for Agent Evaluation.

Implements the Elo rating system used in chess and adapted by DeepMind
for evaluating AlphaZero-style agents. Provides rigorous strength measurement
through head-to-head competition.

Key features:
- Configurable K-factor and initial ratings
- Multi-player tournament support
- Rating uncertainty estimation
- Match history tracking
- Bayesian Elo extensions

Based on:
- Arpad Elo's rating system (1978)
- Bayesian Elo extensions (Coulom, 2008)
- DeepMind's evaluation methodology
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class EloConfig:
    """
    Configuration for Elo rating system.

    Attributes:
        initial_rating: Starting rating for new players
        k_factor_default: Default K-factor for rating adjustments
        k_factor_new_player: K-factor for players with few games
        k_factor_high_rated: K-factor for high-rated players
        new_player_threshold: Games before player is no longer "new"
        high_rating_threshold: Rating above which lower K-factor applies
        min_rating: Minimum allowed rating
        max_rating: Maximum allowed rating
        draw_value: Value assigned to draws (0.5 is standard)
    """

    initial_rating: float = 1500.0
    k_factor_default: float = 32.0
    k_factor_new_player: float = 40.0
    k_factor_high_rated: float = 16.0
    new_player_threshold: int = 30
    high_rating_threshold: float = 2400.0
    min_rating: float = 100.0
    max_rating: float = 4000.0
    draw_value: float = 0.5

    def __post_init__(self):
        """Validate configuration."""
        if self.k_factor_default <= 0:
            raise ValueError(f"k_factor_default must be positive, got {self.k_factor_default}")
        if self.initial_rating <= 0:
            raise ValueError(f"initial_rating must be positive, got {self.initial_rating}")
        if self.min_rating >= self.max_rating:
            raise ValueError(f"min_rating ({self.min_rating}) must be less than max_rating ({self.max_rating})")


@dataclass
class MatchResult:
    """
    Result of a single match between two players.

    Attributes:
        player_a: Identifier for first player
        player_b: Identifier for second player
        score_a: Score for player A (1.0 = win, 0.5 = draw, 0.0 = loss)
        timestamp: When the match occurred
        metadata: Additional match information
    """

    player_a: str
    player_b: str
    score_a: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score_b(self) -> float:
        """Score for player B."""
        return 1.0 - self.score_a

    @property
    def winner(self) -> str | None:
        """Get winner or None for draw."""
        if self.score_a > 0.5:
            return self.player_a
        elif self.score_a < 0.5:
            return self.player_b
        return None

    @classmethod
    def from_winner(
        cls,
        player_a: str,
        player_b: str,
        winner: str | None,
        **kwargs: Any,
    ) -> MatchResult:
        """
        Create match result from winner.

        Args:
            player_a: First player
            player_b: Second player
            winner: Winner's ID, or None for draw

        Returns:
            MatchResult instance
        """
        if winner == player_a:
            score_a = 1.0
        elif winner == player_b:
            score_a = 0.0
        elif winner is None:
            score_a = 0.5
        else:
            raise ValueError(f"Winner '{winner}' must be one of: {player_a}, {player_b}, or None")

        return cls(player_a=player_a, player_b=player_b, score_a=score_a, **kwargs)


@dataclass
class PlayerStats:
    """
    Statistics for a single player.

    Tracks rating history, game count, and performance metrics.
    """

    player_id: str
    rating: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_history: list[tuple[datetime, float]] = field(default_factory=list)
    peak_rating: float = field(default=0.0)
    lowest_rating: float = field(default=float("inf"))

    def __post_init__(self):
        """Initialize peak/lowest if not set."""
        if self.peak_rating == 0.0:
            self.peak_rating = self.rating
        if self.lowest_rating == float("inf"):
            self.lowest_rating = self.rating

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def performance(self) -> float:
        """Calculate performance score (wins + 0.5 * draws) / games."""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played

    def record_rating(self, timestamp: datetime | None = None) -> None:
        """Record current rating in history."""
        if timestamp is None:
            timestamp = datetime.now()
        self.rating_history.append((timestamp, self.rating))
        self.peak_rating = max(self.peak_rating, self.rating)
        self.lowest_rating = min(self.lowest_rating, self.rating)


class EloRatingSystem:
    """
    Elo rating system for evaluating agents.

    Provides methods for:
    - Rating updates after matches
    - Expected score calculation
    - Tournament management
    - Rating persistence and analysis
    """

    def __init__(self, config: EloConfig | None = None):
        """
        Initialize Elo rating system.

        Args:
            config: Rating system configuration
        """
        self.config = config or EloConfig()
        self.players: dict[str, PlayerStats] = {}
        self.match_history: list[MatchResult] = []

    def get_or_create_player(self, player_id: str) -> PlayerStats:
        """
        Get player stats, creating if necessary.

        Args:
            player_id: Player identifier

        Returns:
            PlayerStats for the player
        """
        if player_id not in self.players:
            self.players[player_id] = PlayerStats(
                player_id=player_id,
                rating=self.config.initial_rating,
            )
        return self.players[player_id]

    def get_rating(self, player_id: str) -> float:
        """
        Get current rating for a player.

        Args:
            player_id: Player identifier

        Returns:
            Current Elo rating
        """
        return self.get_or_create_player(player_id).rating

    def set_rating(self, player_id: str, rating: float) -> None:
        """
        Set rating for a player.

        Args:
            player_id: Player identifier
            rating: New rating value
        """
        player = self.get_or_create_player(player_id)
        player.rating = max(self.config.min_rating, min(self.config.max_rating, rating))
        player.record_rating()

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Uses the standard Elo expected score formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score for player A (probability of winning)
        """
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def get_k_factor(self, player: PlayerStats) -> float:
        """
        Get K-factor for a player.

        K-factor determines how much ratings change after a game.
        New players and lower-rated players have higher K-factors.

        Args:
            player: Player stats

        Returns:
            K-factor for this player
        """
        if player.games_played < self.config.new_player_threshold:
            return self.config.k_factor_new_player
        elif player.rating >= self.config.high_rating_threshold:
            return self.config.k_factor_high_rated
        return self.config.k_factor_default

    def update_ratings(
        self,
        player_a: str,
        player_b: str,
        score_a: float,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """
        Update ratings after a match.

        Args:
            player_a: First player ID
            player_b: Second player ID
            score_a: Score for player A (1.0=win, 0.5=draw, 0.0=loss)
            timestamp: Match timestamp
            metadata: Additional match info

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        if timestamp is None:
            timestamp = datetime.now()
        if metadata is None:
            metadata = {}

        # Get player stats
        stats_a = self.get_or_create_player(player_a)
        stats_b = self.get_or_create_player(player_b)

        # Calculate expected scores
        expected_a = self.expected_score(stats_a.rating, stats_b.rating)
        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        # Get K-factors
        k_a = self.get_k_factor(stats_a)
        k_b = self.get_k_factor(stats_b)

        # Calculate rating changes
        delta_a = k_a * (score_a - expected_a)
        delta_b = k_b * (score_b - expected_b)

        # Update ratings
        new_rating_a = stats_a.rating + delta_a
        new_rating_b = stats_b.rating + delta_b

        # Clamp to valid range
        new_rating_a = max(self.config.min_rating, min(self.config.max_rating, new_rating_a))
        new_rating_b = max(self.config.min_rating, min(self.config.max_rating, new_rating_b))

        # Update player stats
        stats_a.rating = new_rating_a
        stats_b.rating = new_rating_b
        stats_a.games_played += 1
        stats_b.games_played += 1

        # Update win/loss/draw counts
        if score_a > 0.5:
            stats_a.wins += 1
            stats_b.losses += 1
        elif score_a < 0.5:
            stats_a.losses += 1
            stats_b.wins += 1
        else:
            stats_a.draws += 1
            stats_b.draws += 1

        # Record rating history
        stats_a.record_rating(timestamp)
        stats_b.record_rating(timestamp)

        # Record match
        self.match_history.append(
            MatchResult(
                player_a=player_a,
                player_b=player_b,
                score_a=score_a,
                timestamp=timestamp,
                metadata=metadata,
            )
        )

        return new_rating_a, new_rating_b

    def record_match(self, result: MatchResult) -> tuple[float, float]:
        """
        Record a match result and update ratings.

        Args:
            result: Match result

        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        return self.update_ratings(
            player_a=result.player_a,
            player_b=result.player_b,
            score_a=result.score_a,
            timestamp=result.timestamp,
            metadata=result.metadata,
        )

    def run_tournament(
        self,
        players: Sequence[str],
        match_fn: Any,  # Callable[[str, str], float | tuple[float, dict]]
        rounds: int = 1,
        include_self_play: bool = False,
    ) -> dict[str, float]:
        """
        Run a round-robin tournament.

        Args:
            players: List of player IDs
            match_fn: Function that plays a match and returns score_a
                      Can also return (score_a, metadata) tuple
            rounds: Number of complete rounds to play
            include_self_play: Whether to include self-play matches

        Returns:
            Dictionary mapping player_id to final rating
        """
        for _ in range(rounds):
            for i, player_a in enumerate(players):
                start_j = i if include_self_play else i + 1
                for player_b in players[start_j:]:
                    if player_a == player_b and not include_self_play:
                        continue

                    # Play match
                    result = match_fn(player_a, player_b)
                    if isinstance(result, tuple):
                        score_a, metadata = result
                    else:
                        score_a, metadata = result, {}

                    # Update ratings
                    self.update_ratings(
                        player_a=player_a,
                        player_b=player_b,
                        score_a=score_a,
                        metadata=metadata,
                    )

        return {p: self.get_rating(p) for p in players}

    def estimate_performance_rating(
        self,
        player_id: str,
        opponent_ratings: Sequence[float],
        scores: Sequence[float],
    ) -> float:
        """
        Estimate performance rating from a series of games.

        Uses the FIDE performance rating formula.

        Args:
            player_id: Player identifier
            opponent_ratings: Ratings of opponents faced
            scores: Scores achieved (1.0, 0.5, or 0.0)

        Returns:
            Estimated performance rating
        """
        if len(opponent_ratings) != len(scores):
            raise ValueError("opponent_ratings and scores must have same length")

        if not opponent_ratings:
            return self.config.initial_rating

        # Calculate average opponent rating
        avg_opponent_rating = np.mean(opponent_ratings)

        # Calculate score percentage
        total_score = sum(scores)
        score_percentage = total_score / len(scores)

        # FIDE performance rating formula
        if score_percentage >= 1.0:
            # Perfect score
            return avg_opponent_rating + 800
        elif score_percentage <= 0.0:
            # Zero score
            return avg_opponent_rating - 800
        else:
            # Standard formula: P = R_avg + 400 * log10(W / L)
            # where W = score, L = 1 - score
            dp = 400 * math.log10(score_percentage / (1 - score_percentage))
            return avg_opponent_rating + dp

    def get_rating_uncertainty(self, player_id: str) -> float:
        """
        Estimate rating uncertainty (standard deviation).

        Uses a simple model based on number of games played.

        Args:
            player_id: Player identifier

        Returns:
            Estimated rating uncertainty
        """
        player = self.get_or_create_player(player_id)

        # Base uncertainty decreases with more games
        base_uncertainty = 350.0
        decay_rate = 0.1
        games = player.games_played

        # Uncertainty formula: σ = base * exp(-decay * games)
        # With minimum uncertainty of ~50
        uncertainty = base_uncertainty * math.exp(-decay_rate * games)
        return max(50.0, uncertainty)

    def get_win_probability(self, player_a: str, player_b: str) -> float:
        """
        Get probability that player A beats player B.

        Args:
            player_a: First player ID
            player_b: Second player ID

        Returns:
            Win probability for player A
        """
        rating_a = self.get_rating(player_a)
        rating_b = self.get_rating(player_b)
        return self.expected_score(rating_a, rating_b)

    def get_leaderboard(self, min_games: int = 0) -> list[PlayerStats]:
        """
        Get sorted leaderboard of players.

        Args:
            min_games: Minimum games required to be included

        Returns:
            List of PlayerStats sorted by rating (descending)
        """
        qualified = [p for p in self.players.values() if p.games_played >= min_games]
        return sorted(qualified, key=lambda x: x.rating, reverse=True)

    def get_head_to_head(self, player_a: str, player_b: str) -> dict[str, Any]:
        """
        Get head-to-head record between two players.

        Args:
            player_a: First player ID
            player_b: Second player ID

        Returns:
            Dictionary with head-to-head statistics
        """
        matches = [
            m
            for m in self.match_history
            if (m.player_a == player_a and m.player_b == player_b)
            or (m.player_a == player_b and m.player_b == player_a)
        ]

        wins_a = sum(
            1
            for m in matches
            if (m.player_a == player_a and m.score_a > 0.5) or (m.player_b == player_a and m.score_a < 0.5)
        )

        draws = sum(1 for m in matches if m.score_a == 0.5)
        wins_b = len(matches) - wins_a - draws

        return {
            "total_matches": len(matches),
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "score_a": wins_a + 0.5 * draws,
            "score_b": wins_b + 0.5 * draws,
        }

    def save(self, path: str | Path) -> None:
        """
        Save rating system state to file.

        Args:
            path: Path to save to (JSON format)
        """
        path = Path(path)

        data = {
            "config": {
                "initial_rating": self.config.initial_rating,
                "k_factor_default": self.config.k_factor_default,
                "k_factor_new_player": self.config.k_factor_new_player,
                "k_factor_high_rated": self.config.k_factor_high_rated,
                "new_player_threshold": self.config.new_player_threshold,
                "high_rating_threshold": self.config.high_rating_threshold,
                "min_rating": self.config.min_rating,
                "max_rating": self.config.max_rating,
                "draw_value": self.config.draw_value,
            },
            "players": {
                player_id: {
                    "rating": stats.rating,
                    "games_played": stats.games_played,
                    "wins": stats.wins,
                    "losses": stats.losses,
                    "draws": stats.draws,
                    "peak_rating": stats.peak_rating,
                    "lowest_rating": stats.lowest_rating,
                    "rating_history": [(ts.isoformat(), rating) for ts, rating in stats.rating_history],
                }
                for player_id, stats in self.players.items()
            },
            "match_history": [
                {
                    "player_a": m.player_a,
                    "player_b": m.player_b,
                    "score_a": m.score_a,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self.match_history
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EloRatingSystem:
        """
        Load rating system state from file.

        Args:
            path: Path to load from

        Returns:
            EloRatingSystem instance
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        # Create config
        config = EloConfig(**data.get("config", {}))
        system = cls(config)

        # Load players
        for player_id, stats_data in data.get("players", {}).items():
            stats = PlayerStats(
                player_id=player_id,
                rating=stats_data["rating"],
                games_played=stats_data.get("games_played", 0),
                wins=stats_data.get("wins", 0),
                losses=stats_data.get("losses", 0),
                draws=stats_data.get("draws", 0),
                peak_rating=stats_data.get("peak_rating", stats_data["rating"]),
                lowest_rating=stats_data.get("lowest_rating", stats_data["rating"]),
                rating_history=[
                    (datetime.fromisoformat(ts), rating) for ts, rating in stats_data.get("rating_history", [])
                ],
            )
            system.players[player_id] = stats

        # Load match history
        for match_data in data.get("match_history", []):
            match = MatchResult(
                player_a=match_data["player_a"],
                player_b=match_data["player_b"],
                score_a=match_data["score_a"],
                timestamp=datetime.fromisoformat(match_data["timestamp"]),
                metadata=match_data.get("metadata", {}),
            )
            system.match_history.append(match)

        return system


# -------------------- Bayesian Elo Extensions --------------------


@dataclass
class BayesianEloConfig(EloConfig):
    """
    Configuration for Bayesian Elo rating system.

    Extends standard Elo with uncertainty estimation.
    """

    # Prior strength (affects how quickly ratings converge)
    prior_games: float = 3.0

    # Advantage for playing first (e.g., white in chess)
    first_player_advantage: float = 32.5

    # Draw probability model parameter
    draw_elo: float = 97.3


class BayesianEloSystem(EloRatingSystem):
    """
    Bayesian Elo rating system with uncertainty estimation.

    Extends the standard Elo system with:
    - Rating uncertainty (confidence intervals)
    - First player advantage modeling
    - More accurate expected score calculation

    Based on Rémi Coulom's Bayesian Elo (2008).
    """

    def __init__(self, config: BayesianEloConfig | None = None):
        """
        Initialize Bayesian Elo system.

        Args:
            config: Bayesian Elo configuration
        """
        super().__init__(config or BayesianEloConfig())

    @property
    def bayesian_config(self) -> BayesianEloConfig:
        """Get config as BayesianEloConfig."""
        return self.config  # type: ignore

    def expected_score_with_draw(
        self,
        rating_a: float,
        rating_b: float,
        is_a_first: bool = True,
    ) -> tuple[float, float, float]:
        """
        Calculate expected win/draw/loss probabilities.

        Uses the Bradley-Terry model with draw handling.

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B
            is_a_first: Whether player A plays first

        Returns:
            Tuple of (win_prob, draw_prob, loss_prob) for player A
        """
        config = self.bayesian_config

        # Apply first player advantage
        if is_a_first:
            effective_a = rating_a + config.first_player_advantage
            effective_b = rating_b
        else:
            effective_a = rating_a
            effective_b = rating_b + config.first_player_advantage

        # Calculate probabilities using logistic model
        diff = (effective_a - effective_b) / 400.0
        gamma_a = math.pow(10.0, diff)
        gamma_b = 1.0
        gamma_d = math.pow(10.0, config.draw_elo / 400.0)

        # Normalize
        total = gamma_a + gamma_b + gamma_d
        win_prob = gamma_a / total
        draw_prob = gamma_d / total
        loss_prob = gamma_b / total

        return win_prob, draw_prob, loss_prob

    def get_confidence_interval(
        self,
        player_id: str,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Calculate confidence interval for player's rating.

        Args:
            player_id: Player identifier
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Get rating and uncertainty
        rating = self.get_rating(player_id)
        uncertainty = self.get_rating_uncertainty(player_id)

        # Use normal distribution approximation
        # For 95% confidence, z ≈ 1.96
        from scipy import stats

        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * uncertainty

        return rating - margin, rating + margin


# -------------------- Utility Functions --------------------


def rating_diff_to_win_probability(rating_diff: float) -> float:
    """
    Convert rating difference to win probability.

    Args:
        rating_diff: Rating difference (higher player - lower player)

    Returns:
        Win probability for higher-rated player
    """
    return 1.0 / (1.0 + math.pow(10.0, -rating_diff / 400.0))


def win_probability_to_rating_diff(win_prob: float) -> float:
    """
    Convert win probability to rating difference.

    Args:
        win_prob: Win probability (0.0 to 1.0)

    Returns:
        Implied rating difference
    """
    if win_prob <= 0.0:
        return -800.0
    if win_prob >= 1.0:
        return 800.0
    return 400.0 * math.log10(win_prob / (1.0 - win_prob))


def calculate_rating_from_performance(
    opponents: Sequence[tuple[float, float]],
) -> float:
    """
    Calculate rating from tournament performance.

    Uses iterative method to find rating that matches observed performance.

    Args:
        opponents: List of (opponent_rating, score) tuples

    Returns:
        Estimated rating
    """
    if not opponents:
        return 1500.0

    # Initial guess: average opponent rating
    avg_opp = np.mean([r for r, _ in opponents])
    total_score = sum(s for _, s in opponents)

    # Iterative refinement
    rating = avg_opp
    for _ in range(20):  # Usually converges in <10 iterations
        expected = sum(rating_diff_to_win_probability(rating - opp_r) for opp_r, _ in opponents)

        if expected < 0.001:
            rating += 50
        elif expected > len(opponents) - 0.001:
            rating -= 50
        else:
            # Newton-Raphson step
            derivative = sum(
                (math.log(10) / 400)
                * rating_diff_to_win_probability(rating - opp_r)
                * (1 - rating_diff_to_win_probability(rating - opp_r))
                for opp_r, _ in opponents
            )
            if abs(derivative) > 1e-6:
                rating += (total_score - expected) / derivative

    return rating
