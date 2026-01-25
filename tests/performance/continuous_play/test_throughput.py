"""
Performance tests for continuous play throughput.

Tests game throughput, learning latency, and memory stability.
"""

import time
from dataclasses import dataclass

import psutil
import pytest

from src.games.chess.continuous_learning import create_learning_session
from src.training.continuous_play_config import ContinuousPlayConfig
from src.training.metrics_aggregator import MetricsAggregator


@dataclass
class PerformanceResult:
    """Container for performance test results."""

    total_games: int
    total_time_seconds: float
    games_per_minute: float
    avg_game_time_seconds: float
    memory_start_mb: float
    memory_end_mb: float
    memory_growth_mb: float


@pytest.mark.performance
@pytest.mark.asyncio
@pytest.mark.timeout(180)  # 3 minute timeout
class TestGameThroughput:
    """Performance tests for game throughput."""

    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    async def test_minimum_game_throughput(self):
        """Test that we achieve minimum game throughput."""
        session = create_learning_session(
            preset="small",
            max_minutes=2,
            max_games=5,
        )

        memory_start = self._get_memory_mb()
        start_time = time.time()

        await session.run_session(max_games=5)

        elapsed = time.time() - start_time
        memory_end = self._get_memory_mb()

        games_per_minute = session.scorecard.total_games / (elapsed / 60)

        result = PerformanceResult(
            total_games=session.scorecard.total_games,
            total_time_seconds=elapsed,
            games_per_minute=games_per_minute,
            avg_game_time_seconds=elapsed / max(1, session.scorecard.total_games),
            memory_start_mb=memory_start,
            memory_end_mb=memory_end,
            memory_growth_mb=memory_end - memory_start,
        )

        # Minimum throughput: 1 game per minute
        # (actual should be higher, but this is a safe minimum)
        assert games_per_minute >= 0.5, f"Throughput too low: {games_per_minute:.2f} games/min"

        # Memory growth should be bounded
        assert result.memory_growth_mb < 500, f"Memory grew too much: {result.memory_growth_mb:.1f}MB"

    async def test_sustained_game_throughput(self):
        """Test throughput over multiple games."""
        session = create_learning_session(
            preset="small",
            max_minutes=3,
            max_games=10,
        )

        game_times = []

        def progress_callback(game_num, max_games, scorecard):
            game_times.append(time.time())

        await session.run_session(
            max_games=10,
            progress_callback=progress_callback,
        )

        # Calculate time between games
        if len(game_times) >= 2:
            intervals = [game_times[i + 1] - game_times[i] for i in range(len(game_times) - 1)]
            avg_interval = sum(intervals) / len(intervals)

            # No game should take more than 60 seconds
            assert max(intervals) < 60, f"Game took too long: {max(intervals):.1f}s"


@pytest.mark.performance
@pytest.mark.asyncio
@pytest.mark.timeout(60)
class TestLearningLatency:
    """Performance tests for learning update latency."""

    async def test_learning_update_speed(self):
        """Test that learning updates complete quickly."""
        from src.games.chess.config import ChessConfig
        from src.games.chess.continuous_learning import (
            ContinuousLearningConfig,
            ContinuousLearningSession,
        )

        chess_config = ChessConfig.from_preset("small")
        learning_config = ContinuousLearningConfig(
            max_session_minutes=2,
            max_games=8,
            min_games_before_learning=3,
            learn_every_n_games=2,
            learning_batch_size=32,  # Small batch for speed
        )

        session = ContinuousLearningSession(chess_config, learning_config)

        learning_times = []

        def learning_callback(loss, game_num):
            learning_times.append(time.time())

        session.learning_config.on_learning_update = learning_callback

        start_time = time.time()
        await session.run_session(max_games=8)
        total_time = time.time() - start_time

        # Calculate learning overhead
        if len(learning_times) >= 2:
            # Time between learning updates should be reasonable
            for i in range(1, len(learning_times)):
                interval = learning_times[i] - learning_times[i - 1]
                # Learning should trigger every few games, not take forever
                assert interval < 30, f"Learning interval too long: {interval:.1f}s"


@pytest.mark.performance
class TestMetricsOverhead:
    """Performance tests for metrics collection overhead."""

    def test_metrics_recording_speed(self):
        """Test that metrics recording is fast."""
        aggregator = MetricsAggregator(
            enable_prometheus=False,
            enable_wandb=False,
        )

        num_samples = 10000
        start_time = time.time()

        for i in range(num_samples):
            aggregator.record_sample("test_metric", float(i))

        elapsed = time.time() - start_time
        samples_per_second = num_samples / elapsed

        # Should be able to record at least 10k samples per second
        assert samples_per_second > 10000, f"Too slow: {samples_per_second:.0f} samples/sec"

    def test_counter_increment_speed(self):
        """Test that counter incrementing is fast."""
        aggregator = MetricsAggregator(
            enable_prometheus=False,
            enable_wandb=False,
        )

        num_increments = 10000
        start_time = time.time()

        for i in range(num_increments):
            aggregator.increment_counter("test_counter")

        elapsed = time.time() - start_time
        increments_per_second = num_increments / elapsed

        # Should be able to increment at least 50k times per second
        assert increments_per_second > 50000, f"Too slow: {increments_per_second:.0f} inc/sec"

    def test_stats_calculation_speed(self):
        """Test that statistics calculation is fast."""
        aggregator = MetricsAggregator(
            buffer_size=1000,
            enable_prometheus=False,
            enable_wandb=False,
        )

        # Fill buffer
        for i in range(1000):
            aggregator.record_sample("test_metric", float(i))

        # Time stats calculation
        num_calcs = 1000
        start_time = time.time()

        for _ in range(num_calcs):
            aggregator.get_stats("test_metric")

        elapsed = time.time() - start_time
        calcs_per_second = num_calcs / elapsed

        # Should calculate stats at least 1000 times per second
        assert calcs_per_second > 1000, f"Too slow: {calcs_per_second:.0f} calcs/sec"


@pytest.mark.performance
class TestMemoryStability:
    """Performance tests for memory stability."""

    def test_metrics_buffer_bounded(self):
        """Test that metrics buffer stays bounded."""
        aggregator = MetricsAggregator(
            buffer_size=100,
            enable_prometheus=False,
            enable_wandb=False,
        )

        initial_memory = psutil.Process().memory_info().rss

        # Add many more samples than buffer size
        for i in range(10000):
            aggregator.record_sample("test", float(i))

        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)

        # Memory growth should be minimal due to bounded buffer
        assert memory_growth < 50, f"Memory grew too much: {memory_growth:.1f}MB"

    def test_config_memory_footprint(self):
        """Test that config objects have minimal memory footprint."""
        configs = []
        initial_memory = psutil.Process().memory_info().rss

        # Create many config objects
        for _ in range(100):
            configs.append(ContinuousPlayConfig.from_env())

        final_memory = psutil.Process().memory_info().rss
        memory_per_config = (final_memory - initial_memory) / (1024 * 100)  # KB per config

        # Each config should be less than 100KB
        assert memory_per_config < 100, f"Config too large: {memory_per_config:.1f}KB"
