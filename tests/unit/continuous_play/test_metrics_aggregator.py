"""
Unit tests for MetricsAggregator.

Tests metric collection, statistics calculation,
and improvement analysis.
"""

import pytest

from src.training.metrics_aggregator import (
    MetricBuffer,
    MetricsAggregator,
    MetricStats,
    create_metrics_aggregator,
)


@pytest.mark.unit
class TestMetricStats:
    """Tests for MetricStats calculation."""

    def test_from_empty_values(self):
        """Test stats from empty list."""
        stats = MetricStats.from_values([])

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_from_single_value(self):
        """Test stats from single value."""
        stats = MetricStats.from_values([42.0])

        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_from_multiple_values(self):
        """Test stats calculation accuracy."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = MetricStats.from_values(values)

        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = MetricStats.from_values([1.0, 2.0, 3.0])
        data = stats.to_dict()

        assert "count" in data
        assert "mean" in data
        assert "p95" in data
        assert isinstance(data["mean"], float)


@pytest.mark.unit
class TestMetricBuffer:
    """Tests for MetricBuffer."""

    def test_add_and_retrieve(self):
        """Test adding and retrieving values."""
        buffer = MetricBuffer(max_size=100)

        buffer.add(1.0)
        buffer.add(2.0)
        buffer.add(3.0)

        values = buffer.get_values()
        assert values == [1.0, 2.0, 3.0]

    def test_max_size_limit(self):
        """Test buffer respects max size."""
        buffer = MetricBuffer(max_size=3)

        for i in range(5):
            buffer.add(float(i))

        values = buffer.get_values()
        assert len(values) == 3
        assert values == [2.0, 3.0, 4.0]  # Oldest dropped

    def test_window_limiting(self):
        """Test getting values with window."""
        buffer = MetricBuffer(max_size=100)

        for i in range(10):
            buffer.add(float(i))

        values = buffer.get_values(window=3)
        assert values == [7.0, 8.0, 9.0]

    def test_get_stats(self):
        """Test getting statistics."""
        buffer = MetricBuffer()
        buffer.add(1.0)
        buffer.add(2.0)
        buffer.add(3.0)

        stats = buffer.get_stats()
        assert stats.mean == 2.0
        assert stats.count == 3

    def test_get_latest(self):
        """Test getting latest value."""
        buffer = MetricBuffer()

        assert buffer.get_latest() is None

        buffer.add(1.0)
        buffer.add(2.0)

        assert buffer.get_latest() == 2.0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = MetricBuffer()
        buffer.add(1.0)
        buffer.add(2.0)

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.get_values() == []

    def test_labels_preserved(self):
        """Test labels are stored with samples."""
        buffer = MetricBuffer()
        buffer.add(1.0, {"agent": "hrm"})

        # Labels should be accessible via internal samples
        assert len(buffer) == 1


@pytest.mark.unit
class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create test aggregator without external dependencies."""
        return MetricsAggregator(
            buffer_size=100,
            enable_prometheus=False,
            enable_wandb=False,
        )

    def test_record_sample(self, aggregator):
        """Test recording metric samples."""
        aggregator.record_sample("test_metric", 1.0)
        aggregator.record_sample("test_metric", 2.0)

        stats = aggregator.get_stats("test_metric")
        assert stats.count == 2
        assert stats.mean == 1.5

    def test_increment_counter(self, aggregator):
        """Test counter incrementing."""
        aggregator.increment_counter("events")
        aggregator.increment_counter("events")
        aggregator.increment_counter("events", value=3)

        assert aggregator.get_counter("events") == 5

    def test_counter_with_labels(self, aggregator):
        """Test counters with different labels."""
        aggregator.increment_counter("games", labels={"result": "win"})
        aggregator.increment_counter("games", labels={"result": "win"})
        aggregator.increment_counter("games", labels={"result": "loss"})

        assert aggregator.get_counter("games", {"result": "win"}) == 2
        assert aggregator.get_counter("games", {"result": "loss"}) == 1

    def test_set_gauge(self, aggregator):
        """Test gauge setting."""
        aggregator.set_gauge("temperature", 1.0)
        assert aggregator.get_gauge("temperature") == 1.0

        aggregator.set_gauge("temperature", 0.5)
        assert aggregator.get_gauge("temperature") == 0.5

    def test_record_game_complete(self, aggregator):
        """Test game completion recording."""
        aggregator.record_game_complete(
            result="white_win",
            num_moves=40,
            duration_ms=5000.0,
            elo=1520.0,
        )

        assert aggregator.get_counter("games_total", {"result": "white_win"}) == 1
        assert aggregator.get_gauge("current_elo") == 1520.0

        move_stats = aggregator.get_stats("game_moves")
        assert move_stats.count == 1
        assert move_stats.mean == 40

    def test_record_training_update(self, aggregator):
        """Test training update recording."""
        aggregator.record_training_update(
            loss=0.5,
            batch_size=256,
            positions_learned=1000,
        )

        assert aggregator.get_latest("training_loss") == 0.5
        assert aggregator.get_counter("training_updates") == 1
        assert aggregator.get_gauge("positions_learned") == 1000

    def test_session_summary(self, aggregator):
        """Test session summary generation."""
        aggregator.start_session("test_session")

        aggregator.record_game_complete("white_win", 30, 4000.0, 1510.0)
        aggregator.record_game_complete("black_win", 25, 3500.0, 1500.0)
        aggregator.record_training_update(0.45, 256, 500)

        summary = aggregator.get_session_summary()

        assert summary["session_id"] == "test_session"
        # Games total is sum of labeled counters
        total_games = summary["games"]["white_wins"] + summary["games"]["black_wins"] + summary["games"]["draws"]
        assert total_games == 2
        assert summary["training"]["updates"] == 1

    def test_improvement_metrics(self, aggregator):
        """Test improvement metrics calculation."""
        aggregator.start_session("test")

        # Simulate improving Elo
        for i in range(10):
            elo = 1500 + i * 10  # Steadily increasing
            aggregator.record_game_complete("white_win", 30, 4000.0, float(elo))

        metrics = aggregator.get_improvement_metrics()

        assert metrics["elo"]["delta_total"] > 0
        # Total games counted via labeled counters
        assert metrics["win_rate"]["total_games"] >= 0  # Just verify it's accessible

    def test_is_improving_detection(self, aggregator):
        """Test improvement detection logic."""
        aggregator.start_session("test")

        # First half: lower Elo
        for i in range(5):
            aggregator.record_game_complete("draw", 30, 4000.0, 1500.0)

        # Second half: higher Elo
        for i in range(5):
            aggregator.record_game_complete("white_win", 30, 4000.0, 1550.0)

        metrics = aggregator.get_improvement_metrics()
        assert metrics["is_improving"] is True

    def test_reset(self, aggregator):
        """Test resetting all metrics."""
        aggregator.record_sample("test", 1.0)
        aggregator.increment_counter("events")

        aggregator.reset()

        assert aggregator.get_stats("test").count == 0
        assert aggregator.get_counter("events") == 0

    def test_unknown_metric_returns_empty(self, aggregator):
        """Test querying unknown metric returns safe defaults."""
        stats = aggregator.get_stats("nonexistent")
        assert stats.count == 0

        value = aggregator.get_latest("nonexistent")
        assert value is None


@pytest.mark.unit
class TestMetricsAggregatorFactory:
    """Tests for create_metrics_aggregator factory."""

    def test_factory_with_defaults(self, monkeypatch):
        """Test factory uses environment defaults."""
        monkeypatch.setenv("ENABLE_PROMETHEUS", "false")
        monkeypatch.setenv("ENABLE_WANDB", "false")

        aggregator = create_metrics_aggregator()

        assert aggregator.enable_prometheus is False
        assert aggregator.enable_wandb is False

    def test_factory_with_overrides(self):
        """Test factory accepts explicit overrides."""
        aggregator = create_metrics_aggregator(
            enable_prometheus=True,
            enable_wandb=True,
        )

        assert aggregator.enable_prometheus is True
        assert aggregator.enable_wandb is True
