"""
Smoke tests for continuous play imports.

Verifies all modules can be imported without errors.
"""

import pytest


@pytest.mark.smoke
class TestImports:
    """Verify all continuous play modules import successfully."""

    def test_import_config(self):
        """Test importing configuration module."""
        from src.training.continuous_play_config import (
            ContinuousPlayConfig,
            LearningConfig,
            MetricsConfig,
            SessionConfig,
            TemperatureSchedule,
            load_config,
        )

        assert ContinuousPlayConfig is not None
        assert SessionConfig is not None
        assert LearningConfig is not None
        assert MetricsConfig is not None
        assert TemperatureSchedule is not None
        assert load_config is not None

    def test_import_orchestrator(self):
        """Test importing orchestrator module."""
        from src.training.continuous_play_orchestrator import (
            ContinuousPlayOrchestrator,
            LiveMetrics,
            MetricsHistory,
            SessionResult,
            create_orchestrator,
        )

        assert ContinuousPlayOrchestrator is not None
        assert SessionResult is not None
        assert LiveMetrics is not None
        assert MetricsHistory is not None
        assert create_orchestrator is not None

    def test_import_metrics_aggregator(self):
        """Test importing metrics aggregator module."""
        from src.training.metrics_aggregator import (
            MetricBuffer,
            MetricsAggregator,
            MetricStats,
            create_metrics_aggregator,
        )

        assert MetricsAggregator is not None
        assert MetricBuffer is not None
        assert MetricStats is not None
        assert create_metrics_aggregator is not None

    def test_import_continuous_learning(self):
        """Test importing existing continuous learning module."""
        from src.games.chess.continuous_learning import (
            ContinuousLearningConfig,
            ContinuousLearningSession,
            GameRecord,
            GameResult,
            OnlineLearner,
            ScoreCard,
            create_learning_session,
        )

        assert ContinuousLearningSession is not None
        assert ContinuousLearningConfig is not None
        assert OnlineLearner is not None
        assert ScoreCard is not None
        assert GameRecord is not None
        assert GameResult is not None
        assert create_learning_session is not None


@pytest.mark.smoke
class TestConfigInstantiation:
    """Verify configurations can be instantiated."""

    def test_session_config_instantiates(self):
        """Test SessionConfig instantiation."""
        from src.training.continuous_play_config import SessionConfig

        config = SessionConfig()
        assert config.session_duration_minutes > 0

    def test_learning_config_instantiates(self):
        """Test LearningConfig instantiation."""
        from src.training.continuous_play_config import LearningConfig

        config = LearningConfig()
        assert config.learning_rate > 0

    def test_metrics_config_instantiates(self):
        """Test MetricsConfig instantiation."""
        from src.training.continuous_play_config import MetricsConfig

        config = MetricsConfig()
        assert config.prometheus_port > 0

    def test_full_config_instantiates(self):
        """Test ContinuousPlayConfig instantiation."""
        from src.training.continuous_play_config import ContinuousPlayConfig

        config = ContinuousPlayConfig.from_env()
        assert config.session is not None
        assert config.learning is not None
        assert config.metrics is not None

    def test_config_validates(self):
        """Test configuration validation."""
        from src.training.continuous_play_config import ContinuousPlayConfig

        config = ContinuousPlayConfig.from_env()
        errors = config.validate()
        assert isinstance(errors, list)


@pytest.mark.smoke
class TestMetricsAggregatorBasics:
    """Verify metrics aggregator basic functionality."""

    def test_aggregator_instantiates(self):
        """Test MetricsAggregator instantiation."""
        from src.training.metrics_aggregator import MetricsAggregator

        aggregator = MetricsAggregator(
            enable_prometheus=False,
            enable_wandb=False,
        )
        assert aggregator is not None

    def test_can_record_sample(self):
        """Test basic sample recording."""
        from src.training.metrics_aggregator import MetricsAggregator

        aggregator = MetricsAggregator(
            enable_prometheus=False,
            enable_wandb=False,
        )
        aggregator.record_sample("test", 1.0)

        stats = aggregator.get_stats("test")
        assert stats.count == 1
