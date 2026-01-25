"""
Unit tests for continuous play configuration.

Tests configuration loading from environment variables,
validation, and preset configurations.
"""

import pytest

from src.training.continuous_play_config import (
    ContinuousPlayConfig,
    LearningConfig,
    MetricsConfig,
    SessionConfig,
    TemperatureSchedule,
    load_config,
)


@pytest.mark.unit
class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SessionConfig()
        assert config.session_duration_minutes == 20
        assert config.max_games == 100
        assert config.max_moves_per_game == 150
        assert config.checkpoint_interval_games == 10

    def test_loads_from_environment(self, monkeypatch):
        """Test loading values from environment variables."""
        monkeypatch.setenv("SESSION_DURATION_MIN", "30")
        monkeypatch.setenv("MAX_GAMES", "50")
        monkeypatch.setenv("MAX_MOVES_PER_GAME", "200")

        config = SessionConfig.from_env()

        assert config.session_duration_minutes == 30
        assert config.max_games == 50
        assert config.max_moves_per_game == 200

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SessionConfig()
        data = config.to_dict()

        assert "session_duration_minutes" in data
        assert "max_games" in data
        assert "checkpoint_dir" in data
        assert isinstance(data["session_duration_minutes"], int)


@pytest.mark.unit
class TestLearningConfig:
    """Tests for LearningConfig."""

    def test_default_values(self):
        """Test default learning configuration."""
        config = LearningConfig()

        assert config.learn_every_n_games == 5
        assert config.min_games_before_learning == 10
        assert config.learning_batch_size == 256
        assert config.learning_rate == 0.001
        assert config.temperature_schedule == TemperatureSchedule.LINEAR_DECAY

    def test_loads_from_environment(self, monkeypatch):
        """Test loading values from environment."""
        monkeypatch.setenv("LEARN_EVERY_N_GAMES", "3")
        monkeypatch.setenv("LEARNING_RATE", "0.0001")
        monkeypatch.setenv("TEMPERATURE_SCHEDULE", "constant")

        config = LearningConfig.from_env()

        assert config.learn_every_n_games == 3
        assert config.learning_rate == 0.0001
        assert config.temperature_schedule == TemperatureSchedule.CONSTANT

    def test_temperature_schedules(self, monkeypatch):
        """Test all temperature schedule types."""
        schedules = ["constant", "linear_decay", "step", "exponential_decay"]

        for schedule in schedules:
            monkeypatch.setenv("TEMPERATURE_SCHEDULE", schedule)
            config = LearningConfig.from_env()
            assert config.temperature_schedule.value == schedule

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LearningConfig()
        data = config.to_dict()

        assert "learning_rate" in data
        assert "temperature_schedule" in data
        assert data["temperature_schedule"] == "linear_decay"


@pytest.mark.unit
class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_values(self):
        """Test default metrics configuration."""
        config = MetricsConfig()

        assert config.enable_prometheus is True
        assert config.enable_wandb is False
        assert config.prometheus_port == 8000
        assert config.generate_html_report is True

    def test_loads_from_environment(self, monkeypatch):
        """Test loading values from environment."""
        monkeypatch.setenv("ENABLE_WANDB", "true")
        monkeypatch.setenv("ENABLE_PROMETHEUS", "false")
        monkeypatch.setenv("PROMETHEUS_PORT", "9090")

        config = MetricsConfig.from_env()

        assert config.enable_wandb is True
        assert config.enable_prometheus is False
        assert config.prometheus_port == 9090

    def test_boolean_parsing(self, monkeypatch):
        """Test boolean environment variable parsing."""
        # Test various true values
        for true_val in ["true", "True", "TRUE", "1", "yes"]:
            monkeypatch.setenv("ENABLE_WANDB", true_val)
            config = MetricsConfig.from_env()
            assert config.enable_wandb is True

        # Test various false values
        for false_val in ["false", "False", "0", "no"]:
            monkeypatch.setenv("ENABLE_WANDB", false_val)
            config = MetricsConfig.from_env()
            assert config.enable_wandb is False


@pytest.mark.unit
class TestContinuousPlayConfig:
    """Tests for unified ContinuousPlayConfig."""

    def test_from_env(self):
        """Test creating config from environment."""
        config = ContinuousPlayConfig.from_env()

        assert config.session is not None
        assert config.learning is not None
        assert config.metrics is not None
        assert config.chess_preset in ["small", "medium", "large"]

    def test_for_quick_test(self):
        """Test quick test preset configuration."""
        config = ContinuousPlayConfig.for_quick_test()

        assert config.session.session_duration_minutes == 1
        assert config.session.max_games == 3
        assert config.learning.min_games_before_learning == 1

    def test_for_development(self):
        """Test development preset configuration."""
        config = ContinuousPlayConfig.for_development()

        assert config.session.session_duration_minutes == 5
        assert config.session.max_games == 10

    def test_for_production(self):
        """Test production preset uses environment."""
        config = ContinuousPlayConfig.for_production()

        # Should use environment defaults
        assert config.session.session_duration_minutes == 20

    def test_validation_valid_config(self):
        """Test validation passes for valid config."""
        config = ContinuousPlayConfig.from_env()
        errors = config.validate()
        assert len(errors) == 0

    def test_validation_invalid_session_duration(self):
        """Test validation fails for invalid session duration."""
        config = ContinuousPlayConfig.from_env()
        config.session.session_duration_minutes = -1

        errors = config.validate()
        assert any("session_duration_minutes" in e for e in errors)

    def test_validation_invalid_learning_rate(self):
        """Test validation fails for invalid learning rate."""
        config = ContinuousPlayConfig.from_env()
        config.learning.learning_rate = -0.001

        errors = config.validate()
        assert any("learning_rate" in e for e in errors)

    def test_validation_invalid_preset(self):
        """Test validation fails for invalid chess preset."""
        config = ContinuousPlayConfig.from_env()
        config.chess_preset = "invalid"

        errors = config.validate()
        assert any("chess_preset" in e for e in errors)

    def test_validation_invalid_device(self):
        """Test validation fails for invalid device."""
        config = ContinuousPlayConfig.from_env()
        config.device = "tpu"  # Not supported

        errors = config.validate()
        assert any("device" in e for e in errors)

    def test_to_dict(self):
        """Test conversion to nested dictionary."""
        config = ContinuousPlayConfig.from_env()
        data = config.to_dict()

        assert "session" in data
        assert "learning" in data
        assert "metrics" in data
        assert "chess_preset" in data
        assert isinstance(data["session"], dict)


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config factory function."""

    def test_load_config_success(self):
        """Test successful config loading."""
        config = load_config()
        assert isinstance(config, ContinuousPlayConfig)

    def test_load_config_raises_on_invalid(self, monkeypatch):
        """Test that load_config raises ValueError for invalid config."""
        monkeypatch.setenv("SESSION_DURATION_MIN", "-10")

        with pytest.raises(ValueError, match="Invalid configuration"):
            load_config()


@pytest.mark.unit
class TestEnvironmentIsolation:
    """Tests ensuring environment variable isolation."""

    def test_env_changes_dont_affect_existing_config(self, monkeypatch):
        """Test that config is snapshot at creation time."""
        # Create config with default
        config1 = ContinuousPlayConfig.from_env()
        original_duration = config1.session.session_duration_minutes

        # Change environment
        monkeypatch.setenv("SESSION_DURATION_MIN", "999")

        # Original config should be unchanged
        assert config1.session.session_duration_minutes == original_duration

        # New config should have new value
        config2 = ContinuousPlayConfig.from_env()
        assert config2.session.session_duration_minutes == 999
