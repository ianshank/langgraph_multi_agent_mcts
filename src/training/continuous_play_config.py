"""
Continuous Play Configuration Module.

All configuration is loaded from environment variables with sensible defaults.
No hardcoded values - everything is configurable via environment.

Best Practices 2025:
- Environment-based configuration
- Dataclass validation
- Type hints throughout
- Factory methods for different presets
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TemperatureSchedule(Enum):
    """Temperature scheduling strategies for exploration."""

    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    STEP = "step"
    EXPONENTIAL_DECAY = "exponential_decay"


def _env_int(key: str, default: int) -> int:
    """Get integer from environment with default."""
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    """Get float from environment with default."""
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment with default."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_str(key: str, default: str) -> str:
    """Get string from environment with default."""
    return os.getenv(key, default)


@dataclass
class SessionConfig:
    """Configuration for a continuous play session.

    All values loaded from environment variables.
    """

    # Session duration and limits
    session_duration_minutes: int = field(default_factory=lambda: _env_int("SESSION_DURATION_MIN", 20))
    max_games: int = field(default_factory=lambda: _env_int("MAX_GAMES", 100))
    max_moves_per_game: int = field(default_factory=lambda: _env_int("MAX_MOVES_PER_GAME", 150))

    # Time control per move (milliseconds)
    time_per_move_ms: int = field(default_factory=lambda: _env_int("TIME_PER_MOVE_MS", 5000))
    increment_per_move_ms: int = field(default_factory=lambda: _env_int("INCREMENT_PER_MOVE_MS", 0))

    # Checkpointing
    checkpoint_interval_games: int = field(default_factory=lambda: _env_int("CHECKPOINT_INTERVAL", 10))
    checkpoint_dir: str = field(default_factory=lambda: _env_str("CHECKPOINT_DIR", "./checkpoints"))
    load_checkpoint_path: str | None = field(default_factory=lambda: _env_str("LOAD_CHECKPOINT", "") or None)

    @classmethod
    def from_env(cls) -> SessionConfig:
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_duration_minutes": self.session_duration_minutes,
            "max_games": self.max_games,
            "max_moves_per_game": self.max_moves_per_game,
            "time_per_move_ms": self.time_per_move_ms,
            "increment_per_move_ms": self.increment_per_move_ms,
            "checkpoint_interval_games": self.checkpoint_interval_games,
            "checkpoint_dir": self.checkpoint_dir,
        }


@dataclass
class LearningConfig:
    """Configuration for online learning during continuous play.

    All values loaded from environment variables.
    """

    # Learning triggers
    learn_every_n_games: int = field(default_factory=lambda: _env_int("LEARN_EVERY_N_GAMES", 5))
    min_games_before_learning: int = field(default_factory=lambda: _env_int("MIN_GAMES_BEFORE_LEARNING", 10))

    # Training parameters
    learning_batch_size: int = field(default_factory=lambda: _env_int("LEARNING_BATCH_SIZE", 256))
    learning_rate: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 0.001))
    weight_decay: float = field(default_factory=lambda: _env_float("WEIGHT_DECAY", 1e-4))

    # Experience buffer
    max_buffer_size: int = field(default_factory=lambda: _env_int("MAX_BUFFER_SIZE", 10000))

    # Temperature scheduling
    temperature_schedule: TemperatureSchedule = field(
        default_factory=lambda: TemperatureSchedule(_env_str("TEMPERATURE_SCHEDULE", "linear_decay"))
    )
    initial_temperature: float = field(default_factory=lambda: _env_float("INITIAL_TEMPERATURE", 1.0))
    final_temperature: float = field(default_factory=lambda: _env_float("FINAL_TEMPERATURE", 0.1))
    temperature_decay_games: int = field(default_factory=lambda: _env_int("TEMPERATURE_DECAY_GAMES", 50))

    # Exploration
    add_noise: bool = field(default_factory=lambda: _env_bool("ADD_NOISE", True))
    noise_weight: float = field(default_factory=lambda: _env_float("NOISE_WEIGHT", 0.25))

    @classmethod
    def from_env(cls) -> LearningConfig:
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "learn_every_n_games": self.learn_every_n_games,
            "min_games_before_learning": self.min_games_before_learning,
            "learning_batch_size": self.learning_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_buffer_size": self.max_buffer_size,
            "temperature_schedule": self.temperature_schedule.value,
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "temperature_decay_games": self.temperature_decay_games,
            "add_noise": self.add_noise,
            "noise_weight": self.noise_weight,
        }


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and monitoring.

    All values loaded from environment variables.
    """

    # Metrics export
    metrics_export_interval_sec: int = field(default_factory=lambda: _env_int("METRICS_EXPORT_INTERVAL", 30))

    # Prometheus
    enable_prometheus: bool = field(default_factory=lambda: _env_bool("ENABLE_PROMETHEUS", True))
    prometheus_port: int = field(default_factory=lambda: _env_int("PROMETHEUS_PORT", 8000))

    # Weights & Biases
    enable_wandb: bool = field(default_factory=lambda: _env_bool("ENABLE_WANDB", False))
    wandb_project: str = field(default_factory=lambda: _env_str("WANDB_PROJECT", "continuous-play"))
    wandb_entity: str = field(default_factory=lambda: _env_str("WANDB_ENTITY", ""))

    # Braintrust
    enable_braintrust: bool = field(default_factory=lambda: _env_bool("ENABLE_BRAINTRUST", False))

    # Reporting
    report_output_dir: str = field(default_factory=lambda: _env_str("REPORT_DIR", "./reports"))
    generate_html_report: bool = field(default_factory=lambda: _env_bool("GENERATE_HTML_REPORT", True))
    generate_json_report: bool = field(default_factory=lambda: _env_bool("GENERATE_JSON_REPORT", True))

    @classmethod
    def from_env(cls) -> MetricsConfig:
        """Create configuration from environment variables."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics_export_interval_sec": self.metrics_export_interval_sec,
            "enable_prometheus": self.enable_prometheus,
            "prometheus_port": self.prometheus_port,
            "enable_wandb": self.enable_wandb,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "enable_braintrust": self.enable_braintrust,
            "report_output_dir": self.report_output_dir,
            "generate_html_report": self.generate_html_report,
            "generate_json_report": self.generate_json_report,
        }


@dataclass
class ContinuousPlayConfig:
    """Unified configuration for continuous play system.

    Combines session, learning, and metrics configuration.
    All values loaded from environment variables.
    """

    session: SessionConfig = field(default_factory=SessionConfig.from_env)
    learning: LearningConfig = field(default_factory=LearningConfig.from_env)
    metrics: MetricsConfig = field(default_factory=MetricsConfig.from_env)

    # Chess configuration preset
    chess_preset: str = field(default_factory=lambda: _env_str("CHESS_PRESET", "small"))

    # Device configuration
    device: str = field(default_factory=lambda: _env_str("DEVICE", "cuda"))

    # Callbacks (set programmatically, not from env)
    on_game_complete: Callable[..., None] | None = None
    on_learning_update: Callable[..., None] | None = None
    on_session_complete: Callable[..., None] | None = None

    @classmethod
    def from_env(cls) -> ContinuousPlayConfig:
        """Create unified configuration from environment variables."""
        return cls(
            session=SessionConfig.from_env(),
            learning=LearningConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )

    @classmethod
    def for_quick_test(cls) -> ContinuousPlayConfig:
        """Create configuration for quick testing (1 minute, 3 games)."""
        config = cls.from_env()
        config.session.session_duration_minutes = 1
        config.session.max_games = 3
        config.learning.min_games_before_learning = 1
        config.learning.learn_every_n_games = 1
        return config

    @classmethod
    def for_development(cls) -> ContinuousPlayConfig:
        """Create configuration for development (5 minutes, 10 games)."""
        config = cls.from_env()
        config.session.session_duration_minutes = 5
        config.session.max_games = 10
        config.learning.min_games_before_learning = 3
        config.learning.learn_every_n_games = 2
        return config

    @classmethod
    def for_production(cls) -> ContinuousPlayConfig:
        """Create configuration for production (from env only)."""
        return cls.from_env()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session": self.session.to_dict(),
            "learning": self.learning.to_dict(),
            "metrics": self.metrics.to_dict(),
            "chess_preset": self.chess_preset,
            "device": self.device,
        }

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Session validation
        if self.session.session_duration_minutes <= 0:
            errors.append("session_duration_minutes must be positive")
        if self.session.max_games <= 0:
            errors.append("max_games must be positive")
        if self.session.max_moves_per_game <= 0:
            errors.append("max_moves_per_game must be positive")

        # Learning validation
        if self.learning.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.learning.learning_batch_size <= 0:
            errors.append("learning_batch_size must be positive")
        if not 0 <= self.learning.initial_temperature <= 2:
            errors.append("initial_temperature should be between 0 and 2")
        if not 0 <= self.learning.final_temperature <= 2:
            errors.append("final_temperature should be between 0 and 2")

        # Chess preset validation
        valid_presets = ["small", "medium", "large"]
        if self.chess_preset not in valid_presets:
            errors.append(f"chess_preset must be one of {valid_presets}")

        # Device validation
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices:
            errors.append(f"device must be one of {valid_devices}")

        return errors


def load_config() -> ContinuousPlayConfig:
    """Load and validate configuration from environment.

    Raises:
        ValueError: If configuration is invalid.
    """
    config = ContinuousPlayConfig.from_env()
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
    return config
