"""
Metrics Aggregator for Continuous Play.

Unified metrics collection combining:
- Prometheus metrics export
- W&B experiment tracking
- In-memory statistics for real-time dashboard
- Improvement analysis

Best Practices 2025:
- Single responsibility
- Observable patterns
- Async-compatible
- Pluggable exporters
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """Single metric sample with timestamp."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of a metric."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: list[float]) -> MetricStats:
        """Calculate statistics from list of values."""
        if not values:
            return cls(
                count=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                p50=0.0,
                p95=0.0,
                p99=0.0,
            )

        sorted_values = sorted(values)
        n = len(sorted_values)

        return cls(
            count=n,
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0.0,
            min=min(values),
            max=max(values),
            p50=sorted_values[int(n * 0.50)] if n > 0 else 0.0,
            p95=sorted_values[int(n * 0.95)] if n > 0 else 0.0,
            p99=sorted_values[int(n * 0.99)] if n > 0 else 0.0,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


class MetricBuffer:
    """Rolling buffer for metric samples."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize buffer.

        Args:
            max_size: Maximum samples to keep
        """
        self.max_size = max_size
        self._samples: deque[MetricSample] = deque(maxlen=max_size)

    def add(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Add a sample."""
        self._samples.append(MetricSample(value=value, labels=labels or {}))

    def get_values(self, window: int | None = None) -> list[float]:
        """Get values, optionally limited to recent window."""
        samples = list(self._samples)
        if window is not None:
            samples = samples[-window:]
        return [s.value for s in samples]

    def get_stats(self, window: int | None = None) -> MetricStats:
        """Get statistics for samples."""
        values = self.get_values(window)
        return MetricStats.from_values(values)

    def get_latest(self) -> float | None:
        """Get most recent value."""
        if self._samples:
            return self._samples[-1].value
        return None

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()

    def __len__(self) -> int:
        """Get number of samples."""
        return len(self._samples)


class MetricsAggregator:
    """
    Aggregates metrics from continuous play sessions.

    Provides:
    - Real-time metric tracking
    - Statistical summaries
    - Export to various backends
    - Improvement analysis
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        enable_prometheus: bool = True,
        enable_wandb: bool = False,
    ) -> None:
        """Initialize aggregator.

        Args:
            buffer_size: Max samples per metric
            enable_prometheus: Enable Prometheus export
            enable_wandb: Enable W&B logging
        """
        self.buffer_size = buffer_size
        self.enable_prometheus = enable_prometheus
        self.enable_wandb = enable_wandb

        # Metric buffers
        self._buffers: dict[str, MetricBuffer] = {}

        # Counters
        self._counters: dict[str, int] = {}

        # Gauges (current values)
        self._gauges: dict[str, float] = {}

        # Prometheus registry (lazy init)
        self._prometheus_metrics: dict[str, Any] = {}

        # Session tracking
        self._session_start: datetime | None = None
        self._session_id: str = ""

        logger.info(
            "MetricsAggregator initialized: prometheus=%s, wandb=%s",
            enable_prometheus,
            enable_wandb,
        )

    def start_session(self, session_id: str) -> None:
        """Start tracking a new session."""
        self._session_start = datetime.now()
        self._session_id = session_id
        self.reset()
        logger.info("Started metrics session: %s", session_id)

    def reset(self) -> None:
        """Reset all metrics."""
        self._buffers.clear()
        self._counters.clear()
        self._gauges.clear()

    def _get_buffer(self, name: str) -> MetricBuffer:
        """Get or create metric buffer."""
        if name not in self._buffers:
            self._buffers[name] = MetricBuffer(self.buffer_size)
        return self._buffers[name]

    # ========== Recording Methods ==========

    def record_sample(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric sample.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the sample
        """
        buffer = self._get_buffer(name)
        buffer.add(value, labels)

        # Update Prometheus if enabled
        if self.enable_prometheus:
            self._update_prometheus_histogram(name, value, labels)

    def increment_counter(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Amount to increment
            labels: Optional labels
        """
        key = self._counter_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value

        # Update Prometheus if enabled
        if self.enable_prometheus:
            self._update_prometheus_counter(name, value, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels
        """
        key = self._counter_key(name, labels)
        self._gauges[key] = value

        # Update Prometheus if enabled
        if self.enable_prometheus:
            self._update_prometheus_gauge(name, value, labels)

    def _counter_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Generate unique key for counter/gauge."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    # ========== Game Metrics ==========

    def record_game_complete(
        self,
        result: str,
        num_moves: int,
        duration_ms: float,
        elo: float,
    ) -> None:
        """Record game completion metrics.

        Args:
            result: Game result (white_win, black_win, draw)
            num_moves: Number of moves in game
            duration_ms: Game duration in milliseconds
            elo: Current Elo estimate
        """
        self.increment_counter("games_total", labels={"result": result})
        self.record_sample("game_moves", num_moves)
        self.record_sample("game_duration_ms", duration_ms)
        self.set_gauge("current_elo", elo)
        self.record_sample("elo_history", elo)

    def record_training_update(
        self,
        loss: float,
        batch_size: int,
        positions_learned: int,
    ) -> None:
        """Record training update metrics.

        Args:
            loss: Training loss
            batch_size: Batch size used
            positions_learned: Total positions learned
        """
        self.record_sample("training_loss", loss)
        self.increment_counter("training_updates")
        self.set_gauge("positions_learned", positions_learned)
        self.set_gauge("experience_buffer_size", batch_size)

    def record_move_time(self, time_ms: float) -> None:
        """Record time for a single move.

        Args:
            time_ms: Move time in milliseconds
        """
        self.record_sample("move_time_ms", time_ms)

    # ========== Query Methods ==========

    def get_stats(self, name: str, window: int | None = None) -> MetricStats:
        """Get statistics for a metric.

        Args:
            name: Metric name
            window: Optional window of recent samples

        Returns:
            MetricStats for the metric
        """
        if name not in self._buffers:
            return MetricStats.from_values([])
        return self._buffers[name].get_stats(window)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get counter value.

        Args:
            name: Counter name
            labels: Optional labels

        Returns:
            Counter value
        """
        key = self._counter_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get gauge value.

        Args:
            name: Gauge name
            labels: Optional labels

        Returns:
            Gauge value
        """
        key = self._counter_key(name, labels)
        return self._gauges.get(key, 0.0)

    def get_latest(self, name: str) -> float | None:
        """Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest value or None
        """
        if name not in self._buffers:
            return None
        return self._buffers[name].get_latest()

    # ========== Summary Methods ==========

    def get_session_summary(self) -> dict[str, Any]:
        """Get summary of current session metrics."""
        white_wins = self.get_counter("games_total", {"result": "white_win"})
        black_wins = self.get_counter("games_total", {"result": "black_win"})
        draws = self.get_counter("games_total", {"result": "draw"})
        total_games = white_wins + black_wins + draws

        return {
            "session_id": self._session_id,
            "session_start": self._session_start.isoformat() if self._session_start else None,
            "games": {
                "total": total_games,
                "white_wins": white_wins,
                "black_wins": black_wins,
                "draws": draws,
            },
            "elo": {
                "current": self.get_gauge("current_elo"),
                "stats": self.get_stats("elo_history").to_dict(),
            },
            "training": {
                "updates": self.get_counter("training_updates"),
                "positions_learned": self.get_gauge("positions_learned"),
                "loss_stats": self.get_stats("training_loss").to_dict(),
            },
            "performance": {
                "game_moves": self.get_stats("game_moves").to_dict(),
                "game_duration_ms": self.get_stats("game_duration_ms").to_dict(),
                "move_time_ms": self.get_stats("move_time_ms").to_dict(),
            },
        }

    def get_improvement_metrics(self) -> dict[str, Any]:
        """Get metrics focused on improvement analysis."""
        elo_stats = self.get_stats("elo_history")
        elo_recent = self.get_stats("elo_history", window=10)
        loss_stats = self.get_stats("training_loss")

        # Calculate improvement indicators
        white_wins = self.get_counter("games_total", {"result": "white_win"})
        black_wins = self.get_counter("games_total", {"result": "black_win"})
        draws = self.get_counter("games_total", {"result": "draw"})
        total_games = white_wins + black_wins + draws
        wins = white_wins + black_wins
        win_rate = wins / total_games if total_games > 0 else 0.0

        return {
            "elo": {
                "current": self.get_gauge("current_elo"),
                "initial": elo_stats.min if elo_stats.count > 0 else 1500.0,
                "delta_total": self.get_gauge("current_elo") - (elo_stats.min if elo_stats.count > 0 else 1500.0),
                "delta_recent_10": elo_recent.max - elo_recent.min if elo_recent.count > 1 else 0.0,
            },
            "training": {
                "avg_loss": loss_stats.mean,
                "loss_trend": "improving"
                if loss_stats.count > 1 and self.get_latest("training_loss") < loss_stats.mean
                else "stable",
                "updates": self.get_counter("training_updates"),
            },
            "win_rate": {
                "overall": win_rate,
                "total_games": total_games,
            },
            "is_improving": self._calculate_is_improving(),
        }

    def _calculate_is_improving(self) -> bool:
        """Determine if the agent is improving."""
        elo_recent = self.get_stats("elo_history", window=10)

        # Improvement criteria:
        # 1. Elo is increasing over recent games
        # 2. Loss is stable or decreasing

        if elo_recent.count < 5:
            return False  # Not enough data

        elo_values = self._buffers.get("elo_history", MetricBuffer()).get_values(10)
        if len(elo_values) >= 2:
            # Check if recent Elo is higher than earlier
            first_half = statistics.mean(elo_values[: len(elo_values) // 2])
            second_half = statistics.mean(elo_values[len(elo_values) // 2 :])
            return second_half > first_half

        return False

    # ========== Prometheus Integration ==========

    def _update_prometheus_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None,
    ) -> None:
        """Update Prometheus histogram."""
        try:
            from prometheus_client import Histogram

            metric_name = f"continuous_play_{name}"
            if metric_name not in self._prometheus_metrics:
                label_names = list(labels.keys()) if labels else []
                self._prometheus_metrics[metric_name] = Histogram(
                    metric_name,
                    f"Continuous play metric: {name}",
                    label_names,
                )

            hist = self._prometheus_metrics[metric_name]
            if labels:
                hist.labels(**labels).observe(value)
            else:
                hist.observe(value)
        except ImportError:
            pass  # Prometheus not available
        except Exception as e:
            logger.debug("Prometheus histogram error: %s", e)

    def _update_prometheus_counter(
        self,
        name: str,
        value: int,
        labels: dict[str, str] | None,
    ) -> None:
        """Update Prometheus counter."""
        try:
            from prometheus_client import Counter

            metric_name = f"continuous_play_{name}"
            if metric_name not in self._prometheus_metrics:
                label_names = list(labels.keys()) if labels else []
                self._prometheus_metrics[metric_name] = Counter(
                    metric_name,
                    f"Continuous play counter: {name}",
                    label_names,
                )

            counter = self._prometheus_metrics[metric_name]
            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)
        except ImportError:
            pass  # Prometheus not available
        except Exception as e:
            logger.debug("Prometheus counter error: %s", e)

    def _update_prometheus_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None,
    ) -> None:
        """Update Prometheus gauge."""
        try:
            from prometheus_client import Gauge

            metric_name = f"continuous_play_{name}"
            if metric_name not in self._prometheus_metrics:
                label_names = list(labels.keys()) if labels else []
                self._prometheus_metrics[metric_name] = Gauge(
                    metric_name,
                    f"Continuous play gauge: {name}",
                    label_names,
                )

            gauge = self._prometheus_metrics[metric_name]
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
        except ImportError:
            pass  # Prometheus not available
        except Exception as e:
            logger.debug("Prometheus gauge error: %s", e)

    # ========== W&B Integration ==========

    def log_to_wandb(self, step: int | None = None) -> None:
        """Log current metrics to W&B.

        Args:
            step: Optional step number
        """
        if not self.enable_wandb:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            metrics = {
                "elo/current": self.get_gauge("current_elo"),
                "games/total": self.get_counter("games_total"),
                "training/loss": self.get_latest("training_loss") or 0.0,
                "training/positions_learned": self.get_gauge("positions_learned"),
                "performance/avg_move_time_ms": self.get_stats("move_time_ms").mean,
            }

            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)

        except ImportError:
            pass  # W&B not available
        except Exception as e:
            logger.debug("W&B logging error: %s", e)


def create_metrics_aggregator(
    enable_prometheus: bool | None = None,
    enable_wandb: bool | None = None,
) -> MetricsAggregator:
    """Factory function to create metrics aggregator.

    Args:
        enable_prometheus: Override Prometheus setting (uses env if None)
        enable_wandb: Override W&B setting (uses env if None)

    Returns:
        Configured MetricsAggregator
    """
    import os

    if enable_prometheus is None:
        enable_prometheus = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    if enable_wandb is None:
        enable_wandb = os.getenv("ENABLE_WANDB", "false").lower() == "true"

    return MetricsAggregator(
        enable_prometheus=enable_prometheus,
        enable_wandb=enable_wandb,
    )
