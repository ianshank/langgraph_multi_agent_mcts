"""
Continuous Play Orchestrator.

Orchestrates continuous self-play learning sessions with:
- Game loop management
- Metrics collection and export
- Learning updates
- Report generation

Best Practices 2025:
- Async-first design
- Dependency injection
- Observable state
- Graceful shutdown
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.training.continuous_play_config import ContinuousPlayConfig, load_config

if TYPE_CHECKING:
    from src.games.chess.continuous_learning import (
        ContinuousLearningSession,
        GameRecord,
        ScoreCard,
    )

logger = logging.getLogger(__name__)


@dataclass
class SessionResult:
    """Result of a continuous play session."""

    scorecard_data: dict[str, Any]
    total_games: int
    session_duration_seconds: float
    games_per_minute: float
    final_elo: float
    elo_delta: float
    total_positions_learned: int
    avg_training_loss: float
    report_path: Path | None
    metrics_path: Path | None
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scorecard": self.scorecard_data,
            "total_games": self.total_games,
            "session_duration_seconds": self.session_duration_seconds,
            "games_per_minute": self.games_per_minute,
            "final_elo": self.final_elo,
            "elo_delta": self.elo_delta,
            "total_positions_learned": self.total_positions_learned,
            "avg_training_loss": self.avg_training_loss,
            "report_path": str(self.report_path) if self.report_path else None,
            "metrics_path": str(self.metrics_path) if self.metrics_path else None,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }


@dataclass
class LiveMetrics:
    """Real-time metrics during session."""

    games_completed: int = 0
    current_game_id: str = ""
    current_fen: str = ""
    last_move: str | None = None
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    positions_learned: int = 0
    last_training_loss: float = 0.0
    current_elo: float = 1500.0
    session_start: datetime | None = None
    last_update: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for UI updates."""
        elapsed = 0.0
        if self.session_start:
            elapsed = (datetime.now() - self.session_start).total_seconds()

        return {
            "games_completed": self.games_completed,
            "current_game_id": self.current_game_id,
            "current_fen": self.current_fen,
            "last_move": self.last_move,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "draws": self.draws,
            "positions_learned": self.positions_learned,
            "last_training_loss": self.last_training_loss,
            "current_elo": self.current_elo,
            "elapsed_seconds": elapsed,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class MetricsHistory:
    """History of metrics for trend analysis."""

    elo_history: list[tuple[int, float]] = field(default_factory=list)  # (game, elo)
    loss_history: list[tuple[int, float]] = field(default_factory=list)  # (game, loss)
    win_rate_history: list[tuple[int, float]] = field(default_factory=list)  # (game, rate)
    game_length_history: list[tuple[int, int]] = field(default_factory=list)  # (game, moves)

    def add_game(
        self,
        game_num: int,
        elo: float,
        win_rate: float,
        game_length: int,
    ) -> None:
        """Record metrics for a completed game."""
        self.elo_history.append((game_num, elo))
        self.win_rate_history.append((game_num, win_rate))
        self.game_length_history.append((game_num, game_length))

    def add_training(self, game_num: int, loss: float) -> None:
        """Record training loss."""
        self.loss_history.append((game_num, loss))

    def get_elo_delta(self, window: int = 10) -> float:
        """Get Elo change over last N games."""
        if len(self.elo_history) < 2:
            return 0.0
        recent = self.elo_history[-window:]
        if len(recent) < 2:
            return 0.0
        return recent[-1][1] - recent[0][1]

    def get_avg_loss(self, window: int = 5) -> float:
        """Get average loss over last N training updates."""
        if not self.loss_history:
            return 0.0
        recent = self.loss_history[-window:]
        return sum(loss for _, loss in recent) / len(recent)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "elo_history": self.elo_history,
            "loss_history": self.loss_history,
            "win_rate_history": self.win_rate_history,
            "game_length_history": self.game_length_history,
        }


class ContinuousPlayOrchestrator:
    """
    Orchestrates continuous play learning sessions.

    Responsibilities:
    - Initialize and manage learning session
    - Collect and export metrics
    - Handle callbacks for UI updates
    - Generate reports
    """

    def __init__(
        self,
        config: ContinuousPlayConfig | None = None,
        session: ContinuousLearningSession | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: Configuration (loaded from env if None)
            session: Pre-created session (created from config if None)
        """
        self.config = config or load_config()
        self._session = session
        self._is_running = False
        self._is_paused = False

        # Metrics
        self.live_metrics = LiveMetrics()
        self.metrics_history = MetricsHistory()
        self._initial_elo = 1500.0

        # Callbacks
        self._game_callbacks: list[Any] = []
        self._learning_callbacks: list[Any] = []
        self._metrics_callbacks: list[Any] = []

        logger.info("ContinuousPlayOrchestrator initialized with config: %s", self.config.chess_preset)

    @property
    def session(self) -> ContinuousLearningSession:
        """Lazy-load the learning session."""
        if self._session is None:
            from src.games.chess.config import ChessConfig
            from src.games.chess.continuous_learning import (
                ContinuousLearningConfig,
                ContinuousLearningSession,
            )

            chess_config = ChessConfig.from_preset(self.config.chess_preset)
            chess_config.device = self.config.device

            learning_config = ContinuousLearningConfig(
                max_session_minutes=self.config.session.session_duration_minutes,
                max_games=self.config.session.max_games,
                max_moves_per_game=self.config.session.max_moves_per_game,
                learn_every_n_games=self.config.learning.learn_every_n_games,
                min_games_before_learning=self.config.learning.min_games_before_learning,
                learning_batch_size=self.config.learning.learning_batch_size,
                learning_rate=self.config.learning.learning_rate,
                temperature_schedule=self.config.learning.temperature_schedule.value,
                initial_temperature=self.config.learning.initial_temperature,
                final_temperature=self.config.learning.final_temperature,
                temperature_decay_games=self.config.learning.temperature_decay_games,
                add_noise=self.config.learning.add_noise,
                noise_weight=self.config.learning.noise_weight,
                time_per_move_ms=self.config.session.time_per_move_ms,
                checkpoint_dir=self.config.session.checkpoint_dir,
                checkpoint_interval=self.config.session.checkpoint_interval_games,
                load_checkpoint_path=self.config.session.load_checkpoint_path,
                on_game_complete=self._on_game_complete,
                on_learning_update=self._on_learning_update,
                on_session_complete=self._on_session_complete,
            )

            self._session = ContinuousLearningSession(chess_config, learning_config)

        return self._session

    def register_game_callback(self, callback: Any) -> None:
        """Register callback for game completion events."""
        self._game_callbacks.append(callback)

    def register_learning_callback(self, callback: Any) -> None:
        """Register callback for learning update events."""
        self._learning_callbacks.append(callback)

    def register_metrics_callback(self, callback: Any) -> None:
        """Register callback for metrics updates."""
        self._metrics_callbacks.append(callback)

    def _on_game_complete(self, record: GameRecord) -> None:
        """Handle game completion."""
        from src.games.chess.continuous_learning import GameResult

        # Update live metrics
        self.live_metrics.games_completed += 1
        self.live_metrics.current_game_id = record.game_id
        self.live_metrics.current_fen = record.final_fen
        self.live_metrics.last_move = record.moves[-1] if record.moves else None
        self.live_metrics.last_update = datetime.now()

        if record.result == GameResult.WHITE_WIN:
            self.live_metrics.white_wins += 1
        elif record.result == GameResult.BLACK_WIN:
            self.live_metrics.black_wins += 1
        else:
            self.live_metrics.draws += 1

        # Update from scorecard
        scorecard = self.session.scorecard
        self.live_metrics.current_elo = scorecard.elo_estimate
        self.live_metrics.positions_learned = scorecard.total_positions_learned

        # Record history
        total = self.live_metrics.games_completed
        win_rate = scorecard.win_rate
        self.metrics_history.add_game(
            game_num=total,
            elo=scorecard.elo_estimate,
            win_rate=win_rate,
            game_length=len(record.moves),
        )

        # Invoke callbacks
        for callback in self._game_callbacks:
            try:
                callback(record, self.live_metrics)
            except Exception as e:
                logger.warning("Game callback error: %s", e)

        # Invoke metrics callbacks
        for callback in self._metrics_callbacks:
            try:
                callback(self.live_metrics.to_dict())
            except Exception as e:
                logger.warning("Metrics callback error: %s", e)

        logger.info(
            "Game %d complete: %s in %d moves, Elo: %.0f",
            total,
            record.result.value,
            len(record.moves),
            scorecard.elo_estimate,
        )

    def _on_learning_update(self, loss: float, game_num: int) -> None:
        """Handle learning update."""
        self.live_metrics.last_training_loss = loss
        self.live_metrics.last_update = datetime.now()
        self.metrics_history.add_training(game_num, loss)

        # Invoke callbacks
        for callback in self._learning_callbacks:
            try:
                callback(loss, game_num)
            except Exception as e:
                logger.warning("Learning callback error: %s", e)

        logger.info("Training update at game %d: loss=%.4f", game_num, loss)

    def _on_session_complete(self, scorecard: ScoreCard) -> None:
        """Handle session completion."""
        logger.info(
            "Session complete: %d games, W:%d B:%d D:%d, Elo: %.0f",
            scorecard.total_games,
            scorecard.white_wins,
            scorecard.black_wins,
            scorecard.draws,
            scorecard.elo_estimate,
        )

    async def run_session(
        self,
        progress_callback: Any | None = None,
    ) -> SessionResult:
        """Run a continuous play session.

        Args:
            progress_callback: Optional callback(game_num, max_games, scorecard)

        Returns:
            SessionResult with all session data
        """
        self._is_running = True
        self._is_paused = False
        start_time = datetime.now()

        # Initialize metrics
        self.live_metrics = LiveMetrics(session_start=start_time)
        self._initial_elo = 1500.0

        logger.info(
            "Starting continuous play session: %d min, %d games max",
            self.config.session.session_duration_minutes,
            self.config.session.max_games,
        )

        try:
            # Run the session
            scorecard = await self.session.run_session(
                max_minutes=self.config.session.session_duration_minutes,
                max_games=self.config.session.max_games,
                progress_callback=progress_callback,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Calculate metrics
            games_per_minute = scorecard.total_games / (duration / 60) if duration > 0 else 0
            elo_delta = scorecard.elo_estimate - self._initial_elo
            avg_loss = self.metrics_history.get_avg_loss()

            # Generate reports
            report_path = None
            metrics_path = None

            if self.config.metrics.generate_html_report:
                report_path = await self._generate_html_report(scorecard, start_time)

            if self.config.metrics.generate_json_report:
                metrics_path = await self._generate_json_report(scorecard, start_time)

            result = SessionResult(
                scorecard_data=scorecard.to_dict(),
                total_games=scorecard.total_games,
                session_duration_seconds=duration,
                games_per_minute=games_per_minute,
                final_elo=scorecard.elo_estimate,
                elo_delta=elo_delta,
                total_positions_learned=scorecard.total_positions_learned,
                avg_training_loss=avg_loss,
                report_path=report_path,
                metrics_path=metrics_path,
                start_time=start_time,
                end_time=end_time,
            )

            logger.info("Session result: %s", result.to_dict())
            return result

        finally:
            self._is_running = False

    async def _generate_html_report(
        self,
        scorecard: ScoreCard,
        start_time: datetime,
    ) -> Path:
        """Generate HTML report."""
        output_dir = Path(self.config.metrics.report_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"session_report_{timestamp}.html"

        # Generate simple HTML report
        html = self._create_html_report(scorecard)
        report_path.write_text(html)

        logger.info("HTML report generated: %s", report_path)
        return report_path

    async def _generate_json_report(
        self,
        scorecard: ScoreCard,
        start_time: datetime,
    ) -> Path:
        """Generate JSON metrics export."""
        output_dir = Path(self.config.metrics.report_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        metrics_path = output_dir / f"session_metrics_{timestamp}.json"

        data = {
            "config": self.config.to_dict(),
            "scorecard": scorecard.to_dict(),
            "metrics_history": self.metrics_history.to_dict(),
            "live_metrics_final": self.live_metrics.to_dict(),
            "generated_at": datetime.now().isoformat(),
        }

        metrics_path.write_text(json.dumps(data, indent=2))
        logger.info("JSON metrics generated: %s", metrics_path)
        return metrics_path

    def _create_html_report(self, scorecard: ScoreCard) -> str:
        """Create HTML report content."""
        elo_delta = scorecard.elo_estimate - self._initial_elo
        elo_class = "positive" if elo_delta >= 0 else "negative"

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Continuous Play Session Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 15px 25px; text-align: center; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 0; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Continuous Play Session Report</h1>
        <p class="timestamp">Generated: {datetime.now().isoformat()}</p>

        <div class="card">
            <h2>Session Summary</h2>
            <div class="metric">
                <div class="metric-value">{scorecard.total_games}</div>
                <div class="metric-label">Total Games</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.white_wins}</div>
                <div class="metric-label">White Wins</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.black_wins}</div>
                <div class="metric-label">Black Wins</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.draws}</div>
                <div class="metric-label">Draws</div>
            </div>
        </div>

        <div class="card">
            <h2>Learning Progress</h2>
            <div class="metric">
                <div class="metric-value">{scorecard.elo_estimate:.0f}</div>
                <div class="metric-label">Final Elo</div>
            </div>
            <div class="metric">
                <div class="metric-value {elo_class}">{elo_delta:+.0f}</div>
                <div class="metric-label">Elo Change</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.total_positions_learned}</div>
                <div class="metric-label">Positions Learned</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.last_loss:.4f}</div>
                <div class="metric-label">Last Loss</div>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="metric">
                <div class="metric-value">{scorecard.avg_game_length:.1f}</div>
                <div class="metric-label">Avg Game Length</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{scorecard.win_streak}</div>
                <div class="metric-label">Best Win Streak</div>
            </div>
        </div>

        <div class="card">
            <h2>Configuration</h2>
            <p><strong>Preset:</strong> {self.config.chess_preset}</p>
            <p><strong>Device:</strong> {self.config.device}</p>
            <p><strong>Session Duration:</strong> {self.config.session.session_duration_minutes} minutes</p>
            <p><strong>Max Games:</strong> {self.config.session.max_games}</p>
            <p><strong>Learning Rate:</strong> {self.config.learning.learning_rate}</p>
            <p><strong>Temperature Schedule:</strong> {self.config.learning.temperature_schedule.value}</p>
        </div>
    </div>
</body>
</html>"""

    def pause(self) -> None:
        """Pause the session."""
        self._is_paused = True
        if self._session:
            self._session.pause()

    def resume(self) -> None:
        """Resume the session."""
        self._is_paused = False
        if self._session:
            self._session.resume()

    def stop(self) -> None:
        """Stop the session."""
        self._is_running = False
        if self._session:
            self._session.stop()

    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self._is_running

    @property
    def is_paused(self) -> bool:
        """Check if session is paused."""
        return self._is_paused

    def get_live_metrics(self) -> dict[str, Any]:
        """Get current live metrics for UI."""
        return self.live_metrics.to_dict()

    def get_improvement_summary(self) -> dict[str, Any]:
        """Get summary of improvement during session."""
        return {
            "elo_delta_total": self.metrics_history.get_elo_delta(window=1000),
            "elo_delta_recent": self.metrics_history.get_elo_delta(window=10),
            "avg_training_loss": self.metrics_history.get_avg_loss(),
            "games_completed": self.live_metrics.games_completed,
            "win_rate": (self.live_metrics.white_wins + self.live_metrics.black_wins)
            / max(1, self.live_metrics.games_completed),
        }


def create_orchestrator(
    preset: str = "default",
) -> ContinuousPlayOrchestrator:
    """Factory function to create orchestrator with preset configuration.

    Args:
        preset: "quick_test", "development", "production", or "default"

    Returns:
        Configured ContinuousPlayOrchestrator
    """
    if preset == "quick_test":
        config = ContinuousPlayConfig.for_quick_test()
    elif preset == "development":
        config = ContinuousPlayConfig.for_development()
    elif preset == "production":
        config = ContinuousPlayConfig.for_production()
    else:
        config = ContinuousPlayConfig.from_env()

    return ContinuousPlayOrchestrator(config)
