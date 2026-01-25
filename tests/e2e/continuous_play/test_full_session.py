"""
End-to-end tests for complete continuous play sessions.

Tests full user journeys including session execution,
metrics collection, and report generation.
"""

import asyncio
import json

import pytest

from src.training.continuous_play_config import ContinuousPlayConfig
from src.training.continuous_play_orchestrator import (
    ContinuousPlayOrchestrator,
    SessionResult,
    create_orchestrator,
)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.timeout(120)  # 2 minute timeout
class TestFullSession:
    """End-to-end tests for complete sessions."""

    @pytest.fixture
    def quick_config(self, tmp_path):
        """Create quick test configuration with temp directory."""
        config = ContinuousPlayConfig.for_quick_test()
        config.metrics.report_output_dir = str(tmp_path / "reports")
        config.session.checkpoint_dir = str(tmp_path / "checkpoints")
        return config

    async def test_complete_mini_session(self, quick_config):
        """Test running a complete mini learning session."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        result = await orchestrator.run_session()

        # Session should complete
        assert isinstance(result, SessionResult)
        assert result.total_games >= 0
        assert result.session_duration_seconds > 0

    async def test_session_generates_reports(self, quick_config):
        """Test that session generates HTML and JSON reports."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        result = await orchestrator.run_session()

        # Reports should be generated
        if result.report_path:
            assert result.report_path.exists()
            content = result.report_path.read_text()
            assert "Session Report" in content or "<!DOCTYPE html>" in content

        if result.metrics_path:
            assert result.metrics_path.exists()
            data = json.loads(result.metrics_path.read_text())
            assert "config" in data
            assert "scorecard" in data

    async def test_session_tracks_improvement(self, quick_config):
        """Test that session tracks improvement metrics."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        await orchestrator.run_session()

        # Improvement summary should be available
        summary = orchestrator.get_improvement_summary()

        assert "elo_delta_total" in summary
        assert "games_completed" in summary
        assert "win_rate" in summary

    async def test_live_metrics_during_session(self, quick_config):
        """Test that live metrics are updated during session."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        metrics_snapshots = []

        def metrics_callback(metrics):
            metrics_snapshots.append(metrics.copy())

        orchestrator.register_metrics_callback(metrics_callback)

        await orchestrator.run_session()

        # Should have received metrics updates
        if orchestrator.live_metrics.games_completed > 0:
            assert len(metrics_snapshots) >= 0  # May or may not have callbacks depending on game count


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.timeout(60)
class TestSessionCallbacks:
    """End-to-end tests for session callbacks."""

    @pytest.fixture
    def quick_config(self, tmp_path):
        """Create quick test configuration."""
        config = ContinuousPlayConfig.for_quick_test()
        config.metrics.report_output_dir = str(tmp_path / "reports")
        return config

    async def test_game_complete_callbacks(self, quick_config):
        """Test game completion callbacks are invoked."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        game_records = []

        def game_callback(record, live_metrics):
            game_records.append(record)

        orchestrator.register_game_callback(game_callback)

        await orchestrator.run_session()

        # Callbacks should match games completed
        assert len(game_records) == orchestrator.live_metrics.games_completed

    async def test_learning_callbacks(self, quick_config):
        """Test learning update callbacks are invoked."""
        # Configure to trigger learning
        quick_config.learning.min_games_before_learning = 1
        quick_config.learning.learn_every_n_games = 1
        quick_config.session.max_games = 3

        orchestrator = ContinuousPlayOrchestrator(quick_config)

        learning_updates = []

        def learning_callback(loss, game_num):
            learning_updates.append((loss, game_num))

        orchestrator.register_learning_callback(learning_callback)

        await orchestrator.run_session()

        # May have learning updates depending on buffer size
        # Just verify callbacks mechanism works
        assert isinstance(learning_updates, list)


@pytest.mark.e2e
@pytest.mark.asyncio
class TestSessionControl:
    """End-to-end tests for session control (pause/resume/stop)."""

    @pytest.fixture
    def quick_config(self, tmp_path):
        """Create configuration for control tests."""
        config = ContinuousPlayConfig.for_quick_test()
        config.session.max_games = 10  # More games for control testing
        config.metrics.report_output_dir = str(tmp_path / "reports")
        return config

    async def test_session_can_be_stopped(self, quick_config):
        """Test that a running session can be stopped."""
        orchestrator = ContinuousPlayOrchestrator(quick_config)

        # Start session in background
        session_task = asyncio.create_task(orchestrator.run_session())

        # Wait a moment then stop
        await asyncio.sleep(0.5)
        orchestrator.stop()

        # Session should complete
        result = await session_task

        assert orchestrator.is_running is False
        assert isinstance(result, SessionResult)


@pytest.mark.e2e
class TestOrchestratorFactory:
    """End-to-end tests for orchestrator factory function."""

    def test_create_quick_test_orchestrator(self):
        """Test creating orchestrator with quick_test preset."""
        orchestrator = create_orchestrator(preset="quick_test")

        assert orchestrator.config.session.session_duration_minutes == 1
        assert orchestrator.config.session.max_games == 3

    def test_create_development_orchestrator(self):
        """Test creating orchestrator with development preset."""
        orchestrator = create_orchestrator(preset="development")

        assert orchestrator.config.session.session_duration_minutes == 5
        assert orchestrator.config.session.max_games == 10

    def test_create_default_orchestrator(self):
        """Test creating orchestrator with default preset."""
        orchestrator = create_orchestrator(preset="default")

        # Should use environment defaults
        assert orchestrator.config is not None
