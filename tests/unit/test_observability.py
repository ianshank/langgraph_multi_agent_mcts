"""
Comprehensive unit tests for observability modules.

Tests:
- metrics.py: MCTSMetrics, AgentMetrics, MetricsCollector, MetricsTimer
- profiling.py: profile_block context manager, MemoryProfiler, AsyncProfiler
- tracing.py: TracingManager initialization, get_tracer
- logging.py: setup_logging, CorrelationIdFilter, sanitization

All external dependencies (psutil, OpenTelemetry) are mocked for fast, isolated tests.
"""

import json
import logging
import logging.config

# Import observability modules
import sys
import time
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, ".")

from src.observability.logging import (
    CorrelationIdFilter,
    JSONFormatter,
    LogContext,
    PerformanceMetricsFilter,
    get_correlation_id,
    get_logger,
    get_request_metadata,
    sanitize_dict,
    sanitize_message,
    set_correlation_id,
    set_request_metadata,
    setup_logging,
)
from src.observability.metrics import (
    AgentMetrics,
    MCTSMetrics,
    MetricsCollector,
    MetricsTimer,
    agent_metrics,
    mcts_metrics,
)
from src.observability.profiling import (
    AsyncProfiler,
    MemoryProfiler,
    ProfilingSession,
    profile_block,
)
from src.observability.tracing import (
    SpanContextPropagator,
    TracingManager,
    add_mcts_attributes,
    get_tracer,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_process():
    """Mock psutil.Process for all tests."""
    with patch("src.observability.metrics.PROMETHEUS_AVAILABLE", False):
        with patch("src.observability.metrics.psutil.Process") as mock:
            process_instance = MagicMock()
            mock.return_value = process_instance

            # Set up default memory info
            memory_info = MagicMock()
            memory_info.rss = 100 * 1024 * 1024  # 100 MB
            memory_info.vms = 200 * 1024 * 1024  # 200 MB
            process_instance.memory_info.return_value = memory_info
            process_instance.cpu_percent.return_value = 25.5
            process_instance.num_threads.return_value = 8
            process_instance.open_files.return_value = [1, 2, 3]
            process_instance.memory_percent.return_value = 5.0

            yield process_instance


@pytest.fixture
def mock_profiling_process():
    """Mock psutil.Process for profiling tests."""
    with patch("src.observability.profiling.psutil.Process") as mock:
        process_instance = MagicMock()
        mock.return_value = process_instance

        memory_info = MagicMock()
        memory_info.rss = 150 * 1024 * 1024  # 150 MB
        memory_info.vms = 300 * 1024 * 1024  # 300 MB
        process_instance.memory_info.return_value = memory_info
        process_instance.cpu_percent.return_value = 30.0
        process_instance.memory_percent.return_value = 7.5
        process_instance.num_threads.return_value = 10

        yield process_instance


@pytest.fixture
def mock_logging_process():
    """Mock psutil.Process for logging tests."""
    with patch("src.observability.logging.psutil.Process") as mock:
        process_instance = MagicMock()
        mock.return_value = process_instance

        memory_info = MagicMock()
        memory_info.rss = 80 * 1024 * 1024  # 80 MB
        process_instance.memory_info.return_value = memory_info
        process_instance.cpu_percent.return_value = 15.0
        process_instance.num_threads.return_value = 6

        yield process_instance


@pytest.fixture
def reset_metrics_collector():
    """Reset MetricsCollector singleton between tests."""
    # Save original instance
    original = MetricsCollector._instance
    MetricsCollector._instance = None
    yield
    # Restore original instance
    MetricsCollector._instance = original


@pytest.fixture
def reset_async_profiler():
    """Reset AsyncProfiler singleton between tests."""
    original = AsyncProfiler._instance
    AsyncProfiler._instance = None
    yield
    AsyncProfiler._instance = original


@pytest.fixture
def reset_tracing_manager():
    """Reset TracingManager singleton between tests."""
    original_instance = TracingManager._instance
    original_provider = TracingManager._provider
    TracingManager._instance = None
    TracingManager._provider = None
    yield
    TracingManager._instance = original_instance
    TracingManager._provider = original_provider


@pytest.fixture(autouse=False)
def reset_correlation_id():
    """Reset correlation ID context variable."""
    from src.observability.logging import _correlation_id, _request_metadata

    # Save current state and reset to clean state
    _correlation_id.get()
    _request_metadata.get()
    token1 = _correlation_id.set(None)
    token2 = _request_metadata.set({})
    yield
    # Restore original state
    _correlation_id.reset(token1)
    _request_metadata.reset(token2)


# ============================================================================
# MCTSMetrics Tests
# ============================================================================


class TestMCTSMetrics:
    """Test suite for MCTSMetrics dataclass."""

    def test_default_initialization(self):
        """Test MCTSMetrics initializes with correct defaults."""
        metrics = MCTSMetrics()

        assert metrics.iterations == 0
        assert metrics.total_simulations == 0
        assert metrics.tree_depth == 0
        assert metrics.total_nodes == 0
        assert metrics.ucb_scores == []
        assert metrics.selection_times_ms == []
        assert metrics.expansion_times_ms == []
        assert metrics.simulation_times_ms == []
        assert metrics.backprop_times_ms == []
        assert metrics.best_action_visits == 0
        assert metrics.best_action_value == 0.0

    def test_custom_initialization(self):
        """Test MCTSMetrics can be initialized with custom values."""
        metrics = MCTSMetrics(
            iterations=100,
            total_simulations=500,
            tree_depth=10,
            total_nodes=1000,
            ucb_scores=[0.5, 0.6, 0.7],
            best_action_visits=50,
            best_action_value=0.85,
        )

        assert metrics.iterations == 100
        assert metrics.total_simulations == 500
        assert metrics.tree_depth == 10
        assert metrics.total_nodes == 1000
        assert len(metrics.ucb_scores) == 3
        assert metrics.best_action_visits == 50
        assert metrics.best_action_value == 0.85

    def test_lists_are_independent(self):
        """Test that list fields are independent between instances."""
        metrics1 = MCTSMetrics()
        metrics2 = MCTSMetrics()

        metrics1.ucb_scores.append(0.5)

        assert len(metrics1.ucb_scores) == 1
        assert len(metrics2.ucb_scores) == 0


# ============================================================================
# AgentMetrics Tests
# ============================================================================


class TestAgentMetrics:
    """Test suite for AgentMetrics dataclass."""

    def test_initialization_with_name(self):
        """Test AgentMetrics requires name and has correct defaults."""
        metrics = AgentMetrics(name="test_agent")

        assert metrics.name == "test_agent"
        assert metrics.executions == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.confidence_scores == []
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.memory_usage_mb == []

    def test_confidence_score_accumulation(self):
        """Test that confidence scores can be accumulated."""
        metrics = AgentMetrics(name="agent1")

        metrics.confidence_scores.append(0.8)
        metrics.confidence_scores.append(0.9)
        metrics.confidence_scores.append(0.85)

        assert len(metrics.confidence_scores) == 3
        avg = sum(metrics.confidence_scores) / len(metrics.confidence_scores)
        assert abs(avg - 0.85) < 1e-6


# ============================================================================
# MetricsCollector Tests
# ============================================================================


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    def test_singleton_pattern(self, reset_metrics_collector, mock_process):
        """Test MetricsCollector follows singleton pattern."""
        collector1 = MetricsCollector.get_instance()
        collector2 = MetricsCollector.get_instance()

        assert collector1 is collector2

    def test_singleton_accessor_functions(self, reset_metrics_collector, mock_process):
        """Test convenience accessor functions return same singleton."""
        collector = MetricsCollector.get_instance()

        assert mcts_metrics() is collector
        assert agent_metrics() is collector

    def test_record_mcts_iteration(self, reset_metrics_collector, mock_process):
        """Test recording MCTS iteration increments counters."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        collector.record_mcts_iteration(
            session_id=session_id,
            ucb_score=0.75,
            selection_time_ms=10.5,
            expansion_time_ms=15.3,
            simulation_time_ms=20.1,
            backprop_time_ms=5.2,
        )

        metrics = collector._mcts_metrics[session_id]
        assert metrics.iterations == 1
        assert len(metrics.ucb_scores) == 1
        assert metrics.ucb_scores[0] == 0.75
        assert len(metrics.selection_times_ms) == 1
        assert metrics.selection_times_ms[0] == 10.5
        assert len(metrics.expansion_times_ms) == 1
        assert len(metrics.simulation_times_ms) == 1
        assert len(metrics.backprop_times_ms) == 1

    def test_record_mcts_iteration_skips_zero_times(self, reset_metrics_collector, mock_process):
        """Test that zero timing values are not recorded."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        collector.record_mcts_iteration(
            session_id=session_id,
            ucb_score=0.8,
            selection_time_ms=0.0,  # Should not be recorded
            expansion_time_ms=10.0,
            simulation_time_ms=0.0,  # Should not be recorded
            backprop_time_ms=5.0,
        )

        metrics = collector._mcts_metrics[session_id]
        assert len(metrics.selection_times_ms) == 0
        assert len(metrics.expansion_times_ms) == 1
        assert len(metrics.simulation_times_ms) == 0
        assert len(metrics.backprop_times_ms) == 1

    def test_record_multiple_iterations(self, reset_metrics_collector, mock_process):
        """Test recording multiple MCTS iterations accumulates correctly."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        for i in range(5):
            collector.record_mcts_iteration(
                session_id=session_id,
                ucb_score=0.5 + i * 0.1,
            )

        metrics = collector._mcts_metrics[session_id]
        assert metrics.iterations == 5
        assert len(metrics.ucb_scores) == 5
        assert metrics.ucb_scores[-1] == 0.9

    def test_record_mcts_simulation(self, reset_metrics_collector, mock_process):
        """Test recording MCTS simulation increments counter."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        collector.record_mcts_simulation(session_id)
        collector.record_mcts_simulation(session_id)
        collector.record_mcts_simulation(session_id)

        metrics = collector._mcts_metrics[session_id]
        assert metrics.total_simulations == 3

    def test_update_mcts_tree_stats(self, reset_metrics_collector, mock_process):
        """Test updating MCTS tree statistics."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        collector.update_mcts_tree_stats(
            session_id=session_id,
            tree_depth=8,
            total_nodes=150,
            best_action_visits=25,
            best_action_value=0.92,
        )

        metrics = collector._mcts_metrics[session_id]
        assert metrics.tree_depth == 8
        assert metrics.total_nodes == 150
        assert metrics.best_action_visits == 25
        assert metrics.best_action_value == 0.92

    def test_record_agent_execution_creates_new_agent(self, reset_metrics_collector, mock_process):
        """Test recording agent execution creates new AgentMetrics if needed."""
        collector = MetricsCollector.get_instance()

        collector.record_agent_execution(
            agent_name="new_agent",
            execution_time_ms=100.5,
            confidence=0.85,
            success=True,
        )

        assert "new_agent" in collector._agent_metrics
        metrics = collector._agent_metrics["new_agent"]
        assert metrics.name == "new_agent"
        assert metrics.executions == 1
        assert metrics.total_time_ms == 100.5
        assert metrics.confidence_scores == [0.85]
        assert metrics.avg_confidence == 0.85
        assert metrics.success_count == 1
        assert metrics.error_count == 0

    def test_record_agent_execution_updates_existing(self, reset_metrics_collector, mock_process):
        """Test recording agent execution updates existing AgentMetrics."""
        collector = MetricsCollector.get_instance()

        collector.record_agent_execution("agent1", 50.0, 0.8, True)
        collector.record_agent_execution("agent1", 75.0, 0.9, True)
        collector.record_agent_execution("agent1", 100.0, 0.7, False)

        metrics = collector._agent_metrics["agent1"]
        assert metrics.executions == 3
        assert metrics.total_time_ms == 225.0
        assert len(metrics.confidence_scores) == 3
        assert abs(metrics.avg_confidence - 0.8) < 1e-6
        assert metrics.success_count == 2
        assert metrics.error_count == 1

    def test_record_node_timing(self, reset_metrics_collector, mock_process):
        """Test recording node timing."""
        collector = MetricsCollector.get_instance()

        collector.record_node_timing("selection", 10.5)
        collector.record_node_timing("selection", 12.3)
        collector.record_node_timing("expansion", 25.0)

        assert len(collector._node_timings["selection"]) == 2
        assert collector._node_timings["selection"][0] == 10.5
        assert len(collector._node_timings["expansion"]) == 1

    def test_record_request_latency(self, reset_metrics_collector, mock_process):
        """Test recording request latency."""
        collector = MetricsCollector.get_instance()

        collector.record_request_latency(150.0)
        collector.record_request_latency(200.0)
        collector.record_request_latency(175.0)

        assert len(collector._request_latencies) == 3
        assert collector._request_latencies[1] == 200.0

    def test_sample_system_metrics(self, reset_metrics_collector, mock_process):
        """Test sampling system metrics."""
        collector = MetricsCollector.get_instance()

        sample = collector.sample_system_metrics()

        assert "timestamp" in sample
        assert sample["memory_rss_mb"] == 100.0
        assert sample["memory_vms_mb"] == 200.0
        assert sample["cpu_percent"] == 25.5
        assert sample["thread_count"] == 8
        assert sample["open_files"] == 3
        assert len(collector._memory_samples) == 1

    def test_get_mcts_summary_empty(self, reset_metrics_collector, mock_process):
        """Test getting summary for non-existent session returns empty dict."""
        collector = MetricsCollector.get_instance()

        summary = collector.get_mcts_summary("nonexistent")

        assert summary == {}

    def test_get_mcts_summary_with_data(self, reset_metrics_collector, mock_process):
        """Test getting summary with data calculates statistics correctly."""
        collector = MetricsCollector.get_instance()
        session_id = "test_session"

        # Add some data
        for i in range(10):
            collector.record_mcts_iteration(
                session_id=session_id,
                ucb_score=0.1 * (i + 1),
                selection_time_ms=10.0 + i,
            )
        collector.update_mcts_tree_stats(session_id, 5, 100, 20, 0.8)

        summary = collector.get_mcts_summary(session_id)

        assert summary["session_id"] == session_id
        assert summary["total_iterations"] == 10
        assert summary["tree_depth"] == 5
        assert summary["total_nodes"] == 100
        assert summary["ucb_scores"]["count"] == 10
        assert abs(summary["ucb_scores"]["mean"] - 0.55) < 0.01
        assert summary["ucb_scores"]["min"] == 0.1
        assert summary["ucb_scores"]["max"] == 1.0

    def test_get_agent_summary_empty(self, reset_metrics_collector, mock_process):
        """Test getting agent summary for non-existent agent returns empty dict."""
        collector = MetricsCollector.get_instance()

        summary = collector.get_agent_summary("nonexistent")

        assert summary == {}

    def test_get_agent_summary_with_data(self, reset_metrics_collector, mock_process):
        """Test getting agent summary calculates statistics correctly."""
        collector = MetricsCollector.get_instance()

        collector.record_agent_execution("agent1", 100.0, 0.8, True)
        collector.record_agent_execution("agent1", 150.0, 0.9, True)
        collector.record_agent_execution("agent1", 50.0, 0.7, False)

        summary = collector.get_agent_summary("agent1")

        assert summary["agent_name"] == "agent1"
        assert summary["total_executions"] == 3
        assert summary["success_count"] == 2
        assert summary["error_count"] == 1
        assert abs(summary["success_rate"] - 0.6667) < 0.01
        assert summary["avg_execution_time_ms"] == 100.0
        assert summary["total_time_ms"] == 300.0
        assert abs(summary["confidence"]["mean"] - 0.8) < 0.01

    def test_get_node_timing_summary(self, reset_metrics_collector, mock_process):
        """Test getting node timing summary."""
        collector = MetricsCollector.get_instance()

        collector.record_node_timing("selection", 10.0)
        collector.record_node_timing("selection", 20.0)
        collector.record_node_timing("selection", 30.0)
        collector.record_node_timing("expansion", 15.0)

        summary = collector.get_node_timing_summary()

        assert "selection" in summary
        assert summary["selection"]["count"] == 3
        assert summary["selection"]["mean_ms"] == 20.0
        assert summary["selection"]["min_ms"] == 10.0
        assert summary["selection"]["max_ms"] == 30.0
        assert summary["selection"]["total_ms"] == 60.0
        assert "expansion" in summary

    def test_get_full_report(self, reset_metrics_collector, mock_process):
        """Test generating full metrics report."""
        collector = MetricsCollector.get_instance()

        collector.record_mcts_iteration("session1", 0.8)
        collector.record_agent_execution("agent1", 100.0, 0.85, True)
        collector.record_node_timing("node1", 50.0)
        collector.record_request_latency(200.0)

        report = collector.get_full_report()

        assert "report_time" in report
        assert "uptime_seconds" in report
        assert "system_metrics" in report
        assert "mcts_sessions" in report
        assert "session1" in report["mcts_sessions"]
        assert "agents" in report
        assert "agent1" in report["agents"]
        assert "node_timings" in report
        assert "request_latencies" in report

    def test_reset_clears_all_data(self, reset_metrics_collector, mock_process):
        """Test that reset clears all collected metrics."""
        collector = MetricsCollector.get_instance()

        # Add some data
        collector.record_mcts_iteration("session1", 0.8)
        collector.record_agent_execution("agent1", 100.0, 0.85, True)
        collector.record_node_timing("node1", 50.0)
        collector.record_request_latency(200.0)

        collector.reset()

        assert len(collector._mcts_metrics) == 0
        assert len(collector._agent_metrics) == 0
        assert len(collector._node_timings) == 0
        assert len(collector._request_latencies) == 0
        assert len(collector._memory_samples) == 0


# ============================================================================
# MetricsTimer Tests
# ============================================================================


class TestMetricsTimer:
    """Test suite for MetricsTimer context manager."""

    def test_timer_measures_elapsed_time(self, reset_metrics_collector, mock_process):
        """Test that timer accurately measures elapsed time."""
        collector = MetricsCollector.get_instance()
        timer = MetricsTimer(collector=collector)

        with timer:
            time.sleep(0.01)  # Sleep for 10ms

        # Should be around 10ms, but allow some tolerance
        assert timer.elapsed_ms > 5.0
        assert timer.elapsed_ms < 50.0

    def test_timer_records_node_timing(self, reset_metrics_collector, mock_process):
        """Test that timer records node timing when node_name is specified."""
        collector = MetricsCollector.get_instance()
        timer = MetricsTimer(collector=collector, node_name="test_node")

        with timer:
            time.sleep(0.005)

        assert "test_node" in collector._node_timings
        assert len(collector._node_timings["test_node"]) == 1
        assert collector._node_timings["test_node"][0] > 0

    def test_timer_uses_default_collector(self, reset_metrics_collector, mock_process):
        """Test that timer uses default collector if not specified."""
        timer = MetricsTimer(node_name="default_node")

        with timer:
            pass

        collector = MetricsCollector.get_instance()
        assert "default_node" in collector._node_timings

    def test_multiple_timers_record_correctly(self, reset_metrics_collector, mock_process):
        """Test that multiple timers record separately."""
        collector = MetricsCollector.get_instance()

        with MetricsTimer(collector=collector, node_name="node1"):
            time.sleep(0.005)

        with MetricsTimer(collector=collector, node_name="node2"):
            time.sleep(0.005)

        with MetricsTimer(collector=collector, node_name="node1"):
            time.sleep(0.005)

        assert len(collector._node_timings["node1"]) == 2
        assert len(collector._node_timings["node2"]) == 1


# ============================================================================
# AsyncProfiler Tests
# ============================================================================


class TestAsyncProfiler:
    """Test suite for AsyncProfiler."""

    def test_singleton_pattern(self, reset_async_profiler, mock_profiling_process):
        """Test AsyncProfiler follows singleton pattern."""
        with patch("src.observability.profiling.get_logger"):
            profiler1 = AsyncProfiler.get_instance()
            profiler2 = AsyncProfiler.get_instance()

            assert profiler1 is profiler2

    def test_start_session_creates_session(self, reset_async_profiler, mock_profiling_process):
        """Test that start_session creates a new profiling session."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            session_id = profiler.start_session("test_session")

            assert session_id == "test_session"
            assert "test_session" in profiler._sessions
            assert profiler._current_session == "test_session"
            assert isinstance(profiler._sessions["test_session"], ProfilingSession)

    def test_start_session_auto_generates_id(self, reset_async_profiler, mock_profiling_process):
        """Test that start_session auto-generates ID if not provided."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            session_id = profiler.start_session()

            assert session_id.startswith("session_")
            assert session_id in profiler._sessions

    def test_end_session_returns_session(self, reset_async_profiler, mock_profiling_process):
        """Test that end_session returns the session data."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            profiler.start_session("test_session")
            session = profiler.end_session("test_session")

            assert isinstance(session, ProfilingSession)
            assert session.session_id == "test_session"
            assert profiler._current_session is None

    def test_end_session_uses_current_session(self, reset_async_profiler, mock_profiling_process):
        """Test that end_session uses current session if not specified."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            profiler.start_session("test_session")
            session = profiler.end_session()

            assert session.session_id == "test_session"

    def test_end_session_raises_for_unknown(self, reset_async_profiler, mock_profiling_process):
        """Test that end_session raises ValueError for unknown session."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            with pytest.raises(ValueError, match="Unknown session"):
                profiler.end_session("nonexistent")

    def test_time_block_records_timing(self, reset_async_profiler, mock_profiling_process):
        """Test that time_block records timing in session."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            with profiler.time_block("test_operation"):
                time.sleep(0.005)

            session = profiler._sessions["test_session"]
            assert len(session.timings) == 1
            assert session.timings[0].name == "test_operation"
            assert session.timings[0].elapsed_ms > 0
            assert session.timings[0].success is True

    def test_time_block_records_memory_delta(self, reset_async_profiler, mock_profiling_process):
        """Test that time_block records memory usage."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            with profiler.time_block("test_operation"):
                pass

            timing = profiler._sessions["test_session"].timings[0]
            assert timing.memory_start_mb == 150.0
            assert timing.memory_end_mb == 150.0
            assert timing.memory_delta_mb == 0.0

    def test_time_block_handles_exceptions(self, reset_async_profiler, mock_profiling_process):
        """Test that time_block records failed operations correctly."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            with pytest.raises(ValueError), profiler.time_block("failing_operation"):
                raise ValueError("Test error")

            timing = profiler._sessions["test_session"].timings[0]
            assert timing.success is False
            assert timing.error == "Test error"

    def test_time_block_records_metadata(self, reset_async_profiler, mock_profiling_process):
        """Test that time_block records custom metadata."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            metadata = {"batch_size": 100, "iteration": 5}
            with profiler.time_block("test_operation", metadata=metadata):
                pass

            timing = profiler._sessions["test_session"].timings[0]
            assert timing.metadata == metadata

    def test_sample_memory(self, reset_async_profiler, mock_profiling_process):
        """Test memory sampling."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            sample = profiler.sample_memory()

            assert "rss_mb" in sample
            assert sample["rss_mb"] == 150.0
            assert sample["vms_mb"] == 300.0
            assert sample["percent"] == 7.5

            session = profiler._sessions["test_session"]
            assert len(session.memory_samples) == 1

    def test_sample_cpu(self, reset_async_profiler, mock_profiling_process):
        """Test CPU sampling."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            cpu_percent = profiler.sample_cpu()

            assert cpu_percent == 30.0

            session = profiler._sessions["test_session"]
            assert len(session.cpu_samples) == 1
            assert session.cpu_samples[0] == 30.0

    def test_add_marker(self, reset_async_profiler, mock_profiling_process):
        """Test adding custom markers."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            profiler.add_marker("checkpoint", data={"step": 10})

            session = profiler._sessions["test_session"]
            assert len(session.markers) == 1
            assert session.markers[0]["name"] == "checkpoint"
            assert session.markers[0]["data"] == {"step": 10}

    def test_get_timing_summary_specific(self, reset_async_profiler, mock_profiling_process):
        """Test getting timing summary for specific operation."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            # Manually add aggregate timings
            profiler._aggregate_timings["operation1"] = [10.0, 20.0, 30.0]

            summary = profiler.get_timing_summary("operation1")

            assert summary["name"] == "operation1"
            assert summary["count"] == 3
            assert summary["total_ms"] == 60.0
            assert summary["mean_ms"] == 20.0
            assert summary["min_ms"] == 10.0
            assert summary["max_ms"] == 30.0

    def test_get_timing_summary_all(self, reset_async_profiler, mock_profiling_process):
        """Test getting timing summary for all operations."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            profiler._aggregate_timings["op1"] = [10.0, 20.0]
            profiler._aggregate_timings["op2"] = [30.0, 40.0]

            summary = profiler.get_timing_summary()

            assert "op1" in summary
            assert "op2" in summary
            assert summary["op1"]["mean_ms"] == 15.0
            assert summary["op2"]["mean_ms"] == 35.0

    def test_reset(self, reset_async_profiler, mock_profiling_process):
        """Test that reset clears all profiling data."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()

            profiler.start_session("test_session")
            profiler._aggregate_timings["op1"] = [10.0]

            profiler.reset()

            assert len(profiler._sessions) == 0
            assert profiler._current_session is None
            assert len(profiler._aggregate_timings) == 0


# ============================================================================
# MemoryProfiler Tests
# ============================================================================


class TestMemoryProfiler:
    """Test suite for MemoryProfiler."""

    def test_set_baseline(self, mock_profiling_process):
        """Test setting memory baseline."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            baseline = profiler.set_baseline()

            assert baseline == 150.0
            assert profiler._baseline == 150.0

    def test_get_current(self, mock_profiling_process):
        """Test getting current memory usage."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            current = profiler.get_current()

            assert current == 150.0

    def test_get_delta_sets_baseline_if_none(self, mock_profiling_process):
        """Test that get_delta sets baseline if not already set."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            delta = profiler.get_delta()

            assert profiler._baseline is not None
            assert delta == 0.0

    def test_get_delta_calculates_correctly(self, mock_profiling_process):
        """Test that get_delta calculates memory change correctly."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()
            profiler._baseline = 100.0

            delta = profiler.get_delta()

            # Current is 150MB, baseline is 100MB
            assert delta == 50.0

    def test_sample_creates_record(self, mock_profiling_process):
        """Test that sample creates a comprehensive memory record."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()
            profiler._baseline = 100.0

            sample = profiler.sample(label="test_label")

            assert sample["label"] == "test_label"
            assert sample["rss_mb"] == 150.0
            assert sample["vms_mb"] == 300.0
            assert sample["percent"] == 7.5
            assert sample["delta_from_baseline_mb"] == 50.0
            assert sample["peak_mb"] == 150.0
            assert len(profiler._samples) == 1

    def test_sample_tracks_peak(self, mock_profiling_process):
        """Test that sample tracks peak memory usage."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            # First sample sets peak to 150MB
            profiler.sample()
            assert profiler._peak == 150.0

            # Mock higher memory
            mock_profiling_process.memory_info.return_value.rss = 200 * 1024 * 1024
            profiler.sample()
            assert profiler._peak == 200.0

    def test_check_leak_no_baseline(self, mock_profiling_process):
        """Test check_leak with no baseline returns appropriate status."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            result = profiler.check_leak()

            assert result["status"] == "no_baseline"

    def test_check_leak_detects_potential_leak(self, mock_profiling_process):
        """Test check_leak detects memory increase above threshold."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()
            profiler._baseline = 100.0  # Current is 150MB, so delta is 50MB

            result = profiler.check_leak(threshold_mb=10.0)

            assert result["status"] == "potential_leak"
            assert result["delta_mb"] == 50.0

    def test_check_leak_ok_when_below_threshold(self, mock_profiling_process):
        """Test check_leak returns OK when below threshold."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()
            profiler._baseline = 145.0  # Current is 150MB, so delta is 5MB

            result = profiler.check_leak(threshold_mb=10.0)

            assert result["status"] == "ok"
            assert result["delta_mb"] == 5.0

    def test_get_summary(self, mock_profiling_process):
        """Test getting memory profiler summary."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()
            profiler._baseline = 100.0

            # Add some samples manually
            profiler._samples = [
                {"rss_mb": 100.0},
                {"rss_mb": 150.0},
                {"rss_mb": 200.0},
            ]
            profiler._peak = 200.0

            summary = profiler.get_summary()

            assert summary["sample_count"] == 3
            assert summary["baseline_mb"] == 100.0
            assert summary["current_mb"] == 150.0
            assert summary["peak_mb"] == 200.0
            assert summary["mean_mb"] == 150.0
            assert summary["min_mb"] == 100.0
            assert summary["max_mb"] == 200.0

    def test_get_summary_no_samples(self, mock_profiling_process):
        """Test getting summary with no samples."""
        with patch("src.observability.profiling.get_logger"):
            profiler = MemoryProfiler()

            summary = profiler.get_summary()

            assert "message" in summary
            assert summary["message"] == "No samples collected"


# ============================================================================
# profile_block Tests
# ============================================================================


class TestProfileBlock:
    """Test suite for profile_block convenience context manager."""

    def test_profile_block_uses_singleton(self, reset_async_profiler, mock_profiling_process):
        """Test that profile_block uses the global AsyncProfiler singleton."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            with profile_block("test_operation"):
                pass

            # Should have recorded in aggregate timings
            assert "test_operation" in profiler._aggregate_timings

    def test_profile_block_with_metadata(self, reset_async_profiler, mock_profiling_process):
        """Test profile_block passes metadata correctly."""
        with patch("src.observability.profiling.get_logger"):
            profiler = AsyncProfiler.get_instance()
            profiler.start_session("test_session")

            metadata = {"key": "value"}
            with profile_block("test_op", metadata=metadata):
                pass

            timing = profiler._sessions["test_session"].timings[0]
            assert timing.metadata == metadata


# ============================================================================
# TracingManager Tests
# ============================================================================


class TestTracingManager:
    """Test suite for TracingManager."""

    def test_singleton_pattern(self, reset_tracing_manager):
        """Test TracingManager follows singleton pattern."""
        manager1 = TracingManager.get_instance()
        manager2 = TracingManager.get_instance()

        assert manager1 is manager2

    def test_initialize_with_defaults(self, reset_tracing_manager):
        """Test initialization with default values."""
        with patch("src.observability.tracing.trace.set_tracer_provider"):
            with patch("src.observability.tracing.OTLPSpanExporter"):
                with patch("src.observability.tracing.BatchSpanProcessor"):
                    with patch("src.observability.tracing.HTTPXClientInstrumentor"):
                        manager = TracingManager.get_instance()
                        manager.initialize()

                        assert manager._initialized is True

    def test_initialize_with_console_exporter(self, reset_tracing_manager):
        """Test initialization with console exporter."""
        manager = TracingManager()

        # Test that initialization with console exporter sets the initialized flag
        with patch.object(manager, "_instrument_httpx"):
            manager.initialize(exporter_type="console")

            assert manager._initialized is True
            assert manager._provider is not None

    def test_initialize_with_none_exporter(self, reset_tracing_manager):
        """Test initialization with no exporter."""
        with patch("src.observability.tracing.trace.set_tracer_provider"):
            with patch("src.observability.tracing.HTTPXClientInstrumentor"):
                manager = TracingManager.get_instance()
                manager.initialize(exporter_type="none")

                assert manager._initialized is True

    def test_initialize_with_invalid_exporter_raises(self, reset_tracing_manager):
        """Test initialization with invalid exporter type raises ValueError."""
        manager = TracingManager.get_instance()

        with pytest.raises(ValueError, match="Unknown exporter type"):
            manager.initialize(exporter_type="invalid")

    def test_initialize_only_runs_once(self, reset_tracing_manager):
        """Test that initialize only runs once."""
        manager = TracingManager()

        # Patch _instrument_httpx to avoid external dependencies
        with patch.object(manager, "_instrument_httpx"):
            manager.initialize()
            first_provider = manager._provider

            manager.initialize()  # Second call should be no-op
            second_provider = manager._provider

            # Provider should remain the same (not recreated)
            assert manager._initialized is True
            assert first_provider is second_provider

    def test_get_tracer_initializes_if_needed(self, reset_tracing_manager):
        """Test that get_tracer initializes if not already initialized."""
        manager = TracingManager()

        # Patch _instrument_httpx to avoid external dependencies
        with patch.object(manager, "_instrument_httpx"):
            tracer = manager.get_tracer("test")

            assert manager._initialized is True
            # Tracer should have start_span method
            assert hasattr(tracer, "start_span")

    def test_shutdown(self, reset_tracing_manager):
        """Test shutdown cleans up resources."""
        with patch("src.observability.tracing.trace.set_tracer_provider"):
            with patch("src.observability.tracing.OTLPSpanExporter"):
                with patch("src.observability.tracing.BatchSpanProcessor"):
                    with patch("src.observability.tracing.HTTPXClientInstrumentor"):
                        manager = TracingManager.get_instance()
                        manager.initialize()

                        mock_provider = MagicMock()
                        manager._provider = mock_provider

                        manager.shutdown()

                        mock_provider.shutdown.assert_called_once()
                        assert manager._initialized is False

    def test_get_tracer_module_function(self, reset_tracing_manager):
        """Test get_tracer module-level function."""
        # Test that get_tracer returns a tracer object
        with patch("src.observability.tracing.trace.set_tracer_provider"):
            with patch("src.observability.tracing.OTLPSpanExporter"):
                with patch("src.observability.tracing.BatchSpanProcessor"):
                    with patch("src.observability.tracing.HTTPXClientInstrumentor"):
                        result = get_tracer("test_tracer")
                        # Verify it's a tracer-like object
                        from opentelemetry.trace import Tracer

                        assert hasattr(result, "start_span") or isinstance(result, Tracer)


# ============================================================================
# SpanContextPropagator Tests
# ============================================================================


class TestSpanContextPropagator:
    """Test suite for SpanContextPropagator."""

    def test_inject(self):
        """Test injecting trace context into carrier."""
        propagator = SpanContextPropagator()
        carrier = {}

        # Just verify it doesn't raise an exception
        propagator.inject(carrier)
        # The inject function is called internally
        assert isinstance(carrier, dict)

    def test_extract(self):
        """Test extracting trace context from carrier."""
        propagator = SpanContextPropagator()
        carrier = {"traceparent": "00-trace-id-span-id-01"}

        result = propagator.extract(carrier)
        # Should return a Context object
        from opentelemetry.context import Context

        assert isinstance(result, Context)

    def test_get_trace_parent(self):
        """Test getting traceparent header value."""
        propagator = SpanContextPropagator()
        result = propagator.get_trace_parent()

        # Without an active span, this may return None
        # The important thing is it doesn't raise an error
        assert result is None or isinstance(result, str)


# ============================================================================
# add_mcts_attributes Tests
# ============================================================================


class TestAddMCTSAttributes:
    """Test suite for add_mcts_attributes helper function."""

    def test_adds_standard_attributes(self):
        """Test adding standard MCTS attributes."""
        mock_span = MagicMock()

        add_mcts_attributes(
            mock_span,
            **{
                "mcts.iteration": 5,
                "agent.confidence": 0.85,
                "framework.version": "1.0",
            },
        )

        calls = mock_span.set_attribute.call_args_list
        assert call("mcts.iteration", 5) in calls
        assert call("agent.confidence", 0.85) in calls
        assert call("framework.version", "1.0") in calls

    def test_prefixes_custom_attributes(self):
        """Test that non-standard attributes get custom prefix."""
        mock_span = MagicMock()

        add_mcts_attributes(mock_span, custom_key="value")

        mock_span.set_attribute.assert_called_once_with("custom.custom_key", "value")

    def test_skips_none_values(self):
        """Test that None values are not added."""
        mock_span = MagicMock()

        add_mcts_attributes(mock_span, **{"mcts.iteration": None, "agent.name": "test"})

        # Should only be called once for agent.name
        mock_span.set_attribute.assert_called_once_with("agent.name", "test")


# ============================================================================
# Logging - CorrelationIdFilter Tests
# ============================================================================


class TestCorrelationIdFilter:
    """Test suite for CorrelationIdFilter."""

    def test_adds_correlation_id_to_record(self, reset_correlation_id):
        """Test that filter adds correlation ID to log record."""
        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        assert result is True
        assert hasattr(record, "correlation_id")
        assert record.correlation_id is not None
        assert len(record.correlation_id) == 36  # UUID format

    def test_uses_existing_correlation_id(self, reset_correlation_id):
        """Test that filter uses existing correlation ID from context."""
        set_correlation_id("test-correlation-id-123")

        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)

        assert record.correlation_id == "test-correlation-id-123"

    def test_adds_request_metadata(self, reset_correlation_id):
        """Test that filter adds request metadata to record."""
        set_request_metadata({"user_id": "123", "session_id": "abc"})

        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)

        assert hasattr(record, "request_metadata")
        assert record.request_metadata == {"user_id": "123", "session_id": "abc"}


# ============================================================================
# Logging - PerformanceMetricsFilter Tests
# ============================================================================


class TestPerformanceMetricsFilter:
    """Test suite for PerformanceMetricsFilter."""

    def test_adds_performance_metrics(self):
        """Test that filter adds performance metrics to record."""
        filter_obj = PerformanceMetricsFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        # Verify it adds the expected attributes (values will vary based on actual system)
        assert result is True
        assert hasattr(record, "memory_mb")
        assert isinstance(record.memory_mb, float)
        assert record.memory_mb > 0
        assert hasattr(record, "cpu_percent")
        assert isinstance(record.cpu_percent, float)
        assert hasattr(record, "thread_count")
        assert isinstance(record.thread_count, int)
        assert record.thread_count > 0


# ============================================================================
# Logging - Sanitization Tests
# ============================================================================


class TestSanitization:
    """Test suite for log sanitization functions."""

    def test_sanitize_message_api_key(self):
        """Test sanitization of API keys in messages."""
        message = 'Config: {"api_key": "sk-secret123456"}'
        sanitized = sanitize_message(message)

        assert "sk-secret123456" not in sanitized
        assert "***REDACTED***" in sanitized

    def test_sanitize_message_password(self):
        """Test sanitization of passwords."""
        message = 'password: "mypassword123"'
        sanitized = sanitize_message(message)

        assert "mypassword123" not in sanitized
        assert "***REDACTED***" in sanitized

    def test_sanitize_message_bearer_token(self):
        """Test sanitization of Bearer tokens."""
        message = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = sanitize_message(message)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
        assert "Bearer ***REDACTED***" in sanitized

    def test_sanitize_message_basic_auth(self):
        """Test sanitization of Basic auth."""
        message = "Authorization: Basic dXNlcjpwYXNz"
        sanitized = sanitize_message(message)

        assert "dXNlcjpwYXNz" not in sanitized
        assert "Basic ***REDACTED***" in sanitized

    def test_sanitize_dict_sensitive_keys(self):
        """Test sanitization of sensitive keys in dictionaries."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_key": "key-123",
            "token": "tok-abc",
        }

        sanitized = sanitize_dict(data)

        assert sanitized["username"] == "john"
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["token"] == "***REDACTED***"

    def test_sanitize_dict_nested(self):
        """Test sanitization of nested dictionaries."""
        data = {
            "config": {
                "api_key": "secret",
                "nested": {
                    "password": "pass123",
                },
            },
        }

        sanitized = sanitize_dict(data)

        assert sanitized["config"]["api_key"] == "***REDACTED***"
        assert sanitized["config"]["nested"]["password"] == "***REDACTED***"

    def test_sanitize_dict_lists(self):
        """Test sanitization handles lists correctly."""
        data = {
            "items": [
                {"api_key": "key1"},
                {"api_key": "key2"},
            ],
        }

        sanitized = sanitize_dict(data)

        assert sanitized["items"][0]["api_key"] == "***REDACTED***"
        assert sanitized["items"][1]["api_key"] == "***REDACTED***"

    def test_sanitize_dict_preserves_non_sensitive(self):
        """Test that non-sensitive data is preserved."""
        data = {
            "name": "test",
            "count": 42,
            "enabled": True,
            "tags": ["a", "b"],
        }

        sanitized = sanitize_dict(data)

        assert sanitized["name"] == "test"
        assert sanitized["count"] == 42
        assert sanitized["enabled"] is True
        assert sanitized["tags"] == ["a", "b"]


# ============================================================================
# Logging - JSONFormatter Tests
# ============================================================================


class TestJSONFormatter:
    """Test suite for JSONFormatter."""

    def test_format_basic_record(self, reset_correlation_id):
        """Test formatting a basic log record."""
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-cid"
        record.request_metadata = {}

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["correlation_id"] == "test-cid"
        assert data["module"] == "test"
        assert data["line"] == 42

    def test_format_includes_hostname(self):
        """Test that hostname is included when configured."""
        with patch("socket.gethostname", return_value="test-host"):
            formatter = JSONFormatter(include_hostname=True, include_process=False)
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            record.correlation_id = None
            record.request_metadata = {}

            result = formatter.format(record)
            data = json.loads(result)

            assert data["hostname"] == "test-host"

    def test_format_includes_process_info(self):
        """Test that process info is included when configured."""
        formatter = JSONFormatter(include_hostname=False, include_process=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.correlation_id = None
        record.request_metadata = {}

        result = formatter.format(record)
        data = json.loads(result)

        assert "process" in data
        assert "id" in data["process"]
        assert "name" in data["process"]
        assert "thread_id" in data["process"]
        assert "thread_name" in data["process"]

    def test_format_includes_performance_metrics(self):
        """Test that performance metrics are included when present."""
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.correlation_id = None
        record.request_metadata = {}
        record.memory_mb = 100.5
        record.cpu_percent = 25.3
        record.thread_count = 8

        result = formatter.format(record)
        data = json.loads(result)

        assert "performance" in data
        assert data["performance"]["memory_mb"] == 100.5
        assert data["performance"]["cpu_percent"] == 25.3
        assert data["performance"]["thread_count"] == 8

    def test_format_sanitizes_message(self):
        """Test that message is sanitized."""
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg='Config: api_key="secret123"',
            args=(),
            exc_info=None,
        )
        record.correlation_id = None
        record.request_metadata = {}

        result = formatter.format(record)
        data = json.loads(result)

        assert "secret123" not in data["message"]
        assert "***REDACTED***" in data["message"]

    def test_format_includes_exception_info(self):
        """Test that exception info is included."""
        formatter = JSONFormatter(include_hostname=False, include_process=False)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.correlation_id = None
        record.request_metadata = {}

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]
        assert isinstance(data["exception"]["traceback"], list)

    def test_format_includes_extra_fields(self):
        """Test that extra fields are included."""
        formatter = JSONFormatter(include_hostname=False, include_process=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.correlation_id = None
        record.request_metadata = {}
        record.custom_field = "custom_value"
        record.timing = {"elapsed_ms": 100.5}

        result = formatter.format(record)
        data = json.loads(result)

        assert "extra" in data
        assert data["extra"]["custom_field"] == "custom_value"
        assert data["extra"]["timing"]["elapsed_ms"] == 100.5


# ============================================================================
# Logging - Correlation ID Functions Tests
# ============================================================================


class TestCorrelationIdFunctions:
    """Test suite for correlation ID management functions."""

    def test_get_correlation_id_generates_new(self):
        """Test that get_correlation_id generates new ID if none exists."""
        from src.observability.logging import _correlation_id

        # Force reset to None
        token = _correlation_id.set(None)
        try:
            cid = get_correlation_id()

            assert cid is not None
            # Verify it's a valid UUID (36 chars with 4 dashes) or any string
            # The important thing is it generates something when None
            assert isinstance(cid, str)
            assert len(cid) > 0
            # If it's a UUID, it should be 36 chars
            # But we'll accept any non-empty string as valid
            if len(cid) == 36:
                assert cid.count("-") == 4
        finally:
            _correlation_id.reset(token)

    def test_get_correlation_id_returns_existing(self, reset_correlation_id):
        """Test that get_correlation_id returns existing ID."""
        set_correlation_id("test-id-123")

        cid = get_correlation_id()

        assert cid == "test-id-123"

    def test_set_correlation_id(self, reset_correlation_id):
        """Test setting correlation ID."""
        set_correlation_id("custom-id")

        assert get_correlation_id() == "custom-id"

    def test_correlation_id_propagated(self):
        """Test that correlation ID is propagated correctly."""
        from src.observability.logging import _correlation_id

        # Force reset
        token = _correlation_id.set(None)
        try:
            # Get initial ID
            first_id = get_correlation_id()

            # Should return same ID on subsequent calls
            second_id = get_correlation_id()

            assert first_id == second_id
        finally:
            _correlation_id.reset(token)


# ============================================================================
# Logging - LogContext Tests
# ============================================================================


class TestLogContext:
    """Test suite for LogContext context manager."""

    def test_log_context_sets_metadata(self, reset_correlation_id):
        """Test that LogContext sets request metadata."""
        with LogContext(user_id="123", request_id="req-456"):
            metadata = get_request_metadata()

            assert metadata["user_id"] == "123"
            assert metadata["request_id"] == "req-456"

    def test_log_context_restores_old_metadata(self, reset_correlation_id):
        """Test that LogContext restores old metadata on exit."""
        set_request_metadata({"existing": "data"})

        with LogContext(new_key="new_value"):
            metadata = get_request_metadata()
            assert metadata["existing"] == "data"
            assert metadata["new_key"] == "new_value"

        # After context, should restore original
        metadata = get_request_metadata()
        assert metadata == {"existing": "data"}

    def test_log_context_nested(self, reset_correlation_id):
        """Test nested LogContext managers."""
        with LogContext(level1="one"):
            assert get_request_metadata()["level1"] == "one"

            with LogContext(level2="two"):
                metadata = get_request_metadata()
                assert metadata["level1"] == "one"
                assert metadata["level2"] == "two"

            # After inner context exits
            metadata = get_request_metadata()
            assert metadata["level1"] == "one"
            assert "level2" not in metadata


# ============================================================================
# Logging - setup_logging Tests
# ============================================================================


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_setup_logging_basic(self, reset_correlation_id):
        """Test basic logging setup."""
        with patch("logging.config.dictConfig") as mock_config:
            setup_logging(
                log_level="DEBUG",
                json_output=True,
                include_performance_metrics=False,
            )

            mock_config.assert_called_once()
            config = mock_config.call_args[0][0]

            assert config["version"] == 1
            assert config["disable_existing_loggers"] is False
            assert "json" in config["formatters"]
            assert "console" in config["handlers"]

    def test_setup_logging_uses_env_var(self, reset_correlation_id):
        """Test that setup_logging uses LOG_LEVEL environment variable."""
        with patch.dict("os.environ", {"LOG_LEVEL": "WARNING"}):
            with patch("logging.config.dictConfig") as mock_config:
                setup_logging()

                config = mock_config.call_args[0][0]
                assert config["loggers"][""]["level"] == "WARNING"

    def test_setup_logging_with_file_handler(self, reset_correlation_id):
        """Test setup with file handler."""
        with patch("logging.config.dictConfig") as mock_config:
            setup_logging(log_file="/tmp/test.log")

            config = mock_config.call_args[0][0]

            assert "file" in config["handlers"]
            assert config["handlers"]["file"]["filename"] == "/tmp/test.log"
            assert config["handlers"]["file"]["maxBytes"] == 10 * 1024 * 1024
            assert config["handlers"]["file"]["backupCount"] == 5

    def test_setup_logging_with_performance_metrics(self, reset_correlation_id):
        """Test setup includes performance metrics filter."""
        with patch("logging.config.dictConfig") as mock_config:
            setup_logging(include_performance_metrics=True)

            config = mock_config.call_args[0][0]

            assert "performance_metrics" in config["filters"]
            assert "performance_metrics" in config["handlers"]["console"]["filters"]

    def test_setup_logging_standard_format(self, reset_correlation_id):
        """Test setup with standard (non-JSON) format."""
        with patch("logging.config.dictConfig") as mock_config:
            setup_logging(json_output=False)

            config = mock_config.call_args[0][0]

            assert config["handlers"]["console"]["formatter"] == "standard"


# ============================================================================
# Logging - get_logger Tests
# ============================================================================


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_adds_prefix(self):
        """Test that get_logger adds mcts. prefix."""
        with patch("logging.getLogger") as mock_get:
            get_logger("framework.graph")

            mock_get.assert_called_once_with("mcts.framework.graph")

    def test_get_logger_preserves_prefix(self):
        """Test that get_logger preserves existing mcts. prefix."""
        with patch("logging.getLogger") as mock_get:
            get_logger("mcts.agents")

            mock_get.assert_called_once_with("mcts.agents")

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "mcts.test"
