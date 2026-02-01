"""
Unit tests for connection pool management.

Tests HTTP connection pooling, metrics collection, and retry logic.

Based on: NEXT_STEPS_PLAN.md Phase 5.1
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Check httpx availability
# =============================================================================

HTTPX_AVAILABLE = importlib.util.find_spec("httpx") is not None

skip_if_no_httpx = pytest.mark.skipif(
    not HTTPX_AVAILABLE,
    reason="httpx not installed",
)


# =============================================================================
# ConnectionPoolConfig Tests
# =============================================================================


class TestConnectionPoolConfig:
    """Tests for ConnectionPoolConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from src.performance.connection_pool import ConnectionPoolConfig

        config = ConnectionPoolConfig()

        assert config.max_connections == 100
        assert config.max_keepalive_connections == 20
        assert config.keepalive_expiry == 30.0
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.http2 is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        from src.performance.connection_pool import ConnectionPoolConfig

        config = ConnectionPoolConfig(
            max_connections=50,
            connect_timeout=5.0,
            http2=False,
        )

        assert config.max_connections == 50
        assert config.connect_timeout == 5.0
        assert config.http2 is False

    def test_retry_status_codes_default(self):
        """Test default retry status codes."""
        from src.performance.connection_pool import ConnectionPoolConfig

        config = ConnectionPoolConfig()

        assert 429 in config.retry_status_codes  # Rate limit
        assert 500 in config.retry_status_codes  # Internal server error
        assert 503 in config.retry_status_codes  # Service unavailable

    @skip_if_no_httpx
    def test_to_httpx_limits(self):
        """Test conversion to httpx.Limits."""
        from src.performance.connection_pool import ConnectionPoolConfig

        config = ConnectionPoolConfig(
            max_connections=50,
            max_keepalive_connections=10,
        )
        limits = config.to_httpx_limits()

        assert limits is not None
        assert limits.max_connections == 50
        assert limits.max_keepalive_connections == 10

    @skip_if_no_httpx
    def test_to_httpx_timeout(self):
        """Test conversion to httpx.Timeout."""
        from src.performance.connection_pool import ConnectionPoolConfig

        config = ConnectionPoolConfig(
            connect_timeout=5.0,
            read_timeout=30.0,
        )
        timeout = config.to_httpx_timeout()

        assert timeout is not None
        assert timeout.connect == 5.0
        assert timeout.read == 30.0


# =============================================================================
# PoolMetrics Tests
# =============================================================================


class TestPoolMetrics:
    """Tests for PoolMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics start at zero."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()

        assert metrics.requests_total == 0
        assert metrics.requests_successful == 0
        assert metrics.requests_failed == 0

    def test_record_successful_request(self):
        """Test recording a successful request."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        metrics.record_request(
            success=True,
            latency_ms=100.0,
            retries=0,
            connection_reused=True,
        )

        assert metrics.requests_total == 1
        assert metrics.requests_successful == 1
        assert metrics.requests_failed == 0
        assert 100.0 in metrics.latency_samples

    def test_record_failed_request(self):
        """Test recording a failed request."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        metrics.record_request(
            success=False,
            latency_ms=500.0,
            retries=3,
        )

        assert metrics.requests_total == 1
        assert metrics.requests_failed == 1
        assert metrics.retries_total == 3

    def test_latency_samples_bounded(self):
        """Test latency samples don't exceed max."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        metrics.max_latency_samples = 10

        for i in range(20):
            metrics.record_request(success=True, latency_ms=float(i))

        assert len(metrics.latency_samples) == 10
        # Should have samples 10-19 (oldest evicted)
        assert metrics.latency_samples[0] == 10.0

    def test_get_stats_returns_dict(self):
        """Test get_stats returns complete dictionary."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        metrics.record_request(success=True, latency_ms=100.0)

        stats = metrics.get_stats()

        assert "requests" in stats
        assert "latency" in stats
        assert "connections" in stats
        assert "health" in stats

    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        for _ in range(8):
            metrics.record_request(success=True, latency_ms=50.0)
        for _ in range(2):
            metrics.record_request(success=False, latency_ms=100.0)

        stats = metrics.get_stats()

        assert stats["requests"]["success_rate"] == 0.8

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        from src.performance.connection_pool import PoolMetrics

        metrics = PoolMetrics()
        for i in range(100):
            metrics.record_request(success=True, latency_ms=float(i + 1))

        stats = metrics.get_stats()

        assert stats["latency"]["min_ms"] == 1.0
        assert stats["latency"]["max_ms"] == 100.0
        assert stats["latency"]["p50_ms"] == 51.0  # Median of 1-100


# =============================================================================
# ConnectionPool Tests
# =============================================================================


@skip_if_no_httpx
class TestConnectionPool:
    """Tests for ConnectionPool class."""

    def test_pool_creation(self):
        """Test connection pool can be created."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)  # Disable HTTP/2 for tests
        pool = ConnectionPool(config)

        assert pool is not None
        assert pool._client is None  # Not created until first request

    def test_pool_with_default_config(self):
        """Test pool with default configuration."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)
        pool = ConnectionPool(config)

        assert pool.config.max_connections == 100

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        """Test _get_client creates httpx client."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)  # Disable HTTP/2 for tests
        pool = ConnectionPool(config)
        client = await pool._get_client()

        assert client is not None
        assert pool._client is client

        await pool.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        """Test _get_client returns same client."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)
        pool = ConnectionPool(config)
        client1 = await pool._get_client()
        client2 = await pool._get_client()

        assert client1 is client2

        await pool.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test pool as context manager."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)
        async with ConnectionPool(config) as pool:
            assert pool is not None
            await pool._get_client()

        # Should be closed after context
        assert pool._closed is True

    def test_get_stats_returns_metrics(self):
        """Test get_stats returns pool metrics."""
        from src.performance.connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(http2=False)
        pool = ConnectionPool(config)
        stats = pool.get_stats()

        assert "requests" in stats
        assert "latency" in stats
        assert "connections" in stats


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateConnectionPool:
    """Tests for create_connection_pool factory function."""

    @skip_if_no_httpx
    def test_factory_creates_pool(self):
        """Test factory creates pool with custom settings."""
        from src.performance.connection_pool import create_connection_pool

        pool = create_connection_pool(
            max_connections=50,
            connect_timeout=5.0,
        )

        assert pool is not None
        assert pool.config.max_connections == 50
        assert pool.config.connect_timeout == 5.0

    def test_factory_returns_none_without_httpx(self):
        """Test factory returns None when httpx unavailable."""
        from src.performance.connection_pool import create_connection_pool

        with patch("src.performance.connection_pool._HAS_HTTPX", False):
            pool = create_connection_pool()

            assert pool is None
