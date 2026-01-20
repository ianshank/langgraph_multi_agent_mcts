"""
Root conftest.py - Shared test fixtures for the entire test suite.

This module provides centralized pytest configuration and fixtures used
across all test modules. Following best practices from 2026:
- Session-scoped fixtures for expensive resources
- Function-scoped fixtures for test isolation
- Async fixtures with proper cleanup
- No hardcoded values - all configurable via environment

Based on: CLAUDE_CODE_IMPLEMENTATION_TEMPLATE.md
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import project modules with graceful fallback
try:
    from src.config.settings import Settings, get_settings, reset_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

try:
    from src.adapters.llm.base import LLMClient, LLMResponse
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
    from src.framework.mcts.config import MCTSConfig
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False

try:
    from src.observability.logging import get_correlation_id, set_correlation_id
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Fast, isolated unit tests")
    config.addinivalue_line("markers", "integration: Component interaction tests")
    config.addinivalue_line("markers", "e2e: End-to-end scenario tests")
    config.addinivalue_line("markers", "slow: Tests taking >10 seconds")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "enterprise: Enterprise use case tests")
    config.addinivalue_line("markers", "mcts: MCTS-related tests")
    config.addinivalue_line("markers", "neural: Neural network tests (requires PyTorch)")
    config.addinivalue_line("markers", "llm: LLM integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment."""
    skip_slow = pytest.mark.skip(reason="Skipping slow tests (use --runslow to include)")
    skip_neural = pytest.mark.skip(reason="Skipping neural tests (PyTorch not available or SKIP_NEURAL=1)")
    skip_llm = pytest.mark.skip(reason="Skipping LLM tests (no API key or SKIP_LLM=1)")

    for item in items:
        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not config.getoption("--runslow", default=False):
            item.add_marker(skip_slow)

        # Skip neural tests if PyTorch unavailable or SKIP_NEURAL set
        if "neural" in item.keywords:
            try:
                import torch  # noqa: F401
                if os.environ.get("SKIP_NEURAL", "0") == "1":
                    item.add_marker(skip_neural)
            except ImportError:
                item.add_marker(skip_neural)

        # Skip LLM tests if no API key or SKIP_LLM set
        if "llm" in item.keywords:
            if os.environ.get("SKIP_LLM", "0") == "1":
                item.add_marker(skip_llm)
            elif not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
                item.add_marker(skip_llm)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )


# =============================================================================
# Session-Scoped Fixtures (Expensive, shared across all tests)
# =============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
    """Provide event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
def test_logger() -> logging.Logger:
    """Create a test logger that doesn't pollute stdout."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    # Use NullHandler to suppress output during tests
    logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture(scope="session")
def torch_device() -> str:
    """Determine the best available torch device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# =============================================================================
# Settings Fixtures
# =============================================================================

@pytest.fixture
def test_settings() -> Generator[Settings, None, None]:
    """
    Create isolated test settings.

    Resets settings after test to prevent pollution.
    """
    if not SETTINGS_AVAILABLE:
        pytest.skip("Settings module not available")

    # Reset any cached settings
    reset_settings()

    # Create test settings with safe defaults
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-key-for-testing-only",
        "MCTS_ENABLED": "true",
        "MCTS_ITERATIONS": "10",
        "SEED": "42",
        "LOG_LEVEL": "DEBUG",
    }):
        settings = get_settings()
        yield settings

    # Cleanup
    reset_settings()


@pytest.fixture
def settings_override():
    """
    Context manager for temporarily overriding settings.

    Usage:
        def test_something(settings_override):
            with settings_override({"MCTS_ITERATIONS": "5"}):
                # Test with overridden settings
    """
    if not SETTINGS_AVAILABLE:
        pytest.skip("Settings module not available")

    from contextlib import contextmanager

    @contextmanager
    def _override(overrides: dict[str, str]):
        reset_settings()
        with patch.dict(os.environ, overrides):
            yield get_settings()
        reset_settings()

    return _override


# =============================================================================
# LLM Client Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create a standard mock LLM response."""
    if not LLM_AVAILABLE:
        pytest.skip("LLM module not available")

    return LLMResponse(
        content="This is a test response from the mock LLM.",
        model="gpt-4-test",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        finish_reason="stop",
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> AsyncMock:
    """
    Create a mock LLM client for testing.

    Returns an AsyncMock that mimics the LLMClient interface.
    """
    if not LLM_AVAILABLE:
        pytest.skip("LLM module not available")

    client = AsyncMock(spec=LLMClient)
    client.generate.return_value = mock_llm_response
    client.generate_with_tools.return_value = mock_llm_response
    client.model = "gpt-4-test"
    client.provider = "openai"

    return client


@pytest.fixture
def mock_llm_client_streaming() -> AsyncMock:
    """Create a mock LLM client that supports streaming."""
    if not LLM_AVAILABLE:
        pytest.skip("LLM module not available")

    async def mock_stream(*args, **kwargs):
        chunks = ["This ", "is ", "a ", "streaming ", "response."]
        for chunk in chunks:
            yield {"content": chunk, "finish_reason": None}
        yield {"content": "", "finish_reason": "stop"}

    client = AsyncMock(spec=LLMClient)
    client.generate_stream = mock_stream
    client.model = "gpt-4-test"
    client.provider = "openai"

    return client


@pytest.fixture
def mock_llm_client_error() -> AsyncMock:
    """Create a mock LLM client that raises errors."""
    if not LLM_AVAILABLE:
        pytest.skip("LLM module not available")

    from src.adapters.llm.exceptions import LLMError

    client = AsyncMock(spec=LLMClient)
    client.generate.side_effect = LLMError("API unavailable")
    client.model = "gpt-4-test"
    client.provider = "openai"

    return client


# =============================================================================
# MCTS Fixtures
# =============================================================================

@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Create a fast MCTS configuration for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS module not available")

    return MCTSConfig(
        seed=42,
        num_iterations=10,  # Low for fast tests
        exploration_weight=1.414,
        max_rollout_depth=5,
        cache_size=100,
    )


@pytest.fixture
def mcts_engine(mcts_config: MCTSConfig) -> MCTSEngine:
    """Create an MCTS engine for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS module not available")

    return MCTSEngine(
        seed=mcts_config.seed,
        exploration_weight=mcts_config.exploration_weight,
    )


@pytest.fixture
def simple_mcts_state() -> MCTSState:
    """Create a simple MCTS state for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS module not available")

    return MCTSState(
        state_id="test_state_001",
        features={
            "query": "test query",
            "depth": 0,
            "domain": "test",
        },
    )


@pytest.fixture
def mcts_node(simple_mcts_state: MCTSState, mcts_engine: MCTSEngine) -> MCTSNode:
    """Create an MCTS node for testing."""
    if not MCTS_AVAILABLE:
        pytest.skip("MCTS module not available")

    return MCTSNode(
        state=simple_mcts_state,
        rng=mcts_engine.rng,
    )


# =============================================================================
# Correlation ID and Logging Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_correlation_id():
    """Reset correlation ID before each test for isolation."""
    if LOGGING_AVAILABLE:
        set_correlation_id(None)
    yield
    if LOGGING_AVAILABLE:
        set_correlation_id(None)


@pytest.fixture
def correlation_id() -> str:
    """Generate and set a correlation ID for the test."""
    if not LOGGING_AVAILABLE:
        return "test-correlation-id"

    import uuid
    test_id = f"test-{uuid.uuid4().hex[:8]}"
    set_correlation_id(test_id)
    return test_id


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def mock_hrm_agent() -> MagicMock:
    """Create a mock HRM agent for testing."""
    agent = MagicMock()
    agent.forward.return_value = MagicMock(
        final_state=MagicMock(),
        subproblems=[],
        halt_step=3,
        total_ponder_cost=0.5,
        convergence_path=[0.5, 0.7, 0.95],
    )
    agent.get_parameter_count.return_value = 1000000
    return agent


@pytest.fixture
def mock_trm_agent() -> MagicMock:
    """Create a mock TRM agent for testing."""
    agent = MagicMock()
    agent.forward.return_value = MagicMock(
        final_prediction=MagicMock(),
        intermediate_predictions=[],
        recursion_depth=5,
        converged=True,
        convergence_step=4,
        residual_norms=[0.1, 0.05, 0.02, 0.01],
    )
    agent.get_parameter_count.return_value = 500000
    return agent


# =============================================================================
# Async Fixtures and Helpers
# =============================================================================

@pytest.fixture
async def async_mock_llm_client(mock_llm_response: LLMResponse) -> AsyncGenerator[AsyncMock, None]:
    """
    Async fixture for LLM client with proper cleanup.
    """
    if not LLM_AVAILABLE:
        pytest.skip("LLM module not available")

    client = AsyncMock(spec=LLMClient)
    client.generate.return_value = mock_llm_response

    yield client

    # Cleanup any pending calls
    client.reset_mock()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_query() -> str:
    """Provide a sample query for testing."""
    return "Analyze the financial health of the target company for the M&A due diligence process."


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Provide sample context data for testing."""
    return {
        "domain": "ma_due_diligence",
        "target_company": "TestCo Inc.",
        "acquirer_company": "AcquirerCorp",
        "deal_value_usd": 100_000_000,
        "jurisdictions": ["US", "EU"],
        "rag_context": "Previous analysis indicates strong revenue growth...",
    }


@pytest.fixture
def sample_agent_state(sample_query: str, sample_context: dict) -> dict[str, Any]:
    """Provide a sample agent state for testing."""
    return {
        "query": sample_query,
        "use_mcts": True,
        "use_rag": True,
        "hrm_results": {},
        "trm_results": {},
        "agent_outputs": [],
        "confidence_scores": {},
        "consensus_reached": False,
        "iteration": 0,
        "max_iterations": 10,
        "context": sample_context,
    }


# =============================================================================
# Enterprise Use Case Fixtures
# =============================================================================

@pytest.fixture
def enterprise_config_overrides() -> dict[str, str]:
    """Provide environment variable overrides for enterprise testing."""
    return {
        "ENTERPRISE_ENABLED": "true",
        "MA_DD_ENABLED": "true",
        "MA_DD_MAX_MCTS_ITERATIONS": "10",
        "MA_DD_RISK_THRESHOLD": "0.6",
        "CLINICAL_TRIAL_ENABLED": "true",
        "CLINICAL_TRIAL_MIN_STATISTICAL_POWER": "0.8",
        "REG_COMPLIANCE_ENABLED": "true",
        "REG_COMPLIANCE_RISK_TOLERANCE": "moderate",
    }


# =============================================================================
# Cleanup and Safety
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Ensure cleanup after each test."""
    yield
    # Add any global cleanup here
    if SETTINGS_AVAILABLE:
        try:
            reset_settings()
        except Exception:
            pass


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def benchmark_iterations() -> int:
    """Number of iterations for benchmark tests."""
    return int(os.environ.get("BENCHMARK_ITERATIONS", "100"))


@pytest.fixture
def performance_threshold_ms() -> float:
    """Maximum allowed time in milliseconds for performance tests."""
    return float(os.environ.get("PERFORMANCE_THRESHOLD_MS", "1000.0"))
