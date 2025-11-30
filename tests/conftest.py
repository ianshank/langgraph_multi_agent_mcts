"""
Pytest Configuration for LangGraph Multi-Agent MCTS Tests.

This module provides:
- Centralized handling of optional dependencies (torch, transformers, peft, langgraph)
- Common fixtures for testing
- Skip markers for tests requiring specific dependencies
- Mock setup for unavailable dependencies during test collection
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# ============================================================================
# DEPENDENCY AVAILABILITY FLAGS
# ============================================================================

# Check for optional dependencies at import time
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers = None

try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    peft = None

try:
    import langgraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    langgraph = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# ============================================================================
# SKIP MARKERS
# ============================================================================

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="Test requires PyTorch"
)

requires_transformers = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="Test requires transformers library"
)

requires_peft = pytest.mark.skipif(
    not PEFT_AVAILABLE,
    reason="Test requires peft library"
)

requires_langgraph = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE,
    reason="Test requires langgraph library"
)

requires_neural = pytest.mark.skipif(
    not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE),
    reason="Test requires PyTorch and transformers"
)


# ============================================================================
# MOCK MODULES FOR COLLECTION PHASE
# ============================================================================

def create_mock_module(name: str) -> MagicMock:
    """Create a mock module for test collection."""
    mock = MagicMock()
    mock.__name__ = name
    mock.__file__ = f"<mock {name}>"
    return mock


# Mock modules that may not be available during test collection
_MOCK_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils",
    "torch.utils.data",
    "transformers",
    "transformers.models",
    "peft",
    "peft.config",
    "langgraph",
    "langgraph.graph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
]


def setup_mock_modules():
    """Set up mock modules for unavailable dependencies."""
    for module_name in _MOCK_MODULES:
        if module_name not in sys.modules:
            # Only mock if not already available
            try:
                __import__(module_name)
            except ImportError:
                sys.modules[module_name] = create_mock_module(module_name)


# Only set up mocks during collection phase
if "pytest" in sys.modules:
    # We're in pytest, set up mocks for unavailable modules
    setup_mock_modules()


# ============================================================================
# COMMON FIXTURES
# ============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.debug = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    return logger


@pytest.fixture
def mock_model_adapter():
    """Create a mock model adapter for testing."""
    from unittest.mock import AsyncMock
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value=MagicMock(text="Generated response"))
    return adapter


@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Sample agent state for testing."""
    return {
        "query": "What is the capital of France?",
        "use_mcts": True,
        "use_rag": False,
        "iteration": 0,
        "max_iterations": 3,
        "agent_outputs": [],
    }


@pytest.fixture
def mcts_config():
    """Create a test MCTS configuration."""
    from src.framework.mcts.config import MCTSConfig
    return MCTSConfig(
        num_iterations=10,
        seed=42,
        exploration_weight=1.414,
        max_rollout_depth=5,
        cache_size_limit=100,
    )


@pytest.fixture
def graph_config():
    """Create a test graph configuration."""
    from src.framework.actions import GraphConfig
    return GraphConfig(
        max_iterations=2,
        enable_parallel_agents=False,
    )


# ============================================================================
# NEURAL FIXTURES (only available with torch)
# ============================================================================

@pytest.fixture
def torch_device() -> str:
    """Get the appropriate torch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.randn(1, 10)


@pytest.fixture
def hrm_config():
    """Create HRM configuration for testing."""
    from src.training.system_config import HRMConfig
    return HRMConfig(
        h_dim=64,
        l_dim=32,
        num_h_layers=1,
        num_l_layers=2,
        max_outer_steps=3,
    )


@pytest.fixture
def trm_config():
    """Create TRM configuration for testing."""
    from src.training.system_config import TRMConfig
    return TRMConfig(
        latent_dim=64,
        hidden_dim=128,
        num_recursions=4,
    )


# ============================================================================
# TEST MARKERS CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "neural: marks tests that require neural network dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Add neural marker to tests in neural-related modules
    for item in items:
        if "neural" in item.nodeid or "hrm" in item.nodeid or "trm" in item.nodeid:
            item.add_marker(pytest.mark.neural)


# ============================================================================
# HELPER FUNCTIONS FOR TESTS
# ============================================================================

def assert_valid_confidence(value: float, name: str = "confidence"):
    """Assert that a value is a valid confidence score in [0, 1]."""
    assert isinstance(value, (int, float)), f"{name} must be numeric"
    assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}"


def assert_valid_mcts_stats(stats: dict[str, Any]):
    """Assert that MCTS stats dictionary has expected fields."""
    required_fields = ["iterations", "cache_hit_rate"]
    for field in required_fields:
        assert field in stats, f"Missing required field: {field}"
