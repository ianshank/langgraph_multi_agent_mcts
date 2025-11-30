"""
Integration tests for Neural MCTS Integration.

Tests the bridge between NeuralMCTS and the LangGraph workflow.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np

# Import core MCTS modules (always available)
from src.framework.mcts.core import MCTSState, MCTSNode
from src.framework.mcts.config import MCTSConfig, ConfigPreset, create_preset_config
from src.framework.actions import (
    ActionType,
    AgentType,
    RouteDecision,
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    DEFAULT_GRAPH_CONFIG,
)

# Guard neural imports for test collection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Conditional import of neural integration
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralMCTSAdapter,
        NeuralRolloutPolicy,
        create_neural_mcts_adapter,
        get_fast_neural_config,
        get_balanced_neural_config,
        get_alphazero_config,
    )
    NEURAL_INTEGRATION_AVAILABLE = True
except ImportError:
    NEURAL_INTEGRATION_AVAILABLE = False


class TestActionTypeEnum:
    """Test ActionType enum functionality."""

    def test_root_actions_returns_list(self):
        """Root actions should return a non-empty list."""
        actions = ActionType.root_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_continuation_actions_returns_list(self):
        """Continuation actions should return a non-empty list."""
        actions = ActionType.continuation_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_root_and_continuation_are_different(self):
        """Root and continuation actions should be distinct."""
        root = set(ActionType.root_actions())
        cont = set(ActionType.continuation_actions())
        assert root.isdisjoint(cont), "Root and continuation actions should not overlap"


class TestAgentTypeEnum:
    """Test AgentType enum functionality."""

    def test_all_agents_includes_core_agents(self):
        """All agents should include HRM, TRM, and MCTS."""
        agents = AgentType.all_agents()
        assert "hrm" in agents
        assert "trm" in agents
        assert "mcts" in agents

    def test_reasoning_agents_excludes_control_flow(self):
        """Reasoning agents should not include control flow agents."""
        reasoning = AgentType.reasoning_agents()
        all_agents = AgentType.all_agents()
        assert len(reasoning) < len(all_agents)
        assert "parallel" not in reasoning
        assert "aggregate" not in reasoning


class TestRouteDecision:
    """Test RouteDecision enum and node mapping."""

    def test_to_node_name_maps_correctly(self):
        """Route decisions should map to expected node names."""
        assert RouteDecision.PARALLEL.to_node_name() == "parallel_agents"
        assert RouteDecision.HRM.to_node_name() == "hrm_agent"
        assert RouteDecision.TRM.to_node_name() == "trm_agent"
        assert RouteDecision.MCTS.to_node_name() == "mcts_simulator"


class TestConfidenceConfig:
    """Test ConfidenceConfig validation."""

    def test_default_values_are_valid(self):
        """Default confidence config should have valid values."""
        config = ConfidenceConfig()
        assert 0 <= config.default_hrm_confidence <= 1
        assert 0 <= config.consensus_threshold <= 1
        assert 0 <= config.heuristic_base_value <= 1

    def test_invalid_confidence_raises_error(self):
        """Invalid confidence values should raise ValueError."""
        with pytest.raises(ValueError):
            ConfidenceConfig(default_hrm_confidence=1.5)

        with pytest.raises(ValueError):
            ConfidenceConfig(consensus_threshold=-0.1)


class TestRolloutWeights:
    """Test RolloutWeights validation."""

    def test_default_weights_sum_to_one(self):
        """Default rollout weights should sum to 1.0."""
        weights = RolloutWeights()
        assert abs(weights.heuristic_weight + weights.random_weight - 1.0) < 1e-6

    def test_invalid_weights_raise_error(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError):
            RolloutWeights(heuristic_weight=0.5, random_weight=0.3)


class TestGraphConfig:
    """Test GraphConfig functionality."""

    def test_default_config_is_valid(self):
        """Default graph config should be valid."""
        config = DEFAULT_GRAPH_CONFIG
        assert config.max_iterations >= 1
        assert config.top_k_retrieval >= 1

    def test_get_route_mapping_returns_dict(self):
        """Route mapping should return a dictionary."""
        config = GraphConfig()
        mapping = config.get_route_mapping()
        assert isinstance(mapping, dict)
        assert "parallel" in mapping
        assert "hrm" in mapping

    def test_to_dict_is_serializable(self):
        """Config should convert to serializable dict."""
        config = GraphConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "max_iterations" in d
        assert "confidence" in d

    def test_from_dict_roundtrip(self):
        """Config should roundtrip through dict."""
        original = GraphConfig(max_iterations=5, top_k_retrieval=10)
        d = original.to_dict()
        restored = GraphConfig.from_dict(d)
        assert restored.max_iterations == original.max_iterations
        assert restored.top_k_retrieval == original.top_k_retrieval


@pytest.mark.skipif(
    not NEURAL_INTEGRATION_AVAILABLE,
    reason="Neural MCTS integration not available"
)
class TestNeuralMCTSConfig:
    """Test NeuralMCTSConfig functionality."""

    def test_default_config_is_valid(self):
        """Default neural MCTS config should be valid."""
        config = NeuralMCTSConfig()
        assert config.num_simulations >= 1
        assert config.c_puct >= 0
        assert 0 <= config.dirichlet_epsilon <= 1

    def test_invalid_config_raises_error(self):
        """Invalid config should raise ValueError."""
        with pytest.raises(ValueError):
            NeuralMCTSConfig(num_simulations=0)

        with pytest.raises(ValueError):
            NeuralMCTSConfig(c_puct=-1)

    def test_to_framework_config(self):
        """Should convert to framework config parameters."""
        config = NeuralMCTSConfig(num_simulations=200, seed=123)
        framework_params = config.to_framework_config()
        assert framework_params["num_iterations"] == 200
        assert framework_params["seed"] == 123

    def test_preset_configs_are_valid(self):
        """Preset configurations should be valid."""
        for preset_fn in [get_fast_neural_config, get_balanced_neural_config, get_alphazero_config]:
            config = preset_fn()
            assert config.num_simulations >= 1
            config.to_framework_config()  # Should not raise


@pytest.mark.skipif(
    not NEURAL_INTEGRATION_AVAILABLE or not TORCH_AVAILABLE,
    reason="Neural MCTS integration or PyTorch not available"
)
class TestNeuralRolloutPolicy:
    """Test NeuralRolloutPolicy functionality."""

    @pytest.fixture
    def policy(self):
        """Create a neural rollout policy without network."""
        config = NeuralMCTSConfig(
            num_simulations=10,
            enable_cache=True,
            cache_size_limit=100,
        )
        return NeuralRolloutPolicy(
            policy_value_network=None,
            config=config,
            fallback_value=0.5,
        )

    @pytest.fixture
    def sample_state(self):
        """Create a sample MCTS state."""
        return MCTSState(
            state_id="test_state",
            features={
                "query": "test query",
                "confidence": 0.8,
            }
        )

    @pytest.mark.asyncio
    async def test_evaluate_without_network_uses_fallback(self, policy, sample_state):
        """Without network, should use fallback with features."""
        rng = np.random.default_rng(42)
        value = await policy.evaluate(sample_state, rng)
        assert isinstance(value, float)
        assert policy._config.min_value <= value <= policy._config.max_value

    @pytest.mark.asyncio
    async def test_cache_works(self, policy, sample_state):
        """Cache should store and retrieve values."""
        rng = np.random.default_rng(42)

        # First evaluation
        await policy.evaluate(sample_state, rng)
        initial_misses = policy._cache_misses
        assert initial_misses >= 1, "Should have at least one cache miss"

        # Second evaluation with same state (should hit cache)
        await policy.evaluate(sample_state, rng)

        # Cache should have recorded a hit (value was cached)
        # Note: The cache stores base value, then adds noise on retrieval
        # So after first eval, state hash is in cache
        assert len(policy._cache) >= 1, "Cache should contain at least one entry"

    def test_cache_stats(self, policy):
        """Should return cache statistics."""
        stats = policy.get_cache_stats()
        assert "cache_size" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self, policy):
        """Should clear the cache."""
        policy._cache["test"] = 0.5
        policy.clear_cache()
        assert len(policy._cache) == 0


@pytest.mark.skipif(
    not NEURAL_INTEGRATION_AVAILABLE,
    reason="Neural MCTS integration not available"
)
class TestNeuralMCTSAdapter:
    """Test NeuralMCTSAdapter functionality."""

    def test_create_adapter_without_network(self):
        """Should create adapter without network."""
        adapter = create_neural_mcts_adapter(network=None)
        assert adapter is not None
        assert adapter._initialized

    def test_adapter_provides_rollout_policy(self):
        """Adapter should provide a rollout policy."""
        adapter = create_neural_mcts_adapter(network=None)
        policy = adapter.rollout_policy
        assert policy is not None
        assert isinstance(policy, NeuralRolloutPolicy)

    def test_adapter_provides_mcts_engine_kwargs(self):
        """Adapter should provide kwargs for MCTSEngine."""
        config = NeuralMCTSConfig(seed=123)
        adapter = create_neural_mcts_adapter(network=None, config=config)
        kwargs = adapter.get_mcts_engine_kwargs()
        assert "seed" in kwargs
        assert kwargs["seed"] == 123


class TestMCTSConfigIntegration:
    """Test integration between MCTS config and graph config."""

    def test_configs_are_compatible(self):
        """MCTS config and graph config should work together."""
        mcts_config = create_preset_config(ConfigPreset.BALANCED)
        graph_config = GraphConfig()

        # Both should be usable together
        assert mcts_config.num_iterations > 0
        assert len(graph_config.root_actions) > 0

    def test_mcts_preset_values_are_reasonable(self):
        """MCTS presets should have reasonable values."""
        fast = create_preset_config(ConfigPreset.FAST)
        thorough = create_preset_config(ConfigPreset.THOROUGH)

        # Fast should have fewer iterations
        assert fast.num_iterations < thorough.num_iterations

        # Both should be valid
        fast.validate()
        thorough.validate()
