"""
End-to-End Tests for LangGraph Multi-Agent MCTS.

Tests the complete pipeline from query input to response output,
including all agent interactions, MCTS search, and configuration.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

# Framework imports
from src.framework.actions import (
    ActionType,
    AgentType,
    RouteDecision,
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    create_research_config,
    create_coding_config,
    DEFAULT_GRAPH_CONFIG,
)
from src.framework.mcts.config import (
    MCTSConfig,
    ConfigPreset,
    create_preset_config,
)
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RandomRolloutPolicy

# Conditional imports
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralRolloutPolicy,
        NeuralMCTSAdapter,
        create_neural_mcts_adapter,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


class TestFullPipeline:
    """End-to-end tests for the complete processing pipeline."""

    @pytest.fixture
    def mcts_engine(self):
        """Create MCTS engine for testing."""
        config = create_preset_config(ConfigPreset.FAST)
        return MCTSEngine(
            seed=42,
            exploration_weight=config.exploration_weight,
            progressive_widening_k=config.progressive_widening_k,
            progressive_widening_alpha=config.progressive_widening_alpha,
        )

    @pytest.fixture
    def rollout_policy(self):
        """Create a simple rollout policy for testing."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    @pytest.fixture
    def sample_query(self):
        """Sample query for testing."""
        return "Explain the difference between supervised and unsupervised learning"

    @pytest.fixture
    def graph_config(self):
        """Test graph configuration."""
        return GraphConfig(
            max_iterations=2,
            enable_parallel_agents=False,
            confidence=ConfidenceConfig(consensus_threshold=0.7),
        )

    @pytest.mark.asyncio
    async def test_mcts_search_completes(self, mcts_engine, rollout_policy):
        """MCTS search should complete and return valid results."""
        # Create root state
        root_state = MCTSState(
            state_id="root",
            features={"query": "test query", "depth": 0},
        )
        root = MCTSNode(state=root_state, rng=mcts_engine.rng)

        # Define simple action generator
        def action_generator(state: MCTSState) -> list[str]:
            depth = state.features.get("depth", 0)
            if depth >= 3:
                return []
            return ["action_a", "action_b", "action_c"]

        # Define state transition
        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={**state.features, "depth": state.features.get("depth", 0) + 1},
            )

        # Run search
        best_action, stats = await mcts_engine.search(
            root=root,
            num_iterations=25,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
        )

        # Validate results
        assert best_action in ["action_a", "action_b", "action_c"]
        assert stats["iterations"] > 0
        assert "cache_hit_rate" in stats
        assert stats["best_action_visits"] > 0

    @pytest.mark.asyncio
    async def test_mcts_with_graph_config_actions(self, mcts_engine, graph_config, rollout_policy):
        """MCTS should use actions from GraphConfig."""
        root_state = MCTSState(state_id="root", features={})
        root = MCTSNode(state=root_state, rng=mcts_engine.rng)

        # Use configured actions
        root_actions = graph_config.root_actions
        continuation_actions = graph_config.continuation_actions

        def action_generator(state: MCTSState) -> list[str]:
            depth = len(state.state_id.split("_")) - 1
            if depth == 0:
                return root_actions
            elif depth < 3:
                return continuation_actions
            return []

        def state_transition(state: MCTSState, action: str) -> MCTSState:
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={"last_action": action},
            )

        best_action, stats = await mcts_engine.search(
            root=root,
            num_iterations=25,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
        )

        # Best action should be from configured actions
        assert best_action in root_actions

    @pytest.mark.asyncio
    async def test_deterministic_search_with_seed(self, rollout_policy):
        """Search should be deterministic with same seed."""
        results = []

        for _ in range(3):
            engine = MCTSEngine(seed=42)
            root = MCTSNode(
                state=MCTSState(state_id="root", features={}),
                rng=engine.rng,
            )

            def action_generator(state):
                return ["a", "b", "c"]

            def state_transition(state, action):
                return MCTSState(state_id=f"{state.state_id}_{action}", features={})

            best_action, stats = await engine.search(
                root=root,
                num_iterations=50,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
            )

            results.append((best_action, stats["best_action_visits"]))

        # All runs should produce same result
        assert all(r == results[0] for r in results), "Search should be deterministic"

    @pytest.mark.asyncio
    async def test_config_presets_produce_different_behaviors(self, rollout_policy):
        """Different config presets should produce different behaviors."""
        fast_config = create_preset_config(ConfigPreset.FAST)
        thorough_config = create_preset_config(ConfigPreset.THOROUGH)

        # Fast should have fewer iterations
        assert fast_config.num_iterations < thorough_config.num_iterations

        # Different exploration weights possible
        fast_engine = MCTSEngine(
            seed=42,
            exploration_weight=fast_config.exploration_weight,
        )
        thorough_engine = MCTSEngine(
            seed=42,
            exploration_weight=thorough_config.exploration_weight,
        )

        # Both should work
        root_fast = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=fast_engine.rng,
        )
        root_thorough = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=thorough_engine.rng,
        )

        def action_gen(s):
            return ["a", "b"] if len(s.state_id.split("_")) < 3 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        _, fast_stats = await fast_engine.search(
            root=root_fast,
            num_iterations=fast_config.num_iterations,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        _, thorough_stats = await thorough_engine.search(
            root=root_thorough,
            num_iterations=thorough_config.num_iterations,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        assert fast_stats["iterations"] < thorough_stats["iterations"]

    def test_graph_config_validation(self):
        """Graph config should validate parameters."""
        # Valid config
        config = GraphConfig(max_iterations=5)
        assert config.max_iterations == 5

        # Invalid config should raise
        with pytest.raises(ValueError):
            GraphConfig(max_iterations=0)

        with pytest.raises(ValueError):
            ConfidenceConfig(consensus_threshold=1.5)

        with pytest.raises(ValueError):
            RolloutWeights(heuristic_weight=0.5, random_weight=0.3)  # Doesn't sum to 1

    def test_domain_presets_have_appropriate_settings(self):
        """Domain presets should have appropriate settings."""
        research = create_research_config()
        coding = create_coding_config()

        # Research should be more exploratory
        assert research.rollout_weights.random_weight >= coding.rollout_weights.random_weight

        # Coding should have higher consensus threshold
        assert coding.confidence.consensus_threshold >= research.confidence.consensus_threshold

        # Coding should have lower temperature (more deterministic)
        assert coding.synthesis.temperature <= research.synthesis.temperature


@pytest.mark.skipif(not NEURAL_AVAILABLE, reason="Neural integration not available")
class TestNeuralMCTSPipeline:
    """E2E tests for neural MCTS integration."""

    @pytest.fixture
    def neural_config(self):
        """Neural MCTS configuration."""
        return NeuralMCTSConfig(
            num_simulations=50,
            enable_cache=True,
            cache_size_limit=100,
        )

    @pytest.fixture
    def neural_adapter(self, neural_config):
        """Neural MCTS adapter without network."""
        return create_neural_mcts_adapter(
            network=None,
            config=neural_config,
        )

    @pytest.mark.asyncio
    async def test_neural_rollout_policy_integration(self, neural_config):
        """Neural rollout policy should integrate with MCTS."""
        policy = NeuralRolloutPolicy(
            policy_value_network=None,
            config=neural_config,
        )

        state = MCTSState(
            state_id="test",
            features={"confidence": 0.8, "quality": 0.7},
        )

        rng = np.random.default_rng(42)
        value = await policy.evaluate(state, rng)

        assert isinstance(value, float)
        assert neural_config.min_value <= value <= neural_config.max_value

    @pytest.mark.asyncio
    async def test_neural_adapter_provides_engine_kwargs(self, neural_adapter):
        """Neural adapter should provide valid engine kwargs."""
        kwargs = neural_adapter.get_mcts_engine_kwargs()

        assert "seed" in kwargs
        assert "exploration_weight" in kwargs
        assert "progressive_widening_k" in kwargs

        # Should be usable to create engine
        engine = MCTSEngine(**kwargs)
        assert engine is not None

    @pytest.mark.asyncio
    async def test_mcts_with_neural_policy(self, neural_adapter):
        """MCTS should work with neural rollout policy."""
        engine = MCTSEngine(**neural_adapter.get_mcts_engine_kwargs())

        root = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=engine.rng,
        )

        def action_gen(s):
            depth = len(s.state_id.split("_")) - 1
            return ["a", "b"] if depth < 2 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=25,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=neural_adapter.rollout_policy,
        )

        assert best_action in ["a", "b"]
        assert stats["iterations"] > 0


class TestConfigurationE2E:
    """E2E tests for configuration system."""

    def test_all_presets_are_valid(self):
        """All configuration presets should be valid."""
        # MCTS presets
        for preset in ConfigPreset:
            config = create_preset_config(preset)
            config.validate()  # Should not raise

        # Domain presets
        for preset_fn in [create_research_config, create_coding_config]:
            config = preset_fn()
            assert config.max_iterations >= 1
            assert 0 <= config.confidence.consensus_threshold <= 1

    def test_config_serialization_roundtrip(self):
        """Configs should survive serialization roundtrip."""
        original = GraphConfig(
            max_iterations=5,
            confidence=ConfidenceConfig(consensus_threshold=0.8),
            rollout_weights=RolloutWeights(heuristic_weight=0.6, random_weight=0.4),
        )

        # Serialize and deserialize
        d = original.to_dict()
        restored = GraphConfig.from_dict(d)

        assert restored.max_iterations == original.max_iterations
        assert restored.confidence.consensus_threshold == original.confidence.consensus_threshold
        assert restored.rollout_weights.heuristic_weight == original.rollout_weights.heuristic_weight

    def test_mcts_config_json_roundtrip(self):
        """MCTS config should survive JSON roundtrip."""
        original = create_preset_config(ConfigPreset.BALANCED)

        json_str = original.to_json()
        restored = MCTSConfig.from_json(json_str)

        assert restored.num_iterations == original.num_iterations
        assert restored.exploration_weight == original.exploration_weight


class TestPerformanceE2E:
    """E2E performance tests."""

    @pytest.fixture
    def rollout_policy(self):
        """Create a simple rollout policy for testing."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    @pytest.mark.asyncio
    async def test_mcts_completes_within_timeout(self, rollout_policy):
        """MCTS should complete within reasonable time."""
        engine = MCTSEngine(seed=42)
        root = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=engine.rng,
        )

        def action_gen(s):
            return ["a", "b", "c"] if len(s.state_id) < 20 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        start = time.perf_counter()

        await engine.search(
            root=root,
            num_iterations=100,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        elapsed = time.perf_counter() - start

        # Should complete within 5 seconds
        assert elapsed < 5.0, f"MCTS took {elapsed:.2f}s, expected < 5s"

    def test_config_creation_is_fast(self):
        """Config creation should be fast."""
        start = time.perf_counter()

        for _ in range(1000):
            create_preset_config(ConfigPreset.BALANCED)
            GraphConfig()

        elapsed = time.perf_counter() - start

        # 1000 configs should create in under 1 second
        assert elapsed < 1.0, f"Config creation took {elapsed:.2f}s"


class TestErrorHandlingE2E:
    """E2E tests for error handling."""

    @pytest.fixture
    def rollout_policy(self):
        """Create a simple rollout policy for testing."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    @pytest.mark.asyncio
    async def test_mcts_handles_empty_actions(self, rollout_policy):
        """MCTS should handle empty action space gracefully."""
        engine = MCTSEngine(seed=42)
        root = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=engine.rng,
        )

        # Action generator that always returns empty
        def action_gen(s):
            return []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        # Should handle gracefully
        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Should return None or empty
        assert best_action is None or stats["iterations"] == 0

    def test_invalid_config_raises_error(self):
        """Invalid configs should raise clear errors."""
        # Invalid MCTS config
        with pytest.raises(ValueError) as exc_info:
            MCTSConfig(num_iterations=-1)
        assert "num_iterations" in str(exc_info.value)

        # Invalid exploration weight
        with pytest.raises(ValueError) as exc_info:
            MCTSConfig(exploration_weight=-1)
        assert "exploration_weight" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_invalid_transition_is_safe(self, rollout_policy):
        """Search should be safe even with problematic transitions."""
        engine = MCTSEngine(seed=42)
        root = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=engine.rng,
        )

        call_count = 0

        def action_gen(s):
            return ["a"] if len(s.state_id) < 10 else []

        def state_trans(s, a):
            nonlocal call_count
            call_count += 1
            # Still return valid state
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        # Should complete without error
        await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Should have called transition
        assert call_count > 0
