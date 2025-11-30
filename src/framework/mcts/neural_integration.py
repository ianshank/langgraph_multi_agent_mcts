"""
Neural MCTS Integration Module.

Bridges the gap between NeuralMCTS (AlphaZero-style) and the LangGraph workflow.
Provides:
- NeuralRolloutPolicy: Uses policy-value network for intelligent rollouts
- NeuralMCTSAdapter: Adapts NeuralMCTS for use with MCTSEngine
- UnifiedMCTSConfig: Merges framework and training configs
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from .core import MCTSState
from .policies import RolloutPolicy


class PolicyValueNetworkProtocol(Protocol):
    """Protocol for policy-value networks."""

    def forward(self, x: Any) -> tuple[Any, Any]:
        """Forward pass returning (policy_logits, value)."""
        ...

    def eval(self) -> None:
        """Set model to evaluation mode."""
        ...


@dataclass
class NeuralMCTSConfig:
    """
    Unified configuration for Neural MCTS.

    Merges parameters from both framework/mcts/config.py and training/system_config.py
    to provide a single source of truth for neural-guided MCTS.
    """

    # Core search parameters (from framework config)
    num_simulations: int = 100
    exploration_weight: float = 1.414  # UCB1 constant (sqrt(2))

    # PUCT parameters (from training config - AlphaZero style)
    c_puct: float = 1.25
    use_puct: bool = True  # Use PUCT instead of UCB1 when neural network available

    # Dirichlet noise for exploration at root
    dirichlet_epsilon: float = 0.25
    dirichlet_alpha: float = 0.3
    add_root_noise: bool = True

    # Temperature for action selection
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_threshold: int = 30  # Step to switch to greedy

    # Virtual loss for parallel search
    virtual_loss: float = 3.0
    num_parallel_workers: int = 4

    # Progressive widening
    progressive_widening_k: float = 1.0
    progressive_widening_alpha: float = 0.5

    # Caching
    enable_cache: bool = True
    cache_size_limit: int = 10000

    # Tree limits
    max_tree_depth: int = 20
    max_rollout_depth: int = 10

    # Value bounds
    min_value: float = -1.0  # AlphaZero uses [-1, 1]
    max_value: float = 1.0

    # Network configuration
    network_device: str = "cpu"
    use_mixed_precision: bool = False

    # Seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.c_puct < 0:
            raise ValueError("c_puct must be >= 0")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

    def to_framework_config(self) -> dict[str, Any]:
        """Convert to framework MCTSConfig parameters."""
        return {
            "num_iterations": self.num_simulations,
            "seed": self.seed,
            "exploration_weight": self.exploration_weight,
            "progressive_widening_k": self.progressive_widening_k,
            "progressive_widening_alpha": self.progressive_widening_alpha,
            "max_rollout_depth": self.max_rollout_depth,
            "max_parallel_rollouts": self.num_parallel_workers,
            "cache_size_limit": self.cache_size_limit,
            "max_tree_depth": self.max_tree_depth,
        }


class NeuralRolloutPolicy(RolloutPolicy):
    """
    Rollout policy that uses a neural network for state evaluation.

    This bridges NeuralMCTS with the standard MCTSEngine by providing
    neural-guided rollout values instead of random/heuristic rollouts.
    """

    def __init__(
        self,
        policy_value_network: PolicyValueNetworkProtocol | None = None,
        config: NeuralMCTSConfig | None = None,
        state_encoder: Any | None = None,
        fallback_value: float = 0.5,
    ):
        """
        Initialize neural rollout policy.

        Args:
            policy_value_network: Neural network for (policy, value) prediction
            config: Neural MCTS configuration
            state_encoder: Function to encode MCTSState to tensor
            fallback_value: Value to return when network unavailable
        """
        if not TORCH_AVAILABLE:
            self._network = None
            self._device = "cpu"
        else:
            self._network = policy_value_network
            if self._network is not None:
                self._network.eval()

        self._config = config or NeuralMCTSConfig()
        self._state_encoder = state_encoder
        self._fallback_value = fallback_value
        self._device = self._config.network_device if TORCH_AVAILABLE else "cpu"

        # Evaluation cache
        self._cache: dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def evaluate(
        self,
        state: MCTSState,
        rng: np.random.Generator,
        max_depth: int = 10,
    ) -> float:
        """
        Evaluate state using neural network.

        Args:
            state: State to evaluate
            rng: Random number generator (for noise)
            max_depth: Maximum depth (unused for neural evaluation)

        Returns:
            Value estimate in [min_value, max_value]
        """
        # Check cache
        state_hash = state.to_hash_key()
        if self._config.enable_cache and state_hash in self._cache:
            self._cache_hits += 1
            cached_value = self._cache[state_hash]
            # Add small noise for exploration
            noise = rng.normal(0, 0.01)
            return float(np.clip(
                cached_value + noise,
                self._config.min_value,
                self._config.max_value
            ))

        self._cache_misses += 1

        # Compute value using network or fallback
        value: float

        # If no network available, use fallback
        if self._network is None or not TORCH_AVAILABLE:
            value = self._fallback_with_features(state, rng)
        else:
            # Encode state to tensor
            state_tensor = self._encode_state(state)
            if state_tensor is None:
                value = self._fallback_with_features(state, rng)
            else:
                # Run network inference
                try:
                    value = await self._neural_evaluate(state_tensor)
                except Exception:
                    # Fallback on any error
                    value = self._fallback_with_features(state, rng)

        # Cache result (both network and fallback values)
        if self._config.enable_cache:
            if len(self._cache) >= self._config.cache_size_limit:
                # Simple cache eviction: clear half
                keys_to_remove = list(self._cache.keys())[: len(self._cache) // 2]
                for key in keys_to_remove:
                    del self._cache[key]
            self._cache[state_hash] = value

        return value

    def _encode_state(self, state: MCTSState) -> Any | None:
        """Encode MCTSState to tensor for neural network."""
        if self._state_encoder is not None:
            try:
                return self._state_encoder(state)
            except Exception:
                return None

        # Default encoding: create feature vector from state features
        if not TORCH_AVAILABLE:
            return None

        features = state.features
        if not features:
            return None

        # Extract numeric features
        feature_values = []
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            elif isinstance(value, bool):
                feature_values.append(1.0 if value else 0.0)

        if not feature_values:
            return None

        # Create tensor
        tensor = torch.tensor(feature_values, dtype=torch.float32)
        return tensor.unsqueeze(0).to(self._device)

    async def _neural_evaluate(self, state_tensor: Any) -> float:
        """Run neural network evaluation."""
        if not TORCH_AVAILABLE or self._network is None:
            return self._fallback_value

        with torch.no_grad():
            if self._config.use_mixed_precision and torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"):
                    _, value = self._network(state_tensor)
            else:
                _, value = self._network(state_tensor)

            # Extract scalar value
            if hasattr(value, "item"):
                value = value.item()
            else:
                value = float(value)

        # Normalize to configured bounds
        return float(np.clip(value, self._config.min_value, self._config.max_value))

    def _fallback_with_features(self, state: MCTSState, rng: np.random.Generator) -> float:
        """Compute fallback value using state features."""
        base_value = self._fallback_value
        features = state.features

        # Incorporate any confidence scores in features
        confidence_boost = 0.0
        for key, value in features.items():
            if "confidence" in key.lower() and isinstance(value, (int, float)):
                confidence_boost += float(value) * 0.1
            elif "quality" in key.lower() and isinstance(value, (int, float)):
                confidence_boost += float(value) * 0.1

        # Add controlled randomness
        noise = rng.normal(0, 0.1)

        value = base_value + confidence_boost + noise
        return float(np.clip(value, self._config.min_value, self._config.max_value))

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
        }

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


@dataclass
class NeuralMCTSAdapter:
    """
    Adapter that wraps NeuralMCTS for use with the graph workflow.

    Provides a clean interface between the AlphaZero-style NeuralMCTS
    and the LangGraph state machine.
    """

    config: NeuralMCTSConfig = field(default_factory=NeuralMCTSConfig)
    policy_value_network: Any | None = None
    state_encoder: Any | None = None

    _rollout_policy: NeuralRolloutPolicy | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self) -> None:
        """Initialize the adapter with neural network."""
        self._rollout_policy = NeuralRolloutPolicy(
            policy_value_network=self.policy_value_network,
            config=self.config,
            state_encoder=self.state_encoder,
        )
        self._initialized = True

    @property
    def rollout_policy(self) -> NeuralRolloutPolicy:
        """Get the neural rollout policy."""
        if not self._initialized:
            self.initialize()
        assert self._rollout_policy is not None
        return self._rollout_policy

    def get_mcts_engine_kwargs(self) -> dict[str, Any]:
        """Get kwargs for MCTSEngine initialization."""
        return {
            "seed": self.config.seed,
            "exploration_weight": self.config.exploration_weight,
            "progressive_widening_k": self.config.progressive_widening_k,
            "progressive_widening_alpha": self.config.progressive_widening_alpha,
            "max_parallel_rollouts": self.config.num_parallel_workers,
            "cache_size_limit": self.config.cache_size_limit,
        }

    async def evaluate_state(
        self,
        state: MCTSState,
        rng: np.random.Generator,
    ) -> tuple[dict[str, float], float]:
        """
        Evaluate a state using neural network.

        Returns both policy (action probabilities) and value.

        Args:
            state: State to evaluate
            rng: Random number generator

        Returns:
            (action_probs, value) tuple
        """
        value = await self.rollout_policy.evaluate(state, rng)

        # For policy, we need the full network output
        # This is a simplified version - full implementation would use NeuralMCTS.search()
        action_probs: dict[str, float] = {}

        return action_probs, value


def create_neural_mcts_adapter(
    network: Any | None = None,
    config: NeuralMCTSConfig | None = None,
    **kwargs: Any,
) -> NeuralMCTSAdapter:
    """
    Factory function to create a NeuralMCTSAdapter.

    Args:
        network: Policy-value network (optional)
        config: Configuration (optional, uses defaults)
        **kwargs: Additional config overrides

    Returns:
        Configured NeuralMCTSAdapter
    """
    if config is None:
        config = NeuralMCTSConfig(**kwargs) if kwargs else NeuralMCTSConfig()
    elif kwargs:
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    adapter = NeuralMCTSAdapter(
        config=config,
        policy_value_network=network,
    )
    adapter.initialize()
    return adapter


# Preset configurations
def get_fast_neural_config() -> NeuralMCTSConfig:
    """Fast neural MCTS for quick evaluations."""
    return NeuralMCTSConfig(
        num_simulations=50,
        c_puct=1.5,
        add_root_noise=False,
        max_rollout_depth=5,
        cache_size_limit=1000,
    )


def get_balanced_neural_config() -> NeuralMCTSConfig:
    """Balanced neural MCTS for typical use."""
    return NeuralMCTSConfig(
        num_simulations=200,
        c_puct=1.25,
        add_root_noise=True,
        max_rollout_depth=10,
        cache_size_limit=10000,
    )


def get_thorough_neural_config() -> NeuralMCTSConfig:
    """Thorough neural MCTS for high-stakes decisions."""
    return NeuralMCTSConfig(
        num_simulations=800,
        c_puct=1.0,
        add_root_noise=True,
        dirichlet_epsilon=0.15,  # Less noise for more exploitation
        max_rollout_depth=20,
        cache_size_limit=50000,
        num_parallel_workers=8,
    )


def get_alphazero_config() -> NeuralMCTSConfig:
    """AlphaZero-style configuration (1600 simulations)."""
    return NeuralMCTSConfig(
        num_simulations=1600,
        c_puct=1.25,
        add_root_noise=True,
        dirichlet_epsilon=0.25,
        dirichlet_alpha=0.3,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=30,
        virtual_loss=3.0,
        num_parallel_workers=8,
        max_rollout_depth=30,
        cache_size_limit=100000,
    )
