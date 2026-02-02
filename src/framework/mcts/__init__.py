"""
MCTS (Monte Carlo Tree Search) module for multi-agent framework.

Provides deterministic, testable MCTS with:
- Progressive widening for controlled branching
- Simulation result caching
- Configurable selection and rollout policies
- Experiment tracking and analysis
- Neural network integration (PUCT, neural rollout)
- Domain adapters for state conversion
"""

from .config import MCTSConfig, create_preset_config
from .core import MCTSEngine, MCTSNode

# Domain adapters
from .domain_adapters import (
    BaseDomainAdapter,
    FeatureAdapterConfig,
    FeatureStateAdapter,
    GridAdapterConfig,
    GridStateAdapter,
    TextAdapterConfig,
    TextStateAdapter,
    create_domain_adapter,
)
from .experiments import ExperimentResult, ExperimentTracker

# Neural policies (optional PyTorch dependency)
from .neural_policies import (
    NeuralPolicyConfig,
    NeuralRolloutPolicy,
    PriorsManager,
    create_neural_rollout_policy,
    is_torch_available,
    puct,
    puct_with_virtual_loss,
    select_child_puct,
)
from .policies import RolloutPolicy, SelectionPolicy, ucb1

__all__ = [
    # Core MCTS
    "MCTSNode",
    "MCTSEngine",
    "ucb1",
    "RolloutPolicy",
    "SelectionPolicy",
    "MCTSConfig",
    "create_preset_config",
    "ExperimentTracker",
    "ExperimentResult",
    # Neural policies
    "puct",
    "puct_with_virtual_loss",
    "NeuralPolicyConfig",
    "NeuralRolloutPolicy",
    "PriorsManager",
    "create_neural_rollout_policy",
    "select_child_puct",
    "is_torch_available",
    # Domain adapters
    "BaseDomainAdapter",
    "GridAdapterConfig",
    "GridStateAdapter",
    "FeatureAdapterConfig",
    "FeatureStateAdapter",
    "TextAdapterConfig",
    "TextStateAdapter",
    "create_domain_adapter",
]
