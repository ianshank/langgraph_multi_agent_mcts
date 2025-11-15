"""
MCTS (Monte Carlo Tree Search) module for multi-agent framework.

Provides deterministic, testable MCTS with:
- Progressive widening for controlled branching
- Simulation result caching
- Configurable selection and rollout policies
- Experiment tracking and analysis
"""

from .core import MCTSNode, MCTSEngine
from .policies import ucb1, RolloutPolicy, SelectionPolicy
from .config import MCTSConfig, create_preset_config
from .experiments import ExperimentTracker, ExperimentResult

__all__ = [
    "MCTSNode",
    "MCTSEngine",
    "ucb1",
    "RolloutPolicy",
    "SelectionPolicy",
    "MCTSConfig",
    "create_preset_config",
    "ExperimentTracker",
    "ExperimentResult",
]
