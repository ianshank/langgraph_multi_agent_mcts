"""
MCTS (Monte Carlo Tree Search) module for multi-agent framework.

Provides deterministic, testable MCTS with:
- Progressive widening for controlled branching
- Simulation result caching
- Configurable selection and rollout policies
- Experiment tracking and analysis
"""

from .config import MCTSConfig, create_preset_config
from .core import MCTSEngine, MCTSNode
from .experiments import ExperimentResult, ExperimentTracker
from .policies import RolloutPolicy, SelectionPolicy, ucb1

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
