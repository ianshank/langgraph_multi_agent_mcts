"""
Training Module for Multi-Agent MCTS.

Provides:
- Experiment tracking (Braintrust, W&B)
- Training pipelines
- Model evaluation
- Artifact management
"""

from .experiment_tracker import (
    BraintrustTracker,
    ExperimentConfig,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)

__all__ = [
    "BraintrustTracker",
    "WandBTracker",
    "UnifiedExperimentTracker",
    "TrainingMetrics",
    "ExperimentConfig",
]

__version__ = "1.0.0"
