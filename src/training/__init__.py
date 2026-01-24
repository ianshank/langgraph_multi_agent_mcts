"""
Training Module for Multi-Agent MCTS.

Provides:
- Experiment tracking (Braintrust, W&B)
- Training pipelines
- Model evaluation with statistical rigor
- Artifact management
"""

from .evaluation_service import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationService,
    EvaluationStrategy,
    GameOutcome,
    GameResult,
    MetricsHistory,
    create_evaluation_service,
)
from .experiment_tracker import (
    BraintrustTracker,
    ExperimentConfig,
    TrainingMetrics,
    UnifiedExperimentTracker,
    WandBTracker,
)

__all__ = [
    # Experiment tracking
    "BraintrustTracker",
    "WandBTracker",
    "UnifiedExperimentTracker",
    "TrainingMetrics",
    "ExperimentConfig",
    # Evaluation service
    "EvaluationService",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationStrategy",
    "GameOutcome",
    "GameResult",
    "MetricsHistory",
    "create_evaluation_service",
]

__version__ = "1.0.0"
