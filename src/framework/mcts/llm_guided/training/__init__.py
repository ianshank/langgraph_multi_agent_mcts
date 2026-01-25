"""
Training Module for LLM-Guided MCTS - Phase 2.

Provides components for training neural networks from MCTS search data:
- MCTSDataset: PyTorch Dataset for training data
- PolicyNetwork: Network for action selection
- ValueNetwork: Network for state value estimation
- DistillationTrainer: Trainer for knowledge distillation from LLM to neural networks
- TrainingMetrics: Metrics tracking during training
"""

from .dataset import (
    MCTSDataset,
    MCTSDatasetConfig,
    TrainingBatch,
    create_dataloaders,
    load_training_data,
)
from .metrics import (
    EvaluationMetrics,
    TrainingMetrics,
    compute_policy_accuracy,
    compute_value_mse,
)
from .networks import (
    CodeEncoder,
    CodeEncoderConfig,
    PolicyNetwork,
    PolicyNetworkConfig,
    ValueNetwork,
    ValueNetworkConfig,
    create_policy_network,
    create_value_network,
)
from .trainer import (
    DistillationTrainer,
    DistillationTrainerConfig,
    TrainingCallback,
    TrainingCheckpoint,
    create_trainer,
)

__all__ = [
    # Dataset
    "MCTSDataset",
    "MCTSDatasetConfig",
    "TrainingBatch",
    "create_dataloaders",
    "load_training_data",
    # Networks
    "CodeEncoder",
    "CodeEncoderConfig",
    "PolicyNetwork",
    "PolicyNetworkConfig",
    "ValueNetwork",
    "ValueNetworkConfig",
    "create_policy_network",
    "create_value_network",
    # Trainer
    "DistillationTrainer",
    "DistillationTrainerConfig",
    "TrainingCallback",
    "TrainingCheckpoint",
    "create_trainer",
    # Metrics
    "TrainingMetrics",
    "EvaluationMetrics",
    "compute_policy_accuracy",
    "compute_value_mse",
]
