"""
Training Metrics for MCTS Knowledge Distillation.

Provides metrics tracking and computation for:
- Policy distillation (KL divergence, accuracy)
- Value prediction (MSE, correlation)
- Training progress (loss curves, learning rates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.observability.logging import get_structured_logger

# Optional PyTorch imports
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

logger = get_structured_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    # Loss values
    policy_loss: float = 0.0
    """Policy distillation loss (KL divergence)."""

    value_loss: float = 0.0
    """Value prediction loss (MSE)."""

    total_loss: float = 0.0
    """Combined total loss."""

    # Policy metrics
    policy_accuracy: float = 0.0
    """Top-1 accuracy for action selection."""

    policy_top3_accuracy: float = 0.0
    """Top-3 accuracy for action selection."""

    policy_kl_divergence: float = 0.0
    """KL divergence from target policy."""

    policy_entropy: float = 0.0
    """Entropy of predicted policy."""

    # Value metrics
    value_mse: float = 0.0
    """Mean squared error for value prediction."""

    value_mae: float = 0.0
    """Mean absolute error for value prediction."""

    value_correlation: float = 0.0
    """Pearson correlation with target values."""

    # Training info
    learning_rate: float = 0.0
    """Current learning rate."""

    gradient_norm: float = 0.0
    """Gradient norm before clipping."""

    num_samples: int = 0
    """Number of samples processed."""

    epoch: int = 0
    """Current epoch."""

    step: int = 0
    """Current global step."""

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "policy_accuracy": self.policy_accuracy,
            "policy_top3_accuracy": self.policy_top3_accuracy,
            "policy_kl_divergence": self.policy_kl_divergence,
            "policy_entropy": self.policy_entropy,
            "value_mse": self.value_mse,
            "value_mae": self.value_mae,
            "value_correlation": self.value_correlation,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "num_samples": self.num_samples,
            "epoch": self.epoch,
            "step": self.step,
        }


@dataclass
class EvaluationMetrics:
    """Metrics from model evaluation."""

    # Overall performance
    policy_accuracy: float = 0.0
    """Top-1 accuracy on validation set."""

    value_mse: float = 0.0
    """MSE on validation set."""

    # Per-depth performance
    accuracy_by_depth: dict[int, float] = field(default_factory=dict)
    """Policy accuracy broken down by tree depth."""

    mse_by_depth: dict[int, float] = field(default_factory=dict)
    """Value MSE broken down by tree depth."""

    # Distribution analysis
    predicted_value_mean: float = 0.0
    """Mean of predicted values."""

    predicted_value_std: float = 0.0
    """Std of predicted values."""

    target_value_mean: float = 0.0
    """Mean of target values."""

    target_value_std: float = 0.0
    """Std of target values."""

    # Sample counts
    total_samples: int = 0
    """Total samples evaluated."""

    samples_by_depth: dict[int, int] = field(default_factory=dict)
    """Sample count by depth."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "policy_accuracy": self.policy_accuracy,
            "value_mse": self.value_mse,
            "accuracy_by_depth": self.accuracy_by_depth,
            "mse_by_depth": self.mse_by_depth,
            "predicted_value_mean": self.predicted_value_mean,
            "predicted_value_std": self.predicted_value_std,
            "target_value_mean": self.target_value_mean,
            "target_value_std": self.target_value_std,
            "total_samples": self.total_samples,
            "samples_by_depth": self.samples_by_depth,
        }


class MetricsAccumulator:
    """Accumulates metrics over batches for epoch-level reporting."""

    def __init__(self):
        """Initialize accumulator."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated values."""
        self._policy_loss_sum = 0.0
        self._value_loss_sum = 0.0
        self._total_loss_sum = 0.0
        self._policy_correct = 0
        self._policy_top3_correct = 0
        self._value_squared_error = 0.0
        self._value_absolute_error = 0.0
        self._predicted_values: list[float] = []
        self._target_values: list[float] = []
        self._num_samples = 0
        self._num_batches = 0

    def update(
        self,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        policy_correct: int,
        policy_top3_correct: int,
        value_predictions: Any,  # torch.Tensor or array
        value_targets: Any,  # torch.Tensor or array
        batch_size: int,
    ) -> None:
        """Update with batch results."""
        self._policy_loss_sum += policy_loss * batch_size
        self._value_loss_sum += value_loss * batch_size
        self._total_loss_sum += total_loss * batch_size
        self._policy_correct += policy_correct
        self._policy_top3_correct += policy_top3_correct
        self._num_samples += batch_size
        self._num_batches += 1

        # Handle tensor conversion
        if _TORCH_AVAILABLE and torch is not None:
            if isinstance(value_predictions, torch.Tensor):
                value_predictions = value_predictions.detach().cpu().numpy()
            if isinstance(value_targets, torch.Tensor):
                value_targets = value_targets.detach().cpu().numpy()

        self._predicted_values.extend(value_predictions.flatten().tolist())
        self._target_values.extend(value_targets.flatten().tolist())

        # Compute running errors
        diff = np.array(value_predictions) - np.array(value_targets)
        self._value_squared_error += np.sum(diff**2)
        self._value_absolute_error += np.sum(np.abs(diff))

    def compute(
        self, learning_rate: float = 0.0, gradient_norm: float = 0.0, epoch: int = 0, step: int = 0
    ) -> TrainingMetrics:
        """Compute final metrics."""
        if self._num_samples == 0:
            return TrainingMetrics()

        n = self._num_samples

        # Compute correlation
        pred = np.array(self._predicted_values)
        targ = np.array(self._target_values)
        if len(pred) > 1 and np.std(pred) > 1e-8 and np.std(targ) > 1e-8:
            correlation = float(np.corrcoef(pred, targ)[0, 1])
        else:
            correlation = 0.0

        return TrainingMetrics(
            policy_loss=self._policy_loss_sum / n,
            value_loss=self._value_loss_sum / n,
            total_loss=self._total_loss_sum / n,
            policy_accuracy=self._policy_correct / n,
            policy_top3_accuracy=self._policy_top3_correct / n,
            value_mse=self._value_squared_error / n,
            value_mae=self._value_absolute_error / n,
            value_correlation=correlation,
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            num_samples=n,
            epoch=epoch,
            step=step,
        )


def compute_policy_accuracy(
    log_probs: Any,  # torch.Tensor [batch, actions]
    targets: Any,  # torch.Tensor [batch, actions]
    action_mask: Any | None = None,  # torch.Tensor [batch, actions]
    k: int = 1,
) -> float:
    """
    Compute top-k accuracy for policy predictions.

    Args:
        log_probs: Predicted log probabilities [batch, actions]
        targets: Target probabilities [batch, actions]
        action_mask: Valid action mask [batch, actions]
        k: Top-k for accuracy computation

    Returns:
        Top-k accuracy as float
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    # Get top-k predictions
    _, pred_topk = torch.topk(log_probs, k=min(k, log_probs.size(1)), dim=-1)

    # Get target action (highest probability)
    target_actions = targets.argmax(dim=-1)

    # Check if target is in top-k predictions
    correct = (pred_topk == target_actions.unsqueeze(-1)).any(dim=-1)

    # Apply mask if provided
    if action_mask is not None:
        valid = action_mask.sum(dim=-1) > 0
        correct = correct & valid
        num_valid = valid.sum().item()
        if num_valid == 0:
            return 0.0
        return float(correct.sum().item() / num_valid)

    return float(correct.float().mean().item())


def compute_value_mse(
    predictions: Any,  # torch.Tensor [batch]
    targets: Any,  # torch.Tensor [batch]
) -> float:
    """
    Compute mean squared error for value predictions.

    Args:
        predictions: Predicted values [batch]
        targets: Target values [batch]

    Returns:
        MSE as float
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    return float(((predictions - targets) ** 2).mean().item())


def compute_kl_divergence(
    log_probs: Any,  # torch.Tensor [batch, actions]
    target_probs: Any,  # torch.Tensor [batch, actions]
    action_mask: Any | None = None,  # torch.Tensor [batch, actions]
) -> float:
    """
    Compute KL divergence between predicted and target policies.

    Args:
        log_probs: Predicted log probabilities [batch, actions]
        target_probs: Target probabilities [batch, actions]
        action_mask: Valid action mask [batch, actions]

    Returns:
        KL divergence as float
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    # KL(target || pred) = sum(target * log(target / pred))
    # = sum(target * log(target)) - sum(target * log(pred))
    # = -H(target) - sum(target * log_pred)

    # Compute cross-entropy term
    cross_entropy = -(target_probs * log_probs)

    if action_mask is not None:
        cross_entropy = cross_entropy * action_mask
        # Normalize by number of valid actions
        cross_entropy = cross_entropy.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-8)
    else:
        cross_entropy = cross_entropy.sum(dim=-1)

    # Target entropy
    target_log = torch.log(target_probs.clamp(min=1e-8))
    target_entropy = -(target_probs * target_log)

    if action_mask is not None:
        target_entropy = target_entropy * action_mask
        target_entropy = target_entropy.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-8)
    else:
        target_entropy = target_entropy.sum(dim=-1)

    # KL = cross_entropy - target_entropy
    kl = cross_entropy - target_entropy

    return float(kl.mean().item())


def compute_policy_entropy(
    log_probs: Any,  # torch.Tensor [batch, actions]
    action_mask: Any | None = None,  # torch.Tensor [batch, actions]
) -> float:
    """
    Compute entropy of predicted policy.

    Args:
        log_probs: Predicted log probabilities [batch, actions]
        action_mask: Valid action mask [batch, actions]

    Returns:
        Entropy as float
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")

    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs)

    if action_mask is not None:
        entropy = entropy * action_mask
        entropy = entropy.sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1e-8)
    else:
        entropy = entropy.sum(dim=-1)

    return float(entropy.mean().item())
