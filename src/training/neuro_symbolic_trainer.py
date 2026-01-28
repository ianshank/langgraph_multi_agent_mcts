"""
Neuro-Symbolic Training Module.

Provides specialized training loops for:
- PonderNet-style adaptive computation (HRM with trainable halting)
- Neural Planner (strategy generation)
- Joint HRM + Planner training

Supports curriculum learning for gradual transition from fixed to adaptive computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import get_linear_schedule_with_warmup

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Import HRM components
from src.agents.hrm_agent import (
    CurriculumPonderScheduler,
    HRMAgent,
    HRMConfig,
    PonderNetLoss,
    create_hrm_agent,
)

# Import Neural Planner components
from src.framework.adapters import (
    NeuralPlanner,
    NeuralPlannerLoss,
    NeuroSymbolicAdapter,
    create_neural_planner,
)

logger = logging.getLogger(__name__)


@dataclass
class NeuroSymbolicConfig:
    """Configuration for Neuro-Symbolic training."""

    # PonderNet configuration
    ponder_lambda: float = 0.01  # KL divergence weight
    geometric_prior_p: float = 0.5  # Geometric prior parameter
    max_ponder_steps: int = 16  # Maximum pondering steps

    # Neural Planner configuration
    planner_hidden_dim: int = 512
    planner_num_strategies: int = 5
    planner_max_steps: int = 8
    planner_num_layers: int = 4
    planner_num_heads: int = 8

    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0

    # Curriculum learning
    curriculum_warmup_epochs: int = 5
    curriculum_transition_epochs: int = 20

    # Loss weights
    task_weight: float = 1.0
    ponder_weight: float = 0.01
    planner_weight: float = 1.0
    consistency_weight: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "training/models/checkpoints/neuro_symbolic"
    save_every_n_epochs: int = 2


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    step: int
    loss: float
    task_loss: float
    ponder_loss: float
    planner_loss: float
    learning_rate: float
    expected_steps: float
    strategy_accuracy: float
    additional_metrics: dict[str, float]


class NeuroSymbolicDataset(Dataset):
    """
    Dataset for Neuro-Symbolic training.

    Provides:
    - Query embeddings
    - Target decomposition strategies
    - Target confidence scores
    - Target number of steps
    """

    def __init__(
        self,
        queries: list[str],
        target_strategies: list[list[int]],
        target_confidences: list[list[float]] | None = None,
        embedding_dim: int = 512,
        max_seq_len: int = 32,
        max_steps: int = 8,
    ):
        """
        Initialize dataset.

        Args:
            queries: List of query strings
            target_strategies: List of strategy sequences (indices)
            target_confidences: Optional confidence scores for each step
            embedding_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            max_steps: Maximum planning steps
        """
        self.queries = queries
        self.target_strategies = target_strategies
        self.target_confidences = target_confidences
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.max_steps = max_steps

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query = self.queries[idx]
        strategies = self.target_strategies[idx]

        # Create mock embedding (in practice, use real embeddings)
        seq_len = min(len(query.split()) + 2, self.max_seq_len)
        embedding = torch.randn(seq_len, self.embedding_dim)

        # Pad strategies to max_steps
        padded_strategies = strategies[: self.max_steps]
        padded_strategies += [0] * (self.max_steps - len(padded_strategies))
        strategies_tensor = torch.tensor(padded_strategies, dtype=torch.long)

        # Confidences
        if self.target_confidences is not None:
            confidences = self.target_confidences[idx][: self.max_steps]
            confidences += [0.5] * (self.max_steps - len(confidences))
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
        else:
            confidences_tensor = torch.ones(self.max_steps) * 0.5

        return {
            "query": query,
            "embedding": embedding,
            "target_strategies": strategies_tensor,
            "target_confidences": confidences_tensor,
            "num_steps": torch.tensor(len(strategies), dtype=torch.long),
        }


def collate_neuro_symbolic(batch: list[dict]) -> dict[str, Any]:
    """Collate function for NeuroSymbolicDataset."""
    queries = [item["query"] for item in batch]

    # Pad embeddings to same length
    max_len = max(item["embedding"].shape[0] for item in batch)
    embeddings = []
    for item in batch:
        emb = item["embedding"]
        if emb.shape[0] < max_len:
            padding = torch.zeros(max_len - emb.shape[0], emb.shape[1])
            emb = torch.cat([emb, padding], dim=0)
        embeddings.append(emb)

    return {
        "queries": queries,
        "embeddings": torch.stack(embeddings),
        "target_strategies": torch.stack([item["target_strategies"] for item in batch]),
        "target_confidences": torch.stack([item["target_confidences"] for item in batch]),
        "num_steps": torch.stack([item["num_steps"] for item in batch]),
    }


class NeuroSymbolicTrainer:
    """
    Trainer for Neuro-Symbolic components.

    Handles training of:
    - HRM Agent with PonderNet (adaptive computation)
    - Neural Planner (strategy generation)
    - Joint training with curriculum learning
    """

    def __init__(
        self,
        config: NeuroSymbolicConfig,
        hrm_config: HRMConfig | None = None,
        device: str | None = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Neuro-symbolic training configuration
            hrm_config: HRM model configuration
            device: Device to use (auto-detected if None)
        """
        self.config = config
        self.hrm_config = hrm_config or HRMConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Models
        self.hrm_agent: HRMAgent | None = None
        self.neural_planner: NeuralPlanner | None = None
        self.adapter: NeuroSymbolicAdapter | None = None

        # Optimizers
        self.hrm_optimizer = None
        self.planner_optimizer = None

        # Schedulers
        self.hrm_scheduler = None
        self.planner_scheduler = None
        self.curriculum_scheduler: CurriculumPonderScheduler | None = None

        # Loss functions
        self.ponder_loss_fn: PonderNetLoss | None = None
        self.planner_loss_fn: NeuralPlannerLoss | None = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.training_history = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized NeuroSymbolicTrainer on device: {self.device}")

    def build_models(self) -> None:
        """Build HRM and Neural Planner models."""
        logger.info("Building Neuro-Symbolic models...")

        # Build HRM Agent with PonderNet
        self.hrm_agent = create_hrm_agent(
            config=self.hrm_config,
            device=self.device,
            use_ponder_net=True,
            ponder_lambda=self.config.ponder_lambda,
            geometric_prior_p=self.config.geometric_prior_p,
        )

        # Build Neural Planner
        self.neural_planner = create_neural_planner(
            hidden_dim=self.config.planner_hidden_dim,
            num_strategies=self.config.planner_num_strategies,
            max_steps=self.config.planner_max_steps,
            num_layers=self.config.planner_num_layers,
            num_heads=self.config.planner_num_heads,
            device=self.device,
        )

        # Build adapter
        self.adapter = NeuroSymbolicAdapter(
            neural_agent=self.hrm_agent,
            neural_planner=self.neural_planner,
            device=self.device,
            use_planner=True,
        )

        # Build loss functions
        self.ponder_loss_fn = PonderNetLoss(
            task_weight=self.config.task_weight,
            kl_weight=self.config.ponder_weight,
            consistency_weight=self.config.consistency_weight,
        )

        self.planner_loss_fn = NeuralPlannerLoss(
            strategy_weight=self.config.planner_weight,
            confidence_weight=0.5,
            diversity_weight=0.1,
        )

        logger.info(
            f"HRM parameters: {self.hrm_agent.get_parameter_count():,}, "
            f"Planner parameters: {sum(p.numel() for p in self.neural_planner.parameters()):,}"
        )

    def setup_optimizers(self, total_steps: int | None = None) -> None:
        """Setup optimizers and learning rate schedulers."""
        # HRM optimizer
        self.hrm_optimizer = AdamW(
            self.hrm_agent.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Planner optimizer
        self.planner_optimizer = AdamW(
            self.neural_planner.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Schedulers
        total_steps = total_steps or self.config.epochs * 1000
        warmup_steps = int(total_steps * 0.1)

        if HAS_TRANSFORMERS:
            self.hrm_scheduler = get_linear_schedule_with_warmup(
                self.hrm_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            self.planner_scheduler = get_linear_schedule_with_warmup(
                self.planner_optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.hrm_scheduler = CosineAnnealingWarmRestarts(self.hrm_optimizer, T_0=100)
            self.planner_scheduler = CosineAnnealingWarmRestarts(self.planner_optimizer, T_0=100)

        # Curriculum scheduler for PonderNet
        self.curriculum_scheduler = CurriculumPonderScheduler(
            warmup_epochs=self.config.curriculum_warmup_epochs,
            curriculum_epochs=self.config.curriculum_transition_epochs,
            initial_lambda_p=0.0,
            final_lambda_p=self.config.ponder_lambda,
            initial_max_steps=3,
            final_max_steps=self.config.max_ponder_steps,
        )

        logger.info("Optimizers and schedulers initialized")

    def train_epoch(
        self,
        dataloader: DataLoader,
        train_hrm: bool = True,
        train_planner: bool = True,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            train_hrm: Whether to train HRM
            train_planner: Whether to train planner

        Returns:
            Dictionary of epoch metrics
        """
        self.hrm_agent.train()
        self.neural_planner.train()

        total_loss = 0.0
        total_hrm_loss = 0.0
        total_planner_loss = 0.0
        total_ponder_cost = 0.0
        total_strategy_acc = 0.0
        num_batches = 0

        accumulation_steps = self.config.gradient_accumulation_steps

        # Update curriculum
        if self.curriculum_scheduler is not None:
            self.curriculum_scheduler.update_ponder_net(self.hrm_agent.ponder_net)
            current_lambda = self.curriculum_scheduler.get_lambda_p()
            current_max_steps = self.curriculum_scheduler.get_max_steps()
            logger.debug(f"Curriculum: lambda_p={current_lambda:.4f}, max_steps={current_max_steps}")

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            embeddings = batch["embeddings"].to(self.device)
            target_strategies = batch["target_strategies"].to(self.device)
            target_confidences = batch["target_confidences"].to(self.device)

            # === Train HRM with PonderNet ===
            hrm_loss = torch.tensor(0.0, device=self.device)
            if train_hrm:
                # Forward pass with ponder output
                hrm_output = self.hrm_agent(
                    embeddings,
                    max_steps=self.curriculum_scheduler.get_max_steps() if self.curriculum_scheduler else None,
                    return_decomposition=True,
                    return_ponder_output=True,
                )

                # Compute task predictions (simplified: use final state)
                # In practice, you'd have a task-specific head
                predictions = hrm_output.final_state.mean(dim=1)  # [batch, h_dim]
                targets = embeddings.mean(dim=1)  # Mock targets

                # Compute loss
                hrm_loss, hrm_loss_dict = self.ponder_loss_fn(
                    hrm_output,
                    predictions,
                    targets,
                    task_loss_fn=nn.MSELoss(),
                )
                hrm_loss = hrm_loss / accumulation_steps

                # Backward
                hrm_loss.backward()

                total_hrm_loss += hrm_loss.item() * accumulation_steps
                if hrm_output.ponder_output is not None:
                    total_ponder_cost += hrm_output.ponder_output.expected_steps

            # === Train Neural Planner ===
            planner_loss = torch.tensor(0.0, device=self.device)
            if train_planner:
                # Forward pass
                planner_output = self.neural_planner(embeddings)

                # Compute loss
                planner_loss, planner_loss_dict = self.planner_loss_fn(
                    planner_output,
                    target_strategies,
                    target_confidences,
                )
                planner_loss = planner_loss / accumulation_steps

                # Backward
                planner_loss.backward()

                total_planner_loss += planner_loss.item() * accumulation_steps
                total_strategy_acc += self.neural_planner.get_strategy_accuracy(planner_output, target_strategies)

            # Combined loss for logging
            total_loss += (hrm_loss.item() + planner_loss.item()) * accumulation_steps
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if train_hrm:
                    torch.nn.utils.clip_grad_norm_(
                        self.hrm_agent.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.hrm_optimizer.step()
                    if self.hrm_scheduler:
                        self.hrm_scheduler.step()
                    self.hrm_optimizer.zero_grad()

                if train_planner:
                    torch.nn.utils.clip_grad_norm_(
                        self.neural_planner.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.planner_optimizer.step()
                    if self.planner_scheduler:
                        self.planner_scheduler.step()
                    self.planner_optimizer.zero_grad()

                self.global_step += 1

            # Logging
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {(hrm_loss.item() + planner_loss.item()) * accumulation_steps:.4f}"
                )

        # Compute averages
        metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "hrm_loss": total_hrm_loss / num_batches if num_batches > 0 else 0.0,
            "planner_loss": total_planner_loss / num_batches if num_batches > 0 else 0.0,
            "ponder_cost": total_ponder_cost / num_batches if num_batches > 0 else 0.0,
            "strategy_accuracy": total_strategy_acc / num_batches if num_batches > 0 else 0.0,
        }

        return metrics

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """
        Evaluate models on validation data.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.hrm_agent.eval()
        self.neural_planner.eval()

        total_loss = 0.0
        total_strategy_acc = 0.0
        total_expected_steps = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch["embeddings"].to(self.device)
                target_strategies = batch["target_strategies"].to(self.device)
                target_confidences = batch["target_confidences"].to(self.device)

                # HRM evaluation
                hrm_output = self.hrm_agent(
                    embeddings,
                    return_decomposition=True,
                    return_ponder_output=True,
                )

                if hrm_output.ponder_output is not None:
                    total_expected_steps += hrm_output.ponder_output.expected_steps

                # Planner evaluation
                planner_output = self.neural_planner(embeddings)
                planner_loss, _ = self.planner_loss_fn(
                    planner_output,
                    target_strategies,
                    target_confidences,
                )
                total_loss += planner_loss.item()

                # Strategy accuracy
                total_strategy_acc += self.neural_planner.get_strategy_accuracy(planner_output, target_strategies)

                num_batches += 1

        metrics = {
            "val_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "val_strategy_accuracy": total_strategy_acc / num_batches if num_batches > 0 else 0.0,
            "val_expected_steps": total_expected_steps / num_batches if num_batches > 0 else 0.0,
        }

        return metrics

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, Any]:
        """
        Run full training loop.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            num_epochs: Number of epochs (defaults to config)

        Returns:
            Training results
        """
        num_epochs = num_epochs or self.config.epochs

        # Build models if not already built
        if self.hrm_agent is None:
            self.build_models()

        # Setup optimizers
        total_steps = num_epochs * len(train_dataloader)
        self.setup_optimizers(total_steps)

        logger.info(f"Starting training for {num_epochs} epochs")

        results = {
            "train_history": [],
            "val_history": [],
            "best_epoch": 0,
            "best_metric": float("inf"),
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_dataloader)
            results["train_history"].append(train_metrics)
            logger.info(f"Train metrics: {train_metrics}")

            # Validate
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                results["val_history"].append(val_metrics)
                logger.info(f"Val metrics: {val_metrics}")

                # Track best
                if val_metrics["val_loss"] < results["best_metric"]:
                    results["best_metric"] = val_metrics["val_loss"]
                    results["best_epoch"] = epoch
                    self.save_checkpoint("best")

            # Curriculum step
            if self.curriculum_scheduler is not None:
                self.curriculum_scheduler.step()

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        # Final checkpoint
        self.save_checkpoint("final")

        logger.info(f"Training complete. Best epoch: {results['best_epoch']}")
        return results

    def save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "hrm_state_dict": self.hrm_agent.state_dict(),
            "planner_state_dict": self.neural_planner.state_dict(),
            "hrm_optimizer_state_dict": self.hrm_optimizer.state_dict() if self.hrm_optimizer else None,
            "planner_optimizer_state_dict": self.planner_optimizer.state_dict() if self.planner_optimizer else None,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config.__dict__,
            "hrm_config": self.hrm_config.__dict__,
        }

        if self.curriculum_scheduler is not None:
            checkpoint["curriculum_state"] = self.curriculum_scheduler.state_dict()

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        # Build models if needed
        if self.hrm_agent is None:
            self.build_models()

        self.hrm_agent.load_state_dict(checkpoint["hrm_state_dict"])
        self.neural_planner.load_state_dict(checkpoint["planner_state_dict"])

        if checkpoint.get("hrm_optimizer_state_dict") and self.hrm_optimizer:
            self.hrm_optimizer.load_state_dict(checkpoint["hrm_optimizer_state_dict"])
        if checkpoint.get("planner_optimizer_state_dict") and self.planner_optimizer:
            self.planner_optimizer.load_state_dict(checkpoint["planner_optimizer_state_dict"])

        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        if checkpoint.get("curriculum_state") and self.curriculum_scheduler:
            self.curriculum_scheduler.load_state_dict(checkpoint["curriculum_state"])

        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")


def create_synthetic_training_data(
    num_samples: int = 1000,
    embedding_dim: int = 512,
    max_steps: int = 8,
    num_strategies: int = 5,
) -> NeuroSymbolicDataset:
    """
    Create synthetic training data for Neuro-Symbolic training.

    This is useful for testing and demonstration purposes.
    In practice, you would use real data with labeled decompositions.

    Args:
        num_samples: Number of samples to generate
        embedding_dim: Dimension of embeddings
        max_steps: Maximum planning steps
        num_strategies: Number of strategies

    Returns:
        NeuroSymbolicDataset with synthetic data
    """
    queries = []
    target_strategies = []
    target_confidences = []

    # Strategy templates
    strategy_templates = [
        ("What is", [0, 0]),  # Direct answer
        ("How do I", [2, 2, 0]),  # Step by step
        ("Research", [1, 1, 1, 0]),  # Deep research
        ("Calculate", [3, 2, 0]),  # Tool use
        ("Compare", [1, 2, 4, 0]),  # Delegation
    ]

    for i in range(num_samples):
        # Select template
        template_idx = i % len(strategy_templates)
        prefix, strategies = strategy_templates[template_idx]

        # Generate query
        query = f"{prefix} topic_{i % 100} with details_{i}"
        queries.append(query)

        # Add some noise to strategies
        noisy_strategies = []
        for s in strategies:
            if np.random.random() < 0.1:
                s = np.random.randint(0, num_strategies)
            noisy_strategies.append(s)
        target_strategies.append(noisy_strategies)

        # Generate confidences
        confidences = [0.5 + 0.4 * np.random.random() for _ in strategies]
        target_confidences.append(confidences)

    return NeuroSymbolicDataset(
        queries=queries,
        target_strategies=target_strategies,
        target_confidences=target_confidences,
        embedding_dim=embedding_dim,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    # Test the trainer
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Neuro-Symbolic Trainer")

    # Create config
    config = NeuroSymbolicConfig(
        epochs=2,
        batch_size=4,
    )

    # Create trainer
    trainer = NeuroSymbolicTrainer(config)
    trainer.build_models()

    # Create synthetic data
    dataset = create_synthetic_training_data(num_samples=100)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_neuro_symbolic,
    )

    # Train
    results = trainer.train(dataloader, num_epochs=2)

    logger.info(f"Training results: {results}")
    logger.info("Neuro-Symbolic Trainer test complete")
