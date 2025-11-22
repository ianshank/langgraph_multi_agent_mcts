"""
Unified Training Orchestrator for LangGraph Multi-Agent MCTS with DeepMind-Style Learning.

Coordinates:
- HRM Agent
- TRM Agent
- Neural MCTS
- Policy-Value Network
- Self-play data generation
- Training loops
- Evaluation
- Checkpointing
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ..agents.hrm_agent import HRMLoss, create_hrm_agent
from ..agents.trm_agent import TRMLoss, create_trm_agent
from ..framework.mcts.neural_mcts import GameState, NeuralMCTS, SelfPlayCollector
from ..models.policy_value_net import (
    AlphaZeroLoss,
    create_policy_value_network,
)
from .performance_monitor import PerformanceMonitor, TimingContext
from .replay_buffer import Experience, PrioritizedReplayBuffer, collate_experiences
from .system_config import SystemConfig


class UnifiedTrainingOrchestrator:
    """
    Complete training pipeline integrating all framework components.

    This orchestrator manages:
    1. Self-play data generation using MCTS
    2. Neural network training (policy-value)
    3. HRM agent training
    4. TRM agent training
    5. Evaluation and checkpointing
    6. Performance monitoring
    """

    def __init__(
        self,
        config: SystemConfig,
        initial_state_fn: Callable[[], GameState],
        board_size: int = 19,
    ):
        """
        Initialize training orchestrator.

        Args:
            config: System configuration
            initial_state_fn: Function that returns initial game state
            board_size: Board/grid size for spatial games
        """
        self.config = config
        self.initial_state_fn = initial_state_fn
        self.board_size = board_size

        # Setup device
        self.device = config.device
        torch.manual_seed(config.seed)

        # Initialize performance monitor
        self.monitor = PerformanceMonitor(
            window_size=100,
            enable_gpu_monitoring=(self.device != "cpu"),
        )

        # Initialize components
        self._initialize_components()

        # Training state
        self.current_iteration = 0
        self.best_win_rate = 0.0
        self.best_model_path = None

        # Setup paths
        self._setup_paths()

        # Setup experiment tracking
        if config.use_wandb:
            self._setup_wandb()

    def _initialize_components(self):
        """Initialize all framework components."""
        print("Initializing components...")

        # Policy-Value Network
        self.policy_value_net = create_policy_value_network(
            config=self.config.neural_net,
            board_size=self.board_size,
            device=self.device,
        )

        print(f"  ✓ Policy-Value Network: {self.policy_value_net.get_parameter_count():,} parameters")

        # HRM Agent
        self.hrm_agent = create_hrm_agent(self.config.hrm, self.device)
        print(f"  ✓ HRM Agent: {self.hrm_agent.get_parameter_count():,} parameters")

        # TRM Agent
        self.trm_agent = create_trm_agent(
            self.config.trm, output_dim=self.config.neural_net.action_size, device=self.device
        )
        print(f"  ✓ TRM Agent: {self.trm_agent.get_parameter_count():,} parameters")

        # Neural MCTS
        self.mcts = NeuralMCTS(
            policy_value_network=self.policy_value_net,
            config=self.config.mcts,
            device=self.device,
        )
        print("  ✓ Neural MCTS initialized")

        # Self-play collector
        self.self_play_collector = SelfPlayCollector(mcts=self.mcts, config=self.config.mcts)

        # Optimizers
        self._setup_optimizers()

        # Loss functions
        self.pv_loss_fn = AlphaZeroLoss(value_loss_weight=1.0)
        self.hrm_loss_fn = HRMLoss(ponder_weight=0.01)
        self.trm_loss_fn = TRMLoss(
            task_loss_fn=nn.MSELoss(),
            supervision_weight_decay=self.config.trm.supervision_weight_decay,
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.training.buffer_size,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=self.config.training.games_per_iteration * 10,
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.config.use_mixed_precision else None

    def _setup_optimizers(self):
        """Setup optimizers and learning rate schedulers."""
        # Policy-Value optimizer
        self.pv_optimizer = torch.optim.SGD(
            self.policy_value_net.parameters(),
            lr=self.config.training.learning_rate,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )

        # HRM optimizer
        self.hrm_optimizer = torch.optim.Adam(self.hrm_agent.parameters(), lr=1e-3)

        # TRM optimizer
        self.trm_optimizer = torch.optim.Adam(self.trm_agent.parameters(), lr=1e-3)

        # Learning rate scheduler for policy-value network
        if self.config.training.lr_schedule == "cosine":
            self.pv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.pv_optimizer, T_max=100)
        elif self.config.training.lr_schedule == "step":
            self.pv_scheduler = torch.optim.lr_scheduler.StepLR(
                self.pv_optimizer,
                step_size=self.config.training.lr_decay_steps,
                gamma=self.config.training.lr_decay_gamma,
            )
        else:
            self.pv_scheduler = None

    def _setup_paths(self):
        """Setup directory paths."""
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_wandb(self):
        """Setup Weights & Biases experiment tracking."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.to_dict(),
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            print("  ✓ Weights & Biases initialized")
        except ImportError:
            print("  ⚠️  wandb not installed, skipping")
            self.config.use_wandb = False

    async def train_iteration(self, iteration: int) -> dict[str, Any]:
        """
        Execute single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Dictionary of metrics
        """
        print(f"\n{'=' * 80}")
        print(f"Training Iteration {iteration}")
        print(f"{'=' * 80}")

        metrics = {}

        # Phase 1: Self-play data generation
        print("\n[1/5] Generating self-play data...")
        with TimingContext(self.monitor, "self_play_generation"):
            game_data = await self._generate_self_play_data()
            metrics["games_generated"] = len(game_data)
            print(f"  Generated {len(game_data)} training examples")

        # Phase 2: Policy-Value network training
        print("\n[2/5] Training Policy-Value Network...")
        with TimingContext(self.monitor, "pv_training"):
            pv_metrics = await self._train_policy_value_network()
            metrics.update(pv_metrics)

        # Phase 3: HRM agent training (optional, if using HRM)
        if hasattr(self, "hrm_agent"):
            print("\n[3/5] Training HRM Agent...")
            with TimingContext(self.monitor, "hrm_training"):
                hrm_metrics = await self._train_hrm_agent()
                metrics.update(hrm_metrics)

        # Phase 4: TRM agent training (optional, if using TRM)
        if hasattr(self, "trm_agent"):
            print("\n[4/5] Training TRM Agent...")
            with TimingContext(self.monitor, "trm_training"):
                trm_metrics = await self._train_trm_agent()
                metrics.update(trm_metrics)

        # Phase 5: Evaluation
        print("\n[5/5] Evaluation...")
        if iteration % self.config.training.checkpoint_interval == 0:
            eval_metrics = await self._evaluate()
            metrics.update(eval_metrics)

            # Save checkpoint if improved
            if eval_metrics.get("win_rate", 0) > self.best_win_rate:
                self.best_win_rate = eval_metrics["win_rate"]
                self._save_checkpoint(iteration, metrics, is_best=True)
                print(f"  ✓ New best model! Win rate: {self.best_win_rate:.2%}")

        # Log metrics
        self._log_metrics(iteration, metrics)

        # Performance check
        self.monitor.alert_if_slow()

        return metrics

    async def _generate_self_play_data(self) -> list[Experience]:
        """Generate training data from self-play games."""
        num_games = self.config.training.games_per_iteration

        # In production, this would use parallel actors
        # For simplicity, we'll do sequential self-play
        all_examples = []

        for game_idx in range(num_games):
            examples = await self.self_play_collector.play_game(
                initial_state=self.initial_state_fn(),
                temperature_threshold=self.config.mcts.temperature_threshold,
            )

            # Convert to Experience objects
            for ex in examples:
                all_examples.append(Experience(state=ex.state, policy=ex.policy_target, value=ex.value_target))

            if (game_idx + 1) % 5 == 0:
                print(f"  Generated {game_idx + 1}/{num_games} games...")

        # Add to replay buffer
        self.replay_buffer.add_batch(all_examples)

        return all_examples

    async def _train_policy_value_network(self) -> dict[str, float]:
        """Train policy-value network on replay buffer data."""
        if not self.replay_buffer.is_ready(self.config.training.batch_size):
            print("  Replay buffer not ready, skipping...")
            return {"policy_loss": 0.0, "value_loss": 0.0}

        self.policy_value_net.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 10  # Train for 10 batches per iteration

        for _ in range(num_batches):
            # Sample batch
            experiences, indices, weights = self.replay_buffer.sample(self.config.training.batch_size)
            states, policies, values = collate_experiences(experiences)

            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            weights = torch.from_numpy(weights).to(self.device)

            # Forward pass
            if self.config.use_mixed_precision and self.scaler:
                with autocast():
                    policy_logits, value_pred = self.policy_value_net(states)
                    loss, loss_dict = self.pv_loss_fn(policy_logits, value_pred, policies, values)
                    # Apply importance sampling weights
                    loss = (loss * weights).mean()

                # Backward pass with mixed precision
                self.pv_optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.pv_optimizer)
                self.scaler.update()
            else:
                policy_logits, value_pred = self.policy_value_net(states)
                loss, loss_dict = self.pv_loss_fn(policy_logits, value_pred, policies, values)
                loss = (loss * weights).mean()

                self.pv_optimizer.zero_grad()
                loss.backward()
                self.pv_optimizer.step()

            # Update priorities in replay buffer
            with torch.no_grad():
                td_errors = torch.abs(value_pred.squeeze() - values)
                self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

            total_policy_loss += loss_dict["policy"]
            total_value_loss += loss_dict["value"]

            # Log losses
            self.monitor.log_loss(loss_dict["policy"], loss_dict["value"], loss_dict["total"])

        # Step learning rate scheduler
        if self.pv_scheduler:
            self.pv_scheduler.step()

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        print(f"  Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

        return {"policy_loss": avg_policy_loss, "value_loss": avg_value_loss}

    async def _train_hrm_agent(self) -> dict[str, float]:
        """Train HRM agent (placeholder for domain-specific implementation)."""
        # This would require domain-specific data and tasks
        # For now, return dummy metrics
        return {"hrm_halt_step": 5.0, "hrm_ponder_cost": 0.1}

    async def _train_trm_agent(self) -> dict[str, float]:
        """Train TRM agent (placeholder for domain-specific implementation)."""
        # This would require domain-specific data and tasks
        # For now, return dummy metrics
        return {"trm_convergence_step": 8.0, "trm_final_residual": 0.01}

    async def _evaluate(self) -> dict[str, float]:
        """Evaluate current model against baseline."""
        # Simplified evaluation: play games against previous best
        # In production, this would be more sophisticated
        win_rate = 0.55  # Placeholder

        return {
            "win_rate": win_rate,
            "eval_games": self.config.training.evaluation_games,
        }

    def _save_checkpoint(self, iteration: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "policy_value_net": self.policy_value_net.state_dict(),
            "hrm_agent": self.hrm_agent.state_dict(),
            "trm_agent": self.trm_agent.state_dict(),
            "pv_optimizer": self.pv_optimizer.state_dict(),
            "hrm_optimizer": self.hrm_optimizer.state_dict(),
            "trm_optimizer": self.trm_optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "best_win_rate": self.best_win_rate,
        }

        # Save regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved: {path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"  ✓ Best model saved: {best_path}")

    def _log_metrics(self, iteration: int, metrics: dict):
        """Log metrics to console and tracking systems."""
        print(f"\n[Metrics Summary - Iteration {iteration}]")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Log to wandb
        if self.config.use_wandb:
            try:
                import wandb

                wandb_metrics = self.monitor.export_to_wandb(iteration)
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics, step=iteration)
            except Exception as e:
                print(f"  ⚠️  Failed to log to wandb: {e}")

    async def train(self, num_iterations: int):
        """
        Run complete training loop.

        Args:
            num_iterations: Number of training iterations
        """
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Total iterations: {num_iterations}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.config.use_mixed_precision}")

        start_time = time.time()

        for iteration in range(1, num_iterations + 1):
            self.current_iteration = iteration

            try:
                _ = await self.train_iteration(iteration)

                # Check early stopping
                if self._should_early_stop(iteration):
                    print("\n⚠️  Early stopping triggered")
                    break

            except KeyboardInterrupt:
                print("\n⚠️  Training interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error in iteration {iteration}: {e}")
                import traceback

                traceback.print_exc()
                break

        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Training completed in {elapsed / 3600:.2f} hours")
        print(f"Best win rate: {self.best_win_rate:.2%}")
        print(f"{'=' * 80}\n")

        # Print final performance summary
        self.monitor.print_summary()

    def _should_early_stop(self, iteration: int) -> bool:
        """Check early stopping criteria."""
        # Placeholder: implement actual early stopping logic
        _ = iteration  # noqa: F841
        return False

    def load_checkpoint(self, path: str):
        """Load checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.policy_value_net.load_state_dict(checkpoint["policy_value_net"])
        self.hrm_agent.load_state_dict(checkpoint["hrm_agent"])
        self.trm_agent.load_state_dict(checkpoint["trm_agent"])

        self.pv_optimizer.load_state_dict(checkpoint["pv_optimizer"])
        self.hrm_optimizer.load_state_dict(checkpoint["hrm_optimizer"])
        self.trm_optimizer.load_state_dict(checkpoint["trm_optimizer"])

        self.current_iteration = checkpoint["iteration"]
        self.best_win_rate = checkpoint.get("best_win_rate", 0.0)

        print(f"✓ Loaded checkpoint from iteration {self.current_iteration}")
