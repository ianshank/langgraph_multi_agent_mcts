"""
Distributed Self-Play Pipeline for AlphaZero-Style Training.

Implements the actor-learner separation pattern from DeepMind's AlphaZero:
- Multiple self-play actors generating game data in parallel
- Single learner training the neural network
- Evaluator assessing model strength through head-to-head matches

Adapted from michaelnny/alpha_zero with enhancements for:
- Multi-GPU support
- Async communication between actors and learner
- Checkpoint-aware training resumption
- Comprehensive metrics and logging

Key components:
1. SelfPlayActor: Generates games using MCTS
2. NetworkLearner: Trains policy-value network
3. ModelEvaluator: Evaluates model strength via Elo
4. DistributedPipeline: Orchestrates all components
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..framework.mcts.batch_mcts import BatchMCTS
    from ..framework.mcts.neural_mcts import GameState, NeuralMCTS
    from ..games.base import GameEnvironment
    from .elo_rating import EloRatingSystem
    from .replay_buffer import Experience


# Type variable for game state
StateT = TypeVar("StateT")


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training pipeline.

    All values are configurable with sensible defaults.
    """

    # Actor configuration
    num_actors: int = 4  # Number of parallel self-play workers
    games_per_actor: int = 25  # Games per actor per iteration
    actor_device: str = "cpu"  # Device for actors (usually CPU)

    # Learner configuration
    learner_device: str = "cuda:0"  # Device for training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_training_steps: int = 1000  # Training steps per iteration

    # Evaluator configuration
    evaluation_games: int = 100  # Games for model evaluation
    evaluation_interval: int = 5  # Evaluate every N iterations
    win_threshold: float = 0.55  # Required win rate to update model

    # Replay buffer configuration
    buffer_capacity: int = 500_000
    min_buffer_size: int = 10_000  # Minimum samples before training

    # Checkpoint configuration
    checkpoint_interval: int = 10
    max_checkpoints: int = 10

    # Communication
    queue_size: int = 1000  # Size of game data queue
    network_sync_interval: int = 100  # Sync network weights every N games

    # MCTS configuration passthrough
    num_simulations: int = 800
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 30

    # Mixed precision training
    use_mixed_precision: bool = True

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        """Validate configuration."""
        if self.num_actors <= 0:
            raise ValueError(f"num_actors must be positive, got {self.num_actors}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class GameTrajectory:
    """
    Complete trajectory from a self-play game.

    Contains all data needed for training.
    """

    states: list[torch.Tensor]  # State tensors
    policies: list[np.ndarray]  # MCTS policy targets
    game_result: float  # Final game outcome (-1, 0, 1)
    move_count: int
    game_id: str = ""
    actor_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_examples(self) -> list[tuple[torch.Tensor, np.ndarray, float]]:
        """
        Convert trajectory to training examples.

        Returns:
            List of (state, policy, value) tuples
        """
        examples = []
        for i, (state, policy) in enumerate(zip(self.states, self.policies, strict=True)):
            # Value from perspective of player who made the move
            # Alternates based on move number (even = player 1)
            value = self.game_result if i % 2 == 0 else -self.game_result
            examples.append((state, policy, value))
        return examples


@dataclass
class TrainingMetrics:
    """Metrics from training."""

    iteration: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    learning_rate: float = 0.0
    games_generated: int = 0
    training_steps: int = 0
    evaluation_win_rate: float = 0.0
    elo_rating: float = 1500.0
    elapsed_time: float = 0.0


# -------------------- Self-Play Actor --------------------


class SelfPlayActor(ABC, Generic[StateT]):
    """
    Abstract base class for self-play actors.

    Actors generate games by playing against themselves using MCTS.
    """

    def __init__(
        self,
        actor_id: int,
        config: DistributedConfig,
        network: nn.Module,
        device: str = "cpu",
    ):
        """
        Initialize actor.

        Args:
            actor_id: Unique actor identifier
            config: Training configuration
            network: Policy-value network
            device: Device for this actor
        """
        self.actor_id = actor_id
        self.config = config
        self.network = network.to(device)
        self.device = device
        self.games_played = 0

    @abstractmethod
    async def play_game(self, initial_state: StateT) -> GameTrajectory:
        """
        Play a single self-play game.

        Args:
            initial_state: Starting game state

        Returns:
            Complete game trajectory
        """
        ...

    @abstractmethod
    def update_network(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Update network weights from learner.

        Args:
            state_dict: New network weights
        """
        ...

    async def generate_games(
        self,
        num_games: int,
        initial_state_fn: Callable[[], StateT],
    ) -> list[GameTrajectory]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate
            initial_state_fn: Factory for initial states

        Returns:
            List of game trajectories
        """
        trajectories = []
        for _ in range(num_games):
            initial_state = initial_state_fn()
            trajectory = await self.play_game(initial_state)
            trajectories.append(trajectory)
            self.games_played += 1
        return trajectories


class MCTSSelfPlayActor(SelfPlayActor):
    """
    Self-play actor using Neural MCTS.

    Generates games by running MCTS at each position.
    """

    def __init__(
        self,
        actor_id: int,
        config: DistributedConfig,
        network: nn.Module,
        mcts: NeuralMCTS,
        device: str = "cpu",
    ):
        """
        Initialize MCTS actor.

        Args:
            actor_id: Unique actor identifier
            config: Training configuration
            network: Policy-value network
            mcts: Neural MCTS instance
            device: Device for this actor
        """
        super().__init__(actor_id, config, network, device)
        self.mcts = mcts

    async def play_game(self, initial_state: GameState) -> GameTrajectory:
        """
        Play a single game using MCTS.

        Args:
            initial_state: Starting game state

        Returns:
            Complete game trajectory
        """
        states: list[torch.Tensor] = []
        policies: list[np.ndarray] = []

        state = initial_state
        move_count = 0

        while not state.is_terminal():
            # Determine temperature based on move number
            temperature = (
                self.config.dirichlet_epsilon
                if move_count < self.config.temperature_threshold
                else 0.0
            )

            # Run MCTS
            action_probs, _ = await self.mcts.search(
                root_state=state,
                num_simulations=self.config.num_simulations,
                temperature=temperature,
                add_root_noise=(move_count < self.config.temperature_threshold),
            )

            # Store training data
            states.append(state.to_tensor())

            # Convert action_probs dict to array
            action_size = len(next(iter(action_probs.values()))) if action_probs else 0
            policy_array = np.zeros(action_size)
            for action, prob in action_probs.items():
                idx = state.action_to_index(action)
                if idx < action_size:
                    policy_array[idx] = prob
            policies.append(policy_array)

            # Select and apply action
            action = self.mcts.select_action(
                action_probs,
                temperature=temperature,
                deterministic=(temperature == 0.0),
            )
            state = state.apply_action(action)
            move_count += 1

        # Get game result
        game_result = state.get_reward()

        return GameTrajectory(
            states=states,
            policies=policies,
            game_result=game_result,
            move_count=move_count,
            actor_id=self.actor_id,
            game_id=f"actor_{self.actor_id}_game_{self.games_played}",
        )

    def update_network(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Update network weights."""
        self.network.load_state_dict(state_dict)
        self.mcts.network = self.network


# -------------------- Network Learner --------------------


class NetworkLearner:
    """
    Trains the policy-value network from self-play data.

    Features:
    - Mixed precision training
    - Learning rate scheduling
    - Gradient clipping
    - Comprehensive metrics
    """

    def __init__(
        self,
        network: nn.Module,
        config: DistributedConfig,
        device: str = "cuda:0",
    ):
        """
        Initialize learner.

        Args:
            network: Policy-value network to train
            config: Training configuration
            device: Device for training
        """
        self.network = network.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated based on actual iterations
            eta_min=1e-5,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Training state
        self.training_steps = 0
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0

    def train_step(
        self,
        states: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
    ) -> dict[str, float]:
        """
        Execute single training step.

        Args:
            states: Batch of state tensors
            target_policies: Target policy distributions
            target_values: Target values

        Returns:
            Dictionary of loss metrics
        """
        self.network.train()

        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        self.optimizer.zero_grad()

        if self.config.use_mixed_precision and self.scaler is not None:
            with autocast():
                policy_logits, value = self.network(states)
                policy_loss, value_loss, total_loss = self._compute_loss(
                    policy_logits, value, target_policies, target_values
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_logits, value = self.network(states)
            policy_loss, value_loss, total_loss = self._compute_loss(
                policy_logits, value, target_policies, target_values
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        self.training_steps += 1
        self.total_policy_loss += policy_loss.item()
        self.total_value_loss += value_loss.item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def _compute_loss(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute AlphaZero loss.

        Args:
            policy_logits: Network policy output (log probs)
            value: Network value output
            target_policy: MCTS policy target
            target_value: Game outcome target

        Returns:
            Tuple of (policy_loss, value_loss, total_loss)
        """
        # Value loss: MSE
        value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), target_value)

        # Policy loss: Cross-entropy
        policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(target_policy * policy_log_probs, dim=-1).mean()

        # Combined loss
        total_loss = policy_loss + value_loss

        return policy_loss, value_loss, total_loss

    def get_network_state(self) -> dict[str, torch.Tensor]:
        """Get current network weights for syncing to actors."""
        return {k: v.cpu() for k, v in self.network.state_dict().items()}

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def get_metrics(self) -> dict[str, float]:
        """Get training metrics."""
        return {
            "avg_policy_loss": self.total_policy_loss / max(1, self.training_steps),
            "avg_value_loss": self.total_value_loss / max(1, self.training_steps),
            "training_steps": self.training_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
        self.training_steps = 0


# -------------------- Model Evaluator --------------------


class ModelEvaluator:
    """
    Evaluates model strength through head-to-head matches.

    Uses Elo rating system to track model improvement.
    """

    def __init__(
        self,
        config: DistributedConfig,
        elo_system: EloRatingSystem | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            config: Training configuration
            elo_system: Elo rating system (created if None)
        """
        self.config = config

        if elo_system is None:
            from .elo_rating import EloConfig, EloRatingSystem

            self.elo_system = EloRatingSystem(EloConfig())
        else:
            self.elo_system = elo_system

    async def evaluate(
        self,
        current_network: nn.Module,
        best_network: nn.Module,
        mcts_factory: Callable[[nn.Module], NeuralMCTS],
        initial_state_fn: Callable[[], GameState],
        num_games: int | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate current model against best model.

        Args:
            current_network: Network being evaluated
            best_network: Current best network
            mcts_factory: Factory to create MCTS instances
            initial_state_fn: Factory for initial states
            num_games: Number of evaluation games

        Returns:
            Evaluation results
        """
        num_games = num_games or self.config.evaluation_games

        current_mcts = mcts_factory(current_network)
        best_mcts = mcts_factory(best_network)

        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(num_games):
            # Alternate who plays first
            current_plays_first = game_idx % 2 == 0

            result = await self._play_evaluation_game(
                current_mcts=current_mcts,
                best_mcts=best_mcts,
                initial_state=initial_state_fn(),
                current_plays_first=current_plays_first,
            )

            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1

        # Calculate win rate
        win_rate = (wins + 0.5 * draws) / num_games

        # Update Elo ratings
        current_id = f"model_iter_{time.time()}"
        best_id = "best_model"

        for _ in range(wins):
            self.elo_system.update_ratings(current_id, best_id, 1.0)
        for _ in range(losses):
            self.elo_system.update_ratings(current_id, best_id, 0.0)
        for _ in range(draws):
            self.elo_system.update_ratings(current_id, best_id, 0.5)

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "is_better": win_rate >= self.config.win_threshold,
            "elo_current": self.elo_system.get_rating(current_id),
            "elo_best": self.elo_system.get_rating(best_id),
        }

    async def _play_evaluation_game(
        self,
        current_mcts: NeuralMCTS,
        best_mcts: NeuralMCTS,
        initial_state: GameState,
        current_plays_first: bool,
    ) -> float:
        """
        Play single evaluation game.

        Args:
            current_mcts: MCTS for current model
            best_mcts: MCTS for best model
            initial_state: Starting state
            current_plays_first: Whether current model plays first

        Returns:
            Result from current model's perspective (1, 0, -1)
        """
        state = initial_state
        is_current_turn = current_plays_first

        while not state.is_terminal():
            mcts = current_mcts if is_current_turn else best_mcts

            # Use deterministic selection for evaluation
            action_probs, _ = await mcts.search(
                root_state=state,
                temperature=0.0,
                add_root_noise=False,
            )

            action = mcts.select_action(action_probs, deterministic=True)
            state = state.apply_action(action)
            is_current_turn = not is_current_turn

        # Get result from current model's perspective
        result = state.get_reward()
        return result if current_plays_first else -result


# -------------------- Distributed Pipeline --------------------


class DistributedSelfPlayPipeline:
    """
    Orchestrates distributed AlphaZero-style training.

    Components:
    - Multiple self-play actors generating games
    - Single learner training the network
    - Evaluator comparing model versions
    - Checkpoint manager for persistence

    This is the main entry point for training.
    """

    def __init__(
        self,
        network: nn.Module,
        config: DistributedConfig,
        initial_state_fn: Callable[[], GameState],
        mcts_factory: Callable[[nn.Module], NeuralMCTS],
        action_size: int,
    ):
        """
        Initialize distributed pipeline.

        Args:
            network: Policy-value network
            config: Training configuration
            initial_state_fn: Factory for initial game states
            mcts_factory: Factory to create MCTS instances
            action_size: Size of action space
        """
        self.config = config
        self.initial_state_fn = initial_state_fn
        self.mcts_factory = mcts_factory
        self.action_size = action_size

        # Create learner
        self.learner = NetworkLearner(network, config, config.learner_device)

        # Create best model copy
        import copy

        self.best_network = copy.deepcopy(network)
        self.best_network.load_state_dict(network.state_dict())

        # Create evaluator
        self.evaluator = ModelEvaluator(config)

        # Create replay buffer
        from .replay_buffer import PrioritizedReplayBuffer

        self.replay_buffer = PrioritizedReplayBuffer(capacity=config.buffer_capacity)

        # Game data queue
        self.game_queue: queue.Queue[GameTrajectory] = queue.Queue(maxsize=config.queue_size)

        # Training state
        self.current_iteration = 0
        self.games_generated = 0
        self.best_win_rate = 0.0

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=config.max_checkpoints,
        )

        # Setup directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    async def train_iteration(self) -> TrainingMetrics:
        """
        Execute single training iteration.

        Returns:
            Training metrics
        """
        start_time = time.time()
        self.current_iteration += 1

        print(f"\n{'=' * 60}")
        print(f"Iteration {self.current_iteration}")
        print(f"{'=' * 60}")

        # Phase 1: Generate self-play games
        print("\n[1/3] Generating self-play games...")
        games_before = self.games_generated
        await self._generate_games()
        games_in_iteration = self.games_generated - games_before

        # Phase 2: Train network
        print("\n[2/3] Training network...")
        training_metrics = await self._train_network()

        # Phase 3: Evaluate (periodically)
        eval_metrics = {"win_rate": 0.0, "elo_current": 1500.0}
        if self.current_iteration % self.config.evaluation_interval == 0:
            print("\n[3/3] Evaluating model...")
            eval_metrics = await self._evaluate_model()

        # Save checkpoint (periodically)
        if self.current_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        elapsed = time.time() - start_time

        metrics = TrainingMetrics(
            iteration=self.current_iteration,
            policy_loss=training_metrics.get("avg_policy_loss", 0.0),
            value_loss=training_metrics.get("avg_value_loss", 0.0),
            total_loss=training_metrics.get("avg_policy_loss", 0.0) + training_metrics.get("avg_value_loss", 0.0),
            learning_rate=training_metrics.get("learning_rate", 0.0),
            games_generated=games_in_iteration,
            training_steps=training_metrics.get("training_steps", 0),
            evaluation_win_rate=eval_metrics.get("win_rate", 0.0),
            elo_rating=eval_metrics.get("elo_current", 1500.0),
            elapsed_time=elapsed,
        )

        self._log_metrics(metrics)

        return metrics

    async def _generate_games(self) -> None:
        """Generate self-play games using multiple actors."""
        games_per_actor = self.config.games_per_actor

        # Create actors
        actors = []
        for i in range(self.config.num_actors):
            network_copy = self._create_network_copy()
            mcts = self.mcts_factory(network_copy)
            actor = MCTSSelfPlayActor(
                actor_id=i,
                config=self.config,
                network=network_copy,
                mcts=mcts,
                device=self.config.actor_device,
            )
            actors.append(actor)

        # Generate games from all actors
        all_trajectories: list[GameTrajectory] = []

        # Run actors (can be parallelized with multiprocessing in production)
        for actor in actors:
            trajectories = await actor.generate_games(games_per_actor, self.initial_state_fn)
            all_trajectories.extend(trajectories)

        # Add to replay buffer
        from .replay_buffer import Experience

        for trajectory in all_trajectories:
            for state, policy, value in trajectory.to_training_examples():
                exp = Experience(state=state, policy=policy, value=value)
                self.replay_buffer.add(exp)
                self.games_generated += 1

        print(f"  Generated {len(all_trajectories)} games, {self.games_generated} total positions")

    async def _train_network(self) -> dict[str, float]:
        """Train network on replay buffer data."""
        if not self.replay_buffer.is_ready(self.config.min_buffer_size):
            print(f"  Buffer not ready ({len(self.replay_buffer)}/{self.config.min_buffer_size})")
            return {}

        self.learner.reset_metrics()

        for step in range(self.config.num_training_steps):
            # Sample batch
            experiences, _, weights = self.replay_buffer.sample(self.config.batch_size)

            # Prepare tensors
            states = torch.stack([e.state for e in experiences])
            policies = torch.tensor(np.array([e.policy for e in experiences]), dtype=torch.float32)
            values = torch.tensor([e.value for e in experiences], dtype=torch.float32)

            # Train step
            self.learner.train_step(states, policies, values)

            if (step + 1) % 100 == 0:
                metrics = self.learner.get_metrics()
                print(
                    f"  Step {step + 1}/{self.config.num_training_steps}: "
                    f"policy_loss={metrics['avg_policy_loss']:.4f}, "
                    f"value_loss={metrics['avg_value_loss']:.4f}"
                )

        # Step scheduler
        self.learner.step_scheduler()

        return self.learner.get_metrics()

    async def _evaluate_model(self) -> dict[str, Any]:
        """Evaluate current model against best model."""
        current_network = self.learner.network

        eval_results = await self.evaluator.evaluate(
            current_network=current_network,
            best_network=self.best_network,
            mcts_factory=self.mcts_factory,
            initial_state_fn=self.initial_state_fn,
        )

        print(
            f"  Evaluation: {eval_results['wins']}W/{eval_results['losses']}L/{eval_results['draws']}D "
            f"(win rate: {eval_results['win_rate']:.2%})"
        )

        # Update best model if improved
        if eval_results["is_better"]:
            print(f"  New best model! (win rate: {eval_results['win_rate']:.2%})")
            self.best_network.load_state_dict(current_network.state_dict())
            self.best_win_rate = eval_results["win_rate"]

        return eval_results

    def _create_network_copy(self) -> nn.Module:
        """Create a copy of the learner's network."""
        import copy

        network_copy = copy.deepcopy(self.learner.network)
        network_copy.load_state_dict(self.learner.get_network_state())
        network_copy.to(self.config.actor_device)
        network_copy.eval()
        return network_copy

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_data = {
            "iteration": self.current_iteration,
            "network_state": self.learner.network.state_dict(),
            "best_network_state": self.best_network.state_dict(),
            "optimizer_state": self.learner.optimizer.state_dict(),
            "scheduler_state": self.learner.scheduler.state_dict(),
            "games_generated": self.games_generated,
            "best_win_rate": self.best_win_rate,
            "config": {
                "num_actors": self.config.num_actors,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
        }

        self.checkpoint_manager.save(checkpoint_data, self.current_iteration)
        print(f"  Checkpoint saved: iteration {self.current_iteration}")

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> None:
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (latest if None)
        """
        if checkpoint_path is None:
            checkpoint_data = self.checkpoint_manager.load_latest()
        else:
            checkpoint_data = self.checkpoint_manager.load(checkpoint_path)

        if checkpoint_data is None:
            print("No checkpoint found")
            return

        self.learner.network.load_state_dict(checkpoint_data["network_state"])
        self.best_network.load_state_dict(checkpoint_data["best_network_state"])
        self.learner.optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        self.learner.scheduler.load_state_dict(checkpoint_data["scheduler_state"])
        self.current_iteration = checkpoint_data["iteration"]
        self.games_generated = checkpoint_data["games_generated"]
        self.best_win_rate = checkpoint_data["best_win_rate"]

        print(f"Loaded checkpoint from iteration {self.current_iteration}")

    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        log_path = Path(self.config.log_dir) / "training_log.jsonl"

        with open(log_path, "a") as f:
            f.write(json.dumps(metrics.__dict__) + "\n")

    async def train(self, num_iterations: int) -> None:
        """
        Run complete training loop.

        Args:
            num_iterations: Number of training iterations
        """
        print(f"\nStarting training for {num_iterations} iterations")
        print(f"Actors: {self.config.num_actors}")
        print(f"Device: {self.config.learner_device}")
        print(f"Batch size: {self.config.batch_size}")

        for _ in range(num_iterations):
            await self.train_iteration()

        print("\nTraining complete!")
        print(f"Total iterations: {self.current_iteration}")
        print(f"Total games: {self.games_generated}")
        print(f"Best win rate: {self.best_win_rate:.2%}")


# -------------------- Checkpoint Manager --------------------


class CheckpointManager:
    """
    Manages training checkpoints.

    Features:
    - Automatic checkpoint rotation
    - Best model tracking
    - Replay buffer state (optional)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: dict[str, Any],
        iteration: int,
        is_best: bool = False,
    ) -> Path:
        """
        Save checkpoint.

        Args:
            data: Checkpoint data
            iteration: Current iteration
            is_best: Whether this is the best model

        Returns:
            Path to saved checkpoint
        """
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration:06d}.pt"
        torch.save(data, checkpoint_path)

        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(data, best_path)

        # Rotate old checkpoints
        self._rotate_checkpoints()

        return checkpoint_path

    def load(self, path: str | Path) -> dict[str, Any] | None:
        """
        Load checkpoint from path.

        Args:
            path: Checkpoint path

        Returns:
            Checkpoint data or None
        """
        path = Path(path)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_latest(self) -> dict[str, Any] | None:
        """
        Load most recent checkpoint.

        Returns:
            Checkpoint data or None
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_iter_*.pt"))
        if not checkpoints:
            return None
        return self.load(checkpoints[-1])

    def load_best(self) -> dict[str, Any] | None:
        """
        Load best model checkpoint.

        Returns:
            Checkpoint data or None
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return self.load(best_path)

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_iter_*.pt"))
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()

    def list_checkpoints(self) -> list[Path]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_dir.glob("checkpoint_iter_*.pt"))
