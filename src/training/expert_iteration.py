"""
Expert Iteration (ExIt) Training Module.

Implements the Expert Iteration algorithm used in AlphaZero:
1. Generate trajectories using MCTS-guided search (Expert)
2. Train neural network to imitate the expert (Apprentice)
3. Use improved network to guide search
4. Repeat for self-improvement

This closes the loop between NeuralMCTS and model training.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, AsyncIterator

import numpy as np

# Optional torch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

from ..framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from ..framework.mcts.config import MCTSConfig

# Optional neural integration import
try:
    from ..framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralMCTSAdapter,
        NeuralRolloutPolicy,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False


logger = logging.getLogger(__name__)


class PolicyValueNetworkProtocol(Protocol):
    """Protocol for policy-value networks."""

    def forward(self, x: Any) -> tuple[Any, Any]:
        """Forward pass returning (policy_logits, value)."""
        ...

    def parameters(self) -> Any:
        """Return model parameters."""
        ...

    def train(self, mode: bool = True) -> Any:
        """Set training mode."""
        ...

    def eval(self) -> Any:
        """Set evaluation mode."""
        ...


@dataclass
class TrajectoryStep:
    """A single step in a trajectory."""

    state: MCTSState
    action: str
    mcts_policy: dict[str, float]  # Visit count distribution from MCTS
    value: float  # Outcome or estimated value


@dataclass
class Trajectory:
    """A complete trajectory from MCTS search."""

    steps: list[TrajectoryStep]
    outcome: float  # Final outcome (-1 to 1)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Return trajectory length."""
        return len(self.steps)


@dataclass
class ExpertIterationConfig:
    """Configuration for Expert Iteration training."""

    # Self-play settings
    num_episodes_per_iteration: int = 100
    max_episode_length: int = 50

    # MCTS settings (for expert)
    mcts_simulations: int = 400
    mcts_exploration_weight: float = 1.414
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_threshold: int = 15

    # Dirichlet noise
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training settings (for apprentice)
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 5

    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000

    # Evaluation
    evaluation_episodes: int = 50
    win_threshold: float = 0.55

    # Checkpointing
    checkpoint_interval: int = 5
    checkpoint_dir: str = "./checkpoints/expert_iteration"

    # Logging
    log_interval: int = 10

    # Hardware
    device: str = "cpu"
    num_workers: int = 4

    # Seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if self.num_episodes_per_iteration < 1:
            raise ValueError("num_episodes_per_iteration must be >= 1")
        if self.mcts_simulations < 1:
            raise ValueError("mcts_simulations must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 < self.win_threshold < 1:
            raise ValueError("win_threshold must be in (0, 1)")


class ReplayBuffer:
    """
    Experience replay buffer for Expert Iteration.

    Stores (state, policy, value) tuples from self-play.
    """

    def __init__(self, max_size: int = 100000, seed: int = 42):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of entries
            seed: Random seed for sampling
        """
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)
        self.buffer: list[tuple[MCTSState, dict[str, float], float]] = []
        self._position = 0

    def add(self, state: MCTSState, policy: dict[str, float], value: float) -> None:
        """Add an entry to the buffer."""
        entry = (state, policy, value)
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self._position] = entry
        self._position = (self._position + 1) % self.max_size

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add all steps from a trajectory."""
        for step in trajectory.steps:
            # Use outcome for value (Monte Carlo return)
            self.add(step.state, step.mcts_policy, trajectory.outcome)

    def sample(self, batch_size: int) -> list[tuple[MCTSState, dict[str, float], float]]:
        """Sample a batch from the buffer."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self._position = 0


class SelfPlayWorker:
    """
    Worker that generates trajectories through self-play.

    Uses MCTS to generate high-quality play data.
    """

    def __init__(
        self,
        policy_value_network: PolicyValueNetworkProtocol | None,
        config: ExpertIterationConfig,
        worker_id: int = 0,
    ):
        """
        Initialize self-play worker.

        Args:
            policy_value_network: Network for neural MCTS guidance
            config: Expert iteration configuration
            worker_id: Worker identifier
        """
        self.network = policy_value_network
        self.config = config
        self.worker_id = worker_id

        # Create MCTS engine
        self.mcts_engine = MCTSEngine(
            seed=config.seed + worker_id,
            exploration_weight=config.mcts_exploration_weight,
        )

        # Create neural adapter if network available
        self.neural_adapter: Any | None = None
        if policy_value_network is not None and NEURAL_AVAILABLE:
            neural_config = NeuralMCTSConfig(
                num_simulations=config.mcts_simulations,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
                temperature_init=config.temperature_init,
                temperature_final=config.temperature_final,
                temperature_threshold=config.temperature_threshold,
            )
            self.neural_adapter = NeuralMCTSAdapter(
                config=neural_config,
                policy_value_network=policy_value_network,
            )
            self.neural_adapter.initialize()

    async def generate_trajectory(
        self,
        initial_state: MCTSState,
        action_generator: Any,
        state_transition: Any,
        is_terminal: Any,
        get_outcome: Any,
    ) -> Trajectory:
        """
        Generate a single trajectory through self-play.

        Args:
            initial_state: Starting state
            action_generator: Function to generate available actions
            state_transition: Function to compute next state
            is_terminal: Function to check if state is terminal
            get_outcome: Function to get outcome value

        Returns:
            Complete trajectory with MCTS policies
        """
        steps: list[TrajectoryStep] = []
        current_state = initial_state
        move_number = 0

        while not is_terminal(current_state) and move_number < self.config.max_episode_length:
            # Create root node
            root = MCTSNode(
                state=current_state,
                rng=self.mcts_engine.rng,
            )

            # Select rollout policy
            if self.neural_adapter is not None:
                rollout_policy = self.neural_adapter.rollout_policy
            else:
                rollout_policy = None

            # Run MCTS search
            best_action, stats = await self.mcts_engine.search(
                root=root,
                num_iterations=self.config.mcts_simulations,
                action_generator=action_generator,
                state_transition=state_transition,
                rollout_policy=rollout_policy,
            )

            # Extract policy from visit counts
            mcts_policy = self._extract_policy(root, move_number)

            # Select action based on temperature
            selected_action = self._select_action(mcts_policy, move_number)

            # Record step
            steps.append(TrajectoryStep(
                state=current_state,
                action=selected_action,
                mcts_policy=mcts_policy,
                value=stats.get("best_action_value", 0.0),
            ))

            # Transition to next state
            current_state = state_transition(current_state, selected_action)
            move_number += 1

        # Get final outcome
        outcome = get_outcome(current_state)

        return Trajectory(
            steps=steps,
            outcome=outcome,
            metadata={
                "worker_id": self.worker_id,
                "length": len(steps),
                "final_state_id": current_state.state_id,
            }
        )

    def _extract_policy(self, root: MCTSNode, move_number: int) -> dict[str, float]:
        """Extract policy from MCTS visit counts."""
        if not root.children:
            return {}

        # Get visit counts
        visits = {child.action: child.visits for child in root.children if child.action}
        total_visits = sum(visits.values())

        if total_visits == 0:
            # Uniform distribution
            n_actions = len(visits)
            return {action: 1.0 / n_actions for action in visits}

        # Apply temperature
        if move_number < self.config.temperature_threshold:
            temp = self.config.temperature_init
        else:
            temp = self.config.temperature_final

        if temp < 0.01:
            # Greedy selection
            max_visits = max(visits.values())
            return {action: 1.0 if v == max_visits else 0.0 for action, v in visits.items()}
        else:
            # Softmax with temperature
            exp_visits = {action: (v / total_visits) ** (1 / temp) for action, v in visits.items()}
            exp_sum = sum(exp_visits.values())
            return {action: v / exp_sum for action, v in exp_visits.items()}

    def _select_action(self, policy: dict[str, float], move_number: int) -> str:
        """Select action from policy distribution."""
        if not policy:
            raise ValueError("Empty policy")

        actions = list(policy.keys())
        probs = list(policy.values())

        # Sample from distribution
        return self.mcts_engine.rng.choice(actions, p=probs)


class ExpertIterationTrainer:
    """
    Main trainer for Expert Iteration.

    Coordinates self-play generation and neural network training.
    """

    def __init__(
        self,
        policy_value_network: PolicyValueNetworkProtocol,
        config: ExpertIterationConfig,
        state_encoder: Any | None = None,
    ):
        """
        Initialize Expert Iteration trainer.

        Args:
            policy_value_network: Network to train
            config: Training configuration
            state_encoder: Function to encode states for network
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Expert Iteration training")

        self.network = policy_value_network
        self.config = config
        self.state_encoder = state_encoder

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size,
            seed=config.seed,
        )

        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Metrics tracking
        self.iteration = 0
        self.best_network_version = 0
        self.metrics_history: list[dict[str, Any]] = []

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up workers
        self.workers = [
            SelfPlayWorker(
                policy_value_network=policy_value_network,
                config=config,
                worker_id=i,
            )
            for i in range(config.num_workers)
        ]

        logger.info(
            f"ExpertIterationTrainer initialized with {config.num_workers} workers, "
            f"buffer_size={config.buffer_size}"
        )

    async def run_iteration(
        self,
        action_generator: Any,
        state_transition: Any,
        is_terminal: Any,
        get_outcome: Any,
        initial_state_fn: Any,
    ) -> dict[str, Any]:
        """
        Run a single iteration of Expert Iteration.

        Args:
            action_generator: Function to generate actions
            state_transition: Function for state transitions
            is_terminal: Function to check terminal states
            get_outcome: Function to get outcome value
            initial_state_fn: Function to create initial states

        Returns:
            Dictionary of metrics for this iteration
        """
        start_time = time.time()
        self.iteration += 1

        logger.info(f"Starting Expert Iteration {self.iteration}")

        # Phase 1: Self-play generation
        logger.info("Phase 1: Generating self-play trajectories")
        trajectories = await self._generate_self_play(
            action_generator=action_generator,
            state_transition=state_transition,
            is_terminal=is_terminal,
            get_outcome=get_outcome,
            initial_state_fn=initial_state_fn,
        )

        # Add trajectories to buffer
        for traj in trajectories:
            self.replay_buffer.add_trajectory(traj)

        # Phase 2: Train network
        train_metrics = {}
        if len(self.replay_buffer) >= self.config.min_buffer_size:
            logger.info("Phase 2: Training neural network")
            train_metrics = self._train_network()
        else:
            logger.info(
                f"Skipping training - buffer size {len(self.replay_buffer)} "
                f"< min {self.config.min_buffer_size}"
            )

        # Compute metrics
        elapsed_time = time.time() - start_time
        avg_trajectory_length = np.mean([t.length for t in trajectories])
        avg_outcome = np.mean([t.outcome for t in trajectories])

        metrics = {
            "iteration": self.iteration,
            "elapsed_time": elapsed_time,
            "num_trajectories": len(trajectories),
            "avg_trajectory_length": avg_trajectory_length,
            "avg_outcome": avg_outcome,
            "buffer_size": len(self.replay_buffer),
            **train_metrics,
        }

        self.metrics_history.append(metrics)

        # Checkpoint if needed
        if self.iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

        logger.info(
            f"Iteration {self.iteration} complete: "
            f"trajectories={len(trajectories)}, "
            f"avg_length={avg_trajectory_length:.1f}, "
            f"avg_outcome={avg_outcome:.3f}, "
            f"time={elapsed_time:.1f}s"
        )

        return metrics

    async def _generate_self_play(
        self,
        action_generator: Any,
        state_transition: Any,
        is_terminal: Any,
        get_outcome: Any,
        initial_state_fn: Any,
    ) -> list[Trajectory]:
        """Generate self-play trajectories using workers."""
        trajectories: list[Trajectory] = []
        episodes_per_worker = self.config.num_episodes_per_iteration // len(self.workers)

        # Generate in parallel using workers
        async def worker_task(worker: SelfPlayWorker, num_episodes: int) -> list[Trajectory]:
            results = []
            for _ in range(num_episodes):
                initial_state = initial_state_fn()
                traj = await worker.generate_trajectory(
                    initial_state=initial_state,
                    action_generator=action_generator,
                    state_transition=state_transition,
                    is_terminal=is_terminal,
                    get_outcome=get_outcome,
                )
                results.append(traj)
            return results

        # Run workers concurrently
        tasks = [
            worker_task(worker, episodes_per_worker)
            for worker in self.workers
        ]
        results = await asyncio.gather(*tasks)

        for worker_trajectories in results:
            trajectories.extend(worker_trajectories)

        return trajectories

    def _train_network(self) -> dict[str, Any]:
        """Train network on replay buffer data."""
        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)

            if not batch:
                continue

            # Prepare batch tensors
            states, policies, values = self._prepare_batch(batch)
            if states is None:
                continue

            # Forward pass
            policy_logits, value_preds = self.network(states)

            # Compute losses
            policy_loss = self._compute_policy_loss(policy_logits, policies)
            value_loss = self._compute_value_loss(value_preds, values)
            total_loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        self.network.eval()

        if num_batches == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "num_batches": num_batches,
        }

    def _prepare_batch(
        self,
        batch: list[tuple[MCTSState, dict[str, float], float]],
    ) -> tuple[Any, Any, Any] | tuple[None, None, None]:
        """Prepare batch tensors from replay buffer samples."""
        if self.state_encoder is None:
            logger.warning("No state encoder provided - skipping batch")
            return None, None, None

        states_list = []
        policies_list = []
        values_list = []

        for state, policy, value in batch:
            # Encode state
            encoded = self.state_encoder(state)
            if encoded is None:
                continue

            states_list.append(encoded)
            values_list.append(value)

            # Convert policy dict to tensor (placeholder - needs action space mapping)
            policy_tensor = torch.zeros(len(policy))
            for i, (action, prob) in enumerate(policy.items()):
                if i < len(policy_tensor):
                    policy_tensor[i] = prob
            policies_list.append(policy_tensor)

        if not states_list:
            return None, None, None

        states = torch.stack(states_list).to(self.config.device)
        policies = torch.stack(policies_list).to(self.config.device)
        values = torch.tensor(values_list, dtype=torch.float32).to(self.config.device)

        return states, policies, values

    def _compute_policy_loss(self, logits: Any, targets: Any) -> Any:
        """Compute cross-entropy policy loss."""
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return -torch.sum(targets * log_probs, dim=-1).mean()

    def _compute_value_loss(self, preds: Any, targets: Any) -> Any:
        """Compute MSE value loss."""
        return torch.nn.functional.mse_loss(preds.squeeze(-1), targets)

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"

        checkpoint = {
            "iteration": self.iteration,
            "network_state_dict": self.network.state_dict() if hasattr(self.network, 'state_dict') else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "config": self.config.__dict__,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.iteration = checkpoint["iteration"]
        if checkpoint["network_state_dict"] and hasattr(self.network, 'load_state_dict'):
            self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics_history = checkpoint["metrics_history"]

        logger.info(f"Loaded checkpoint from {path} (iteration {self.iteration})")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of training metrics."""
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-10:]  # Last 10 iterations

        return {
            "total_iterations": self.iteration,
            "recent_avg_outcome": np.mean([m.get("avg_outcome", 0) for m in recent]),
            "recent_avg_policy_loss": np.mean([m.get("policy_loss", 0) for m in recent if "policy_loss" in m]),
            "recent_avg_value_loss": np.mean([m.get("value_loss", 0) for m in recent if "value_loss" in m]),
            "buffer_size": len(self.replay_buffer),
        }


def create_expert_iteration_trainer(
    policy_value_network: PolicyValueNetworkProtocol,
    config: ExpertIterationConfig | None = None,
    state_encoder: Any | None = None,
) -> ExpertIterationTrainer:
    """
    Factory function to create Expert Iteration trainer.

    Args:
        policy_value_network: Network to train
        config: Optional configuration (uses defaults if None)
        state_encoder: Optional state encoder function

    Returns:
        Configured ExpertIterationTrainer
    """
    if config is None:
        config = ExpertIterationConfig()

    return ExpertIterationTrainer(
        policy_value_network=policy_value_network,
        config=config,
        state_encoder=state_encoder,
    )
