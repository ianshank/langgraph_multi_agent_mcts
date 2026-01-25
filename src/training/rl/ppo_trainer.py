"""
Proximal Policy Optimization (PPO) Trainer.

Implements the PPO algorithm for fine-tuning agents based on MCTS feedback.
"""

from typing import Any

import torch.nn as nn
import torch.optim as optim


class PPOTrainer:
    """
    PPO training loop for Neural Agent.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def compute_advantages(
        self, rewards: list[float], values: list[float], dones: list[bool], next_value: float
    ) -> list[float]:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        last_gae_lam = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - int(dones[step])) - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - int(dones[step])) * last_gae_lam
            advantages.insert(0, last_gae_lam)
            next_value = values[step]

        return advantages

    def update(self, rollouts: list[dict[str, Any]]):
        """
        Update policy and value networks using a batch of rollouts.

        Rollout expected format:
        {
            "states": Tensor,
            "actions": Tensor,
            "log_probs": Tensor (old),
            "rewards": List[float],
            "values": List[float], # Estimated by Critic during collection
            "dones": List[bool]
        }
        """
        # 1. Flatten Trajectories
        # For simplicity in this initial MVP, we assume `rollouts` is already a processed batch
        # of tensors or we process them here.

        pass  # To be fleshed out with specific Tensor handling based on agent output shape
