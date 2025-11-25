"""
Curiosity Module - Intrinsic motivation and exploration.

Enhances MCTS with curiosity-driven intrinsic rewards.
High curiosity = stronger exploration of novel states.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..collections import AsyncBoundedCache, BoundedCounter, BoundedHistory
from ..config import CuriosityConfig
from ..profiles import PersonalityProfile

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class NoveltyRecord:
    """Record of state novelty evaluation.

    Attributes:
        state_hash: Hash of the state
        novelty_score: Computed novelty [0.0, 1.0]
        visit_count: Number of times visited
        intrinsic_reward: Computed intrinsic reward
    """

    state_hash: str
    novelty_score: float
    visit_count: int
    intrinsic_reward: float


class CuriosityModule:
    """Module for curiosity-driven exploration.

    Implements intrinsic motivation through:
    - Visit-based novelty (episodic)
    - Prediction-error novelty (RND-style)
    - UCB exploration bonuses

    Implements:
    - PersonalityModuleProtocol
    - MCTSInfluencer
    - PromptAugmenter

    Attributes:
        personality: Agent's personality profile
        config: Module configuration

    Example:
        >>> profile = PersonalityProfile(curiosity=0.85)
        >>> module = CuriosityModule(profile)
        >>> reward = module.compute_intrinsic_reward("state_hash", "action", 0.5)
    """

    def __init__(
        self,
        personality: PersonalityProfile,
        config: CuriosityConfig | None = None,
    ) -> None:
        """Initialize curiosity module.

        Args:
            personality: Agent's personality profile
            config: Optional module configuration
        """
        self.personality = personality
        self.config = config or CuriosityConfig()

        # State visit tracking
        self._state_visits: BoundedCounter = BoundedCounter(
            max_count=1_000_000,
            max_keys=self.config.max_state_memory,
        )

        # Novelty history
        self._novelty_history: BoundedHistory[NoveltyRecord] = BoundedHistory(
            max_size=10000
        )

        # Async cache for expensive computations
        self._novelty_cache: AsyncBoundedCache[float] = AsyncBoundedCache(
            max_size=self.config.max_state_memory,
            ttl_seconds=3600,  # 1 hour TTL
        )

        # Intrinsic reward statistics
        self._total_intrinsic_reward: float = 0.0
        self._reward_count: int = 0

        # Known state embeddings for RND-style novelty
        self._known_embeddings: list[NDArray[np.float32]] = []

        logger.info(
            "CuriosityModule initialized with trait_value=%.2f",
            self.trait_value,
        )

    @property
    def module_name(self) -> str:
        """Module identifier."""
        return "curiosity"

    @property
    def trait_value(self) -> float:
        """Current curiosity trait value."""
        return self.personality.curiosity

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Add curiosity context before processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified context with curiosity information
        """
        if hasattr(context, "metadata"):
            context.metadata["curiosity_level"] = self.trait_value
            context.metadata["exploration_enabled"] = self.trait_value > 0.5

        return context

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Track exploration after processing.

        Args:
            context: Original context
            result: Agent result to modify

        Returns:
            Modified result with curiosity tracking
        """
        if hasattr(result, "metadata"):
            result.metadata["curiosity_influence"] = self.trait_value

        return result

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on curiosity.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        # Higher curiosity = higher temperature (more exploration)
        if "temperature" in config:
            config["temperature"] = config["temperature"] * (
                1.0 + self.trait_value * 0.5
            )

        return config

    def compute_intrinsic_reward(
        self,
        state_hash: str,
        action: str,
        uncertainty: float = 0.5,
    ) -> float:
        """Compute curiosity-driven intrinsic reward.

        Novel states receive higher rewards.

        Args:
            state_hash: Hash of current state
            action: Action being evaluated
            uncertainty: Model uncertainty for this state-action [0.0, 1.0]

        Returns:
            Intrinsic reward value
        """
        # Get visit count for this state
        visit_count = self._state_visits.get(state_hash)

        # Increment visit count
        try:
            self._state_visits.increment(state_hash)
        except (ValueError, Exception) as e:
            logger.warning("Failed to increment state visits: %s", e)

        # Visit-based novelty (episodic) - decays with visits
        visit_novelty = 1.0 / (1.0 + visit_count)

        # Uncertainty-based novelty
        uncertainty_bonus = uncertainty * 0.5

        # Combine novelties, weighted by curiosity trait
        intrinsic_reward = self.trait_value * (
            visit_novelty * 0.6
            + uncertainty_bonus * 0.4
            + (1.0 - self.trait_value) * 0.1  # Small baseline
        )

        # Apply scale
        intrinsic_reward *= self.config.intrinsic_reward_scale

        # Track statistics
        self._total_intrinsic_reward += intrinsic_reward
        self._reward_count += 1

        # Record for analysis
        record = NoveltyRecord(
            state_hash=state_hash,
            novelty_score=visit_novelty,
            visit_count=visit_count + 1,
            intrinsic_reward=intrinsic_reward,
        )
        self._novelty_history.append(record)

        return intrinsic_reward

    def get_exploration_bonus(
        self,
        base_weight: float,
        iteration: int,
        state_novelty: float,
    ) -> float:
        """Calculate exploration bonus for MCTS UCB.

        Args:
            base_weight: Base UCB exploration weight
            iteration: Current MCTS iteration
            state_novelty: Novelty score of current state [0.0, 1.0]

        Returns:
            Modified exploration weight
        """
        # Decay exploration over iterations
        decay = self.config.exploration_bonus_decay ** iteration

        # Curiosity-driven bonus
        curiosity_bonus = self.trait_value * state_novelty * decay

        # Final weight
        return base_weight * (1.0 + curiosity_bonus)

    def modify_rollout_policy(
        self,
        policy: str,
        context: dict[str, Any],
    ) -> str:
        """Modify MCTS rollout policy based on curiosity.

        Args:
            policy: Current policy name
            context: Decision context

        Returns:
            Modified policy name
        """
        # High curiosity favors more random/exploratory policies
        if self.trait_value > 0.8:
            return "random"  # Maximum exploration
        elif self.trait_value > 0.5:
            return "hybrid"  # Mix of random and greedy
        else:
            return policy  # Keep current policy

    def adjusted_ucb_score(
        self,
        node_value: float,
        node_visits: int,
        parent_visits: int,
        intrinsic_reward: float,
        exploration_weight: float = 1.414,
    ) -> float:
        """UCB1 with curiosity-driven exploration bonus.

        Args:
            node_value: Node value estimate
            node_visits: Visits to this node
            parent_visits: Visits to parent node
            intrinsic_reward: Intrinsic reward for this node
            exploration_weight: Base exploration weight

        Returns:
            UCB score with curiosity bonus
        """
        if node_visits == 0:
            return float("inf")  # Unexplored nodes prioritized

        # Standard UCB exploitation term
        exploitation = node_value / node_visits

        # Exploration term
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / node_visits
        )

        # Curiosity-driven bonus
        curiosity_bonus = self.trait_value * intrinsic_reward

        return exploitation + exploration + curiosity_bonus

    async def augment_prompt(
        self,
        base_prompt: str,
        context: AgentContext,
    ) -> str:
        """Add curiosity-specific prompt instructions.

        Args:
            base_prompt: Original prompt text
            context: Agent context

        Returns:
            Augmented prompt
        """
        if self.trait_value > 0.7:
            augmentation = (
                "\n\nExplore creative and unconventional approaches. "
                "Consider novel perspectives and unexpected solutions."
            )
            return base_prompt + augmentation
        elif self.trait_value > 0.5:
            augmentation = (
                "\n\nBe open to exploring different approaches while "
                "maintaining focus on the core objective."
            )
            return base_prompt + augmentation

        return base_prompt

    def get_system_message_suffix(self) -> str:
        """Get suffix to add to system message.

        Returns:
            System message suffix
        """
        if self.trait_value > 0.8:
            return (
                "You are naturally curious and enjoy exploring "
                "unconventional ideas."
            )
        elif self.trait_value > 0.5:
            return "You balance focused work with openness to new ideas."
        return ""

    def compute_prediction_error_novelty(
        self,
        state_embedding: NDArray[np.float32],
    ) -> float:
        """Compute prediction-based novelty using RND-style approach.

        Higher error = more novel state.

        Args:
            state_embedding: State embedding vector

        Returns:
            Novelty score [0.0, 1.0]
        """
        if len(self._known_embeddings) == 0:
            self._known_embeddings.append(state_embedding)
            return 1.0  # First state is maximally novel

        # Compute similarity to known states
        similarities: list[float] = []
        for known in self._known_embeddings:
            # Cosine similarity
            dot_product = float(np.dot(state_embedding, known))
            norm_product = float(
                np.linalg.norm(state_embedding) * np.linalg.norm(known)
            )
            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0.0
            similarities.append(similarity)

        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity  # High novelty if dissimilar

        # Store if sufficiently novel
        if novelty > self.config.novelty_threshold:
            if len(self._known_embeddings) < 1000:  # Cap storage
                self._known_embeddings.append(state_embedding)

        return novelty

    def _hash_state(self, state: Any) -> str:
        """Create hashable state representation.

        Args:
            state: State to hash

        Returns:
            Hash string
        """
        state_str = str(state)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        avg_reward = (
            self._total_intrinsic_reward / self._reward_count
            if self._reward_count > 0
            else 0.0
        )

        return {
            "trait_value": self.trait_value,
            "unique_states_visited": self._state_visits.total_keys(),
            "total_visits": self._state_visits.total_count(),
            "average_intrinsic_reward": avg_reward,
            "novelty_records": len(self._novelty_history),
            "known_embeddings": len(self._known_embeddings),
            "cache_hit_rate": self._novelty_cache.hit_rate,
        }
