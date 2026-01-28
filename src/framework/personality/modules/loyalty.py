"""
Loyalty Module - Goal persistence and behavioral consistency.

Ensures consistent goal pursuit and commitment to established objectives.
High loyalty = resistance to goal drift, consistent decision patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..collections import BoundedCounter, BoundedHistory
from ..config import LoyaltyConfig
from ..profiles import PersonalityProfile

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class GoalRecord:
    """Record of a committed goal.

    Attributes:
        goal: Goal identifier
        priority: Goal priority [0.0, 1.0]
        committed_at: When the goal was committed
        attempts: Number of attempts made
        status: Current status (active, achieved, abandoned)
    """

    goal: str
    priority: float
    committed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    attempts: int = 0
    status: str = "active"


class LoyaltyModule:
    """Module for goal persistence and consistency tracking.

    High loyalty = higher persistence threshold, more consistent actions.

    Implements:
    - PersonalityModuleProtocol
    - GoalTracker

    Attributes:
        personality: Agent's personality profile
        config: Module configuration

    Example:
        >>> profile = PersonalityProfile(loyalty=0.95)
        >>> module = LoyaltyModule(profile)
        >>> module.commit_to_goal("complete_task", priority=0.9)
        >>> alignment = module.evaluate_goal_alignment("work_on_task", ["complete_task"])
    """

    def __init__(
        self,
        personality: PersonalityProfile,
        config: LoyaltyConfig | None = None,
    ) -> None:
        """Initialize loyalty module.

        Args:
            personality: Agent's personality profile
            config: Optional module configuration
        """
        self.personality = personality
        self.config = config or LoyaltyConfig()

        # Goal tracking
        self._goal_history: BoundedHistory[GoalRecord] = BoundedHistory(
            max_size=self.config.max_goal_history
        )
        self._active_goals: dict[str, GoalRecord] = {}

        # Action consistency tracking
        self._action_counts: BoundedCounter = BoundedCounter(
            max_count=self.config.max_action_memory,
            max_keys=self.config.max_action_memory,
        )

        logger.info(
            "LoyaltyModule initialized with trait_value=%.2f",
            self.trait_value,
        )

    @property
    def module_name(self) -> str:
        """Module identifier."""
        return "loyalty"

    @property
    def trait_value(self) -> float:
        """Current loyalty trait value."""
        return self.personality.loyalty

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Add loyalty context before processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified context with loyalty information
        """
        # Add goal alignment info to context
        if hasattr(context, "metadata"):
            context.metadata["loyalty_score"] = self.trait_value
            context.metadata["active_goals"] = list(self._active_goals.keys())
            context.metadata["goal_persistence"] = self._calculate_persistence()

        return context

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Track action consistency after processing.

        Args:
            context: Original context
            result: Agent result to modify

        Returns:
            Modified result with loyalty tracking
        """
        # Track this action for consistency
        if hasattr(result, "response") and hasattr(context, "query"):
            action_key = f"{context.query[:50]}:{result.response[:50]}"
            try:
                self._action_counts.increment(action_key)
            except (ValueError, Exception) as e:
                logger.warning("Failed to track action: %s", e)

        # Add loyalty influence to metadata
        if hasattr(result, "metadata"):
            result.metadata["loyalty_influence"] = self.trait_value

        return result

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on loyalty.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        # Higher loyalty = more deterministic behavior
        if "temperature" in config:
            # Reduce temperature for high loyalty (more consistent)
            config["temperature"] = config["temperature"] * (
                1.0 - self.trait_value * 0.3
            )

        return config

    def commit_to_goal(
        self,
        goal: str,
        priority: float = 1.0,
    ) -> None:
        """Register commitment to a goal.

        Args:
            goal: Goal identifier
            priority: Goal priority [0.0, 1.0]
        """
        if not 0.0 <= priority <= 1.0:
            raise ValueError(f"Priority must be in [0.0, 1.0], got {priority}")

        record = GoalRecord(goal=goal, priority=priority)
        self._active_goals[goal] = record
        self._goal_history.append(record)

        logger.info(
            "Committed to goal '%s' with priority %.2f",
            goal,
            priority,
        )

    def evaluate_goal_alignment(
        self,
        action: str,
        current_goals: list[str],
    ) -> float:
        """Evaluate action alignment with committed goals.

        Args:
            action: Action being evaluated
            current_goals: Currently active goals

        Returns:
            Alignment score [0.0, 1.0]
        """
        if not self._active_goals:
            return 1.0  # No conflicts if no prior commitments

        # Check consistency with committed goals
        alignment_scores: list[float] = []

        for goal in self._active_goals:
            if goal in current_goals:
                # Goal still active - reward consistency
                alignment_scores.append(1.0)
            else:
                # Goal changed - penalize based on loyalty
                penalty = self.trait_value * 0.5
                alignment_scores.append(1.0 - penalty)

        # Weight by loyalty trait
        if not alignment_scores:
            return 1.0

        base_alignment = float(np.mean(alignment_scores))
        loyalty_weighted = (
            base_alignment * self.trait_value
            + (1.0 - self.trait_value) * 0.5
        )

        return loyalty_weighted

    def should_persist_on_goal(
        self,
        goal: str,
        difficulty: float,
        attempts: int,
    ) -> tuple[bool, str]:
        """Determine if agent should persist despite difficulty.

        High loyalty = higher persistence threshold.

        Args:
            goal: Goal identifier
            difficulty: Current difficulty score [0.0, 1.0]
            attempts: Number of attempts so far

        Returns:
            Tuple of (should_persist, explanation)
        """
        # Get goal record
        record = self._active_goals.get(goal)
        if not record:
            return True, "Goal not tracked, defaulting to persistence"

        # Loyalty-adjusted persistence threshold
        base_threshold = 5  # attempts before reconsidering
        loyalty_multiplier = 1.0 + (
            self.trait_value * self.config.persistence_multiplier
        )
        persistence_threshold = int(base_threshold * loyalty_multiplier)

        # Update attempts
        record.attempts = attempts

        if attempts < persistence_threshold:
            explanation = (
                f"Maintaining commitment to '{goal}' "
                f"(attempt {attempts}/{persistence_threshold}). "
                f"High goal fidelity ({self.trait_value:.0%}) "
                f"guides continued pursuit."
            )
            return True, explanation
        else:
            explanation = (
                f"Reconsidering approach to '{goal}' "
                f"after {attempts} attempts. "
                f"Loyalty ({self.trait_value:.0%}) does not "
                f"preclude strategic adaptation."
            )
            return False, explanation

    def mark_goal_achieved(self, goal: str) -> bool:
        """Mark a goal as achieved.

        Args:
            goal: Goal identifier

        Returns:
            True if goal was active, False otherwise
        """
        if goal in self._active_goals:
            self._active_goals[goal].status = "achieved"
            del self._active_goals[goal]
            logger.info("Goal '%s' marked as achieved", goal)
            return True
        return False

    def mark_goal_abandoned(self, goal: str, reason: str = "") -> bool:
        """Mark a goal as abandoned.

        Args:
            goal: Goal identifier
            reason: Reason for abandonment

        Returns:
            True if goal was active, False otherwise
        """
        if goal in self._active_goals:
            self._active_goals[goal].status = "abandoned"
            del self._active_goals[goal]
            logger.info(
                "Goal '%s' marked as abandoned: %s",
                goal,
                reason or "No reason provided",
            )
            return True
        return False

    def get_active_goals(self) -> list[str]:
        """Get list of active goals.

        Returns:
            List of active goal identifiers
        """
        return list(self._active_goals.keys())

    def get_consistency_score(self, context: str = "") -> float:
        """Get action consistency score for a context.

        Args:
            context: Optional context to filter by

        Returns:
            Consistency score [0.0, 1.0]
        """
        if self._action_counts.total_keys() == 0:
            return 1.0  # Perfect consistency if no history

        # Get top actions
        top_actions = self._action_counts.top_k(10)
        if not top_actions:
            return 1.0

        # Calculate concentration ratio
        total = self._action_counts.total_count()
        top_count = sum(count for _, count in top_actions)

        # Higher concentration = higher consistency
        consistency = top_count / total if total > 0 else 1.0

        # Weight by loyalty
        return consistency * self.trait_value + (1.0 - self.trait_value) * 0.5

    def _calculate_persistence(self) -> float:
        """Calculate overall persistence score.

        Returns:
            Persistence score [0.0, 1.0]
        """
        if not self._active_goals:
            return 1.0

        # Average attempts relative to threshold
        base_threshold = 5
        loyalty_multiplier = 1.0 + (
            self.trait_value * self.config.persistence_multiplier
        )

        persistence_scores: list[float] = []
        for record in self._active_goals.values():
            threshold = base_threshold * loyalty_multiplier
            persistence = 1.0 - (record.attempts / (threshold * 2))
            persistence_scores.append(max(0.0, min(1.0, persistence)))

        return float(np.mean(persistence_scores)) if persistence_scores else 1.0

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        return {
            "trait_value": self.trait_value,
            "active_goals": len(self._active_goals),
            "total_goals_tracked": len(self._goal_history),
            "action_patterns_tracked": self._action_counts.total_keys(),
            "consistency_score": self.get_consistency_score(),
            "persistence_score": self._calculate_persistence(),
        }
