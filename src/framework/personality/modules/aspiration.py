"""
Aspiration Module - Goal management and achievement standards.

Manages goal-setting, progress tracking, and achievement pursuit.
High aspiration = ambitious goals, persistent optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ..collections import BoundedHistory
from ..config import AspirationConfig
from ..profiles import PersonalityProfile

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class GoalTarget:
    """Target goal with aspiration-adjusted metrics.

    Attributes:
        goal_id: Unique goal identifier
        description: Goal description
        base_target: Original target value
        ambitious_target: Aspiration-adjusted target
        current_value: Current progress value
        created_at: When the goal was created
        priority: Goal priority [0.0, 1.0]
    """

    goal_id: str
    description: str
    base_target: float
    ambitious_target: float
    current_value: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    priority: float = 0.5


class AspirationModule:
    """Module for goal management and achievement standards.

    High aspiration = ambitious goals, persistent optimization.

    Implements:
    - PersonalityModuleProtocol
    - ConfidenceCalibrator

    Attributes:
        personality: Agent's personality profile
        config: Module configuration

    Example:
        >>> profile = PersonalityProfile(aspiration=0.9)
        >>> module = AspirationModule(profile)
        >>> target = module.set_goal_ambition_level(0.8)  # Returns ~1.0-1.4
    """

    def __init__(
        self,
        personality: PersonalityProfile,
        config: AspirationConfig | None = None,
    ) -> None:
        """Initialize aspiration module.

        Args:
            personality: Agent's personality profile
            config: Optional module configuration
        """
        self.personality = personality
        self.config = config or AspirationConfig()

        # Active goals
        self._active_goals: dict[str, GoalTarget] = {}

        # Achieved goals history
        self._achieved_goals: BoundedHistory[GoalTarget] = BoundedHistory(
            max_size=self.config.performance_history_size
        )

        # Performance history
        self._performance_history: BoundedHistory[float] = BoundedHistory(
            max_size=self.config.performance_history_size
        )

        # Current standards level
        self._current_standards: float = 0.5

        logger.info(
            "AspirationModule initialized with trait_value=%.2f",
            self.trait_value,
        )

    @property
    def module_name(self) -> str:
        """Module identifier."""
        return "aspiration"

    @property
    def trait_value(self) -> float:
        """Current aspiration trait value."""
        return self.personality.aspiration

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Add aspiration context before processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified context with aspiration information
        """
        if hasattr(context, "metadata"):
            context.metadata["aspiration_level"] = self.trait_value
            context.metadata["current_standards"] = self._current_standards
            context.metadata["active_goals_count"] = len(self._active_goals)

        return context

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Track performance after processing.

        Args:
            context: Original context
            result: Agent result to modify

        Returns:
            Modified result with aspiration tracking
        """
        # Track performance if confidence available
        if hasattr(result, "confidence"):
            self._performance_history.append(result.confidence)

        if hasattr(result, "metadata"):
            result.metadata["aspiration_influence"] = self.trait_value

        return result

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on aspiration.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        # Higher aspiration = more iterations/depth for better results
        if "max_iterations" in config:
            multiplier = 1.0 + self.trait_value * 0.5
            config["max_iterations"] = int(config["max_iterations"] * multiplier)

        return config

    def set_goal_ambition_level(self, base_goal_value: float) -> float:
        """Scale goal targets based on aspiration level.

        Higher aspiration = more ambitious targets.

        Args:
            base_goal_value: Base target value

        Returns:
            Aspiration-adjusted target value
        """
        # Aspiration amplifies goal targets
        ambition_multiplier = 1.0 + (self.trait_value * 0.5)  # Up to 1.5x
        return base_goal_value * ambition_multiplier

    def create_goal(
        self,
        goal_id: str,
        description: str,
        base_target: float,
        priority: float = 0.5,
    ) -> GoalTarget:
        """Create a new goal with aspiration-adjusted target.

        Args:
            goal_id: Unique identifier
            description: Goal description
            base_target: Base target value
            priority: Goal priority [0.0, 1.0]

        Returns:
            Created GoalTarget
        """
        if len(self._active_goals) >= self.config.max_active_goals:
            # Remove lowest priority goal
            min_goal = min(
                self._active_goals.values(),
                key=lambda g: g.priority,
            )
            del self._active_goals[min_goal.goal_id]
            logger.info("Removed lowest priority goal: %s", min_goal.goal_id)

        ambitious_target = self.set_goal_ambition_level(base_target)

        goal = GoalTarget(
            goal_id=goal_id,
            description=description,
            base_target=base_target,
            ambitious_target=ambitious_target,
            priority=priority,
        )

        self._active_goals[goal_id] = goal
        logger.info(
            "Created goal '%s': base=%.2f, ambitious=%.2f",
            goal_id,
            base_target,
            ambitious_target,
        )

        return goal

    def update_goal_progress(self, goal_id: str, new_value: float) -> bool:
        """Update progress on a goal.

        Args:
            goal_id: Goal identifier
            new_value: New progress value

        Returns:
            True if goal was found and updated
        """
        if goal_id not in self._active_goals:
            return False

        goal = self._active_goals[goal_id]
        goal.current_value = new_value

        # Check if goal achieved
        if new_value >= goal.ambitious_target:
            self._achieved_goals.append(goal)
            del self._active_goals[goal_id]
            logger.info(
                "Goal '%s' achieved! Value: %.2f >= Target: %.2f",
                goal_id,
                new_value,
                goal.ambitious_target,
            )

        return True

    def evaluate_progress_satisfaction(
        self,
        current_value: float,
        target_value: float,
    ) -> tuple[float, str]:
        """Assess satisfaction with current progress.

        High aspiration = higher standards for satisfaction.

        Args:
            current_value: Current progress value
            target_value: Target value

        Returns:
            Tuple of (satisfaction_score [0.0, 1.0], explanation)
        """
        if target_value <= 0:
            return 1.0, "Target is zero or negative"

        progress_ratio = current_value / target_value

        # Aspiration-adjusted satisfaction threshold
        satisfaction_threshold = 0.7 + (self.trait_value * 0.2)  # 0.7-0.9

        if progress_ratio >= satisfaction_threshold:
            explanation = (
                f"Progress satisfactory ({progress_ratio:.1%} of target). "
                f"Current achievement meets high standards."
            )
            return 1.0, explanation
        else:
            shortfall = satisfaction_threshold - progress_ratio
            satisfaction = progress_ratio / satisfaction_threshold
            explanation = (
                f"Progress ({progress_ratio:.1%}) below threshold "
                f"({satisfaction_threshold:.1%}). "
                f"Continued optimization required."
            )
            return satisfaction, explanation

    def calibrate_confidence(
        self,
        raw_confidence: float,
        context: dict[str, Any],
    ) -> float:
        """Adjust confidence based on aspiration.

        High aspiration = higher confidence requirements.

        Args:
            raw_confidence: Raw confidence score [0.0, 1.0]
            context: Decision context

        Returns:
            Calibrated confidence score [0.0, 1.0]
        """
        # Higher aspiration raises the bar for confidence
        threshold = 0.5 + (self.trait_value * 0.3)  # 0.5-0.8

        if raw_confidence >= threshold:
            # Above threshold: full confidence
            return raw_confidence
        else:
            # Below threshold: reduce confidence proportionally
            return raw_confidence * (raw_confidence / threshold)

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold.

        Returns:
            Minimum acceptable confidence [0.0, 1.0]
        """
        return 0.5 + (self.trait_value * 0.3)

    def should_raise_standards(
        self,
        window_size: int = 10,
    ) -> tuple[bool, str]:
        """Determine if performance standards should increase.

        High aspiration = proactive standard elevation.

        Args:
            window_size: Number of recent performances to consider

        Returns:
            Tuple of (should_raise, explanation)
        """
        recent = self._performance_history.get_recent(window_size)

        if len(recent) < window_size:
            return False, "Insufficient data for standard adjustment"

        avg_performance = float(np.mean(recent))
        performance_trend = float(np.polyfit(range(len(recent)), recent, 1)[0])

        # High aspiration triggers standard raises with positive trends
        raise_threshold = self.config.standard_raise_threshold + (
            self.trait_value * 0.15
        )

        if avg_performance > raise_threshold and performance_trend > 0:
            self._current_standards = min(
                1.0, self._current_standards + 0.1
            )
            explanation = (
                f"Consistent high performance ({avg_performance:.2f}). "
                f"Raising standards to {self._current_standards:.2f}. "
                f"Excellence requires perpetual refinement."
            )
            return True, explanation
        else:
            explanation = (
                f"Current standards appropriate for performance level "
                f"({avg_performance:.2f})."
            )
            return False, explanation

    def prioritize_goals(
        self,
        goals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank goals by importance and alignment with aspiration.

        Args:
            goals: List of goal dictionaries

        Returns:
            Sorted list of goals by priority
        """
        scored_goals: list[dict[str, Any]] = []

        for goal in goals:
            # Base score from goal attributes
            importance = goal.get("importance", 0.5)
            feasibility = goal.get("feasibility", 0.5)
            base_score = importance * feasibility

            # Aspiration boosts challenging goals
            difficulty = goal.get("difficulty", 0.5)
            aspiration_bonus = difficulty * self.trait_value * 0.3

            # Final score
            total_score = base_score + aspiration_bonus

            scored_goals.append(
                {
                    **goal,
                    "priority_score": total_score,
                }
            )

        # Sort by priority descending
        return sorted(
            scored_goals,
            key=lambda g: g["priority_score"],
            reverse=True,
        )

    def generate_stretch_goal(
        self,
        current_capabilities: dict[str, float],
    ) -> dict[str, Any]:
        """Create challenging but achievable stretch goal.

        Args:
            current_capabilities: Current capability levels

        Returns:
            Stretch goal dictionary
        """
        if not current_capabilities:
            return {
                "capability": "general",
                "current_level": 0.5,
                "target_level": 0.7,
                "improvement_required": 0.2,
                "rationale": "Default stretch goal for improvement.",
            }

        # Identify best capability
        best_capability = max(
            current_capabilities,
            key=lambda k: current_capabilities[k],
        )
        current_level = current_capabilities[best_capability]

        # Aspiration-scaled improvement target
        improvement_factor = 1.1 + (self.trait_value * 0.3)  # 1.1x to 1.4x
        target_level = min(1.0, current_level * improvement_factor)

        return {
            "capability": best_capability,
            "current_level": current_level,
            "target_level": target_level,
            "improvement_required": target_level - current_level,
            "rationale": (
                f"Pursuing excellence in {best_capability}. "
                f"Current proficiency ({current_level:.2f}) provides foundation "
                f"for advancement to {target_level:.2f}."
            ),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        recent_performance = self._performance_history.get_recent(10)
        avg_performance = (
            float(np.mean(recent_performance)) if recent_performance else 0.0
        )

        return {
            "trait_value": self.trait_value,
            "active_goals": len(self._active_goals),
            "achieved_goals": len(self._achieved_goals),
            "current_standards": self._current_standards,
            "average_recent_performance": avg_performance,
            "confidence_threshold": self.get_confidence_threshold(),
        }
