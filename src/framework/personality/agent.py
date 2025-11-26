"""
Personality-Driven Agent - Integration class combining all personality modules.

Wraps any AsyncAgentBase with personality-driven behavior using the
decorator pattern.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import PersonalityConfig
from .profiles import PersonalityProfile
from .protocols import PersonalityModuleProtocol

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult, AsyncAgentBase

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Record of a personality-influenced decision.

    Attributes:
        timestamp: When the decision was made
        state: State at decision time
        action: Chosen action
        scores: Score breakdown by factor
        explanation: Human-readable explanation
        duration_ms: Processing time in milliseconds
    """

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    state: Any = None
    action: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    duration_ms: float = 0.0


class PersonalityDrivenAgent:
    """Multi-agent system with integrated personality traits.

    Wraps an existing AsyncAgentBase and applies personality
    modules to influence behavior through pre/post-processing hooks.

    Implements decorator pattern - can be used wherever the
    base agent is expected.

    Attributes:
        base_agent: The wrapped agent
        personality: Personality profile
        modules: Dictionary of personality modules

    Example:
        >>> agent = PersonalityDrivenAgent(
        ...     base_agent=hrm_agent,
        ...     personality=PersonalityProfile(loyalty=0.95, curiosity=0.85),
        ... )
        >>> result = await agent.process(context)
    """

    def __init__(
        self,
        base_agent: AsyncAgentBase,
        personality: PersonalityProfile,
        modules: dict[str, PersonalityModuleProtocol] | None = None,
        config: PersonalityConfig | None = None,
    ) -> None:
        """Initialize personality-driven agent.

        Args:
            base_agent: The agent to wrap
            personality: Personality profile
            modules: Pre-created personality modules
            config: Personality configuration
        """
        self.base_agent = base_agent
        self.personality = personality
        self.config = config or PersonalityConfig()

        # Create modules if not provided
        if modules is None:
            from .factory import PersonalityFactory

            modules = PersonalityFactory.create_modules(personality, self.config)

        self.modules = modules

        # Decision history
        self._decision_history: list[DecisionRecord] = []

        # Current goals (from loyalty module)
        self._current_goals: list[str] = []

        # Delegate attributes from base agent
        self.name = f"Personality[{getattr(base_agent, 'name', 'Agent')}]"

        logger.info(
            "PersonalityDrivenAgent initialized: name=%s, "
            "loyalty=%.2f, curiosity=%.2f, aspiration=%.2f, "
            "ethical=%.2f, transparency=%.2f",
            self.name,
            personality.loyalty,
            personality.curiosity,
            personality.aspiration,
            personality.ethical_weight,
            personality.transparency,
        )

    # ==================== AsyncAgentBase Protocol ====================

    async def initialize(self) -> None:
        """Initialize base agent."""
        if hasattr(self.base_agent, "initialize"):
            await self.base_agent.initialize()

    async def shutdown(self) -> None:
        """Shutdown base agent."""
        if hasattr(self.base_agent, "shutdown"):
            await self.base_agent.shutdown()

    async def process(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """Process query with personality influence.

        Args:
            context: Agent context

        Returns:
            AgentResult with personality influence
        """
        start_time = datetime.now(timezone.utc)

        # Apply pre-processing hooks
        modified_context = await self._apply_pre_process_hooks(context)

        # Process with base agent
        result = await self.base_agent.process(modified_context)

        # Apply post-processing hooks
        modified_result = await self._apply_post_process_hooks(
            modified_context, result
        )

        # Record decision
        duration_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        self._record_decision(modified_context, modified_result, duration_ms)

        return modified_result

    async def pre_process(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Apply personality pre-processing.

        Args:
            context: Agent context

        Returns:
            Modified context
        """
        # First base agent's pre-processing
        if hasattr(self.base_agent, "pre_process"):
            context = await self.base_agent.pre_process(context)

        # Then personality modules
        return await self._apply_pre_process_hooks(context)

    async def post_process(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Apply personality post-processing.

        Args:
            context: Original context
            result: Agent result

        Returns:
            Modified result
        """
        # First personality modules
        result = await self._apply_post_process_hooks(context, result)

        # Then base agent's post-processing
        if hasattr(self.base_agent, "post_process"):
            result = await self.base_agent.post_process(context, result)

        return result

    # ==================== Personality-Specific Methods ====================

    async def make_decision(
        self,
        state: Any,
        available_actions: list[Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        """Make personality-driven decision.

        Integrates all personality dimensions into decision-making.

        Args:
            state: Current state
            available_actions: Available actions
            context: Optional decision context

        Returns:
            Tuple of (chosen_action, explanation)
        """
        if not available_actions:
            return None, "No actions available"

        context = context or {}
        decision_start = datetime.now(timezone.utc)

        # 1. Get scores from each personality dimension
        scores: dict[str, dict[Any, float]] = {}

        # Curiosity-driven novelty scores
        curiosity_module = self.modules.get("curiosity")
        if curiosity_module and hasattr(curiosity_module, "compute_intrinsic_reward"):
            novelty_scores = {}
            for action in available_actions:
                state_hash = str(hash(f"{state}_{action}"))
                novelty = curiosity_module.compute_intrinsic_reward(
                    state_hash, str(action), 0.5
                )
                novelty_scores[action] = novelty
            scores["curiosity"] = novelty_scores

        # Ethical evaluation
        ethical_module = self.modules.get("ethical")
        if ethical_module and hasattr(ethical_module, "evaluate_action_ethics"):
            ethical_scores = {}
            for action in available_actions:
                score, _ = ethical_module.evaluate_action_ethics(
                    action=str(action),
                    context=context,
                    consequences=context.get("consequences", {}),
                )
                ethical_scores[action] = score
            scores["ethical"] = ethical_scores

        # Loyalty/goal alignment
        loyalty_module = self.modules.get("loyalty")
        if loyalty_module and hasattr(loyalty_module, "evaluate_goal_alignment"):
            alignment_scores = {}
            for action in available_actions:
                alignment = loyalty_module.evaluate_goal_alignment(
                    str(action), self._current_goals
                )
                alignment_scores[action] = alignment
            scores["loyalty"] = alignment_scores

        # Aspiration-based value estimation
        aspiration_module = self.modules.get("aspiration")
        if aspiration_module and hasattr(aspiration_module, "set_goal_ambition_level"):
            aspiration_scores = {}
            for action in available_actions:
                base_value = context.get("action_values", {}).get(action, 0.5)
                ambitious_value = aspiration_module.set_goal_ambition_level(
                    base_value
                )
                aspiration_scores[action] = ambitious_value
            scores["aspiration"] = aspiration_scores

        # 2. Combine scores with personality weights
        final_scores: dict[Any, float] = {}

        for action in available_actions:
            combined_score = 0.0

            # Get score for each dimension
            curiosity_score = scores.get("curiosity", {}).get(action, 0.5)
            ethical_score = scores.get("ethical", {}).get(action, 1.0)
            loyalty_score = scores.get("loyalty", {}).get(action, 1.0)
            aspiration_score = scores.get("aspiration", {}).get(action, 0.5)

            # Weighted combination
            combined_score = (
                aspiration_score * 0.25
                + ethical_score * self.personality.ethical_weight * 0.35
                + loyalty_score * self.personality.loyalty * 0.20
                + curiosity_score * self.personality.curiosity * 0.20
            )

            final_scores[action] = combined_score

        # 3. Select best action
        best_action = max(final_scores, key=lambda a: final_scores[a])

        # 4. Generate explanation
        transparency_module = self.modules.get("transparency")
        explanation = self._generate_decision_explanation(
            best_action,
            final_scores,
            scores,
        )

        # 5. Record decision
        duration_ms = (
            datetime.now(timezone.utc) - decision_start
        ).total_seconds() * 1000

        record = DecisionRecord(
            state=state,
            action=str(best_action),
            scores={"final": final_scores[best_action], **scores},
            explanation=explanation,
            duration_ms=duration_ms,
        )
        self._decision_history.append(record)

        return best_action, explanation

    async def set_goal(
        self,
        goal: str,
        priority: float = 1.0,
    ) -> str:
        """Establish new goal with loyalty commitment.

        Args:
            goal: Goal description
            priority: Goal priority [0.0, 1.0]

        Returns:
            Explanation of commitment
        """
        loyalty_module = self.modules.get("loyalty")
        if loyalty_module and hasattr(loyalty_module, "commit_to_goal"):
            loyalty_module.commit_to_goal(goal, priority)
            self._current_goals.append(goal)

        aspiration_module = self.modules.get("aspiration")
        ambitious_target = priority
        if aspiration_module and hasattr(aspiration_module, "set_goal_ambition_level"):
            ambitious_target = aspiration_module.set_goal_ambition_level(priority)

        explanation = (
            f"Goal established: '{goal}' (priority: {priority:.2f}, "
            f"aspiration-adjusted: {ambitious_target:.2f}). "
            f"High loyalty ({self.personality.loyalty:.0%}) ensures consistent pursuit. "
            f"Ambitious standards ({self.personality.aspiration:.0%}) drive excellence."
        )

        return explanation

    async def reflect_on_performance(
        self,
        recent_results: list[float],
    ) -> str:
        """Analyze performance and adjust standards.

        Args:
            recent_results: Recent performance scores

        Returns:
            Performance reflection summary
        """
        parts: list[str] = [
            "Performance reflection:",
            f"Recent results: {np.mean(recent_results):.2f} avg over {len(recent_results)} episodes",
        ]

        # Check aspiration module for standards
        aspiration_module = self.modules.get("aspiration")
        if aspiration_module and hasattr(aspiration_module, "should_raise_standards"):
            # Update performance history
            for result in recent_results:
                aspiration_module._performance_history.append(result)

            should_raise, explanation = aspiration_module.should_raise_standards()
            parts.append(explanation)

        # Check loyalty for goal persistence
        loyalty_module = self.modules.get("loyalty")
        if loyalty_module and hasattr(loyalty_module, "should_persist_on_goal"):
            for goal in self._current_goals[:3]:
                should_persist, persist_explanation = loyalty_module.should_persist_on_goal(
                    goal, difficulty=0.5, attempts=5
                )
                parts.append(persist_explanation)

        return "\n\n".join(parts)

    def generate_personality_report(self) -> str:
        """Generate transparent report of personality configuration.

        Returns:
            Formatted personality report
        """
        parts = [
            "═" * 60,
            "PERSONALITY PROFILE REPORT",
            "═" * 60,
            "\nCore Traits:",
            f"  Loyalty:         {self.personality.loyalty:.0%}  │ Goal persistence",
            f"  Curiosity:       {self.personality.curiosity:.0%}  │ Exploration drive",
            f"  Aspiration:      {self.personality.aspiration:.0%}  │ Goal ambition",
            f"  Ethical Weight:  {self.personality.ethical_weight:.0%}  │ Ethics priority",
            f"  Transparency:    {self.personality.transparency:.0%}  │ Explainability",
            "\nBehavioral Implications:",
        ]

        # Calculate behavioral implications
        persistence_threshold = int(5 * (1 + self.personality.loyalty * 2))
        exploration_bonus = int(self.personality.curiosity * 100)
        standards = int(100 + self.personality.aspiration * 50)

        parts.extend(
            [
                f"  • Goal persistence threshold: {persistence_threshold} attempts",
                f"  • Exploration bonus: {exploration_bonus}% weight in selection",
                f"  • Performance standards: {standards}% of baseline",
                f"  • Ethical violation penalty: {int(self.personality.ethical_weight * 100)}% score reduction",
                f"  • Explanation depth: {'Comprehensive' if self.personality.transparency >= 0.8 else 'Standard'}",
                "\nDecision History:",
                f"  • Total decisions made: {len(self._decision_history)}",
                f"  • Goals committed: {len(self._current_goals)}",
            ]
        )

        # Module statistics
        for name, module in self.modules.items():
            if hasattr(module, "get_stats"):
                stats = module.get_stats()
                parts.append(f"\n{name.upper()} Module Stats:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        parts.append(f"  • {key}: {value:.2f}")
                    else:
                        parts.append(f"  • {key}: {value}")

        parts.append("═" * 60)
        return "\n".join(parts)

    # ==================== Private Helper Methods ====================

    async def _apply_pre_process_hooks(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Apply all personality pre-process hooks.

        Args:
            context: Agent context

        Returns:
            Modified context
        """
        for module in self.modules.values():
            if hasattr(module, "pre_process_hook"):
                context = await module.pre_process_hook(context)

        return context

    async def _apply_post_process_hooks(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Apply all personality post-process hooks.

        Args:
            context: Original context
            result: Agent result

        Returns:
            Modified result
        """
        for module in self.modules.values():
            if hasattr(module, "post_process_hook"):
                result = await module.post_process_hook(context, result)

        return result

    def _record_decision(
        self,
        context: AgentContext,
        result: AgentResult,
        duration_ms: float,
    ) -> None:
        """Record decision for audit/analysis.

        Args:
            context: Decision context
            result: Decision result
            duration_ms: Processing duration
        """
        record = DecisionRecord(
            state=getattr(context, "query", str(context)),
            action=getattr(result, "response", str(result))[:100],
            scores={
                "confidence": getattr(result, "confidence", 0.0),
            },
            duration_ms=duration_ms,
        )
        self._decision_history.append(record)

    def _generate_decision_explanation(
        self,
        best_action: Any,
        final_scores: dict[Any, float],
        component_scores: dict[str, dict[Any, float]],
    ) -> str:
        """Generate explanation for decision.

        Args:
            best_action: Selected action
            final_scores: Final scores per action
            component_scores: Scores per personality dimension

        Returns:
            Human-readable explanation
        """
        parts = [
            f"Selected action: {best_action}",
            f"Score: {final_scores[best_action]:.3f}",
            "\nPersonality influence breakdown:",
        ]

        for dimension, scores in component_scores.items():
            if best_action in scores:
                parts.append(f"  • {dimension}: {scores[best_action]:.3f}")

        # Add alternatives
        sorted_actions = sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )
        if len(sorted_actions) > 1:
            parts.append("\nAlternatives considered:")
            for action, score in sorted_actions[1:4]:  # Top 3 alternatives
                parts.append(f"  • {action}: {score:.3f}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PersonalityDrivenAgent("
            f"name={self.name}, "
            f"loyalty={self.personality.loyalty:.2f}, "
            f"curiosity={self.personality.curiosity:.2f})"
        )
