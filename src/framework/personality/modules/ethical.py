"""
Ethical Reasoning Module - Multi-framework ethical evaluation.

Evaluates decisions through ethical frameworks.
High ethical_weight = stronger constraints on acceptable actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ..collections import BoundedHistory
from ..config import EthicalConfig
from ..exceptions import EthicalViolationError, ErrorSeverity
from ..profiles import PersonalityProfile

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult

logger = logging.getLogger(__name__)


class EthicalFramework(str, Enum):
    """Supported ethical frameworks."""

    UTILITARIAN = "utilitarian"  # Maximize benefit, minimize harm
    DEONTOLOGICAL = "deontological"  # Rules and duties
    VIRTUE_ETHICS = "virtue_ethics"  # Character and virtues
    CARE_ETHICS = "care_ethics"  # Relationships and care


@dataclass
class EthicalAssessment:
    """Assessment of an action's ethical implications.

    Attributes:
        action: Action being evaluated
        framework_scores: Scores per ethical framework
        overall_score: Combined ethical score
        is_permitted: Whether action is ethically permitted
        violation_type: Type of violation if any
        reasoning: Explanation of assessment
        timestamp: When assessment was made
    """

    action: str
    framework_scores: dict[str, float]
    overall_score: float
    is_permitted: bool
    violation_type: str | None = None
    reasoning: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class DilemmaResolution:
    """Record of an ethical dilemma resolution.

    Attributes:
        options: Options that were considered
        chosen_option: Index of chosen option
        reasoning: Explanation for choice
        confidence: Confidence in resolution
        timestamp: When resolution was made
    """

    options: list[dict[str, Any]]
    chosen_option: int
    reasoning: str
    confidence: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EthicalReasoningModule:
    """Module for multi-framework ethical evaluation.

    High ethical_weight = stronger constraints on acceptable actions.

    Implements:
    - PersonalityModuleProtocol
    - EthicalEvaluator

    Attributes:
        personality: Agent's personality profile
        config: Module configuration

    Example:
        >>> profile = PersonalityProfile(ethical_weight=0.92)
        >>> module = EthicalReasoningModule(profile)
        >>> score, assessment = module.evaluate_action_ethics(
        ...     "action", {"context": "data"}, {"benefits": [], "harms": []}
        ... )
    """

    # Absolute prohibitions - actions that are never permitted
    PROHIBITIONS: list[tuple[str, str]] = [
        ("deception", "Intentional deception violates core commitment to truth"),
        ("harm_to_innocents", "Harm to innocent parties is impermissible"),
        ("rights_violation", "Fundamental rights must be respected"),
        ("exploitation", "Exploitative actions undermine dignity"),
        ("discrimination", "Discriminatory actions violate equality principles"),
    ]

    def __init__(
        self,
        personality: PersonalityProfile,
        config: EthicalConfig | None = None,
    ) -> None:
        """Initialize ethical reasoning module.

        Args:
            personality: Agent's personality profile
            config: Optional module configuration
        """
        self.personality = personality
        self.config = config or EthicalConfig()

        # Core principles with baseline weights
        self._principles: dict[str, float] = {
            "beneficence": 1.0,  # Maximize benefit
            "non_maleficence": 1.0,  # Minimize harm
            "autonomy": 0.9,  # Respect agency
            "justice": 0.95,  # Fair treatment
            "transparency": 0.88,  # Honest communication
            "accountability": 0.92,  # Take responsibility
        }

        # Violation history
        self._violation_history: BoundedHistory[EthicalAssessment] = (
            BoundedHistory(max_size=self.config.max_violation_history)
        )

        # Dilemma resolutions
        self._dilemma_resolutions: BoundedHistory[DilemmaResolution] = (
            BoundedHistory(max_size=self.config.max_dilemma_resolutions)
        )

        logger.info(
            "EthicalReasoningModule initialized with trait_value=%.2f",
            self.trait_value,
        )

    @property
    def module_name(self) -> str:
        """Module identifier."""
        return "ethical"

    @property
    def trait_value(self) -> float:
        """Current ethical weight trait value."""
        return self.personality.ethical_weight

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Add ethical context before processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified context with ethical information
        """
        if hasattr(context, "metadata"):
            context.metadata["ethical_weight"] = self.trait_value
            context.metadata["ethical_mode"] = (
                "strict" if self.config.strict_mode else "standard"
            )

        return context

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Evaluate ethical implications after processing.

        Args:
            context: Original context
            result: Agent result to modify

        Returns:
            Modified result with ethical assessment
        """
        if hasattr(result, "metadata"):
            result.metadata["ethical_influence"] = self.trait_value

        return result

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on ethical weight.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        # Higher ethical weight = more conservative behavior
        if "risk_threshold" in config:
            # Lower risk threshold for higher ethics
            config["risk_threshold"] = config["risk_threshold"] * (
                1.0 - self.trait_value * 0.3
            )

        return config

    def evaluate_action_ethics(
        self,
        action: str,
        context: dict[str, Any],
        consequences: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Comprehensive ethical evaluation of proposed action.

        Args:
            action: Action being evaluated
            context: Decision context
            consequences: Predicted consequences

        Returns:
            Tuple of (ethical_score [0.0, 1.0], detailed_assessment)
        """
        assessment: dict[str, Any] = {}

        # 1. Check for absolute prohibitions
        prohibited, prohibition_reason = self._check_prohibitions(
            action, context
        )
        if prohibited:
            assessment["prohibited"] = True
            assessment["reason"] = prohibition_reason

            # Record violation
            violation = EthicalAssessment(
                action=action,
                framework_scores={},
                overall_score=0.0,
                is_permitted=False,
                violation_type="prohibition",
                reasoning=prohibition_reason,
            )
            self._violation_history.append(violation)

            return 0.0, assessment

        # 2. Evaluate across frameworks
        consequence_score = self._evaluate_consequences(consequences)
        assessment["consequences"] = consequence_score

        duty_score = self._evaluate_duties(action, context)
        assessment["duties"] = duty_score

        virtue_score = self._evaluate_virtues(action, context)
        assessment["virtues"] = virtue_score

        # 3. Weighted combination
        ethical_score = (
            consequence_score * 0.4
            + duty_score * 0.35
            + virtue_score * 0.25
        )

        # 4. Apply ethical_weight scaling
        # High ethical_weight makes violations more costly
        if ethical_score < 0.5:
            penalty = (0.5 - ethical_score) * self.trait_value
            ethical_score = max(0.0, ethical_score - penalty)

        assessment["final_score"] = ethical_score
        assessment["ethical_weight_applied"] = self.trait_value

        # 5. Determine if permitted
        is_permitted = ethical_score >= 0.3  # Minimum threshold

        if self.config.strict_mode and ethical_score < 0.5:
            is_permitted = False

        assessment["is_permitted"] = is_permitted

        return ethical_score, assessment

    def _evaluate_consequences(
        self,
        consequences: dict[str, Any],
    ) -> float:
        """Evaluate based on outcomes (consequentialism).

        Args:
            consequences: Predicted consequences

        Returns:
            Consequence score [0.0, 1.0]
        """
        benefits = consequences.get("benefits", [])
        harms = consequences.get("harms", [])

        benefit_score = sum(
            b.get("magnitude", 0) for b in benefits
            if isinstance(b, dict)
        )
        harm_score = sum(
            h.get("magnitude", 0) for h in harms
            if isinstance(h, dict)
        )

        # Net utility
        total = benefit_score + harm_score
        if total == 0:
            return 0.5

        net_score = (benefit_score - harm_score) / (total + 1)
        return max(0.0, min(1.0, (net_score + 1) / 2))

    def _evaluate_duties(
        self,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """Evaluate based on moral duties (deontology).

        Args:
            action: Action being evaluated
            context: Decision context

        Returns:
            Duty score [0.0, 1.0]
        """
        duties = context.get("applicable_duties", [])

        if not duties:
            return 0.7  # Neutral if no specific duties

        fulfillment_scores: list[float] = []
        action_info = context.get("action_info", {})

        for duty in duties:
            fulfills = action_info.get("fulfills_duty", {}).get(duty, False)
            violates = action_info.get("violates_duty", {}).get(duty, False)

            if fulfills:
                fulfillment_scores.append(1.0)
            elif violates:
                fulfillment_scores.append(0.0)
            else:
                fulfillment_scores.append(0.5)  # Neutral

        return float(np.mean(fulfillment_scores)) if fulfillment_scores else 0.7

    def _evaluate_virtues(
        self,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """Evaluate based on virtuous character traits.

        Args:
            action: Action being evaluated
            context: Decision context

        Returns:
            Virtue score [0.0, 1.0]
        """
        virtues = ["honesty", "courage", "compassion", "integrity", "wisdom"]
        action_info = context.get("action_info", {})

        virtue_scores: list[float] = []
        for virtue in virtues:
            demonstrates = action_info.get("demonstrates_virtue", {}).get(
                virtue, 0.5
            )
            virtue_scores.append(demonstrates)

        return float(np.mean(virtue_scores))

    def _check_prohibitions(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Check for absolute ethical violations.

        Args:
            action: Action being evaluated
            context: Decision context

        Returns:
            Tuple of (is_prohibited, reason)
        """
        action_info = context.get("action_info", {})

        for prohibition, reason in self.PROHIBITIONS:
            if action_info.get(prohibition, False):
                return True, reason

        return False, None

    def check_ethical_constraints(
        self,
        action: str,
    ) -> tuple[bool, str | None]:
        """Check if action violates ethical constraints.

        Args:
            action: Action being evaluated

        Returns:
            Tuple of (is_allowed, violation_reason if not allowed)
        """
        # Simple keyword-based check for demonstration
        prohibited_terms = [
            "deceive",
            "lie",
            "harm",
            "exploit",
            "discriminate",
        ]

        action_lower = action.lower()
        for term in prohibited_terms:
            if term in action_lower:
                return False, f"Action contains prohibited term: {term}"

        return True, None

    def resolve_ethical_dilemma(
        self,
        options: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> tuple[int, str]:
        """Resolve conflict between competing ethical options.

        Args:
            options: List of option dictionaries with 'action' and 'consequences'
            context: Decision context

        Returns:
            Tuple of (selected_index, reasoning)
        """
        if not options:
            return -1, "No options provided"

        # Evaluate all options
        scores: list[float] = []
        assessments: list[dict[str, Any]] = []

        for option in options:
            score, assessment = self.evaluate_action_ethics(
                action=option.get("action", ""),
                context=context,
                consequences=option.get("consequences", {}),
            )
            scores.append(score)
            assessments.append(assessment)

        # Find best option
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]

        # High ethical_weight makes us more conservative
        certainty_threshold = 0.6 + (self.trait_value * 0.2)

        # Check if scores are too close
        if len(set(scores)) > 1:
            sorted_scores = sorted(scores, reverse=True)
            score_diff = sorted_scores[0] - sorted_scores[1]

            if score_diff < 0.1:
                # Too close to call - defer
                reasoning = (
                    f"Ethical considerations are closely matched "
                    f"(top scores: {sorted_scores[0]:.2f}, {sorted_scores[1]:.2f}). "
                    f"Given strong commitment to ethical action "
                    f"(weight: {self.trait_value:.2f}), human guidance recommended."
                )

                # Record dilemma
                resolution = DilemmaResolution(
                    options=options,
                    chosen_option=-1,  # Defer
                    reasoning=reasoning,
                    confidence=score_diff,
                )
                self._dilemma_resolutions.append(resolution)

                return -1, reasoning

        # Select best option
        chosen = max_indices[0]
        reasoning = (
            f"Option {chosen + 1} ethically preferable "
            f"(score: {scores[chosen]:.2f}). "
            f"Primary factors: {self._summarize_assessment(assessments[chosen])}. "
            f"Decision reflects {self.trait_value:.0%} weighting of ethics."
        )

        # Record resolution
        resolution = DilemmaResolution(
            options=options,
            chosen_option=chosen,
            reasoning=reasoning,
            confidence=max_score,
        )
        self._dilemma_resolutions.append(resolution)

        return chosen, reasoning

    def _summarize_assessment(self, assessment: dict[str, Any]) -> str:
        """Create human-readable assessment summary.

        Args:
            assessment: Assessment dictionary

        Returns:
            Summary string
        """
        parts: list[str] = []

        if assessment.get("consequences", 0) > 0.7:
            parts.append("positive consequences")
        if assessment.get("duties", 0) > 0.7:
            parts.append("duty fulfillment")
        if assessment.get("virtues", 0) > 0.7:
            parts.append("virtuous character")

        return ", ".join(parts) if parts else "balanced considerations"

    def generate_ethical_explanation(
        self,
        action: str,
        score: float,
        assessment: dict[str, Any],
    ) -> str:
        """Generate transparent explanation of ethical reasoning.

        Args:
            action: Action being explained
            score: Ethical score
            assessment: Assessment details

        Returns:
            Human-readable explanation
        """
        parts = [
            f"Ethical evaluation of '{action}': {score:.2f}/1.00",
            f"\nEthical framework weight: {self.trait_value:.0%}",
        ]

        # Framework scores
        if "consequences" in assessment:
            parts.append(
                f"\nConsequences: {assessment['consequences']:.2f} "
                "(benefit-harm analysis)"
            )

        if "duties" in assessment:
            parts.append(
                f"Duty compliance: {assessment['duties']:.2f} "
                "(deontological assessment)"
            )

        if "virtues" in assessment:
            parts.append(
                f"Character alignment: {assessment['virtues']:.2f} "
                "(virtue ethics)"
            )

        # Prohibitions
        if assessment.get("prohibited", False):
            parts.append(
                f"\n⚠️ ETHICAL PROHIBITION: {assessment.get('reason', 'Unknown')}"
            )

        # Conclusion
        if score >= 0.7:
            conclusion = "Action ethically sound and acceptable."
        elif score >= 0.4:
            conclusion = "Action ethically ambiguous; careful consideration required."
        else:
            conclusion = "Action raises significant ethical concerns."

        parts.append(f"\nConclusion: {conclusion}")

        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        return {
            "trait_value": self.trait_value,
            "strict_mode": self.config.strict_mode,
            "violations_recorded": len(self._violation_history),
            "dilemmas_resolved": len(self._dilemma_resolutions),
            "principles": self._principles,
        }
