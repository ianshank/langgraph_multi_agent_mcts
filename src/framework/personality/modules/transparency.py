"""
Transparency Module - Decision explainability and logging.

Ensures decisions are explainable and reasoning is accessible.
High transparency = detailed explanations, reasoning visibility.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from ..collections import BoundedHistory, TimeAwareBoundedHistory
from ..config import TransparencyConfig
from ..profiles import PersonalityProfile

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult

logger = logging.getLogger(__name__)

ExplanationLevel = Literal["basic", "detailed", "expert"]


@dataclass
class PIIPattern:
    """Pattern for PII detection.

    Attributes:
        name: Pattern name
        pattern: Regex pattern
        replacement: Replacement text
    """

    name: str
    pattern: re.Pattern[str]
    replacement: str


@dataclass
class DecisionLog:
    """Record of a single decision.

    Attributes:
        decision_id: Unique identifier
        timestamp: When decision was made
        state_id: State identifier
        action: Chosen action
        rationale: Structured explanation
        confidence: Confidence score
        metadata: Additional context
    """

    decision_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    state_id: str = ""
    action: str = ""
    rationale: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class TransparencyModule:
    """Module for decision explainability and logging.

    High transparency = detailed explanations, reasoning visibility.

    Implements:
    - PersonalityModuleProtocol
    - ExplainabilityProvider

    Attributes:
        personality: Agent's personality profile
        config: Module configuration

    Example:
        >>> profile = PersonalityProfile(transparency=0.88)
        >>> module = TransparencyModule(profile)
        >>> log = await module.log_decision(
        ...     state_id="s1",
        ...     action="explore",
        ...     rationale={"score": 0.8},
        ...     confidence=0.9,
        ... )
    """

    # PII patterns for masking
    PII_PATTERNS: list[PIIPattern] = [
        PIIPattern(
            name="email",
            pattern=re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ),
            replacement="[REDACTED_EMAIL]",
        ),
        PIIPattern(
            name="phone",
            pattern=re.compile(
                r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b"
            ),
            replacement="[REDACTED_PHONE]",
        ),
        PIIPattern(
            name="ssn",
            pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            replacement="[REDACTED_SSN]",
        ),
        PIIPattern(
            name="ip_address",
            pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            replacement="[REDACTED_IP]",
        ),
    ]

    def __init__(
        self,
        personality: PersonalityProfile,
        config: TransparencyConfig | None = None,
    ) -> None:
        """Initialize transparency module.

        Args:
            personality: Agent's personality profile
            config: Optional module configuration
        """
        self.personality = personality
        self.config = config or TransparencyConfig()

        # Decision logs with time-based retention
        self._decision_logs: TimeAwareBoundedHistory[DecisionLog] = (
            TimeAwareBoundedHistory(
                max_size=self.config.max_decision_log_size,
                retention_period=timedelta(days=self.config.log_retention_days),
            )
        )

        # Access log for audit
        self._access_log: BoundedHistory[dict[str, Any]] = BoundedHistory(
            max_size=1000
        )

        # Explanation templates
        self._templates: dict[str, str] = self._load_templates()

        logger.info(
            "TransparencyModule initialized with trait_value=%.2f",
            self.trait_value,
        )

    @property
    def module_name(self) -> str:
        """Module identifier."""
        return "transparency"

    @property
    def trait_value(self) -> float:
        """Current transparency trait value."""
        return self.personality.transparency

    def _load_templates(self) -> dict[str, str]:
        """Load explanation templates.

        Returns:
            Dictionary of template strings
        """
        return {
            "action_selection": (
                "Selected {action} based on {primary_factor} "
                "(weight: {weight:.2f}). "
                "Alternative options: {alternatives}. "
                "Key factors: {factors}."
            ),
            "goal_prioritization": (
                "Prioritized {goal} due to {rationale}. "
                "Progress: {progress:.1%}. "
                "Expected outcome: {outcome}."
            ),
            "ethical_decision": (
                "Ethical analysis: {summary}. "
                "Score: {score:.2f}. "
                "Framework: {framework}. "
                "Principles: {principles}."
            ),
            "uncertainty": (
                "Confidence: {confidence:.1%}. "
                "Uncertainty sources: {uncertainty_factors}. "
                "Mitigation: {strategies}."
            ),
        }

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Add transparency context before processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified context with transparency information
        """
        if hasattr(context, "metadata"):
            context.metadata["transparency_level"] = self.trait_value
            context.metadata["log_decisions"] = self.trait_value >= 0.5

        return context

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Log decision after processing.

        Args:
            context: Original context
            result: Agent result to modify

        Returns:
            Modified result with transparency tracking
        """
        if hasattr(result, "metadata"):
            result.metadata["transparency_influence"] = self.trait_value

        return result

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on transparency.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        # Higher transparency = more verbose output
        if "verbosity" in config:
            if self.trait_value > 0.8:
                config["verbosity"] = "high"
            elif self.trait_value > 0.5:
                config["verbosity"] = "medium"

        return config

    async def log_decision(
        self,
        state_id: str,
        action: str,
        rationale: dict[str, Any],
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> DecisionLog | None:
        """Log a decision with full context.

        Only logs if personality.transparency >= 0.5.

        Args:
            state_id: State identifier
            action: Chosen action
            rationale: Structured explanation
            confidence: Decision confidence [0.0, 1.0]
            metadata: Additional context

        Returns:
            DecisionLog if logged, None otherwise
        """
        # Only log if transparency is sufficient
        if self.trait_value < 0.5:
            return None

        # Validate inputs
        if not action or not action.strip():
            raise ValueError("Action cannot be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")

        # Mask PII if enabled
        if self.config.enable_pii_masking:
            action = self._mask_pii(action)
            rationale = self._mask_pii_dict(rationale)

        # Create log entry
        log_entry = DecisionLog(
            state_id=state_id,
            action=action,
            rationale=dict(rationale),
            confidence=confidence,
            metadata=dict(metadata) if metadata else {},
        )

        # Store
        self._decision_logs.append(log_entry)

        logger.debug(
            "Decision logged: id=%s, action=%s, confidence=%.2f",
            log_entry.decision_id,
            action,
            confidence,
        )

        return log_entry

    async def generate_explanation(
        self,
        decision: dict[str, Any],
        context: AgentContext,
        verbosity: str = "moderate",
    ) -> str:
        """Generate human-readable explanation of a decision.

        Args:
            decision: Decision data to explain
            context: Agent context
            verbosity: Detail level (brief, moderate, detailed)

        Returns:
            Human-readable explanation
        """
        # Map verbosity to level
        level_map = {
            "brief": "basic",
            "moderate": "detailed",
            "detailed": "expert",
        }
        level: ExplanationLevel = level_map.get(verbosity, "detailed")  # type: ignore

        return self._generate_explanation_by_level(decision, level)

    def _generate_explanation_by_level(
        self,
        decision: dict[str, Any],
        level: ExplanationLevel,
    ) -> str:
        """Generate explanation at specified detail level.

        Args:
            decision: Decision data
            level: Detail level

        Returns:
            Explanation string
        """
        if level == "basic":
            return self._generate_basic_explanation(decision)
        elif level == "detailed":
            return self._generate_detailed_explanation(decision)
        else:  # expert
            return self._generate_expert_explanation(decision)

    def _generate_basic_explanation(self, decision: dict[str, Any]) -> str:
        """Generate simple text explanation.

        Args:
            decision: Decision data

        Returns:
            Basic explanation string
        """
        action = decision.get("action", "unknown action")
        confidence = decision.get("confidence", 0.0)
        rationale = decision.get("rationale", {})

        # Find primary factor
        primary_reason = "analysis"
        if rationale:
            # Get highest-valued factor
            numeric_factors = {
                k: v for k, v in rationale.items()
                if isinstance(v, (int, float))
            }
            if numeric_factors:
                primary_factor = max(
                    numeric_factors.items(),
                    key=lambda x: x[1],
                )
                primary_reason = f"{primary_factor[0]} ({primary_factor[1]:.2f})"

        return (
            f"Chose '{action}' with confidence {confidence:.2f} "
            f"based on {primary_reason}."
        )

    def _generate_detailed_explanation(self, decision: dict[str, Any]) -> str:
        """Generate structured explanation with key factors.

        Args:
            decision: Decision data

        Returns:
            Detailed explanation string
        """
        parts = [self._generate_basic_explanation(decision)]

        rationale = decision.get("rationale", {})
        if rationale:
            parts.append("\nKey factors:")
            for key, value in rationale.items():
                if isinstance(value, (int, float)):
                    parts.append(f"  • {key}: {value:.2f}")
                else:
                    parts.append(f"  • {key}: {value}")

        metadata = decision.get("metadata", {})
        if metadata:
            parts.append("\nContext:")
            for key, value in list(metadata.items())[:5]:  # Limit to 5
                parts.append(f"  • {key}: {value}")

        return "\n".join(parts)

    def _generate_expert_explanation(self, decision: dict[str, Any]) -> str:
        """Generate complete explanation with all details.

        Args:
            decision: Decision data

        Returns:
            Expert-level explanation string
        """
        parts = [
            "═" * 50,
            "DECISION ANALYSIS REPORT",
            "═" * 50,
            self._generate_detailed_explanation(decision),
            "\nTransparency Configuration:",
            f"  • Trait value: {self.trait_value:.2f}",
            f"  • PII masking: {'Enabled' if self.config.enable_pii_masking else 'Disabled'}",
            f"  • Log retention: {self.config.log_retention_days} days",
            "═" * 50,
        ]

        return "\n".join(parts)

    def get_key_factors(
        self,
        decision: dict[str, Any],
    ) -> list[tuple[str, float, str]]:
        """Extract key factors from decision.

        Args:
            decision: Decision data

        Returns:
            List of (factor_name, weight, description) tuples
        """
        factors: list[tuple[str, float, str]] = []
        rationale = decision.get("rationale", {})

        for key, value in rationale.items():
            if isinstance(value, (int, float)):
                description = f"Contributed {value:.2f} to decision"
                factors.append((key, float(value), description))

        # Sort by weight descending
        factors.sort(key=lambda x: x[1], reverse=True)

        return factors

    def generate_confidence_report(
        self,
        confidence_components: dict[str, float],
    ) -> str:
        """Generate transparent confidence breakdown.

        Args:
            confidence_components: Component scores

        Returns:
            Formatted confidence report
        """
        overall = sum(confidence_components.values()) / len(confidence_components)

        parts = [
            f"Decision confidence: {overall:.1%}",
            "\nComponent breakdown:",
        ]

        for component, score in sorted(
            confidence_components.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            parts.append(f"  {component:20s} [{bar}] {score:.1%}")

        # Add interpretation for high transparency
        if self.trait_value >= 0.8:
            parts.append("\nConfidence interpretation:")
            if overall >= 0.8:
                parts.append("  High confidence. Decision well-supported.")
            elif overall >= 0.5:
                parts.append("  Moderate confidence. Monitoring recommended.")
            else:
                parts.append("  Low confidence. Human oversight advisable.")

        return "\n".join(parts)

    def explain_uncertainty(
        self,
        uncertainty_sources: list[str],
        mitigation_steps: list[str],
    ) -> str:
        """Generate transparent uncertainty communication.

        Args:
            uncertainty_sources: Sources of uncertainty
            mitigation_steps: Steps to mitigate

        Returns:
            Formatted uncertainty explanation
        """
        parts = ["Current decision involves these uncertainties:"]

        for i, source in enumerate(uncertainty_sources, 1):
            parts.append(f"  {i}. {source}")

        parts.append("\nMitigation approaches:")
        for i, step in enumerate(mitigation_steps, 1):
            parts.append(f"  {i}. {step}")

        parts.append(
            f"\nTransparency note: Full disclosure provided "
            f"(transparency: {self.trait_value:.0%}). "
            f"Honest communication enables informed collaboration."
        )

        return "\n".join(parts)

    def generate_audit_trail(
        self,
        lookback: int = 10,
    ) -> str:
        """Create auditable record of recent decisions.

        Args:
            lookback: Number of recent decisions

        Returns:
            Formatted audit trail
        """
        recent = self._decision_logs.get_recent(lookback)

        parts = [
            f"Decision Audit Trail (most recent {len(recent)} decisions):",
            "═" * 60,
        ]

        for i, entry in enumerate(reversed(recent), 1):
            parts.append(
                f"\n[{i}] {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            parts.append(f"ID: {entry.decision_id}")
            parts.append(f"Action: {entry.action}")
            parts.append(f"Confidence: {entry.confidence:.2f}")
            parts.append("-" * 60)

        return "\n".join(parts)

    def _mask_pii(self, text: str) -> str:
        """Mask PII in text.

        Args:
            text: Text to mask

        Returns:
            Text with PII masked
        """
        result = text
        for pattern in self.PII_PATTERNS:
            result = pattern.pattern.sub(pattern.replacement, result)
        return result

    def _mask_pii_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively mask PII in dictionary.

        Args:
            data: Dictionary to process

        Returns:
            Dictionary with PII masked
        """
        result: dict[str, Any] = {}

        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._mask_pii(value)
            elif isinstance(value, dict):
                result[key] = self._mask_pii_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self._mask_pii(v) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value

        return result

    def get_decision_log(
        self,
        decision_id: UUID,
        requester_id: str | None = None,
    ) -> DecisionLog | None:
        """Retrieve a specific decision log.

        Args:
            decision_id: Decision identifier
            requester_id: ID of requester (for audit)

        Returns:
            DecisionLog if found, None otherwise
        """
        # Record access
        self._access_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requester_id": requester_id,
                "decision_id": str(decision_id),
            }
        )

        # Find in logs
        for log in self._decision_logs:
            if log.decision_id == decision_id:
                return log

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics.

        Returns:
            Dictionary with module statistics
        """
        return {
            "trait_value": self.trait_value,
            "logs_stored": len(self._decision_logs),
            "access_count": len(self._access_log),
            "pii_masking_enabled": self.config.enable_pii_masking,
            "retention_days": self.config.log_retention_days,
        }
