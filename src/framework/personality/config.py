"""
Configuration classes for personality-driven agents.

Provides:
- PersonalityConfig for system-wide settings
- Module-specific configuration classes
- Backward-compatible optional personality support
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .profiles import PersonalityProfile


@dataclass
class LoyaltyConfig:
    """Configuration for LoyaltyModule.

    Attributes:
        max_goal_history: Maximum goal history entries
        max_action_memory: Maximum action consistency entries
        consistency_window: Window for consistency calculations
        persistence_multiplier: Base persistence factor
    """

    max_goal_history: int = 1000
    max_action_memory: int = 5000
    consistency_window: int = 100
    persistence_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_goal_history <= 0:
            raise ValueError("max_goal_history must be positive")
        if self.max_action_memory <= 0:
            raise ValueError("max_action_memory must be positive")
        if self.consistency_window <= 0:
            raise ValueError("consistency_window must be positive")


@dataclass
class CuriosityConfig:
    """Configuration for CuriosityModule.

    Attributes:
        max_state_memory: Maximum states to track
        intrinsic_reward_scale: Scale for intrinsic rewards
        novelty_threshold: Threshold for considering state novel
        exploration_bonus_decay: Decay rate for exploration bonus
    """

    max_state_memory: int = 10000
    intrinsic_reward_scale: float = 0.1
    novelty_threshold: float = 0.3
    exploration_bonus_decay: float = 0.99

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_state_memory <= 0:
            raise ValueError("max_state_memory must be positive")
        if not 0.0 <= self.intrinsic_reward_scale <= 10.0:
            raise ValueError(
                "intrinsic_reward_scale must be in [0.0, 10.0]"
            )
        if not 0.0 <= self.novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in [0.0, 1.0]")
        if not 0.5 <= self.exploration_bonus_decay <= 1.0:
            raise ValueError("exploration_bonus_decay must be in [0.5, 1.0]")


@dataclass
class AspirationConfig:
    """Configuration for AspirationModule.

    Attributes:
        max_active_goals: Maximum concurrent goals
        performance_history_size: Performance tracking window
        standard_raise_threshold: Threshold for raising standards
    """

    max_active_goals: int = 100
    performance_history_size: int = 1000
    standard_raise_threshold: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_active_goals <= 0:
            raise ValueError("max_active_goals must be positive")
        if not 0.0 <= self.standard_raise_threshold <= 1.0:
            raise ValueError(
                "standard_raise_threshold must be in [0.0, 1.0]"
            )


@dataclass
class EthicalConfig:
    """Configuration for EthicalReasoningModule.

    Attributes:
        max_violation_history: Maximum violations to track
        max_dilemma_resolutions: Maximum dilemmas to store
        violation_severity_levels: Number of severity levels
        strict_mode: If True, always reject violations
    """

    max_violation_history: int = 1000
    max_dilemma_resolutions: int = 500
    violation_severity_levels: int = 5
    strict_mode: bool = False

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if not 3 <= self.violation_severity_levels <= 10:
            raise ValueError(
                "violation_severity_levels must be in [3, 10]"
            )


@dataclass
class TransparencyConfig:
    """Configuration for TransparencyModule.

    Attributes:
        max_decision_log_size: Maximum decision log entries
        max_template_count: Maximum explanation templates
        enable_pii_masking: Enable PII detection and masking
        log_retention_days: Days to retain decision logs
    """

    max_decision_log_size: int = 5000
    max_template_count: int = 50
    enable_pii_masking: bool = True
    log_retention_days: int = 30

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_decision_log_size <= 0:
            raise ValueError("max_decision_log_size must be positive")
        if not 1 <= self.log_retention_days <= 365:
            raise ValueError("log_retention_days must be in [1, 365]")


@dataclass
class PersonalityConfig:
    """Master configuration for personality-driven agents.

    Provides system-wide personality settings with backward
    compatibility through optional enablement.

    Attributes:
        enabled: Whether personality system is active
        default_profile: Default personality profile for agents
        hrm_profile: Optional profile for HRM agent
        trm_profile: Optional profile for TRM agent
        mcts_curiosity_weight: How much curiosity affects MCTS
        meta_controller_loyalty_weight: Loyalty influence on routing
        module_configs: Per-module configuration
    """

    # Global settings
    enabled: bool = False

    # Default profiles
    default_profile: PersonalityProfile = field(
        default_factory=PersonalityProfile.default
    )

    # Agent-specific profiles (optional overrides)
    hrm_profile: PersonalityProfile | None = None
    trm_profile: PersonalityProfile | None = None
    mcts_profile: PersonalityProfile | None = None

    # Influence weights
    mcts_curiosity_weight: float = 0.3
    meta_controller_loyalty_weight: float = 0.2
    ethical_veto_threshold: float = 0.3  # Below this, action blocked

    # Module configs
    loyalty_config: LoyaltyConfig = field(default_factory=LoyaltyConfig)
    curiosity_config: CuriosityConfig = field(default_factory=CuriosityConfig)
    aspiration_config: AspirationConfig = field(
        default_factory=AspirationConfig
    )
    ethical_config: EthicalConfig = field(default_factory=EthicalConfig)
    transparency_config: TransparencyConfig = field(
        default_factory=TransparencyConfig
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.mcts_curiosity_weight <= 1.0:
            raise ValueError("mcts_curiosity_weight must be in [0.0, 1.0]")
        if not 0.0 <= self.meta_controller_loyalty_weight <= 1.0:
            raise ValueError(
                "meta_controller_loyalty_weight must be in [0.0, 1.0]"
            )
        if not 0.0 <= self.ethical_veto_threshold <= 1.0:
            raise ValueError("ethical_veto_threshold must be in [0.0, 1.0]")

    def get_profile_for_agent(self, agent_name: str) -> PersonalityProfile:
        """Get personality profile for a specific agent.

        Args:
            agent_name: Agent identifier (e.g., 'hrm', 'trm', 'mcts')

        Returns:
            Agent-specific profile or default profile
        """
        agent_lower = agent_name.lower()

        if "hrm" in agent_lower and self.hrm_profile:
            return self.hrm_profile
        if "trm" in agent_lower and self.trm_profile:
            return self.trm_profile
        if "mcts" in agent_lower and self.mcts_profile:
            return self.mcts_profile

        return self.default_profile

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of config
        """
        return {
            "enabled": self.enabled,
            "default_profile": self.default_profile.to_dict(),
            "hrm_profile": (
                self.hrm_profile.to_dict() if self.hrm_profile else None
            ),
            "trm_profile": (
                self.trm_profile.to_dict() if self.trm_profile else None
            ),
            "mcts_profile": (
                self.mcts_profile.to_dict() if self.mcts_profile else None
            ),
            "mcts_curiosity_weight": self.mcts_curiosity_weight,
            "meta_controller_loyalty_weight": (
                self.meta_controller_loyalty_weight
            ),
            "ethical_veto_threshold": self.ethical_veto_threshold,
        }

    @classmethod
    def disabled(cls) -> PersonalityConfig:
        """Create disabled personality config.

        Returns:
            PersonalityConfig with enabled=False
        """
        return cls(enabled=False)

    @classmethod
    def with_defaults(cls) -> PersonalityConfig:
        """Create enabled config with default profiles.

        Returns:
            PersonalityConfig with enabled=True and default profiles
        """
        return cls(
            enabled=True,
            default_profile=PersonalityProfile.default(),
        )

    @classmethod
    def high_performance(cls) -> PersonalityConfig:
        """Create config optimized for performance.

        Returns:
            PersonalityConfig with high-performer profiles
        """
        return cls(
            enabled=True,
            default_profile=PersonalityProfile.high_performer(),
            hrm_profile=PersonalityProfile(
                loyalty=0.85,
                curiosity=0.6,
                aspiration=0.9,
                ethical_weight=0.8,
                transparency=0.7,
            ),
            trm_profile=PersonalityProfile(
                loyalty=0.8,
                curiosity=0.5,
                aspiration=0.85,
                ethical_weight=0.75,
                transparency=0.65,
            ),
        )

    @classmethod
    def exploration_focused(cls) -> PersonalityConfig:
        """Create config optimized for exploration.

        Returns:
            PersonalityConfig with explorer profiles
        """
        return cls(
            enabled=True,
            default_profile=PersonalityProfile.explorer(),
            mcts_curiosity_weight=0.5,  # Higher curiosity influence
        )

    @classmethod
    def ethics_focused(cls) -> PersonalityConfig:
        """Create config with strong ethical constraints.

        Returns:
            PersonalityConfig with principled profiles
        """
        return cls(
            enabled=True,
            default_profile=PersonalityProfile.principled(),
            ethical_veto_threshold=0.5,  # Higher threshold
            ethical_config=EthicalConfig(strict_mode=True),
        )
