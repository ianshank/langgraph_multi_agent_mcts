"""
Personality-driven agent framework for Multi-Agent MCTS.

This module provides personality traits that influence agent behavior:
- Loyalty: Goal persistence and consistency
- Curiosity: Intrinsic motivation and exploration
- Aspiration: Goal management and achievement standards
- Ethical reasoning: Multi-framework ethical evaluation
- Transparency: Decision explainability and logging

Example:
    >>> from src.framework.personality import PersonalityProfile, PersonalityDrivenAgent
    >>> profile = PersonalityProfile(
    ...     loyalty=0.95,
    ...     curiosity=0.85,
    ...     aspiration=0.9,
    ...     ethical_weight=0.92,
    ...     transparency=0.88,
    ... )
    >>> agent = PersonalityDrivenAgent(
    ...     base_agent=hrm_agent,
    ...     personality=profile,
    ... )
"""

from __future__ import annotations

# Public API exports
from .profiles import PersonalityProfile, TraitValue
from .config import PersonalityConfig
from .protocols import (
    PersonalityModuleProtocol,
    MCTSInfluencer,
    PromptAugmenter,
    ConfidenceCalibrator,
    ExplainabilityProvider,
)
from .exceptions import (
    PersonalityError,
    TraitValidationError,
    EthicalViolationError,
    TransparencyError,
)
from .collections import BoundedHistory, BoundedCounter
from .agent import PersonalityDrivenAgent
from .factory import PersonalityFactory, PersonalityAgentBuilder

# Modules
from .modules import (
    LoyaltyModule,
    CuriosityModule,
    AspirationModule,
    EthicalReasoningModule,
    TransparencyModule,
)

__all__ = [
    # Core
    "PersonalityProfile",
    "PersonalityDrivenAgent",
    "PersonalityConfig",
    "PersonalityFactory",
    "PersonalityAgentBuilder",
    # Types
    "TraitValue",
    # Protocols
    "PersonalityModuleProtocol",
    "MCTSInfluencer",
    "PromptAugmenter",
    "ConfidenceCalibrator",
    "ExplainabilityProvider",
    # Modules
    "LoyaltyModule",
    "CuriosityModule",
    "AspirationModule",
    "EthicalReasoningModule",
    "TransparencyModule",
    # Collections
    "BoundedHistory",
    "BoundedCounter",
    # Exceptions
    "PersonalityError",
    "TraitValidationError",
    "EthicalViolationError",
    "TransparencyError",
]

__version__ = "0.1.0"
