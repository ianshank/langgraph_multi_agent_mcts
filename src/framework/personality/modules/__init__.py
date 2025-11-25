"""
Personality trait modules.

Each module implements a specific personality trait:
- LoyaltyModule: Goal persistence and consistency
- CuriosityModule: Intrinsic motivation and exploration
- AspirationModule: Goal management and standards
- EthicalReasoningModule: Multi-framework ethical evaluation
- TransparencyModule: Decision explainability
"""

from __future__ import annotations

from .loyalty import LoyaltyModule
from .curiosity import CuriosityModule
from .aspiration import AspirationModule
from .ethical import EthicalReasoningModule
from .transparency import TransparencyModule

__all__ = [
    "LoyaltyModule",
    "CuriosityModule",
    "AspirationModule",
    "EthicalReasoningModule",
    "TransparencyModule",
]
