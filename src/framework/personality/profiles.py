"""
Personality profile definitions with Pydantic validation.

Provides:
- Type-safe personality trait configuration
- Strict validation for trait bounds [0.0, 1.0]
- NaN/Inf rejection for security
- Immutable profile with update methods
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Type aliases
TraitValue = Annotated[float, "Must be in range [0.0, 1.0]"]
TraitName = Literal[
    "loyalty", "curiosity", "aspiration", "ethical_weight", "transparency"
]

# Constants
MIN_TRAIT_VALUE: float = 0.0
MAX_TRAIT_VALUE: float = 1.0
DEFAULT_TRAIT_VALUE: float = 0.5


class PersonalityProfile(BaseModel):
    """Validated personality profile with security controls.

    All traits are constrained to [0.0, 1.0] range.
    Prevents numerical exploits and ensures type safety.

    Attributes:
        loyalty: Goal persistence and team alignment [0.0=independent, 1.0=committed]
        curiosity: Exploration vs exploitation [0.0=focused, 1.0=exploratory]
        aspiration: Goal ambition level [0.0=conservative, 1.0=ambitious]
        ethical_weight: Priority for ethical considerations [0.0=pragmatic, 1.0=principled]
        transparency: Explainability preference [0.0=minimal, 1.0=comprehensive]

    Example:
        >>> profile = PersonalityProfile(
        ...     loyalty=0.95,
        ...     curiosity=0.85,
        ...     aspiration=0.9,
        ... )
        >>> profile.trait_vector
        array([0.95, 0.85, 0.9, 0.5, 0.5], dtype=float32)
    """

    model_config = ConfigDict(
        strict=True,
        validate_assignment=True,
        extra="forbid",
        frozen=True,  # Immutable after creation
    )

    # Core personality traits
    loyalty: float = Field(
        default=DEFAULT_TRAIT_VALUE,
        ge=MIN_TRAIT_VALUE,
        le=MAX_TRAIT_VALUE,
        description="Goal persistence and team alignment [0.0-1.0]",
    )

    curiosity: float = Field(
        default=DEFAULT_TRAIT_VALUE,
        ge=MIN_TRAIT_VALUE,
        le=MAX_TRAIT_VALUE,
        description="Exploration vs exploitation [0.0-1.0]",
    )

    aspiration: float = Field(
        default=DEFAULT_TRAIT_VALUE,
        ge=MIN_TRAIT_VALUE,
        le=MAX_TRAIT_VALUE,
        description="Goal ambition level [0.0-1.0]",
    )

    ethical_weight: float = Field(
        default=DEFAULT_TRAIT_VALUE,
        ge=MIN_TRAIT_VALUE,
        le=MAX_TRAIT_VALUE,
        description="Priority for ethical considerations [0.0-1.0]",
    )

    transparency: float = Field(
        default=DEFAULT_TRAIT_VALUE,
        ge=MIN_TRAIT_VALUE,
        le=MAX_TRAIT_VALUE,
        description="Explainability preference [0.0-1.0]",
    )

    # Metadata
    profile_version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Profile schema version (semver)",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Profile creation timestamp (UTC)",
    )

    @field_validator(
        "loyalty",
        "curiosity",
        "aspiration",
        "ethical_weight",
        "transparency",
    )
    @classmethod
    def validate_trait_safety(cls, v: float) -> float:
        """Validate trait is safe (not NaN/Inf) - security check.

        Args:
            v: Trait value to validate

        Returns:
            Validated trait value

        Raises:
            ValueError: If value is NaN or Inf
        """
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"Trait value must be a valid number, got: {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_trait_consistency(self) -> PersonalityProfile:
        """Validate logical consistency between traits.

        Warns for potentially inconsistent configurations that
        could lead to unpredictable behavior.
        """
        # High ethical weight + very high risk tolerance warning
        if self.ethical_weight > 0.8 and self.aspiration > 0.95:
            warnings.warn(
                "High ethical weight combined with very high aspiration "
                "may create goal conflicts. Consider balancing these traits.",
                UserWarning,
                stacklevel=2,
            )

        # Low transparency warning
        if self.transparency < 0.3:
            warnings.warn(
                "Low transparency setting may limit explainability and "
                "make debugging difficult.",
                UserWarning,
                stacklevel=2,
            )

        # Extreme curiosity without aspiration warning
        if self.curiosity > 0.9 and self.aspiration < 0.3:
            warnings.warn(
                "High curiosity with low aspiration may result in "
                "exploration without goal progress.",
                UserWarning,
                stacklevel=2,
            )

        return self

    @property
    def trait_vector(self) -> NDArray[np.float32]:
        """Get numpy vector of trait values.

        Returns:
            5-element numpy array in order:
            [loyalty, curiosity, aspiration, ethical_weight, transparency]
        """
        return np.array(
            [
                self.loyalty,
                self.curiosity,
                self.aspiration,
                self.ethical_weight,
                self.transparency,
            ],
            dtype=np.float32,
        )

    @property
    def trait_sum(self) -> float:
        """Get sum of all trait values."""
        return sum(
            [
                self.loyalty,
                self.curiosity,
                self.aspiration,
                self.ethical_weight,
                self.transparency,
            ]
        )

    @property
    def trait_mean(self) -> float:
        """Get mean of all trait values."""
        return self.trait_sum / 5.0

    def get_trait(self, name: TraitName) -> float:
        """Type-safe trait getter.

        Args:
            name: Trait name to retrieve

        Returns:
            Trait value
        """
        return getattr(self, name)

    def with_trait(self, name: TraitName, value: float) -> PersonalityProfile:
        """Create new profile with updated trait (immutable update).

        Args:
            name: Trait to update
            value: New trait value [0.0, 1.0]

        Returns:
            New PersonalityProfile with updated trait
        """
        return self.model_copy(update={name: value})

    def with_traits(self, **traits: float) -> PersonalityProfile:
        """Create new profile with multiple updated traits.

        Args:
            **traits: Trait name-value pairs to update

        Returns:
            New PersonalityProfile with updated traits
        """
        return self.model_copy(update=traits)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary for serialization.

        Returns:
            Dictionary with all profile fields
        """
        return {
            "loyalty": self.loyalty,
            "curiosity": self.curiosity,
            "aspiration": self.aspiration,
            "ethical_weight": self.ethical_weight,
            "transparency": self.transparency,
            "profile_version": self.profile_version,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityProfile:
        """Create profile from dictionary with validation.

        Args:
            data: Dictionary with profile fields

        Returns:
            Validated PersonalityProfile
        """
        # Handle ISO datetime string
        if "created_at" in data and isinstance(data["created_at"], str):
            data = data.copy()
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    @classmethod
    def default(cls) -> PersonalityProfile:
        """Create profile with all default values.

        Returns:
            PersonalityProfile with default traits (0.5 each)
        """
        return cls()

    @classmethod
    def high_performer(cls) -> PersonalityProfile:
        """Create high-performance profile preset.

        Returns:
            PersonalityProfile optimized for goal achievement
        """
        return cls(
            loyalty=0.85,
            curiosity=0.7,
            aspiration=0.9,
            ethical_weight=0.8,
            transparency=0.75,
        )

    @classmethod
    def explorer(cls) -> PersonalityProfile:
        """Create exploration-focused profile preset.

        Returns:
            PersonalityProfile optimized for exploration
        """
        return cls(
            loyalty=0.6,
            curiosity=0.95,
            aspiration=0.7,
            ethical_weight=0.7,
            transparency=0.8,
        )

    @classmethod
    def principled(cls) -> PersonalityProfile:
        """Create ethics-focused profile preset.

        Returns:
            PersonalityProfile with high ethical weight
        """
        return cls(
            loyalty=0.8,
            curiosity=0.65,
            aspiration=0.75,
            ethical_weight=0.95,
            transparency=0.9,
        )


# Legacy dataclass support (for backward compatibility)
@dataclass(frozen=True, slots=True)
class PersonalityTraits:
    """Legacy dataclass for personality traits.

    Note: Prefer using PersonalityProfile (Pydantic model) for
    new code. This class is provided for backward compatibility.

    Attributes:
        loyalty: Goal persistence [0.0, 1.0]
        curiosity: Exploration tendency [0.0, 1.0]
        aspiration: Goal ambition [0.0, 1.0]
        ethical_weight: Ethics priority [0.0, 1.0]
        transparency: Explainability [0.0, 1.0]
    """

    # Class-level constants
    MIN_VALUE: ClassVar[float] = 0.0
    MAX_VALUE: ClassVar[float] = 1.0

    loyalty: float = DEFAULT_TRAIT_VALUE
    curiosity: float = DEFAULT_TRAIT_VALUE
    aspiration: float = DEFAULT_TRAIT_VALUE
    ethical_weight: float = DEFAULT_TRAIT_VALUE
    transparency: float = DEFAULT_TRAIT_VALUE

    # Computed field
    _trait_vector: NDArray[np.float32] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate all traits after initialization."""
        for trait_name in [
            "loyalty",
            "curiosity",
            "aspiration",
            "ethical_weight",
            "transparency",
        ]:
            value = getattr(self, trait_name)
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{trait_name} must be numeric, got {type(value)}"
                )
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"{trait_name} cannot be NaN or Inf")
            if not self.MIN_VALUE <= value <= self.MAX_VALUE:
                raise ValueError(
                    f"{trait_name}={value} out of bounds "
                    f"[{self.MIN_VALUE}, {self.MAX_VALUE}]"
                )

        # Initialize computed field
        object.__setattr__(
            self,
            "_trait_vector",
            np.array(
                [
                    self.loyalty,
                    self.curiosity,
                    self.aspiration,
                    self.ethical_weight,
                    self.transparency,
                ],
                dtype=np.float32,
            ),
        )

    @property
    def trait_vector(self) -> NDArray[np.float32]:
        """Get numpy vector of traits."""
        return self._trait_vector

    def to_pydantic(self) -> PersonalityProfile:
        """Convert to Pydantic PersonalityProfile.

        Returns:
            PersonalityProfile with same trait values
        """
        return PersonalityProfile(
            loyalty=self.loyalty,
            curiosity=self.curiosity,
            aspiration=self.aspiration,
            ethical_weight=self.ethical_weight,
            transparency=self.transparency,
        )


def validate_trait_value(value: float, trait_name: str = "trait") -> float:
    """Validate a single trait value.

    Args:
        value: Value to validate
        trait_name: Name for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is invalid
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{trait_name} must be numeric, got {type(value)}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{trait_name} cannot be NaN or Inf")
    if not MIN_TRAIT_VALUE <= value <= MAX_TRAIT_VALUE:
        raise ValueError(
            f"{trait_name}={value} out of bounds "
            f"[{MIN_TRAIT_VALUE}, {MAX_TRAIT_VALUE}]"
        )
    return float(value)
