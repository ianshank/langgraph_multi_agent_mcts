"""
Custom exceptions for personality-driven agents.

Provides:
- Hierarchical exception structure
- Detailed error context
- Type-safe error handling
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorSeverity(str, Enum):
    """Severity levels for personality-related errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PersonalityError(Exception):
    """Base exception for personality module errors.

    Attributes:
        message: Human-readable error description
        severity: Error severity level
        context: Additional context for debugging
        original_error: Original exception if wrapping
    """

    message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: dict[str, Any] | None = None
    original_error: Exception | None = None

    def __str__(self) -> str:
        msg = f"[{self.severity.value.upper()}] {self.message}"
        if self.context:
            msg += f" | Context: {self.context}"
        if self.original_error:
            msg += f" | Caused by: {self.original_error}"
        return msg

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"severity={self.severity!r})"
        )


@dataclass
class TraitValidationError(PersonalityError):
    """Error raised when trait validation fails.

    Attributes:
        trait_name: Name of the invalid trait
        trait_value: The invalid value provided
        expected_range: Description of valid range
    """

    trait_name: str = ""
    trait_value: float | None = None
    expected_range: str = "[0.0, 1.0]"

    def __post_init__(self) -> None:
        if not self.message and self.trait_name:
            self.message = (
                f"Trait '{self.trait_name}' validation failed: "
                f"got {self.trait_value}, expected {self.expected_range}"
            )


@dataclass
class EthicalViolationError(PersonalityError):
    """Error raised when an action violates ethical constraints.

    Attributes:
        action: The violating action
        framework: Ethical framework that was violated
        violation_type: Type of ethical violation
        score: Ethical score that triggered the violation
    """

    action: str = ""
    framework: str = ""
    violation_type: str = ""
    score: float = 0.0

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"Ethical violation in '{self.framework}' framework: "
                f"action '{self.action}' scored {self.score:.2f} "
                f"({self.violation_type})"
            )
        self.severity = ErrorSeverity.HIGH


@dataclass
class TransparencyError(PersonalityError):
    """Error raised when transparency/explainability fails.

    Attributes:
        decision_id: ID of the decision being explained
        explanation_type: Type of explanation requested
    """

    decision_id: str = ""
    explanation_type: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"Failed to generate '{self.explanation_type}' explanation "
                f"for decision '{self.decision_id}'"
            )


@dataclass
class MemoryLimitError(PersonalityError):
    """Error raised when memory limits are exceeded.

    Attributes:
        collection_name: Name of the collection
        current_size: Current collection size
        max_size: Maximum allowed size
    """

    collection_name: str = ""
    current_size: int = 0
    max_size: int = 0

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"Memory limit exceeded for '{self.collection_name}': "
                f"{self.current_size} >= {self.max_size}"
            )
        self.severity = ErrorSeverity.HIGH


@dataclass
class ModuleInitializationError(PersonalityError):
    """Error raised when module initialization fails.

    Attributes:
        module_name: Name of the module
        required_dependencies: Missing dependencies
    """

    module_name: str = ""
    required_dependencies: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.message:
            deps = ", ".join(self.required_dependencies or [])
            self.message = (
                f"Failed to initialize '{self.module_name}' module. "
                f"Missing dependencies: {deps}"
            )
        self.severity = ErrorSeverity.CRITICAL
