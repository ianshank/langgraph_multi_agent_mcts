"""``HarnessOutcome`` — discriminated union for control-loop verdicts.

The control loop branches on ``kind`` rather than on raw exceptions so that
retryable failures, hook violations, terminal errors, and budget exhaustion
all flow through the same typed channel. The outermost facade flattens these
into the existing ``AgentResult`` shape only when handing back to
``GraphBuilder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.framework.harness.state import Severity, VerificationResult


@dataclass(frozen=True)
class Continue:
    """Loop should proceed to the next iteration."""

    kind: Literal["continue"] = "continue"
    note: str = ""


@dataclass(frozen=True)
class Retryable:
    """Recoverable failure — loop may retry within budget."""

    kind: Literal["retryable"] = "retryable"
    reason: str = ""
    cause: str | None = None  # exception class name, never the instance
    severity: Severity = Severity.WARNING


@dataclass(frozen=True)
class HookViolation:
    """A guardrail hook rejected the iteration."""

    kind: Literal["hook_violation"] = "hook_violation"
    hook_name: str = ""
    detail: str = ""
    severity: Severity = Severity.WARNING


@dataclass(frozen=True)
class Terminal:
    """Verifier accepted (or unrecoverable rejection) — loop must stop."""

    kind: Literal["terminal"] = "terminal"
    accepted: bool = False
    verification: VerificationResult | None = None
    note: str = ""


@dataclass(frozen=True)
class BudgetExhausted:
    """Token, time, or iteration budget consumed."""

    kind: Literal["budget_exhausted"] = "budget_exhausted"
    budget: Literal["tokens", "time", "iterations"] = "iterations"
    consumed: float = 0.0
    limit: float = 0.0


HarnessOutcome = Continue | Retryable | HookViolation | Terminal | BudgetExhausted


@dataclass(frozen=True)
class PhaseResult:
    """Returned by each phase function: state delta + verdict."""

    outcome: HarnessOutcome = field(default_factory=Continue)
    tokens_used: int = 0
    duration_ms: float = 0.0
    notes: str = ""


def is_terminal(outcome: HarnessOutcome) -> bool:
    """Convenience: should the loop stop on this outcome?"""
    return isinstance(outcome, (Terminal, BudgetExhausted))


__all__ = [
    "BudgetExhausted",
    "Continue",
    "HarnessOutcome",
    "HookViolation",
    "PhaseResult",
    "Retryable",
    "Terminal",
    "is_terminal",
]
