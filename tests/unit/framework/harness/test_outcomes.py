"""Unit tests for the ``HarnessOutcome`` discriminated union."""

from __future__ import annotations

import pytest

from src.framework.harness.outcomes import (
    BudgetExhausted,
    Continue,
    HookViolation,
    PhaseResult,
    Retryable,
    Terminal,
    is_terminal,
)
from src.framework.harness.state import Severity, VerificationResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (Continue(), False),
        (Retryable(reason="timeout"), False),
        (HookViolation(hook_name="ruff", detail="bad"), False),
        (Terminal(accepted=True), True),
        (BudgetExhausted(budget="iterations", limit=5, consumed=5), True),
    ],
)
def test_is_terminal_matches_kind(outcome: object, expected: bool) -> None:
    """``is_terminal`` returns ``True`` only for ``Terminal`` and ``BudgetExhausted``."""
    assert is_terminal(outcome) is expected  # type: ignore[arg-type]


def test_outcomes_have_distinct_kind_tags() -> None:
    """Each outcome variant has a unique ``kind`` literal so callers can pattern-match."""
    kinds = {
        Continue().kind,
        Retryable().kind,
        HookViolation().kind,
        Terminal().kind,
        BudgetExhausted().kind,
    }
    assert len(kinds) == 5


def test_phase_result_defaults_to_continue() -> None:
    """A bare ``PhaseResult()`` is a non-terminal continuation."""
    pr = PhaseResult()
    assert isinstance(pr.outcome, Continue)
    assert not is_terminal(pr.outcome)


def test_terminal_carries_verification() -> None:
    """``Terminal`` can wrap a ``VerificationResult`` for downstream consumers."""
    v = VerificationResult(passed=True, score=1.0)
    t = Terminal(accepted=True, verification=v)
    assert t.verification is v


def test_retryable_severity_default() -> None:
    """Retryable outcomes default to ``WARNING`` severity."""
    assert Retryable().severity is Severity.WARNING
