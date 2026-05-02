"""Default :class:`OutputVerifier` implementations.

The verifier decides whether the iteration's outputs satisfy the task's
acceptance criteria. Two implementations:

* :class:`AcceptanceCriteriaVerifier` — searches observation payloads for
  configured success markers; accepts when all criteria match.
* :class:`AlwaysAccept` / :class:`AlwaysReject` — degenerate verifiers used
  in tests to drive the loop into specific terminal states.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from src.framework.harness.state import (
    AcceptanceCriterion,
    Observation,
    Plan,
    Task,
    VerificationResult,
)


@dataclass
class AcceptanceCriteriaVerifier:
    """Pattern-based verifier: each criterion matches if its ``check`` regex
    appears in any observation's payload (or, if ``check`` is empty, the
    description itself is used as a literal needle).
    """

    logger: logging.Logger = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    async def verify(
        self,
        observations: Sequence[Observation],
        task: Task,
        plan: Plan | None,
    ) -> VerificationResult:
        if not task.acceptance_criteria:
            # No criteria → accept iff at least one observation succeeded.
            passed = bool(observations) and all(o.success for o in observations)
            return VerificationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                notes="no acceptance criteria defined; accepted on observation success",
            )

        haystack = "\n".join(o.payload for o in observations if o.success)
        failed: list[str] = []
        passes = 0
        for crit in task.acceptance_criteria:
            if self._matches(crit, haystack):
                passes += 1
            else:
                failed.append(crit.id)
        total = len(task.acceptance_criteria)
        score = passes / total if total else 0.0
        passed = passes == total
        self.logger.debug(
            "verifier task=%s passed=%s score=%.2f failed=%d/%d",
            task.id,
            passed,
            score,
            len(failed),
            total,
        )
        return VerificationResult(
            passed=passed,
            score=score,
            failed_criteria=tuple(failed),
            notes=f"{passes}/{total} criteria satisfied",
        )

    @staticmethod
    def _matches(crit: AcceptanceCriterion, haystack: str) -> bool:
        needle = crit.check or crit.description
        if not needle:
            return False
        try:
            return re.search(needle, haystack, re.IGNORECASE) is not None
        except re.error:
            return needle.lower() in haystack.lower()


@dataclass
class AlwaysAccept:
    """Test-only verifier that accepts unconditionally."""

    async def verify(self, observations: Sequence[Observation], task: Task, plan: Plan | None) -> VerificationResult:
        return VerificationResult(passed=True, score=1.0, notes="always-accept")


@dataclass
class AlwaysReject:
    """Test-only verifier that rejects unconditionally."""

    async def verify(self, observations: Sequence[Observation], task: Task, plan: Plan | None) -> VerificationResult:
        return VerificationResult(passed=False, score=0.0, notes="always-reject")


__all__ = ["AcceptanceCriteriaVerifier", "AlwaysAccept", "AlwaysReject"]
