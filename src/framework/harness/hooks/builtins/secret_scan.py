"""``SecretScanHook`` — refuse outputs that look like leaked secrets."""

from __future__ import annotations

import re

from src.framework.harness.hooks.base import BaseHook
from src.framework.harness.outcomes import HookViolation
from src.framework.harness.protocols import HookCost
from src.framework.harness.state import HarnessState

# Conservative defaults — narrow patterns to keep false positives low.
_DEFAULT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret_key", re.compile(r"(?i)aws(.{0,20})?(secret|access).{0,20}?[:=]\s*[A-Za-z0-9/+=]{40}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


class SecretScanHook(BaseHook):
    """Scan recent observation payloads and the response text for leaked secrets.

    Medium-cost (regex over potentially-long text); short-circuits because
    leaking a secret is unrecoverable.
    """

    def __init__(
        self,
        *,
        name: str = "secret_scan",
        short_circuit: bool = True,
        patterns: tuple[tuple[str, re.Pattern[str]], ...] | None = None,
    ) -> None:
        super().__init__(name=name, cost_class=HookCost.MEDIUM, short_circuit=short_circuit)
        self._patterns = patterns or _DEFAULT_PATTERNS

    async def check(self, state: HarnessState) -> HookViolation | None:
        haystacks = [state.last_response_text]
        haystacks.extend(o.payload for o in state.last_observations)
        for haystack in haystacks:
            if not haystack:
                continue
            for label, pattern in self._patterns:
                if pattern.search(haystack):
                    return HookViolation(
                        hook_name=self.name,
                        detail=f"possible secret detected: {label}",
                    )
        return None


__all__ = ["SecretScanHook"]
