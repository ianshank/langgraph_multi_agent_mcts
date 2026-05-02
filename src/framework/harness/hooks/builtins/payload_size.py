"""``PayloadSizeHook`` — bounds the size of the last-rendered context payload."""

from __future__ import annotations

from src.framework.harness.hooks.base import BaseHook
from src.framework.harness.outcomes import HookViolation
from src.framework.harness.protocols import HookCost
from src.framework.harness.state import HarnessState


class PayloadSizeHook(BaseHook):
    """Reject iterations whose composed context exceeds a character limit.

    Cheap (string measurement only); short-circuits by default because the
    LLM call would burn tokens on a payload we already know is too big.
    """

    def __init__(
        self,
        *,
        max_chars: int,
        name: str = "payload_size",
        short_circuit: bool = True,
    ) -> None:
        super().__init__(name=name, cost_class=HookCost.CHEAP, short_circuit=short_circuit)
        self._max_chars = max_chars

    async def check(self, state: HarnessState) -> HookViolation | None:
        if state.last_context is None:
            return None
        rendered = state.last_context.render()
        size = sum(len(m["content"]) for m in rendered)
        if size > self._max_chars:
            return HookViolation(
                hook_name=self.name,
                detail=f"context size {size} exceeds limit {self._max_chars}",
            )
        return None


__all__ = ["PayloadSizeHook"]
