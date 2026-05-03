"""``RequiredMetadataKeysHook`` — enforce required keys on the task metadata."""

from __future__ import annotations

from collections.abc import Iterable

from src.framework.harness.hooks.base import BaseHook
from src.framework.harness.outcomes import HookViolation
from src.framework.harness.protocols import HookCost
from src.framework.harness.state import HarnessState


class RequiredMetadataKeysHook(BaseHook):
    """Fail if any of the required metadata keys are missing from ``state.task``.

    Cheap (dict membership only). Useful for enforcing that callers tag tasks
    with provenance fields like ``owner`` or ``ticket`` before any LLM work.
    """

    def __init__(
        self,
        *,
        required: Iterable[str],
        name: str = "required_metadata_keys",
        short_circuit: bool = True,
    ) -> None:
        super().__init__(name=name, cost_class=HookCost.CHEAP, short_circuit=short_circuit)
        self._required = tuple(required)

    async def check(self, state: HarnessState) -> HookViolation | None:
        if state.task is None:
            return None
        missing = [k for k in self._required if k not in state.task.metadata]
        if missing:
            return HookViolation(
                hook_name=self.name,
                detail=f"required metadata keys missing: {', '.join(missing)}",
            )
        return None


__all__ = ["RequiredMetadataKeysHook"]
