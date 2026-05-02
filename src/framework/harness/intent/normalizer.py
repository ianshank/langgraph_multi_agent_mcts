"""Default intent normaliser.

Translates the loose ``str | dict`` payload that arrives from a CLI, an
``EvaluationHarness`` adapter, or a ``GraphBuilder`` invocation into a typed
:class:`Task`. The normaliser is intentionally permissive about input shape
but strict about output: the resulting :class:`Task` is the contract every
later phase consumes.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from src.framework.harness.settings import HarnessSettings
from src.framework.harness.state import AcceptanceCriterion, Task


class DefaultIntentNormalizer:
    """Convert raw input into a :class:`Task`.

    Recognised dict keys (all optional except ``goal`` or ``query``):

    * ``id`` — task id; auto-generated if absent
    * ``goal`` / ``query`` — primary objective string
    * ``acceptance_criteria`` — list of ``str`` descriptions or
      ``{"id": ..., "description": ..., "check": ...}`` mappings
    * ``constraints`` — list of strings
    * ``metadata`` — passthrough dict
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    async def normalize(self, raw: str | dict[str, Any], settings: HarnessSettings) -> Task:
        """Normalise ``raw`` into a :class:`Task`."""
        if isinstance(raw, str):
            return self._from_string(raw)
        if isinstance(raw, dict):
            return self._from_dict(raw)
        raise TypeError(f"DefaultIntentNormalizer cannot accept payload of type {type(raw).__name__}")

    def _from_string(self, raw: str) -> Task:
        if not raw.strip():
            raise ValueError("intent string is empty")
        task = Task(id=str(uuid4()), goal=raw.strip(), raw=raw)
        self._logger.debug("normalised string intent id=%s len=%d", task.id, len(raw))
        return task

    def _from_dict(self, raw: dict[str, Any]) -> Task:
        goal = raw.get("goal") or raw.get("query")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("intent dict requires non-empty 'goal' or 'query' string")
        task_id = str(raw.get("id") or uuid4())
        criteria = tuple(self._coerce_criteria(raw.get("acceptance_criteria") or ()))
        constraints = tuple(str(c) for c in raw.get("constraints") or ())
        metadata = dict(raw.get("metadata") or {})
        task = Task(
            id=task_id,
            goal=goal.strip(),
            acceptance_criteria=criteria,
            constraints=constraints,
            metadata=metadata,
            raw=str(raw),
        )
        self._logger.debug(
            "normalised dict intent id=%s criteria=%d constraints=%d",
            task.id,
            len(criteria),
            len(constraints),
        )
        return task

    @staticmethod
    def _coerce_criteria(items: Any) -> list[AcceptanceCriterion]:
        out: list[AcceptanceCriterion] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                out.append(AcceptanceCriterion(id=f"c{index}", description=item))
            elif isinstance(item, dict):
                out.append(
                    AcceptanceCriterion(
                        id=str(item.get("id") or f"c{index}"),
                        description=str(item.get("description") or ""),
                        check=str(item.get("check") or ""),
                    )
                )
            else:
                raise TypeError(f"acceptance_criteria[{index}] must be str or dict, got {type(item).__name__}")
        return out


__all__ = ["DefaultIntentNormalizer"]
