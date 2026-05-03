"""State, task, plan, observation, and context types for the harness.

These are pure data structures with no behavior. The control loop reads and
writes them; everything else (hooks, tools, topologies) treats them as
immutable inputs and produces new instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from src.utils.time_utils import utc_now


class Severity(str, Enum):
    """Outcome severity hint for downstream loggers / callers."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class AcceptanceCriterion:
    """A single, mechanically checkable acceptance criterion."""

    id: str
    description: str
    check: str = ""  # human-readable; verifier owns the actual mechanism


@dataclass(frozen=True)
class Task:
    """Normalised intent — the harness never operates on raw user prose.

    ``raw`` is preserved for audit logs but never consumed by downstream
    phases; ``goal`` and ``acceptance_criteria`` are the operational contract.
    """

    id: str
    goal: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PlanStep:
    """One ordered step in a plan."""

    id: str
    description: str
    expected_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """Strategic plan produced by the planner before any worker executes."""

    task_id: str
    summary: str
    steps: tuple[PlanStep, ...] = ()
    rationale: str = ""
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ContextPayload:
    """Concrete prompt-context payload for the Reason phase.

    Keeping this typed (rather than a free-form ``dict``) makes context
    construction inspectable and unit-testable.
    """

    system_prompt: str
    task_brief: str
    plan_brief: str = ""
    memory_excerpt: str = ""
    spec_excerpt: str = ""
    rag_snippets: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self) -> list[dict[str, str]]:
        """Convert to the OpenAI/Anthropic chat ``messages`` shape."""
        system = "\n\n".join(
            section
            for section in (
                self.system_prompt,
                ("# Task\n" + self.task_brief) if self.task_brief else "",
                ("# Plan\n" + self.plan_brief) if self.plan_brief else "",
                ("# Memory\n" + self.memory_excerpt) if self.memory_excerpt else "",
                ("# Spec\n" + self.spec_excerpt) if self.spec_excerpt else "",
                ("# Retrieved\n" + "\n---\n".join(self.rag_snippets)) if self.rag_snippets else "",
            )
            if section
        )
        return [{"role": "system", "content": system}]


@dataclass(frozen=True)
class ToolInvocation:
    """A pending tool call dispatched by the model."""

    id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Result of a tool invocation, post-truncation."""

    invocation_id: str
    tool_name: str
    success: bool
    payload: str
    spillover_path: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of an :class:`OutputVerifier` check."""

    passed: bool
    score: float = 0.0
    failed_criteria: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class HarnessState:
    """Mutable per-run state threaded through the control loop.

    The runner enforces single-writer semantics: only the runner's own
    coroutine mutates this object, while phase functions read it and return
    deltas.
    """

    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    iteration: int = 0
    task: Task | None = None
    plan: Plan | None = None
    last_context: ContextPayload | None = None
    last_response_text: str = ""
    pending_tool_calls: tuple[ToolInvocation, ...] = ()
    last_observations: tuple[Observation, ...] = ()
    last_verification: VerificationResult | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    tokens_consumed: int = 0
    started_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record(self, phase: str, payload: dict[str, Any]) -> None:
        """Append a structured history entry; cheap, used by every phase."""
        self.history.append(
            {
                "iteration": self.iteration,
                "phase": phase,
                "at": utc_now().isoformat(),
                **payload,
            }
        )


__all__ = [
    "AcceptanceCriterion",
    "ContextPayload",
    "HarnessState",
    "Observation",
    "Plan",
    "PlanStep",
    "Severity",
    "Task",
    "ToolInvocation",
    "VerificationResult",
]
