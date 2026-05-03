"""Agent harness framework.

Public surface intentionally narrow — most callers should go through
:class:`HarnessFactory` rather than instantiating internals directly.
"""

from src.framework.harness.loop.facade import HarnessAgentAdapter
from src.framework.harness.loop.runner import HarnessRunner, RunResult
from src.framework.harness.outcomes import (
    BudgetExhausted,
    Continue,
    HarnessOutcome,
    HookViolation,
    Retryable,
    Terminal,
    is_terminal,
)
from src.framework.harness.settings import (
    AggregationPolicy,
    HarnessPermissions,
    HarnessSettings,
    TopologyName,
    get_harness_settings,
    reset_harness_settings,
)

__all__ = [
    "AggregationPolicy",
    "BudgetExhausted",
    "Continue",
    "HarnessAgentAdapter",
    "HarnessOutcome",
    "HarnessPermissions",
    "HarnessRunner",
    "HarnessSettings",
    "HookViolation",
    "Retryable",
    "RunResult",
    "Terminal",
    "TopologyName",
    "get_harness_settings",
    "is_terminal",
    "reset_harness_settings",
]
