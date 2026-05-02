"""``AsyncAgentBase`` facade over a :class:`HarnessRunner`.

Allows the harness to drop into existing call sites — :class:`GraphBuilder`,
:class:`EvaluationHarness`, and any other consumer that expects
``AsyncAgentBase.process()`` — without modifying those call sites. The
facade flattens the runner's typed :class:`RunResult` into the legacy
``AgentResult`` shape, populating the keys ``GraphBuilder`` is known to
read (``metadata.confidence``, ``metadata.decomposition_quality_score``,
``intermediate_steps``).
"""

from __future__ import annotations

import logging
from typing import Any

from src.framework.agents.base import AgentContext, AgentResult, AsyncAgentBase
from src.framework.harness.loop.runner import HarnessRunner, RunResult
from src.framework.harness.outcomes import Terminal


class HarnessAgentAdapter(AsyncAgentBase):
    """Wrap a :class:`HarnessRunner` as an :class:`AsyncAgentBase`."""

    def __init__(
        self,
        runner: HarnessRunner,
        *,
        name: str = "harness",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_adapter=runner.llm,
            logger=logger or logging.getLogger(__name__),
            name=name,
        )
        self._runner = runner

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Run the harness against the agent context's ``query``."""
        run_result = await self._runner.run(self._context_to_intent(context))
        return self._run_result_to_agent_result(run_result)

    @staticmethod
    def _context_to_intent(context: AgentContext) -> dict[str, Any]:
        """Project ``AgentContext`` into the runner's intent payload."""
        return {
            "id": context.session_id,
            "goal": context.query,
            "metadata": {
                "rag_context": context.rag_context,
                **context.metadata,
                **context.additional_context,
            },
        }

    def _run_result_to_agent_result(self, run: RunResult) -> AgentResult:
        accepted = isinstance(run.outcome, Terminal) and run.outcome.accepted
        verification = run.state.last_verification
        confidence = run.confidence
        decomposition_score = (run.state.plan and (1.0 if run.state.plan.steps else 0.0)) or 0.0
        return AgentResult(
            response=run.response_text,
            confidence=confidence,
            agent_name=self.name,
            processing_time_ms=run.duration_ms,
            metadata={
                "outcome": run.outcome.kind,
                "iterations": run.iterations,
                "correlation_id": run.metadata.get("correlation_id", ""),
                "tokens_consumed": run.metadata.get("tokens_consumed", 0),
                # Keys that GraphBuilder._parallel_agents_node reads:
                "decomposition_quality_score": decomposition_score,
                "confidence": confidence,
                "verification_passed": bool(verification and verification.passed),
                "verification_score": verification.score if verification else 0.0,
            },
            intermediate_steps=[dict(entry) for entry in run.state.history],
            success=accepted or (verification.passed if verification else False),
            error=None if accepted else f"outcome={run.outcome.kind}",
        )


__all__ = ["HarnessAgentAdapter"]
