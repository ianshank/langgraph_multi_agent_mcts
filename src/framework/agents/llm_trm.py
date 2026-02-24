"""
LLM-backed Task Refinement Module (TRM) agent.

Replaces the PyTorch-based TRM with an LLM-powered agent that iteratively
refines an answer by generating an initial response, critiquing it, and
producing a strengthened final answer.

Implements the AsyncAgentBase interface so it can be used as a drop-in
replacement in the LangGraph graph nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapters.llm.base import LLMClient
from src.framework.agents.base import AgentContext, AgentResult, AsyncAgentBase


REFINEMENT_PROMPT = """\
You are a Task Refinement Module (TRM). Your job is to produce a high-quality
answer through iterative self-refinement.

## Instructions
1. Write an initial answer to the query
2. Critically review your answer for gaps, errors, and unclear points
3. Produce a refined, strengthened final answer

## Query
{query}

{rag_section}

## Response Format
### Initial Answer
<your first-pass answer>

### Critical Review
- <weakness or gap 1>
- <weakness or gap 2>

### Refined Answer
<improved final answer addressing the weaknesses>
"""


class LLMTRMAgent(AsyncAgentBase):
    """
    LLM-powered Task Refinement Module.

    Generates an initial answer, self-critiques, and refines it into a
    stronger response. Compatible with LangGraph graph nodes via the
    AsyncAgentBase.process() interface.
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str = "LLM_TRM",
        temperature: float = 0.5,
        max_tokens: int = 1500,
        **config: Any,
    ):
        super().__init__(model_adapter, logger, name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Generate, critique, and refine an answer."""
        query = context.query

        # Build RAG section if available
        rag_section = ""
        if context.rag_context:
            rag_section = f"## Additional Context\n{context.rag_context}\n"

        prompt = REFINEMENT_PROMPT.format(query=query, rag_section=rag_section)

        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Extract quality signal from response structure
        text = response.text
        has_initial = "initial" in text.lower()
        has_review = "review" in text.lower() or "weakness" in text.lower()
        has_refined = "refined" in text.lower() or "improved" in text.lower()
        quality_score = 0.5
        if has_initial:
            quality_score += 0.1
        if has_review:
            quality_score += 0.2
        if has_refined:
            quality_score += 0.15
        if len(text) > 300:
            quality_score += 0.05
        quality_score = min(quality_score, 1.0)

        return AgentResult(
            response=text,
            confidence=quality_score,
            metadata={
                "final_quality_score": quality_score,
                "has_initial_answer": has_initial,
                "has_critical_review": has_review,
                "has_refined_answer": has_refined,
                "strategy": "iterative_refinement",
            },
            token_usage=response.usage,
        )
