"""
LLM-backed Hierarchical Reasoning Module (HRM) agent.

Replaces the PyTorch-based HRM with an LLM-powered agent that decomposes
queries into sub-problems, solves each, and synthesizes a final answer.

Implements the AsyncAgentBase interface so it can be used as a drop-in
replacement in the LangGraph graph nodes.
"""

from __future__ import annotations

import logging
from typing import Any

from src.adapters.llm.base import LLMClient
from src.framework.agents.base import AgentContext, AgentResult, AsyncAgentBase


DECOMPOSITION_PROMPT = """\
You are a Hierarchical Reasoning Module (HRM). Your job is to decompose
complex questions into sub-problems, solve each independently, then
synthesize a final answer.

## Instructions
1. Identify 2-4 sub-problems from the query
2. Solve each sub-problem with a brief, focused answer
3. Synthesize the sub-problem answers into a comprehensive final answer

## Query
{query}

{rag_section}

## Response Format
### Sub-problems
1. <sub-problem 1>
   **Answer:** <answer>
2. <sub-problem 2>
   **Answer:** <answer>

### Synthesis
<Final integrated answer combining insights from all sub-problems>
"""


class LLMHRMAgent(AsyncAgentBase):
    """
    LLM-powered Hierarchical Reasoning Module.

    Decomposes queries into sub-problems, solves each via LLM calls,
    and synthesizes a comprehensive answer. Compatible with LangGraph
    graph nodes via the AsyncAgentBase.process() interface.
    """

    def __init__(
        self,
        model_adapter: LLMClient,
        logger: Any = None,
        name: str = "LLM_HRM",
        temperature: float = 0.5,
        max_tokens: int = 1500,
        **config: Any,
    ):
        super().__init__(model_adapter, logger, name, **config)
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def _process_impl(self, context: AgentContext) -> AgentResult:
        """Decompose the query and synthesize an answer."""
        query = context.query

        # Build RAG section if available
        rag_section = ""
        if context.rag_context:
            rag_section = f"## Additional Context\n{context.rag_context}\n"

        prompt = DECOMPOSITION_PROMPT.format(query=query, rag_section=rag_section)

        response = await self.generate_llm_response(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        # Extract quality signal from response structure
        text = response.text
        has_subproblems = "sub-problem" in text.lower() or "###" in text
        has_synthesis = "synthesis" in text.lower() or "final" in text.lower()
        quality_score = 0.5
        if has_subproblems:
            quality_score += 0.2
        if has_synthesis:
            quality_score += 0.15
        if len(text) > 300:
            quality_score += 0.1
        quality_score = min(quality_score, 1.0)

        return AgentResult(
            response=text,
            confidence=quality_score,
            metadata={
                "decomposition_quality_score": quality_score,
                "has_subproblems": has_subproblems,
                "has_synthesis": has_synthesis,
                "strategy": "hierarchical_decomposition",
            },
            token_usage=response.usage,
        )
