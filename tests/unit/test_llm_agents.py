"""
Tests for LLM-backed HRM and TRM agents.

Uses a mock LLM client to verify agent behavior without API calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Mock LLM adapter implementing the LLMClient protocol
# ---------------------------------------------------------------------------


@dataclass
class MockLLMResponse:
    text: str
    usage: dict = field(default_factory=lambda: {"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50})
    model: str = "mock"
    raw_response: Any = None
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


class MockLLMAdapter:
    """Mock LLM client that implements the LLMClient protocol."""

    def __init__(self, response_text: str | None = None):
        self._response_text = response_text
        self.call_count = 0
        self.last_prompt = None

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> MockLLMResponse:
        self.call_count += 1
        self.last_prompt = prompt

        if self._response_text:
            return MockLLMResponse(text=self._response_text)

        # Generate strategy-specific mock responses
        if prompt and "hierarchical" in prompt.lower():
            return MockLLMResponse(
                text=(
                    "### Sub-problems\n"
                    "1. Core architecture\n"
                    "   **Answer:** Use event-driven design\n"
                    "2. Scalability\n"
                    "   **Answer:** Horizontal scaling with sharding\n\n"
                    "### Synthesis\n"
                    "The final integrated answer combines event-driven architecture "
                    "with horizontal scaling to meet all requirements."
                )
            )
        elif prompt and "refinement" in prompt.lower():
            return MockLLMResponse(
                text=(
                    "### Initial Answer\n"
                    "Use a monolithic approach for simplicity.\n\n"
                    "### Critical Review\n"
                    "- Weakness: Doesn't scale for large teams\n"
                    "- Weakness: Deployment coupling\n\n"
                    "### Refined Answer\n"
                    "Start with a modular monolith that can be improved "
                    "and decomposed into services as needs evolve."
                )
            )

        return MockLLMResponse(text="Default mock response for testing.")


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine synchronously for testing."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# LLMHRMAgent tests
# ---------------------------------------------------------------------------


class TestLLMHRMAgent:
    def _make_agent(self, response_text: str | None = None):
        from src.framework.agents.llm_hrm import LLMHRMAgent
        adapter = MockLLMAdapter(response_text)
        return LLMHRMAgent(model_adapter=adapter, name="test_hrm"), adapter

    def test_process_returns_dict(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="How to design a database?"))
        assert isinstance(result, dict)
        assert "response" in result
        assert "metadata" in result

    def test_response_is_non_empty(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="Explain microservices"))
        assert len(result["response"]) > 0

    def test_metadata_contains_quality_score(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="Design a cache"))
        assert "decomposition_quality_score" in result["metadata"]
        score = result["metadata"]["decomposition_quality_score"]
        assert 0.0 <= score <= 1.0

    def test_metadata_indicates_strategy(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="test"))
        assert result["metadata"]["strategy"] == "hierarchical_decomposition"

    def test_structured_response_gets_higher_score(self):
        # Response with sub-problems and synthesis should score higher
        structured = (
            "### Sub-problems\n1. Sub-problem A\n   **Answer:** X\n\n"
            "### Synthesis\nFinal integrated answer combining all insights."
        )
        agent_structured, _ = self._make_agent(structured)
        result_structured = run_async(agent_structured.process(query="test"))

        plain = "Here is a short answer."
        agent_plain, _ = self._make_agent(plain)
        result_plain = run_async(agent_plain.process(query="test"))

        assert (
            result_structured["metadata"]["decomposition_quality_score"]
            >= result_plain["metadata"]["decomposition_quality_score"]
        )

    def test_rag_context_included_in_prompt(self):
        agent, adapter = self._make_agent("Response")
        run_async(agent.process(query="test", rag_context="Relevant context here"))
        assert "Relevant context here" in adapter.last_prompt

    def test_calls_llm_exactly_once(self):
        agent, adapter = self._make_agent()
        run_async(agent.process(query="test"))
        assert adapter.call_count == 1


# ---------------------------------------------------------------------------
# LLMTRMAgent tests
# ---------------------------------------------------------------------------


class TestLLMTRMAgent:
    def _make_agent(self, response_text: str | None = None):
        from src.framework.agents.llm_trm import LLMTRMAgent
        adapter = MockLLMAdapter(response_text)
        return LLMTRMAgent(model_adapter=adapter, name="test_trm"), adapter

    def test_process_returns_dict(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="How to scale a system?"))
        assert isinstance(result, dict)
        assert "response" in result
        assert "metadata" in result

    def test_response_is_non_empty(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="Explain caching"))
        assert len(result["response"]) > 0

    def test_metadata_contains_quality_score(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="Design an API"))
        assert "final_quality_score" in result["metadata"]
        score = result["metadata"]["final_quality_score"]
        assert 0.0 <= score <= 1.0

    def test_metadata_indicates_strategy(self):
        agent, _ = self._make_agent()
        result = run_async(agent.process(query="test"))
        assert result["metadata"]["strategy"] == "iterative_refinement"

    def test_refined_response_gets_higher_score(self):
        refined = (
            "### Initial Answer\nBasic approach.\n\n"
            "### Critical Review\n- Weakness: incomplete\n\n"
            "### Refined Answer\nImproved comprehensive approach."
        )
        agent_refined, _ = self._make_agent(refined)
        result_refined = run_async(agent_refined.process(query="test"))

        plain = "Short answer."
        agent_plain, _ = self._make_agent(plain)
        result_plain = run_async(agent_plain.process(query="test"))

        assert (
            result_refined["metadata"]["final_quality_score"]
            >= result_plain["metadata"]["final_quality_score"]
        )

    def test_rag_context_included_in_prompt(self):
        agent, adapter = self._make_agent("Response")
        run_async(agent.process(query="test", rag_context="Extra context"))
        assert "Extra context" in adapter.last_prompt

    def test_calls_llm_exactly_once(self):
        agent, adapter = self._make_agent()
        run_async(agent.process(query="test"))
        assert adapter.call_count == 1


# ---------------------------------------------------------------------------
# Agent compatibility tests
# ---------------------------------------------------------------------------


class TestAgentCompatibility:
    """Verify the LLM agents produce the dict format expected by graph.py."""

    def test_hrm_output_matches_graph_expectations(self):
        """graph.py expects {"response": ..., "metadata": {...decomposition_quality_score...}}"""
        from src.framework.agents.llm_hrm import LLMHRMAgent
        adapter = MockLLMAdapter()
        agent = LLMHRMAgent(model_adapter=adapter)
        result = run_async(agent.process(query="test"))

        # GraphBuilder._hrm_agent_node reads these keys
        assert "response" in result
        assert "metadata" in result
        assert "decomposition_quality_score" in result["metadata"]

    def test_trm_output_matches_graph_expectations(self):
        """graph.py expects {"response": ..., "metadata": {...final_quality_score...}}"""
        from src.framework.agents.llm_trm import LLMTRMAgent
        adapter = MockLLMAdapter()
        agent = LLMTRMAgent(model_adapter=adapter)
        result = run_async(agent.process(query="test"))

        # GraphBuilder._trm_agent_node reads these keys
        assert "response" in result
        assert "metadata" in result
        assert "final_quality_score" in result["metadata"]
