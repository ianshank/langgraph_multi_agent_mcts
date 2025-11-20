"""
Contract tests for API and interface compliance.

Contract tests validate that components adhere to their defined
interfaces and contracts. This ensures:
- Interface compatibility
- Protocol compliance
- Consistent behavior across implementations
- Safe refactoring

Best Practices 2025:
- Test protocol compliance
- Validate interface contracts
- Check backward compatibility
- Document expected behaviors
"""

import pytest
from typing import Protocol, runtime_checkable

from src.adapters.llm.base import LLMClient, LLMResponse
from src.framework.agents.base import AgentContext, AgentResult


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol defining LLM client contract."""

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate completion from prompt."""
        ...

    async def stream(self, prompt: str, **kwargs):
        """Stream completion from prompt."""
        ...


class TestLLMClientContract:
    """Contract tests for LLM client implementations."""

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_openai_client_implements_protocol(self):
        """
        Contract: OpenAI client should implement LLM client interface.

        This ensures the OpenAI adapter adheres to the standard interface.
        """
        from src.adapters.llm.openai_client import OpenAIClient

        # Create client (even without API key for contract test)
        client = OpenAIClient(
            api_key="test-key",  # Won't actually call API
            model="gpt-4",
            timeout=30.0,
        )

        # Verify interface compliance
        assert hasattr(client, "generate")
        assert callable(client.generate)
        assert hasattr(client, "close")
        assert callable(client.close)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_anthropic_client_implements_protocol(self):
        """
        Contract: Anthropic client should implement LLM client interface.
        """
        from src.adapters.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(
            api_key="test-key",
            model="claude-3-sonnet-20240229",
            timeout=30.0,
        )

        assert hasattr(client, "generate")
        assert callable(client.generate)
        assert hasattr(client, "close")
        assert callable(client.close)

    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_lmstudio_client_implements_protocol(self):
        """
        Contract: LM Studio client should implement LLM client interface.
        """
        from src.adapters.llm.lmstudio_client import LMStudioClient

        client = LMStudioClient(
            base_url="http://localhost:1234/v1",
            model="local-model",
            timeout=30.0,
        )

        assert hasattr(client, "generate")
        assert callable(client.generate)
        assert hasattr(client, "close")
        assert callable(client.close)

    @pytest.mark.contract
    def test_llm_response_structure(self):
        """
        Contract: LLM responses should have required fields.

        This validates the response data structure contract.
        """
        response = LLMResponse(
            text="Test response",
            model="test-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )

        # Required fields
        assert hasattr(response, "text")
        assert hasattr(response, "model")
        assert hasattr(response, "usage")
        assert hasattr(response, "prompt_tokens")
        assert hasattr(response, "completion_tokens")
        assert hasattr(response, "total_tokens")

        # Type validation
        assert isinstance(response.text, str)
        assert isinstance(response.model, str)
        assert isinstance(response.prompt_tokens, int)
        assert isinstance(response.completion_tokens, int)
        assert isinstance(response.total_tokens, int)

        # Invariants
        assert response.total_tokens >= response.prompt_tokens
        assert response.total_tokens >= response.completion_tokens


class TestAgentContract:
    """Contract tests for agent implementations."""

    @pytest.mark.contract
    def test_agent_context_structure(self):
        """
        Contract: Agent context should have required fields.
        """
        context = AgentContext(
            query="test query",
            session_id="test-session",
        )

        # Required fields
        assert hasattr(context, "query")
        assert hasattr(context, "session_id")
        assert hasattr(context, "rag_context")
        assert hasattr(context, "metadata")
        assert hasattr(context, "conversation_history")
        assert hasattr(context, "max_iterations")
        assert hasattr(context, "temperature")

        # Type validation
        assert isinstance(context.query, str)
        assert isinstance(context.session_id, str)
        assert isinstance(context.metadata, dict)
        assert isinstance(context.conversation_history, list)
        assert isinstance(context.max_iterations, int)
        assert isinstance(context.temperature, float)

    @pytest.mark.contract
    def test_agent_result_structure(self):
        """
        Contract: Agent result should have required fields.
        """
        result = AgentResult(
            response="test response",
            confidence=0.95,
            agent_name="TestAgent",
        )

        # Required fields
        assert hasattr(result, "response")
        assert hasattr(result, "confidence")
        assert hasattr(result, "metadata")
        assert hasattr(result, "agent_name")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "token_usage")
        assert hasattr(result, "success")
        assert hasattr(result, "error")

        # Type validation
        assert isinstance(result.response, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.agent_name, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.metadata, dict)

        # Invariants
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms >= 0

    @pytest.mark.contract
    def test_agent_context_serialization(self):
        """
        Contract: Agent context should be serializable to dict.
        """
        context = AgentContext(
            query="test query",
            metadata={"key": "value"},
        )

        # Should have to_dict method
        assert hasattr(context, "to_dict")
        assert callable(context.to_dict)

        # Serialization should produce dict
        serialized = context.to_dict()
        assert isinstance(serialized, dict)
        assert "query" in serialized
        assert "metadata" in serialized

    @pytest.mark.contract
    def test_agent_result_serialization(self):
        """
        Contract: Agent result should be serializable to dict.
        """
        result = AgentResult(
            response="test",
            confidence=0.9,
        )

        # Should have to_dict method
        assert hasattr(result, "to_dict")
        assert callable(result.to_dict)

        # Serialization should produce dict
        serialized = result.to_dict()
        assert isinstance(serialized, dict)
        assert "response" in serialized
        assert "confidence" in serialized


class TestMCTSContract:
    """Contract tests for MCTS components."""

    @pytest.mark.contract
    def test_mcts_state_has_hash_method(self):
        """
        Contract: MCTS states should be hashable for caching.
        """
        from src.framework.mcts.core import MCTSState

        state = MCTSState(state_id="test", features={"key": "value"})

        # Should have hash method
        assert hasattr(state, "to_hash_key")
        assert callable(state.to_hash_key)

        # Hash should be string
        hash_key = state.to_hash_key()
        assert isinstance(hash_key, str)
        assert len(hash_key) > 0

    @pytest.mark.contract
    def test_mcts_node_structure(self):
        """
        Contract: MCTS nodes should have required attributes.
        """
        from src.framework.mcts.core import MCTSNode, MCTSState

        state = MCTSState(state_id="test", features={})
        node = MCTSNode(state=state)

        # Required attributes
        assert hasattr(node, "state")
        assert hasattr(node, "parent")
        assert hasattr(node, "action")
        assert hasattr(node, "children")
        assert hasattr(node, "visits")
        assert hasattr(node, "value_sum")
        assert hasattr(node, "depth")

        # Type validation
        assert isinstance(node.children, list)
        assert isinstance(node.visits, int)
        assert isinstance(node.depth, int)

        # Invariants
        assert node.visits >= 0
        assert node.depth >= 0

    @pytest.mark.contract
    def test_mcts_node_value_property(self):
        """
        Contract: MCTS nodes should provide value property.
        """
        from src.framework.mcts.core import MCTSNode, MCTSState

        state = MCTSState(state_id="test", features={})
        node = MCTSNode(state=state)

        # Should have value property
        assert hasattr(node, "value")

        # Value should be float
        value = node.value
        assert isinstance(value, float)

        # Initial value should be 0
        assert value == 0.0


class TestFactoryContract:
    """Contract tests for factory implementations."""

    @pytest.mark.contract
    def test_llm_factory_has_create_method(self):
        """
        Contract: LLM factory should have create method.
        """
        from src.framework.factories import LLMClientFactory

        # Check the class has the required methods without instantiation
        assert hasattr(LLMClientFactory, "create")
        assert hasattr(LLMClientFactory, "create_from_settings")
        assert hasattr(LLMClientFactory, "_get_default_model")

    @pytest.mark.contract
    def test_mcts_factory_has_create_method(self):
        """
        Contract: MCTS factory should have create method.
        """
        from src.framework.factories import MCTSEngineFactory

        # Check the class has the required methods
        assert hasattr(MCTSEngineFactory, "create")
        assert hasattr(MCTSEngineFactory, "_get_preset_config")


class TestValidationContract:
    """Contract tests for validation models."""

    @pytest.mark.contract
    def test_query_input_validation(self):
        """
        Contract: QueryInput should validate input.
        """
        from src.models.validation import QueryInput

        # Valid request should work
        request = QueryInput(query="test query")
        assert request.query == "test query"

        # Should have required fields
        assert hasattr(request, "query")

    @pytest.mark.contract
    def test_query_input_pydantic_compliance(self):
        """
        Contract: QueryInput should be a Pydantic model.
        """
        from src.models.validation import QueryInput
        from pydantic import BaseModel

        # Should inherit from BaseModel
        assert issubclass(QueryInput, BaseModel)

        # Should support dict conversion
        request = QueryInput(query="test")
        assert hasattr(request, "model_dump")
        data = request.model_dump()
        assert isinstance(data, dict)
        assert "query" in data


# Run contract tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "contract"])
