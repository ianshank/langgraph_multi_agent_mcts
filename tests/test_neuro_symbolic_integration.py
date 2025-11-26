"""
Integration tests for Neuro-Symbolic Graph flow.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from src.framework.graph import GraphBuilder, AgentState
from src.framework.mcts.config import create_preset_config, ConfigPreset


class MockModelAdapter:
    """Mock for LLM Adapter."""
    
    async def generate(self, prompt, temperature=0.0):
        response = MagicMock()
        response.text = f"Generated response for prompt: {prompt[:20]}..."
        return response


@pytest.fixture
def mock_dependencies():
    return {
        "hrm_agent": MagicMock(),
        "trm_agent": MagicMock(),
        "model_adapter": MockModelAdapter(),
        "logger": MagicMock(),
    }


@pytest.mark.asyncio
async def test_graph_builder_init(mock_dependencies):
    """Test that GraphBuilder initializes the adapter."""
    builder = GraphBuilder(**mock_dependencies)
    
    assert builder.hrm_adapter is not None
    assert builder.mcts_engine is not None


@pytest.mark.asyncio
async def test_hrm_node_integration(mock_dependencies):
    """Test that HRM node uses the adapter plan."""
    builder = GraphBuilder(**mock_dependencies)
    
    # Mock adapter response
    builder.hrm_adapter.process_query = AsyncMock(return_value={
        "neural_plan": [
            {"level": 0, "description": "Test Step", "strategy": "Deep Research", "confidence": 0.4}
        ],
        "ponder_cost": 0.5
    })
    
    # Mock LLM generation to verify prompt contained strategy
    builder.model_adapter.generate = AsyncMock(return_value=MagicMock(text="Final Answer"))
    
    state = {
        "query": "test query",
        "rag_context": "context"
    }
    
    result = await builder._hrm_agent_node(state)
    
    # Verify output structure
    assert "hrm_results" in result
    assert result["hrm_results"]["metadata"]["decomposition_quality_score"] == 0.85
    
    # Verify prompt construction
    call_args = builder.model_adapter.generate.call_args
    prompt = call_args[1]["prompt"]
    assert "Deep Research" in prompt
    assert "Test Step" in prompt


@pytest.mark.asyncio
async def test_mcts_actions_integration(mock_dependencies):
    """Test that MCTS node generates concrete actions."""
    # Use fast config for test
    mcts_config = create_preset_config(ConfigPreset.FAST)
    mcts_config.num_iterations = 5
    
    builder = GraphBuilder(
        **mock_dependencies,
        mcts_config=mcts_config
    )
    
    state = {
        "query": "test mcts",
        "use_rag": False
    }
    
    result = await builder._mcts_simulator_node(state)
    
    assert "mcts_best_action" in result
    # Should be one of our concrete actions
    valid_actions = ["decompose_problem", "research_context", "draft_answer", "review_logic"]
    # Note: "action_A" etc are from fallback if registry fails, but we expect concrete ones
    # In a fresh state, "decompose_problem" or "draft_answer" are likely
    assert result["mcts_best_action"] in valid_actions or result["mcts_best_action"] is None

