"""
Unit tests for Neuro-Symbolic Adapter.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.framework.adapters import NeuroSymbolicAdapter, NeuralPlanStep


class MockHRMAgent:
    """Mock HRM Agent for testing."""
    
    def __init__(self):
        self.config = MagicMock()
        self.config.h_dim = 768
        self.device = "cpu"
        
    def to(self, device):
        self.device = device
        
    def eval(self):
        pass


@pytest.fixture
def mock_agent():
    return MockHRMAgent()


@pytest.fixture
def adapter(mock_agent):
    return NeuroSymbolicAdapter(mock_agent)


@pytest.mark.asyncio
async def test_process_query_fallback(adapter):
    """Test fallback mechanism when agent doesn't have forward method."""
    query = "Test query"
    
    result = await adapter.process_query(query)
    
    assert "neural_plan" in result
    assert "ponder_cost" in result
    assert len(result["neural_plan"]) > 0
    
    step = result["neural_plan"][0]
    assert "description" in step
    assert "strategy" in step
    assert "confidence" in step


@pytest.mark.asyncio
async def test_process_query_with_forward(adapter):
    """Test processing when agent has forward method."""
    
    # Mock the agent's forward method
    adapter.agent.forward = MagicMock()
    
    # Create mock output with confidences that map to different strategies:
    # >= 0.8 -> Direct Answer
    # 0.6-0.8 -> Tool Use
    # 0.4-0.6 -> Step By Step
    # < 0.4 -> Deep Research
    mock_output = MagicMock()
    mock_output.subproblems = [
        MagicMock(level=0, description="Step 1", confidence=0.9),  # Direct Answer
        MagicMock(level=1, description="Step 2", confidence=0.3)   # Deep Research (< 0.4)
    ]
    mock_output.total_ponder_cost = 1.5
    mock_output.halt_step = 2
    mock_output.convergence_path = [0.5, 0.9]
    mock_output.ponder_output = None  # No PonderNet output
    
    adapter.agent.forward.return_value = mock_output
    
    result = await adapter.process_query("Complex query")
    
    assert len(result["neural_plan"]) == 2
    assert result["neural_plan"][0]["strategy"] == "Direct Answer"  # High confidence >= 0.8
    assert result["neural_plan"][1]["strategy"] == "Deep Research"  # Low confidence < 0.4
    assert result["ponder_cost"] == 1.5


def test_embed_query(adapter):
    """Test internal embedding logic."""
    query = "test embedding"
    tensor = adapter._embed_query(query)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 3  # [batch, seq, dim]
    assert tensor.shape[0] == 1

