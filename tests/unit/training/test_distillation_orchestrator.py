"""
Unit tests for DistillationOrchestrator.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.training.distillation_orchestrator import DistillationOrchestrator
from src.training.system_config import SystemConfig

class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 32

    def forward(self, texts):
        batch_size = len(texts)
        # return dummy embeddings [B, S, H]
        return torch.randn(batch_size, 10, self.hidden_size)

@pytest.fixture
def mock_config():
    return SystemConfig(device="cpu")

@pytest.fixture
def orchestrator(mock_config, tmp_path):
    with patch("src.training.distillation_orchestrator.SystemEncoder", return_value=MockEncoder()), \
         patch("src.training.distillation_orchestrator.SystemDecoder") as MockDecoderParams: # Use class patch
        
        # Setup Mock Decoder Instance
        mock_decoder_instance = MagicMock()
        mock_decoder_instance.parameters.return_value = [torch.randn(1, requires_grad=True)]
        mock_decoder_instance.tokenizer = MagicMock()
        mock_decoder_instance.tokenizer.return_value = MagicMock(input_ids=torch.ones(1, 10, dtype=torch.long))

        # Setup forward return
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.0, requires_grad=True)
        mock_decoder_instance.return_value = mock_output
        
        MockDecoderParams.return_value = mock_decoder_instance
        
        orch = DistillationOrchestrator(
            config=mock_config,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "models"
        )
        # Inject mocks again
        orch.encoder = MockEncoder() 
        orch.decoder = mock_decoder_instance
        return orch

def test_distill_hrm(orchestrator):
    """Test HRM distillation loop."""
    # Mock dataset
    with patch("src.training.distillation_orchestrator.DistillationDataset") as MockDataset, \
         patch("torch.save") as mock_save:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        mock_ds.__getitem__.return_value = {
            "problem": "Write foo",
            "target_decomposition": ["Step 1", "Step 2"]
        }
        MockDataset.return_value = mock_ds
        
        # Run distillation
        metrics = orchestrator.distill_hrm(batch_size=2, num_epochs=1)
        
        assert "avg_loss" in metrics
        assert orchestrator.hrm_agent is not None
        assert mock_save.called

def test_distill_trm(orchestrator):
    """Test TRM distillation loop."""
    # Mock dataset
    with patch("src.training.distillation_orchestrator.DistillationDataset") as MockDataset, \
         patch("torch.save") as mock_save:
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        mock_ds.__getitem__.return_value = {
            "initial_code": "def foo(): pass",
            "target_code": "def foo(): return 1",
            "success": True
        }
        MockDataset.return_value = mock_ds
        
        # Run distillation
        metrics = orchestrator.distill_trm(batch_size=2, num_epochs=1)
        
        assert "avg_loss" in metrics
        assert orchestrator.trm_agent is not None
        assert mock_save.called
