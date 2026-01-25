"""
End-to-End Test for the Neural Distillation Pipeline.
"""

import pytest
import shutil
import json
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.training.distillation_orchestrator import DistillationOrchestrator
from src.training.system_config import SystemConfig
from src.training.distillation_dataset import TrainingExample, EpisodeMetadata

@pytest.fixture
def e2e_env(tmp_path):
    """Setup temporary environment for E2E test."""
    data_dir = tmp_path / "training_data"
    output_dir = tmp_path / "distilled_models"
    data_dir.mkdir()
    output_dir.mkdir()
    
    return data_dir, output_dir

def create_synthetic_data(data_dir: Path):
    """Generate synthetic JSONL data for training."""
    
    # Create an episode trace
    metadata = EpisodeMetadata(
        episode_id="e2e_test_1",
        problem_type="code_generation",
        difficulty="easy",
        start_time=0.0,
        outcome=1.0,
        solution_found=True
    )
    
    examples = []
    
    # 1. Decomposition Example
    examples.append(TrainingExample(
        state_code="",
        state_problem="Write a function to add two numbers",
        state_hash="hash1",
        depth=0,
        llm_action_probs={},
        mcts_action_probs={},
        llm_value_estimate=1.0,
        outcome=1.0,
        episode_id="e2e_test_1",
        timestamp=1.0,
        visits=0,
        q_value=0.0,
        action="decompose",
        test_results={
            "decomposition": {
                "subproblems": ["validate inputs", "return sum"],
                "levels": [0, 0],
                "confidences": [0.9, 0.9]
            }
        }
    ))
    
    # 2. Refinement Example
    examples.append(TrainingExample(
        state_code="def add(a, b): pass",
        state_problem="Write a function to add two numbers",
        state_hash="hash2",
        depth=0,
        llm_action_probs={},
        mcts_action_probs={},
        llm_value_estimate=0.5,
        outcome=1.0,
        episode_id="e2e_test_1",
        timestamp=2.0,
        visits=0,
        q_value=0.0,
        action="refine",
        test_results={
            "refined_code": "def add(a, b): return a + b",
            "refinement_metadata": {"improvement": 1.0}
        }
    ))
    
    # Save to file
    filepath = data_dir / "episode_e2e_1.jsonl"
    with open(filepath, "w") as f:
        f.write(json.dumps({"_metadata": metadata.to_dict()}) + "\n")
        for ex in examples:
            f.write(ex.to_json() + "\n")

    return filepath

def test_e2e_distillation_loop(e2e_env):
    """
    Test the full loop: Data -> Train -> Save Models.
    Matches the flow described in the verification plan.
    """
    data_dir, output_dir = e2e_env
    create_synthetic_data(data_dir)
    
    # Use CPU config for test
    config = SystemConfig(device="cpu")
    
    # Mock Encoder/Decoder to avoid downloading real models during test
    # (unless we want to verify real model loading, which might be slow/heavy)
    # Ideally E2E tests run real logic, but for CI speed we often mock the heavy weights.
    # Let's mock the internal transformer loading but keep the logic flow.
    
    with patch("src.agents.common.system_encoder.AutoTokenizer") as MockEncTok, \
         patch("src.agents.common.system_encoder.AutoModel") as MockEncModel, \
         patch("src.agents.common.text_decoder.AutoTokenizer") as MockDecTok, \
         patch("src.agents.common.text_decoder.GPT2LMHeadModel") as MockDecModel, \
         patch("src.agents.common.text_decoder.GPT2Config") as MockDecConfig:
         
        # Configure minimal mocks to pass shape checks
        enc_model_instance = MockEncModel.from_pretrained.return_value
        enc_model_instance.config.hidden_size = 32
        enc_model_instance.to.return_value = enc_model_instance # Allow chaining
        enc_model_instance.return_value.last_hidden_state = torch.randn(1, 10, 32)
        
        # Encoder tokenizer needs to return dict with tensors
        enc_tok_ret = MagicMock()
        enc_tok_ret.to.return_value = enc_tok_ret
        enc_tok_ret.__getitem__.side_effect = lambda k: torch.randn(1, 10)
        MockEncTok.from_pretrained.return_value.return_value = enc_tok_ret
        
        MockDecConfig.from_pretrained.return_value.n_embd = 32
        
        dec_model = MagicMock()
        dec_model.loss = torch.tensor(0.1, requires_grad=True) # allow backward
        dec_model.parameters.return_value = [torch.tensor(0.0, requires_grad=True)] # allow optimizer
        dec_model.to.return_value = dec_model # Allow chaining
        
        # Mocking return value of forward pass to have .loss attribute
        dec_output = MagicMock()
        dec_output.loss = torch.tensor(0.1, requires_grad=True)
        dec_model.return_value = dec_output
        
        MockDecModel.from_pretrained.return_value = dec_model
        
        # Instantiate Orchestrator
        orchestrator = DistillationOrchestrator(
            config=config,
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # 1. Train HRM
        # We need to ensure SystemEncoder/Decoder are initialized
        # They are initialized in __init__ of orchestrator using our mocks
        
        metrics_hrm = orchestrator.distill_hrm(batch_size=1, num_epochs=1)
        assert metrics_hrm["avg_loss"] is not None
        assert (output_dir / "hrm_agent.pt").exists()
        
        # 2. Train TRM
        metrics_trm = orchestrator.distill_trm(batch_size=1, num_epochs=1)
        assert metrics_trm["avg_loss"] is not None
        assert (output_dir / "trm_agent.pt").exists()
        assert (output_dir / "system_decoder.pt").exists()
