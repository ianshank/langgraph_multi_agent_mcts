"""
Unit tests for DistillationDataset.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.training.distillation_dataset import DistillationDataset, DistillationTask
from src.framework.mcts.llm_guided.data_collector import TrainingExample, EpisodeMetadata

@pytest.fixture
def sample_data(tmp_path):
    """Create a temporary JSONL file with sample training data."""
    file_path = tmp_path / "episode_test_123.jsonl"
    
    metadata = EpisodeMetadata(
        episode_id="test_ep",
        problem_type="code_generation",
        difficulty="easy",
        start_time=1000.0,
        outcome=1.0
    )
    
    example1 = TrainingExample(
        state_code="def foo(): pass",
        state_problem="Write foo",
        state_hash="hash1",
        depth=1,
        llm_action_probs={"action1": 0.9},
        mcts_action_probs={"action1": 0.95},
        llm_value_estimate=0.8,
        outcome=1.0,
        episode_id="test_ep",
        timestamp=1001.0,
        visits=10,
        q_value=0.9,
        action="action1",
        test_results={"decomposition": ["sub1"]},
        parent_visits=5
    )
    
    example2 = TrainingExample(
        state_code="def foo(): return 1",
        state_problem="Write foo",
        state_hash="hash2",
        depth=2,
        llm_action_probs={"action2": 0.8},
        mcts_action_probs={"action2": 0.85},
        llm_value_estimate=0.9,
        outcome=1.0,
        episode_id="test_ep",
        timestamp=1002.0,
        visits=20,
        q_value=0.95,
        action="action2",
        test_results={"refined_code": "def foo(): return 1"},
        parent_visits=10
    )

    with open(file_path, "w") as f:
        f.write(json.dumps({"_metadata": metadata.to_dict()}) + "\n")
        f.write(example1.to_json() + "\n")
        f.write(example2.to_json() + "\n")
        
    return tmp_path

def test_load_dataset(sample_data):
    """Test loading the dataset."""
    dataset = DistillationDataset(
        data_dir=sample_data,
        task_type=DistillationTask.POLICY_VALUE
    )
    
    assert len(dataset) == 2
    assert len(dataset.metadata) == 1
    assert dataset.metadata[0].episode_id == "test_ep"

def test_policy_value_mapping(sample_data):
    """Test mapping for policy-value task."""
    dataset = DistillationDataset(
        data_dir=sample_data,
        task_type=DistillationTask.POLICY_VALUE
    )
    
    sample = dataset[0]
    assert sample["state_code"] == "def foo(): pass"
    assert sample["target_policy"] == {"action1": 0.95}
    assert sample["target_value"] == 1.0

def test_hrm_mapping(sample_data):
    """Test mapping for HRM task."""
    dataset = DistillationDataset(
        data_dir=sample_data,
        task_type=DistillationTask.HRM_DECOMPOSITION
    )
    
    sample = dataset[0]
    assert sample["problem"] == "Write foo"
    assert sample["target_decomposition"] == ["sub1"]

def test_meta_controller_mapping(sample_data):
    """Test mapping for meta-controller task."""
    dataset = DistillationDataset(
        data_dir=sample_data,
        task_type=DistillationTask.META_CONTROLLER
    )
    
    sample = dataset[0]
    assert sample["features"]["visits"] == 10
    assert sample["outcome"] == 1.0
