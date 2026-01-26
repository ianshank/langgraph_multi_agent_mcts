"""
Unit tests for LLM-Guided MCTS Training Module (Phase 2).

Tests:
- MCTSDataset and data loading
- PolicyNetwork and ValueNetwork
- DistillationTrainer
- Training metrics computation
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# Check for numpy availability
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Check for PyTorch availability
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Skip all tests if PyTorch or numpy not available
pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE or not _NUMPY_AVAILABLE, reason="PyTorch and numpy required for training tests"
)


class TestMCTSDatasetConfig:
    """Tests for MCTSDatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDatasetConfig

        config = MCTSDatasetConfig()

        assert config.max_code_length == 2048
        assert config.max_problem_length == 1024
        assert config.max_actions == 10
        assert config.min_visits == 1

    def test_config_validation(self):
        """Test configuration validation."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDatasetConfig

        config = MCTSDatasetConfig(max_code_length=0)

        with pytest.raises(ValueError, match="max_code_length must be >= 1"):
            config.validate()

    def test_custom_config(self):
        """Test custom configuration values."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDatasetConfig

        config = MCTSDatasetConfig(
            max_code_length=1024,
            max_actions=5,
            min_visits=10,
        )

        config.validate()
        assert config.max_code_length == 1024
        assert config.max_actions == 5
        assert config.min_visits == 10


class TestRawExample:
    """Tests for RawExample data structure."""

    def test_create_raw_example(self):
        """Test creating a raw example."""
        from src.framework.mcts.llm_guided.training.dataset import RawExample

        example = RawExample(
            state_code="def foo(): pass",
            state_problem="Write a function",
            state_hash="abc123",
            depth=2,
            llm_action_probs={"a": 0.5, "b": 0.5},
            mcts_action_probs={"a": 0.7, "b": 0.3},
            llm_value_estimate=0.6,
            outcome=1.0,
            episode_id="ep_001",
            visits=10,
            q_value=0.8,
        )

        assert example.state_code == "def foo(): pass"
        assert example.depth == 2
        assert example.visits == 10


class TestMCTSDataset:
    """Tests for MCTSDataset."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a temporary directory with sample training data."""
        data_dir = tmp_path / "training_data"
        data_dir.mkdir()

        # Create a sample episode file
        episode_file = data_dir / "episode_001.jsonl"
        examples = [
            {
                "_metadata": {"episode_id": "ep_001", "solution_found": True},
            },
            {
                "state_code": "def solution(): pass",
                "state_problem": "Write a solution",
                "state_hash": "hash1",
                "depth": 1,
                "llm_action_probs": {"variant_0": 0.6, "variant_1": 0.4},
                "mcts_action_probs": {"variant_0": 0.8, "variant_1": 0.2},
                "llm_value_estimate": 0.5,
                "outcome": 1.0,
                "episode_id": "ep_001",
                "visits": 5,
                "q_value": 0.7,
            },
            {
                "state_code": "def solution():\n    return 42",
                "state_problem": "Write a solution",
                "state_hash": "hash2",
                "depth": 2,
                "llm_action_probs": {"variant_0": 0.5, "variant_1": 0.5},
                "mcts_action_probs": {"variant_0": 0.9, "variant_1": 0.1},
                "llm_value_estimate": 0.8,
                "outcome": 1.0,
                "episode_id": "ep_001",
                "visits": 10,
                "q_value": 0.9,
            },
        ]

        with open(episode_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        return data_dir

    @patch("src.framework.mcts.llm_guided.training.dataset.MCTSDataset._init_tokenizer")
    def test_dataset_loading(self, mock_tokenizer, sample_data_dir):
        """Test loading dataset from files."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDataset, MCTSDatasetConfig

        mock_tokenizer.return_value = None

        config = MCTSDatasetConfig(
            data_dir=sample_data_dir,
            exclude_root_nodes=False,
        )
        dataset = MCTSDataset(config=config)

        # Should have 2 examples (excluding metadata)
        assert len(dataset) == 2

    @patch("src.framework.mcts.llm_guided.training.dataset.MCTSDataset._init_tokenizer")
    def test_dataset_filtering(self, mock_tokenizer, sample_data_dir):
        """Test dataset filtering by min_visits."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDataset, MCTSDatasetConfig

        mock_tokenizer.return_value = None

        config = MCTSDatasetConfig(
            data_dir=sample_data_dir,
            min_visits=8,
            exclude_root_nodes=False,
        )
        dataset = MCTSDataset(config=config)

        # Should have only 1 example with visits >= 8
        assert len(dataset) == 1

    @patch("src.framework.mcts.llm_guided.training.dataset.MCTSDataset._init_tokenizer")
    def test_dataset_getitem(self, mock_tokenizer, sample_data_dir):
        """Test getting items from dataset."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDataset, MCTSDatasetConfig

        mock_tokenizer.return_value = None

        config = MCTSDatasetConfig(
            data_dir=sample_data_dir,
            exclude_root_nodes=False,
        )
        dataset = MCTSDataset(config=config)

        item = dataset[0]

        assert "code_tokens" in item
        assert "problem_tokens" in item
        assert "llm_policy" in item
        assert "mcts_policy" in item
        assert "outcome" in item
        assert isinstance(item["code_tokens"], torch.Tensor)

    @patch("src.framework.mcts.llm_guided.training.dataset.MCTSDataset._init_tokenizer")
    def test_dataset_statistics(self, mock_tokenizer, sample_data_dir):
        """Test dataset statistics computation."""
        from src.framework.mcts.llm_guided.training.dataset import MCTSDataset, MCTSDatasetConfig

        mock_tokenizer.return_value = None

        config = MCTSDatasetConfig(
            data_dir=sample_data_dir,
            exclude_root_nodes=False,
        )
        dataset = MCTSDataset(config=config)
        stats = dataset.get_statistics()

        assert stats["num_examples"] == 2
        assert stats["num_episodes"] == 1
        assert "depth_stats" in stats
        assert "visits_stats" in stats


class TestTrainingBatch:
    """Tests for TrainingBatch."""

    def test_batch_to_device(self):
        """Test moving batch to device."""
        from src.framework.mcts.llm_guided.training.dataset import TrainingBatch

        batch = TrainingBatch(
            code_tokens=torch.zeros(2, 10),
            code_attention_mask=torch.ones(2, 10),
            problem_tokens=torch.zeros(2, 10),
            problem_attention_mask=torch.ones(2, 10),
            llm_policy=torch.zeros(2, 5),
            mcts_policy=torch.zeros(2, 5),
            action_mask=torch.ones(2, 5),
            llm_value=torch.zeros(2),
            outcome=torch.zeros(2),
            q_value=torch.zeros(2),
            episode_ids=["ep1", "ep2"],
            depths=torch.zeros(2),
            visits=torch.zeros(2),
        )

        # Should not raise
        moved = batch.to("cpu")
        assert moved.code_tokens.device == torch.device("cpu")


class TestCodeEncoder:
    """Tests for CodeEncoder network."""

    def test_encoder_creation(self):
        """Test creating code encoder."""
        from src.framework.mcts.llm_guided.training.networks import CodeEncoder, CodeEncoderConfig

        config = CodeEncoderConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        encoder = CodeEncoder(config)

        assert encoder.output_dim == 64

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        from src.framework.mcts.llm_guided.training.networks import CodeEncoder, CodeEncoderConfig

        config = CodeEncoderConfig(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_seq_length=128,
        )
        encoder = CodeEncoder(config)

        # Create dummy input
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)

        output = encoder(input_ids, attention_mask)

        assert output.shape == (2, 64)


class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    @pytest.fixture
    def policy_network(self):
        """Create a small policy network for testing."""
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            PolicyNetwork,
            PolicyNetworkConfig,
        )

        encoder_config = CodeEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )
        config = PolicyNetworkConfig(
            encoder_config=encoder_config,
            max_actions=5,
            hidden_dim=32,
        )
        return PolicyNetwork(config)

    def test_policy_forward(self, policy_network):
        """Test policy network forward pass."""
        batch_size = 2
        seq_len = 16

        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)
        action_mask = torch.ones(batch_size, 5)

        log_probs = policy_network(code_tokens, code_mask, problem_tokens, problem_mask, action_mask)

        assert log_probs.shape == (batch_size, 5)
        # Log probs should be <= 0
        assert (log_probs <= 0).all()
        # Should sum to ~1 (log space)
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_policy_predict(self, policy_network):
        """Test policy network predict method."""
        batch_size = 2
        seq_len = 16

        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)

        probs = policy_network.predict(code_tokens, code_mask, problem_tokens, problem_mask)

        assert probs.shape == (batch_size, 5)
        # Probs should be in [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()


class TestValueNetwork:
    """Tests for ValueNetwork."""

    @pytest.fixture
    def value_network(self):
        """Create a small value network for testing."""
        from src.framework.mcts.llm_guided.training.networks import (
            CodeEncoderConfig,
            ValueNetwork,
            ValueNetworkConfig,
        )

        encoder_config = CodeEncoderConfig(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )
        config = ValueNetworkConfig(
            encoder_config=encoder_config,
            hidden_dim=32,
        )
        return ValueNetwork(config)

    def test_value_forward(self, value_network):
        """Test value network forward pass."""
        batch_size = 2
        seq_len = 16

        code_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        problem_tokens = torch.randint(0, 1000, (batch_size, seq_len))
        problem_mask = torch.ones(batch_size, seq_len)

        values = value_network(code_tokens, code_mask, problem_tokens, problem_mask)

        assert values.shape == (batch_size,)
        # Values should be in [-1, 1] (tanh output)
        assert (values >= -1).all() and (values <= 1).all()


class TestTrainingMetrics:
    """Tests for training metrics."""

    def test_metrics_accumulator(self):
        """Test metrics accumulator."""
        from src.framework.mcts.llm_guided.training.metrics import MetricsAccumulator

        accumulator = MetricsAccumulator()

        # Add some batches
        for _ in range(5):
            accumulator.update(
                policy_loss=0.5,
                value_loss=0.3,
                total_loss=0.8,
                policy_correct=8,
                policy_top3_correct=9,
                value_predictions=np.array([0.5, 0.6, 0.7]),
                value_targets=np.array([0.6, 0.6, 0.8]),
                batch_size=10,
            )

        metrics = accumulator.compute()

        assert metrics.policy_loss == pytest.approx(0.5, abs=0.01)
        assert metrics.value_loss == pytest.approx(0.3, abs=0.01)
        assert metrics.num_samples == 50
        assert metrics.policy_accuracy == pytest.approx(0.8, abs=0.01)

    def test_compute_policy_accuracy(self):
        """Test policy accuracy computation."""
        from src.framework.mcts.llm_guided.training.metrics import compute_policy_accuracy

        log_probs = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]]))
        targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        accuracy = compute_policy_accuracy(log_probs, targets, k=1)

        # First example: pred=0, target=0 (correct)
        # Second example: pred=1, target=1 (correct)
        assert accuracy == pytest.approx(1.0)

    def test_compute_value_mse(self):
        """Test value MSE computation."""
        from src.framework.mcts.llm_guided.training.metrics import compute_value_mse

        predictions = torch.tensor([0.5, 0.6, 0.7])
        targets = torch.tensor([0.4, 0.6, 0.8])

        mse = compute_value_mse(predictions, targets)

        expected = ((0.1**2) + 0 + (0.1**2)) / 3
        assert mse == pytest.approx(expected, abs=0.001)


class TestDistillationTrainerConfig:
    """Tests for DistillationTrainerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from src.framework.mcts.llm_guided.training.trainer import DistillationTrainerConfig

        config = DistillationTrainerConfig()

        assert config.num_epochs == 10
        assert config.learning_rate == 1e-4
        assert config.policy_loss_weight == 1.0
        assert config.value_loss_weight == 1.0

    def test_config_validation(self):
        """Test configuration validation."""
        from src.framework.mcts.llm_guided.training.trainer import DistillationTrainerConfig

        config = DistillationTrainerConfig(num_epochs=0)

        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            config.validate()


class TestTrainingCheckpoint:
    """Tests for TrainingCheckpoint."""

    def test_checkpoint_save_load(self, tmp_path):
        """Test saving and loading checkpoints."""
        from src.framework.mcts.llm_guided.training.trainer import TrainingCheckpoint

        checkpoint = TrainingCheckpoint(
            epoch=5,
            step=1000,
            best_metric=0.5,
            metrics_history=[{"loss": 0.6}, {"loss": 0.5}],
            config={"learning_rate": 0.001},
        )

        filepath = tmp_path / "checkpoint.pt"
        checkpoint.save(filepath)

        loaded = TrainingCheckpoint.load(filepath)

        assert loaded.epoch == 5
        assert loaded.step == 1000
        assert loaded.best_metric == 0.5
        assert len(loaded.metrics_history) == 2


class TestCreateNetworkFactories:
    """Tests for network factory functions."""

    def test_create_policy_network(self):
        """Test policy network factory."""
        from src.framework.mcts.llm_guided.training.networks import create_policy_network

        network = create_policy_network(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            max_actions=10,
        )

        assert network is not None
        assert network._config.max_actions == 10

    def test_create_value_network(self):
        """Test value network factory."""
        from src.framework.mcts.llm_guided.training.networks import create_value_network

        network = create_value_network(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )

        assert network is not None
        assert network._config.hidden_dim == 64
