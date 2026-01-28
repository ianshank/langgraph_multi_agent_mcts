"""Tests for neural network trainer."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")
from torch.utils.data import DataLoader

from src.models.policy_network import PolicyLoss, PolicyNetwork
from src.models.value_network import ValueLoss, ValueNetwork
from src.training.neural_trainer import (
    NeuralTrainer,
    PolicyDataset,
    TrainingConfig,
    ValueDataset,
    train_policy_network,
    train_value_network,
)


class TestPolicyDataset:
    """Test PolicyDataset class."""

    def test_initialization(self):
        """Test dataset creation."""
        states = torch.randn(100, 10)
        actions = torch.randint(0, 5, (100,))
        values = torch.randn(100)

        dataset = PolicyDataset(states, actions, values)

        assert len(dataset) == 100

    def test_getitem(self):
        """Test item retrieval."""
        states = torch.randn(10, 10)
        actions = torch.randint(0, 5, (10,))
        values = torch.randn(10)

        dataset = PolicyDataset(states, actions, values)
        state, action, value = dataset[0]

        assert state.shape == (10,)
        assert isinstance(action.item(), int)
        assert isinstance(value.item(), float)


class TestValueDataset:
    """Test ValueDataset class."""

    def test_initialization(self):
        """Test dataset creation."""
        states = torch.randn(100, 10)
        values = torch.randn(100)

        dataset = ValueDataset(states, values)

        assert len(dataset) == 100

    def test_getitem(self):
        """Test item retrieval."""
        states = torch.randn(10, 10)
        values = torch.randn(10)

        dataset = ValueDataset(states, values)
        state, value = dataset[0]

        assert state.shape == (10,)
        assert isinstance(value.item(), float)


class TestNeuralTrainer:
    """Test NeuralTrainer class."""

    @pytest.fixture
    def policy_net(self):
        """Create test policy network."""
        return PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])

    @pytest.fixture
    def value_net(self):
        """Create test value network."""
        return ValueNetwork(state_dim=10, hidden_dims=[32, 16])

    @pytest.fixture
    def config(self, tmp_path):
        """Create test config."""
        return TrainingConfig(
            learning_rate=0.001,
            batch_size=4,
            num_epochs=2,
            checkpoint_dir=str(tmp_path),
            save_every=1,
            early_stopping_patience=5,
        )

    @pytest.fixture
    def policy_data(self):
        """Create test policy data."""
        states = torch.randn(20, 10)
        actions = torch.randint(0, 5, (20,))
        values = torch.randn(20)
        return PolicyDataset(states, actions, values)

    @pytest.fixture
    def value_data(self):
        """Create test value data."""
        states = torch.randn(20, 10)
        values = torch.randn(20)
        return ValueDataset(states, values)

    def test_trainer_initialization(self, policy_net, config):
        """Test trainer initialization."""
        loss_fn = PolicyLoss()
        trainer = NeuralTrainer(policy_net, loss_fn, config)

        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")

    def test_policy_training(self, policy_net, config, policy_data):
        """Test policy network training."""
        loss_fn = PolicyLoss()
        trainer = NeuralTrainer(policy_net, loss_fn, config, model_name="test_policy")

        train_loader = DataLoader(policy_data, batch_size=4, shuffle=True)

        # Train for one epoch
        train_loss, metrics = trainer.train_epoch(train_loader)

        assert train_loss > 0
        assert "total" in metrics

    def test_value_training(self, value_net, config, value_data):
        """Test value network training."""
        loss_fn = ValueLoss()
        trainer = NeuralTrainer(value_net, loss_fn, config, model_name="test_value")

        train_loader = DataLoader(value_data, batch_size=4, shuffle=True)

        # Train for one epoch
        train_loss, metrics = trainer.train_epoch(train_loader)

        assert train_loss > 0
        assert "total" in metrics

    def test_validation(self, policy_net, config, policy_data):
        """Test validation."""
        loss_fn = PolicyLoss()
        trainer = NeuralTrainer(policy_net, loss_fn, config)

        val_loader = DataLoader(policy_data, batch_size=4, shuffle=False)

        val_loss, metrics = trainer.validate(val_loader)

        assert val_loss > 0

    def test_full_training(self, policy_net, config, policy_data):
        """Test full training loop."""
        loss_fn = PolicyLoss()
        trainer = NeuralTrainer(policy_net, loss_fn, config)

        train_loader = DataLoader(policy_data, batch_size=4, shuffle=True)
        val_loader = DataLoader(policy_data, batch_size=4, shuffle=False)

        history = trainer.train(train_loader, val_loader)

        assert len(history) == config.num_epochs
        assert all(m.train_loss > 0 for m in history)

    def test_checkpoint_save_load(self, policy_net, config, policy_data, tmp_path):
        """Test checkpoint saving and loading."""
        loss_fn = PolicyLoss()
        trainer = NeuralTrainer(policy_net, loss_fn, config)

        # Train for one epoch
        train_loader = DataLoader(policy_data, batch_size=4, shuffle=True)
        trainer.train_epoch(train_loader)
        trainer.current_epoch = 1

        # Save checkpoint
        checkpoint_name = "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_name)

        # Create new trainer
        new_policy_net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])
        new_trainer = NeuralTrainer(new_policy_net, loss_fn, config)

        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_name)

        assert new_trainer.current_epoch == 1

    def test_early_stopping(self, value_net, tmp_path):
        """Test early stopping."""
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=4,
            num_epochs=100,
            checkpoint_dir=str(tmp_path),
            early_stopping_patience=3,
            min_delta=0.0001,
        )

        # Create data where training won't improve
        states = torch.randn(20, 10)
        values = torch.randn(20)
        dataset = ValueDataset(states, values)

        loss_fn = ValueLoss()
        trainer = NeuralTrainer(value_net, loss_fn, config)

        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # Should stop early due to no improvement
        history = trainer.train(train_loader, val_loader)

        # Should stop before 100 epochs
        assert len(history) < 100


def test_train_policy_network_convenience():
    """Test convenience function for policy training."""
    # Create small dataset
    states = torch.randn(20, 10)
    actions = torch.randint(0, 5, (20,))
    values = torch.randn(20)

    train_dataset = PolicyDataset(states, actions, values)
    val_dataset = PolicyDataset(states[:5], actions[:5], values[:5])

    policy_net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])

    config = TrainingConfig(num_epochs=2, batch_size=4, log_every=100)

    trainer = train_policy_network(policy_net, train_dataset, val_dataset, config)

    assert len(trainer.training_history) == 2


def test_train_value_network_convenience():
    """Test convenience function for value training."""
    # Create small dataset
    states = torch.randn(20, 10)
    values = torch.randn(20)

    train_dataset = ValueDataset(states, values)
    val_dataset = ValueDataset(states[:5], values[:5])

    value_net = ValueNetwork(state_dim=10, hidden_dims=[16])

    config = TrainingConfig(num_epochs=2, batch_size=4, log_every=100)

    trainer = train_value_network(value_net, train_dataset, val_dataset, config)

    assert len(trainer.training_history) == 2
