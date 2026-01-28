"""Tests for value network implementation."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")

from src.models.value_network import (
    EnsembleValueNetwork,
    TemporalDifferenceLoss,
    ValueLoss,
    ValueNetwork,
    ValueOutput,
    create_value_network,
)


class TestValueNetwork:
    """Test ValueNetwork class."""

    @pytest.fixture
    def value_net(self):
        """Create test value network."""
        return ValueNetwork(state_dim=10, hidden_dims=[64, 32])

    @pytest.fixture
    def batch_states(self):
        """Create batch of test states."""
        return torch.randn(4, 10)

    def test_initialization(self, value_net):
        """Test network initialization."""
        assert value_net.state_dim == 10
        assert value_net.hidden_dims == [64, 32]
        assert value_net.get_parameter_count() > 0

    def test_forward_pass(self, value_net, batch_states):
        """Test forward pass."""
        output = value_net(batch_states)

        assert isinstance(output, ValueOutput)
        assert output.value.shape == (4, 1)

        # Test output activation
        if value_net.output_activation == "tanh":
            assert (output.value >= -1).all() and (output.value <= 1).all()
        elif value_net.output_activation == "sigmoid":
            assert (output.value >= 0).all() and (output.value <= 1).all()

    def test_evaluate_single_state(self, value_net):
        """Test evaluating single state."""
        state = torch.randn(10)
        value = value_net.evaluate(state)

        assert isinstance(value, float)

    def test_evaluate_batch(self, value_net, batch_states):
        """Test batch evaluation."""
        values = value_net.evaluate_batch(batch_states)

        assert values.shape == (4,)

    def test_get_confidence(self, value_net):
        """Test confidence estimation."""
        state = torch.randn(10)
        confidence = value_net.get_confidence(state)

        assert 0.0 <= confidence <= 1.0

    def test_uncertainty_estimation(self):
        """Test uncertainty head."""
        net = ValueNetwork(state_dim=10, estimate_uncertainty=True)
        state = torch.randn(4, 10)
        output = net(state)

        assert output.uncertainty is not None
        assert output.uncertainty.shape == (4, 1)
        assert (output.uncertainty > 0).all()  # Uncertainty should be positive


class TestValueLoss:
    """Test ValueLoss class."""

    @pytest.fixture
    def value_net(self):
        """Create test value network."""
        return ValueNetwork(state_dim=10, hidden_dims=[32, 16])

    def test_mse_loss(self, value_net):
        """Test MSE loss."""
        loss_fn = ValueLoss(loss_type="mse")

        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)

        loss, loss_dict = loss_fn(predictions, targets, model=value_net)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "mse" in loss_dict
        assert "total" in loss_dict

    def test_huber_loss(self, value_net):
        """Test Huber loss."""
        loss_fn = ValueLoss(loss_type="huber")

        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)

        loss, loss_dict = loss_fn(predictions, targets, model=value_net)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "huber" in loss_dict

    def test_uncertainty_loss(self, value_net):
        """Test loss with uncertainty."""
        loss_fn = ValueLoss(uncertainty_weight=0.1)

        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        uncertainty = torch.ones(4, 1) * 0.5

        loss, loss_dict = loss_fn(predictions, targets, uncertainty=uncertainty, model=value_net)

        assert "uncertainty" in loss_dict


class TestTemporalDifferenceLoss:
    """Test TD loss."""

    def test_td_loss(self):
        """Test temporal difference loss."""
        loss_fn = TemporalDifferenceLoss(gamma=0.99)

        values = torch.randn(4)
        rewards = torch.randn(4)
        next_values = torch.randn(4)
        dones = torch.tensor([0, 0, 0, 1], dtype=torch.bool)

        loss, loss_dict = loss_fn(values, rewards, next_values, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "td_loss" in loss_dict
        assert "mean_td_error" in loss_dict


class TestEnsembleValueNetwork:
    """Test ensemble value network."""

    def test_ensemble_prediction(self):
        """Test ensemble forward pass."""
        ensemble = EnsembleValueNetwork(state_dim=10, num_networks=3)
        state = torch.randn(4, 10)

        output = ensemble(state)

        assert output.value.shape == (4, 1)
        assert output.uncertainty is not None
        assert output.uncertainty.shape == (4, 1)

    def test_ensemble_evaluate(self):
        """Test ensemble evaluation."""
        ensemble = EnsembleValueNetwork(state_dim=10, num_networks=3)
        state = torch.randn(10)

        value, uncertainty = ensemble.evaluate(state)

        assert isinstance(value, float)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0


def test_create_value_network():
    """Test factory function."""
    config = {"hidden_dims": [128, 64], "dropout": 0.2, "output_activation": "sigmoid"}

    net = create_value_network(state_dim=20, config=config, device="cpu")

    assert net.state_dim == 20
    assert net.hidden_dims == [128, 64]
    assert net.dropout == 0.2
    assert net.output_activation == "sigmoid"
