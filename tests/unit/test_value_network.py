"""
Tests for value network module.

Tests ValueOutput, ValueNetwork forward pass, evaluation,
uncertainty estimation, and different output activations.
"""

import pytest
import torch

from src.models.value_network import ValueNetwork, ValueOutput


@pytest.mark.unit
class TestValueOutput:
    """Tests for ValueOutput dataclass."""

    def test_basic(self):
        vo = ValueOutput(value=torch.randn(2, 1))
        assert vo.features is None
        assert vo.uncertainty is None

    def test_with_all_fields(self):
        vo = ValueOutput(
            value=torch.randn(2, 1),
            features=torch.randn(2, 64),
            uncertainty=torch.randn(2, 1),
        )
        assert vo.features is not None
        assert vo.uncertainty is not None


@pytest.mark.unit
class TestValueNetwork:
    """Tests for ValueNetwork."""

    def test_init(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16])
        assert net.state_dim == 10

    def test_default_hidden_dims(self):
        net = ValueNetwork(state_dim=10)
        assert net.hidden_dims == [512, 256, 128]

    def test_forward_tanh(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16], output_activation="tanh")
        x = torch.randn(4, 10)
        output = net(x)
        assert output.value.shape == (4, 1)
        assert (output.value >= -1.0).all()
        assert (output.value <= 1.0).all()

    def test_forward_sigmoid(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16], output_activation="sigmoid")
        x = torch.randn(4, 10)
        output = net(x)
        assert (output.value >= 0.0).all()
        assert (output.value <= 1.0).all()

    def test_forward_no_activation(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16], output_activation=None)
        x = torch.randn(4, 10)
        output = net(x)
        assert output.value.shape == (4, 1)

    def test_forward_return_features(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16])
        x = torch.randn(4, 10)
        output = net(x, return_features=True)
        assert output.features is not None
        assert output.features.shape[0] == 4

    def test_forward_no_features_by_default(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16])
        x = torch.randn(4, 10)
        output = net(x)
        assert output.features is None

    def test_uncertainty_estimation(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16], estimate_uncertainty=True)
        x = torch.randn(4, 10)
        output = net(x)
        assert output.uncertainty is not None
        assert output.uncertainty.shape == (4, 1)
        assert (output.uncertainty >= 0).all()  # Softplus ensures positive

    def test_no_uncertainty_by_default(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[32, 16])
        x = torch.randn(4, 10)
        output = net(x)
        assert output.uncertainty is None

    def test_no_batch_norm(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[16], use_batch_norm=False)
        x = torch.randn(4, 10)
        output = net(x)
        assert output.value.shape == (4, 1)

    def test_evaluate_single_state(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[16])
        state = torch.randn(10)
        value = net.evaluate(state)
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0

    def test_evaluate_batched_state(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[16])
        state = torch.randn(1, 10)
        value = net.evaluate(state)
        assert isinstance(value, float)

    def test_evaluate_batch(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[16])
        states = torch.randn(8, 10)
        values = net.evaluate_batch(states)
        assert values.shape == (8,)

    def test_gradient_flow(self):
        net = ValueNetwork(state_dim=10, hidden_dims=[16])
        x = torch.randn(4, 10, requires_grad=True)
        output = net(x)
        loss = output.value.sum()
        loss.backward()
        assert x.grad is not None
