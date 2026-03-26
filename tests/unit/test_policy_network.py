"""
Tests for policy network module.

Tests PolicyOutput, ActionSelection, PolicyNetwork forward pass,
action selection, and helper methods.
"""

import pytest
import torch

from src.models.policy_network import ActionSelection, PolicyNetwork, PolicyOutput


@pytest.mark.unit
class TestPolicyOutput:
    """Tests for PolicyOutput dataclass."""

    def test_basic(self):
        po = PolicyOutput(
            policy_logits=torch.randn(2, 5),
            state_value=torch.randn(2, 1),
        )
        assert po.action_probs is None
        assert po.entropy is None

    def test_with_probs(self):
        po = PolicyOutput(
            policy_logits=torch.randn(2, 5),
            state_value=torch.randn(2, 1),
            action_probs=torch.softmax(torch.randn(2, 5), dim=-1),
            entropy=torch.tensor([1.5, 1.2]),
        )
        assert po.action_probs is not None
        assert po.entropy is not None


@pytest.mark.unit
class TestActionSelection:
    """Tests for ActionSelection dataclass."""

    def test_basic(self):
        sel = ActionSelection(action=2, log_prob=-0.5, confidence=0.8, entropy=1.2)
        assert sel.action == 2
        assert sel.log_prob == -0.5
        assert sel.confidence == 0.8
        assert sel.entropy == 1.2


@pytest.mark.unit
class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_init(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])
        assert net.state_dim == 10
        assert net.action_dim == 5

    def test_init_default_hidden_dims(self):
        net = PolicyNetwork(state_dim=10, action_dim=5)
        assert net.hidden_dims == [256, 256, 128]

    def test_init_gelu_activation(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16], activation="gelu")
        x = torch.randn(4, 10)
        output = net(x)
        assert output.policy_logits.shape == (4, 5)

    def test_init_tanh_activation(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16], activation="tanh")
        x = torch.randn(4, 10)
        output = net(x)
        assert output.policy_logits.shape == (4, 5)

    def test_init_unknown_activation(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            PolicyNetwork(state_dim=10, action_dim=5, activation="swish")

    def test_init_no_batch_norm(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16], use_batch_norm=False)
        x = torch.randn(4, 10)
        output = net(x)
        assert output.policy_logits.shape == (4, 5)

    def test_forward(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])
        x = torch.randn(4, 10)
        output = net(x)
        assert output.policy_logits.shape == (4, 5)
        assert output.state_value.shape == (4, 1)
        assert output.action_probs is None

    def test_forward_with_probs(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])
        x = torch.randn(4, 10)
        output = net(x, return_probs=True)
        assert output.action_probs is not None
        assert output.entropy is not None
        # Probs should sum to 1
        sums = output.action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_select_action_deterministic(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(10)
        sel = net.select_action(state, deterministic=True)
        assert isinstance(sel, ActionSelection)
        assert 0 <= sel.action < 5
        assert sel.confidence > 0
        assert sel.entropy >= 0

    def test_select_action_stochastic(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(10)
        sel = net.select_action(state, deterministic=False)
        assert 0 <= sel.action < 5

    def test_select_action_with_batch_dim(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(1, 10)
        sel = net.select_action(state, deterministic=True)
        assert 0 <= sel.action < 5

    def test_select_action_top_k(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(10)
        sel = net.select_action(state, top_k=2)
        assert 0 <= sel.action < 5

    def test_select_action_temperature(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(10)
        # Low temperature should be more deterministic
        sel_low = net.select_action(state, temperature=0.01, deterministic=True)
        sel_high = net.select_action(state, temperature=100.0, deterministic=True)
        # Both should be valid actions
        assert 0 <= sel_low.action < 5
        assert 0 <= sel_high.action < 5
        # Low temp should have higher confidence
        assert sel_low.confidence >= sel_high.confidence

    def test_get_action_probs(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        state = torch.randn(1, 10)
        probs = net.get_action_probs(state)
        assert probs.shape == (1, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_gradient_flow(self):
        net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[16])
        x = torch.randn(4, 10, requires_grad=True)
        output = net(x)
        loss = output.policy_logits.sum() + output.state_value.sum()
        loss.backward()
        assert x.grad is not None
