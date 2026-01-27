"""Tests for policy network implementation."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch required")

from src.models.policy_network import (
    ActionSelection,
    PolicyLoss,
    PolicyNetwork,
    PolicyOutput,
    create_policy_network,
)


class TestPolicyNetwork:
    """Test PolicyNetwork class."""

    @pytest.fixture
    def policy_net(self):
        """Create test policy network."""
        return PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])

    @pytest.fixture
    def batch_states(self):
        """Create batch of test states."""
        return torch.randn(4, 10)

    def test_initialization(self, policy_net):
        """Test network initialization."""
        assert policy_net.state_dim == 10
        assert policy_net.action_dim == 5
        assert policy_net.hidden_dims == [32, 16]
        assert policy_net.get_parameter_count() > 0

    def test_forward_pass(self, policy_net, batch_states):
        """Test forward pass."""
        output = policy_net(batch_states)

        assert isinstance(output, PolicyOutput)
        assert output.policy_logits.shape == (4, 5)
        assert output.state_value.shape == (4, 1)

    def test_forward_with_probs(self, policy_net, batch_states):
        """Test forward pass with probability computation."""
        output = policy_net(batch_states, return_probs=True)

        assert output.action_probs is not None
        assert output.action_probs.shape == (4, 5)
        assert torch.allclose(output.action_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert output.entropy is not None
        assert output.entropy.shape == (4,)

    def test_select_action(self, policy_net):
        """Test action selection."""
        state = torch.randn(10)
        selection = policy_net.select_action(state)

        assert isinstance(selection, ActionSelection)
        assert 0 <= selection.action < 5
        assert selection.log_prob <= 0.0
        assert 0.0 <= selection.confidence <= 1.0
        assert selection.entropy >= 0.0

    def test_select_action_deterministic(self, policy_net):
        """Test deterministic action selection."""
        state = torch.randn(10)
        selection1 = policy_net.select_action(state, deterministic=True)
        selection2 = policy_net.select_action(state, deterministic=True)

        assert selection1.action == selection2.action

    def test_select_action_temperature(self, policy_net):
        """Test temperature effect on action selection."""
        state = torch.randn(10)

        # High temperature = more exploration
        high_temp_entropies = []
        for _ in range(10):
            selection = policy_net.select_action(state, temperature=2.0)
            high_temp_entropies.append(selection.entropy)

        # Low temperature = more exploitation
        low_temp_entropies = []
        for _ in range(10):
            selection = policy_net.select_action(state, temperature=0.5)
            low_temp_entropies.append(selection.entropy)

        # Higher temperature should lead to higher entropy on average
        assert sum(high_temp_entropies) / len(high_temp_entropies) > sum(low_temp_entropies) / len(low_temp_entropies)

    def test_select_action_top_k(self, policy_net):
        """Test top-k action filtering."""
        state = torch.randn(10)
        selection = policy_net.select_action(state, top_k=3)

        # Action should be from top-3
        assert 0 <= selection.action < 5

    def test_get_action_probs(self, policy_net):
        """Test getting action probabilities."""
        state = torch.randn(10)
        probs = policy_net.get_action_probs(state)

        assert probs.shape == (1, 5)
        assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-5)
        assert (probs >= 0).all()

    def test_evaluate_actions(self, policy_net, batch_states):
        """Test action evaluation for training."""
        actions = torch.randint(0, 5, (4,))
        log_probs, entropy = policy_net.evaluate_actions(batch_states, actions)

        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)
        assert (log_probs <= 0).all()
        assert (entropy >= 0).all()


class TestPolicyLoss:
    """Test PolicyLoss class."""

    @pytest.fixture
    def loss_fn(self):
        """Create test loss function."""
        return PolicyLoss(policy_weight=1.0, value_weight=0.5, entropy_weight=0.01, l2_weight=0.0001)

    @pytest.fixture
    def policy_net(self):
        """Create test policy network."""
        return PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])

    @pytest.fixture
    def policy_output(self, policy_net):
        """Create test policy output."""
        states = torch.randn(4, 10)
        return policy_net(states, return_probs=True)

    def test_supervised_loss_with_actions(self, loss_fn, policy_output, policy_net):
        """Test supervised learning with action labels."""
        target_actions = torch.randint(0, 5, (4,))
        target_values = torch.randn(4)

        loss, loss_dict = loss_fn(
            policy_output=policy_output,
            target_actions=target_actions,
            target_values=target_values,
            model=policy_net,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "policy_ce" in loss_dict
        assert "value_mse" in loss_dict
        assert "entropy" in loss_dict
        assert "l2" in loss_dict
        assert "total" in loss_dict

    def test_supervised_loss_with_distribution(self, loss_fn, policy_output, policy_net):
        """Test supervised learning with target distribution."""
        target_policy = torch.softmax(torch.randn(4, 5), dim=-1)
        target_values = torch.randn(4)

        loss, loss_dict = loss_fn(
            policy_output=policy_output, target_policy=target_policy, target_values=target_values, model=policy_net
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert "policy_supervised" in loss_dict

    def test_policy_gradient_loss(self, loss_fn, policy_output, policy_net):
        """Test policy gradient loss."""
        target_actions = torch.randint(0, 5, (4,))
        advantages = torch.randn(4)
        target_values = torch.randn(4)

        loss, loss_dict = loss_fn(
            policy_output=policy_output,
            target_actions=target_actions,
            advantages=advantages,
            target_values=target_values,
            model=policy_net,
        )

        assert isinstance(loss, torch.Tensor)
        assert "policy_gradient" in loss_dict


def test_create_policy_network():
    """Test factory function."""
    config = {"hidden_dims": [64, 32], "dropout": 0.2, "activation": "gelu"}

    net = create_policy_network(state_dim=20, action_dim=10, config=config, device="cpu")

    assert net.state_dim == 20
    assert net.action_dim == 10
    assert net.hidden_dims == [64, 32]
    assert net.dropout == 0.2


def test_backward_pass():
    """Test that gradients flow correctly."""
    net = PolicyNetwork(state_dim=10, action_dim=5, hidden_dims=[32, 16])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    state = torch.randn(4, 10)
    actions = torch.randint(0, 5, (4,))
    values = torch.randn(4)

    # Forward pass
    output = net(state, return_probs=True)

    # Compute loss
    loss_fn = PolicyLoss()
    loss, _ = loss_fn(policy_output=output, target_actions=actions, target_values=values, model=net)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    assert any(p.grad is not None for p in net.parameters())


def test_batch_independence():
    """Test that predictions are independent across batch dimension."""
    net = PolicyNetwork(state_dim=10, action_dim=5)
    net.eval()  # Set to eval mode for batch norm

    # Create two different states
    state1 = torch.randn(1, 10)
    state2 = torch.randn(1, 10)

    # Get individual predictions
    with torch.no_grad():
        output1 = net(state1)
        output2 = net(state2)

        # Get batch prediction
        batch_states = torch.cat([state1, state2], dim=0)
        batch_output = net(batch_states)

        # Check that batch predictions match individual predictions
        assert torch.allclose(batch_output.policy_logits[0], output1.policy_logits[0], atol=1e-5)
        assert torch.allclose(batch_output.policy_logits[1], output2.policy_logits[0], atol=1e-5)
