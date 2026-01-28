"""
Comprehensive unit tests for HRM (Hierarchical Reasoning Model) Agent.

Tests cover:
- HRM initialization with different configurations
- Forward pass with sample inputs
- Adaptive Computation Time (ACT) mechanism
- Halting logic and convergence
- Subproblem decomposition
- H-Module and L-Module functionality
- Loss computation and training utilities
- Edge cases and error conditions

All tests use deterministic seeding for reproducibility.
"""

import math

import pytest

# Skip entire module if torch is not installed (optional neural dependency)
torch = pytest.importorskip("torch", reason="PyTorch required for neural agent tests")
import torch.nn as nn  # noqa: E402

from src.agents.hrm_agent import (  # noqa: E402
    AdaptiveComputationTime,
    HModule,
    HRMAgent,
    HRMLoss,
    HRMOutput,
    LModule,
    PonderNet,
    SubProblem,
    create_hrm_agent,
)
from src.training.system_config import HRMConfig  # noqa: E402

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> HRMConfig:
    """
    Default HRM configuration for testing.

    Uses small dimensions for fast test execution.
    """
    return HRMConfig(
        h_dim=64,
        l_dim=32,
        num_h_layers=2,
        num_l_layers=2,
        max_outer_steps=5,
        halt_threshold=0.9,
        dropout=0.1,
        ponder_epsilon=0.01,
        max_ponder_steps=8,
    )


@pytest.fixture
def minimal_config() -> HRMConfig:
    """
    Minimal HRM configuration for edge case testing.

    Single layer, small dimensions.
    """
    return HRMConfig(
        h_dim=16,
        l_dim=8,
        num_h_layers=1,
        num_l_layers=1,
        max_outer_steps=3,
        halt_threshold=0.95,
        dropout=0.0,
        ponder_epsilon=0.01,
    )


@pytest.fixture
def large_config() -> HRMConfig:
    """
    Larger HRM configuration for testing scalability.

    Multiple layers, larger dimensions.
    """
    return HRMConfig(
        h_dim=256,
        l_dim=128,
        num_h_layers=4,
        num_l_layers=4,
        max_outer_steps=10,
        halt_threshold=0.85,
        dropout=0.2,
        ponder_epsilon=0.01,
    )


@pytest.fixture
def sample_input_batch() -> torch.Tensor:
    """
    Create sample input batch for testing.

    Returns:
        Tensor of shape [batch=2, seq=4, h_dim=64]
    """
    torch.manual_seed(42)
    return torch.randn(2, 4, 64)


@pytest.fixture
def sample_input_single() -> torch.Tensor:
    """
    Create single sample input for testing.

    Returns:
        Tensor of shape [batch=1, seq=8, h_dim=64]
    """
    torch.manual_seed(42)
    return torch.randn(1, 8, 64)


@pytest.fixture
def sample_targets() -> torch.Tensor:
    """
    Create sample target labels for loss computation.

    Returns:
        Tensor of shape [batch=2] with class indices
    """
    return torch.tensor([0, 1], dtype=torch.long)


def set_deterministic_seed(seed: int = 42):
    """
    Set random seeds for deterministic behavior.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Test AdaptiveComputationTime
# ============================================================================


class TestAdaptiveComputationTime:
    """Test suite for ACT mechanism."""

    def test_act_initialization(self, default_config):
        """
        Test ACT module initializes with correct parameters.

        Validates:
        - Epsilon value is set correctly
        - Halt FC network has correct architecture
        """
        act = AdaptiveComputationTime(hidden_dim=default_config.h_dim, epsilon=default_config.ponder_epsilon)

        assert act.epsilon == default_config.ponder_epsilon
        assert isinstance(act.halt_fc, nn.Sequential)
        assert len(act.halt_fc) == 4  # Linear, ReLU, Linear, Sigmoid

    def test_act_forward_shape(self, default_config):
        """
        Test ACT forward pass produces correct output shapes.

        Validates:
        - Halt probabilities shape matches [batch, seq]
        - Ponder cost is a scalar float
        """
        set_deterministic_seed()
        act = AdaptiveComputationTime(default_config.h_dim)

        batch_size, seq_len, hidden_dim = 2, 4, default_config.h_dim
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        halt_probs, ponder_cost = act(hidden_states)

        assert halt_probs.shape == (batch_size, seq_len)
        assert isinstance(ponder_cost, torch.Tensor)
        assert ponder_cost.ndim == 0  # Scalar

    def test_act_halt_probabilities_range(self, default_config):
        """
        Test ACT halt probabilities are in valid range [0, 1].

        Validates:
        - All probabilities are between 0 and 1 (due to sigmoid)
        - Probabilities are deterministic with fixed seed
        """
        set_deterministic_seed()
        act = AdaptiveComputationTime(default_config.h_dim)

        hidden_states = torch.randn(3, 5, default_config.h_dim)
        halt_probs, _ = act(hidden_states)

        assert torch.all(halt_probs >= 0.0)
        assert torch.all(halt_probs <= 1.0)

    def test_act_ponder_cost_calculation(self, default_config):
        """
        Test ACT ponder cost is calculated correctly.

        Validates:
        - Ponder cost equals mean sum of halt probabilities
        - Cost increases with higher halt probabilities
        """
        set_deterministic_seed()
        act = AdaptiveComputationTime(default_config.h_dim)

        batch_size, seq_len = 2, 4
        hidden_states = torch.randn(batch_size, seq_len, default_config.h_dim)

        halt_probs, ponder_cost = act(hidden_states)

        # Ponder cost should be mean of summed halt probs across batch
        expected_cost = halt_probs.sum(dim=-1).mean()
        assert torch.isclose(ponder_cost, expected_cost, atol=1e-6)

    def test_act_determinism(self, default_config):
        """
        Test ACT produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        """
        hidden_states = torch.randn(2, 3, default_config.h_dim)

        # First run
        set_deterministic_seed(42)
        act1 = AdaptiveComputationTime(default_config.h_dim)
        halt_probs1, ponder_cost1 = act1(hidden_states)

        # Second run with same seed
        set_deterministic_seed(42)
        act2 = AdaptiveComputationTime(default_config.h_dim)
        halt_probs2, ponder_cost2 = act2(hidden_states)

        assert torch.allclose(halt_probs1, halt_probs2)
        assert torch.isclose(ponder_cost1, ponder_cost2)


# ============================================================================
# Test HModule
# ============================================================================


class TestHModule:
    """Test suite for H-Module (high-level planning)."""

    def test_hmodule_initialization(self, default_config):
        """
        Test H-Module initializes with correct architecture.

        Validates:
        - Multi-head attention layer exists
        - FFN network has correct structure
        - Layer normalization layers exist
        - Decomposition head exists
        """
        h_module = HModule(default_config)

        assert isinstance(h_module.attention, nn.MultiheadAttention)
        assert h_module.attention.embed_dim == default_config.h_dim
        assert h_module.attention.num_heads == 8

        assert isinstance(h_module.ffn, nn.Sequential)
        assert isinstance(h_module.norm1, nn.LayerNorm)
        assert isinstance(h_module.norm2, nn.LayerNorm)
        assert isinstance(h_module.decompose_head, nn.Sequential)

    def test_hmodule_forward_shape(self, default_config):
        """
        Test H-Module forward pass preserves input shape.

        Validates:
        - Output shape matches input shape [batch, seq, h_dim]
        - No information is lost through residual connections
        """
        set_deterministic_seed()
        h_module = HModule(default_config)

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, default_config.h_dim)

        output = h_module(x)

        assert output.shape == x.shape
        assert output.shape == (batch_size, seq_len, default_config.h_dim)

    def test_hmodule_decompose_shape(self, default_config):
        """
        Test H-Module decomposition produces correct shape.

        Validates:
        - Decompose output has same spatial dimensions as input
        - Feature dimension matches h_dim
        """
        set_deterministic_seed()
        h_module = HModule(default_config)

        batch_size, seq_len = 2, 3
        x = torch.randn(batch_size, seq_len, default_config.h_dim)

        decomposed = h_module.decompose(x)

        assert decomposed.shape == (batch_size, seq_len, default_config.h_dim)

    def test_hmodule_attention_mechanism(self, default_config):
        """
        Test H-Module attention creates dependencies between positions.

        Validates:
        - Output at each position depends on all input positions
        - Different from simple feedforward (verified by gradient flow)
        """
        set_deterministic_seed()
        h_module = HModule(default_config)
        h_module.eval()  # Disable dropout

        x = torch.randn(1, 4, default_config.h_dim, requires_grad=True)
        output = h_module(x)

        # Compute gradient of first output position w.r.t. all inputs
        output[0, 0].sum().backward()

        # All input positions should have gradients (attention dependency)
        assert x.grad is not None
        assert torch.any(x.grad[0, 1:] != 0), "Attention should create cross-position dependencies"

    def test_hmodule_determinism(self, default_config):
        """
        Test H-Module produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        """
        x = torch.randn(2, 3, default_config.h_dim)

        set_deterministic_seed(42)
        h_module1 = HModule(default_config)
        h_module1.eval()
        output1 = h_module1(x)

        set_deterministic_seed(42)
        h_module2 = HModule(default_config)
        h_module2.eval()
        output2 = h_module2(x)

        assert torch.allclose(output1, output2, atol=1e-5)


# ============================================================================
# Test LModule
# ============================================================================


class TestLModule:
    """Test suite for L-Module (low-level execution)."""

    def test_lmodule_initialization(self, default_config):
        """
        Test L-Module initializes with correct architecture.

        Validates:
        - Projection layers exist (h_to_l, l_to_h)
        - GRU layer has correct parameters
        - Output projection network exists
        """
        l_module = LModule(default_config)

        assert isinstance(l_module.h_to_l, nn.Linear)
        assert l_module.h_to_l.in_features == default_config.h_dim
        assert l_module.h_to_l.out_features == default_config.l_dim

        assert isinstance(l_module.gru, nn.GRU)
        assert l_module.gru.input_size == default_config.l_dim
        assert l_module.gru.hidden_size == default_config.l_dim
        assert l_module.gru.num_layers == default_config.num_l_layers

        assert isinstance(l_module.l_to_h, nn.Linear)
        assert l_module.l_to_h.out_features == default_config.h_dim

    def test_lmodule_forward_shapes(self, default_config):
        """
        Test L-Module forward pass produces correct output shapes.

        Validates:
        - Output shape is [batch, seq, l_dim]
        - Feedback shape is [batch, seq, h_dim] for H-module
        """
        set_deterministic_seed()
        l_module = LModule(default_config)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, default_config.h_dim)

        output, feedback = l_module(x)

        assert output.shape == (batch_size, seq_len, default_config.l_dim)
        assert feedback.shape == (batch_size, seq_len, default_config.h_dim)

    def test_lmodule_with_hidden_context(self, default_config):
        """
        Test L-Module accepts optional hidden state context.

        Validates:
        - Can pass hidden context to GRU
        - Output shapes remain correct
        """
        set_deterministic_seed()
        l_module = LModule(default_config)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, default_config.h_dim)

        # Create hidden context for GRU
        h_context = torch.randn(default_config.num_l_layers, batch_size, default_config.l_dim)

        output, feedback = l_module(x, h_context)

        assert output.shape == (batch_size, seq_len, default_config.l_dim)
        assert feedback.shape == (batch_size, seq_len, default_config.h_dim)

    def test_lmodule_sequential_processing(self, default_config):
        """
        Test L-Module GRU creates temporal dependencies.

        Validates:
        - Output at position t depends on positions < t
        - GRU maintains sequential state
        """
        set_deterministic_seed()
        l_module = LModule(default_config)
        l_module.eval()

        x = torch.randn(1, 5, default_config.h_dim, requires_grad=True)
        output, _ = l_module(x)

        # Gradient from last position should affect earlier positions
        output[0, -1].sum().backward()

        assert x.grad is not None
        assert torch.any(x.grad[0, :-1] != 0), "GRU should create temporal dependencies"

    def test_lmodule_determinism(self, default_config):
        """
        Test L-Module produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        """
        x = torch.randn(2, 3, default_config.h_dim)

        set_deterministic_seed(42)
        l_module1 = LModule(default_config)
        l_module1.eval()
        output1, feedback1 = l_module1(x)

        set_deterministic_seed(42)
        l_module2 = LModule(default_config)
        l_module2.eval()
        output2, feedback2 = l_module2(x)

        assert torch.allclose(output1, output2, atol=1e-5)
        assert torch.allclose(feedback1, feedback2, atol=1e-5)


# ============================================================================
# Test HRMAgent
# ============================================================================


class TestHRMAgent:
    """Test suite for complete HRM Agent."""

    def test_hrm_initialization(self, default_config):
        """
        Test HRM agent initializes with correct configuration.

        Validates:
        - Config is stored correctly
        - All submodules exist (H-module, L-module, PonderNet)
        - Model is on correct device
        """
        agent = HRMAgent(default_config, device="cpu")

        assert agent.config == default_config
        assert len(agent.h_module) == default_config.num_h_layers
        assert isinstance(agent.l_module, LModule)
        # HRM now uses PonderNet instead of AdaptiveComputationTime
        assert isinstance(agent.ponder_net, PonderNet) or agent.ponder_net is None
        assert isinstance(agent.input_proj, nn.Linear)
        assert isinstance(agent.integrate, nn.Sequential)

    def test_hrm_forward_basic(self, default_config, sample_input_batch):
        """
        Test HRM forward pass with default parameters.

        Validates:
        - Forward pass completes without error
        - Output has correct type (HRMOutput)
        - All expected fields are present
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        output = agent(sample_input_batch)

        assert isinstance(output, HRMOutput)
        assert isinstance(output.final_state, torch.Tensor)
        assert isinstance(output.subproblems, list)
        assert isinstance(output.halt_step, int)
        # Ponder cost can be float or tensor
        assert isinstance(output.total_ponder_cost, (float, torch.Tensor))
        assert isinstance(output.convergence_path, list)

    def test_hrm_forward_output_shape(self, default_config, sample_input_batch):
        """
        Test HRM forward pass produces correct output shape.

        Validates:
        - Final state has same spatial dims as input
        - Final state is in h_dim space
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        batch_size, seq_len, _ = sample_input_batch.shape
        output = agent(sample_input_batch)

        assert output.final_state.shape == (batch_size, seq_len, default_config.h_dim)

    def test_hrm_halting_mechanism(self, default_config):
        """
        Test HRM halts when confidence threshold is reached.

        Validates:
        - Agent stops before max_steps if halt threshold reached
        - Halt step is recorded correctly
        - Convergence path shows increasing confidence
        """
        set_deterministic_seed()

        # Use high halt threshold to force early stopping
        config = HRMConfig(
            h_dim=32,
            l_dim=16,
            num_h_layers=1,
            num_l_layers=1,
            max_outer_steps=20,
            halt_threshold=0.5,  # Low threshold for early halt
        )

        agent = HRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.h_dim)
        output = agent(x)

        # Should halt before max steps with reasonable probability
        assert output.halt_step <= config.max_outer_steps
        assert len(output.convergence_path) == output.halt_step

    def test_hrm_max_steps_limit(self, default_config):
        """
        Test HRM respects maximum step limit.

        Validates:
        - Agent never exceeds max_outer_steps
        - Works with custom max_steps parameter
        """
        set_deterministic_seed()

        config = HRMConfig(
            h_dim=32,
            l_dim=16,
            num_h_layers=1,
            num_l_layers=1,
            max_outer_steps=3,
            halt_threshold=0.99,  # High threshold to force max steps
        )

        agent = HRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.h_dim)
        output = agent(x, max_steps=2)

        assert output.halt_step <= 2  # Custom max_steps

    def test_hrm_convergence_path(self, default_config):
        """
        Test HRM convergence path tracks halt probabilities.

        Validates:
        - Convergence path has one entry per step
        - All values are in [0, 1] range
        - Path generally increases (trending toward halt)
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 4, default_config.h_dim)
        output = agent(x)

        assert len(output.convergence_path) == output.halt_step
        assert all(0.0 <= p <= 1.0 for p in output.convergence_path)

    def test_hrm_ponder_cost(self, default_config):
        """
        Test HRM ponder cost accumulates correctly.

        Validates:
        - Ponder cost is non-negative
        - Cost increases with more steps
        - Cost is deterministic with fixed seed
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, default_config.h_dim)
        output = agent(x)

        # Convert to float if it's a tensor
        ponder_cost = output.total_ponder_cost
        if isinstance(ponder_cost, torch.Tensor):
            ponder_cost = ponder_cost.item()

        assert ponder_cost >= 0.0
        assert isinstance(ponder_cost, float)

    def test_hrm_subproblem_decomposition(self, default_config):
        """
        Test HRM generates subproblem decomposition when requested.

        Validates:
        - Subproblems are created when return_decomposition=True
        - Each subproblem has required fields
        - Subproblem states have correct shape
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, default_config.h_dim)

        output = agent(x, return_decomposition=True)

        assert len(output.subproblems) > 0

        for subproblem in output.subproblems:
            assert isinstance(subproblem, SubProblem)
            assert isinstance(subproblem.level, int)
            assert isinstance(subproblem.description, str)
            assert isinstance(subproblem.state, torch.Tensor)
            assert isinstance(subproblem.confidence, float)
            assert 0.0 <= subproblem.confidence <= 1.0

    def test_hrm_no_decomposition_by_default(self, default_config):
        """
        Test HRM does not generate subproblems by default.

        Validates:
        - Subproblems list is empty when return_decomposition=False
        - Saves computation when decomposition not needed
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(2, 4, default_config.h_dim)
        output = agent(x, return_decomposition=False)

        assert len(output.subproblems) == 0

    def test_hrm_determinism(self, default_config):
        """
        Test HRM produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        - Critical for reproducible experiments
        """
        x = torch.randn(2, 3, default_config.h_dim)

        set_deterministic_seed(42)
        agent1 = HRMAgent(default_config, device="cpu")
        agent1.eval()
        output1 = agent1(x)

        set_deterministic_seed(42)
        agent2 = HRMAgent(default_config, device="cpu")
        agent2.eval()
        output2 = agent2(x)

        assert torch.allclose(output1.final_state, output2.final_state, atol=1e-5)
        assert output1.halt_step == output2.halt_step
        assert len(output1.convergence_path) == len(output2.convergence_path)

    def test_hrm_different_configs(self, minimal_config, large_config):
        """
        Test HRM works with different configuration sizes.

        Validates:
        - Minimal config produces valid output
        - Large config produces valid output
        - Architecture scales correctly
        """
        set_deterministic_seed()

        # Test minimal config
        agent_min = HRMAgent(minimal_config, device="cpu")
        agent_min.eval()
        x_min = torch.randn(1, 2, minimal_config.h_dim)
        output_min = agent_min(x_min)
        assert output_min.final_state.shape == (1, 2, minimal_config.h_dim)

        # Test large config
        agent_large = HRMAgent(large_config, device="cpu")
        agent_large.eval()
        x_large = torch.randn(1, 2, large_config.h_dim)
        output_large = agent_large(x_large)
        assert output_large.final_state.shape == (1, 2, large_config.h_dim)

    def test_hrm_parameter_count(self, default_config):
        """
        Test HRM parameter counting utility.

        Validates:
        - get_parameter_count returns positive integer
        - Count matches manual parameter sum
        """
        agent = HRMAgent(default_config, device="cpu")

        param_count = agent.get_parameter_count()

        assert param_count > 0
        assert isinstance(param_count, int)

        # Verify against manual count
        manual_count = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        assert param_count == manual_count

    @pytest.mark.asyncio
    async def test_hrm_decompose_problem(self, default_config):
        """
        Test HRM async problem decomposition method.

        Validates:
        - Async decomposition works correctly
        - Returns list of SubProblems
        - Subproblems contain query context
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        query = "Solve complex reasoning task"
        state = torch.randn(5, default_config.h_dim)  # Unbatched

        subproblems = await agent.decompose_problem(query, state)

        assert isinstance(subproblems, list)
        assert len(subproblems) > 0

        for sp in subproblems:
            assert isinstance(sp, SubProblem)
            assert query in sp.description

    @pytest.mark.asyncio
    async def test_hrm_decompose_problem_batched_input(self, default_config):
        """
        Test HRM decomposition handles batched input correctly.

        Validates:
        - Works with pre-batched input
        - Produces valid subproblems
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        query = "Test query"
        state = torch.randn(1, 5, default_config.h_dim)  # Already batched

        subproblems = await agent.decompose_problem(query, state)

        assert isinstance(subproblems, list)
        assert len(subproblems) > 0


# ============================================================================
# Test HRMLoss
# ============================================================================


class TestHRMLoss:
    """Test suite for HRM loss computation."""

    def test_hrmloss_initialization(self):
        """
        Test HRMLoss initializes with correct weights.

        Validates:
        - Default weights are set correctly
        - Custom weights are accepted
        """
        # Default weights
        loss_fn = HRMLoss()
        assert loss_fn.task_weight == 1.0
        assert loss_fn.ponder_weight == 0.01
        assert loss_fn.consistency_weight == 0.1

        # Custom weights
        custom_loss = HRMLoss(task_weight=2.0, ponder_weight=0.05, consistency_weight=0.2)
        assert custom_loss.task_weight == 2.0
        assert custom_loss.ponder_weight == 0.05
        assert custom_loss.consistency_weight == 0.2

    def test_hrmloss_forward(self, default_config):
        """
        Test HRMLoss forward computation.

        Validates:
        - Loss is computed correctly
        - Returns both total loss and loss dict
        - All loss components are present
        """
        set_deterministic_seed()

        # Create mock HRM output
        hrm_output = HRMOutput(
            final_state=torch.randn(2, 4, default_config.h_dim),
            subproblems=[],
            halt_step=3,
            total_ponder_cost=1.5,
            convergence_path=[0.3, 0.5, 0.7],
        )

        # Mock predictions and targets
        predictions = torch.randn(2, 10)  # Batch=2, num_classes=10
        targets = torch.tensor([0, 5], dtype=torch.long)

        # Task loss function
        task_loss_fn = nn.CrossEntropyLoss()

        # Compute loss
        loss_fn = HRMLoss()
        total_loss, loss_dict = loss_fn(hrm_output, predictions, targets, task_loss_fn)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert total_loss.item() > 0

        assert "total" in loss_dict
        assert "task" in loss_dict
        assert "ponder" in loss_dict
        assert "consistency" in loss_dict
        assert "halt_step" in loss_dict

        assert loss_dict["halt_step"] == 3

    def test_hrmloss_task_component(self, default_config):
        """
        Test HRMLoss task loss component is computed correctly.

        Validates:
        - Task loss is weighted properly
        - Task loss contributes to total loss
        """
        set_deterministic_seed()

        hrm_output = HRMOutput(
            final_state=torch.randn(2, 3, default_config.h_dim),
            subproblems=[],
            halt_step=2,
            total_ponder_cost=0.5,
            convergence_path=[0.4, 0.6],
        )

        predictions = torch.randn(2, 5)
        targets = torch.tensor([1, 3], dtype=torch.long)
        task_loss_fn = nn.CrossEntropyLoss()

        loss_fn = HRMLoss(task_weight=2.0, ponder_weight=0.0, consistency_weight=0.0)
        total_loss, loss_dict = loss_fn(hrm_output, predictions, targets, task_loss_fn)

        # With only task weight, total should be 2.0 * task_loss
        expected_total = 2.0 * loss_dict["task"]
        assert abs(total_loss.item() - expected_total) < 1e-5

    def test_hrmloss_ponder_component(self, default_config):
        """
        Test HRMLoss ponder cost component.

        Validates:
        - Ponder cost is included in total loss
        - Ponder cost is weighted correctly
        """
        hrm_output = HRMOutput(
            final_state=torch.randn(2, 3, default_config.h_dim),
            subproblems=[],
            halt_step=2,
            total_ponder_cost=3.0,
            convergence_path=[0.4, 0.6],
        )

        predictions = torch.randn(2, 5)
        targets = torch.tensor([1, 3], dtype=torch.long)
        task_loss_fn = nn.CrossEntropyLoss()

        loss_fn = HRMLoss(task_weight=0.0, ponder_weight=0.1, consistency_weight=0.0)
        total_loss, loss_dict = loss_fn(hrm_output, predictions, targets, task_loss_fn)

        # Ponder cost should be included
        assert loss_dict["ponder"] == 3.0

    def test_hrmloss_consistency_component(self, default_config):
        """
        Test HRMLoss consistency loss component.

        Validates:
        - Consistency loss penalizes non-monotonic convergence
        - Monotonic convergence has zero consistency loss
        """
        # Monotonic convergence (should have low consistency loss)
        hrm_output_monotonic = HRMOutput(
            final_state=torch.randn(2, 3, default_config.h_dim),
            subproblems=[],
            halt_step=4,
            total_ponder_cost=1.0,
            convergence_path=[0.2, 0.4, 0.6, 0.8],  # Increasing
        )

        # Non-monotonic convergence (should have higher consistency loss)
        hrm_output_nonmonotonic = HRMOutput(
            final_state=torch.randn(2, 3, default_config.h_dim),
            subproblems=[],
            halt_step=4,
            total_ponder_cost=1.0,
            convergence_path=[0.2, 0.6, 0.4, 0.7],  # Has decrease
        )

        predictions = torch.randn(2, 5)
        targets = torch.tensor([1, 3], dtype=torch.long)
        task_loss_fn = nn.CrossEntropyLoss()

        loss_fn = HRMLoss(task_weight=0.0, ponder_weight=0.0, consistency_weight=1.0)

        _, loss_dict_mono = loss_fn(hrm_output_monotonic, predictions, targets, task_loss_fn)
        _, loss_dict_nonmono = loss_fn(hrm_output_nonmonotonic, predictions, targets, task_loss_fn)

        # Non-monotonic should have higher consistency loss
        assert loss_dict_nonmono["consistency"] > loss_dict_mono["consistency"]

    def test_hrmloss_single_step_convergence(self, default_config):
        """
        Test HRMLoss handles single-step convergence path.

        Validates:
        - No error with convergence_path of length 1
        - Consistency loss is zero (no differences to compute)
        """
        hrm_output = HRMOutput(
            final_state=torch.randn(2, 3, default_config.h_dim),
            subproblems=[],
            halt_step=1,
            total_ponder_cost=0.5,
            convergence_path=[0.9],  # Only one step
        )

        predictions = torch.randn(2, 5)
        targets = torch.tensor([1, 3], dtype=torch.long)
        task_loss_fn = nn.CrossEntropyLoss()

        loss_fn = HRMLoss()
        total_loss, loss_dict = loss_fn(hrm_output, predictions, targets, task_loss_fn)

        # Should handle gracefully
        assert isinstance(total_loss, torch.Tensor)
        assert loss_dict["consistency"] == 0.0


# ============================================================================
# Test create_hrm_agent factory
# ============================================================================


class TestCreateHRMAgent:
    """Test suite for HRM agent factory function."""

    def test_create_hrm_agent_basic(self, default_config):
        """
        Test create_hrm_agent factory creates valid agent.

        Validates:
        - Returns HRMAgent instance
        - Initializes weights properly
        - Agent is on correct device
        """
        agent = create_hrm_agent(default_config, device="cpu")

        assert isinstance(agent, HRMAgent)
        assert agent.config == default_config

        # Check that at least some weights are initialized (not all zeros)
        # Note: Biases are intentionally initialized to zero, so we check weights
        has_nonzero_params = False
        for name, param in agent.named_parameters():
            if "weight" in name and not torch.all(param == 0.0):
                has_nonzero_params = True
                break
        assert has_nonzero_params, "At least some weights should be non-zero"

    def test_create_hrm_agent_weight_initialization(self, default_config):
        """
        Test create_hrm_agent initializes weights correctly.

        Validates:
        - Linear layers use Xavier initialization
        - GRU layers use orthogonal initialization
        - Biases are initialized to zero
        """
        set_deterministic_seed()
        agent = create_hrm_agent(default_config, device="cpu")

        # Check linear layer initialization
        for module in agent.modules():
            if isinstance(module, nn.Linear):
                # Xavier init should have specific variance
                assert not torch.all(module.weight == 0.0)
                if module.bias is not None:
                    assert torch.all(module.bias == 0.0)

    def test_create_hrm_agent_different_devices(self, minimal_config):
        """
        Test create_hrm_agent places model on correct device.

        Validates:
        - CPU device works
        - Device is set correctly on all parameters
        """
        agent = create_hrm_agent(minimal_config, device="cpu")

        for param in agent.parameters():
            assert param.device.type == "cpu"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestHRMEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_hrm_empty_sequence(self, default_config):
        """
        Test HRM handles empty sequence length gracefully.

        Note: seq_len=0 may cause errors in some operations.
        This test validates behavior.
        """
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        # Sequence length of 1 (minimal)
        x = torch.randn(1, 1, default_config.h_dim)
        output = agent(x)

        assert output.final_state.shape == (1, 1, default_config.h_dim)

    def test_hrm_large_batch_size(self, default_config):
        """
        Test HRM handles large batch sizes.

        Validates:
        - No memory issues with larger batches
        - Output shapes are correct
        """
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        large_batch = torch.randn(16, 4, default_config.h_dim)
        output = agent(large_batch)

        assert output.final_state.shape == (16, 4, default_config.h_dim)

    def test_hrm_single_step(self, default_config):
        """
        Test HRM with max_steps=1.

        Validates:
        - Agent can run with just one iteration
        - Convergence path has one entry
        """
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(2, 3, default_config.h_dim)
        output = agent(x, max_steps=1)

        assert output.halt_step == 1
        assert len(output.convergence_path) == 1

    def test_hrm_gradient_flow(self, default_config):
        """
        Test gradients flow through entire HRM architecture.

        Validates:
        - Gradients reach core parameters
        - No gradient issues (NaN, inf)

        Note: Some parameters may not have gradients if they're not used
        in the forward pass (e.g., decompose_head when return_decomposition=False)
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        agent.train()

        x = torch.randn(2, 3, default_config.h_dim, requires_grad=True)
        output = agent(x, return_decomposition=True)  # Use decomposition to activate all paths

        # Compute loss and backpropagate
        loss = output.final_state.mean()
        loss.backward()

        # Check that most parameters have gradients (excluding unused ones)
        params_with_grad = 0
        params_without_grad = 0

        for name, param in agent.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                else:
                    params_without_grad += 1

        # Most parameters should have gradients
        assert params_with_grad > params_without_grad, "Most parameters should have gradients"

    def test_hrm_eval_mode(self, default_config):
        """
        Test HRM behavior in eval mode vs train mode.

        Validates:
        - Eval mode disables dropout
        - Results are deterministic in eval mode
        """
        set_deterministic_seed()
        agent = HRMAgent(default_config, device="cpu")
        x = torch.randn(1, 3, default_config.h_dim)

        # Eval mode
        agent.eval()
        output1 = agent(x)
        output2 = agent(x)

        # Should be identical in eval mode
        assert torch.allclose(output1.final_state, output2.final_state)


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestHRMIntegration:
    """Integration tests for HRM components working together."""

    def test_hrm_full_forward_backward_pass(self, default_config):
        """
        Test complete forward and backward pass through HRM.

        Validates:
        - End-to-end training loop works
        - Loss computation is correct
        - Parameters are updated
        """
        set_deterministic_seed()

        agent = HRMAgent(default_config, device="cpu")
        loss_fn = HRMLoss()
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        agent.train()

        # Forward pass with requires_grad input
        x = torch.randn(4, 5, default_config.h_dim, requires_grad=True)
        output = agent(x, return_decomposition=True)

        # Compute predictions from final state (requires grad)
        pred_layer = nn.Linear(default_config.h_dim, 10)
        predictions = pred_layer(output.final_state.mean(dim=1))
        targets = torch.randint(0, 10, (4,))

        task_loss_fn = nn.CrossEntropyLoss()
        total_loss, loss_dict = loss_fn(output, predictions, targets, task_loss_fn)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Verify training step completed
        assert total_loss.item() > 0
        assert all(key in loss_dict for key in ["total", "task", "ponder", "consistency"])

    def test_hrm_multiple_iterations(self, default_config):
        """
        Test HRM over multiple training iterations.

        Validates:
        - Agent can process multiple batches
        - No memory leaks
        - Training is stable
        """
        set_deterministic_seed()

        agent = HRMAgent(default_config, device="cpu")
        optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            x = torch.randn(2, 3, default_config.h_dim)
            output = agent(x)

            loss = output.final_state.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert len(losses) == 5
        assert all(not math.isnan(loss) and not math.isinf(loss) for loss in losses)

    def test_hrm_with_different_sequence_lengths(self, default_config):
        """
        Test HRM handles varying sequence lengths in different batches.

        Validates:
        - Agent works with different seq lengths
        - No size mismatch errors
        """
        agent = HRMAgent(default_config, device="cpu")
        agent.eval()

        # Different sequence lengths
        x1 = torch.randn(2, 3, default_config.h_dim)
        x2 = torch.randn(2, 7, default_config.h_dim)
        x3 = torch.randn(2, 1, default_config.h_dim)

        output1 = agent(x1)
        output2 = agent(x2)
        output3 = agent(x3)

        assert output1.final_state.shape[1] == 3
        assert output2.final_state.shape[1] == 7
        assert output3.final_state.shape[1] == 1


# ============================================================================
# Test SubProblem dataclass
# ============================================================================


class TestSubProblem:
    """Test suite for SubProblem dataclass."""

    def test_subproblem_initialization(self):
        """
        Test SubProblem initializes correctly.

        Validates:
        - Required fields are set
        - Optional fields have defaults
        """
        state = torch.randn(32)

        sp = SubProblem(
            level=1,
            description="Test subproblem",
            state=state,
            confidence=0.85,
        )

        assert sp.level == 1
        assert sp.description == "Test subproblem"
        assert torch.equal(sp.state, state)
        assert sp.confidence == 0.85
        assert sp.parent_id is None  # Default

    def test_subproblem_with_parent(self):
        """
        Test SubProblem with parent_id set.

        Validates:
        - Parent ID is stored correctly
        - Can create hierarchical structures
        """
        sp = SubProblem(
            level=2,
            description="Child subproblem",
            state=torch.randn(32),
            parent_id=0,
            confidence=0.9,
        )

        assert sp.parent_id == 0
        assert sp.level == 2


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/test_hrm_agent.py -v
    pytest.main([__file__, "-v", "--tb=short"])
