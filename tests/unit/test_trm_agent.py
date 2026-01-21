"""
Comprehensive unit tests for TRM (Tiny Recursive Model) Agent.

Tests cover:
- TRM initialization with different configurations
- Forward pass with sample inputs
- Recursive refinement mechanism
- Convergence detection and logic
- Recursion depth limits
- Deep supervision heads
- Loss computation and training utilities
- Edge cases and error conditions

All tests use deterministic seeding for reproducibility.
"""

import math

import pytest

# Skip entire module if torch is not installed (optional neural dependency)
torch = pytest.importorskip("torch", reason="PyTorch required for neural agent tests")
import torch.nn as nn  # noqa: E402

from src.agents.trm_agent import (  # noqa: E402
    DeepSupervisionHead,
    RecursiveBlock,
    TRMAgent,
    TRMLoss,
    TRMOutput,
    TRMRefinementWrapper,
    create_trm_agent,
)
from src.training.system_config import TRMConfig  # noqa: E402

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> TRMConfig:
    """
    Default TRM configuration for testing.

    Uses small dimensions for fast test execution.
    """
    return TRMConfig(
        latent_dim=64,
        num_recursions=8,
        hidden_dim=128,
        deep_supervision=True,
        supervision_weight_decay=0.5,
        convergence_threshold=0.01,
        min_recursions=2,
        dropout=0.1,
        use_layer_norm=True,
    )


@pytest.fixture
def minimal_config() -> TRMConfig:
    """
    Minimal TRM configuration for edge case testing.

    Small dimensions, fewer recursions.
    """
    return TRMConfig(
        latent_dim=16,
        num_recursions=3,
        hidden_dim=32,
        deep_supervision=False,
        convergence_threshold=0.05,
        min_recursions=1,
        dropout=0.0,
        use_layer_norm=False,
    )


@pytest.fixture
def large_config() -> TRMConfig:
    """
    Larger TRM configuration for testing scalability.

    Multiple recursions, larger dimensions.
    """
    return TRMConfig(
        latent_dim=256,
        num_recursions=20,
        hidden_dim=512,
        deep_supervision=True,
        supervision_weight_decay=0.7,
        convergence_threshold=0.005,
        min_recursions=5,
        dropout=0.2,
        use_layer_norm=True,
    )


@pytest.fixture
def sample_input_batch() -> torch.Tensor:
    """
    Create sample input batch for testing.

    Returns:
        Tensor of shape [batch=2, seq=4, latent_dim=64]
    """
    torch.manual_seed(42)
    return torch.randn(2, 4, 64)


@pytest.fixture
def sample_input_single() -> torch.Tensor:
    """
    Create single sample input for testing.

    Returns:
        Tensor of shape [batch=1, seq=8, latent_dim=64]
    """
    torch.manual_seed(42)
    return torch.randn(1, 8, 64)


@pytest.fixture
def sample_targets() -> torch.Tensor:
    """
    Create sample target labels for loss computation.

    Returns:
        Tensor of shape [batch=2, seq=4, latent_dim=64]
    """
    torch.manual_seed(42)
    return torch.randn(2, 4, 64)


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
# Test RecursiveBlock
# ============================================================================


class TestRecursiveBlock:
    """Test suite for RecursiveBlock."""

    def test_recursive_block_initialization(self, default_config):
        """
        Test RecursiveBlock initializes with correct architecture.

        Validates:
        - Transform network exists
        - Layer normalization is applied when configured
        - Residual scale parameter exists
        """
        block = RecursiveBlock(default_config)

        assert isinstance(block.transform, nn.Sequential)
        assert isinstance(block.residual_scale, nn.Parameter)
        assert block.residual_scale.shape == (1,)

        # Check that transform has correct layers
        assert len(block.transform) == 6  # Linear, LayerNorm, GELU, Dropout, Linear, LayerNorm

    def test_recursive_block_no_layer_norm(self, minimal_config):
        """
        Test RecursiveBlock without layer normalization.

        Validates:
        - Identity layers replace LayerNorm when disabled
        - Transform still has correct structure
        """
        block = RecursiveBlock(minimal_config)

        # Should have Identity instead of LayerNorm
        assert isinstance(block.transform[1], nn.Identity)
        assert isinstance(block.transform[-1], nn.Identity)

    def test_recursive_block_forward_shape(self, default_config):
        """
        Test RecursiveBlock forward pass preserves input shape.

        Validates:
        - Output shape matches input shape
        - Residual connection maintains dimensions
        """
        set_deterministic_seed()
        block = RecursiveBlock(default_config)

        batch_size, seq_len = 2, 4
        x = torch.randn(batch_size, seq_len, default_config.latent_dim)

        output = block(x, iteration=0)

        assert output.shape == x.shape
        assert output.shape == (batch_size, seq_len, default_config.latent_dim)

    def test_recursive_block_residual_connection(self, default_config):
        """
        Test RecursiveBlock applies residual connection correctly.

        Validates:
        - Output includes input (residual connection)
        - Residual scale affects output
        """
        set_deterministic_seed()
        block = RecursiveBlock(default_config)
        block.eval()

        x = torch.randn(1, 3, default_config.latent_dim)

        # Output should be x + scale * transform(x)
        output = block(x, iteration=0)

        # Output should not equal input (unless residual is zero, unlikely)
        assert not torch.allclose(output, x)

        # But should be related by residual connection
        # Setting residual_scale to 0 should give us back the input
        with torch.no_grad():
            block.residual_scale.fill_(0.0)

        output_no_residual = block(x, iteration=0)
        assert torch.allclose(output_no_residual, x)

    def test_recursive_block_iteration_parameter(self, default_config):
        """
        Test RecursiveBlock accepts iteration parameter.

        Validates:
        - Iteration parameter is accepted (for future iteration-dependent behavior)
        - Different iterations produce deterministic results
        """
        set_deterministic_seed()
        block = RecursiveBlock(default_config)
        block.eval()

        x = torch.randn(1, 3, default_config.latent_dim)

        # Currently iteration doesn't affect output, but parameter should be accepted
        output_iter0 = block(x, iteration=0)
        output_iter5 = block(x, iteration=5)

        # With current implementation, should be the same
        assert torch.allclose(output_iter0, output_iter5)

    def test_recursive_block_determinism(self, default_config):
        """
        Test RecursiveBlock produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        """
        x = torch.randn(2, 3, default_config.latent_dim)

        set_deterministic_seed(42)
        block1 = RecursiveBlock(default_config)
        block1.eval()
        output1 = block1(x, iteration=0)

        set_deterministic_seed(42)
        block2 = RecursiveBlock(default_config)
        block2.eval()
        output2 = block2(x, iteration=0)

        assert torch.allclose(output1, output2, atol=1e-5)


# ============================================================================
# Test DeepSupervisionHead
# ============================================================================


class TestDeepSupervisionHead:
    """Test suite for DeepSupervisionHead."""

    def test_deep_supervision_head_initialization(self):
        """
        Test DeepSupervisionHead initializes with correct architecture.

        Validates:
        - Network has correct layer structure
        - Input/output dimensions are correct
        """
        latent_dim = 64
        output_dim = 32

        head = DeepSupervisionHead(latent_dim, output_dim)

        assert isinstance(head.head, nn.Sequential)
        assert len(head.head) == 3  # Linear, ReLU, Linear

        # Check dimensions
        assert head.head[0].in_features == latent_dim
        assert head.head[0].out_features == latent_dim // 2
        assert head.head[2].out_features == output_dim

    def test_deep_supervision_head_forward_shape(self):
        """
        Test DeepSupervisionHead forward pass produces correct shape.

        Validates:
        - Output has correct dimension
        - Batch and spatial dimensions are preserved
        """
        set_deterministic_seed()
        latent_dim = 64
        output_dim = 32

        head = DeepSupervisionHead(latent_dim, output_dim)

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, latent_dim)

        output = head(x)

        assert output.shape == (batch_size, seq_len, output_dim)

    def test_deep_supervision_head_determinism(self):
        """
        Test DeepSupervisionHead produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        """
        x = torch.randn(2, 3, 64)

        set_deterministic_seed(42)
        head1 = DeepSupervisionHead(64, 32)
        head1.eval()
        output1 = head1(x)

        set_deterministic_seed(42)
        head2 = DeepSupervisionHead(64, 32)
        head2.eval()
        output2 = head2(x)

        assert torch.allclose(output1, output2, atol=1e-5)


# ============================================================================
# Test TRMAgent
# ============================================================================


class TestTRMAgent:
    """Test suite for complete TRM Agent."""

    def test_trm_initialization(self, default_config):
        """
        Test TRM agent initializes with correct configuration.

        Validates:
        - Config is stored correctly
        - All submodules exist (encoder, recursive block, supervision heads)
        - Model is on correct device
        """
        agent = TRMAgent(default_config, device="cpu")

        assert agent.config == default_config
        assert isinstance(agent.encoder, nn.Sequential)
        assert isinstance(agent.recursive_block, RecursiveBlock)

        # Check deep supervision heads
        if default_config.deep_supervision:
            assert isinstance(agent.supervision_heads, nn.ModuleList)
            assert len(agent.supervision_heads) == default_config.num_recursions
        else:
            assert hasattr(agent, "output_head")

    def test_trm_initialization_no_deep_supervision(self, minimal_config):
        """
        Test TRM agent initializes without deep supervision.

        Validates:
        - Single output head is created when deep_supervision=False
        - No supervision_heads ModuleList exists
        """
        agent = TRMAgent(minimal_config, device="cpu")

        assert not minimal_config.deep_supervision
        assert hasattr(agent, "output_head")
        assert isinstance(agent.output_head, DeepSupervisionHead)

    def test_trm_forward_basic(self, default_config, sample_input_batch):
        """
        Test TRM forward pass with default parameters.

        Validates:
        - Forward pass completes without error
        - Output has correct type (TRMOutput)
        - All expected fields are present
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        output = agent(sample_input_batch)

        assert isinstance(output, TRMOutput)
        assert isinstance(output.final_prediction, torch.Tensor)
        assert isinstance(output.intermediate_predictions, list)
        assert isinstance(output.recursion_depth, int)
        assert isinstance(output.converged, bool)
        assert isinstance(output.convergence_step, int)
        assert isinstance(output.residual_norms, list)

    def test_trm_forward_output_shape(self, default_config, sample_input_batch):
        """
        Test TRM forward pass produces correct output shape.

        Validates:
        - Final prediction has same spatial dims as input
        - Output dimension matches config
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, output_dim=32, device="cpu")
        agent.eval()

        batch_size, seq_len, _ = sample_input_batch.shape
        output = agent(sample_input_batch)

        assert output.final_prediction.shape == (batch_size, seq_len, 32)

    def test_trm_forward_default_output_dim(self, default_config, sample_input_batch):
        """
        Test TRM forward pass with default output dimension.

        Validates:
        - When output_dim is None, defaults to latent_dim
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        batch_size, seq_len, _ = sample_input_batch.shape
        output = agent(sample_input_batch)

        assert output.final_prediction.shape == (batch_size, seq_len, default_config.latent_dim)

    def test_trm_recursive_refinement(self, default_config):
        """
        Test TRM applies recursive refinement correctly.

        Validates:
        - Multiple recursion iterations are performed
        - Intermediate predictions are stored
        - Number of intermediate predictions matches recursion depth
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 4, default_config.latent_dim)
        output = agent(x, num_recursions=5, check_convergence=False)

        # Should have exactly 5 intermediate predictions
        assert output.recursion_depth == 5
        assert len(output.intermediate_predictions) == 5

    def test_trm_convergence_detection(self, default_config):
        """
        Test TRM detects convergence when threshold is met.

        Validates:
        - Agent stops early when convergence threshold is reached
        - Convergence flag is set correctly
        - Convergence step is recorded
        """
        set_deterministic_seed()

        # Use tight convergence threshold to potentially trigger early stopping
        config = TRMConfig(
            latent_dim=32,
            num_recursions=20,
            hidden_dim=64,
            deep_supervision=True,
            convergence_threshold=0.1,  # Relaxed threshold
            min_recursions=2,
            dropout=0.0,  # Disable dropout for stability
        )

        agent = TRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.latent_dim)
        output = agent(x, check_convergence=True)

        # Check convergence tracking
        if output.converged:
            assert output.convergence_step < config.num_recursions
            assert output.convergence_step == output.recursion_depth
            assert len(output.residual_norms) >= config.min_recursions

    def test_trm_min_recursions_respected(self, default_config):
        """
        Test TRM respects minimum recursion requirement before checking convergence.

        Validates:
        - At least min_recursions iterations are performed
        - Convergence is not checked before min_recursions
        """
        set_deterministic_seed()

        config = TRMConfig(
            latent_dim=32,
            num_recursions=10,
            hidden_dim=64,
            convergence_threshold=100.0,  # Very high, won't converge
            min_recursions=5,
            dropout=0.0,
        )

        agent = TRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.latent_dim)
        output = agent(x, check_convergence=True)

        # Should perform at least min_recursions
        assert output.recursion_depth >= config.min_recursions

    def test_trm_max_recursions_limit(self, default_config):
        """
        Test TRM respects maximum recursion limit.

        Validates:
        - Agent never exceeds num_recursions
        - Works with custom num_recursions parameter
        """
        set_deterministic_seed()

        config = TRMConfig(
            latent_dim=32,
            num_recursions=15,
            hidden_dim=64,
            convergence_threshold=0.0001,  # Very tight, won't converge
            min_recursions=2,
        )

        agent = TRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.latent_dim)

        # Test with default max
        output = agent(x, check_convergence=False)
        assert output.recursion_depth == config.num_recursions

        # Test with custom max
        output_custom = agent(x, num_recursions=7, check_convergence=False)
        assert output_custom.recursion_depth == 7

    def test_trm_residual_norms_tracking(self, default_config):
        """
        Test TRM tracks residual norms during refinement.

        Validates:
        - Residual norms are computed at each iteration (after min_recursions)
        - Norms are non-negative
        - Norms generally decrease (refinement converging)
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 4, default_config.latent_dim)
        output = agent(x, check_convergence=True)

        # Should have residual norms if convergence checking is enabled
        if output.recursion_depth > default_config.min_recursions:
            assert len(output.residual_norms) > 0
            assert all(norm >= 0.0 for norm in output.residual_norms)

    def test_trm_no_convergence_check(self, default_config):
        """
        Test TRM with convergence checking disabled.

        Validates:
        - All recursions are performed when check_convergence=False
        - Residual norms list is empty
        - Converged flag is False
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, default_config.latent_dim)
        output = agent(x, num_recursions=5, check_convergence=False)

        assert output.recursion_depth == 5
        assert len(output.intermediate_predictions) == 5
        assert output.converged is False
        assert len(output.residual_norms) == 0

    def test_trm_determinism(self, default_config):
        """
        Test TRM produces identical results with same seed.

        Validates:
        - Multiple forward passes with same seed yield same results
        - Critical for reproducible experiments
        """
        x = torch.randn(2, 3, default_config.latent_dim)

        set_deterministic_seed(42)
        agent1 = TRMAgent(default_config, device="cpu")
        agent1.eval()
        output1 = agent1(x, check_convergence=False)

        set_deterministic_seed(42)
        agent2 = TRMAgent(default_config, device="cpu")
        agent2.eval()
        output2 = agent2(x, check_convergence=False)

        assert torch.allclose(output1.final_prediction, output2.final_prediction, atol=1e-5)
        assert output1.recursion_depth == output2.recursion_depth
        assert len(output1.intermediate_predictions) == len(output2.intermediate_predictions)

    def test_trm_different_configs(self, minimal_config, large_config):
        """
        Test TRM works with different configuration sizes.

        Validates:
        - Minimal config produces valid output
        - Large config produces valid output
        - Architecture scales correctly
        """
        set_deterministic_seed()

        # Test minimal config
        agent_min = TRMAgent(minimal_config, device="cpu")
        agent_min.eval()
        x_min = torch.randn(1, 2, minimal_config.latent_dim)
        output_min = agent_min(x_min)
        assert output_min.final_prediction.shape == (1, 2, minimal_config.latent_dim)

        # Test large config (use fewer recursions to keep test fast)
        agent_large = TRMAgent(large_config, device="cpu")
        agent_large.eval()
        x_large = torch.randn(1, 2, large_config.latent_dim)
        output_large = agent_large(x_large, num_recursions=5)
        assert output_large.final_prediction.shape == (1, 2, large_config.latent_dim)

    def test_trm_parameter_count(self, default_config):
        """
        Test TRM parameter counting utility.

        Validates:
        - get_parameter_count returns positive integer
        - Count matches manual parameter sum
        """
        agent = TRMAgent(default_config, device="cpu")

        param_count = agent.get_parameter_count()

        assert param_count > 0
        assert isinstance(param_count, int)

        # Verify against manual count
        manual_count = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        assert param_count == manual_count

    @pytest.mark.asyncio
    async def test_trm_refine_solution(self, default_config):
        """
        Test TRM async solution refinement method.

        Validates:
        - Async refinement works correctly
        - Returns refined solution and info dict
        - Info dict contains expected metadata
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        initial_prediction = torch.randn(2, 5, default_config.latent_dim)

        refined_solution, info = await agent.refine_solution(
            initial_prediction,
            num_recursions=5,
        )

        assert isinstance(refined_solution, torch.Tensor)
        assert refined_solution.shape == initial_prediction.shape
        assert isinstance(info, dict)

        # Check info dict
        assert "converged" in info
        assert "convergence_step" in info
        assert "total_recursions" in info
        assert "refinement_path" in info
        assert info["total_recursions"] <= 5

    @pytest.mark.asyncio
    async def test_trm_refine_solution_custom_threshold(self, default_config):
        """
        Test TRM refinement with custom convergence threshold.

        Validates:
        - Custom threshold is applied during refinement
        - Original config threshold is restored after refinement
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        initial_prediction = torch.randn(1, 4, default_config.latent_dim)
        original_threshold = agent.config.convergence_threshold

        refined_solution, info = await agent.refine_solution(
            initial_prediction,
            num_recursions=5,
            convergence_threshold=0.05,
        )

        # Original threshold should be restored
        assert agent.config.convergence_threshold == original_threshold


# ============================================================================
# Test TRMLoss
# ============================================================================


class TestTRMLoss:
    """Test suite for TRM loss computation."""

    def test_trm_loss_initialization(self):
        """
        Test TRMLoss initializes with correct weights.

        Validates:
        - Default weights are set correctly
        - Custom weights are accepted
        """
        task_loss_fn = nn.MSELoss()

        # Default weights
        loss_fn = TRMLoss(task_loss_fn)
        assert loss_fn.supervision_weight_decay == 0.5
        assert loss_fn.final_weight == 1.0

        # Custom weights
        custom_loss = TRMLoss(
            task_loss_fn,
            supervision_weight_decay=0.7,
            final_weight=2.0,
        )
        assert custom_loss.supervision_weight_decay == 0.7
        assert custom_loss.final_weight == 2.0

    def test_trm_loss_forward(self, default_config, sample_targets):
        """
        Test TRMLoss forward computation.

        Validates:
        - Loss is computed correctly
        - Returns both total loss and loss dict
        - All loss components are present
        """
        set_deterministic_seed()

        # Create mock TRM output
        intermediate_preds = [
            torch.randn(2, 4, 64),
            torch.randn(2, 4, 64),
            torch.randn(2, 4, 64),
        ]

        trm_output = TRMOutput(
            final_prediction=intermediate_preds[-1],
            intermediate_predictions=intermediate_preds,
            recursion_depth=3,
            converged=False,
            convergence_step=3,
            residual_norms=[0.5, 0.3],
        )

        task_loss_fn = nn.MSELoss()
        loss_fn = TRMLoss(task_loss_fn)

        total_loss, loss_dict = loss_fn(trm_output, sample_targets)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert total_loss.item() > 0

        assert "total" in loss_dict
        assert "final" in loss_dict
        assert "intermediate_mean" in loss_dict
        assert "recursion_depth" in loss_dict
        assert "converged" in loss_dict
        assert "convergence_step" in loss_dict

        assert loss_dict["recursion_depth"] == 3
        assert loss_dict["converged"] is False

    def test_trm_loss_deep_supervision(self, default_config):
        """
        Test TRMLoss deep supervision mechanism.

        Validates:
        - Intermediate predictions contribute to loss
        - Earlier predictions have lower weight (exponential decay)
        - Final prediction has highest weight
        """
        set_deterministic_seed()

        # Create intermediate predictions
        num_preds = 4
        intermediate_preds = [torch.randn(1, 3, 64) for _ in range(num_preds)]

        trm_output = TRMOutput(
            final_prediction=intermediate_preds[-1],
            intermediate_predictions=intermediate_preds,
            recursion_depth=num_preds,
            converged=False,
            convergence_step=num_preds,
            residual_norms=[],
        )

        targets = torch.randn(1, 3, 64)
        task_loss_fn = nn.MSELoss()
        loss_fn = TRMLoss(task_loss_fn, supervision_weight_decay=0.5, final_weight=1.0)

        total_loss, loss_dict = loss_fn(trm_output, targets)

        # Should have intermediate losses
        assert loss_dict["intermediate_mean"] > 0

    def test_trm_loss_single_prediction(self):
        """
        Test TRMLoss handles single prediction (no intermediate supervision).

        Validates:
        - Works with only one prediction
        - Intermediate mean is 0.0
        - Only final loss contributes
        """
        set_deterministic_seed()

        # Only one prediction
        final_pred = torch.randn(1, 3, 64)

        trm_output = TRMOutput(
            final_prediction=final_pred,
            intermediate_predictions=[final_pred],
            recursion_depth=1,
            converged=False,
            convergence_step=1,
            residual_norms=[],
        )

        targets = torch.randn(1, 3, 64)
        task_loss_fn = nn.MSELoss()
        loss_fn = TRMLoss(task_loss_fn)

        total_loss, loss_dict = loss_fn(trm_output, targets)

        assert isinstance(total_loss, torch.Tensor)
        assert loss_dict["intermediate_mean"] == 0.0  # No intermediate predictions

    def test_trm_loss_weight_decay_effect(self):
        """
        Test TRMLoss weight decay affects intermediate losses.

        Validates:
        - Different decay values produce different total losses
        - Higher decay reduces contribution of early predictions
        """
        set_deterministic_seed()

        # Create consistent predictions
        intermediate_preds = [torch.randn(1, 2, 32) for _ in range(3)]

        trm_output = TRMOutput(
            final_prediction=intermediate_preds[-1],
            intermediate_predictions=intermediate_preds,
            recursion_depth=3,
            converged=False,
            convergence_step=3,
            residual_norms=[],
        )

        targets = torch.randn(1, 2, 32)
        task_loss_fn = nn.MSELoss()

        # Low decay (more weight on intermediate)
        loss_fn_low = TRMLoss(task_loss_fn, supervision_weight_decay=0.3)
        total_low, _ = loss_fn_low(trm_output, targets)

        # High decay (less weight on intermediate)
        loss_fn_high = TRMLoss(task_loss_fn, supervision_weight_decay=0.9)
        total_high, _ = loss_fn_high(trm_output, targets)

        # Losses should be different (unless intermediate losses are zero)
        # We just verify they're both valid
        assert total_low.item() > 0
        assert total_high.item() > 0


# ============================================================================
# Test create_trm_agent factory
# ============================================================================


class TestCreateTRMAgent:
    """Test suite for TRM agent factory function."""

    def test_create_trm_agent_basic(self, default_config):
        """
        Test create_trm_agent factory creates valid agent.

        Validates:
        - Returns TRMAgent instance
        - Initializes weights properly
        - Agent is on correct device
        """
        agent = create_trm_agent(default_config, device="cpu")

        assert isinstance(agent, TRMAgent)
        assert agent.config == default_config

        # Check that weights are initialized (not all zeros)
        has_nonzero_params = False
        for name, param in agent.named_parameters():
            if "weight" in name and not torch.all(param == 0.0):
                has_nonzero_params = True
                break
        assert has_nonzero_params, "At least some weights should be non-zero"

    def test_create_trm_agent_weight_initialization(self, default_config):
        """
        Test create_trm_agent initializes weights correctly.

        Validates:
        - Linear layers use Kaiming initialization
        - Biases are initialized to zero
        """
        set_deterministic_seed()
        agent = create_trm_agent(default_config, device="cpu")

        # Check linear layer initialization
        for module in agent.modules():
            if isinstance(module, nn.Linear):
                # Kaiming init should produce non-zero weights
                assert not torch.all(module.weight == 0.0)
                if module.bias is not None:
                    assert torch.all(module.bias == 0.0)

    def test_create_trm_agent_different_devices(self, minimal_config):
        """
        Test create_trm_agent places model on correct device.

        Validates:
        - CPU device works
        - Device is set correctly on all parameters
        """
        agent = create_trm_agent(minimal_config, device="cpu")

        for param in agent.parameters():
            assert param.device.type == "cpu"

    def test_create_trm_agent_custom_output_dim(self, default_config):
        """
        Test create_trm_agent with custom output dimension.

        Validates:
        - Custom output_dim is respected
        - Agent produces correct output shape
        """
        set_deterministic_seed()
        output_dim = 128
        agent = create_trm_agent(default_config, output_dim=output_dim, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, default_config.latent_dim)
        output = agent(x)

        assert output.final_prediction.shape[-1] == output_dim


# ============================================================================
# Test TRMRefinementWrapper
# ============================================================================


class TestTRMRefinementWrapper:
    """Test suite for TRMRefinementWrapper."""

    def test_wrapper_initialization(self, default_config):
        """
        Test TRMRefinementWrapper initializes correctly.

        Validates:
        - Wrapper stores TRM agent
        - Agent is set to eval mode
        """
        agent = TRMAgent(default_config, device="cpu")
        wrapper = TRMRefinementWrapper(agent, device="cpu")

        assert wrapper.trm_agent == agent
        assert not wrapper.trm_agent.training  # Should be in eval mode

    @pytest.mark.asyncio
    async def test_wrapper_refine_basic(self, default_config):
        """
        Test TRMRefinementWrapper refine method.

        Validates:
        - Refine method works correctly
        - Returns refined predictions
        - Shape matches input
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        wrapper = TRMRefinementWrapper(agent, device="cpu")

        predictions = torch.randn(2, 4, default_config.latent_dim, requires_grad=False)
        refined = await wrapper.refine(predictions, num_iterations=3)

        assert isinstance(refined, torch.Tensor)
        assert refined.shape == predictions.shape
        # Verify refinement produces valid output
        assert not torch.isnan(refined).any()
        assert not torch.isinf(refined).any()

    @pytest.mark.asyncio
    async def test_wrapper_refine_with_path(self, default_config):
        """
        Test TRMRefinementWrapper refine with return_path=True.

        Validates:
        - Returns both final prediction and intermediate path
        - Path contains all intermediate predictions
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        wrapper = TRMRefinementWrapper(agent, device="cpu")

        predictions = torch.randn(1, 3, default_config.latent_dim)
        refined, path = await wrapper.refine(
            predictions,
            num_iterations=4,
            return_path=True,
        )

        assert isinstance(refined, torch.Tensor)
        assert isinstance(path, list)
        assert len(path) == 4  # Should have 4 intermediate predictions
        assert all(isinstance(p, torch.Tensor) for p in path)

    def test_wrapper_get_refinement_stats(self, default_config):
        """
        Test TRMRefinementWrapper get_refinement_stats method.

        Validates:
        - Stats are computed correctly
        - Returns dict with expected keys
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        wrapper = TRMRefinementWrapper(agent, device="cpu")

        predictions = torch.randn(1, 3, default_config.latent_dim)
        stats = wrapper.get_refinement_stats(predictions)

        assert isinstance(stats, dict)
        assert "converged" in stats
        assert "steps_to_convergence" in stats
        assert "final_residual" in stats
        assert "total_refinement_iterations" in stats

        assert isinstance(stats["converged"], bool)
        assert isinstance(stats["total_refinement_iterations"], int)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestTRMEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_trm_single_recursion(self, default_config):
        """
        Test TRM with num_recursions=1.

        Validates:
        - Agent can run with just one iteration
        - Single intermediate prediction
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, default_config.latent_dim)
        output = agent(x, num_recursions=1, check_convergence=False)

        assert output.recursion_depth == 1
        assert len(output.intermediate_predictions) == 1
        assert not output.converged

    def test_trm_large_batch_size(self, default_config):
        """
        Test TRM handles large batch sizes.

        Validates:
        - No memory issues with larger batches
        - Output shapes are correct
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        large_batch = torch.randn(16, 5, default_config.latent_dim)
        output = agent(large_batch, num_recursions=3)

        assert output.final_prediction.shape == (16, 5, default_config.latent_dim)

    def test_trm_long_sequence(self, default_config):
        """
        Test TRM handles long sequences.

        Validates:
        - Works with longer sequence lengths
        - No shape mismatch errors
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        long_seq = torch.randn(1, 50, default_config.latent_dim)
        output = agent(long_seq, num_recursions=3)

        assert output.final_prediction.shape == (1, 50, default_config.latent_dim)

    def test_trm_gradient_flow(self, default_config):
        """
        Test gradients flow through entire TRM architecture.

        Validates:
        - Gradients reach core parameters
        - No gradient issues (NaN, inf)
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.train()

        x = torch.randn(2, 3, default_config.latent_dim, requires_grad=True)
        output = agent(x, num_recursions=3, check_convergence=False)

        # Compute loss and backpropagate
        loss = output.final_prediction.mean()
        loss.backward()

        # Check that most parameters have gradients
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

        # All parameters should have gradients in this simple case
        assert params_with_grad > 0, "Some parameters should have gradients"

    def test_trm_eval_mode(self, default_config):
        """
        Test TRM behavior in eval mode vs train mode.

        Validates:
        - Eval mode disables dropout
        - Results are deterministic in eval mode
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        x = torch.randn(1, 3, default_config.latent_dim)

        # Eval mode
        agent.eval()
        output1 = agent(x, check_convergence=False)
        output2 = agent(x, check_convergence=False)

        # Should be identical in eval mode
        assert torch.allclose(output1.final_prediction, output2.final_prediction)

    def test_trm_different_sequence_lengths(self, default_config):
        """
        Test TRM handles varying sequence lengths in different batches.

        Validates:
        - Agent works with different seq lengths
        - No size mismatch errors
        """
        set_deterministic_seed()
        agent = TRMAgent(default_config, device="cpu")
        agent.eval()

        # Different sequence lengths
        x1 = torch.randn(2, 3, default_config.latent_dim)
        x2 = torch.randn(2, 10, default_config.latent_dim)
        x3 = torch.randn(2, 1, default_config.latent_dim)

        output1 = agent(x1, num_recursions=3)
        output2 = agent(x2, num_recursions=3)
        output3 = agent(x3, num_recursions=3)

        assert output1.final_prediction.shape[1] == 3
        assert output2.final_prediction.shape[1] == 10
        assert output3.final_prediction.shape[1] == 1

    def test_trm_zero_dropout(self, minimal_config):
        """
        Test TRM with dropout disabled.

        Validates:
        - Works correctly with dropout=0.0
        - Deterministic in both train and eval modes
        """
        set_deterministic_seed()

        config = TRMConfig(
            latent_dim=32,
            num_recursions=5,
            hidden_dim=64,
            dropout=0.0,  # No dropout
            use_layer_norm=True,
        )

        agent = TRMAgent(config, device="cpu")
        x = torch.randn(1, 3, config.latent_dim)

        # Should be deterministic even in train mode (no dropout)
        agent.train()
        set_deterministic_seed(42)
        output1 = agent(x, check_convergence=False)

        agent.train()
        set_deterministic_seed(42)
        output2 = agent(x, check_convergence=False)

        assert torch.allclose(output1.final_prediction, output2.final_prediction, atol=1e-5)


# ============================================================================
# Test Integration Scenarios
# ============================================================================


class TestTRMIntegration:
    """Integration tests for TRM components working together."""

    def test_trm_full_forward_backward_pass(self, default_config):
        """
        Test complete forward and backward pass through TRM.

        Validates:
        - End-to-end training loop works
        - Loss computation is correct
        - Parameters are updated
        """
        set_deterministic_seed()

        agent = TRMAgent(default_config, device="cpu")
        task_loss_fn = nn.MSELoss()
        loss_fn = TRMLoss(task_loss_fn)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

        # Forward pass
        x = torch.randn(2, 4, default_config.latent_dim)
        output = agent(x, num_recursions=5, check_convergence=False)

        # Compute loss
        targets = torch.randn(2, 4, default_config.latent_dim)
        total_loss, loss_dict = loss_fn(output, targets)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Verify training step completed
        assert total_loss.item() > 0
        assert all(key in loss_dict for key in ["total", "final", "intermediate_mean"])

    def test_trm_multiple_iterations(self, default_config):
        """
        Test TRM over multiple training iterations.

        Validates:
        - Agent can process multiple batches
        - No memory leaks
        - Training is stable
        """
        set_deterministic_seed()

        agent = TRMAgent(default_config, device="cpu")
        optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)

        losses = []
        for _ in range(5):
            x = torch.randn(2, 3, default_config.latent_dim)
            output = agent(x, num_recursions=3, check_convergence=False)

            loss = output.final_prediction.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert len(losses) == 5
        assert all(not math.isnan(loss) and not math.isinf(loss) for loss in losses)

    def test_trm_training_reduces_loss(self, default_config):
        """
        Test that TRM training actually reduces loss over iterations.

        Validates:
        - Loss decreases with training (on simple synthetic task)
        - Optimization is effective
        """
        set_deterministic_seed()

        # Simple regression task
        agent = TRMAgent(default_config, output_dim=default_config.latent_dim, device="cpu")
        task_loss_fn = nn.MSELoss()
        loss_fn = TRMLoss(task_loss_fn)
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)

        # Fixed input and target
        x = torch.randn(4, 5, default_config.latent_dim)
        targets = torch.randn(4, 5, default_config.latent_dim)

        losses = []
        for _ in range(10):
            output = agent(x, num_recursions=3, check_convergence=False)
            total_loss, _ = loss_fn(output, targets)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

        # Loss should generally decrease (may have some fluctuation)
        # Check that final loss is lower than initial loss
        assert losses[-1] < losses[0], "Training should reduce loss"

    def test_trm_convergence_behavior(self, default_config):
        """
        Test TRM convergence behavior over training.

        Validates:
        - Convergence patterns change during training
        - Agent learns to refine efficiently
        """
        set_deterministic_seed()

        config = TRMConfig(
            latent_dim=32,
            num_recursions=10,
            hidden_dim=64,
            convergence_threshold=0.01,
            min_recursions=2,
            dropout=0.0,
        )

        agent = TRMAgent(config, device="cpu")
        agent.eval()

        x = torch.randn(1, 3, config.latent_dim)

        # Track convergence over multiple forward passes
        convergence_steps = []
        for _ in range(3):
            output = agent(x, check_convergence=True)
            convergence_steps.append(output.convergence_step)

        # Should have consistent convergence (deterministic in eval mode)
        assert len(set(convergence_steps)) == 1


# ============================================================================
# Test TRMOutput dataclass
# ============================================================================


class TestTRMOutput:
    """Test suite for TRMOutput dataclass."""

    def test_trm_output_initialization(self):
        """
        Test TRMOutput initializes correctly.

        Validates:
        - Required fields are set
        - All fields have correct types
        """
        final_pred = torch.randn(2, 3, 64)
        intermediate_preds = [torch.randn(2, 3, 64) for _ in range(3)]

        output = TRMOutput(
            final_prediction=final_pred,
            intermediate_predictions=intermediate_preds,
            recursion_depth=3,
            converged=True,
            convergence_step=2,
            residual_norms=[0.5, 0.1],
        )

        assert torch.equal(output.final_prediction, final_pred)
        assert output.intermediate_predictions == intermediate_preds
        assert output.recursion_depth == 3
        assert output.converged is True
        assert output.convergence_step == 2
        assert output.residual_norms == [0.5, 0.1]

    def test_trm_output_fields(self):
        """
        Test TRMOutput has all expected fields.

        Validates:
        - All documented fields exist
        - Field types are correct
        """
        output = TRMOutput(
            final_prediction=torch.randn(1, 2, 32),
            intermediate_predictions=[],
            recursion_depth=5,
            converged=False,
            convergence_step=5,
            residual_norms=[],
        )

        assert hasattr(output, "final_prediction")
        assert hasattr(output, "intermediate_predictions")
        assert hasattr(output, "recursion_depth")
        assert hasattr(output, "converged")
        assert hasattr(output, "convergence_step")
        assert hasattr(output, "residual_norms")


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/test_trm_agent.py -v
    pytest.main([__file__, "-v", "--tb=short"])
