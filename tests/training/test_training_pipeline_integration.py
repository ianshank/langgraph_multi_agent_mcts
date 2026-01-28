"""
Integration tests for the unified training pipeline.

Tests end-to-end training flow including:
- HRM agent training with proper loss computation
- TRM agent training with deep supervision
- Self-play evaluation
- Checkpoint saving and loading
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn


@pytest.fixture
def mock_game_state():
    """Create a mock game state for testing."""
    state = MagicMock()
    state.is_terminal.side_effect = [False, False, False, True]
    state.get_reward.return_value = 1.0
    state.apply_action.return_value = state
    state.get_legal_actions.return_value = ["a", "b", "c"]

    # For tensor conversion
    state.to_tensor.return_value = torch.randn(1, 19, 19)

    return state


@pytest.fixture
def initial_state_fn(mock_game_state):
    """Create initial state function."""
    return lambda: mock_game_state


@pytest.fixture
def system_config():
    """Create minimal system configuration for testing."""
    from src.training.system_config import (
        HRMConfig,
        MCTSConfig,
        NeuralNetConfig,
        SystemConfig,
        TrainingConfig,
        TRMConfig,
    )

    return SystemConfig(
        # Minimal sizes for testing
        neural_net=NeuralNetConfig(
            num_res_blocks=1,
            num_channels=16,
            policy_head_channels=2,
            value_head_channels=1,
            action_size=10,
        ),
        hrm=HRMConfig(
            h_dim=32,
            l_dim=16,
            num_h_layers=1,
            num_l_layers=1,
            max_outer_steps=2,
            halt_threshold=0.9,
            ponder_epsilon=0.01,
            ponder_weight=0.01,
            dropout=0.0,
        ),
        trm=TRMConfig(
            latent_dim=32,
            hidden_dim=64,
            num_recursions=2,
            min_recursions=1,
            convergence_threshold=0.01,
            supervision_weight_decay=0.5,
            use_layer_norm=True,
            deep_supervision=True,
            dropout=0.0,
        ),
        mcts=MCTSConfig(
            num_simulations=4,
            c_puct=1.0,
            dirichlet_alpha=0.3,
            temperature_threshold=10,
        ),
        training=TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            buffer_size=100,
            games_per_iteration=1,
            checkpoint_interval=1,
            evaluation_games=2,
            win_threshold=0.5,
            patience=10,
            min_delta=0.01,
            hrm_train_batches=2,
            trm_train_batches=2,
            eval_temperature=0.0,
            gradient_clip_norm=1.0,
        ),
        device="cpu",
        seed=42,
        use_mixed_precision=False,
        checkpoint_dir=tempfile.mkdtemp(),
        data_dir=tempfile.mkdtemp(),
        log_dir=tempfile.mkdtemp(),
        use_wandb=False,
    )


class TestHRMTrainerIntegration:
    """Integration tests for HRM training."""

    @pytest.fixture
    def hrm_setup(self, system_config):
        """Setup HRM training components."""
        from src.agents.hrm_agent import HRMLoss, create_hrm_agent
        from src.training.agent_trainer import HRMTrainer, HRMTrainingConfig

        agent = create_hrm_agent(system_config.hrm, device="cpu")
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        loss_fn = HRMLoss(ponder_weight=0.01)
        config = HRMTrainingConfig(
            batch_size=4,
            num_batches=2,
            gradient_clip_norm=1.0,
        )

        trainer = HRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        return trainer, agent

    @pytest.mark.asyncio
    async def test_hrm_train_step(self, hrm_setup):
        """Test single HRM training step."""
        trainer, agent = hrm_setup

        # Create synthetic data matching agent dimensions
        states = torch.randn(4, 8, 32)  # batch, seq, h_dim
        targets = torch.randn(4, 8, 32)

        metrics = await trainer.train_step(states, targets)

        assert metrics.loss > 0
        assert metrics.samples_processed == 4
        assert metrics.gradient_norm >= 0
        assert "hrm_task_loss" in metrics.component_losses
        assert "hrm_ponder_cost" in metrics.component_losses
        assert "hrm_halt_step" in metrics.component_losses

    @pytest.mark.asyncio
    async def test_hrm_train_epoch(self, hrm_setup):
        """Test HRM training epoch."""
        from src.training.agent_trainer import DummyDataLoader

        trainer, agent = hrm_setup

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=32,
            output_dim=32,
            num_batches=2,
            device="cpu",
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "hrm_halt_step" in metrics


class TestTRMTrainerIntegration:
    """Integration tests for TRM training."""

    @pytest.fixture
    def trm_setup(self, system_config):
        """Setup TRM training components."""
        from src.agents.trm_agent import TRMLoss, create_trm_agent
        from src.training.agent_trainer import TRMTrainer, TRMTrainingConfig

        agent = create_trm_agent(
            system_config.trm,
            output_dim=10,
            device="cpu",
        )
        optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
        loss_fn = TRMLoss(
            task_loss_fn=nn.MSELoss(),
            supervision_weight_decay=0.5,
        )
        config = TRMTrainingConfig(
            batch_size=4,
            num_batches=2,
            gradient_clip_norm=1.0,
        )

        trainer = TRMTrainer(
            agent=agent,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            device="cpu",
        )

        return trainer, agent

    @pytest.mark.asyncio
    async def test_trm_train_step(self, trm_setup):
        """Test single TRM training step."""
        trainer, agent = trm_setup

        # Create synthetic data matching agent dimensions
        inputs = torch.randn(4, 8, 32)  # batch, seq, latent_dim
        targets = torch.randn(4, 8, 10)  # batch, seq, output_dim

        metrics = await trainer.train_step(inputs, targets)

        assert metrics.loss > 0
        assert metrics.samples_processed == 4
        assert metrics.gradient_norm >= 0
        assert "trm_final_loss" in metrics.component_losses
        assert "trm_convergence_step" in metrics.component_losses

    @pytest.mark.asyncio
    async def test_trm_train_epoch(self, trm_setup):
        """Test TRM training epoch."""
        from src.training.agent_trainer import DummyDataLoader

        trainer, agent = trm_setup

        data_loader = DummyDataLoader(
            batch_size=4,
            input_dim=32,
            output_dim=10,
            num_batches=2,
            device="cpu",
        )

        metrics = await trainer.train_epoch(data_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0
        assert "trm_convergence_step" in metrics


class TestSelfPlayEvaluatorIntegration:
    """Integration tests for self-play evaluation."""

    @pytest.fixture
    def evaluator_setup(self, mock_game_state, initial_state_fn):
        """Setup evaluation components."""
        from src.training.agent_trainer import EvaluationConfig, SelfPlayEvaluator

        # Mock MCTS
        mock_mcts = MagicMock()
        mock_mcts.search = AsyncMock(return_value=({"a": 0.5, "b": 0.3, "c": 0.2}, 0.6))

        config = EvaluationConfig(
            num_games=2,
            temperature=0.0,
            mcts_iterations=4,
            win_threshold=0.5,
        )

        evaluator = SelfPlayEvaluator(
            mcts=mock_mcts,
            initial_state_fn=initial_state_fn,
            config=config,
            device="cpu",
        )

        return evaluator, mock_mcts

    @pytest.mark.asyncio
    async def test_play_game(self, evaluator_setup):
        """Test playing a single game."""
        evaluator, mock_mcts = evaluator_setup

        # Create mock models
        model1 = MagicMock()
        model2 = MagicMock()

        result, stats = await evaluator.play_game(model1, model2)

        assert result in [-1, 0, 1]
        assert "moves" in stats
        assert "model1_avg_mcts_value" in stats

    @pytest.mark.asyncio
    async def test_evaluate(self, evaluator_setup):
        """Test full evaluation."""
        evaluator, mock_mcts = evaluator_setup

        # Create mock models
        current_model = MagicMock()
        best_model = MagicMock()

        metrics = await evaluator.evaluate(current_model, best_model)

        assert "win_rate" in metrics
        assert "wins" in metrics
        assert "losses" in metrics
        assert "draws" in metrics
        assert "eval_games" in metrics
        assert metrics["eval_games"] == 2


class TestUnifiedOrchestratorIntegration:
    """Integration tests for the unified training orchestrator."""

    @pytest.fixture
    def orchestrator(self, system_config, initial_state_fn):
        """Create orchestrator instance."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        return UnifiedTrainingOrchestrator(
            config=system_config,
            initial_state_fn=initial_state_fn,
            board_size=19,
        )

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes all components."""
        assert orchestrator.policy_value_net is not None
        assert orchestrator.hrm_agent is not None
        assert orchestrator.trm_agent is not None
        assert orchestrator.mcts is not None
        assert orchestrator.replay_buffer is not None

    def test_parameter_count(self, orchestrator):
        """Test that models have parameters."""
        pv_params = orchestrator.policy_value_net.get_parameter_count()
        hrm_params = orchestrator.hrm_agent.get_parameter_count()
        trm_params = orchestrator.trm_agent.get_parameter_count()

        assert pv_params > 0
        assert hrm_params > 0
        assert trm_params > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_train_iteration(self, orchestrator):
        """Test single training iteration."""
        # This is a slow test that runs actual training
        with patch.object(
            orchestrator.self_play_collector,
            "play_game",
            new_callable=AsyncMock,
        ) as mock_play:
            # Mock game results
            mock_example = MagicMock()
            mock_example.state = torch.randn(19, 19)
            mock_example.policy_target = torch.rand(10)
            mock_example.value_target = torch.tensor([0.5])
            mock_play.return_value = [mock_example] * 10

            metrics = await orchestrator.train_iteration(iteration=1)

            assert "games_generated" in metrics
            assert "policy_loss" in metrics
            assert "value_loss" in metrics

    def test_save_and_load_checkpoint(self, orchestrator, system_config):
        """Test checkpoint saving and loading."""
        # Save checkpoint
        orchestrator._save_checkpoint(
            iteration=1,
            metrics={"test_metric": 0.5},
            is_best=True,
        )

        # Verify checkpoint file exists
        checkpoint_path = Path(system_config.checkpoint_dir) / "checkpoint_iter_1.pt"
        assert checkpoint_path.exists()

        # Verify best model file exists
        best_path = Path(system_config.checkpoint_dir) / "best_model.pt"
        assert best_path.exists()

        # Load checkpoint
        orchestrator.load_checkpoint(str(checkpoint_path))
        assert orchestrator.current_iteration == 1


class TestDataLoaderIntegration:
    """Integration tests for data loader utilities."""

    def test_dummy_data_loader(self):
        """Test dummy data loader generates correct shapes."""
        from src.training.agent_trainer import DummyDataLoader

        loader = DummyDataLoader(
            batch_size=8,
            input_dim=64,
            output_dim=32,
            num_batches=5,
            device="cpu",
        )

        batches = list(loader)

        assert len(batches) == 5
        for inputs, targets in batches:
            assert inputs.shape[0] == 8
            assert inputs.shape[2] == 64
            assert targets.shape[0] == 8
            assert targets.shape[2] == 32

    def test_create_data_loader_from_empty_buffer(self):
        """Test data loader creation with empty buffer falls back to dummy."""
        from src.training.agent_trainer import create_data_loader_from_buffer

        loader = create_data_loader_from_buffer(
            replay_buffer=None,
            batch_size=8,
            input_dim=64,
            output_dim=32,
        )

        # Should get a DummyDataLoader
        batches = list(loader)
        assert len(batches) > 0


class TestEndToEndTraining:
    """End-to-end training tests."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_minimal_training_loop(self, system_config, initial_state_fn):
        """Test a minimal training loop runs without errors."""
        from src.training.unified_orchestrator import UnifiedTrainingOrchestrator

        orchestrator = UnifiedTrainingOrchestrator(
            config=system_config,
            initial_state_fn=initial_state_fn,
            board_size=19,
        )

        # Mock self-play to avoid long running games
        with patch.object(
            orchestrator.self_play_collector,
            "play_game",
            new_callable=AsyncMock,
        ) as mock_play:
            mock_example = MagicMock()
            mock_example.state = torch.randn(19, 19)
            mock_example.policy_target = torch.rand(10)
            mock_example.value_target = torch.tensor([0.5])
            mock_play.return_value = [mock_example] * 10

            # Run a single iteration
            metrics = await orchestrator.train_iteration(iteration=1)

            assert metrics is not None
            assert "policy_loss" in metrics or "games_generated" in metrics
