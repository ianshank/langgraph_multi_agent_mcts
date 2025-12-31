"""
Integration tests for AlphaZero-style training pipeline.

Tests the complete integration of:
- Game environments (Go, Gomoku, Chess)
- Batch MCTS
- Distributed self-play pipeline
- Elo rating evaluation
- Checkpoint management
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.games.gomoku import GomokuGameState, create_gomoku_state
from src.games.go import GoGameState, create_go_state
from src.games.base import GameRegistry, PlayerColor
from src.training.elo_rating import EloRatingSystem
from src.training.replay_buffer import Experience, PrioritizedReplayBuffer
from src.framework.mcts.batch_mcts import BatchMCTS, BatchMCTSConfig


class SimplePolicyValueNet(nn.Module):
    """Simple network for testing."""

    def __init__(self, input_channels: int, action_size: int, board_size: int):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(32 * board_size * board_size, action_size)
        self.fc_value = nn.Linear(32 * board_size * board_size, 1)
        self.board_size = board_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.relu(self.conv(x))
        x = x.view(batch_size, -1)
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value


class TestGameRegistryIntegration:
    """Integration tests for game registry."""

    def test_registered_games(self):
        """Test that all games are properly registered."""
        games = GameRegistry.list_games()
        assert "gomoku" in games
        assert "go" in games

    def test_create_gomoku(self):
        """Test creating Gomoku via registry."""
        state = GameRegistry.create("gomoku")
        assert state is not None
        assert not state.is_terminal()

    def test_create_go(self):
        """Test creating Go via registry."""
        state = GameRegistry.create("go")
        assert state is not None
        assert not state.is_terminal()


class TestGameEnvironmentIntegration:
    """Integration tests for game environments."""

    @pytest.mark.parametrize("game_name", ["gomoku", "go"])
    def test_full_game_simulation(self, game_name):
        """Test simulating a complete game."""
        state = GameRegistry.create(game_name)

        move_count = 0
        max_moves = 100

        while not state.is_terminal() and move_count < max_moves:
            actions = state.get_legal_actions()
            if not actions:
                break

            # Random action selection
            action = np.random.choice(actions)
            state = state.apply_action(action)
            move_count += 1

        # Game should have progressed
        assert state.metadata.move_number > 0

    @pytest.mark.parametrize("game_name", ["gomoku", "go"])
    def test_tensor_shapes(self, game_name):
        """Test tensor generation consistency."""
        state = GameRegistry.create(game_name)

        tensor1 = state.to_tensor()
        assert tensor1.dim() == 3  # C, H, W

        # After move, tensor shape should be same
        action = state.get_legal_actions()[0]
        new_state = state.apply_action(action)
        tensor2 = new_state.to_tensor()

        assert tensor1.shape == tensor2.shape

    @pytest.mark.parametrize("game_name", ["gomoku", "go"])
    def test_hash_consistency(self, game_name):
        """Test that same positions have same hash."""
        state1 = GameRegistry.create(game_name)
        state2 = GameRegistry.create(game_name)

        assert state1.get_state_hash() == state2.get_state_hash()

        # Apply same action to both
        action = state1.get_legal_actions()[0]
        state1 = state1.apply_action(action)
        state2 = state2.apply_action(action)

        assert state1.get_state_hash() == state2.get_state_hash()


class TestReplayBufferIntegration:
    """Integration tests for replay buffer with game data."""

    def test_store_game_experiences(self):
        """Test storing game experiences in buffer."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        state = create_gomoku_state(board_size=5, win_length=3)

        # Simulate a short game
        for _ in range(10):
            if state.is_terminal():
                break

            tensor = state.to_tensor()
            actions = state.get_legal_actions()
            policy = np.zeros(state.config.action_size)
            policy[actions] = 1.0 / len(actions)

            exp = Experience(state=tensor, policy=policy, value=0.0)
            buffer.add(exp)

            state = state.apply_action(np.random.choice(actions))

        assert len(buffer) > 0

    def test_sample_batch(self):
        """Test sampling batch from buffer."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        state = create_gomoku_state(board_size=5, win_length=3)

        # Add experiences
        for _ in range(100):
            if state.is_terminal():
                state = create_gomoku_state(board_size=5, win_length=3)

            tensor = state.to_tensor()
            actions = state.get_legal_actions()
            policy = np.zeros(state.config.action_size)
            policy[actions] = 1.0 / len(actions)

            exp = Experience(state=tensor, policy=policy, value=np.random.randn())
            buffer.add(exp)

            state = state.apply_action(np.random.choice(actions))

        # Sample batch
        assert buffer.is_ready(32)
        experiences, indices, weights = buffer.sample(32)

        assert len(experiences) == 32
        assert len(indices) == 32
        assert len(weights) == 32


class TestBatchMCTSIntegration:
    """Integration tests for batch MCTS."""

    @pytest.fixture
    def gomoku_setup(self):
        """Create Gomoku setup for testing."""
        state = create_gomoku_state(board_size=5, win_length=3)
        config = BatchMCTSConfig(
            num_simulations=10,
            batch_size=4,
            c_puct=1.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        network = SimplePolicyValueNet(
            input_channels=9,  # 4*2+1
            action_size=25,
            board_size=5,
        )
        return state, config, network

    @pytest.mark.asyncio
    async def test_batch_mcts_search(self, gomoku_setup):
        """Test batch MCTS search."""
        state, config, network = gomoku_setup

        mcts = BatchMCTS(
            policy_value_network=network,
            config=config,
            state_to_tensor_fn=lambda s: s.to_tensor(),
            get_legal_actions_fn=lambda s: list(s.get_legal_actions()),
            apply_action_fn=lambda s, a: s.apply_action(a),
            is_terminal_fn=lambda s: s.is_terminal(),
            get_reward_fn=lambda s: s.get_reward(PlayerColor.BLACK),
            get_state_hash_fn=lambda s: s.get_state_hash(),
            action_size=25,
            device="cpu",
        )

        action_probs, root = await mcts.search(state, num_simulations=10)

        assert action_probs.shape == (25,)
        assert np.abs(action_probs.sum() - 1.0) < 0.01
        assert root.visit_count > 0

    @pytest.mark.asyncio
    async def test_batch_mcts_action_selection(self, gomoku_setup):
        """Test action selection from MCTS."""
        state, config, network = gomoku_setup

        mcts = BatchMCTS(
            policy_value_network=network,
            config=config,
            state_to_tensor_fn=lambda s: s.to_tensor(),
            get_legal_actions_fn=lambda s: list(s.get_legal_actions()),
            apply_action_fn=lambda s, a: s.apply_action(a),
            is_terminal_fn=lambda s: s.is_terminal(),
            get_reward_fn=lambda s: s.get_reward(PlayerColor.BLACK),
            get_state_hash_fn=lambda s: s.get_state_hash(),
            action_size=25,
            device="cpu",
        )

        action_probs, _ = await mcts.search(state, num_simulations=10)
        action = mcts.select_action(action_probs, temperature=1.0)

        assert 0 <= action < 25
        assert action in state.get_legal_actions()


class TestEloIntegration:
    """Integration tests for Elo rating with games."""

    def test_tournament_simulation(self):
        """Test simulating a tournament with Elo tracking."""
        elo = EloRatingSystem()

        # Simulate matches
        players = ["agent_v1", "agent_v2", "agent_v3"]

        for _ in range(10):
            p1, p2 = np.random.choice(players, 2, replace=False)
            # Random result
            score = np.random.choice([1.0, 0.5, 0.0])
            elo.update_ratings(p1, p2, score)

        # Check ratings exist
        for player in players:
            rating = elo.get_rating(player)
            assert rating > 0

        # Check leaderboard
        leaderboard = elo.get_leaderboard()
        assert len(leaderboard) == 3


class TestCheckpointIntegration:
    """Integration tests for checkpoint management."""

    def test_save_load_training_state(self):
        """Test saving and loading training state."""
        from src.training.distributed_pipeline import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=5)

            # Create sample data
            network = SimplePolicyValueNet(9, 25, 5)
            checkpoint_data = {
                "iteration": 10,
                "network_state": network.state_dict(),
                "games_generated": 1000,
                "best_win_rate": 0.65,
            }

            # Save
            path = manager.save(checkpoint_data, iteration=10)
            assert path.exists()

            # Load
            loaded = manager.load(path)
            assert loaded["iteration"] == 10
            assert loaded["games_generated"] == 1000

    def test_checkpoint_rotation(self):
        """Test old checkpoint rotation."""
        from src.training.distributed_pipeline import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_checkpoints=3)

            # Save more than max
            for i in range(5):
                manager.save({"iteration": i}, iteration=i)

            # Should only keep last 3
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3


class TestEndToEndTraining:
    """End-to-end training tests (lightweight)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mini_training_loop(self):
        """Test a minimal training loop."""
        from src.training.replay_buffer import Experience
        from src.training.distributed_pipeline import NetworkLearner, DistributedConfig

        # Setup
        config = DistributedConfig(
            num_actors=1,
            games_per_actor=2,
            batch_size=8,
            num_training_steps=2,
            use_mixed_precision=False,
        )

        network = SimplePolicyValueNet(9, 25, 5)
        learner = NetworkLearner(network, config, device="cpu")
        buffer = PrioritizedReplayBuffer(capacity=1000)

        # Generate some fake data
        for _ in range(50):
            state_tensor = torch.randn(9, 5, 5)
            policy = np.random.randn(25)
            policy = np.exp(policy) / np.exp(policy).sum()
            value = np.random.randn()

            exp = Experience(state=state_tensor, policy=policy, value=value)
            buffer.add(exp)

        # Training step
        assert buffer.is_ready(8)
        experiences, _, _ = buffer.sample(8)

        states = torch.stack([e.state for e in experiences])
        policies = torch.tensor(np.array([e.policy for e in experiences]), dtype=torch.float32)
        values = torch.tensor([e.value for e in experiences], dtype=torch.float32)

        metrics = learner.train_step(states, policies, values)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert metrics["total_loss"] > 0
