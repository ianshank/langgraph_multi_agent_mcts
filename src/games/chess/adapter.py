"""
Chess Adapter for MCTS Integration.

This module adapts the generic MCTS engine to work specifically with the Chess domain,
providing a simplified interface for the API to interact with.
"""

from typing import Any, Optional
import torch
from pathlib import Path

from src.framework.mcts.neural_mcts import NeuralMCTS
from src.framework.mcts.config import MCTSConfig
from src.games.chess.state import ChessGameState, create_state_from_fen
from src.training.system_config import SystemConfig
from src.observability.logging import get_structured_logger
from src.games.chess.training import ChessNeuralNetwork

class ChessAdapter:
    """
    Adapter to run MCTS on Chess games using the NeuralMCTS engine.
    """

    def __init__(self, config: SystemConfig, model_path: Optional[str] = None):
        self._config = config
        self._logger = get_structured_logger(__name__)
        
        # Initialize Neural Network
        self._device = config.device
        self._network = ChessNeuralNetwork(config.neural_net)
        self._network.to(self._device)
        self._network.eval()

        if model_path:
            self._load_model(model_path)
        
        # Initialize MCTS Engine
        # We reuse the specific MCTS config for chess if available, or system default
        self._mcts_config = config.mcts
        
        # MCTS instance is created per-search or persisted? 
        # For AlphaZero, MCTS tree is often persistent across moves in a game, but for an API we might reconstruct.
        # For this MVP, we will create a fresh search for each request or allow passing a tree if stateful.
        self._mcts: Optional[NeuralMCTS] = None

    def _load_model(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self._device)
            self._network.load_state_dict(state_dict)
            self._logger.info(f"Loaded chess model from {path}")
        except Exception as e:
            self._logger.error(f"Failed to load model {path}: {e}")
            raise

    def search(self, fen: str, simulations: Optional[int] = None) -> dict[str, Any]:
        """
        Run MCTS search on a given FEN position.

        Args:
            fen: The board position in FEN notation.
            simulations: Overide default simulation count.

        Returns:
            Dictionary containing best move, value, and search stats.
        """
        root_state = create_state_from_fen(fen)
        
        # Create MCTS instance
        num_sims = simulations or self._mcts_config.num_simulations
        
        # We need to construct a lightweight config copy if we override sims
        # (Assuming MCTSConfig is a dataclass we can copy/update)
        
        mcts = NeuralMCTS(
            config=self._mcts_config,
            model=self._network,
            device=self._device
        )
        
        self._logger.info(f"Starting search on {fen} with {num_sims} simulations")
        
        # Run search
        root_node = mcts.search(root_state, num_simulations=num_sims)
        
        # Extract best action (most visited)
        best_action = mcts.get_best_action(root_node)
        
        # Extract principal variation (if implemented) or stats
        # For visualization, we might want the policy distribution at the root
        policy_dist = root_node.visit_counts
        
        # Calculate root value (average value of root node)
        root_value = root_node.total_value / max(1, root_node.visit_count)

        return {
            "best_move": best_action,
            "root_value": float(root_value),
            "simulations": root_node.visit_count,
            "policy": {k: v for k, v in policy_dist.items()}, # UCI -> count
            "nodes_explored": len(mcts.transposition_table) # Approximate
        }
        
    def get_legal_moves(self, fen: str) -> list[str]:
        """Helper to get legal moves for UI validation."""
        state = create_state_from_fen(fen)
        return state.get_legal_actions()
