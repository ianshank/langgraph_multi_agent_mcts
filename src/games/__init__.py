"""
Games module for domain-specific implementations.

This module contains game implementations that follow the GameState interface
for use with the Neural MCTS and ensemble agent framework.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.games.chess import ChessConfig, ChessGameState

__all__ = ["ChessConfig", "ChessGameState"]
