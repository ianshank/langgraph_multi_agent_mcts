"""
Chess Engines Package.

Provides adapters for external chess engines like Stockfish.
"""

from src.games.chess.engines.stockfish_adapter import (
    StockfishAdapter,
    StockfishAnalysis,
    StockfishConfig,
    create_stockfish_adapter,
)

__all__ = [
    "StockfishAdapter",
    "StockfishAnalysis",
    "StockfishConfig",
    "create_stockfish_adapter",
]
