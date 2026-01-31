"""
Chess Observability Package.

Provides logging, metrics, and debugging infrastructure
for chess gameplay and verification.
"""

from src.games.chess.observability.decorators import (
    traced_move_selection,
    verified_game_play,
    with_verification_context,
)
from src.games.chess.observability.logger import (
    ChessVerificationLogger,
    get_chess_logger,
)
from src.games.chess.observability.metrics import (
    ChessMetricsCollector,
    ChessVerificationMetrics,
)

__all__ = [
    # Logger
    "ChessVerificationLogger",
    "get_chess_logger",
    # Metrics
    "ChessMetricsCollector",
    "ChessVerificationMetrics",
    # Decorators
    "traced_move_selection",
    "verified_game_play",
    "with_verification_context",
]
