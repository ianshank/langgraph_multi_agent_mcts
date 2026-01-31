"""
Chess Constants and Shared Values.

Centralizes all chess-related constants to avoid duplication and ensure
consistency across the codebase. All values are configurable via settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import Settings

# Standard starting position FEN
STARTING_FEN: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Default FEN truncation length for logging (overridable via settings)
DEFAULT_FEN_LOG_TRUNCATE_LENGTH: int = 40

# Invalid pawn ranks (0-indexed: rank 0 = 1st rank, rank 7 = 8th rank)
INVALID_PAWN_RANKS: frozenset[int] = frozenset({0, 7})


def get_fen_truncate_length(settings: "Settings | None" = None) -> int:
    """Get the FEN truncation length from settings or default.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        FEN truncation length for logging
    """
    if settings is None:
        try:
            from src.config.settings import get_settings
            settings = get_settings()
        except Exception:
            return DEFAULT_FEN_LOG_TRUNCATE_LENGTH

    return getattr(settings, "CHESS_FEN_LOG_TRUNCATE_LENGTH", DEFAULT_FEN_LOG_TRUNCATE_LENGTH)


def truncate_fen(fen: str, length: int | None = None, settings: "Settings | None" = None) -> str:
    """Truncate FEN string for logging purposes.

    Args:
        fen: FEN string to truncate
        length: Optional explicit length. If not provided, uses settings.
        settings: Optional settings instance for length lookup.

    Returns:
        Truncated FEN with "..." suffix if truncated
    """
    if length is None:
        length = get_fen_truncate_length(settings)

    if len(fen) > length:
        return fen[:length] + "..."
    return fen


def get_piece_values(settings: "Settings | None" = None) -> dict[int, int]:
    """Get piece values from settings for material evaluation.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        Dictionary mapping chess piece types to centipawn values
    """
    import chess

    if settings is None:
        try:
            from src.config.settings import get_settings
            settings = get_settings()
        except Exception:
            # Return defaults if settings unavailable
            return {
                chess.PAWN: 100,
                chess.KNIGHT: 320,
                chess.BISHOP: 330,
                chess.ROOK: 500,
                chess.QUEEN: 900,
            }

    return {
        chess.PAWN: getattr(settings, "CHESS_PIECE_VALUE_PAWN", 100),
        chess.KNIGHT: getattr(settings, "CHESS_PIECE_VALUE_KNIGHT", 320),
        chess.BISHOP: getattr(settings, "CHESS_PIECE_VALUE_BISHOP", 330),
        chess.ROOK: getattr(settings, "CHESS_PIECE_VALUE_ROOK", 500),
        chess.QUEEN: getattr(settings, "CHESS_PIECE_VALUE_QUEEN", 900),
    }


def get_routing_scores(settings: "Settings | None" = None) -> dict[str, float]:
    """Get routing consistency scores from settings.

    Args:
        settings: Optional settings instance. If not provided, uses get_settings().

    Returns:
        Dictionary of routing score keys to values
    """
    if settings is None:
        try:
            from src.config.settings import get_settings
            settings = get_settings()
        except Exception:
            # Return defaults if settings unavailable
            return {
                "match": 1.0,
                "middlegame_fallback": 0.7,
                "phase_appropriate": 0.8,
                "phase_mismatch": 0.4,
                "default": 0.5,
            }

    return {
        "match": getattr(settings, "CHESS_ROUTING_SCORE_MATCH", 1.0),
        "middlegame_fallback": getattr(settings, "CHESS_ROUTING_SCORE_MIDDLEGAME_FALLBACK", 0.7),
        "phase_appropriate": getattr(settings, "CHESS_ROUTING_SCORE_PHASE_APPROPRIATE", 0.8),
        "phase_mismatch": getattr(settings, "CHESS_ROUTING_SCORE_PHASE_MISMATCH", 0.4),
        "default": getattr(settings, "CHESS_ROUTING_SCORE_DEFAULT", 0.5),
    }


__all__ = [
    "STARTING_FEN",
    "DEFAULT_FEN_LOG_TRUNCATE_LENGTH",
    "INVALID_PAWN_RANKS",
    "get_fen_truncate_length",
    "truncate_fen",
    "get_piece_values",
    "get_routing_scores",
]
