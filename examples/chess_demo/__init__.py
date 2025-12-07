"""
Chess Demo Module.

Multi-agent ensemble chess demo with chessboard.js UI and learning dashboard.

Components:
- chess_state: Chess game state implementing MCTS GameState interface
- chess_network: Policy-value neural network for chess
- chess_ensemble: Multi-agent ensemble (HRM, TRM, MCTS, Symbolic)
- learning_dashboard: Comprehensive analytics dashboard
- app: Flask web application with chessboard.js frontend
"""

from .chess_state import ChessState, ChessConfig, CHESS_AVAILABLE
from .chess_ensemble import (
    ChessEnsemble,
    EnsembleConfig,
    AgentType,
    AgentResult,
    LearningRecord,
)
from .learning_dashboard import (
    LearningDashboard,
    DashboardConfig,
    create_dashboard,
)

__all__ = [
    # State
    "ChessState",
    "ChessConfig",
    "CHESS_AVAILABLE",
    # Ensemble
    "ChessEnsemble",
    "EnsembleConfig",
    "AgentType",
    "AgentResult",
    "LearningRecord",
    # Dashboard
    "LearningDashboard",
    "DashboardConfig",
    "create_dashboard",
]
