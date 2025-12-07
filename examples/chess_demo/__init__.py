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

from .chess_ensemble import (
    AgentResult,
    AgentType,
    ChessEnsemble,
    EnsembleConfig,
    LearningRecord,
)
from .chess_state import CHESS_AVAILABLE, ChessConfig, ChessState
from .learning_dashboard import (
    DashboardConfig,
    LearningDashboard,
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
