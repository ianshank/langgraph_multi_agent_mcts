"""
Chess Play Verification Package.

Provides comprehensive verification components for chess gameplay,
including move validation, game verification, and ensemble consistency checking.

Components:
- MoveValidator: Validates chess moves including edge cases
- ChessGameVerifier: Verifies complete games and move sequences
- EnsembleConsistencyChecker: Checks consistency between sub-agents
- ChessVerificationFactory: Factory for creating verification components
"""

from src.games.chess.verification.ensemble_checker import (
    EnsembleCheckerConfig,
    EnsembleConsistencyChecker,
    create_ensemble_checker,
)
from src.games.chess.verification.factory import (
    ChessVerificationFactory,
    VerificationBuilder,
    create_verification_factory,
)
from src.games.chess.verification.game_verifier import (
    ChessGameVerifier,
    GameVerifierConfig,
    create_game_verifier,
)
from src.games.chess.verification.move_validator import (
    MoveValidator,
    MoveValidatorConfig,
    create_move_validator,
)
from src.games.chess.verification.protocols import (
    ChessGameVerifierProtocol,
    EnsembleConsistencyCheckerProtocol,
    MoveValidatorProtocol,
    SubAgentVerifierProtocol,
)
from src.games.chess.verification.types import (
    BatchVerificationResult,
    EnsembleConsistencyResult,
    GameResult,
    GameVerificationResult,
    MoveSequenceResult,
    MoveType,
    MoveValidationResult,
    PositionVerificationResult,
    VerificationIssue,
    VerificationSeverity,
)

__all__ = [
    # Protocols
    "MoveValidatorProtocol",
    "ChessGameVerifierProtocol",
    "EnsembleConsistencyCheckerProtocol",
    "SubAgentVerifierProtocol",  # Reserved for future use
    # Components
    "MoveValidator",
    "MoveValidatorConfig",
    "ChessGameVerifier",
    "GameVerifierConfig",
    "EnsembleConsistencyChecker",
    "EnsembleCheckerConfig",
    # Factory
    "ChessVerificationFactory",
    "VerificationBuilder",
    # Factory functions
    "create_move_validator",
    "create_game_verifier",
    "create_ensemble_checker",
    "create_verification_factory",
    # Types
    "MoveValidationResult",
    "MoveType",
    "GameVerificationResult",
    "PositionVerificationResult",
    "MoveSequenceResult",
    "EnsembleConsistencyResult",
    "BatchVerificationResult",
    "GameResult",
    "VerificationIssue",
    "VerificationSeverity",
]
