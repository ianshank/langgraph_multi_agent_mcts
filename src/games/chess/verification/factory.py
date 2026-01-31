"""
Chess Verification Factory.

Provides factory classes for creating verification components
in a consistent, testable, and modular way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging

    from src.games.chess.ensemble_agent import ChessEnsembleAgent

from src.config.settings import LogLevel, Settings, get_settings
from src.games.chess.action_space import ChessActionEncoder
from src.games.chess.config import ChessActionSpaceConfig, ChessConfig
from src.games.chess.constants import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD,
)
from src.games.chess.verification.ensemble_checker import (
    EnsembleCheckerConfig,
    EnsembleConsistencyChecker,
)
from src.games.chess.verification.game_verifier import (
    ChessGameVerifier,
    GameVerifierConfig,
)
from src.games.chess.verification.move_validator import (
    MoveValidator,
    MoveValidatorConfig,
)
from src.observability.logging import StructuredLogger, get_structured_logger


class ChessVerificationFactory:
    """Factory for creating chess verification components.

    Provides a unified interface for creating verification components
    with consistent configuration and dependency injection.

    Example:
        >>> factory = ChessVerificationFactory()
        >>> validator = factory.create_move_validator()
        >>> verifier = factory.create_game_verifier()
        >>> checker = factory.create_ensemble_checker(ensemble_agent)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        chess_config: ChessConfig | None = None,
        logger: StructuredLogger | logging.Logger | None = None,
    ) -> None:
        """Initialize the verification factory.

        Args:
            settings: Optional settings instance (uses defaults if None)
            chess_config: Optional chess configuration
            logger: Optional logger instance
        """
        self._settings = settings or get_settings()
        self._chess_config = chess_config
        self._logger = logger or get_structured_logger("chess.verification.factory")

        # Cache for reusable components
        self._action_encoder: ChessActionEncoder | None = None

    @property
    def settings(self) -> Settings:
        """Get the settings instance."""
        return self._settings

    @property
    def chess_config(self) -> ChessConfig | None:
        """Get the chess configuration."""
        return self._chess_config

    def create_move_validator(
        self,
        action_config: ChessActionSpaceConfig | None = None,
        validator_config: MoveValidatorConfig | None = None,
        reuse_encoder: bool = True,
    ) -> MoveValidator:
        """Create a MoveValidator instance.

        Args:
            action_config: Optional action space configuration
            validator_config: Optional validator configuration
            reuse_encoder: Whether to reuse cached action encoder

        Returns:
            Configured MoveValidator instance
        """
        # Get action config from chess_config if available
        if action_config is None and self._chess_config is not None:
            action_config = self._chess_config.action_space

        action_config = action_config or ChessActionSpaceConfig()

        # Get or create action encoder
        if reuse_encoder and self._action_encoder is not None:
            encoder = self._action_encoder
        else:
            encoder = ChessActionEncoder(action_config)
            if reuse_encoder:
                self._action_encoder = encoder

        # Create validator config with settings
        if validator_config is None:
            validator_config = MoveValidatorConfig(
                validate_encoding=True,
                validate_legality=True,
                log_validations=self._settings.LOG_LEVEL == LogLevel.DEBUG,
            )

        self._logger.debug(
            "Creating MoveValidator",
            extra={"action_size": action_config.total_actions},
        )

        return MoveValidator(
            action_encoder=encoder,
            config=validator_config,
        )

    def create_game_verifier(
        self,
        verifier_config: GameVerifierConfig | None = None,
        move_validator: MoveValidator | None = None,
    ) -> ChessGameVerifier:
        """Create a ChessGameVerifier instance.

        Args:
            verifier_config: Optional verifier configuration
            move_validator: Optional move validator (created if None)

        Returns:
            Configured ChessGameVerifier instance
        """
        # Create move validator if not provided
        if move_validator is None:
            move_validator = self.create_move_validator()

        # Create verifier config with settings
        if verifier_config is None:
            verifier_config = GameVerifierConfig(
                verify_encoding=True,
                verify_terminal_state=True,
                verify_result=True,
                log_verifications=True,
            )

        self._logger.debug("Creating ChessGameVerifier")

        return ChessGameVerifier(
            move_validator=move_validator,
            config=verifier_config,
        )

    def create_ensemble_checker(
        self,
        ensemble_agent: ChessEnsembleAgent | None = None,
        checker_config: EnsembleCheckerConfig | None = None,
    ) -> EnsembleConsistencyChecker:
        """Create an EnsembleConsistencyChecker instance.

        Args:
            ensemble_agent: Optional ensemble agent (can be set later)
            checker_config: Optional checker configuration

        Returns:
            Configured EnsembleConsistencyChecker instance
        """
        # Create checker config with settings
        if checker_config is None:
            checker_config = EnsembleCheckerConfig(
                log_checks=True,
            )

        self._logger.debug("Creating EnsembleConsistencyChecker")

        return EnsembleConsistencyChecker(
            ensemble_agent=ensemble_agent,
            config=checker_config,
        )

    def create_all_verifiers(
        self,
        ensemble_agent: ChessEnsembleAgent | None = None,
    ) -> dict[str, Any]:
        """Create all verification components.

        Args:
            ensemble_agent: Optional ensemble agent

        Returns:
            Dictionary containing all verification components
        """
        move_validator = self.create_move_validator()
        game_verifier = self.create_game_verifier(move_validator=move_validator)
        ensemble_checker = self.create_ensemble_checker(ensemble_agent=ensemble_agent)

        return {
            "move_validator": move_validator,
            "game_verifier": game_verifier,
            "ensemble_checker": ensemble_checker,
        }


class VerificationBuilder:
    """Fluent builder for verification components.

    Provides a fluent API for building verification components
    with custom configurations.

    Example:
        >>> verifier = (
        ...     VerificationBuilder()
        ...     .with_encoding_validation(True)
        ...     .with_stop_on_first_error(True)
        ...     .build_game_verifier()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._settings: Settings | None = None
        self._chess_config: ChessConfig | None = None
        self._validate_encoding: bool = True
        self._validate_legality: bool = True
        self._verify_terminal: bool = True
        self._verify_result: bool = True
        self._stop_on_first_error: bool = False
        self._log_validations: bool = False
        self._agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD
        self._divergence_threshold: float = DEFAULT_CONFIDENCE_DIVERGENCE_THRESHOLD

    def with_settings(self, settings: Settings) -> VerificationBuilder:
        """Set the settings instance.

        Args:
            settings: Settings instance

        Returns:
            Self for chaining
        """
        self._settings = settings
        return self

    def with_chess_config(self, config: ChessConfig) -> VerificationBuilder:
        """Set the chess configuration.

        Args:
            config: Chess configuration

        Returns:
            Self for chaining
        """
        self._chess_config = config
        return self

    def with_encoding_validation(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable encoding validation.

        Args:
            enabled: Whether to enable encoding validation

        Returns:
            Self for chaining
        """
        self._validate_encoding = enabled
        return self

    def with_legality_validation(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable legality validation.

        Args:
            enabled: Whether to enable legality validation

        Returns:
            Self for chaining
        """
        self._validate_legality = enabled
        return self

    def with_terminal_verification(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable terminal state verification.

        Args:
            enabled: Whether to enable terminal verification

        Returns:
            Self for chaining
        """
        self._verify_terminal = enabled
        return self

    def with_result_verification(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable result verification.

        Args:
            enabled: Whether to enable result verification

        Returns:
            Self for chaining
        """
        self._verify_result = enabled
        return self

    def with_stop_on_first_error(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable stopping on first error.

        Args:
            enabled: Whether to stop on first error

        Returns:
            Self for chaining
        """
        self._stop_on_first_error = enabled
        return self

    def with_logging(self, enabled: bool) -> VerificationBuilder:
        """Enable or disable validation logging.

        Args:
            enabled: Whether to enable logging

        Returns:
            Self for chaining
        """
        self._log_validations = enabled
        return self

    def with_agreement_threshold(self, threshold: float) -> VerificationBuilder:
        """Set the agreement threshold for ensemble checking.

        Args:
            threshold: Agreement threshold (0-1)

        Returns:
            Self for chaining
        """
        self._agreement_threshold = threshold
        return self

    def with_divergence_threshold(self, threshold: float) -> VerificationBuilder:
        """Set the divergence threshold for ensemble checking.

        Args:
            threshold: Divergence threshold (0-1)

        Returns:
            Self for chaining
        """
        self._divergence_threshold = threshold
        return self

    def build_move_validator(self) -> MoveValidator:
        """Build a MoveValidator with current configuration.

        Returns:
            Configured MoveValidator instance
        """
        config = MoveValidatorConfig(
            validate_encoding=self._validate_encoding,
            validate_legality=self._validate_legality,
            log_validations=self._log_validations,
        )

        action_config = None
        if self._chess_config is not None:
            action_config = self._chess_config.action_space

        encoder = ChessActionEncoder(action_config or ChessActionSpaceConfig())

        return MoveValidator(
            action_encoder=encoder,
            config=config,
        )

    def build_game_verifier(self) -> ChessGameVerifier:
        """Build a ChessGameVerifier with current configuration.

        Returns:
            Configured ChessGameVerifier instance
        """
        move_validator_config = MoveValidatorConfig(
            validate_encoding=self._validate_encoding,
            validate_legality=self._validate_legality,
            log_validations=self._log_validations,
        )

        verifier_config = GameVerifierConfig(
            verify_encoding=self._validate_encoding,
            verify_terminal_state=self._verify_terminal,
            verify_result=self._verify_result,
            stop_on_first_error=self._stop_on_first_error,
            log_verifications=self._log_validations,
            move_validator_config=move_validator_config,
        )

        return ChessGameVerifier(config=verifier_config)

    def build_ensemble_checker(
        self,
        ensemble_agent: ChessEnsembleAgent | None = None,
    ) -> EnsembleConsistencyChecker:
        """Build an EnsembleConsistencyChecker with current configuration.

        Args:
            ensemble_agent: Optional ensemble agent

        Returns:
            Configured EnsembleConsistencyChecker instance
        """
        config = EnsembleCheckerConfig(
            agreement_threshold=self._agreement_threshold,
            confidence_divergence_threshold=self._divergence_threshold,
            log_checks=self._log_validations,
        )

        return EnsembleConsistencyChecker(
            ensemble_agent=ensemble_agent,
            config=config,
        )


def create_verification_factory(
    settings: Settings | None = None,
    chess_config: ChessConfig | None = None,
) -> ChessVerificationFactory:
    """Factory function to create a ChessVerificationFactory.

    Args:
        settings: Optional settings instance
        chess_config: Optional chess configuration

    Returns:
        Configured ChessVerificationFactory instance
    """
    return ChessVerificationFactory(
        settings=settings,
        chess_config=chess_config,
    )
