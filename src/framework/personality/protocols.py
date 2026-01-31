"""
Protocol interfaces for personality modules.

Follows Interface Segregation Principle (ISP) by providing
granular interfaces for specific capabilities.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.framework.agents.base import AgentContext, AgentResult


@runtime_checkable
class PersonalityModuleProtocol(Protocol):
    """Base protocol for all personality modules.

    All personality modules must implement pre/post process hooks
    and configuration influence methods.
    """

    @property
    def module_name(self) -> str:
        """Module identifier string."""
        ...

    @property
    def trait_value(self) -> float:
        """Current trait value [0.0, 1.0]."""
        ...

    async def pre_process_hook(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """Modify context before agent processing.

        Args:
            context: Agent context to modify

        Returns:
            Modified agent context
        """
        ...

    async def post_process_hook(
        self,
        context: AgentContext,
        result: AgentResult,
    ) -> AgentResult:
        """Modify result after agent processing.

        Args:
            context: Original agent context
            result: Agent result to modify

        Returns:
            Modified agent result
        """
        ...

    def influence_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Modify agent configuration based on trait.

        Args:
            config: Configuration dictionary

        Returns:
            Modified configuration
        """
        ...


@runtime_checkable
class MCTSInfluencer(Protocol):
    """Protocol for modules that influence MCTS behavior.

    Use this interface for modules that need to modify
    MCTS exploration/exploitation parameters.
    """

    def get_exploration_bonus(
        self,
        base_weight: float,
        iteration: int,
        state_novelty: float,
    ) -> float:
        """Calculate exploration bonus based on trait.

        Args:
            base_weight: Base UCB exploration weight
            iteration: Current MCTS iteration
            state_novelty: Novelty score of current state [0.0, 1.0]

        Returns:
            Modified exploration weight
        """
        ...

    def modify_rollout_policy(
        self,
        policy: str,
        context: dict[str, Any],
    ) -> str:
        """Modify MCTS rollout policy based on trait.

        Args:
            policy: Current policy name (random, greedy, hybrid)
            context: Decision context

        Returns:
            Modified policy name
        """
        ...

    def compute_intrinsic_reward(
        self,
        state_hash: str,
        action: str,
        uncertainty: float,
    ) -> float:
        """Compute intrinsic reward for curiosity-driven exploration.

        Args:
            state_hash: Hash of current state
            action: Action being evaluated
            uncertainty: Model uncertainty for this state-action

        Returns:
            Intrinsic reward value
        """
        ...


@runtime_checkable
class PromptAugmenter(Protocol):
    """Protocol for modules that augment LLM prompts.

    Use this interface for modules that add personality-specific
    instructions or context to prompts.
    """

    async def augment_prompt(
        self,
        base_prompt: str,
        context: AgentContext,
    ) -> str:
        """Add trait-specific prompt instructions.

        Args:
            base_prompt: Original prompt text
            context: Agent context with additional info

        Returns:
            Augmented prompt text
        """
        ...

    def get_system_message_suffix(self) -> str:
        """Get suffix to add to system message.

        Returns:
            System message suffix string
        """
        ...


@runtime_checkable
class ConfidenceCalibrator(Protocol):
    """Protocol for modules that calibrate confidence scores.

    Use this interface for modules that adjust confidence
    based on personality traits.
    """

    def calibrate_confidence(
        self,
        raw_confidence: float,
        context: dict[str, Any],
    ) -> float:
        """Adjust confidence based on trait.

        Args:
            raw_confidence: Raw confidence score [0.0, 1.0]
            context: Decision context

        Returns:
            Calibrated confidence score [0.0, 1.0]
        """
        ...

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold for this trait.

        Returns:
            Minimum acceptable confidence [0.0, 1.0]
        """
        ...


@runtime_checkable
class ExplainabilityProvider(Protocol):
    """Protocol for modules that provide explanations.

    Use this interface for modules that generate
    human-readable decision explanations.
    """

    async def generate_explanation(
        self,
        decision: dict[str, Any],
        context: AgentContext,
        verbosity: str,
    ) -> str:
        """Generate human-readable explanation.

        Args:
            decision: Decision data to explain
            context: Agent context
            verbosity: Detail level (brief, moderate, detailed)

        Returns:
            Human-readable explanation string
        """
        ...

    def get_key_factors(
        self,
        decision: dict[str, Any],
    ) -> list[tuple[str, float, str]]:
        """Extract key factors from decision.

        Args:
            decision: Decision data

        Returns:
            List of (factor_name, factor_weight, description) tuples
        """
        ...


@runtime_checkable
class GoalTracker(Protocol):
    """Protocol for modules that track goals and commitments.

    Use this interface for loyalty and aspiration modules.
    """

    def commit_to_goal(
        self,
        goal: str,
        priority: float,
    ) -> None:
        """Register commitment to a goal.

        Args:
            goal: Goal identifier
            priority: Goal priority [0.0, 1.0]
        """
        ...

    def evaluate_goal_alignment(
        self,
        action: str,
        current_goals: list[str],
    ) -> float:
        """Evaluate action alignment with committed goals.

        Args:
            action: Action being evaluated
            current_goals: Currently active goals

        Returns:
            Alignment score [0.0, 1.0]
        """
        ...

    def should_persist_on_goal(
        self,
        goal: str,
        difficulty: float,
        attempts: int,
    ) -> tuple[bool, str]:
        """Determine if agent should persist despite difficulty.

        Args:
            goal: Goal identifier
            difficulty: Current difficulty score [0.0, 1.0]
            attempts: Number of attempts so far

        Returns:
            Tuple of (should_persist, explanation)
        """
        ...


@runtime_checkable
class EthicalEvaluator(Protocol):
    """Protocol for ethical evaluation modules.

    Use this interface for modules that evaluate
    ethical implications of actions.
    """

    def evaluate_action_ethics(
        self,
        action: str,
        context: dict[str, Any],
        consequences: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate ethical implications of an action.

        Args:
            action: Action being evaluated
            context: Decision context
            consequences: Predicted consequences

        Returns:
            Tuple of (ethical_score [0.0, 1.0], detailed_assessment)
        """
        ...

    def check_ethical_constraints(
        self,
        action: str,
    ) -> tuple[bool, str | None]:
        """Check if action violates ethical constraints.

        Args:
            action: Action being evaluated

        Returns:
            Tuple of (is_allowed, violation_reason if not allowed)
        """
        ...

    def resolve_ethical_dilemma(
        self,
        options: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> tuple[int, str]:
        """Resolve conflict between competing ethical options.

        Args:
            options: List of option dictionaries
            context: Decision context

        Returns:
            Tuple of (selected_index, reasoning)
        """
        ...
