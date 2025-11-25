"""
Abstract base class for Neural Meta-Controllers.

Provides the foundation for neural network-based meta-controllers that
dynamically select which agent (HRM, TRM, or MCTS) should handle a query.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetaControllerFeatures:
    """
    Features extracted from the current agent state for meta-controller prediction.

    These features capture the current state of the multi-agent system,
    including confidence scores from different agents and contextual information.
    """

    hrm_confidence: float
    """Confidence score from the HRM (Human Response Model) agent."""

    trm_confidence: float
    """Confidence score from the TRM (Task Response Model) agent."""

    mcts_value: float
    """Value estimate from the MCTS (Monte Carlo Tree Search) process."""

    consensus_score: float
    """Agreement score between different agents."""

    last_agent: str
    """Name of the last agent used ('hrm', 'trm', 'mcts', or 'none')."""

    iteration: int
    """Current iteration number in the reasoning process."""

    query_length: int
    """Length of the input query in characters."""

    has_rag_context: bool
    """Whether RAG (Retrieval-Augmented Generation) context is available."""

    rag_relevance_score: float = 0.0
    """Relevance score of RAG context to the query (0.0 to 1.0)."""

    is_technical_query: bool = False
    """Whether the query is technical in nature."""


@dataclass
class MetaControllerPrediction:
    """
    Prediction output from the meta-controller.

    Contains the selected agent and associated confidence/probability information.
    """

    agent: str
    """Name of the selected agent ('hrm', 'trm', or 'mcts')."""

    confidence: float
    """Confidence score for the prediction (0.0 to 1.0)."""

    probabilities: dict[str, float] = field(default_factory=dict)
    """Probability distribution over all possible agents."""


class AbstractMetaController(ABC):
    """
    Abstract base class for neural meta-controllers.

    This class defines the interface that all meta-controller implementations
    must follow. Meta-controllers are responsible for deciding which agent
    should handle a given query based on the current system state.

    Attributes:
        AGENT_NAMES: List of valid agent names that can be selected.
        name: Name of this meta-controller instance.
        seed: Random seed for reproducibility.
    """

    AGENT_NAMES = ["hrm", "trm", "mcts"]

    def __init__(self, name: str, seed: int = 42) -> None:
        """
        Initialize the meta-controller.

        Args:
            name: Name identifier for this meta-controller instance.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self.name = name
        self.seed = seed

    @abstractmethod
    def predict(self, features: MetaControllerFeatures) -> MetaControllerPrediction:
        """
        Predict which agent should handle the current query.

        Args:
            features: Features extracted from the current agent state.

        Returns:
            Prediction containing the selected agent and confidence scores.
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model file or directory.
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the current model to disk.

        Args:
            path: Path where the model should be saved.
        """
        pass

    def extract_features(self, state: dict[str, Any]) -> MetaControllerFeatures:
        """
        Extract meta-controller features from an AgentState dictionary.

        This method converts raw state information into the structured
        MetaControllerFeatures format required for prediction.

        Args:
            state: Dictionary containing agent state information.
                   Expected keys include:
                   - 'hrm_confidence' or nested in 'agent_confidences'
                   - 'trm_confidence' or nested in 'agent_confidences'
                   - 'mcts_value' or nested in 'mcts_state'
                   - 'consensus_score'
                   - 'last_agent'
                   - 'iteration'
                   - 'query' or 'query_length'
                   - 'rag_context' or 'has_rag_context'

        Returns:
            MetaControllerFeatures instance with extracted values.

        Example:
            >>> state = {
            ...     'agent_confidences': {'hrm': 0.8, 'trm': 0.6},
            ...     'mcts_state': {'value': 0.75},
            ...     'consensus_score': 0.7,
            ...     'last_agent': 'hrm',
            ...     'iteration': 2,
            ...     'query': 'What is machine learning?',
            ...     'rag_context': 'ML is a subset of AI...'
            ... }
            >>> features = controller.extract_features(state)
        """
        # Extract HRM confidence
        if "hrm_confidence" in state:
            hrm_confidence = float(state["hrm_confidence"])
        elif "agent_confidences" in state and isinstance(state["agent_confidences"], dict):
            hrm_confidence = float(state["agent_confidences"].get("hrm", 0.0))
        else:
            hrm_confidence = 0.0

        # Extract TRM confidence
        if "trm_confidence" in state:
            trm_confidence = float(state["trm_confidence"])
        elif "agent_confidences" in state and isinstance(state["agent_confidences"], dict):
            trm_confidence = float(state["agent_confidences"].get("trm", 0.0))
        else:
            trm_confidence = 0.0

        # Extract MCTS value
        if "mcts_value" in state:
            mcts_value = float(state["mcts_value"])
        elif "mcts_state" in state and isinstance(state["mcts_state"], dict):
            mcts_value = float(state["mcts_state"].get("value", 0.0))
        else:
            mcts_value = 0.0

        # Extract consensus score
        consensus_score = float(state.get("consensus_score", 0.0))

        # Extract last agent
        last_agent = str(state.get("last_agent", "none"))
        if last_agent not in self.AGENT_NAMES and last_agent != "none":
            last_agent = "none"

        # Extract iteration
        iteration = int(state.get("iteration", 0))

        # Extract query length
        if "query_length" in state:
            query_length = int(state["query_length"])
        elif "query" in state and isinstance(state["query"], str):
            query_length = len(state["query"])
        else:
            query_length = 0

        # Extract has_rag_context
        if "has_rag_context" in state:
            has_rag_context = bool(state["has_rag_context"])
        elif "rag_context" in state:
            has_rag_context = state["rag_context"] is not None and len(str(state["rag_context"])) > 0
        else:
            has_rag_context = False

        return MetaControllerFeatures(
            hrm_confidence=hrm_confidence,
            trm_confidence=trm_confidence,
            mcts_value=mcts_value,
            consensus_score=consensus_score,
            last_agent=last_agent,
            iteration=iteration,
            query_length=query_length,
            has_rag_context=has_rag_context,
        )
