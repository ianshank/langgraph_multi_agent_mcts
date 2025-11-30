"""
Action Types and Graph Configuration Module.

Provides:
- ActionType enum for MCTS action space
- AgentType enum for routing decisions
- GraphConfig for workflow parameters
- Removes all hardcoded values from graph.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    """
    MCTS action types for tree expansion.

    Root-level actions represent initial strategic choices.
    Continuation actions represent follow-up decisions.
    """

    # Root-level actions (depth 0)
    EXPLORE_BREADTH = "explore_breadth"
    EXPLORE_DEPTH = "explore_depth"
    SYNTHESIZE = "synthesize"
    DELEGATE = "delegate"

    # Continuation actions (depth 1+)
    CONTINUE = "continue"
    REFINE = "refine"
    FALLBACK = "fallback"
    ESCALATE = "escalate"

    @classmethod
    def root_actions(cls) -> list[str]:
        """Return action strings for root level."""
        return [
            cls.EXPLORE_BREADTH.value,
            cls.EXPLORE_DEPTH.value,
            cls.SYNTHESIZE.value,
            cls.DELEGATE.value,
        ]

    @classmethod
    def continuation_actions(cls) -> list[str]:
        """Return action strings for continuation levels."""
        return [
            cls.CONTINUE.value,
            cls.REFINE.value,
            cls.FALLBACK.value,
            cls.ESCALATE.value,
        ]


class AgentType(Enum):
    """
    Agent types for routing decisions.

    Used by the meta-controller and graph workflow routing.
    """

    HRM = "hrm"
    TRM = "trm"
    MCTS = "mcts"
    PARALLEL = "parallel"
    AGGREGATE = "aggregate"

    @classmethod
    def all_agents(cls) -> list[str]:
        """Return all agent type strings."""
        return [member.value for member in cls]

    @classmethod
    def reasoning_agents(cls) -> list[str]:
        """Return agents that perform reasoning (not control flow)."""
        return [cls.HRM.value, cls.TRM.value, cls.MCTS.value]


class RouteDecision(Enum):
    """
    Route decision types for workflow control.

    Maps to graph node names for conditional routing.
    """

    PARALLEL = "parallel"
    HRM = "hrm"
    TRM = "trm"
    MCTS = "mcts"
    AGGREGATE = "aggregate"
    SYNTHESIZE = "synthesize"
    ITERATE = "iterate"

    def to_node_name(self) -> str:
        """Convert route decision to graph node name."""
        node_mapping = {
            RouteDecision.PARALLEL: "parallel_agents",
            RouteDecision.HRM: "hrm_agent",
            RouteDecision.TRM: "trm_agent",
            RouteDecision.MCTS: "mcts_simulator",
            RouteDecision.AGGREGATE: "aggregate_results",
            RouteDecision.SYNTHESIZE: "synthesize",
            RouteDecision.ITERATE: "route_decision",
        }
        return node_mapping[self]


@dataclass
class ConfidenceConfig:
    """
    Configuration for confidence thresholds and defaults.

    Centralizes all confidence-related magic numbers.
    """

    # Default confidence when metadata is missing
    default_hrm_confidence: float = 0.5
    default_trm_confidence: float = 0.5
    default_mcts_confidence: float = 0.5
    default_adk_confidence: float = 0.8

    # Agent output fallback confidence
    agent_fallback_confidence: float = 0.7

    # Consensus evaluation
    consensus_threshold: float = 0.75
    perfect_consensus_score: float = 1.0

    # Heuristic weights for MCTS
    heuristic_base_value: float = 0.5
    confidence_weight_multiplier: float = 0.2

    def __post_init__(self):
        """Validate confidence values are in [0, 1]."""
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


@dataclass
class RolloutWeights:
    """
    Configuration for rollout policy weights.

    Controls the balance between heuristic and random components.
    """

    heuristic_weight: float = 0.7
    random_weight: float = 0.3

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.heuristic_weight + self.random_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total} "
                f"(heuristic={self.heuristic_weight}, random={self.random_weight})"
            )


@dataclass
class SynthesisConfig:
    """
    Configuration for response synthesis.

    Controls LLM parameters for final response generation.
    """

    temperature: float = 0.5
    max_tokens: int = 2048

    def __post_init__(self):
        """Validate synthesis parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass
class GraphConfig:
    """
    Complete configuration for LangGraph workflow.

    Centralizes all configurable parameters for the multi-agent graph.
    This eliminates hardcoded values throughout graph.py.
    """

    # Agent configuration
    max_iterations: int = 3
    enable_parallel_agents: bool = True

    # RAG configuration
    top_k_retrieval: int = 5
    use_rag_by_default: bool = True

    # Confidence configuration
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)

    # Rollout configuration
    rollout_weights: RolloutWeights = field(default_factory=RolloutWeights)

    # Synthesis configuration
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    # Action configuration (can be customized per domain)
    root_actions: list[str] = field(default_factory=ActionType.root_actions)
    continuation_actions: list[str] = field(default_factory=ActionType.continuation_actions)

    # ADK trigger keywords (domain-specific)
    adk_triggers: dict[str, list[str]] = field(default_factory=lambda: {
        "deep_search": ["research", "investigate", "explore"],
        "ml_engineering": ["train", "model", "fine-tune"],
        "data_science": ["analyze", "data", "statistics"],
    })

    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")
        if self.top_k_retrieval < 1:
            raise ValueError(f"top_k_retrieval must be >= 1, got {self.top_k_retrieval}")

    def get_route_mapping(self) -> dict[str, str]:
        """
        Get the routing map from decision strings to node names.

        Returns:
            Dictionary mapping route decisions to graph node names.
        """
        return {
            RouteDecision.PARALLEL.value: RouteDecision.PARALLEL.to_node_name(),
            RouteDecision.HRM.value: RouteDecision.HRM.to_node_name(),
            RouteDecision.TRM.value: RouteDecision.TRM.to_node_name(),
            RouteDecision.MCTS.value: RouteDecision.MCTS.to_node_name(),
            RouteDecision.AGGREGATE.value: RouteDecision.AGGREGATE.to_node_name(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "max_iterations": self.max_iterations,
            "enable_parallel_agents": self.enable_parallel_agents,
            "top_k_retrieval": self.top_k_retrieval,
            "use_rag_by_default": self.use_rag_by_default,
            "confidence": self.confidence.__dict__,
            "rollout_weights": self.rollout_weights.__dict__,
            "synthesis": self.synthesis.__dict__,
            "root_actions": self.root_actions,
            "continuation_actions": self.continuation_actions,
            "adk_triggers": self.adk_triggers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphConfig:
        """Create configuration from dictionary."""
        config = cls()

        # Update simple fields
        for key in ["max_iterations", "enable_parallel_agents", "top_k_retrieval",
                    "use_rag_by_default", "root_actions", "continuation_actions", "adk_triggers"]:
            if key in data:
                setattr(config, key, data[key])

        # Update nested configs
        if "confidence" in data:
            config.confidence = ConfidenceConfig(**data["confidence"])
        if "rollout_weights" in data:
            config.rollout_weights = RolloutWeights(**data["rollout_weights"])
        if "synthesis" in data:
            config.synthesis = SynthesisConfig(**data["synthesis"])

        return config


# Default configuration instance
DEFAULT_GRAPH_CONFIG = GraphConfig()


# Domain-specific preset configurations
def create_research_config() -> GraphConfig:
    """Configuration optimized for research tasks."""
    return GraphConfig(
        max_iterations=5,
        enable_parallel_agents=True,
        top_k_retrieval=10,
        confidence=ConfidenceConfig(
            consensus_threshold=0.8,
            heuristic_base_value=0.4,
        ),
        rollout_weights=RolloutWeights(
            heuristic_weight=0.6,
            random_weight=0.4,
        ),
        synthesis=SynthesisConfig(temperature=0.7),
        adk_triggers={
            "deep_search": ["research", "investigate", "literature", "papers"],
            "academic_research": ["cite", "reference", "scholarly"],
        },
    )


def create_coding_config() -> GraphConfig:
    """Configuration optimized for coding tasks."""
    return GraphConfig(
        max_iterations=4,
        enable_parallel_agents=False,  # Sequential for code coherence
        confidence=ConfidenceConfig(
            consensus_threshold=0.85,  # Higher bar for code
            heuristic_base_value=0.6,
        ),
        rollout_weights=RolloutWeights(
            heuristic_weight=0.8,
            random_weight=0.2,
        ),
        synthesis=SynthesisConfig(temperature=0.3),  # More deterministic
        root_actions=["implement", "debug", "refactor", "test"],
        continuation_actions=["continue", "fix", "optimize", "validate"],
    )


def create_creative_config() -> GraphConfig:
    """Configuration optimized for creative tasks."""
    return GraphConfig(
        max_iterations=3,
        enable_parallel_agents=True,
        confidence=ConfidenceConfig(
            consensus_threshold=0.6,  # Lower bar for creativity
            heuristic_base_value=0.3,
        ),
        rollout_weights=RolloutWeights(
            heuristic_weight=0.4,
            random_weight=0.6,  # More exploration
        ),
        synthesis=SynthesisConfig(temperature=0.9),  # More creative
    )
