"""
Meta Controller Test Fixtures - Deterministic test scenarios for meta controller agent selection.

Provides:
- Deterministic test scenarios for meta controller
- Sample features and agent states
- Training batch generation
- Expected prediction rules for validation
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np

# Type alias for AgentState structure from src.framework.graph
AgentStateDict = Dict[str, Any]


@dataclass
class MetaControllerFeatures:
    """
    Features extracted for meta controller decision making.

    Contains normalized scores and metrics used to select the optimal agent.
    """

    # Agent confidence scores (0.0 to 1.0)
    hrm_confidence: float
    trm_confidence: float
    mcts_confidence: float

    # Quality metrics
    hrm_decomposition_quality: float
    trm_final_quality: float

    # MCTS statistics
    mcts_best_action_value: float
    mcts_cache_hit_rate: float
    mcts_exploration_ratio: float

    # Consensus and iteration info
    consensus_score: float
    iteration_count: int

    # Query complexity indicators
    query_length: int
    query_complexity_score: float

    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector for model input."""
        return np.array(
            [
                self.hrm_confidence,
                self.trm_confidence,
                self.mcts_confidence,
                self.hrm_decomposition_quality,
                self.trm_final_quality,
                self.mcts_best_action_value,
                self.mcts_cache_hit_rate,
                self.mcts_exploration_ratio,
                self.consensus_score,
                self.iteration_count / 10.0,  # Normalize iteration count
                self.query_length / 1000.0,  # Normalize query length
                self.query_complexity_score,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "MetaControllerFeatures":
        """Create features from numpy vector."""
        return cls(
            hrm_confidence=float(vector[0]),
            trm_confidence=float(vector[1]),
            mcts_confidence=float(vector[2]),
            hrm_decomposition_quality=float(vector[3]),
            trm_final_quality=float(vector[4]),
            mcts_best_action_value=float(vector[5]),
            mcts_cache_hit_rate=float(vector[6]),
            mcts_exploration_ratio=float(vector[7]),
            consensus_score=float(vector[8]),
            iteration_count=int(vector[9] * 10),
            query_length=int(vector[10] * 1000),
            query_complexity_score=float(vector[11]),
        )


# ============================================================================
# META CONTROLLER TEST FIXTURE CLASS
# ============================================================================


class MetaControllerTestFixture:
    """
    Fixture for deterministic meta controller testing.

    Provides sample features, agent states, and training data generation.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize fixture with deterministic seeding.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_sample_features(self) -> MetaControllerFeatures:
        """
        Create sample MetaControllerFeatures with realistic values.

        Returns:
            MetaControllerFeatures instance with balanced, realistic values
        """
        return MetaControllerFeatures(
            hrm_confidence=0.75,
            trm_confidence=0.72,
            mcts_confidence=0.68,
            hrm_decomposition_quality=0.78,
            trm_final_quality=0.74,
            mcts_best_action_value=0.65,
            mcts_cache_hit_rate=0.45,
            mcts_exploration_ratio=0.35,
            consensus_score=0.73,
            iteration_count=2,
            query_length=150,
            query_complexity_score=0.6,
        )

    def create_sample_agent_state(self) -> AgentStateDict:
        """
        Create sample AgentState matching TypedDict structure from src.framework.graph.

        Returns:
            Dictionary matching AgentState TypedDict structure
        """
        return {
            # Input
            "query": "Analyze the strategic implications of deploying multiple reconnaissance units in urban terrain.",
            "use_mcts": True,
            "use_rag": True,
            # RAG context
            "rag_context": "Urban operations require careful consideration of civilian presence and limited lines of sight.",
            "retrieved_docs": [
                {"content": "Urban terrain analysis guidelines.", "metadata": {"source": "doctrine_manual", "page": 42}}
            ],
            # Agent results
            "hrm_results": {
                "response": "Decomposed into 3 sub-problems: unit positioning, communication lines, civilian safety.",
                "metadata": {
                    "decomposition_quality_score": 0.82,
                    "num_subproblems": 3,
                    "reasoning_depth": 4,
                },
            },
            "trm_results": {
                "response": "Sequential reasoning chain established with 5 steps leading to tactical recommendation.",
                "metadata": {
                    "final_quality_score": 0.79,
                    "num_reasoning_steps": 5,
                    "chain_coherence": 0.85,
                },
            },
            "agent_outputs": [
                {
                    "agent": "hrm",
                    "response": "Decomposed analysis complete.",
                    "confidence": 0.82,
                },
                {
                    "agent": "trm",
                    "response": "Reasoning chain complete.",
                    "confidence": 0.79,
                },
            ],
            # MCTS simulation
            "mcts_root": None,  # MCTSNode would be here
            "mcts_iterations": 100,
            "mcts_best_action": "action_A",
            "mcts_stats": {
                "best_action_value": 0.72,
                "best_action_visits": 45,
                "root_visits": 100,
                "cache_hit_rate": 0.38,
                "cache_hits": 38,
                "cache_misses": 62,
                "total_simulations": 100,
                "iterations": 100,
                "unique_nodes": 87,
                "max_depth_reached": 5,
            },
            "mcts_config": {
                "seed": 42,
                "num_iterations": 100,
                "exploration_weight": 1.414,
            },
            # Evaluation
            "confidence_scores": {
                "hrm": 0.82,
                "trm": 0.79,
                "mcts": 0.72,
            },
            "consensus_reached": False,
            "consensus_score": 0.78,
            # Control flow
            "iteration": 1,
            "max_iterations": 3,
            # Output (may not be present yet)
            "final_response": "",
            "metadata": {},
        }

    def create_training_batch(self, batch_size: int = 32) -> Tuple[List[MetaControllerFeatures], List[str]]:
        """
        Create a training batch with features and corresponding labels.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (features_list, labels_list) where labels are agent names
        """
        features_list = []
        labels_list = []

        for i in range(batch_size):
            # Generate varied features using deterministic RNG
            scenario_type = i % 4  # Cycle through scenarios

            if scenario_type == 0:
                # High HRM confidence scenario
                features = self._create_hrm_favorable_features()
                labels_list.append("hrm")
            elif scenario_type == 1:
                # High TRM confidence scenario
                features = self._create_trm_favorable_features()
                labels_list.append("trm")
            elif scenario_type == 2:
                # MCTS favorable scenario
                features = self._create_mcts_favorable_features()
                labels_list.append("mcts")
            else:
                # Balanced/ambiguous scenario
                features = self._create_balanced_features()
                # For balanced, choose based on slight differences
                best_agent = self.get_expected_prediction(features)
                labels_list.append(best_agent)

            # Add some noise to make each sample unique
            features = self._add_noise_to_features(features, noise_scale=0.05)
            features_list.append(features)

        return features_list, labels_list

    def get_expected_prediction(self, features: MetaControllerFeatures) -> str:
        """
        Returns expected agent based on simple rules (highest confidence).

        This implements a simple rule-based selection for testing purposes.

        Args:
            features: MetaControllerFeatures instance

        Returns:
            Agent name string: "hrm", "trm", or "mcts"
        """
        # Compute weighted scores for each agent
        hrm_score = (
            features.hrm_confidence * 0.4 + features.hrm_decomposition_quality * 0.4 + features.consensus_score * 0.2
        )

        trm_score = features.trm_confidence * 0.4 + features.trm_final_quality * 0.4 + features.consensus_score * 0.2

        mcts_score = (
            features.mcts_confidence * 0.3
            + features.mcts_best_action_value * 0.3
            + features.mcts_cache_hit_rate * 0.2
            + features.mcts_exploration_ratio * 0.2
        )

        # Return agent with highest score
        scores = {
            "hrm": hrm_score,
            "trm": trm_score,
            "mcts": mcts_score,
        }

        return max(scores, key=scores.get)

    def _create_hrm_favorable_features(self) -> MetaControllerFeatures:
        """Create features where HRM should be chosen."""
        return MetaControllerFeatures(
            hrm_confidence=0.92,
            trm_confidence=0.65,
            mcts_confidence=0.58,
            hrm_decomposition_quality=0.88,
            trm_final_quality=0.62,
            mcts_best_action_value=0.55,
            mcts_cache_hit_rate=0.30,
            mcts_exploration_ratio=0.25,
            consensus_score=0.75,
            iteration_count=1,
            query_length=200,
            query_complexity_score=0.7,
        )

    def _create_trm_favorable_features(self) -> MetaControllerFeatures:
        """Create features where TRM should be chosen."""
        return MetaControllerFeatures(
            hrm_confidence=0.68,
            trm_confidence=0.91,
            mcts_confidence=0.60,
            hrm_decomposition_quality=0.65,
            trm_final_quality=0.89,
            mcts_best_action_value=0.58,
            mcts_cache_hit_rate=0.35,
            mcts_exploration_ratio=0.28,
            consensus_score=0.78,
            iteration_count=2,
            query_length=180,
            query_complexity_score=0.65,
        )

    def _create_mcts_favorable_features(self) -> MetaControllerFeatures:
        """Create features where MCTS should be chosen."""
        return MetaControllerFeatures(
            hrm_confidence=0.62,
            trm_confidence=0.64,
            mcts_confidence=0.88,
            hrm_decomposition_quality=0.60,
            trm_final_quality=0.62,
            mcts_best_action_value=0.85,
            mcts_cache_hit_rate=0.75,
            mcts_exploration_ratio=0.70,
            consensus_score=0.65,
            iteration_count=3,
            query_length=250,
            query_complexity_score=0.8,
        )

    def _create_balanced_features(self) -> MetaControllerFeatures:
        """Create balanced/ambiguous features."""
        return MetaControllerFeatures(
            hrm_confidence=0.72,
            trm_confidence=0.71,
            mcts_confidence=0.70,
            hrm_decomposition_quality=0.73,
            trm_final_quality=0.72,
            mcts_best_action_value=0.69,
            mcts_cache_hit_rate=0.50,
            mcts_exploration_ratio=0.48,
            consensus_score=0.72,
            iteration_count=2,
            query_length=175,
            query_complexity_score=0.68,
        )

    def _add_noise_to_features(
        self, features: MetaControllerFeatures, noise_scale: float = 0.05
    ) -> MetaControllerFeatures:
        """Add small random noise to features for variety."""

        def clip_value(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            noise = self.rng.normal(0, noise_scale)
            return float(np.clip(val + noise, min_val, max_val))

        return MetaControllerFeatures(
            hrm_confidence=clip_value(features.hrm_confidence),
            trm_confidence=clip_value(features.trm_confidence),
            mcts_confidence=clip_value(features.mcts_confidence),
            hrm_decomposition_quality=clip_value(features.hrm_decomposition_quality),
            trm_final_quality=clip_value(features.trm_final_quality),
            mcts_best_action_value=clip_value(features.mcts_best_action_value),
            mcts_cache_hit_rate=clip_value(features.mcts_cache_hit_rate),
            mcts_exploration_ratio=clip_value(features.mcts_exploration_ratio),
            consensus_score=clip_value(features.consensus_score),
            iteration_count=features.iteration_count,  # Keep discrete
            query_length=features.query_length,  # Keep discrete
            query_complexity_score=clip_value(features.query_complexity_score),
        )


# ============================================================================
# SAMPLE TEST DATA SCENARIOS
# ============================================================================


def create_high_hrm_confidence_scenario() -> Dict[str, Any]:
    """
    Create scenario where HRM should be chosen.

    High decomposition quality and HRM confidence indicate complex
    hierarchical problem that benefits from HRM approach.

    Returns:
        Scenario dictionary with features and expected agent
    """
    return {
        "name": "high_hrm_confidence",
        "description": "HRM excels due to high decomposition quality and confidence",
        "features": MetaControllerFeatures(
            hrm_confidence=0.95,
            trm_confidence=0.60,
            mcts_confidence=0.55,
            hrm_decomposition_quality=0.92,
            trm_final_quality=0.58,
            mcts_best_action_value=0.52,
            mcts_cache_hit_rate=0.25,
            mcts_exploration_ratio=0.20,
            consensus_score=0.70,
            iteration_count=1,
            query_length=300,
            query_complexity_score=0.85,
        ),
        "expected_agent": "hrm",
        "reasoning": "Complex hierarchical problem requiring decomposition into sub-problems",
    }


def create_high_trm_confidence_scenario() -> Dict[str, Any]:
    """
    Create scenario where TRM should be chosen.

    High reasoning chain quality and TRM confidence indicate sequential
    reasoning problem that benefits from TRM approach.

    Returns:
        Scenario dictionary with features and expected agent
    """
    return {
        "name": "high_trm_confidence",
        "description": "TRM excels due to high reasoning quality and confidence",
        "features": MetaControllerFeatures(
            hrm_confidence=0.58,
            trm_confidence=0.94,
            mcts_confidence=0.52,
            hrm_decomposition_quality=0.55,
            trm_final_quality=0.91,
            mcts_best_action_value=0.50,
            mcts_cache_hit_rate=0.30,
            mcts_exploration_ratio=0.22,
            consensus_score=0.72,
            iteration_count=2,
            query_length=220,
            query_complexity_score=0.75,
        ),
        "expected_agent": "trm",
        "reasoning": "Sequential reasoning problem requiring step-by-step logical chain",
    }


def create_mcts_favorable_scenario() -> Dict[str, Any]:
    """
    Create scenario where MCTS should be chosen.

    High exploration metrics and simulation success indicate
    decision problem that benefits from tree search.

    Returns:
        Scenario dictionary with features and expected agent
    """
    return {
        "name": "mcts_favorable",
        "description": "MCTS excels due to high action value and cache efficiency",
        "features": MetaControllerFeatures(
            hrm_confidence=0.55,
            trm_confidence=0.58,
            mcts_confidence=0.90,
            hrm_decomposition_quality=0.52,
            trm_final_quality=0.56,
            mcts_best_action_value=0.88,
            mcts_cache_hit_rate=0.80,
            mcts_exploration_ratio=0.75,
            consensus_score=0.60,
            iteration_count=3,
            query_length=280,
            query_complexity_score=0.82,
        ),
        "expected_agent": "mcts",
        "reasoning": "Decision optimization problem requiring exploration of action space",
    }


def create_balanced_scenario() -> Dict[str, Any]:
    """
    Create balanced/ambiguous scenario.

    All agents have similar confidence levels, requiring careful
    feature weighting to make selection decision.

    Returns:
        Scenario dictionary with features and expected agent
    """
    features = MetaControllerFeatures(
        hrm_confidence=0.74,
        trm_confidence=0.73,
        mcts_confidence=0.72,
        hrm_decomposition_quality=0.75,
        trm_final_quality=0.74,
        mcts_best_action_value=0.71,
        mcts_cache_hit_rate=0.55,
        mcts_exploration_ratio=0.52,
        consensus_score=0.74,
        iteration_count=2,
        query_length=190,
        query_complexity_score=0.70,
    )

    # Determine expected agent based on scoring
    fixture = MetaControllerTestFixture(seed=42)
    expected = fixture.get_expected_prediction(features)

    return {
        "name": "balanced_scenario",
        "description": "Ambiguous case with similar confidence levels across agents",
        "features": features,
        "expected_agent": expected,
        "reasoning": "Close competition requiring weighted scoring to determine best agent",
    }


# ============================================================================
# PRESET TEST CONFIGURATIONS
# ============================================================================


TEST_SCENARIOS = {
    "high_hrm_confidence": create_high_hrm_confidence_scenario(),
    "high_trm_confidence": create_high_trm_confidence_scenario(),
    "mcts_favorable": create_mcts_favorable_scenario(),
    "balanced_scenario": create_balanced_scenario(),
}
"""Pre-configured test scenarios for meta controller validation."""


DETERMINISTIC_TEST_SEEDS = [42, 123, 456, 789, 1024]
"""Standard seeds for determinism testing."""


def get_all_test_scenarios() -> List[Dict[str, Any]]:
    """
    Get all predefined test scenarios.

    Returns:
        List of all scenario dictionaries
    """
    return list(TEST_SCENARIOS.values())


def validate_scenario_predictions() -> Dict[str, bool]:
    """
    Validate that fixture predictions match expected outcomes.

    Returns:
        Dictionary mapping scenario names to validation results
    """
    results = {}
    fixture = MetaControllerTestFixture(seed=42)

    for name, scenario in TEST_SCENARIOS.items():
        predicted = fixture.get_expected_prediction(scenario["features"])
        results[name] = predicted == scenario["expected_agent"]

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def features_to_agent_state(
    features: MetaControllerFeatures,
    query: str = "Test query for meta controller evaluation.",
) -> AgentStateDict:
    """
    Convert MetaControllerFeatures to full AgentState dictionary.

    Args:
        features: MetaControllerFeatures instance
        query: Query string to use

    Returns:
        Full AgentState dictionary matching TypedDict structure
    """
    return {
        "query": query,
        "use_mcts": True,
        "use_rag": True,
        "rag_context": "Retrieved context for query.",
        "retrieved_docs": [],
        "hrm_results": {
            "response": "HRM analysis complete.",
            "metadata": {
                "decomposition_quality_score": features.hrm_decomposition_quality,
            },
        },
        "trm_results": {
            "response": "TRM reasoning complete.",
            "metadata": {
                "final_quality_score": features.trm_final_quality,
            },
        },
        "agent_outputs": [
            {"agent": "hrm", "response": "HRM output", "confidence": features.hrm_confidence},
            {"agent": "trm", "response": "TRM output", "confidence": features.trm_confidence},
            {"agent": "mcts", "response": "MCTS output", "confidence": features.mcts_confidence},
        ],
        "mcts_stats": {
            "best_action_value": features.mcts_best_action_value,
            "cache_hit_rate": features.mcts_cache_hit_rate,
            "exploration_ratio": features.mcts_exploration_ratio,
            "iterations": 100,
        },
        "confidence_scores": {
            "hrm": features.hrm_confidence,
            "trm": features.trm_confidence,
            "mcts": features.mcts_confidence,
        },
        "consensus_score": features.consensus_score,
        "iteration": features.iteration_count,
        "max_iterations": 3,
        "agent_outputs": [],
    }


def agent_state_to_features(state: AgentStateDict) -> MetaControllerFeatures:
    """
    Extract MetaControllerFeatures from AgentState dictionary.

    Args:
        state: AgentState dictionary

    Returns:
        MetaControllerFeatures instance
    """
    confidence_scores = state.get("confidence_scores", {})
    mcts_stats = state.get("mcts_stats", {})
    hrm_results = state.get("hrm_results", {})
    trm_results = state.get("trm_results", {})

    return MetaControllerFeatures(
        hrm_confidence=confidence_scores.get("hrm", 0.5),
        trm_confidence=confidence_scores.get("trm", 0.5),
        mcts_confidence=confidence_scores.get("mcts", 0.5),
        hrm_decomposition_quality=hrm_results.get("metadata", {}).get("decomposition_quality_score", 0.5),
        trm_final_quality=trm_results.get("metadata", {}).get("final_quality_score", 0.5),
        mcts_best_action_value=mcts_stats.get("best_action_value", 0.5),
        mcts_cache_hit_rate=mcts_stats.get("cache_hit_rate", 0.0),
        mcts_exploration_ratio=mcts_stats.get("exploration_ratio", 0.0),
        consensus_score=state.get("consensus_score", 0.5),
        iteration_count=state.get("iteration", 0),
        query_length=len(state.get("query", "")),
        query_complexity_score=0.5,  # Would need NLP analysis for real value
    )


# ============================================================================
# DEMONSTRATION
# ============================================================================


if __name__ == "__main__":
    print("=== Meta Controller Test Fixtures Demonstration ===\n")

    # Create fixture
    fixture = MetaControllerTestFixture(seed=42)

    # 1. Sample features
    print("1. Sample MetaControllerFeatures:")
    sample_features = fixture.create_sample_features()
    print(f"   HRM Confidence: {sample_features.hrm_confidence}")
    print(f"   TRM Confidence: {sample_features.trm_confidence}")
    print(f"   MCTS Confidence: {sample_features.mcts_confidence}")
    print(f"   Expected Agent: {fixture.get_expected_prediction(sample_features)}\n")

    # 2. Sample agent state
    print("2. Sample AgentState:")
    sample_state = fixture.create_sample_agent_state()
    print(f"   Query: {sample_state['query'][:60]}...")
    print(f"   HRM Quality: {sample_state['hrm_results']['metadata']['decomposition_quality_score']}")
    print(f"   TRM Quality: {sample_state['trm_results']['metadata']['final_quality_score']}")
    print(f"   MCTS Best Value: {sample_state['mcts_stats']['best_action_value']}\n")

    # 3. Training batch
    print("3. Training Batch Generation:")
    features_list, labels_list = fixture.create_training_batch(batch_size=8)
    label_counts = {label: labels_list.count(label) for label in set(labels_list)}
    print(f"   Batch Size: {len(features_list)}")
    print(f"   Label Distribution: {label_counts}\n")

    # 4. Test scenarios
    print("4. Predefined Test Scenarios:")
    for name, scenario in TEST_SCENARIOS.items():
        expected = scenario["expected_agent"]
        predicted = fixture.get_expected_prediction(scenario["features"])
        match = "PASS" if expected == predicted else "FAIL"
        print(f"   {name}: Expected={expected}, Predicted={predicted} [{match}]")

    print("\n5. Scenario Validation:")
    validation = validate_scenario_predictions()
    all_pass = all(validation.values())
    print(f"   All scenarios valid: {all_pass}")

    print("\n=== Demonstration Complete ===")
