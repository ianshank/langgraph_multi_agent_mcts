"""
MCTS Test Fixtures - Deterministic test scenarios and known-good tree structures.

Provides:
- Deterministic test scenarios
- Known-good tree structures for validation
- Seed-based reproducible tests
- Helper functions for test setup
"""

from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.config import MCTSConfig, create_preset_config, ConfigPreset
from src.framework.mcts.policies import (
    RandomRolloutPolicy,
    HybridRolloutPolicy,
    GreedyRolloutPolicy,
    SelectionPolicy,
)
from src.framework.mcts.experiments import ExperimentTracker


# ============================================================================
# KNOWN DETERMINISTIC SCENARIOS
# ============================================================================


def create_simple_binary_tree_scenario() -> Dict[str, Any]:
    """
    Create a simple binary decision tree scenario.

    Returns a scenario where optimal path is deterministic given seed.
    """
    return {
        "name": "simple_binary_tree",
        "description": "Binary tree with known optimal path",
        "actions_per_state": ["left", "right"],
        "max_depth": 3,
        "optimal_path": ["left", "left", "left"],  # With seed 42
        "expected_value_range": (0.7, 0.9),
    }


def create_tactical_decision_scenario() -> Dict[str, Any]:
    """
    Create a tactical decision scenario with multiple actions.

    Simulates a military tactical decision tree.
    """
    return {
        "name": "tactical_decision",
        "description": "Tactical decision tree with defensive/offensive choices",
        "root_actions": ["defend_position", "advance_cautiously", "flanking_maneuver", "retreat"],
        "second_level_actions": ["consolidate", "reinforce", "exploit", "disengage"],
        "max_depth": 4,
        "heuristics": {
            "defend_position": 0.7,
            "advance_cautiously": 0.6,
            "flanking_maneuver": 0.5,
            "retreat": 0.4,
        },
    }


def create_progressive_widening_test_scenario() -> Dict[str, Any]:
    """
    Create scenario to test progressive widening behavior.

    Large action space that should be controlled by progressive widening.
    """
    return {
        "name": "progressive_widening_test",
        "description": "Large action space for testing progressive widening",
        "num_actions": 20,
        "expected_expansions": {
            "k=1.0, alpha=0.5": {
                "visits_1": 1,   # After 1 visit, expand to 1 child
                "visits_2": 2,   # After 2 visits, can have 2 children
                "visits_4": 2,   # After 4 visits, can have 4 children (but 2**0.5=1.41, so 4 > 1.41*2)
            },
            "k=2.0, alpha=0.5": {
                "visits_1": 0,   # Threshold = 2*0^0.5 = 0, so expand
                "visits_2": 1,   # Threshold = 2*1^0.5 = 2, so 2 > 2 = False
                "visits_4": 1,   # Threshold = 2*1^0.5 = 2, so 4 > 2 = True
            },
        },
    }


# ============================================================================
# DETERMINISTIC TEST FIXTURES
# ============================================================================


class DeterministicTestFixture:
    """
    Fixture for deterministic MCTS testing.

    Ensures identical results for same seed across runs.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_engine(self, config: Optional[MCTSConfig] = None) -> MCTSEngine:
        """Create MCTS engine with fixture seed."""
        if config is None:
            config = MCTSConfig(seed=self.seed)
        else:
            config.seed = self.seed

        return MCTSEngine(
            seed=config.seed,
            exploration_weight=config.exploration_weight,
            progressive_widening_k=config.progressive_widening_k,
            progressive_widening_alpha=config.progressive_widening_alpha,
            max_parallel_rollouts=config.max_parallel_rollouts,
            cache_size_limit=config.cache_size_limit,
        )

    def create_root_state(self, state_id: str = "root") -> MCTSState:
        """Create deterministic root state."""
        return MCTSState(
            state_id=state_id,
            features={"created_by": "fixture", "seed": self.seed},
        )

    def create_root_node(self, engine: MCTSEngine) -> MCTSNode:
        """Create root node with engine's RNG."""
        root_state = self.create_root_state()
        return MCTSNode(state=root_state, rng=engine.rng)

    def binary_action_generator(self, state: MCTSState) -> List[str]:
        """Generate binary actions (left/right)."""
        depth = len(state.state_id.split("_")) - 1
        if depth < 3:
            return ["left", "right"]
        return []

    def tactical_action_generator(self, state: MCTSState) -> List[str]:
        """Generate tactical actions based on depth."""
        depth = len(state.state_id.split("_")) - 1
        if depth == 0:
            return ["defend", "advance", "flank", "retreat"]
        elif depth < 4:
            return ["consolidate", "reinforce", "exploit"]
        return []

    def simple_state_transition(
        self, state: MCTSState, action: str
    ) -> MCTSState:
        """Simple state transition appending action to ID."""
        new_id = f"{state.state_id}_{action}"
        new_features = state.features.copy()
        new_features["last_action"] = action
        new_features["depth"] = len(new_id.split("_")) - 1
        return MCTSState(state_id=new_id, features=new_features)


class KnownGoodTreeFixture:
    """
    Fixture with known-good tree structures for validation.

    Used to verify MCTS algorithms produce expected results.
    """

    @staticmethod
    def create_balanced_binary_tree(depth: int = 3) -> MCTSNode:
        """
        Create a balanced binary tree with known values.

        Args:
            depth: Tree depth

        Returns:
            Root node of balanced tree
        """
        rng = np.random.default_rng(42)

        def build_tree(node: MCTSNode, current_depth: int):
            if current_depth >= depth:
                node.terminal = True
                return

            # Add left child
            left_state = MCTSState(
                state_id=f"{node.state.state_id}_left",
                features={"side": "left", "depth": current_depth + 1},
            )
            left_child = node.add_child("left", left_state)

            # Add right child
            right_state = MCTSState(
                state_id=f"{node.state.state_id}_right",
                features={"side": "right", "depth": current_depth + 1},
            )
            right_child = node.add_child("right", right_state)

            # Recursively build subtrees
            build_tree(left_child, current_depth + 1)
            build_tree(right_child, current_depth + 1)

        root_state = MCTSState(state_id="root", features={"depth": 0})
        root = MCTSNode(state=root_state, rng=rng)
        build_tree(root, 0)

        return root

    @staticmethod
    def create_imbalanced_tree() -> MCTSNode:
        """
        Create an imbalanced tree for testing robust selection.

        One branch has high value but low visits,
        another has moderate value but high visits.
        """
        rng = np.random.default_rng(42)

        root_state = MCTSState(state_id="root", features={})
        root = MCTSNode(state=root_state, rng=rng)

        # High value, low visits child
        high_value_state = MCTSState(state_id="high_value", features={})
        high_value_child = root.add_child("high_value_action", high_value_state)
        high_value_child.visits = 10
        high_value_child.value_sum = 9.0  # Average = 0.9

        # Moderate value, high visits child
        moderate_state = MCTSState(state_id="moderate", features={})
        moderate_child = root.add_child("moderate_action", moderate_state)
        moderate_child.visits = 100
        moderate_child.value_sum = 70.0  # Average = 0.7

        # Low value, medium visits child
        low_value_state = MCTSState(state_id="low_value", features={})
        low_value_child = root.add_child("low_value_action", low_value_state)
        low_value_child.visits = 50
        low_value_child.value_sum = 25.0  # Average = 0.5

        # Update root visits
        root.visits = 160

        return root

    @staticmethod
    def create_known_optimal_tree() -> Tuple[MCTSNode, str, float]:
        """
        Create tree with known optimal action.

        Returns:
            Tuple of (root_node, optimal_action, optimal_value)
        """
        rng = np.random.default_rng(42)

        root_state = MCTSState(state_id="root", features={})
        root = MCTSNode(state=root_state, rng=rng)

        # Create children with known values
        optimal_state = MCTSState(state_id="optimal", features={})
        optimal_child = root.add_child("optimal_action", optimal_state)
        optimal_child.visits = 200
        optimal_child.value_sum = 180.0  # Average = 0.9

        suboptimal_state = MCTSState(state_id="suboptimal", features={})
        suboptimal_child = root.add_child("suboptimal_action", suboptimal_state)
        suboptimal_child.visits = 150
        suboptimal_child.value_sum = 90.0  # Average = 0.6

        bad_state = MCTSState(state_id="bad", features={})
        bad_child = root.add_child("bad_action", bad_state)
        bad_child.visits = 50
        bad_child.value_sum = 15.0  # Average = 0.3

        root.visits = 400
        root.value_sum = 285.0

        return root, "optimal_action", 0.9


# ============================================================================
# SEED-BASED REPRODUCIBLE TEST HELPERS
# ============================================================================


async def run_determinism_verification(
    num_runs: int = 3,
    num_iterations: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Verify MCTS produces identical results with same seed.

    Args:
        num_runs: Number of runs to compare
        num_iterations: MCTS iterations per run
        seed: Random seed

    Returns:
        Verification results
    """
    results = []
    fixture = DeterministicTestFixture(seed=seed)

    for run in range(num_runs):
        # Create fresh engine with same seed
        engine = fixture.create_engine()
        root = fixture.create_root_node(engine)

        # Use random rollout for determinism
        rollout_policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.3)

        # Run MCTS
        best_action, stats = await engine.search(
            root=root,
            num_iterations=num_iterations,
            action_generator=fixture.binary_action_generator,
            state_transition=fixture.simple_state_transition,
            rollout_policy=rollout_policy,
            selection_policy=SelectionPolicy.MAX_VISITS,
        )

        results.append(
            {
                "run": run,
                "best_action": best_action,
                "best_value": stats["best_action_value"],
                "best_visits": stats["best_action_visits"],
                "root_visits": stats["root_visits"],
            }
        )

    # Check for consistency
    is_deterministic = True
    inconsistencies = []

    for i in range(1, len(results)):
        if results[i]["best_action"] != results[0]["best_action"]:
            is_deterministic = False
            inconsistencies.append(f"Run {i}: different best_action")
        if abs(results[i]["best_value"] - results[0]["best_value"]) > 1e-6:
            is_deterministic = False
            inconsistencies.append(f"Run {i}: different best_value")
        if results[i]["best_visits"] != results[0]["best_visits"]:
            is_deterministic = False
            inconsistencies.append(f"Run {i}: different best_visits")

    return {
        "seed": seed,
        "num_runs": num_runs,
        "num_iterations": num_iterations,
        "is_deterministic": is_deterministic,
        "inconsistencies": inconsistencies,
        "results": results,
    }


async def test_progressive_widening_behavior(
    k: float = 1.0,
    alpha: float = 0.5,
    num_iterations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test progressive widening with given parameters.

    Args:
        k: Progressive widening coefficient
        alpha: Progressive widening exponent
        num_iterations: Number of iterations
        seed: Random seed

    Returns:
        Analysis of widening behavior
    """
    config = MCTSConfig(
        seed=seed,
        num_iterations=num_iterations,
        progressive_widening_k=k,
        progressive_widening_alpha=alpha,
    )

    fixture = DeterministicTestFixture(seed=seed)
    engine = fixture.create_engine(config)
    root = fixture.create_root_node(engine)

    # Large action space to test widening
    def large_action_generator(state: MCTSState) -> List[str]:
        depth = len(state.state_id.split("_")) - 1
        if depth < 3:
            return [f"action_{i}" for i in range(10)]
        return []

    rollout_policy = RandomRolloutPolicy()

    best_action, stats = await engine.search(
        root=root,
        num_iterations=num_iterations,
        action_generator=large_action_generator,
        state_transition=fixture.simple_state_transition,
        rollout_policy=rollout_policy,
    )

    # Analyze widening behavior
    max_children = len(root.children)
    total_nodes = engine.count_nodes(root)
    tree_depth = engine.get_tree_depth(root)

    return {
        "k": k,
        "alpha": alpha,
        "num_iterations": num_iterations,
        "root_children": max_children,
        "total_nodes": total_nodes,
        "tree_depth": tree_depth,
        "best_action": best_action,
        "root_visits": root.visits,
        "widening_controlled": max_children < 10,  # Should be less than max available
    }


async def test_cache_effectiveness(
    num_iterations: int = 100,
    cache_enabled: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test simulation caching effectiveness.

    Args:
        num_iterations: MCTS iterations
        cache_enabled: Whether caching is enabled
        seed: Random seed

    Returns:
        Cache performance analysis
    """
    config = MCTSConfig(
        seed=seed,
        num_iterations=num_iterations,
        enable_cache=cache_enabled,
        cache_size_limit=10000 if cache_enabled else 0,
    )

    fixture = DeterministicTestFixture(seed=seed)
    engine = fixture.create_engine(config)

    if not cache_enabled:
        engine.cache_size_limit = 0

    root = fixture.create_root_node(engine)
    rollout_policy = RandomRolloutPolicy()

    best_action, stats = await engine.search(
        root=root,
        num_iterations=num_iterations,
        action_generator=fixture.binary_action_generator,
        state_transition=fixture.simple_state_transition,
        rollout_policy=rollout_policy,
    )

    return {
        "cache_enabled": cache_enabled,
        "num_iterations": num_iterations,
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "cache_hit_rate": stats["cache_hit_rate"],
        "total_simulations": stats["total_simulations"],
    }


# ============================================================================
# PRESET TEST CONFIGURATIONS
# ============================================================================


DETERMINISTIC_TEST_SEEDS = [42, 123, 456, 789, 1024]
"""Standard seeds for determinism testing."""

KNOWN_GOOD_RESULTS = {
    "seed_42_binary_50iter": {
        "expected_best_action": "left",  # May vary based on rollout
        "expected_visits_range": (20, 35),
        "expected_value_range": (0.4, 0.7),
    },
    "seed_42_tactical_100iter": {
        "expected_best_action": "defend",  # Highest heuristic value
        "expected_visits_range": (30, 60),
    },
}
"""Known-good results for regression testing."""


def get_test_config_matrix() -> List[MCTSConfig]:
    """
    Get matrix of test configurations.

    Returns:
        List of configs for comprehensive testing
    """
    configs = []

    # Vary exploration weight
    for c in [0.5, 1.0, 1.414, 2.0]:
        configs.append(
            MCTSConfig(
                name=f"exploration_c{c}",
                num_iterations=50,
                exploration_weight=c,
                seed=42,
            )
        )

    # Vary progressive widening
    for k in [0.5, 1.0, 2.0]:
        for alpha in [0.3, 0.5, 0.7]:
            configs.append(
                MCTSConfig(
                    name=f"widening_k{k}_a{alpha}",
                    num_iterations=50,
                    progressive_widening_k=k,
                    progressive_widening_alpha=alpha,
                    seed=42,
                )
            )

    return configs


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================


async def demonstrate_deterministic_mcts():
    """
    Demonstrate deterministic MCTS behavior.

    Shows that same seed produces identical results.
    """
    print("=== Deterministic MCTS Demonstration ===\n")

    # Verify determinism
    print("1. Verifying deterministic behavior...")
    result = await run_determinism_verification(num_runs=3, num_iterations=50, seed=42)
    print(f"   Is deterministic: {result['is_deterministic']}")
    if result["inconsistencies"]:
        print(f"   Inconsistencies: {result['inconsistencies']}")
    print(f"   Results: {result['results']}\n")

    # Test progressive widening
    print("2. Testing progressive widening...")
    widening_result = await test_progressive_widening_behavior(k=1.0, alpha=0.5)
    print(f"   Root children: {widening_result['root_children']}")
    print(f"   Total nodes: {widening_result['total_nodes']}")
    print(f"   Widening controlled: {widening_result['widening_controlled']}\n")

    # Test caching
    print("3. Testing cache effectiveness...")
    cache_result = await test_cache_effectiveness(num_iterations=100)
    print(f"   Cache hit rate: {cache_result['cache_hit_rate']:.2%}")
    print(f"   Cache hits: {cache_result['cache_hits']}")
    print(f"   Cache misses: {cache_result['cache_misses']}\n")

    # Use known-good tree
    print("4. Testing with known-good tree structure...")
    root, optimal_action, optimal_value = KnownGoodTreeFixture.create_known_optimal_tree()
    engine = MCTSEngine(seed=42)
    selected_action = engine._select_best_action(root, SelectionPolicy.MAX_VISITS)
    print(f"   Known optimal action: {optimal_action}")
    print(f"   Selected action: {selected_action}")
    print(f"   Match: {selected_action == optimal_action}\n")

    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    asyncio.run(demonstrate_deterministic_mcts())
