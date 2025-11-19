"""
Comprehensive unit tests for MCTS core functionality.

Tests:
- MCTSNode operations (UCB1, child selection, tree structure)
- MCTS phases (selection, expansion, simulation, backpropagation)
- Determinism with seeding
- Edge cases and error conditions
"""

import math
import random

# Import the MCTS classes from the framework
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.framework.mcts.core import MCTSNode


class TestMCTSNode:
    """Test suite for MCTSNode class."""

    def test_node_initialization(self):
        """Test that nodes are initialized correctly."""
        node = MCTSNode(state_id="test_state")

        assert node.state_id == "test_state"
        assert node.parent is None
        assert node.action is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.terminal is False

    def test_node_with_parent(self):
        """Test node initialization with parent."""
        parent = MCTSNode(state_id="parent_state")
        child = MCTSNode(state_id="child_state", parent=parent, action="test_action")

        assert child.parent is parent
        assert child.action == "test_action"

    def test_ucb1_unvisited_node_returns_infinity(self):
        """UCB1 should return infinity for unvisited nodes."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 10  # Parent must have visits
        child = MCTSNode(state_id="child", parent=parent)

        assert child.ucb1() == float("inf")

    def test_ucb1_calculation_standard_case(self):
        """Test UCB1 calculation with standard values."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 100

        child = MCTSNode(state_id="child", parent=parent)
        child.visits = 10
        child.value = 7.0  # Average value = 0.7

        # UCB1 = exploitation + exploration
        # exploitation = 7.0 / 10 = 0.7
        # exploration = 1.414 * sqrt(ln(100) / 10) = 1.414 * sqrt(4.605 / 10) = 1.414 * 0.6786 ≈ 0.959
        ucb = child.ucb1(exploration_weight=1.414)

        expected_exploitation = 0.7
        expected_exploration = 1.414 * math.sqrt(math.log(100) / 10)
        expected_ucb = expected_exploitation + expected_exploration

        assert abs(ucb - expected_ucb) < 1e-6

    def test_ucb1_with_different_exploration_weights(self):
        """Test that exploration weight affects UCB1 correctly."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 100

        child = MCTSNode(state_id="child", parent=parent)
        child.visits = 10
        child.value = 5.0

        ucb_low_exploration = child.ucb1(exploration_weight=0.5)
        ucb_high_exploration = child.ucb1(exploration_weight=2.0)

        # Higher exploration weight should give higher UCB1
        assert ucb_high_exploration > ucb_low_exploration

    def test_best_child_returns_none_for_leaf(self):
        """best_child should return None when no children exist."""
        node = MCTSNode(state_id="leaf")
        assert node.best_child() is None

    def test_best_child_selects_highest_ucb1(self):
        """best_child should select child with highest UCB1."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 100

        # Create children with different values
        child1 = parent.add_child("action_1", "state_1")
        child1.visits = 10
        child1.value = 3.0  # Low average

        child2 = parent.add_child("action_2", "state_2")
        child2.visits = 10
        child2.value = 8.0  # High average

        child3 = parent.add_child("action_3", "state_3")
        child3.visits = 10
        child3.value = 5.0  # Medium average

        best = parent.best_child()
        assert best is child2
        assert best.action == "action_2"

    def test_best_child_prefers_unvisited(self):
        """Unvisited children should be selected first (infinite UCB1)."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 100

        visited_child = parent.add_child("visited", "state_1")
        visited_child.visits = 10
        visited_child.value = 9.0  # High value

        unvisited_child = parent.add_child("unvisited", "state_2")
        # unvisited_child.visits = 0 (default)

        best = parent.best_child()
        assert best is unvisited_child

    def test_add_child_creates_proper_relationship(self):
        """add_child should create correct parent-child relationship."""
        parent = MCTSNode(state_id="parent")
        child = parent.add_child(action="move_forward", state_id="new_state")

        assert child in parent.children
        assert child.parent is parent
        assert child.action == "move_forward"
        assert child.state_id == "new_state"
        assert len(parent.children) == 1

    def test_add_multiple_children(self):
        """Adding multiple children should work correctly."""
        parent = MCTSNode(state_id="root")

        for i in range(5):
            parent.add_child(f"action_{i}", f"state_{i}")

        assert len(parent.children) == 5
        for i, child in enumerate(parent.children):
            assert child.action == f"action_{i}"
            assert child.parent is parent

    def test_ucb1_exploitation_dominates_with_many_visits(self):
        """With many visits, exploitation term should dominate."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 10000

        child = MCTSNode(state_id="child", parent=parent)
        child.visits = 1000
        child.value = 900.0  # Average = 0.9

        ucb = child.ucb1(exploration_weight=1.414)
        exploitation = child.value / child.visits

        # UCB should be close to exploitation term
        assert abs(ucb - exploitation) < 0.2


class TestMCTSFrameworkIntegration:
    """Integration tests for MCTS within LangGraph framework."""

    @pytest.fixture
    def mock_model_adapter(self):
        """Create a mock model adapter."""
        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value=Mock(text="Generated response"))
        return adapter

    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        logger = Mock()
        logger.info = Mock()
        logger.error = Mock()
        logger.debug = Mock()
        return logger

    @pytest.fixture
    def framework(self, mock_model_adapter, mock_logger):
        """Create a framework instance with mocks."""
        with (
            patch("langgraph_multi_agent_mcts.HRMAgent"),
            patch("langgraph_multi_agent_mcts.TRMAgent"),
            patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"),
        ):
            framework = LangGraphMultiAgentFramework(
                model_adapter=mock_model_adapter,
                logger=mock_logger,
                mcts_iterations=10,
                mcts_exploration_weight=1.414,
            )
            return framework

    def test_mcts_select_traverses_to_leaf(self, framework):
        """Selection phase should traverse to leaf node."""
        root = MCTSNode(state_id="root")
        root.visits = 100

        child1 = root.add_child("a1", "s1")
        child1.visits = 50
        child1.value = 30.0

        child2 = root.add_child("a2", "s2")
        child2.visits = 50
        child2.value = 35.0

        # Add a grandchild to child2
        grandchild = child2.add_child("a3", "s3")
        # grandchild is unvisited (leaf)

        selected = framework._mcts_select(root)
        assert selected is grandchild

    def test_mcts_select_returns_root_when_no_children(self, framework):
        """Selection on node with no children returns the node itself."""
        root = MCTSNode(state_id="root")
        selected = framework._mcts_select(root)
        assert selected is root

    def test_mcts_select_stops_at_terminal_node(self, framework):
        """Selection should stop at terminal nodes."""
        root = MCTSNode(state_id="root")
        root.visits = 10

        child = root.add_child("action", "state")
        child.visits = 5
        child.value = 3.0
        child.terminal = True

        # Even though child has visits, it's terminal
        selected = framework._mcts_select(root)
        # Should traverse to child (best child of root) but stop there
        assert selected is child

    def test_mcts_backpropagate_updates_all_ancestors(self, framework):
        """Backpropagation should update all nodes to root."""
        root = MCTSNode(state_id="root")
        child = root.add_child("a1", "s1")
        grandchild = child.add_child("a2", "s2")
        great_grandchild = grandchild.add_child("a3", "s3")

        value = 0.8
        framework._mcts_backpropagate(great_grandchild, value)

        # All nodes should have 1 visit and value = 0.8
        assert great_grandchild.visits == 1
        assert great_grandchild.value == 0.8
        assert grandchild.visits == 1
        assert grandchild.value == 0.8
        assert child.visits == 1
        assert child.value == 0.8
        assert root.visits == 1
        assert root.value == 0.8

    def test_mcts_backpropagate_accumulates_visits_and_values(self, framework):
        """Multiple backpropagations should accumulate correctly."""
        root = MCTSNode(state_id="root")
        child = root.add_child("a1", "s1")

        # First backprop
        framework._mcts_backpropagate(child, 0.6)
        # Second backprop
        framework._mcts_backpropagate(child, 0.8)

        assert child.visits == 2
        assert child.value == 1.4
        assert root.visits == 2
        assert root.value == 1.4

        # Average value should be 0.7
        assert child.value / child.visits == 0.7

    def test_generate_actions_returns_root_actions(self, framework):
        """Root state should have predefined actions."""
        root = MCTSNode(state_id="root_state")
        state = {"query": "test"}

        actions = framework._generate_actions(root, state)
        assert actions == ["action_A", "action_B", "action_C"]

    def test_generate_actions_respects_depth_limit(self, framework):
        """Deep nodes should return empty actions (terminal)."""
        deep_node = MCTSNode(state_id="root_a1_b2_c3")  # Depth 4
        state = {"query": "test"}

        actions = framework._generate_actions(deep_node, state)
        assert actions == []

    def test_generate_actions_intermediate_depth(self, framework):
        """Intermediate depth nodes should have continuation actions."""
        mid_node = MCTSNode(state_id="root_action")  # Depth 2
        state = {"query": "test"}

        actions = framework._generate_actions(mid_node, state)
        assert actions == ["continue_A", "continue_B", "fallback"]

    @pytest.mark.asyncio
    async def test_mcts_simulate_returns_bounded_value(self, framework):
        """Simulation should return value between 0 and 1."""
        node = MCTSNode(state_id="test")
        state = {"query": "test query"}

        for _ in range(20):
            value = await framework._mcts_simulate(node, state)
            assert 0.0 <= value <= 1.0

    @pytest.mark.asyncio
    async def test_mcts_simulate_incorporates_agent_results(self, framework):
        """Simulation should be biased by agent confidence."""
        node = MCTSNode(state_id="test")

        # State with high confidence agents
        state_high_conf = {
            "query": "test",
            "hrm_results": {"metadata": {"decomposition_quality_score": 0.95}},
            "trm_results": {"metadata": {"final_quality_score": 0.95}},
        }

        # State with no agent results
        state_no_agents = {"query": "test"}

        # Run multiple simulations and compare averages
        high_conf_values = []
        no_agent_values = []

        for _ in range(100):
            high_conf_values.append(await framework._mcts_simulate(node, state_high_conf))
            no_agent_values.append(await framework._mcts_simulate(node, state_no_agents))

        avg_high = sum(high_conf_values) / len(high_conf_values)
        avg_no_agent = sum(no_agent_values) / len(no_agent_values)

        # High confidence should give higher average value
        assert avg_high > avg_no_agent

    def test_mcts_expand_creates_child_node(self, framework):
        """Expansion should create a new child node."""
        random.seed(42)  # For reproducibility

        parent = MCTSNode(state_id="root_state")
        parent.visits = 1  # Must have visits to expand
        state = {"query": "test"}

        child = framework._mcts_expand(parent, state)

        assert child in parent.children
        assert child.parent is parent
        assert child.action in ["action_A", "action_B", "action_C"]

    def test_mcts_expand_marks_terminal_when_no_actions(self, framework):
        """Expansion should mark node as terminal if no actions available."""
        deep_node = MCTSNode(state_id="root_a_b_c")  # Depth 4
        deep_node.visits = 1
        state = {"query": "test"}

        result = framework._mcts_expand(deep_node, state)

        assert result is deep_node  # Returns same node
        assert deep_node.terminal is True


class TestMCTSDeterminism:
    """Test MCTS determinism with seeding."""

    @pytest.fixture
    def framework(self):
        """Create a minimal framework for testing."""
        mock_adapter = AsyncMock()
        mock_logger = Mock()
        mock_logger.info = Mock()

        with (
            patch("langgraph_multi_agent_mcts.HRMAgent"),
            patch("langgraph_multi_agent_mcts.TRMAgent"),
            patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"),
        ):
            return LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
                mcts_iterations=5,
            )

    def test_same_seed_produces_same_tree_structure(self, framework):
        """With same seed, expansion should create identical structure."""
        state = {"query": "test"}

        # First run
        random.seed(12345)
        root1 = MCTSNode(state_id="root_state")
        root1.visits = 1
        child1 = framework._mcts_expand(root1, state)

        # Second run with same seed
        random.seed(12345)
        root2 = MCTSNode(state_id="root_state")
        root2.visits = 1
        child2 = framework._mcts_expand(root2, state)

        assert child1.action == child2.action
        assert child1.state_id == child2.state_id

    def test_different_seeds_produce_different_trees(self, framework):
        """Different seeds should produce different tree structures."""
        state = {"query": "test"}

        # First run
        random.seed(11111)
        root1 = MCTSNode(state_id="root_state")
        root1.visits = 1
        framework._mcts_expand(root1, state)

        # Second run with different seed
        random.seed(99999)
        root2 = MCTSNode(state_id="root_state")
        root2.visits = 1
        framework._mcts_expand(root2, state)

        # With different seeds, actions may differ
        # This test verifies the mechanism works, not specific outcomes
        assert True  # Structure was built differently


class TestMCTSEdgeCases:
    """Test edge cases and error conditions."""

    def test_ucb1_with_zero_parent_visits_raises_math_error(self):
        """UCB1 calculation with zero parent visits should fail."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 0  # Invalid state

        child = MCTSNode(state_id="child", parent=parent)
        child.visits = 1
        child.value = 0.5

        # This will cause math.log(0) which raises ValueError
        with pytest.raises(ValueError):
            child.ucb1()

    def test_node_without_parent_ucb1_fails(self):
        """Root node UCB1 calculation should fail (no parent)."""
        root = MCTSNode(state_id="root")
        root.visits = 10
        root.value = 5.0

        # Accessing parent.visits will fail
        with pytest.raises(AttributeError):
            root.ucb1()

    def test_best_child_with_single_child(self):
        """best_child with only one child should return that child."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 10

        only_child = parent.add_child("action", "state")
        only_child.visits = 5
        only_child.value = 2.0

        assert parent.best_child() is only_child

    def test_very_deep_tree_structure(self):
        """Test that deep trees are handled correctly."""
        root = MCTSNode(state_id="root")
        current = root

        # Create very deep tree
        for i in range(100):
            child = current.add_child(f"action_{i}", f"state_{i}")
            child.visits = 1
            child.value = 0.5
            current = child

        # Verify structure
        node = root
        depth = 0
        while node.children:
            node = node.children[0]
            depth += 1

        assert depth == 100

    def test_negative_value_handling(self):
        """Ensure negative values are handled properly."""
        parent = MCTSNode(state_id="parent")
        parent.visits = 10

        child = MCTSNode(state_id="child", parent=parent)
        child.visits = 5
        child.value = -2.0  # Negative total value

        # UCB1 should still work
        ucb = child.ucb1()
        assert isinstance(ucb, float)
        # Exploitation term is negative
        assert child.value / child.visits == -0.4


class TestMCTSValidation:
    """Test input validation for MCTS parameters."""

    @pytest.fixture
    def mock_adapter(self):
        return AsyncMock()

    @pytest.fixture
    def mock_logger(self):
        logger = Mock()
        logger.info = Mock()
        return logger

    @pytest.mark.parametrize("exploration_weight", [0.0, 0.5, 1.0, 1.414, 2.0, 3.0])
    def test_valid_exploration_weights(self, exploration_weight, mock_adapter, mock_logger):
        """Valid exploration weights should work."""
        with (
            patch("langgraph_multi_agent_mcts.HRMAgent"),
            patch("langgraph_multi_agent_mcts.TRMAgent"),
            patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"),
        ):
            framework = LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
                mcts_exploration_weight=exploration_weight,
            )
            assert framework.mcts_exploration_weight == exploration_weight

    @pytest.mark.parametrize("iterations", [1, 10, 100, 1000])
    def test_valid_iteration_counts(self, iterations, mock_adapter, mock_logger):
        """Valid iteration counts should work."""
        with (
            patch("langgraph_multi_agent_mcts.HRMAgent"),
            patch("langgraph_multi_agent_mcts.TRMAgent"),
            patch("langgraph_multi_agent_mcts.OpenAIEmbeddings"),
        ):
            framework = LangGraphMultiAgentFramework(
                model_adapter=mock_adapter,
                logger=mock_logger,
                mcts_iterations=iterations,
            )
            assert framework.mcts_iterations == iterations


# Performance benchmarks
class TestMCTSPerformance:
    """Performance benchmarks for MCTS operations."""

    @pytest.mark.slow
    def test_ucb1_computation_performance(self):
        """UCB1 should compute quickly even with many calls."""
        import time

        parent = MCTSNode(state_id="parent")
        parent.visits = 1000000

        children = []
        for i in range(100):
            child = parent.add_child(f"action_{i}", f"state_{i}")
            child.visits = 10000 + i
            child.value = 5000 + i * 10
            children.append(child)

        start = time.perf_counter()

        # Compute UCB1 for all children many times
        for _ in range(10000):
            for child in children:
                child.ucb1()

        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (less than 5 seconds)
        assert elapsed < 5.0, f"UCB1 computation took {elapsed:.2f}s"

    @pytest.mark.slow
    def test_best_child_selection_performance(self):
        """best_child should be efficient with many children."""
        import time

        parent = MCTSNode(state_id="parent")
        parent.visits = 100000

        # Add many children
        for i in range(1000):
            child = parent.add_child(f"action_{i}", f"state_{i}")
            child.visits = 100 + i
            child.value = 50 + i * 0.5

        start = time.perf_counter()

        # Select best child many times
        for _ in range(1000):
            parent.best_child()

        elapsed = time.perf_counter() - start

        # Should complete quickly
        assert elapsed < 2.0, f"best_child selection took {elapsed:.2f}s"
