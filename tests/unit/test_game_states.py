"""
Unit tests for GameState implementations.

Tests ReasoningState, PlanningState, and DecisionState classes.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import os

import pytest

# Set environment variables before importing modules
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

# Import with graceful fallback
try:
    from src.framework.mcts.game_states import (
        DecisionState,
        PlanningState,
        ReasoningState,
        create_game_state,
    )

    GAME_STATES_AVAILABLE = True
except ImportError:
    GAME_STATES_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not GAME_STATES_AVAILABLE, reason="GameStates module not available"),
]


class TestReasoningState:
    """Test ReasoningState class."""

    @pytest.fixture
    def basic_state(self):
        """Create basic reasoning state."""
        return ReasoningState(
            problem="What is 2 + 2?",
            max_steps=10,
        )

    def test_initial_state(self, basic_state):
        """Test initial state properties."""
        assert basic_state.problem == "What is 2 + 2?"
        assert basic_state.reasoning_steps == []
        assert basic_state.confidence == 0.0
        assert basic_state.max_steps == 10
        assert not basic_state.is_terminal()

    def test_get_legal_actions_early(self, basic_state):
        """Test legal actions in early stage."""
        actions = basic_state.get_legal_actions()

        assert len(actions) > 0
        action_types = [a["type"] for a in actions]
        assert "decompose" in action_types or "infer" in action_types

    def test_apply_decompose_action(self, basic_state):
        """Test applying decompose action."""
        action = {"type": "decompose", "step": 0}
        new_state = basic_state.apply_action(action)

        assert len(new_state.reasoning_steps) == 1
        assert "[DECOMPOSE]" in new_state.reasoning_steps[0]
        assert new_state.confidence > basic_state.confidence

    def test_apply_infer_action(self, basic_state):
        """Test applying infer action."""
        action = {"type": "infer", "step": 0}
        new_state = basic_state.apply_action(action)

        assert len(new_state.reasoning_steps) == 1
        assert "[INFER]" in new_state.reasoning_steps[0]

    def test_apply_conclude_action(self):
        """Test applying conclude action."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.8,
            max_steps=10,
        )

        action = {"type": "conclude", "step": 2}
        new_state = state.apply_action(action)

        assert "[CONCLUDE]" in new_state.reasoning_steps[-1]
        assert new_state.is_terminal()

    def test_apply_backtrack_action(self):
        """Test applying backtrack action."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.5,
            max_steps=10,
        )

        action = {"type": "backtrack", "step": 2}
        new_state = state.apply_action(action)

        assert len(new_state.reasoning_steps) == 1
        assert new_state.confidence < state.confidence

    def test_is_terminal_max_steps(self, basic_state):
        """Test terminal when max steps reached."""
        state = basic_state
        for i in range(10):
            action = {"type": "infer", "step": i}
            state = state.apply_action(action)

        assert state.is_terminal()

    def test_is_terminal_concluded(self, basic_state):
        """Test terminal when concluded."""
        action = {"type": "conclude", "step": 0}
        new_state = basic_state.apply_action(action)

        assert new_state.is_terminal()

    def test_get_reward_non_terminal(self, basic_state):
        """Test reward is 0 for non-terminal state."""
        assert basic_state.get_reward() == 0.0

    def test_get_reward_terminal(self):
        """Test reward calculation for terminal state."""
        state = ReasoningState(
            problem="Test",
            reasoning_steps=["[CONCLUDE] Answer"],
            confidence=0.8,
            max_steps=10,
        )

        reward = state.get_reward()
        assert 0 < reward <= 1.0

    def test_to_tensor_shape(self, basic_state):
        """Test tensor output shape."""
        tensor = basic_state.to_tensor()

        assert tensor.dim() == 1
        assert tensor.shape[0] > 0

    def test_get_hash_deterministic(self, basic_state):
        """Test hash is deterministic."""
        hash1 = basic_state.get_hash()
        hash2 = basic_state.get_hash()

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_get_hash_different_for_different_states(self):
        """Test different states have different hashes."""
        state1 = ReasoningState(problem="Problem 1")
        state2 = ReasoningState(problem="Problem 2")

        assert state1.get_hash() != state2.get_hash()

    def test_action_to_index(self, basic_state):
        """Test action to index mapping."""
        action = {"type": "decompose"}
        idx = basic_state.action_to_index(action)

        assert isinstance(idx, int)
        assert idx >= 0

    def test_immutability(self, basic_state):
        """Test state immutability after action."""
        original_steps = len(basic_state.reasoning_steps)

        action = {"type": "infer", "step": 0}
        new_state = basic_state.apply_action(action)

        # Original should be unchanged
        assert len(basic_state.reasoning_steps) == original_steps
        assert len(new_state.reasoning_steps) == original_steps + 1


class TestPlanningState:
    """Test PlanningState class."""

    @pytest.fixture
    def basic_state(self):
        """Create basic planning state."""
        return PlanningState(
            goal="Complete task",
            current_state="Start",
            available_actions=["analyze", "execute", "verify", "finish"],
            resources={"time": 10.0, "compute": 5.0},
            max_actions=10,
        )

    def test_initial_state(self, basic_state):
        """Test initial state properties."""
        assert basic_state.goal == "Complete task"
        assert basic_state.current_state == "Start"
        assert len(basic_state.available_actions) == 4
        assert basic_state.resources["time"] == 10.0
        assert not basic_state.is_terminal()

    def test_get_legal_actions(self, basic_state):
        """Test legal actions based on resources."""
        actions = basic_state.get_legal_actions()

        assert len(actions) > 0
        for action in actions:
            assert "name" in action
            assert "cost" in action

    def test_apply_action_deducts_resources(self, basic_state):
        """Test resource deduction after action."""
        actions = basic_state.get_legal_actions()
        analyze_action = next((a for a in actions if a["name"] == "analyze"), None)

        if analyze_action:
            new_state = basic_state.apply_action(analyze_action)

            # Resources should be deducted
            for resource, _cost in analyze_action["cost"].items():
                assert new_state.resources[resource] < basic_state.resources[resource]

    def test_is_terminal_finish_action(self, basic_state):
        """Test terminal when finish action taken."""
        finish_action = {"name": "finish", "cost": {}, "step": 0}
        new_state = basic_state.apply_action(finish_action)

        assert new_state.is_terminal()

    def test_is_terminal_max_actions(self, basic_state):
        """Test terminal when max actions reached."""
        state = basic_state
        for i in range(10):
            action = {"name": "wait", "cost": {}, "step": i}
            state = state.apply_action(action)

        assert state.is_terminal()

    def test_get_reward_with_finish(self):
        """Test reward when finished properly."""
        state = PlanningState(
            goal="Test",
            current_state="Done",
            completed_actions=[{"name": "finish"}],
            resources={"time": 5.0},
            max_actions=10,
        )

        reward = state.get_reward()
        assert reward > 0

    def test_to_tensor(self, basic_state):
        """Test tensor conversion."""
        tensor = basic_state.to_tensor()

        assert tensor.dim() == 1
        assert tensor.shape[0] > 0

    def test_get_hash(self, basic_state):
        """Test hash generation."""
        hash_val = basic_state.get_hash()

        assert isinstance(hash_val, str)
        assert len(hash_val) == 16


class TestDecisionState:
    """Test DecisionState class."""

    @pytest.fixture
    def basic_state(self):
        """Create basic decision state."""
        return DecisionState(
            context="Choose best option",
            options=[
                {"id": "opt_a", "value": 10},
                {"id": "opt_b", "value": 20},
                {"id": "opt_c", "value": 15},
            ],
            max_evaluations=5,
        )

    def test_initial_state(self, basic_state):
        """Test initial state properties."""
        assert basic_state.context == "Choose best option"
        assert len(basic_state.options) == 3
        assert basic_state.evaluated_options == {}
        assert not basic_state.is_terminal()

    def test_get_legal_actions_evaluate(self, basic_state):
        """Test legal actions include evaluations."""
        actions = basic_state.get_legal_actions()

        eval_actions = [a for a in actions if a["type"] == "evaluate"]
        assert len(eval_actions) == 3  # All options unevaluated

    def test_apply_evaluate_action(self, basic_state):
        """Test applying evaluate action."""
        action = {"type": "evaluate", "option_id": "opt_a", "option": {"id": "opt_a"}}
        new_state = basic_state.apply_action(action)

        assert "opt_a" in new_state.evaluated_options
        assert len(new_state.decision_history) == 1

    def test_apply_compare_action(self):
        """Test applying compare action."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}, {"id": "b"}],
            evaluated_options={"a": 0.5, "b": 0.7},
            max_evaluations=5,
        )

        action = {"type": "compare", "options": ["a", "b"]}
        new_state = state.apply_action(action)

        assert len(new_state.decision_history) == 1
        assert new_state.decision_history[0]["action"] == "compare"

    def test_apply_decide_action(self):
        """Test applying decide action."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            evaluated_options={"a": 0.8},
            max_evaluations=5,
        )

        action = {"type": "decide", "best_option": "a"}
        new_state = state.apply_action(action)

        assert new_state.is_terminal()
        assert new_state.decision_history[-1]["final"] is True

    def test_get_reward(self):
        """Test reward calculation."""
        state = DecisionState(
            context="Test",
            options=[{"id": "a"}],
            evaluated_options={"a": 0.8},
            decision_history=[{"final": True}],
            max_evaluations=5,
        )

        reward = state.get_reward()
        assert 0 <= reward <= 1.0

    def test_to_tensor(self, basic_state):
        """Test tensor conversion."""
        tensor = basic_state.to_tensor()

        assert tensor.dim() == 1

    def test_action_to_index(self, basic_state):
        """Test action to index mapping."""
        eval_idx = basic_state.action_to_index({"type": "evaluate"})
        compare_idx = basic_state.action_to_index({"type": "compare"})
        decide_idx = basic_state.action_to_index({"type": "decide"})

        assert eval_idx == 0
        assert compare_idx == 1
        assert decide_idx == 2


class TestCreateGameState:
    """Test create_game_state factory function."""

    def test_create_reasoning_state(self):
        """Test creating reasoning state."""
        state = create_game_state(
            "reasoning",
            problem="Test problem",
            max_steps=5,
        )

        assert isinstance(state, ReasoningState)
        assert state.problem == "Test problem"
        assert state.max_steps == 5

    def test_create_planning_state(self):
        """Test creating planning state."""
        state = create_game_state(
            "planning",
            goal="Test goal",
            current_state="Start",
            available_actions=["action1"],
        )

        assert isinstance(state, PlanningState)
        assert state.goal == "Test goal"

    def test_create_decision_state(self):
        """Test creating decision state."""
        state = create_game_state(
            "decision",
            context="Test context",
            options=[{"id": "opt1"}],
        )

        assert isinstance(state, DecisionState)
        assert state.context == "Test context"

    def test_invalid_state_type(self):
        """Test invalid state type raises error."""
        with pytest.raises(ValueError, match="Unknown state type"):
            create_game_state("invalid_type")

    def test_ignores_invalid_kwargs(self):
        """Test factory ignores invalid kwargs."""
        state = create_game_state(
            "reasoning",
            problem="Test",
            invalid_field="should_be_ignored",
        )

        assert isinstance(state, ReasoningState)
        assert not hasattr(state, "invalid_field")
