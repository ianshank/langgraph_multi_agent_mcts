"""
Unit tests for M&A Due Diligence MCTS action space.

Tests cover:
- ACTIONS_BY_PHASE and META_ACTIONS definitions
- get_available_actions with filtering and meta-action logic
- apply_action immutability and state transitions
- Phase transition helpers (next/previous)
- Action effect tracking in metadata
- get_action_description and get_phase_description
"""

from __future__ import annotations

import pytest

from src.enterprise.use_cases.ma_due_diligence.actions import (
    ACTIONS_BY_PHASE,
    META_ACTIONS,
    apply_action,
    get_action_description,
    get_available_actions,
    get_phase_description,
)
from src.enterprise.use_cases.ma_due_diligence.state import (
    DueDiligencePhase,
    MADueDiligenceState,
)


def _make_state(**kwargs) -> MADueDiligenceState:
    defaults = {
        "state_id": "test-state",
        "target_company": "TargetCo",
        "acquirer_company": "AcquirerCo",
    }
    defaults.update(kwargs)
    return MADueDiligenceState(**defaults)


@pytest.mark.unit
class TestActionDefinitions:
    """Tests for action constants."""

    def test_all_phases_have_actions(self) -> None:
        for phase in DueDiligencePhase:
            assert phase in ACTIONS_BY_PHASE
            assert len(ACTIONS_BY_PHASE[phase]) > 0

    def test_meta_actions_exist(self) -> None:
        assert "escalate_to_expert" in META_ACTIONS
        assert "request_additional_docs" in META_ACTIONS
        assert "proceed_to_next_phase" in META_ACTIONS
        assert "revisit_previous_phase" in META_ACTIONS
        assert "schedule_management_meeting" in META_ACTIONS

    def test_no_duplicate_actions_within_phase(self) -> None:
        for phase, actions in ACTIONS_BY_PHASE.items():
            assert len(actions) == len(set(actions)), f"Duplicate actions in {phase.name}"


@pytest.mark.unit
class TestGetAvailableActions:
    """Tests for get_available_actions function."""

    def test_returns_phase_actions(self) -> None:
        state = _make_state(phase=DueDiligencePhase.INITIAL_SCREENING)
        actions = get_available_actions(state, include_meta=False, exclude_recent=False)
        assert actions == ACTIONS_BY_PHASE[DueDiligencePhase.INITIAL_SCREENING]

    def test_includes_meta_actions(self) -> None:
        state = _make_state(phase=DueDiligencePhase.FINANCIAL_ANALYSIS)
        actions = get_available_actions(state, include_meta=True, exclude_recent=False)
        # Should include proceed_to_next_phase and revisit_previous_phase (not initial or final)
        assert "proceed_to_next_phase" in actions
        assert "revisit_previous_phase" in actions
        assert "escalate_to_expert" in actions

    def test_no_proceed_at_final_phase(self) -> None:
        state = _make_state(phase=DueDiligencePhase.FINAL_RECOMMENDATION)
        actions = get_available_actions(state, include_meta=True, exclude_recent=False)
        assert "proceed_to_next_phase" not in actions

    def test_no_revisit_at_initial_phase(self) -> None:
        state = _make_state(phase=DueDiligencePhase.INITIAL_SCREENING)
        actions = get_available_actions(state, include_meta=True, exclude_recent=False)
        assert "revisit_previous_phase" not in actions

    def test_excludes_recent_actions(self) -> None:
        state = _make_state(
            phase=DueDiligencePhase.INITIAL_SCREENING,
            action_history=["analyze_financial_overview", "review_company_structure"],
        )
        actions = get_available_actions(state, include_meta=False, exclude_recent=True, recent_window=5)
        assert "analyze_financial_overview" not in actions
        assert "review_company_structure" not in actions
        assert "identify_key_stakeholders" in actions

    def test_recent_window_limits_exclusion(self) -> None:
        # Action outside the window should still be available
        history = ["analyze_financial_overview"] + ["filler"] * 10
        state = _make_state(
            phase=DueDiligencePhase.INITIAL_SCREENING,
            action_history=history,
        )
        actions = get_available_actions(state, include_meta=False, exclude_recent=True, recent_window=3)
        # "analyze_financial_overview" is outside the last 3, so it should be available
        assert "analyze_financial_overview" in actions

    def test_exclude_recent_false(self) -> None:
        state = _make_state(
            phase=DueDiligencePhase.INITIAL_SCREENING,
            action_history=["analyze_financial_overview"],
        )
        actions = get_available_actions(state, include_meta=False, exclude_recent=False)
        assert "analyze_financial_overview" in actions


@pytest.mark.unit
class TestApplyAction:
    """Tests for apply_action state transition function."""

    def test_immutability(self) -> None:
        state = _make_state()
        original_id = state.state_id
        original_history = list(state.action_history)
        new_state = apply_action(state, "analyze_financial_overview")
        # Original state unchanged
        assert state.state_id == original_id
        assert state.action_history == original_history
        # New state modified
        assert new_state.state_id != original_id
        assert len(new_state.action_history) == 1

    def test_action_appended_to_history(self) -> None:
        state = _make_state()
        new_state = apply_action(state, "deep_dive_revenue")
        assert new_state.action_history[-1] == "deep_dive_revenue"

    def test_state_id_updated(self) -> None:
        state = _make_state()
        new_state = apply_action(state, "review_contracts")
        assert new_state.state_id.startswith("test-state_")
        assert new_state.state_id != state.state_id

    def test_features_updated(self) -> None:
        state = _make_state()
        new_state = apply_action(state, "analyze_financial_overview")
        assert new_state.features["last_action"] == "analyze_financial_overview"
        assert new_state.features["action_count"] == 1

    def test_proceed_to_next_phase(self) -> None:
        state = _make_state(phase=DueDiligencePhase.INITIAL_SCREENING)
        new_state = apply_action(state, "proceed_to_next_phase")
        assert new_state.phase == DueDiligencePhase.FINANCIAL_ANALYSIS
        assert "phase_transitions" in new_state.metadata
        assert len(new_state.metadata["phase_transitions"]) == 1
        assert new_state.metadata["phase_transitions"][0]["from"] == "INITIAL_SCREENING"
        assert new_state.metadata["phase_transitions"][0]["to"] == "FINANCIAL_ANALYSIS"

    def test_proceed_at_final_phase_stays(self) -> None:
        state = _make_state(phase=DueDiligencePhase.FINAL_RECOMMENDATION)
        new_state = apply_action(state, "proceed_to_next_phase")
        assert new_state.phase == DueDiligencePhase.FINAL_RECOMMENDATION

    def test_revisit_previous_phase(self) -> None:
        state = _make_state(phase=DueDiligencePhase.LEGAL_REVIEW)
        new_state = apply_action(state, "revisit_previous_phase")
        assert new_state.phase == DueDiligencePhase.FINANCIAL_ANALYSIS
        assert new_state.metadata["backtrack_count"] == 1

    def test_revisit_at_initial_phase_stays(self) -> None:
        state = _make_state(phase=DueDiligencePhase.INITIAL_SCREENING)
        new_state = apply_action(state, "revisit_previous_phase")
        assert new_state.phase == DueDiligencePhase.INITIAL_SCREENING

    def test_multiple_backtracks_increment_count(self) -> None:
        state = _make_state(phase=DueDiligencePhase.LEGAL_REVIEW)
        state1 = apply_action(state, "revisit_previous_phase")
        # Move forward again then back
        state2 = apply_action(state1, "proceed_to_next_phase")
        state3 = apply_action(state2, "revisit_previous_phase")
        assert state3.metadata["backtrack_count"] == 2

    def test_action_category_tracking(self) -> None:
        state = _make_state()
        new_state = apply_action(state, "analyze_financial_overview")
        assert "actions_by_category" in new_state.metadata
        assert new_state.metadata["actions_by_category"]["analyze"] == 1

    def test_risk_check_metadata(self) -> None:
        state = _make_state(phase=DueDiligencePhase.LEGAL_REVIEW)
        new_state = apply_action(state, "check_litigation_history")
        assert new_state.metadata.get("risk_checks", 0) == 1

    def test_synergy_check_metadata(self) -> None:
        state = _make_state(phase=DueDiligencePhase.SYNERGY_EXPLORATION)
        new_state = apply_action(state, "identify_revenue_synergies")
        assert new_state.metadata.get("synergy_checks", 0) == 1

    def test_document_review_metadata(self) -> None:
        state = _make_state(phase=DueDiligencePhase.LEGAL_REVIEW)
        new_state = apply_action(state, "review_contracts")
        assert new_state.metadata.get("document_reviews", 0) == 1

    def test_sequential_actions(self) -> None:
        state = _make_state()
        s1 = apply_action(state, "analyze_financial_overview")
        s2 = apply_action(s1, "review_company_structure")
        s3 = apply_action(s2, "proceed_to_next_phase")
        assert len(s3.action_history) == 3
        assert s3.phase == DueDiligencePhase.FINANCIAL_ANALYSIS
        assert s3.features["action_count"] == 3


@pytest.mark.unit
class TestGetActionDescription:
    """Tests for get_action_description function."""

    def test_known_action(self) -> None:
        desc = get_action_description("deep_dive_revenue")
        assert "revenue" in desc.lower()

    def test_unknown_action_fallback(self) -> None:
        desc = get_action_description("some_custom_action")
        assert "some custom action" in desc.lower()

    def test_meta_actions_described(self) -> None:
        desc = get_action_description("proceed_to_next_phase")
        assert "next" in desc.lower()

    def test_returns_string(self) -> None:
        for action_list in ACTIONS_BY_PHASE.values():
            for action in action_list[:2]:  # Spot check a couple per phase
                desc = get_action_description(action)
                assert isinstance(desc, str)
                assert len(desc) > 0


@pytest.mark.unit
class TestGetPhaseDescription:
    """Tests for get_phase_description function."""

    def test_all_phases_described(self) -> None:
        for phase in DueDiligencePhase:
            desc = get_phase_description(phase)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_specific_descriptions(self) -> None:
        desc = get_phase_description(DueDiligencePhase.INITIAL_SCREENING)
        assert "screening" in desc.lower()
        desc = get_phase_description(DueDiligencePhase.FINAL_RECOMMENDATION)
        assert "recommendation" in desc.lower()
