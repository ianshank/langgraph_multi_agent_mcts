"""
Tests for M&A Due Diligence use case.

Comprehensive tests for the M&A due diligence implementation
including state management, agents, actions, and rewards.
"""

from __future__ import annotations

import pytest

# Import modules with error handling
try:
    from src.enterprise.use_cases.ma_due_diligence import (
        ACTIONS_BY_PHASE,
        ComplianceCheckAgent,
        DocumentAnalysisAgent,
        DueDiligencePhase,
        IdentifiedRisk,
        MADueDiligence,
        MADueDiligenceReward,
        MADueDiligenceState,
        RiskIdentificationAgent,
        RiskLevel,
        SynergyExplorationAgent,
        apply_action,
        get_available_actions,
    )

    MA_AVAILABLE = True
except ImportError:
    MA_AVAILABLE = False


pytestmark = [pytest.mark.enterprise, pytest.mark.ma_due_diligence]


@pytest.mark.unit
class TestMADueDiligenceState:
    """Tests for M&A Due Diligence state."""

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_state_initialization(self, ma_due_diligence_state):
        """Test state initializes correctly."""
        assert ma_due_diligence_state.state_id == "test_ma_state_001"
        assert ma_due_diligence_state.target_company == "TestCo Inc."
        assert ma_due_diligence_state.domain == "ma_due_diligence"

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_compute_risk_score(self, ma_due_diligence_state):
        """Test risk score computation."""
        score = ma_due_diligence_state.compute_risk_score()

        assert 0 <= score <= 1
        assert ma_due_diligence_state.overall_risk_score == score

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_compute_synergy_confidence(self, ma_due_diligence_state):
        """Test synergy confidence computation."""
        confidence = ma_due_diligence_state.compute_synergy_confidence()

        assert 0 <= confidence <= 1
        assert ma_due_diligence_state.synergy_confidence == confidence

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_get_critical_risks(self):
        """Test filtering critical risks."""
        state = MADueDiligenceState(
            state_id="test",
            risks_identified=[
                IdentifiedRisk(
                    risk_id="R1",
                    category="financial",
                    description="Minor issue",
                    severity=RiskLevel.LOW,
                    probability=0.3,
                    impact=0.2,
                ),
                IdentifiedRisk(
                    risk_id="R2",
                    category="legal",
                    description="Critical issue",
                    severity=RiskLevel.CRITICAL,
                    probability=0.8,
                    impact=0.9,
                ),
            ],
        )

        critical = state.get_critical_risks()
        assert len(critical) == 1
        assert critical[0].risk_id == "R2"

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_phase_progression(self):
        """Test phase index calculation."""
        for i, phase in enumerate(DueDiligencePhase):
            state = MADueDiligenceState(state_id="test", phase=phase)
            assert state.phase.to_index() == i

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_to_summary(self, ma_due_diligence_state):
        """Test summary generation."""
        summary = ma_due_diligence_state.to_summary()

        assert "state_id" in summary
        assert "phase" in summary
        assert "target_company" in summary
        assert "risks_identified" in summary
        assert "synergies_found" in summary

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_to_mcts_state(self, ma_due_diligence_state):
        """Test conversion to MCTS state."""
        ma_due_diligence_state.update_features()
        mcts_state = ma_due_diligence_state.to_mcts_state()

        assert mcts_state.state_id == ma_due_diligence_state.state_id
        assert "domain" in mcts_state.features


@pytest.mark.unit
class TestMADueDiligenceActions:
    """Tests for M&A action space."""

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_actions_defined_for_all_phases(self):
        """Test that actions are defined for all phases."""
        for phase in DueDiligencePhase:
            actions = ACTIONS_BY_PHASE.get(phase, [])
            assert len(actions) > 0, f"No actions defined for {phase}"

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_get_available_actions(self, ma_due_diligence_state):
        """Test getting available actions for state."""
        actions = get_available_actions(ma_due_diligence_state)

        assert len(actions) > 0
        assert isinstance(actions, list)
        assert all(isinstance(a, str) for a in actions)

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_get_available_actions_excludes_recent(self):
        """Test that recent actions are excluded."""
        state = MADueDiligenceState(
            state_id="test",
            phase=DueDiligencePhase.FINANCIAL_ANALYSIS,
            action_history=["deep_dive_revenue", "analyze_cost_structure"],
        )

        actions = get_available_actions(state, exclude_recent=True, recent_window=5)

        # Recent actions should not be in available
        assert "deep_dive_revenue" not in actions
        assert "analyze_cost_structure" not in actions

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_apply_action_creates_new_state(self, ma_due_diligence_state):
        """Test that apply_action creates a new state."""
        original_id = ma_due_diligence_state.state_id

        new_state = apply_action(ma_due_diligence_state, "deep_dive_revenue")

        # Should be new state
        assert new_state.state_id != original_id

        # Original should be unchanged
        assert ma_due_diligence_state.state_id == original_id
        assert len(ma_due_diligence_state.action_history) == 0
        assert len(new_state.action_history) == 1

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_apply_action_phase_transition(self):
        """Test phase transitions via actions."""
        state = MADueDiligenceState(
            state_id="test",
            phase=DueDiligencePhase.INITIAL_SCREENING,
        )

        new_state = apply_action(state, "proceed_to_next_phase")

        assert new_state.phase == DueDiligencePhase.FINANCIAL_ANALYSIS

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_apply_action_backtrack(self):
        """Test backtracking to previous phase."""
        state = MADueDiligenceState(
            state_id="test",
            phase=DueDiligencePhase.LEGAL_REVIEW,
        )

        new_state = apply_action(state, "revisit_previous_phase")

        assert new_state.phase == DueDiligencePhase.FINANCIAL_ANALYSIS


@pytest.mark.unit
class TestMADueDiligenceReward:
    """Tests for M&A reward function."""

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_reward_function_bounds(self, ma_config, ma_due_diligence_state):
        """Test reward values are bounded [0, 1]."""
        reward_fn = MADueDiligenceReward(config=ma_config)

        actions = get_available_actions(ma_due_diligence_state)
        for action in actions[:5]:  # Test subset for speed
            reward = reward_fn.evaluate(ma_due_diligence_state, action, {})
            assert 0 <= reward <= 1, f"Reward {reward} out of bounds for {action}"

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_reward_components(self, ma_config, ma_due_diligence_state):
        """Test reward component extraction."""
        reward_fn = MADueDiligenceReward(config=ma_config)

        components = reward_fn.get_components(
            ma_due_diligence_state,
            "deep_dive_revenue",
            {},
        )

        assert "information_gain" in components
        assert "risk_discovery" in components
        assert "timeline_efficiency" in components
        assert "total_reward" in components

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_reward_diminishing_returns(self, ma_config):
        """Test diminishing returns for repeated actions."""
        reward_fn = MADueDiligenceReward(config=ma_config)

        # Fresh state
        state1 = MADueDiligenceState(state_id="test", action_history=[])
        reward1 = reward_fn.evaluate(state1, "deep_dive_revenue", {})

        # State with repeated similar actions
        state2 = MADueDiligenceState(
            state_id="test",
            action_history=["deep_dive_revenue", "deep_dive_revenue"],
        )
        reward2 = reward_fn.evaluate(state2, "deep_dive_revenue", {})

        # Should have diminishing returns
        assert reward2 < reward1


@pytest.mark.unit
class TestMADueDiligenceAgents:
    """Tests for M&A domain agents."""

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_document_analysis_agent(self, mock_llm_client, ma_due_diligence_state):
        """Test DocumentAnalysisAgent processes queries."""
        agent = DocumentAnalysisAgent(llm_client=mock_llm_client)

        result = await agent.process(
            query="Analyze the financial statements",
            domain_state=ma_due_diligence_state,
            context={},
        )

        assert "agent" in result
        assert result["agent"] == "document_analysis"
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_risk_identification_agent(self, mock_llm_client, ma_due_diligence_state):
        """Test RiskIdentificationAgent identifies risks."""
        agent = RiskIdentificationAgent(llm_client=mock_llm_client)

        result = await agent.process(
            query="Identify financial risks",
            domain_state=ma_due_diligence_state,
            context={},
        )

        assert "agent" in result
        assert result["agent"] == "risk_identification"
        assert "risks" in result
        assert "risk_count" in result

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_synergy_exploration_agent(self, mock_llm_client, ma_due_diligence_state):
        """Test SynergyExplorationAgent finds synergies."""
        agent = SynergyExplorationAgent(llm_client=mock_llm_client)

        result = await agent.process(
            query="Identify synergy opportunities",
            domain_state=ma_due_diligence_state,
            context={},
        )

        assert "agent" in result
        assert result["agent"] == "synergy_exploration"
        assert "synergies" in result
        assert "total_estimated_value" in result

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_compliance_check_agent(self, mock_llm_client, ma_due_diligence_state):
        """Test ComplianceCheckAgent checks compliance."""
        agent = ComplianceCheckAgent(
            config={"jurisdictions": ["US", "EU"]},
            llm_client=mock_llm_client,
        )

        result = await agent.process(
            query="Check regulatory compliance",
            domain_state=ma_due_diligence_state,
            context={},
        )

        assert "agent" in result
        assert result["agent"] == "compliance_check"
        assert "jurisdictions_checked" in result
        assert "US" in result["jurisdictions_checked"]


@pytest.mark.integration
class TestMADueDiligenceUseCase:
    """Integration tests for the full M&A use case."""

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_use_case_initialization(self, ma_config, mock_llm_client):
        """Test use case initializes correctly."""
        use_case = MADueDiligence(
            config=ma_config,
            llm_client=mock_llm_client,
        )

        assert use_case.name == "ma_due_diligence"
        assert use_case.domain == "finance"
        assert use_case.config == ma_config

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    def test_get_initial_state(self, ma_config, mock_llm_client, sample_ma_context):
        """Test initial state creation."""
        use_case = MADueDiligence(config=ma_config, llm_client=mock_llm_client)

        state = use_case.get_initial_state(
            query="Analyze acquisition target",
            context=sample_ma_context,
        )

        assert state.target_company == sample_ma_context["target_company"]
        assert state.deal_value == sample_ma_context["deal_value"]
        assert state.phase == DueDiligencePhase.INITIAL_SCREENING

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_process_query(
        self,
        ma_config,
        mock_llm_client,
        sample_ma_query,
        sample_ma_context,
    ):
        """Test processing a full query."""
        use_case = MADueDiligence(config=ma_config, llm_client=mock_llm_client)

        result = await use_case.process(
            query=sample_ma_query,
            context=sample_ma_context,
            use_mcts=False,  # Skip MCTS for speed
        )

        # Check result structure
        assert "result" in result
        assert "confidence" in result
        assert "domain_state" in result
        assert "agent_results" in result
        assert "risk_analysis" in result
        assert "synergy_analysis" in result

        # Check use case identification
        assert result["use_case"] == "ma_due_diligence"
        assert result["domain"] == "finance"

    @pytest.mark.skipif(not MA_AVAILABLE, reason="MA module not available")
    @pytest.mark.asyncio
    async def test_process_generates_recommendation(
        self,
        ma_config,
        mock_llm_client,
        sample_ma_query,
        sample_ma_context,
    ):
        """Test that processing generates a recommendation."""
        use_case = MADueDiligence(config=ma_config, llm_client=mock_llm_client)

        result = await use_case.process(
            query=sample_ma_query,
            context=sample_ma_context,
            use_mcts=False,
        )

        assert "recommendation" in result
        assert result["recommendation"] is not None
        assert any(keyword in result["recommendation"].upper() for keyword in ["PROCEED", "CAUTION", "NOT PROCEED"])
