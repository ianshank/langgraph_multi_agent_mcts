"""
MCTS action space for M&A Due Diligence.

Defines available actions at each due diligence phase and
state transition functions for MCTS exploration.
"""

from __future__ import annotations

import copy

from .state import DueDiligencePhase, MADueDiligenceState

# Action definitions per phase - no hardcoded values, easily configurable
ACTIONS_BY_PHASE: dict[DueDiligencePhase, list[str]] = {
    DueDiligencePhase.INITIAL_SCREENING: [
        "analyze_financial_overview",
        "review_company_structure",
        "identify_key_stakeholders",
        "assess_market_position",
        "evaluate_strategic_fit",
    ],
    DueDiligencePhase.FINANCIAL_ANALYSIS: [
        "deep_dive_revenue",
        "analyze_cost_structure",
        "review_working_capital",
        "assess_debt_obligations",
        "evaluate_cash_flow",
        "analyze_profitability_trends",
        "review_financial_projections",
    ],
    DueDiligencePhase.LEGAL_REVIEW: [
        "review_contracts",
        "check_litigation_history",
        "verify_ip_ownership",
        "assess_regulatory_compliance",
        "review_employment_agreements",
        "analyze_liability_exposure",
    ],
    DueDiligencePhase.OPERATIONAL_ASSESSMENT: [
        "evaluate_operations",
        "assess_supply_chain",
        "review_hr_structure",
        "analyze_it_infrastructure",
        "evaluate_facilities",
        "assess_quality_systems",
    ],
    DueDiligencePhase.TECHNOLOGY_EVALUATION: [
        "assess_tech_stack",
        "review_security_posture",
        "evaluate_scalability",
        "check_tech_debt",
        "analyze_data_assets",
        "review_ip_portfolio",
    ],
    DueDiligencePhase.SYNERGY_EXPLORATION: [
        "identify_revenue_synergies",
        "identify_cost_synergies",
        "assess_integration_complexity",
        "estimate_synergy_timeline",
        "evaluate_cultural_fit",
        "analyze_customer_overlap",
    ],
    DueDiligencePhase.RISK_CONSOLIDATION: [
        "consolidate_risks",
        "prioritize_risks",
        "develop_mitigations",
        "escalate_critical_risks",
        "quantify_risk_impact",
        "prepare_risk_matrix",
    ],
    DueDiligencePhase.FINAL_RECOMMENDATION: [
        "generate_recommendation",
        "prepare_executive_summary",
        "finalize_valuation_adjustment",
        "document_key_findings",
        "prepare_integration_plan",
    ],
}

# Meta-actions available in all phases
META_ACTIONS: list[str] = [
    "escalate_to_expert",
    "request_additional_docs",
    "proceed_to_next_phase",
    "revisit_previous_phase",
    "schedule_management_meeting",
]


def get_available_actions(
    state: MADueDiligenceState,
    include_meta: bool = True,
    exclude_recent: bool = True,
    recent_window: int = 5,
) -> list[str]:
    """
    Generate available actions based on current state.

    Args:
        state: Current due diligence state
        include_meta: Whether to include meta-actions
        exclude_recent: Whether to exclude recently taken actions
        recent_window: Number of recent actions to consider

    Returns:
        List of available action strings
    """
    phase_actions = ACTIONS_BY_PHASE.get(state.phase, [])

    # Filter based on progress
    if exclude_recent:
        recent_actions = set(state.action_history[-recent_window:])
        available = [a for a in phase_actions if a not in recent_actions]
    else:
        available = list(phase_actions)

    # Add meta-actions if requested
    if include_meta:
        # Filter meta-actions based on state
        filtered_meta = []
        for action in META_ACTIONS:
            if action == "proceed_to_next_phase":
                # Only allow if not at final phase
                if state.phase != DueDiligencePhase.FINAL_RECOMMENDATION:
                    filtered_meta.append(action)
            elif action == "revisit_previous_phase":
                # Only allow if not at initial phase
                if state.phase != DueDiligencePhase.INITIAL_SCREENING:
                    filtered_meta.append(action)
            else:
                filtered_meta.append(action)

        available.extend(filtered_meta)

    return available


def apply_action(
    state: MADueDiligenceState,
    action: str,
) -> MADueDiligenceState:
    """
    State transition function for MCTS.

    Creates a new state after applying the action.
    Does not modify the original state (immutable transition).

    Args:
        state: Current state
        action: Action to apply

    Returns:
        New state after action
    """
    # Deep copy to ensure immutability
    new_state = copy.deepcopy(state)

    # Update state ID
    action_hash = hash(action) % 10000
    new_state.state_id = f"{state.state_id}_{action_hash}"

    # Record action in history
    new_state.action_history.append(action)

    # Handle phase transitions
    if action == "proceed_to_next_phase":
        new_state = _transition_to_next_phase(new_state)
    elif action == "revisit_previous_phase":
        new_state = _transition_to_previous_phase(new_state)
    else:
        # Apply action effects
        new_state = _apply_action_effects(new_state, action)

    # Update features for MCTS
    new_state.update_features()
    new_state.features["last_action"] = action
    new_state.features["action_count"] = len(new_state.action_history)

    return new_state


def _transition_to_next_phase(state: MADueDiligenceState) -> MADueDiligenceState:
    """Transition state to next phase."""
    phases = list(DueDiligencePhase)
    current_idx = phases.index(state.phase)

    if current_idx < len(phases) - 1:
        state.phase = phases[current_idx + 1]
        state.metadata["phase_transitions"] = state.metadata.get("phase_transitions", [])
        state.metadata["phase_transitions"].append(
            {
                "from": phases[current_idx].name,
                "to": state.phase.name,
                "action_count": len(state.action_history),
            }
        )

    return state


def _transition_to_previous_phase(state: MADueDiligenceState) -> MADueDiligenceState:
    """Transition state to previous phase."""
    phases = list(DueDiligencePhase)
    current_idx = phases.index(state.phase)

    if current_idx > 0:
        state.phase = phases[current_idx - 1]
        state.metadata["backtrack_count"] = state.metadata.get("backtrack_count", 0) + 1

    return state


def _apply_action_effects(
    state: MADueDiligenceState,
    action: str,
) -> MADueDiligenceState:
    """
    Apply the effects of a non-transition action.

    In a full implementation, this would update state based on
    action outcomes (e.g., adding discovered risks, documents analyzed).
    """
    # Track actions by category
    action_category = action.split("_")[0] if "_" in action else action
    state.metadata.setdefault("actions_by_category", {})
    state.metadata["actions_by_category"].setdefault(action_category, 0)
    state.metadata["actions_by_category"][action_category] += 1

    # Simulate action effects based on action type
    if "risk" in action or "litigation" in action or "compliance" in action:
        state.metadata.setdefault("risk_checks", 0)
        state.metadata["risk_checks"] += 1

    if "synerg" in action:
        state.metadata.setdefault("synergy_checks", 0)
        state.metadata["synergy_checks"] += 1

    if "document" in action or "review" in action:
        state.metadata.setdefault("document_reviews", 0)
        state.metadata["document_reviews"] += 1

    return state


def get_action_description(action: str) -> str:
    """
    Get human-readable description for an action.

    Args:
        action: Action name

    Returns:
        Human-readable description
    """
    descriptions = {
        # Initial Screening
        "analyze_financial_overview": "Analyze high-level financial metrics and trends",
        "review_company_structure": "Review corporate structure and ownership",
        "identify_key_stakeholders": "Identify key management and stakeholders",
        "assess_market_position": "Assess competitive position and market share",
        "evaluate_strategic_fit": "Evaluate strategic alignment with acquirer",
        # Financial Analysis
        "deep_dive_revenue": "Detailed analysis of revenue streams and growth",
        "analyze_cost_structure": "Analyze cost structure and margins",
        "review_working_capital": "Review working capital requirements",
        "assess_debt_obligations": "Assess debt levels and covenants",
        "evaluate_cash_flow": "Evaluate cash flow generation and stability",
        # Legal Review
        "review_contracts": "Review material contracts and agreements",
        "check_litigation_history": "Check pending and historical litigation",
        "verify_ip_ownership": "Verify intellectual property ownership",
        "assess_regulatory_compliance": "Assess regulatory compliance status",
        # Meta-actions
        "proceed_to_next_phase": "Move to the next due diligence phase",
        "revisit_previous_phase": "Return to previous phase for additional work",
        "escalate_to_expert": "Escalate issue to domain expert",
        "request_additional_docs": "Request additional documentation",
    }

    return descriptions.get(action, f"Execute action: {action.replace('_', ' ')}")


def get_phase_description(phase: DueDiligencePhase) -> str:
    """
    Get human-readable description for a phase.

    Args:
        phase: Due diligence phase

    Returns:
        Human-readable description
    """
    descriptions = {
        DueDiligencePhase.INITIAL_SCREENING: "Initial target screening and evaluation",
        DueDiligencePhase.FINANCIAL_ANALYSIS: "Deep financial analysis and modeling",
        DueDiligencePhase.LEGAL_REVIEW: "Legal and regulatory due diligence",
        DueDiligencePhase.OPERATIONAL_ASSESSMENT: "Operational capability assessment",
        DueDiligencePhase.TECHNOLOGY_EVALUATION: "Technology and IP evaluation",
        DueDiligencePhase.SYNERGY_EXPLORATION: "Synergy identification and quantification",
        DueDiligencePhase.RISK_CONSOLIDATION: "Risk consolidation and mitigation planning",
        DueDiligencePhase.FINAL_RECOMMENDATION: "Final recommendation and deal structuring",
    }

    return descriptions.get(phase, phase.name)
