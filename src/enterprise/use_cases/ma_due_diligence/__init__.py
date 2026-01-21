"""
M&A Due Diligence Use Case.

MCTS-guided multi-agent system that explores thousands of due diligence
pathways simultaneously, identifying hidden risks and synergies in
corporate acquisitions.

Target Buyers: Investment banks, PE firms (Goldman, KKR, Blackstone)
Revenue Potential: $5M-20M ARR
"""

from __future__ import annotations

from .actions import ACTIONS_BY_PHASE, apply_action, get_available_actions
from .agents import (
    ComplianceCheckAgent,
    DocumentAnalysisAgent,
    RiskIdentificationAgent,
    SynergyExplorationAgent,
)
from .reward import MADueDiligenceReward
from .state import (
    AnalyzedDocument,
    DueDiligencePhase,
    IdentifiedRisk,
    MADueDiligenceState,
    RiskLevel,
    SynergyOpportunity,
)
from .use_case import MADueDiligence

__all__ = [
    "MADueDiligence",
    "MADueDiligenceState",
    "DueDiligencePhase",
    "IdentifiedRisk",
    "RiskLevel",
    "AnalyzedDocument",
    "SynergyOpportunity",
    "DocumentAnalysisAgent",
    "RiskIdentificationAgent",
    "SynergyExplorationAgent",
    "ComplianceCheckAgent",
    "MADueDiligenceReward",
    "get_available_actions",
    "apply_action",
    "ACTIONS_BY_PHASE",
]
