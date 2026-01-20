"""
Adapter for integrating enterprise use cases with the meta-controller.

Extends the existing Meta-Controller pattern to support
domain-specific routing decisions for enterprise use cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.meta_controller.base import (
        AbstractMetaController,
    )

from ..config.enterprise_settings import EnterpriseDomain


@dataclass
class EnterpriseMetaControllerFeatures:
    """Extended features for enterprise routing decisions."""

    # Base meta-controller features
    hrm_confidence: float = 0.0
    trm_confidence: float = 0.0
    mcts_value: float = 0.0
    consensus_score: float = 0.0
    last_agent: str = "none"
    iteration: int = 0
    query_length: int = 0
    has_rag_context: bool = False

    # Enterprise-specific features
    detected_domain: EnterpriseDomain | None = None
    domain_confidence: float = 0.0
    requires_compliance_check: bool = False
    estimated_complexity: float = 0.5
    regulatory_jurisdictions: list[str] = field(default_factory=list)
    is_time_sensitive: bool = False
    requires_expert_review: bool = False


class EnterpriseMetaControllerAdapter:
    """
    Adapts enterprise use cases for meta-controller routing.

    Provides domain detection and routing logic for
    enterprise-specific queries, integrating with the
    existing meta-controller infrastructure.

    Example:
        >>> adapter = EnterpriseMetaControllerAdapter()
        >>> features = adapter.extract_enterprise_features(state, query)
        >>> route = adapter.route_to_enterprise(features)
    """

    # Extended agent names for enterprise
    ENTERPRISE_AGENTS = [
        "hrm",
        "trm",
        "mcts",
        "enterprise_ma",
        "enterprise_clinical",
        "enterprise_regulatory",
    ]

    def __init__(
        self,
        base_controller: AbstractMetaController | None = None,
        domain_detection_threshold: float = 0.05,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            base_controller: Optional base meta-controller
            domain_detection_threshold: Confidence threshold for domain detection
            logger: Optional logger instance
        """
        self._base_controller = base_controller
        self._detection_threshold = domain_detection_threshold
        self._logger = logger or logging.getLogger(__name__)

        # Domain detection patterns
        self._domain_patterns = {
            EnterpriseDomain.MA_DUE_DILIGENCE: {
                "keywords": [
                    "acquisition",
                    "merger",
                    "due diligence",
                    "m&a",
                    "target company",
                    "synergy",
                    "valuation",
                    "deal structure",
                    "acquirer",
                    "buyout",
                    "takeover",
                    "corporate transaction",
                ],
                "weight": 1.0,
            },
            EnterpriseDomain.CLINICAL_TRIAL: {
                "keywords": [
                    "clinical trial",
                    "fda",
                    "ema",
                    "regulatory approval",
                    "phase 1",
                    "phase 2",
                    "phase 3",
                    "endpoint",
                    "cohort",
                    "sample size",
                    "statistical power",
                    "placebo",
                    "randomized",
                ],
                "weight": 1.0,
            },
            EnterpriseDomain.REGULATORY_COMPLIANCE: {
                "keywords": [
                    "compliance",
                    "regulation",
                    "audit",
                    "enforcement",
                    "jurisdiction",
                    "gdpr",
                    "sox",
                    "hipaa",
                    "gap analysis",
                    "remediation",
                    "policy",
                    "regulatory",
                    "obligation",
                ],
                "weight": 1.0,
            },
        }

    def detect_domain(self, query: str) -> tuple[EnterpriseDomain | None, float]:
        """
        Detect enterprise domain from query text.

        Args:
            query: User query

        Returns:
            Tuple of (detected_domain, confidence)
        """
        query_lower = query.lower()
        domain_scores: dict[EnterpriseDomain, float] = {}

        for domain, config in self._domain_patterns.items():
            keywords: list[str] = config["keywords"]
            weight: float = config["weight"]

            matches = sum(1 for kw in keywords if kw in query_lower)
            score = (matches / len(keywords)) * weight if keywords else 0.0
            domain_scores[domain] = score

        if not domain_scores:
            return None, 0.0

        best_domain = max(domain_scores, key=lambda d: domain_scores[d])
        best_score = domain_scores[best_domain]

        if best_score >= self._detection_threshold:
            self._logger.debug(f"Detected domain: {best_domain.value} (confidence={best_score:.2f})")
            return best_domain, best_score

        return None, best_score

    def extract_enterprise_features(
        self,
        state: dict[str, Any],
        query: str,
    ) -> EnterpriseMetaControllerFeatures:
        """
        Extract features including enterprise-specific attributes.

        Args:
            state: Current agent state
            query: User query

        Returns:
            EnterpriseMetaControllerFeatures instance
        """
        # Detect enterprise domain
        detected_domain, domain_confidence = self.detect_domain(query)

        features = EnterpriseMetaControllerFeatures(
            hrm_confidence=state.get("confidence_scores", {}).get("hrm", 0.0),
            trm_confidence=state.get("confidence_scores", {}).get("trm", 0.0),
            mcts_value=state.get("mcts_best_value", 0.0),
            consensus_score=state.get("consensus_score", 0.0),
            last_agent=state.get("last_agent", "none"),
            iteration=state.get("iteration", 0),
            query_length=len(query),
            has_rag_context=bool(state.get("rag_context")),
            detected_domain=detected_domain,
            domain_confidence=domain_confidence,
            requires_compliance_check=self._requires_compliance(query),
            estimated_complexity=self._estimate_complexity(query, state),
            regulatory_jurisdictions=self._extract_jurisdictions(query),
            is_time_sensitive=self._is_time_sensitive(query),
            requires_expert_review=self._requires_expert(query, state),
        )

        return features

    def route_to_enterprise(
        self,
        features: EnterpriseMetaControllerFeatures,
    ) -> str:
        """
        Determine routing to enterprise use case.

        Args:
            features: Extracted features

        Returns:
            Route string (e.g., 'enterprise_ma', 'hrm', etc.)
        """
        if features.detected_domain and features.domain_confidence >= self._detection_threshold:
            domain_routes = {
                EnterpriseDomain.MA_DUE_DILIGENCE: "enterprise_ma",
                EnterpriseDomain.CLINICAL_TRIAL: "enterprise_clinical",
                EnterpriseDomain.REGULATORY_COMPLIANCE: "enterprise_regulatory",
            }
            route = domain_routes.get(features.detected_domain, "hrm")
            self._logger.info(f"Routing to enterprise: {route}")
            return route

        # Fall back based on other features
        if features.requires_compliance_check:
            return "enterprise_regulatory"

        if features.estimated_complexity > 0.7:
            return "mcts"  # Use MCTS for complex queries

        return "hrm"  # Default to HRM

    def _requires_compliance(self, query: str) -> bool:
        """Check if query requires compliance verification."""
        compliance_keywords = [
            "compliance",
            "compliant",
            "regulation",
            "regulatory",
            "audit",
            "legal",
            "jurisdiction",
            "policy",
            "requirement",
            "gdpr",
            "hipaa",
            "sox",
        ]
        return any(kw in query.lower() for kw in compliance_keywords)

    def _estimate_complexity(self, query: str, state: dict[str, Any]) -> float:
        """Estimate query complexity on 0-1 scale."""
        # Length factor
        length_factor = min(len(query) / 1000, 0.3)

        # Technical keyword factor
        technical_keywords = [
            "analyze",
            "evaluate",
            "assess",
            "compare",
            "simulate",
            "optimize",
            "predict",
            "recommend",
        ]
        technical_factor = sum(0.1 for kw in technical_keywords if kw in query.lower())

        # State complexity factor
        state_factor = 0.0
        if state.get("iteration", 0) > 3:
            state_factor = 0.1
        if not state.get("consensus_reached", True):
            state_factor += 0.1

        return min(length_factor + technical_factor + state_factor, 1.0)

    def _extract_jurisdictions(self, query: str) -> list[str]:
        """Extract mentioned jurisdictions from query."""
        jurisdictions = []
        jurisdiction_keywords = {
            "US": ["united states", "us", "american", "federal"],
            "EU": ["european union", "eu", "europe", "gdpr"],
            "UK": ["united kingdom", "uk", "british", "england"],
            "APAC": ["asia pacific", "apac", "asia", "china", "japan"],
        }

        query_lower = query.lower()
        for jurisdiction, keywords in jurisdiction_keywords.items():
            if any(kw in query_lower for kw in keywords):
                jurisdictions.append(jurisdiction)

        return jurisdictions

    def _is_time_sensitive(self, query: str) -> bool:
        """Check if query indicates time sensitivity."""
        time_keywords = [
            "urgent",
            "deadline",
            "asap",
            "immediately",
            "time-sensitive",
            "critical",
            "priority",
        ]
        return any(kw in query.lower() for kw in time_keywords)

    def _requires_expert(self, query: str, state: dict[str, Any]) -> bool:
        """Determine if expert review is required."""
        # Complex queries or low confidence
        if state.get("confidence_scores", {}).get("max", 1.0) < 0.5:
            return True

        # Specific expert-requiring topics
        expert_keywords = [
            "legal opinion",
            "expert review",
            "specialist",
            "regulatory interpretation",
            "complex transaction",
        ]
        return any(kw in query.lower() for kw in expert_keywords)
