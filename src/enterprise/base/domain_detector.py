"""
Centralized domain detection for enterprise use cases.

This module provides a single source of truth for domain detection logic,
eliminating duplication across the enterprise module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.enterprise.config.enterprise_settings import EnterpriseDomain


@dataclass
class DomainPattern:
    """Configuration for a domain's detection pattern."""

    keywords: list[str]
    weight: float = 1.0


@dataclass
class DetectionResult:
    """Result of domain detection."""

    domain: EnterpriseDomain | None
    confidence: float
    all_scores: dict[EnterpriseDomain, float] = field(default_factory=dict)

    @property
    def is_detected(self) -> bool:
        """Check if a domain was confidently detected."""
        return self.domain is not None


class DomainDetector:
    """
    Centralized domain detection for enterprise use cases.

    This class provides a single source of truth for detecting which
    enterprise domain a query belongs to, based on keyword matching.

    Example:
        >>> detector = DomainDetector()
        >>> result = detector.detect("Analyze acquisition target TestCo")
        >>> print(result.domain)  # EnterpriseDomain.MA_DUE_DILIGENCE
    """

    # Default patterns for each domain
    DEFAULT_PATTERNS: dict[EnterpriseDomain, DomainPattern] = {
        EnterpriseDomain.MA_DUE_DILIGENCE: DomainPattern(
            keywords=[
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
            weight=1.0,
        ),
        EnterpriseDomain.CLINICAL_TRIAL: DomainPattern(
            keywords=[
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
            weight=1.0,
        ),
        EnterpriseDomain.REGULATORY_COMPLIANCE: DomainPattern(
            keywords=[
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
            weight=1.0,
        ),
    }

    # Keywords that indicate compliance requirements
    COMPLIANCE_KEYWORDS: list[str] = [
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

    # Keywords for complexity estimation
    TECHNICAL_KEYWORDS: list[str] = [
        "analyze",
        "evaluate",
        "compare",
        "assess",
        "optimize",
        "design",
        "implement",
        "integrate",
        "transform",
    ]

    def __init__(
        self,
        detection_threshold: float = 0.05,
        patterns: dict[EnterpriseDomain, DomainPattern] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the domain detector.

        Args:
            detection_threshold: Minimum confidence score to consider a domain detected
            patterns: Optional custom patterns (defaults to DEFAULT_PATTERNS)
            logger: Optional logger instance
        """
        self._threshold = detection_threshold
        self._patterns = patterns or self.DEFAULT_PATTERNS.copy()
        self._logger = logger or logging.getLogger(__name__)

    @property
    def detection_threshold(self) -> float:
        """Get the current detection threshold."""
        return self._threshold

    @detection_threshold.setter
    def detection_threshold(self, value: float) -> None:
        """Set the detection threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Detection threshold must be between 0.0 and 1.0")
        self._threshold = value

    def detect(self, query: str) -> DetectionResult:
        """
        Detect the enterprise domain from a query.

        Args:
            query: The user query to analyze

        Returns:
            DetectionResult with domain, confidence, and all scores
        """
        query_lower = query.lower()
        domain_scores: dict[EnterpriseDomain, float] = {}

        for domain, pattern in self._patterns.items():
            matches = sum(1 for kw in pattern.keywords if kw in query_lower)
            score = (matches / len(pattern.keywords)) * pattern.weight if pattern.keywords else 0.0
            domain_scores[domain] = score

        if not domain_scores:
            return DetectionResult(domain=None, confidence=0.0, all_scores={})

        best_domain = max(domain_scores, key=lambda d: domain_scores[d])
        best_score = domain_scores[best_domain]

        if best_score >= self._threshold:
            self._logger.debug(f"Detected domain: {best_domain.value} (confidence={best_score:.3f})")
            return DetectionResult(
                domain=best_domain,
                confidence=best_score,
                all_scores=domain_scores,
            )

        return DetectionResult(
            domain=None,
            confidence=best_score,
            all_scores=domain_scores,
        )

    def requires_compliance(self, query: str) -> bool:
        """
        Check if a query requires compliance verification.

        Args:
            query: The user query to check

        Returns:
            True if the query contains compliance-related keywords
        """
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.COMPLIANCE_KEYWORDS)

    def estimate_complexity(self, query: str, state: dict[str, Any] | None = None) -> float:
        """
        Estimate the complexity of a query on a 0-1 scale.

        Args:
            query: The user query
            state: Optional state dictionary with additional context

        Returns:
            Complexity score between 0.0 and 1.0
        """
        state = state or {}

        # Length factor (longer queries tend to be more complex)
        length_factor = min(len(query) / 1000, 0.3)

        # Technical keyword factor
        query_lower = query.lower()
        tech_matches = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw in query_lower)
        tech_factor = min(tech_matches * 0.1, 0.3)

        # State complexity factor
        state_factor = 0.0
        if state.get("rag_context"):
            state_factor += 0.1
        if state.get("previous_responses"):
            state_factor += 0.1

        return min(length_factor + tech_factor + state_factor, 1.0)

    def extract_jurisdictions(self, query: str) -> list[str]:
        """
        Extract regulatory jurisdictions mentioned in a query.

        Args:
            query: The user query

        Returns:
            List of detected jurisdiction codes
        """
        jurisdiction_patterns = {
            "US": ["united states", "usa", "u.s.", "us ", " us", "federal", "sec", "fda"],
            "EU": ["european union", "eu ", " eu", "gdpr", "ema", "european"],
            "UK": ["united kingdom", "uk ", " uk", "fca", "british"],
            "APAC": ["asia", "apac", "china", "japan", "singapore"],
        }

        query_lower = query.lower()
        detected = []

        for code, patterns in jurisdiction_patterns.items():
            if any(p in query_lower for p in patterns):
                detected.append(code)

        return detected if detected else ["US"]  # Default to US

    def get_enterprise_route(self, query: str) -> str:
        """
        Determine the appropriate enterprise route for a query.

        Args:
            query: The user query

        Returns:
            Route string (e.g., 'enterprise_ma', 'enterprise_clinical', etc.)
        """
        result = self.detect(query)

        if result.is_detected:
            route_map = {
                EnterpriseDomain.MA_DUE_DILIGENCE: "enterprise_ma",
                EnterpriseDomain.CLINICAL_TRIAL: "enterprise_clinical",
                EnterpriseDomain.REGULATORY_COMPLIANCE: "enterprise_regulatory",
            }
            return route_map.get(result.domain, "hrm")  # type: ignore[arg-type]

        # Fallback checks
        if self.requires_compliance(query):
            return "enterprise_regulatory"

        if self.estimate_complexity(query) > 0.7:
            return "mcts"

        return "hrm"

    def add_pattern(self, domain: EnterpriseDomain, pattern: DomainPattern) -> None:
        """
        Add or update a domain pattern.

        Args:
            domain: The enterprise domain
            pattern: The detection pattern to use
        """
        self._patterns[domain] = pattern
        self._logger.info(f"Updated pattern for domain: {domain.value}")

    def get_pattern(self, domain: EnterpriseDomain) -> DomainPattern | None:
        """
        Get the pattern for a specific domain.

        Args:
            domain: The enterprise domain

        Returns:
            The DomainPattern or None if not found
        """
        return self._patterns.get(domain)


# Singleton instance for convenient access
_default_detector: DomainDetector | None = None


def get_domain_detector() -> DomainDetector:
    """
    Get the default domain detector singleton.

    Returns:
        The default DomainDetector instance
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = DomainDetector()
    return _default_detector
