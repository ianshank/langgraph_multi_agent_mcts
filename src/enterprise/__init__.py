"""
Enterprise Use Case Framework for LangGraph Multi-Agent MCTS.

This module provides a reusable, extensible framework for building
enterprise AI applications with MCTS-guided multi-agent orchestration.

Supported Use Cases:
- M&A Due Diligence: Strategic exploration of acquisition pathways
- Clinical Trial Design: MCTS simulation of trial configurations
- Regulatory Compliance: Multi-jurisdictional compliance automation

Key Features:
- Dynamic configuration via Pydantic Settings (no hardcoded values)
- Factory pattern for component creation
- Protocol-based interfaces for extensibility
- Full integration with LangGraph and Meta-Controller

Example:
    >>> from src.enterprise import EnterpriseUseCaseFactory, EnterpriseDomain
    >>> factory = EnterpriseUseCaseFactory()
    >>> use_case = factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)
    >>> result = await use_case.process(query="Analyze target company")
"""

from __future__ import annotations

# Base classes and protocols (alphabetically sorted)
from .base.domain_detector import (
    DetectionResult,
    DomainDetector,
    DomainPattern,
    get_domain_detector,
)
from .base.use_case import (
    BaseDomainState,
    BaseUseCase,
    DomainAgentProtocol,
    RewardFunctionProtocol,
    UseCaseProtocol,
)

# Configuration (alphabetically sorted)
from .config.enterprise_settings import (
    BaseUseCaseConfig,
    ClinicalTrialConfig,
    EnterpriseDomain,
    EnterpriseSettings,
    MADueDiligenceConfig,
    RegulatoryComplianceConfig,
    get_enterprise_settings,
    reset_enterprise_settings,
)

# Factories (alphabetically sorted)
from .factories.use_case_factory import (
    EnterpriseUseCaseFactory,
    UseCaseFactory,
)

__all__ = [
    # Base classes and protocols (alphabetically sorted)
    "BaseDomainState",
    "BaseUseCase",
    "DetectionResult",
    "DomainAgentProtocol",
    "DomainDetector",
    "DomainPattern",
    "RewardFunctionProtocol",
    "UseCaseProtocol",
    "get_domain_detector",
    # Configuration (alphabetically sorted)
    "BaseUseCaseConfig",
    "ClinicalTrialConfig",
    "EnterpriseDomain",
    "EnterpriseSettings",
    "MADueDiligenceConfig",
    "RegulatoryComplianceConfig",
    "get_enterprise_settings",
    "reset_enterprise_settings",
    # Factories (alphabetically sorted)
    "EnterpriseUseCaseFactory",
    "UseCaseFactory",
]

__version__ = "1.0.0"
