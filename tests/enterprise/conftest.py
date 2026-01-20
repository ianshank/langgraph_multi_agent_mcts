"""
Shared fixtures for enterprise use case tests.

Provides fixtures for testing the enterprise module including
configurations, mock agents, and sample states.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import enterprise modules with graceful fallback
try:
    from src.enterprise.config.enterprise_settings import (
        ClinicalTrialConfig,
        EnterpriseSettings,
        MADueDiligenceConfig,
        RegulatoryComplianceConfig,
        reset_enterprise_settings,
    )
    from src.enterprise.factories.use_case_factory import EnterpriseUseCaseFactory
    from src.enterprise.use_cases.ma_due_diligence.state import (
        DueDiligencePhase,
        IdentifiedRisk,
        MADueDiligenceState,
        RiskLevel,
        SynergyOpportunity,
    )

    ENTERPRISE_AVAILABLE = True
except ImportError:
    ENTERPRISE_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_enterprise_settings_fixture():
    """Reset enterprise settings before and after each test."""
    if ENTERPRISE_AVAILABLE:
        reset_enterprise_settings()
    yield
    if ENTERPRISE_AVAILABLE:
        reset_enterprise_settings()


@pytest.fixture
def enterprise_env_vars() -> dict[str, str]:
    """Provide environment variable overrides for enterprise testing."""
    return {
        "ENTERPRISE_ENABLED": "true",
        "MA_DD_ENABLED": "true",
        "MA_DD_MAX_MCTS_ITERATIONS": "10",
        "MA_DD_RISK_THRESHOLD": "0.6",
        "MA_DD_CRITICAL_RISK_THRESHOLD": "0.85",
        "CLINICAL_TRIAL_ENABLED": "true",
        "CLINICAL_TRIAL_MIN_STATISTICAL_POWER": "0.8",
        "REG_COMPLIANCE_ENABLED": "true",
        "REG_COMPLIANCE_RISK_TOLERANCE": "moderate",
    }


@pytest.fixture
def enterprise_settings(enterprise_env_vars) -> EnterpriseSettings:
    """Create test enterprise settings."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    with patch.dict(os.environ, enterprise_env_vars):
        reset_enterprise_settings()
        return EnterpriseSettings()


@pytest.fixture
def ma_config() -> MADueDiligenceConfig:
    """Create M&A Due Diligence configuration for testing."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    return MADueDiligenceConfig(
        enabled=True,
        max_mcts_iterations=10,  # Low for fast tests
        risk_threshold=0.6,
        critical_risk_threshold=0.85,
        confidence_threshold=0.7,
        jurisdictions=["US", "EU"],
    )


@pytest.fixture
def clinical_config() -> ClinicalTrialConfig:
    """Create Clinical Trial configuration for testing."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    return ClinicalTrialConfig(
        enabled=True,
        max_mcts_iterations=10,
        min_statistical_power=0.8,
        alpha_level=0.05,
    )


@pytest.fixture
def compliance_config() -> RegulatoryComplianceConfig:
    """Create Regulatory Compliance configuration for testing."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    return RegulatoryComplianceConfig(
        enabled=True,
        max_mcts_iterations=10,
        risk_tolerance="moderate",
        jurisdictions=["US", "EU", "UK"],
    )


@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create mock LLM client for testing."""
    client = AsyncMock()
    client.generate.return_value = MagicMock(
        content="Test response from LLM",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    client.model = "gpt-4-test"
    client.provider = "openai"
    return client


@pytest.fixture
def use_case_factory(enterprise_settings, mock_llm_client) -> EnterpriseUseCaseFactory:
    """Create factory with test configuration."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    return EnterpriseUseCaseFactory(
        enterprise_settings=enterprise_settings,
        llm_client=mock_llm_client,
    )


@pytest.fixture
def ma_due_diligence_state() -> MADueDiligenceState:
    """Create sample M&A due diligence state."""
    if not ENTERPRISE_AVAILABLE:
        pytest.skip("Enterprise module not available")

    return MADueDiligenceState(
        state_id="test_ma_state_001",
        target_company="TestCo Inc.",
        acquirer_company="AcquirerCorp",
        deal_value=100_000_000.0,
        phase=DueDiligencePhase.FINANCIAL_ANALYSIS,
        risks_identified=[
            IdentifiedRisk(
                risk_id="RISK_001",
                category="financial",
                description="Working capital concerns",
                severity=RiskLevel.MEDIUM,
                probability=0.6,
                impact=0.7,
            ),
        ],
        synergies_found=[
            SynergyOpportunity(
                synergy_id="SYN_001",
                category="cost",
                description="IT consolidation",
                estimated_value=2_000_000.0,
                probability=0.8,
                timeline_months=12,
            ),
        ],
    )


@pytest.fixture
def sample_ma_query() -> str:
    """Provide sample M&A due diligence query."""
    return "Analyze the acquisition target TestCo Inc. for potential M&A due diligence. Focus on financial health and synergy opportunities."


@pytest.fixture
def sample_clinical_query() -> str:
    """Provide sample clinical trial query."""
    return "Design a Phase 2 clinical trial for our new oncology drug. We need FDA approval pathway analysis."


@pytest.fixture
def sample_compliance_query() -> str:
    """Provide sample regulatory compliance query."""
    return "Conduct a GDPR compliance gap analysis for our EU operations and identify remediation priorities."


@pytest.fixture
def sample_ma_context() -> dict:
    """Provide sample context for M&A due diligence."""
    return {
        "target_company": "TestCo Inc.",
        "acquirer_company": "AcquirerCorp",
        "deal_value": 100_000_000,
        "jurisdictions": ["US", "EU"],
        "deal_rationale": "Strategic expansion into new market",
    }
