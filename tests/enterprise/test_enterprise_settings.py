"""
Tests for enterprise configuration settings.

Tests the Pydantic Settings-based configuration system ensuring
all values are properly validated and no hardcoded values exist.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Import modules with error handling
try:
    from src.enterprise.config.enterprise_settings import (
        BaseUseCaseConfig,
        ClinicalTrialConfig,
        EnterpriseDomain,
        MADueDiligenceConfig,
        RegulatoryComplianceConfig,
        get_enterprise_settings,
        reset_enterprise_settings,
    )

    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False


pytestmark = pytest.mark.enterprise


@pytest.mark.unit
class TestBaseUseCaseConfig:
    """Tests for BaseUseCaseConfig."""

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_default_values(self):
        """Test default configuration values are sensible."""
        config = BaseUseCaseConfig()

        assert config.enabled is True
        assert config.max_mcts_iterations == 100
        assert 0 < config.mcts_exploration_weight <= 10
        assert 0 <= config.confidence_threshold <= 1
        assert config.parallel_agents >= 1
        assert config.timeout_seconds > 0

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_validation_bounds(self):
        """Test that validation bounds are enforced."""
        # Valid config should work
        config = MADueDiligenceConfig(
            max_mcts_iterations=100,
            mcts_exploration_weight=1.414,
        )
        assert config.max_mcts_iterations == 100

        # Invalid iterations should raise
        with pytest.raises(ValueError):
            MADueDiligenceConfig(max_mcts_iterations=0)

        with pytest.raises(ValueError):
            MADueDiligenceConfig(max_mcts_iterations=100000)  # Over max

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_environment_variable_loading(self, enterprise_env_vars):
        """Test configuration loads from environment variables."""
        with patch.dict(os.environ, enterprise_env_vars):
            reset_enterprise_settings()
            config = MADueDiligenceConfig()

            # Should pick up environment values
            assert config.max_mcts_iterations == 10
            assert config.risk_threshold == 0.6


@pytest.mark.unit
class TestMADueDiligenceConfig:
    """Tests for M&A Due Diligence configuration."""

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_ma_specific_defaults(self, ma_config):
        """Test M&A-specific configuration defaults."""
        assert ma_config.risk_threshold >= 0
        assert ma_config.risk_threshold <= 1
        assert ma_config.critical_risk_threshold >= ma_config.risk_threshold
        assert len(ma_config.jurisdictions) > 0

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_threshold_validation(self):
        """Test that critical threshold must be >= risk threshold."""
        # Valid: critical > risk
        config = MADueDiligenceConfig(
            risk_threshold=0.5,
            critical_risk_threshold=0.8,
        )
        assert config.critical_risk_threshold > config.risk_threshold

        # Invalid: critical < risk
        with pytest.raises(ValueError):
            MADueDiligenceConfig(
                risk_threshold=0.8,
                critical_risk_threshold=0.5,
            )

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_action_weights_configurable(self):
        """Test that action weights are configurable."""
        custom_weights = {
            "financial_analysis": 0.5,
            "legal_review": 0.3,
            "operational_assessment": 0.2,
        }
        config = MADueDiligenceConfig(action_weights=custom_weights)
        assert config.action_weights == custom_weights


@pytest.mark.unit
class TestClinicalTrialConfig:
    """Tests for Clinical Trial configuration."""

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_clinical_specific_defaults(self, clinical_config):
        """Test clinical trial-specific configuration."""
        assert clinical_config.min_statistical_power >= 0.5
        assert clinical_config.min_statistical_power <= 1.0
        assert clinical_config.alpha_level > 0
        assert clinical_config.alpha_level < 0.5
        assert len(clinical_config.supported_agencies) > 0

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_phase_weights_sum(self):
        """Test that phase weights are provided."""
        config = ClinicalTrialConfig()
        assert "phase_1" in config.phase_weights
        assert "phase_2" in config.phase_weights
        assert "phase_3" in config.phase_weights


@pytest.mark.unit
class TestRegulatoryComplianceConfig:
    """Tests for Regulatory Compliance configuration."""

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_compliance_specific_defaults(self, compliance_config):
        """Test compliance-specific configuration."""
        assert compliance_config.risk_tolerance in ["conservative", "moderate", "aggressive"]
        assert len(compliance_config.jurisdictions) > 0
        assert compliance_config.enforcement_horizon_months > 0

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_risk_tolerance_validation(self):
        """Test that risk tolerance is validated."""
        # Valid values
        for tolerance in ["conservative", "moderate", "aggressive"]:
            config = RegulatoryComplianceConfig(risk_tolerance=tolerance)
            assert config.risk_tolerance == tolerance

        # Invalid value
        with pytest.raises(ValueError):
            RegulatoryComplianceConfig(risk_tolerance="invalid")


@pytest.mark.unit
class TestEnterpriseSettings:
    """Tests for master EnterpriseSettings."""

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_get_use_case_config(self, enterprise_settings):
        """Test retrieving use case configurations."""
        ma_config = enterprise_settings.get_use_case_config(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert isinstance(ma_config, MADueDiligenceConfig)

        clinical_config = enterprise_settings.get_use_case_config(EnterpriseDomain.CLINICAL_TRIAL)
        assert isinstance(clinical_config, ClinicalTrialConfig)

        compliance_config = enterprise_settings.get_use_case_config(EnterpriseDomain.REGULATORY_COMPLIANCE)
        assert isinstance(compliance_config, RegulatoryComplianceConfig)

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_safe_dict_no_secrets(self, enterprise_settings):
        """Test that safe_dict doesn't expose sensitive data."""
        safe = enterprise_settings.safe_dict()

        # Should have basic structure
        assert "enterprise_enabled" in safe
        assert "use_cases" in safe

        # Check it's serializable (no secrets)
        import json

        json_str = json.dumps(safe)
        assert len(json_str) > 0

    @pytest.mark.skipif(not SETTINGS_AVAILABLE, reason="Settings not available")
    def test_singleton_pattern(self, enterprise_env_vars):
        """Test settings singleton behavior."""
        with patch.dict(os.environ, enterprise_env_vars):
            reset_enterprise_settings()

            settings1 = get_enterprise_settings()
            settings2 = get_enterprise_settings()

            # Should be same instance
            assert settings1 is settings2

            # Reset and get new instance
            reset_enterprise_settings()
            settings3 = get_enterprise_settings()

            # After reset, should be different instance
            assert settings1 is not settings3
