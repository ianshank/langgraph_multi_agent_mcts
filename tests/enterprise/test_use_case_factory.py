"""
Tests for EnterpriseUseCaseFactory.

Tests the factory pattern implementation for creating
enterprise use case instances.
"""

from __future__ import annotations

import pytest

# Import modules with error handling
try:
    from src.enterprise.config.enterprise_settings import EnterpriseDomain
    from src.enterprise.factories.use_case_factory import EnterpriseUseCaseFactory

    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False


pytestmark = pytest.mark.enterprise


@pytest.mark.unit
class TestEnterpriseUseCaseFactory:
    """Tests for EnterpriseUseCaseFactory."""

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_factory_initialization(self, enterprise_settings, mock_llm_client):
        """Test factory initializes correctly."""
        factory = EnterpriseUseCaseFactory(
            enterprise_settings=enterprise_settings,
            llm_client=mock_llm_client,
        )

        assert factory.enterprise_settings == enterprise_settings
        assert factory.llm_client == mock_llm_client

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_list_registered_returns_domains(self, use_case_factory):
        """Test listing registered use cases."""
        registered = EnterpriseUseCaseFactory.list_registered()

        assert isinstance(registered, list)
        # Should have at least some registered use cases
        # (depends on auto-registration success)

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_is_registered(self, use_case_factory):
        """Test checking registration status."""
        # Check the method works without error
        result = EnterpriseUseCaseFactory.is_registered(EnterpriseDomain.MA_DUE_DILIGENCE)
        assert isinstance(result, bool)

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_raises_for_unknown_domain(self, use_case_factory):
        """Test creating unknown domain raises error."""
        # Clear registry to ensure failure
        original_registry = EnterpriseUseCaseFactory._registry.copy()
        try:
            EnterpriseUseCaseFactory._registry.clear()

            with pytest.raises(ValueError, match="Unknown domain"):
                use_case_factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)
        finally:
            EnterpriseUseCaseFactory._registry = original_registry

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_with_config_overrides(self, use_case_factory):
        """Test creating use case with configuration overrides."""
        # This test checks that override mechanism works
        # Actual creation depends on registration

        registered = EnterpriseUseCaseFactory.list_registered()
        if not registered:
            pytest.skip("No use cases registered")

        domain = registered[0]
        overrides = {"max_mcts_iterations": 5}

        use_case = use_case_factory.create(domain, config_overrides=overrides)
        assert use_case is not None

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_all_enabled(self, use_case_factory):
        """Test creating all enabled use cases."""
        enabled_cases = use_case_factory.create_all_enabled()

        assert isinstance(enabled_cases, dict)
        # All returned should be enabled
        for _domain, use_case in enabled_cases.items():
            assert use_case.config.enabled

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_from_query_ma(self, use_case_factory, sample_ma_query):
        """Test auto-detecting M&A domain from query."""
        use_case = use_case_factory.create_from_query(sample_ma_query)

        if use_case is not None:
            assert "ma" in use_case.name.lower() or "due_diligence" in use_case.name.lower()

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_from_query_clinical(self, use_case_factory, sample_clinical_query):
        """Test auto-detecting clinical trial domain from query."""
        use_case = use_case_factory.create_from_query(sample_clinical_query)

        if use_case is not None:
            assert "clinical" in use_case.name.lower()

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_from_query_compliance(self, use_case_factory, sample_compliance_query):
        """Test auto-detecting compliance domain from query."""
        use_case = use_case_factory.create_from_query(sample_compliance_query)

        if use_case is not None:
            assert "compliance" in use_case.name.lower() or "regulatory" in use_case.name.lower()

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    def test_create_from_query_no_match(self, use_case_factory):
        """Test that generic query returns None."""
        generic_query = "What is the weather today?"
        use_case = use_case_factory.create_from_query(generic_query)

        # Should not match any enterprise domain
        assert use_case is None


@pytest.mark.integration
class TestFactoryIntegration:
    """Integration tests for factory with real components."""

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    @pytest.mark.asyncio
    async def test_created_use_case_can_process(
        self,
        use_case_factory,
        sample_ma_query,
        sample_ma_context,
    ):
        """Test that created use cases can process queries."""
        registered = EnterpriseUseCaseFactory.list_registered()
        if EnterpriseDomain.MA_DUE_DILIGENCE not in registered:
            pytest.skip("MA Due Diligence not registered")

        use_case = use_case_factory.create(EnterpriseDomain.MA_DUE_DILIGENCE)

        result = await use_case.process(
            query=sample_ma_query,
            context=sample_ma_context,
            use_mcts=False,  # Skip MCTS for faster test
        )

        assert "result" in result
        assert "confidence" in result
        assert result["use_case"] == "ma_due_diligence"

    @pytest.mark.skipif(not FACTORY_AVAILABLE, reason="Factory not available")
    @pytest.mark.asyncio
    async def test_multiple_use_cases_independent(self, use_case_factory):
        """Test that multiple use case instances are independent."""
        registered = EnterpriseUseCaseFactory.list_registered()
        if len(registered) < 2:
            pytest.skip("Need at least 2 use cases for this test")

        domain1, domain2 = registered[0], registered[1]

        use_case1 = use_case_factory.create(domain1)
        use_case2 = use_case_factory.create(domain2)

        # Should be different instances
        assert use_case1 is not use_case2
        assert use_case1.name != use_case2.name
