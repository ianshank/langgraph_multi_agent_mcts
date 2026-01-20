"""
Integration tests for enterprise module.

Tests the integration layer including LangGraph extension
and meta-controller adapter.
"""

from __future__ import annotations

import pytest

# Import modules with error handling
try:
    from src.enterprise.config.enterprise_settings import EnterpriseDomain
    from src.enterprise.factories.use_case_factory import EnterpriseUseCaseFactory
    from src.enterprise.integration import (
        EnterpriseAgentState,
        EnterpriseGraphBuilder,
        EnterpriseMetaControllerAdapter,
    )

    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


pytestmark = pytest.mark.enterprise


@pytest.mark.unit
class TestEnterpriseMetaControllerAdapter:
    """Tests for EnterpriseMetaControllerAdapter."""

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        adapter = EnterpriseMetaControllerAdapter(
            domain_detection_threshold=0.7,
        )

        assert adapter._detection_threshold == 0.7

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_detect_ma_domain(self, sample_ma_query):
        """Test detecting M&A domain from query."""
        adapter = EnterpriseMetaControllerAdapter()

        domain, confidence = adapter.detect_domain(sample_ma_query)

        assert domain == EnterpriseDomain.MA_DUE_DILIGENCE
        assert confidence > 0

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_detect_clinical_domain(self, sample_clinical_query):
        """Test detecting clinical trial domain from query."""
        adapter = EnterpriseMetaControllerAdapter()

        domain, confidence = adapter.detect_domain(sample_clinical_query)

        assert domain == EnterpriseDomain.CLINICAL_TRIAL
        assert confidence > 0

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_detect_compliance_domain(self, sample_compliance_query):
        """Test detecting compliance domain from query."""
        adapter = EnterpriseMetaControllerAdapter()

        domain, confidence = adapter.detect_domain(sample_compliance_query)

        assert domain == EnterpriseDomain.REGULATORY_COMPLIANCE
        assert confidence > 0

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_no_domain_detected_for_generic(self):
        """Test no domain detected for generic query."""
        adapter = EnterpriseMetaControllerAdapter()

        domain, confidence = adapter.detect_domain("What is the weather?")

        # Should not match any domain with high confidence
        # (domain may be returned but confidence should be low)
        assert confidence < 0.7

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_extract_enterprise_features(self, sample_ma_query):
        """Test feature extraction."""
        adapter = EnterpriseMetaControllerAdapter()

        features = adapter.extract_enterprise_features(
            state={"iteration": 1, "confidence_scores": {"hrm": 0.8}},
            query=sample_ma_query,
        )

        assert features.detected_domain is not None
        assert features.domain_confidence > 0
        assert features.query_length == len(sample_ma_query)
        assert features.iteration == 1

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_route_to_enterprise(self, sample_ma_query):
        """Test routing decision."""
        adapter = EnterpriseMetaControllerAdapter()

        features = adapter.extract_enterprise_features(
            state={},
            query=sample_ma_query,
        )

        route = adapter.route_to_enterprise(features)

        assert route == "enterprise_ma"

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_requires_compliance_detection(self):
        """Test compliance requirement detection."""
        adapter = EnterpriseMetaControllerAdapter()

        assert adapter._requires_compliance("Check GDPR compliance status")
        assert adapter._requires_compliance("Audit our regulatory requirements")
        assert not adapter._requires_compliance("Analyze revenue growth")

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_complexity_estimation(self):
        """Test complexity estimation."""
        adapter = EnterpriseMetaControllerAdapter()

        # Short simple query
        simple_complexity = adapter._estimate_complexity(
            "Hello",
            {},
        )

        # Long complex query
        complex_complexity = adapter._estimate_complexity(
            "Analyze and evaluate the comprehensive financial statements "
            "and compare against industry benchmarks to optimize our strategy",
            {"iteration": 5},
        )

        assert complex_complexity > simple_complexity

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_jurisdiction_extraction(self):
        """Test jurisdiction extraction from query."""
        adapter = EnterpriseMetaControllerAdapter()

        jurisdictions = adapter._extract_jurisdictions(
            "Analyze GDPR compliance for our EU operations and US data centers"
        )

        assert "EU" in jurisdictions
        assert "US" in jurisdictions


@pytest.mark.unit
class TestEnterpriseGraphBuilder:
    """Tests for EnterpriseGraphBuilder."""

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_builder_initialization(self, use_case_factory):
        """Test graph builder initializes correctly."""
        builder = EnterpriseGraphBuilder(
            use_case_factory=use_case_factory,
        )

        assert builder._factory == use_case_factory

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_should_route_to_enterprise(self, use_case_factory, sample_ma_query):
        """Test enterprise routing detection."""
        builder = EnterpriseGraphBuilder(use_case_factory=use_case_factory)

        state = EnterpriseAgentState(query=sample_ma_query)
        should_route = builder.should_route_to_enterprise(state)

        assert should_route is True

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_should_not_route_generic(self, use_case_factory):
        """Test generic queries don't route to enterprise."""
        builder = EnterpriseGraphBuilder(use_case_factory=use_case_factory)

        state = EnterpriseAgentState(query="What time is it?")
        should_route = builder.should_route_to_enterprise(state)

        assert should_route is False

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    def test_get_enterprise_route(self, use_case_factory, sample_ma_query):
        """Test getting specific enterprise route."""
        builder = EnterpriseGraphBuilder(use_case_factory=use_case_factory)

        state = EnterpriseAgentState(query=sample_ma_query)
        route = builder.get_enterprise_route(state)

        assert "enterprise" in route or route == "standard"

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    @pytest.mark.asyncio
    async def test_process_enterprise_query(
        self,
        use_case_factory,
        sample_ma_query,
    ):
        """Test processing enterprise query through builder."""
        builder = EnterpriseGraphBuilder(use_case_factory=use_case_factory)

        state = EnterpriseAgentState(
            query=sample_ma_query,
            use_mcts=False,
        )

        result = await builder.process_enterprise_query(state)

        # Result should have enterprise fields
        assert "enterprise_domain" in result


@pytest.mark.integration
class TestFullIntegration:
    """Full integration tests."""

    @pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration not available")
    @pytest.mark.asyncio
    async def test_end_to_end_ma_query(
        self,
        enterprise_settings,
        mock_llm_client,
        sample_ma_query,
        sample_ma_context,
    ):
        """Test end-to-end M&A query processing."""
        factory = EnterpriseUseCaseFactory(
            enterprise_settings=enterprise_settings,
            llm_client=mock_llm_client,
        )
        builder = EnterpriseGraphBuilder(use_case_factory=factory)
        adapter = EnterpriseMetaControllerAdapter()

        # Detect domain
        domain, confidence = adapter.detect_domain(sample_ma_query)
        assert domain == EnterpriseDomain.MA_DUE_DILIGENCE

        # Route to enterprise
        features = adapter.extract_enterprise_features({}, sample_ma_query)
        route = adapter.route_to_enterprise(features)
        assert route == "enterprise_ma"

        # Process through builder
        state = EnterpriseAgentState(
            query=sample_ma_query,
            use_mcts=False,
        )
        result = await builder.process_enterprise_query(state)

        # Verify result structure
        if result.get("enterprise_domain"):
            assert "agent_outputs" in result
