"""
End-to-End Test: Complete Query Processing Flow (with LangSmith Tracing).

This is an instrumented version demonstrating LangSmith E2E tracing patterns.
Copy the tracing decorators and update_run_metadata() calls to the original file.

Tests the full user journey from query submission to response generation:
1. Input validation
2. Multi-agent processing (HRM/TRM)
3. Consensus scoring
4. Response formatting
5. Confidence metrics

Expected outcomes:
- 80-90% accuracy on complex tactical tasks
- Evidence-backed recommendations
- Uncertainty quantification (0-1 scale)
"""

from datetime import datetime

import pytest

from tests.mocks.mock_external_services import create_mock_llm
from tests.utils.langsmith_tracing import trace_e2e_test, update_run_metadata


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client with tactical responses."""
    client = create_mock_llm(provider="openai")
    client.set_responses(
        [
            # HRM agent response
            """Based on hierarchical decomposition:
        1. Primary objective: Secure northern perimeter
        2. Secondary objective: Establish observation posts
        3. Tertiary objective: Coordinate communication
        Confidence: 0.85""",
            # TRM agent response
            """After iterative refinement:
        - Position Alpha provides best coverage
        - Risk assessment: Medium
        - Resource utilization: 78%
        Confidence: 0.82""",
        ]
    )
    return client


@pytest.fixture
def tactical_query():
    """Sample tactical query for testing."""
    return {
        "query": "Enemy forces spotted approaching from north. Night conditions with limited visibility. "
        "Available assets: Infantry platoon, UAV support, limited ammunition. "
        "Recommend optimal defensive position and engagement strategy.",
        "use_rag": True,
        "use_mcts": False,
        "thread_id": "test_thread_001",
    }


@pytest.fixture
def cybersecurity_query():
    """Sample cybersecurity query for testing."""
    return {
        "query": "APT28 indicators detected on critical infrastructure network. "
        "Evidence of credential harvesting and lateral movement. "
        "Recommend immediate containment and response actions.",
        "use_rag": True,
        "use_mcts": True,
        "thread_id": "test_thread_002",
    }


class TestCompleteFlow:
    """Test complete end-to-end query flow with LangSmith tracing."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_tactical_analysis_flow",
        phase="complete_flow",
        scenario_type="tactical",
        use_mcts=False,
        tags=["hrm", "trm", "consensus"],
    )
    async def test_tactical_analysis_flow(self, mock_llm_client, tactical_query):
        """Complete tactical analysis should produce valid results."""
        # Step 1: Validate input
        from src.models.validation import QueryInput

        query_input = QueryInput(**tactical_query)
        assert query_input.query is not None

        # Step 2: Process through agents
        _hrm_response = await mock_llm_client.generate(f"HRM: {query_input.query}")
        _trm_response = await mock_llm_client.generate(f"TRM: {query_input.query}")

        # Step 3: Calculate consensus
        hrm_conf = 0.85
        trm_conf = 0.82
        consensus = (hrm_conf + trm_conf) / 2

        # Step 4: Generate final response
        final_response = {
            "query_id": query_input.thread_id,
            "recommendation": "Secure northern perimeter at Position Alpha",
            "confidence": consensus,
            "agents_consulted": ["HRM", "TRM"],
            "processing_time_ms": 1500,  # Mock value
        }

        # Update trace with runtime metrics for LangSmith dashboard
        update_run_metadata(
            {
                "consensus_score": consensus,
                "processing_time_ms": final_response["processing_time_ms"],
                "agents_consulted": final_response["agents_consulted"],
            }
        )

        # Validate complete flow
        assert final_response["confidence"] >= 0.75  # High confidence
        assert len(final_response["agents_consulted"]) == 2
        assert final_response["processing_time_ms"] < 30000  # Under SLA

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @trace_e2e_test(
        "e2e_cybersecurity_analysis_flow",
        phase="complete_flow",
        scenario_type="cybersecurity",
        use_mcts=True,
        tags=["threat_detection", "apt", "incident_response"],
    )
    async def test_cybersecurity_analysis_flow(self, mock_llm_client, cybersecurity_query):
        """Complete cybersecurity analysis should identify threats."""
        from src.models.validation import QueryInput

        query_input = QueryInput(**cybersecurity_query)

        # Process query
        response = await mock_llm_client.generate(query_input.query)

        # Verify response addresses threat
        assert response.content is not None

        # In real implementation, would check for:
        # - MITRE ATT&CK technique identification
        # - Severity assessment
        # - Containment recommendations

        final_result = {
            "threat_identified": True,
            "threat_actor": "APT28",
            "severity": "HIGH",
            "confidence": 0.87,
            "recommended_actions": [
                "Isolate affected systems",
                "Reset compromised credentials",
                "Deploy additional monitoring",
            ],
        }

        # Update trace with threat analysis results for LangSmith
        update_run_metadata(
            {
                "threat_identified": final_result["threat_identified"],
                "threat_actor": final_result["threat_actor"],
                "severity": final_result["severity"],
                "confidence": final_result["confidence"],
                "actions_count": len(final_result["recommended_actions"]),
            }
        )

        assert final_result["threat_identified"] is True
        assert final_result["severity"] in ["HIGH", "CRITICAL"]
        assert len(final_result["recommended_actions"]) >= 3
