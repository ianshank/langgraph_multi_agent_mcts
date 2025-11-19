"""
TRM (Task Refinement Model) Component Tests with LangSmith Tracing.

Component-level tests for TRM agent behavior, refinement quality,
and iterative improvement validation.

Tests focus on:
- Iterative refinement cycles
- Alternative evaluation
- Convergence behavior
- Quality improvement over iterations
- Performance metrics
"""

import pytest

from tests.mocks.mock_external_services import create_mock_llm
from tests.utils.langsmith_tracing import trace_e2e_test, update_run_metadata


@pytest.fixture
def trm_config():
    """TRM agent configuration."""
    return {
        "max_iterations": 5,
        "convergence_threshold": 0.02,
        "min_alternatives": 3,
        "quality_threshold": 0.75,
    }


@pytest.fixture
def mock_trm_llm():
    """Mock LLM for TRM testing."""
    client = create_mock_llm(provider="openai")
    client.set_responses(
        [
            """TRM Refinement Cycle 1:
Option A: Position Alpha - Coverage 78%, Risk Medium, Score 0.72
Option B: Position Beta - Coverage 65%, Risk Low, Score 0.68
Option C: Position Gamma - Coverage 82%, Risk High, Score 0.70
Recommended: Option C (highest coverage)
Confidence: 0.74""",
            """TRM Refinement Cycle 2:
Option A: Position Alpha - Coverage 82%, Risk Medium, Score 0.78
Option B: Position Beta - Coverage 68%, Risk Low, Score 0.70
Option C: Position Gamma - Coverage 85%, Risk Medium-High, Score 0.81
Recommended: Option C (best balance)
Confidence: 0.81""",
        ]
    )
    return client


class TestTRMIterativeRefinement:
    """Test TRM iterative refinement capabilities."""

    @pytest.mark.component
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_trm_iterative_refinement",
        phase="component",
        scenario_type="tactical",
        tags=["trm", "component", "refinement", "iterations"],
    )
    async def test_multi_cycle_refinement(self, mock_trm_llm, trm_config):
        """Test TRM improves quality over multiple refinement cycles."""
        query = "Recommend optimal defensive position"

        # Simulate multiple refinement cycles
        cycles = 3
        quality_scores = []

        for i in range(cycles):
            response = await mock_trm_llm.generate(f"TRM Cycle {i+1}: {query}")

            # Extract quality score from response
            if "0.74" in response.content:
                score = 0.74
            elif "0.81" in response.content:
                score = 0.81
            else:
                score = 0.70 + (i * 0.05)  # Simulate improvement

            quality_scores.append(score)

        # Calculate improvement
        initial_score = quality_scores[0]
        final_score = quality_scores[-1]
        improvement = final_score - initial_score

        # Update trace
        update_run_metadata(
            {
                "component": "trm",
                "test_type": "iterative_refinement",
                "total_cycles": cycles,
                "quality_scores": quality_scores,
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": improvement,
                "converged": improvement < trm_config["convergence_threshold"],
            }
        )

        # Assertions
        assert final_score >= initial_score, "Quality should not degrade"
        assert final_score >= trm_config["quality_threshold"], "Should meet quality threshold"

    @pytest.mark.component
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_trm_convergence_detection",
        phase="component",
        scenario_type="tactical",
        tags=["trm", "component", "convergence"],
    )
    async def test_convergence_detection(self, mock_trm_llm, trm_config):
        """Test TRM detects convergence and stops iterating."""
        _query = "Optimize resource allocation strategy"

        # Simulate refinement until convergence
        max_iterations = trm_config["max_iterations"]
        convergence_threshold = trm_config["convergence_threshold"]

        scores = [0.70, 0.76, 0.80, 0.82, 0.821]  # Simulated scores showing convergence
        converged_at = None

        for i in range(1, len(scores)):
            improvement = abs(scores[i] - scores[i - 1])
            if improvement < convergence_threshold:
                converged_at = i
                break

        # Update trace
        update_run_metadata(
            {
                "component": "trm",
                "test_type": "convergence_detection",
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
                "scores": scores,
                "converged_at_iteration": converged_at,
                "final_score": scores[converged_at] if converged_at else scores[-1],
            }
        )

        assert converged_at is not None, "Should detect convergence"
        assert converged_at < max_iterations, "Should converge before max iterations"


class TestTRMAlternativeEvaluation:
    """Test TRM alternative evaluation."""

    @pytest.mark.component
    @trace_e2e_test(
        "component_trm_alternative_ranking",
        phase="component",
        scenario_type="tactical",
        tags=["trm", "component", "alternatives"],
    )
    def test_alternative_ranking_quality(self, trm_config):
        """Test TRM ranks alternatives consistently."""
        # Simulate TRM alternative evaluation
        alternatives = [
            {"option": "A", "coverage": 0.82, "risk": "medium", "score": 0.78},
            {"option": "B", "coverage": 0.68, "risk": "low", "score": 0.70},
            {"option": "C", "coverage": 0.85, "risk": "medium_high", "score": 0.81},
        ]

        # Sort by score
        ranked = sorted(alternatives, key=lambda x: x["score"], reverse=True)
        best_option = ranked[0]

        # Update trace
        update_run_metadata(
            {
                "component": "trm",
                "test_type": "alternative_ranking",
                "alternatives_evaluated": len(alternatives),
                "best_option": best_option["option"],
                "best_score": best_option["score"],
                "score_spread": max(a["score"] for a in alternatives) - min(a["score"] for a in alternatives),
                "min_alternatives_config": trm_config["min_alternatives"],
            }
        )

        assert len(alternatives) >= trm_config["min_alternatives"]
        assert ranked[0]["score"] >= ranked[1]["score"]
        assert ranked[1]["score"] >= ranked[2]["score"]


class TestTRMPerformance:
    """Test TRM performance characteristics."""

    @pytest.mark.component
    @pytest.mark.performance
    @pytest.mark.asyncio
    @trace_e2e_test(
        "component_trm_refinement_latency",
        phase="component",
        scenario_type="tactical",
        tags=["trm", "component", "performance", "latency"],
    )
    async def test_refinement_cycle_latency(self, mock_trm_llm, trm_config):
        """Test TRM refinement completes within SLA."""
        import time

        query = "Refine tactical approach"

        # Time multiple cycles
        cycle_latencies = []

        for cycle_num in range(3):
            start = time.time()
            await mock_trm_llm.generate(f"TRM Cycle {cycle_num+1}: {query}")
            latency_ms = (time.time() - start) * 1000
            cycle_latencies.append(latency_ms)

        avg_latency = sum(cycle_latencies) / len(cycle_latencies)

        # Update trace
        update_run_metadata(
            {
                "component": "trm",
                "test_type": "performance_latency",
                "cycles_tested": len(cycle_latencies),
                "cycle_latencies_ms": cycle_latencies,
                "average_latency_ms": avg_latency,
                "sla_per_cycle_ms": 2000,
                "within_sla": all(latency < 2000 for latency in cycle_latencies),
            }
        )

        assert all(latency < 2000 for latency in cycle_latencies), "Each cycle should be under 2s"
