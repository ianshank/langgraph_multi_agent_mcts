"""
Integration tests for Neural Components in UnifiedSearchOrchestrator.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.framework.mcts.llm_guided.integration import (
    AgentType,
    HRMAdapter,
    IntegrationConfig,
    TRMAdapter,
    UnifiedSearchOrchestrator,
)
from src.training.system_config import SystemConfig


@pytest.fixture
def mock_components():
    llm_client = AsyncMock()
    hrm_agent = MagicMock()
    trm_agent = MagicMock()
    # Mocks for encoder/decoder
    encoder = MagicMock()
    # encoder output needs to be a tensor with shape [1, S, H]
    import torch

    encoder.return_value = torch.randn(1, 10, 32)
    encoder.hidden_size = 32

    decoder = MagicMock()
    # decoder generate returns list of strings
    decoder.generate.return_value = ["step 1\nstep 2"]

    return {
        "llm_client": llm_client,
        "hrm_agent": hrm_agent,
        "trm_agent": trm_agent,
        "encoder": encoder,
        "decoder": decoder,
    }


@pytest.mark.asyncio
async def test_orchestrator_neural_decomposition_flow(mock_components):
    """Verify that orchestrator routes to neural decomposition when enabled."""

    # 1. Setup Orchestrator with Neural Agents
    config = IntegrationConfig(use_hrm_decomposition=True, distillation_mode=True)

    # We need to mock the internal MCTS engine and Router to control flow
    with (
        patch("src.framework.mcts.llm_guided.integration.LLMGuidedMCTSEngine") as MockMCTS,
        patch("src.framework.mcts.llm_guided.integration.MetaControllerAdapter") as MockMeta,
    ):
        # Setup MCTS Mock
        mcts_instance = AsyncMock()
        mcts_instance.search.return_value = MagicMock(
            solution_found=True,
            best_code="def solution(): pass",
            best_value=1.0,
            num_iterations=10,
            num_expansions=10,
            llm_calls=5,
            tokens_used=100,
        )
        # Mock Data Collector inside Engine
        data_collector = MagicMock()
        mcts_instance._data_collector = data_collector
        MockMCTS.return_value = mcts_instance

        # Setup Router Mock
        router_instance = MagicMock()
        router_instance.route.return_value = MagicMock(selected_agent=AgentType.HRM, confidence=0.9)
        MockMeta.return_value = router_instance

        # Initialize Orchestrator
        # We manually construct adapters to inject our mocks
        hrm_adapter = HRMAdapter(
            hrm_agent=mock_components["hrm_agent"],
            llm_client=mock_components["llm_client"],
            encoder=mock_components["encoder"],
            decoder=mock_components["decoder"],
        )

        trm_adapter = TRMAdapter(
            trm_agent=mock_components["trm_agent"],
            llm_client=mock_components["llm_client"],
            encoder=mock_components["encoder"],
            decoder=mock_components["decoder"],
        )

        orchestrator = UnifiedSearchOrchestrator(
            llm_client=mock_components["llm_client"],
            mcts_config=SystemConfig(),
            integration_config=config,
            hrm_adapter=hrm_adapter,
            trm_adapter=trm_adapter,
            meta_controller_adapter=router_instance,
        )
        # Inject the mock engine directly
        orchestrator._mcts_engine = mcts_instance

        # 2. Run Search
        result = await orchestrator.search("Write a function", ["assert True"])

        # 3. Verify Neural Decomposition was used
        # Check encoder called
        mock_components["encoder"].assert_called()
        # Check decoder called
        mock_components["decoder"].generate.assert_called()

        # 4. Verify Data Recording
        data_collector.start_episode.assert_called_once()
        data_collector.record_decomposition.assert_called_once()

        # Check that decomposition result propagated
        assert result.decomposition is not None
        assert result.decomposition.subproblems == ["step 1", "step 2"]


@pytest.mark.asyncio
async def test_orchestrator_neural_refinement_flow(mock_components):
    """Verify that orchestrator routes to neural refinement when enabled."""

    config = IntegrationConfig(use_trm_refinement=True, distillation_mode=True)

    with (
        patch("src.framework.mcts.llm_guided.integration.LLMGuidedMCTSEngine") as MockMCTS,
        patch("src.framework.mcts.llm_guided.integration.MetaControllerAdapter") as MockMeta,
    ):
        mcts_instance = AsyncMock()
        # Initial MCTS finds solution, but we might refine it anyway
        mcts_instance.search.return_value = MagicMock(
            solution_found=True, best_code="def solution(): pass", best_value=0.8
        )
        data_collector = MagicMock()
        mcts_instance._data_collector = data_collector
        MockMCTS.return_value = mcts_instance

        router_instance = MagicMock()
        # Route to TRM/LLM_MCTS to allow refinement
        router_instance.route.return_value = MagicMock(selected_agent=AgentType.LLM_MCTS, confidence=0.9)
        MockMeta.return_value = router_instance

        # Setup Adapters
        hrm_adapter = HRMAdapter(None, mock_components["llm_client"])
        trm_adapter = TRMAdapter(
            trm_agent=mock_components["trm_agent"],
            llm_client=mock_components["llm_client"],
            encoder=mock_components["encoder"],
            decoder=mock_components["decoder"],
        )

        orchestrator = UnifiedSearchOrchestrator(
            llm_client=mock_components["llm_client"],
            mcts_config=SystemConfig(),
            integration_config=config,
            hrm_adapter=hrm_adapter,
            trm_adapter=trm_adapter,
            meta_controller_adapter=router_instance,
        )
        orchestrator._mcts_engine = mcts_instance

        # Configure decoder to return refined code
        mock_components["decoder"].generate.return_value = ["def refined(): pass"]

        # Run Search
        result = await orchestrator.search("Write a function", ["test"])

        # Verify Refinement
        mock_components["trm_agent"].assert_called()
        data_collector.record_refinement.assert_called_once()
        assert result.refinement is not None
        assert result.refinement.refined_code == "def refined(): pass"
