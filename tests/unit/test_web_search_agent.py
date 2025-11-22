"""
Unit tests for Web Search Agent.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.agents.web_search_agent import WebSearchAgent, WebSearchAgentConfig
from src.adapters.web_search import WebSearchResponse, SearchResult


@pytest.fixture
def mock_model_adapter():
    """Create mock model adapter."""
    adapter = AsyncMock()
    adapter.generate = AsyncMock(
        return_value=Mock(text="Synthesized answer based on search results")
    )
    return adapter


@pytest.fixture
def mock_search_client():
    """Create mock search client."""
    client = AsyncMock()
    client.search = AsyncMock(
        return_value=WebSearchResponse(
            query="test query",
            results=[
                SearchResult(
                    title="Test Result",
                    url="https://test.com",
                    snippet="Test snippet",
                    source="test.com",
                    relevance_score=0.9,
                ),
            ],
            provider="test",
            search_time_ms=100.0,
        )
    )
    client.stats = {
        "request_count": 1,
        "cache": {"hits": 0, "misses": 1},
    }
    return client


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = AsyncMock()
    store.add_documents = AsyncMock()
    return store


@pytest.mark.asyncio
class TestWebSearchAgent:
    """Tests for WebSearchAgent."""

    async def test_successful_search(self, mock_model_adapter, mock_search_client):
        """Test successful web search."""
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_search_client,
        )

        result = await agent.process("test query")

        assert "response" in result
        assert "metadata" in result
        assert "sources" in result
        assert len(result["sources"]) > 0
        assert mock_search_client.search.called

    async def test_synthesis_with_llm(self, mock_model_adapter, mock_search_client):
        """Test result synthesis with LLM."""
        config = WebSearchAgentConfig(enable_synthesis=True)
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_search_client,
            config=config,
        )

        result = await agent.process("test query")

        # Should use LLM for synthesis
        assert mock_model_adapter.generate.called
        assert "Synthesized answer" in result["response"]

    async def test_synthesis_without_llm(self, mock_search_client):
        """Test result synthesis without LLM (fallback)."""
        config = WebSearchAgentConfig(enable_synthesis=False)
        agent = WebSearchAgent(
            model_adapter=None,
            search_client=mock_search_client,
            config=config,
        )

        result = await agent.process("test query")

        # Should use simple synthesis
        assert "response" in result
        assert "Test Result" in result["response"]

    async def test_vector_storage(
        self, mock_model_adapter, mock_search_client, mock_vector_store
    ):
        """Test storing search results in vector DB."""
        config = WebSearchAgentConfig(enable_vector_storage=True)
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_search_client,
            vector_store=mock_vector_store,
            config=config,
        )

        await agent.process("test query")

        # Should store results in vector DB
        assert mock_vector_store.add_documents.called

    async def test_no_search_client(self, mock_model_adapter):
        """Test behavior when search client is not available."""
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=None,
        )

        result = await agent.process("test query")

        assert "error" in result["metadata"]
        assert result["metadata"]["error"] == "search_client_unavailable"

    async def test_multi_query_search(self, mock_model_adapter, mock_search_client):
        """Test multi-query concurrent search."""
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_search_client,
        )

        queries = ["query 1", "query 2", "query 3"]
        result = await agent.multi_query_search(queries)

        assert "metadata" in result
        assert result["metadata"]["query_count"] == 3
        assert mock_search_client.search.call_count == 3

    async def test_agent_stats(self, mock_model_adapter, mock_search_client):
        """Test agent statistics tracking."""
        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_search_client,
        )

        await agent.process("test query")

        stats = agent.stats
        assert stats["search_count"] == 1
        assert stats["total_results_retrieved"] > 0
        assert "search_client" in stats

    async def test_caching(self, mock_model_adapter):
        """Test result caching."""
        # Create search client with cache
        mock_client = AsyncMock()
        cached_response = WebSearchResponse(
            query="test",
            results=[],
            provider="test",
            cached=True,
        )
        mock_client.search = AsyncMock(return_value=cached_response)

        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_client,
        )

        result = await agent.process("test")

        # Should indicate cached result
        assert result["metadata"]["cached"] is True

    async def test_error_handling(self, mock_model_adapter):
        """Test error handling during search."""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(
            side_effect=Exception("Network error")
        )

        agent = WebSearchAgent(
            model_adapter=mock_model_adapter,
            search_client=mock_client,
        )

        result = await agent.process("test query")

        # Should return error information
        assert "error" in result["metadata"]
        assert "Network error" in str(result["metadata"]["error"])
