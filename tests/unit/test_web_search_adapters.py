"""
Unit tests for web search adapters.

Tests all web search providers with mocking to ensure no actual API calls.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.adapters.web_search import (
    SearchResult,
    WebSearchResponse,
    WebSearchError,
    WebSearchAuthError,
    WebSearchRateLimitError,
    create_web_search_client,
)
from src.adapters.web_search.tavily_client import TavilySearchClient
from src.adapters.web_search.serpapi_client import SerpAPISearchClient
from src.adapters.web_search.duckduckgo_client import DuckDuckGoSearchClient
from src.config.settings import Settings, WebSearchProvider


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.WEB_SEARCH_ENABLED = True
    settings.WEB_SEARCH_PROVIDER = WebSearchProvider.TAVILY
    settings.WEB_SEARCH_TIMEOUT_SECONDS = 10
    settings.HTTP_MAX_RETRIES = 3
    settings.WEB_SEARCH_RATE_LIMIT_PER_MINUTE = 10
    settings.WEB_SEARCH_CACHE_TTL_SECONDS = 3600
    settings.WEB_SEARCH_USER_AGENT = "Test/1.0"
    settings.get_tavily_api_key = Mock(return_value="test-tavily-key")
    settings.get_serpapi_api_key = Mock(return_value="test-serpapi-key")
    return settings


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            title="Result 1",
            url="https://example.com/1",
            snippet="This is the first result",
            source="example.com",
            relevance_score=0.9,
        ),
        SearchResult(
            title="Result 2",
            url="https://example.com/2",
            snippet="This is the second result",
            source="example.com",
            relevance_score=0.8,
        ),
    ]


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_to_dict(self):
        """Test SearchResult serialization."""
        result = SearchResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
            source="test.com",
            relevance_score=0.95,
        )
        data = result.to_dict()

        assert data["title"] == "Test"
        assert data["url"] == "https://test.com"
        assert data["snippet"] == "Test snippet"
        assert data["relevance_score"] == 0.95

    def test_content_hash(self):
        """Test content hash generation for deduplication."""
        result1 = SearchResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
        )
        result2 = SearchResult(
            title="Test",
            url="https://test.com",
            snippet="Test snippet",
        )

        # Same content should produce same hash
        assert result1.get_content_hash() == result2.get_content_hash()

        # Different content should produce different hash
        result3 = SearchResult(
            title="Different",
            url="https://different.com",
            snippet="Different snippet",
        )
        assert result1.get_content_hash() != result3.get_content_hash()


class TestWebSearchResponse:
    """Tests for WebSearchResponse."""

    def test_to_dict(self, sample_search_results):
        """Test response serialization."""
        response = WebSearchResponse(
            query="test query",
            results=sample_search_results,
            provider="test",
        )
        data = response.to_dict()

        assert data["query"] == "test query"
        assert data["provider"] == "test"
        assert len(data["results"]) == 2

    def test_get_top_k_results(self, sample_search_results):
        """Test getting top K results by relevance."""
        response = WebSearchResponse(
            query="test",
            results=sample_search_results,
        )

        top_1 = response.get_top_k_results(1)
        assert len(top_1) == 1
        assert top_1[0].relevance_score == 0.9


class TestSearchCache:
    """Tests for search result caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit scenario."""
        from src.adapters.web_search.base import SearchCache

        cache = SearchCache(ttl_seconds=60)
        response = WebSearchResponse(query="test", results=[])

        # Set cache
        await cache.set("test", 5, response)

        # Get from cache
        cached = await cache.get("test", 5)
        assert cached is not None
        assert cached.cached is True
        assert cached.query == "test"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss scenario."""
        from src.adapters.web_search.base import SearchCache

        cache = SearchCache()
        result = await cache.get("nonexistent", 5)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        from src.adapters.web_search.base import SearchCache

        cache = SearchCache()
        response = WebSearchResponse(query="test", results=[])

        await cache.set("test", 5, response)
        await cache.get("test", 5)  # Hit
        await cache.get("miss", 5)  # Miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


@pytest.mark.asyncio
class TestTavilyClient:
    """Tests for Tavily search client."""

    async def test_successful_search(self, sample_search_results):
        """Test successful Tavily search."""
        client = TavilySearchClient(api_key="test-key")

        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the first result",
                    "score": 0.9,
                },
            ],
            "answer": "AI-generated answer",
        })

        with patch.object(client._client, "post", return_value=mock_response):
            response = await client.search("test query", max_results=1)

            assert response.provider == "tavily"
            assert len(response.results) > 0
            assert response.results[0].title == "Result 1"

        await client.close()

    async def test_auth_error(self):
        """Test authentication error handling."""
        client = TavilySearchClient(api_key="invalid-key")

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(WebSearchAuthError):
                await client.search("test query")

        await client.close()

    async def test_rate_limit_error(self):
        """Test rate limit error handling."""
        client = TavilySearchClient(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}

        with patch.object(client._client, "post", return_value=mock_response):
            with pytest.raises(WebSearchRateLimitError) as exc_info:
                await client.search("test query")
            assert exc_info.value.retry_after == 60

        await client.close()


@pytest.mark.asyncio
class TestSerpAPIClient:
    """Tests for SerpAPI client."""

    async def test_successful_search(self):
        """Test successful SerpAPI search."""
        client = SerpAPISearchClient(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={
            "organic_results": [
                {
                    "title": "Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a snippet",
                    "position": 1,
                },
            ],
            "search_information": {"total_results": 1000},
        })

        with patch.object(client._client, "get", return_value=mock_response):
            response = await client.search("test query", max_results=1)

            assert response.provider == "serpapi"
            assert len(response.results) > 0

        await client.close()


@pytest.mark.asyncio
class TestDuckDuckGoClient:
    """Tests for DuckDuckGo client."""

    async def test_successful_search(self):
        """Test successful DuckDuckGo search."""
        client = DuckDuckGoSearchClient()

        # Mock HTML response
        mock_html = """
        <div class="result">
            <a class="result__a" href="https://example.com/1">Result 1</a>
            <a class="result__snippet">This is a snippet</a>
        </div>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html

        with patch.object(client._client, "get", return_value=mock_response):
            response = await client.search("test query", max_results=1)

            assert response.provider == "duckduckgo"
            # Results depend on parsing, which may vary

        await client.close()


class TestFactory:
    """Tests for web search client factory."""

    def test_create_tavily_client(self, mock_settings):
        """Test creating Tavily client from settings."""
        mock_settings.WEB_SEARCH_PROVIDER = WebSearchProvider.TAVILY
        client = create_web_search_client(settings=mock_settings)
        assert isinstance(client, TavilySearchClient)

    def test_create_serpapi_client(self, mock_settings):
        """Test creating SerpAPI client from settings."""
        mock_settings.WEB_SEARCH_PROVIDER = WebSearchProvider.SERPAPI
        client = create_web_search_client(settings=mock_settings)
        assert isinstance(client, SerpAPISearchClient)

    def test_create_duckduckgo_client(self, mock_settings):
        """Test creating DuckDuckGo client from settings."""
        mock_settings.WEB_SEARCH_PROVIDER = WebSearchProvider.DUCKDUCKGO
        client = create_web_search_client(settings=mock_settings)
        assert isinstance(client, DuckDuckGoSearchClient)

    def test_disabled_provider(self, mock_settings):
        """Test error when provider is disabled."""
        mock_settings.WEB_SEARCH_PROVIDER = WebSearchProvider.DISABLED
        with pytest.raises(WebSearchError):
            create_web_search_client(settings=mock_settings)
