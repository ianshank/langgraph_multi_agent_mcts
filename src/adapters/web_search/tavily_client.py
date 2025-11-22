"""
Tavily AI Search API client.

Tavily provides an optimized search API specifically designed for LLMs and RAG applications.
Features: AI-optimized results, content extraction, and relevance scoring.
"""

import logging
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseWebSearchClient, SearchResult, WebSearchResponse
from .exceptions import (
    WebSearchAPIError,
    WebSearchAuthError,
    WebSearchNetworkError,
    WebSearchParseError,
    WebSearchRateLimitError,
    WebSearchTimeoutError,
)

logger = logging.getLogger(__name__)


class TavilySearchClient(BaseWebSearchClient):
    """
    Tavily AI Search API client.

    Optimized for AI applications with features like:
    - Content extraction for RAG
    - AI-powered relevance scoring
    - Rich metadata extraction
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_limit_per_minute: int | None = 10,
        cache_ttl_seconds: int = 3600,
        user_agent: str = "LangGraph-Multi-Agent-MCTS/0.1.0",
        base_url: str = "https://api.tavily.com",
    ):
        """
        Initialize Tavily client.

        Args:
            api_key: Tavily API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit_per_minute: Requests per minute limit
            cache_ttl_seconds: Cache TTL in seconds
            user_agent: User agent string
            base_url: Tavily API base URL
        """
        super().__init__(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            rate_limit_per_minute=rate_limit_per_minute,
            cache_ttl_seconds=cache_ttl_seconds,
            user_agent=user_agent,
        )
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": user_agent,
                "Content-Type": "application/json",
            },
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((WebSearchNetworkError, WebSearchTimeoutError)),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        search_type: str = "general",
        **kwargs: Any,
    ) -> WebSearchResponse:
        """
        Perform Tavily search.

        Args:
            query: Search query
            max_results: Max results (1-20)
            include_raw_content: Include full content extraction
            search_type: "general" or "news"
            **kwargs: Additional Tavily parameters

        Returns:
            WebSearchResponse with results
        """
        # Check cache first
        cached = await self._cache.get(query, max_results)
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached

        # Apply rate limiting
        await self._apply_rate_limit()

        start_time = time.perf_counter()

        # Build request payload
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(max_results, 20),
            "include_answer": kwargs.get("include_answer", False),
            "include_raw_content": include_raw_content,
            "include_images": kwargs.get("include_images", False),
            "search_depth": kwargs.get("search_depth", "basic"),  # "basic" or "advanced"
        }

        if search_type == "news":
            payload["topic"] = "news"

        try:
            logger.debug(f"Tavily search: {query[:50]}... (max_results={max_results})")

            response = await self._client.post(
                f"{self.base_url}/search",
                json=payload,
            )

            search_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle errors
            if response.status_code == 401:
                raise WebSearchAuthError("Invalid Tavily API key", status_code=401, provider="tavily")
            elif response.status_code == 429:
                raise WebSearchRateLimitError(
                    "Tavily rate limit exceeded",
                    retry_after=int(response.headers.get("Retry-After", 60)),
                    provider="tavily",
                )
            elif response.status_code >= 400:
                raise WebSearchAPIError(
                    f"Tavily API error: {response.text}",
                    status_code=response.status_code,
                    provider="tavily",
                )

            data = response.json()

            # Parse results
            results = self._parse_results(data, include_raw_content)

            search_response = WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                provider="tavily",
                search_time_ms=search_time_ms,
                cached=False,
                metadata={
                    "answer": data.get("answer"),  # AI-generated answer if requested
                    "images": data.get("images", []),
                    "search_depth": payload["search_depth"],
                },
            )

            # Update stats and cache
            self._update_stats(search_response)
            await self._cache.set(query, max_results, search_response)

            logger.info(
                f"Tavily search completed: {len(results)} results in {search_time_ms:.0f}ms "
                f"(cached={search_response.cached})"
            )

            return search_response

        except httpx.TimeoutException as e:
            self._failed_requests += 1
            raise WebSearchTimeoutError(f"Tavily request timed out: {e}") from e
        except httpx.NetworkError as e:
            self._failed_requests += 1
            raise WebSearchNetworkError(f"Tavily network error: {e}") from e
        except Exception as e:
            self._failed_requests += 1
            if isinstance(e, (WebSearchAuthError, WebSearchRateLimitError, WebSearchAPIError)):
                raise
            raise WebSearchParseError(f"Failed to parse Tavily response: {e}") from e

    def _parse_results(self, data: dict, include_raw_content: bool) -> list[SearchResult]:
        """Parse Tavily API response into SearchResult objects."""
        results = []

        for item in data.get("results", []):
            try:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source=self._extract_domain(item.get("url", "")),
                    raw_content=item.get("raw_content") if include_raw_content else None,
                    relevance_score=item.get("score", 0.0),
                    metadata={
                        "published_date": item.get("published_date"),
                    },
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse Tavily result: {e}")
                continue

        return results

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    async def close(self) -> None:
        """Clean up resources."""
        await super().close()
        await self._client.aclose()
