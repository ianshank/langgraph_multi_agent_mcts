"""
SerpAPI client for Google Search integration.

SerpAPI provides access to Google Search results with structured data extraction.
"""

import logging
import time
from datetime import datetime
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


class SerpAPISearchClient(BaseWebSearchClient):
    """
    SerpAPI client for Google Search results.

    Provides structured access to Google Search with rich metadata and snippets.
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_limit_per_minute: int | None = 10,
        cache_ttl_seconds: int = 3600,
        user_agent: str = "LangGraph-Multi-Agent-MCTS/0.1.0",
        base_url: str = "https://serpapi.com",
    ):
        """
        Initialize SerpAPI client.

        Args:
            api_key: SerpAPI key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit_per_minute: Requests per minute limit
            cache_ttl_seconds: Cache TTL in seconds
            user_agent: User agent string
            base_url: SerpAPI base URL
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
        Perform SerpAPI search.

        Args:
            query: Search query
            max_results: Max results (1-20)
            include_raw_content: Not supported by SerpAPI directly
            search_type: "general" or "news"
            **kwargs: Additional SerpAPI parameters

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

        # Build request parameters
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": min(max_results, 20),
            "engine": "google",  # Default to Google
        }

        if search_type == "news":
            params["tbm"] = "nws"  # Google News

        # Add optional parameters
        if "location" in kwargs:
            params["location"] = kwargs["location"]
        if "gl" in kwargs:  # Geographic location
            params["gl"] = kwargs["gl"]
        if "hl" in kwargs:  # Interface language
            params["hl"] = kwargs["hl"]

        try:
            logger.debug(f"SerpAPI search: {query[:50]}... (max_results={max_results})")

            response = await self._client.get(
                f"{self.base_url}/search",
                params=params,
            )

            search_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle errors
            if response.status_code == 401:
                raise WebSearchAuthError("Invalid SerpAPI key", status_code=401, provider="serpapi")
            elif response.status_code == 429:
                raise WebSearchRateLimitError(
                    "SerpAPI rate limit exceeded",
                    retry_after=60,
                    provider="serpapi",
                )
            elif response.status_code >= 400:
                raise WebSearchAPIError(
                    f"SerpAPI error: {response.text}",
                    status_code=response.status_code,
                    provider="serpapi",
                )

            data = response.json()

            # Parse results
            results = self._parse_results(data, search_type)

            search_response = WebSearchResponse(
                query=query,
                results=results,
                total_results=data.get("search_information", {}).get("total_results", len(results)),
                provider="serpapi",
                search_time_ms=search_time_ms,
                cached=False,
                metadata={
                    "search_metadata": data.get("search_metadata", {}),
                    "answer_box": data.get("answer_box"),
                    "knowledge_graph": data.get("knowledge_graph"),
                },
            )

            # Update stats and cache
            self._update_stats(search_response)
            await self._cache.set(query, max_results, search_response)

            logger.info(
                f"SerpAPI search completed: {len(results)} results in {search_time_ms:.0f}ms "
                f"(cached={search_response.cached})"
            )

            return search_response

        except httpx.TimeoutException as e:
            self._failed_requests += 1
            raise WebSearchTimeoutError(f"SerpAPI request timed out: {e}") from e
        except httpx.NetworkError as e:
            self._failed_requests += 1
            raise WebSearchNetworkError(f"SerpAPI network error: {e}") from e
        except Exception as e:
            self._failed_requests += 1
            if isinstance(e, (WebSearchAuthError, WebSearchRateLimitError, WebSearchAPIError)):
                raise
            raise WebSearchParseError(f"Failed to parse SerpAPI response: {e}") from e

    def _parse_results(self, data: dict, search_type: str) -> list[SearchResult]:
        """Parse SerpAPI response into SearchResult objects."""
        results = []

        # Different result keys for different search types
        if search_type == "news":
            result_key = "news_results"
        else:
            result_key = "organic_results"

        for item in data.get(result_key, []):
            try:
                # Extract published date if available
                published_date = None
                if "date" in item:
                    try:
                        published_date = datetime.fromisoformat(item["date"])
                    except Exception:
                        pass

                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source=item.get("source", self._extract_domain(item.get("link", ""))),
                    published_date=published_date,
                    relevance_score=1.0 - (item.get("position", 0) / 100.0),  # Approximate from position
                    metadata={
                        "position": item.get("position"),
                        "displayed_link": item.get("displayed_link"),
                        "thumbnail": item.get("thumbnail"),
                    },
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to parse SerpAPI result: {e}")
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
