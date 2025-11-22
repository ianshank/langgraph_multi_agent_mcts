"""
DuckDuckGo Search client (no API key required).

Provides privacy-focused search using DuckDuckGo's HTML search interface.
Note: Less reliable than API-based providers but requires no authentication.
"""

import logging
import re
import time
from typing import Any
from urllib.parse import quote_plus

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import BaseWebSearchClient, SearchResult, WebSearchResponse
from .exceptions import (
    WebSearchAPIError,
    WebSearchNetworkError,
    WebSearchParseError,
    WebSearchTimeoutError,
)

logger = logging.getLogger(__name__)


class DuckDuckGoSearchClient(BaseWebSearchClient):
    """
    DuckDuckGo search client using HTML scraping.

    Privacy-focused search with no API key required.
    Note: Results may be less comprehensive than API-based providers.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_limit_per_minute: int | None = 10,
        cache_ttl_seconds: int = 3600,
        user_agent: str = "LangGraph-Multi-Agent-MCTS/0.1.0",
        base_url: str = "https://duckduckgo.com",
    ):
        """
        Initialize DuckDuckGo client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            rate_limit_per_minute: Requests per minute limit
            cache_ttl_seconds: Cache TTL in seconds
            user_agent: User agent string
            base_url: DuckDuckGo base URL
        """
        super().__init__(
            api_key=None,  # No API key needed
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
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
            },
            follow_redirects=True,
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
        Perform DuckDuckGo search.

        Args:
            query: Search query
            max_results: Max results (1-20)
            include_raw_content: Not supported for DuckDuckGo
            search_type: "general" only (news not supported)
            **kwargs: Additional parameters (region, safe_search)

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

        # Build URL
        params = {
            "q": query,
            "kl": kwargs.get("region", "us-en"),  # Region
        }

        if kwargs.get("safe_search", True):
            params["kp"] = "1"  # Safe search

        try:
            logger.debug(f"DuckDuckGo search: {query[:50]}... (max_results={max_results})")

            response = await self._client.get(
                f"{self.base_url}/html/",
                params=params,
            )

            search_time_ms = (time.perf_counter() - start_time) * 1000

            # Handle errors
            if response.status_code >= 400:
                raise WebSearchAPIError(
                    f"DuckDuckGo error: HTTP {response.status_code}",
                    status_code=response.status_code,
                    provider="duckduckgo",
                )

            # Parse HTML results
            results = self._parse_html_results(response.text, max_results)

            search_response = WebSearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                provider="duckduckgo",
                search_time_ms=search_time_ms,
                cached=False,
                metadata={
                    "region": params.get("kl"),
                    "safe_search": "kp" in params,
                },
            )

            # Update stats and cache
            self._update_stats(search_response)
            await self._cache.set(query, max_results, search_response)

            logger.info(
                f"DuckDuckGo search completed: {len(results)} results in {search_time_ms:.0f}ms "
                f"(cached={search_response.cached})"
            )

            return search_response

        except httpx.TimeoutException as e:
            self._failed_requests += 1
            raise WebSearchTimeoutError(f"DuckDuckGo request timed out: {e}") from e
        except httpx.NetworkError as e:
            self._failed_requests += 1
            raise WebSearchNetworkError(f"DuckDuckGo network error: {e}") from e
        except Exception as e:
            self._failed_requests += 1
            if isinstance(e, WebSearchAPIError):
                raise
            raise WebSearchParseError(f"Failed to parse DuckDuckGo response: {e}") from e

    def _parse_html_results(self, html: str, max_results: int) -> list[SearchResult]:
        """
        Parse DuckDuckGo HTML results.

        This is a simple regex-based parser. For production, consider using BeautifulSoup.
        """
        results = []

        try:
            # Simple regex patterns for DuckDuckGo HTML structure
            # Pattern to match result blocks
            result_pattern = re.compile(
                r'<div class="result.*?">.*?'
                r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>.*?'
                r"</div>",
                re.DOTALL,
            )

            matches = result_pattern.findall(html)

            for i, match in enumerate(matches[:max_results]):
                url, title, snippet = match

                # Clean HTML tags
                title = re.sub(r"<[^>]+>", "", title).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet).strip()

                # Decode URL if needed
                url = url.strip()

                if url and title:
                    result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=self._extract_domain(url),
                        relevance_score=1.0 - (i / 100.0),  # Approximate from position
                        metadata={"position": i + 1},
                    )
                    results.append(result)

        except Exception as e:
            logger.warning(f"Failed to parse DuckDuckGo HTML: {e}")

        # If regex parsing failed, try a simpler approach
        if not results:
            logger.warning("Regex parsing failed, using fallback method")
            results = self._parse_html_fallback(html, max_results)

        return results

    def _parse_html_fallback(self, html: str, max_results: int) -> list[SearchResult]:
        """
        Fallback HTML parser using simpler pattern matching.

        This is even more basic but more robust.
        """
        results = []

        try:
            # Extract URLs
            url_pattern = re.compile(r'href="(https?://[^"]+)"')
            urls = url_pattern.findall(html)

            # Filter out DuckDuckGo internal URLs
            external_urls = [
                url
                for url in urls
                if not any(
                    domain in url for domain in ["duckduckgo.com", "duck.co", "duckduckgo.com/y.js", "duckduckgo.com/d.js"]
                )
            ]

            # Take first N unique URLs
            seen = set()
            for i, url in enumerate(external_urls):
                if url not in seen and len(results) < max_results:
                    seen.add(url)
                    results.append(
                        SearchResult(
                            title=self._extract_domain(url),
                            url=url,
                            snippet=f"Result {i + 1}",
                            source=self._extract_domain(url),
                            relevance_score=1.0 - (i / 100.0),
                            metadata={"position": i + 1, "fallback_parse": True},
                        )
                    )

        except Exception as e:
            logger.error(f"Fallback parsing also failed: {e}")

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
