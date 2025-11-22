"""
Base web search client interface and data structures.

Defines the protocol that all web search provider adapters must implement,
enabling seamless switching between providers.
"""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """Represents a single search result."""

    title: str
    url: str
    snippet: str  # Short description/summary
    source: str = ""  # Domain or publisher name
    raw_content: str | None = None  # Full text content for RAG (optional)
    published_date: datetime | None = None
    relevance_score: float = 0.0  # Provider-specific relevance (0-1)
    metadata: dict = field(default_factory=dict)  # Provider-specific metadata

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "raw_content": self.raw_content,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }

    def get_content_hash(self) -> str:
        """Generate deterministic hash of content for deduplication."""
        content = f"{self.url}:{self.title}:{self.snippet}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class WebSearchResponse:
    """Standardized response from any web search provider."""

    query: str
    results: list[SearchResult]
    total_results: int = 0  # Estimated total (if available)
    provider: str = ""
    search_time_ms: float = 0.0
    cached: bool = False
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "provider": self.provider,
            "search_time_ms": self.search_time_ms,
            "cached": self.cached,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def get_top_k_results(self, k: int) -> list[SearchResult]:
        """Get top k results by relevance score."""
        return sorted(self.results, key=lambda r: r.relevance_score, reverse=True)[:k]


class SearchCache:
    """
    Simple in-memory cache for search results.

    Uses LRU eviction when size limit is reached.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        """
        Initialize search cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum number of entries to cache
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, tuple[WebSearchResponse, float]] = {}
        self._access_times: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, max_results: int) -> str:
        """Create cache key from query parameters."""
        content = f"{query.lower().strip()}:{max_results}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(self, query: str, max_results: int) -> WebSearchResponse | None:
        """Get cached result if available and not expired."""
        async with self._lock:
            key = self._make_key(query, max_results)
            if key in self._cache:
                response, timestamp = self._cache[key]
                age = time.time() - timestamp

                if age < self.ttl_seconds:
                    self._access_times[key] = time.time()
                    self._hits += 1
                    # Mark as cached
                    response.cached = True
                    return response
                else:
                    # Expired
                    del self._cache[key]
                    del self._access_times[key]

            self._misses += 1
            return None

    async def set(self, query: str, max_results: int, response: WebSearchResponse) -> None:
        """Cache a search response."""
        async with self._lock:
            key = self._make_key(query, max_results)

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]

            self._cache[key] = (response, time.time())
            self._access_times[key] = time.time()

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
        }


@runtime_checkable
class WebSearchClient(Protocol):
    """
    Protocol for web search clients.

    All web search provider adapters must implement this interface.
    """

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
        Perform a web search.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (1-20)
            include_raw_content: Fetch full page content for RAG
            search_type: Type of search ("general", "news", "academic", etc.)
            **kwargs: Provider-specific parameters

        Returns:
            WebSearchResponse with results

        Raises:
            WebSearchError: Base exception for all search errors
        """
        ...


class BaseWebSearchClient(ABC):
    """
    Abstract base class for web search clients.

    Provides common functionality and enforces the interface contract.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        rate_limit_per_minute: int | None = None,
        cache_ttl_seconds: int = 3600,
        user_agent: str = "LangGraph-Multi-Agent-MCTS/0.1.0",
    ):
        """
        Initialize web search client.

        Args:
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_per_minute: Rate limit (requests per minute), None to disable
            cache_ttl_seconds: Cache time-to-live in seconds
            user_agent: User agent string for requests
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._request_count = 0
        self._failed_requests = 0
        self._total_search_time_ms = 0.0

        # Initialize cache
        self._cache = SearchCache(ttl_seconds=cache_ttl_seconds, max_size=100)

        # Rate limiter (to be implemented similar to LLM client if needed)
        self._rate_limiter = None
        if rate_limit_per_minute:
            # Import and use rate limiter from LLM base
            from src.adapters.llm.base import TokenBucketRateLimiter

            self._rate_limiter = TokenBucketRateLimiter(rate_per_minute=rate_limit_per_minute)

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        include_raw_content: bool = False,
        search_type: str = "general",
        **kwargs: Any,
    ) -> WebSearchResponse:
        """Perform a web search."""
        pass

    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting if configured."""
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

    def _update_stats(self, response: WebSearchResponse) -> None:
        """Update internal statistics."""
        self._request_count += 1
        self._total_search_time_ms += response.search_time_ms

    @property
    def stats(self) -> dict:
        """Get client statistics."""
        base_stats = {
            "request_count": self._request_count,
            "failed_requests": self._failed_requests,
            "total_search_time_ms": self._total_search_time_ms,
            "avg_search_time_ms": (
                self._total_search_time_ms / self._request_count if self._request_count > 0 else 0.0
            ),
        }

        # Include cache stats
        base_stats["cache"] = self._cache.stats

        # Include rate limiter stats if available
        if self._rate_limiter is not None:
            base_stats.update(self._rate_limiter.stats)

        return base_stats

    async def close(self) -> None:
        """Clean up resources."""
        await self._cache.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
