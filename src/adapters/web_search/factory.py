"""
Factory for creating web search clients based on configuration.

Provides a unified interface for instantiating different search providers.
"""

import logging
from typing import Any

from src.config.settings import Settings, WebSearchProvider, get_settings

from .base import WebSearchClient
from .duckduckgo_client import DuckDuckGoSearchClient
from .exceptions import WebSearchError
from .serpapi_client import SerpAPISearchClient
from .tavily_client import TavilySearchClient

logger = logging.getLogger(__name__)


def create_web_search_client(
    settings: Settings | None = None,
    provider: str | WebSearchProvider | None = None,
    **kwargs: Any,
) -> WebSearchClient:
    """
    Create a web search client based on configuration.

    Args:
        settings: Settings instance (uses global if None)
        provider: Override provider from settings
        **kwargs: Additional client parameters

    Returns:
        WebSearchClient instance

    Raises:
        WebSearchError: If provider is invalid or credentials missing
    """
    if settings is None:
        settings = get_settings()

    # Determine provider
    if provider is None:
        provider = settings.WEB_SEARCH_PROVIDER
    elif isinstance(provider, str):
        try:
            provider = WebSearchProvider(provider.lower())
        except ValueError:
            raise WebSearchError(f"Invalid web search provider: {provider}")

    # Check if web search is enabled
    if not settings.WEB_SEARCH_ENABLED and provider != WebSearchProvider.DISABLED:
        logger.warning(
            f"Web search is disabled but provider {provider} requested. "
            "Set WEB_SEARCH_ENABLED=True to enable web search."
        )

    # Common parameters from settings
    client_params = {
        "timeout": kwargs.get("timeout", settings.WEB_SEARCH_TIMEOUT_SECONDS),
        "max_retries": kwargs.get("max_retries", settings.HTTP_MAX_RETRIES),
        "rate_limit_per_minute": kwargs.get("rate_limit_per_minute", settings.WEB_SEARCH_RATE_LIMIT_PER_MINUTE),
        "cache_ttl_seconds": kwargs.get("cache_ttl_seconds", settings.WEB_SEARCH_CACHE_TTL_SECONDS),
        "user_agent": kwargs.get("user_agent", settings.WEB_SEARCH_USER_AGENT),
    }

    # Create client based on provider
    if provider == WebSearchProvider.TAVILY:
        api_key = settings.get_tavily_api_key()
        if not api_key:
            raise WebSearchError(
                "Tavily API key not found. Set TAVILY_API_KEY environment variable."
            )

        logger.info("Creating Tavily search client")
        return TavilySearchClient(
            api_key=api_key,
            **client_params,
        )

    elif provider == WebSearchProvider.SERPAPI:
        api_key = settings.get_serpapi_api_key()
        if not api_key:
            raise WebSearchError(
                "SerpAPI key not found. Set SERPAPI_API_KEY environment variable."
            )

        logger.info("Creating SerpAPI search client")
        return SerpAPISearchClient(
            api_key=api_key,
            **client_params,
        )

    elif provider == WebSearchProvider.DUCKDUCKGO:
        logger.info("Creating DuckDuckGo search client (no API key required)")
        return DuckDuckGoSearchClient(
            **{k: v for k, v in client_params.items() if k != "api_key"}
        )

    elif provider == WebSearchProvider.DISABLED:
        raise WebSearchError(
            "Web search is disabled. Set WEB_SEARCH_PROVIDER to a valid provider "
            "(tavily, serpapi, duckduckgo)"
        )

    else:
        raise WebSearchError(f"Unsupported web search provider: {provider}")


# Convenience function for common use case
async def search_web(
    query: str,
    *,
    max_results: int = 5,
    include_raw_content: bool = False,
    settings: Settings | None = None,
    **kwargs: Any,
):
    """
    Convenience function for performing web search.

    Args:
        query: Search query
        max_results: Maximum results to return
        include_raw_content: Include full page content
        settings: Settings instance
        **kwargs: Additional search parameters

    Returns:
        WebSearchResponse

    Example:
        >>> results = await search_web("LangGraph multi-agent systems")
        >>> for result in results.results:
        ...     print(f"{result.title}: {result.url}")
    """
    client = create_web_search_client(settings=settings)
    try:
        return await client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            **kwargs,
        )
    finally:
        await client.close()
