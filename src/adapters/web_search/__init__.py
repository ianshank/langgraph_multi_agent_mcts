"""
Web Search Adapters for Multi-Agent MCTS Framework.

Provides unified interface for multiple web search providers:
- Tavily: Advanced AI search API
- SerpAPI: Google Search API
- DuckDuckGo: Privacy-focused search (no API key required)

All adapters follow the WebSearchClient protocol for provider-agnostic usage.
"""

from .base import SearchResult, WebSearchClient, WebSearchResponse
from .exceptions import (
    WebSearchAPIError,
    WebSearchAuthError,
    WebSearchError,
    WebSearchNetworkError,
    WebSearchRateLimitError,
    WebSearchTimeoutError,
)
from .factory import create_web_search_client

__all__ = [
    # Base protocol and data structures
    "WebSearchClient",
    "SearchResult",
    "WebSearchResponse",
    # Exceptions
    "WebSearchError",
    "WebSearchAPIError",
    "WebSearchAuthError",
    "WebSearchRateLimitError",
    "WebSearchTimeoutError",
    "WebSearchNetworkError",
    # Factory
    "create_web_search_client",
]
