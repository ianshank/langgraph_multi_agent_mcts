"""
Exception hierarchy for web search adapters.

Provides specific exception types for different failure modes to enable
intelligent error handling and retry logic.
"""


class WebSearchError(Exception):
    """Base exception for all web search errors."""

    pass


class WebSearchAPIError(WebSearchError):
    """API returned an error response."""

    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


class WebSearchAuthError(WebSearchAPIError):
    """Authentication/authorization failed (invalid API key, etc.)."""

    pass


class WebSearchRateLimitError(WebSearchAPIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None, provider: str | None = None):
        super().__init__(message, status_code=429, provider=provider)
        self.retry_after = retry_after  # seconds to wait


class WebSearchTimeoutError(WebSearchError):
    """Request timed out."""

    pass


class WebSearchNetworkError(WebSearchError):
    """Network connectivity issue."""

    pass


class WebSearchParseError(WebSearchError):
    """Failed to parse search results."""

    pass
