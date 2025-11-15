"""
Custom exceptions for LLM client operations.

Provides a hierarchy of structured exceptions for better error handling
and debugging across different LLM providers.
"""


class LLMClientError(Exception):
    """Base exception for all LLM client errors."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [f"[{self.provider}] {self.message}"]
        if self.status_code:
            parts.append(f"(status: {self.status_code})")
        return " ".join(parts)


class LLMAuthenticationError(LLMClientError):
    """Authentication failed - invalid or missing API key."""

    def __init__(self, provider: str, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            provider=provider,
            status_code=401,
        )


class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded - too many requests."""

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
        message: str = "Rate limit exceeded",
    ):
        super().__init__(
            message=message,
            provider=provider,
            status_code=429,
            retry_after=retry_after,
        )


class LLMQuotaExceededError(LLMClientError):
    """Quota or credits exhausted."""

    def __init__(self, provider: str, message: str = "Quota exceeded"):
        super().__init__(
            message=message,
            provider=provider,
            status_code=402,
        )


class LLMModelNotFoundError(LLMClientError):
    """Requested model not available."""

    def __init__(self, provider: str, model: str):
        super().__init__(
            message=f"Model '{model}' not found or not available",
            provider=provider,
            status_code=404,
        )


class LLMContextLengthError(LLMClientError):
    """Input exceeds model's context window."""

    def __init__(
        self,
        provider: str,
        token_count: int | None = None,
        max_tokens: int | None = None,
    ):
        message = "Context length exceeded"
        if token_count and max_tokens:
            message = f"Context length exceeded: {token_count} tokens provided, max is {max_tokens}"
        super().__init__(
            message=message,
            provider=provider,
            status_code=400,
        )


class LLMInvalidRequestError(LLMClientError):
    """Invalid request parameters."""

    def __init__(self, provider: str, message: str = "Invalid request parameters"):
        super().__init__(
            message=message,
            provider=provider,
            status_code=400,
        )


class LLMTimeoutError(LLMClientError):
    """Request timed out."""

    def __init__(self, provider: str, timeout: float):
        super().__init__(
            message=f"Request timed out after {timeout}s",
            provider=provider,
            status_code=408,
        )


class LLMConnectionError(LLMClientError):
    """Failed to connect to the API endpoint."""

    def __init__(self, provider: str, url: str | None = None):
        message = "Failed to connect to API"
        if url:
            message = f"Failed to connect to {url}"
        super().__init__(
            message=message,
            provider=provider,
        )


class LLMServerError(LLMClientError):
    """Server-side error from the LLM provider."""

    def __init__(
        self,
        provider: str,
        status_code: int = 500,
        message: str = "Server error",
    ):
        super().__init__(
            message=message,
            provider=provider,
            status_code=status_code,
        )


class LLMResponseParseError(LLMClientError):
    """Failed to parse response from LLM provider."""

    def __init__(self, provider: str, raw_response: str | None = None):
        message = "Failed to parse response"
        if raw_response:
            preview = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
            message = f"Failed to parse response: {preview}"
        super().__init__(
            message=message,
            provider=provider,
        )


class LLMStreamError(LLMClientError):
    """Error during streaming response."""

    def __init__(self, provider: str, message: str = "Stream interrupted"):
        super().__init__(
            message=message,
            provider=provider,
        )


class LLMContentFilterError(LLMClientError):
    """Content blocked by safety filters."""

    def __init__(self, provider: str, reason: str | None = None):
        message = "Content blocked by safety filters"
        if reason:
            message = f"Content blocked: {reason}"
        super().__init__(
            message=message,
            provider=provider,
            status_code=400,
        )


class CircuitBreakerOpenError(LLMClientError):
    """Circuit breaker is open, requests are being blocked."""

    def __init__(
        self,
        provider: str,
        failure_count: int,
        reset_time: float,
    ):
        super().__init__(
            message=f"Circuit breaker open after {failure_count} failures. Resets in {reset_time:.1f}s",
            provider=provider,
        )
        self.failure_count = failure_count
        self.reset_time = reset_time
