"""
Base LLM client interface for provider-agnostic model access.

This module defines the protocol and data structures for LLM clients,
enabling seamless switching between providers (OpenAI, Anthropic, LM Studio, etc.)
"""

from typing import Protocol, Any, AsyncIterator, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    text: str
    usage: dict = field(default_factory=dict)
    model: str = ""
    raw_response: Any = None
    finish_reason: str = "stop"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in request/response."""
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        """Tokens used in prompt."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Tokens used in completion."""
        return self.usage.get("completion_tokens", 0)


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""

    id: str
    name: str
    arguments: dict
    type: str = "function"


@dataclass
class LLMToolResponse(LLMResponse):
    """Response containing tool calls."""

    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    This protocol defines the interface that all LLM provider adapters must implement.
    Using Protocol allows for structural subtyping (duck typing) while maintaining
    type safety.
    """

    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts in OpenAI format [{"role": "...", "content": "..."}]
            prompt: Simple string prompt (converted to single user message)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions for function calling
            stream: If True, returns AsyncIterator[str] for streaming
            stop: Stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse if stream=False, AsyncIterator[str] if stream=True

        Raises:
            LLMClientError: Base exception for all client errors
        """
        ...


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Provides common functionality and enforces the interface contract.
    All concrete implementations should inherit from this class.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default",
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for authentication
            model: Model identifier
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._request_count = 0
        self._total_tokens_used = 0

    @abstractmethod
    async def generate(
        self,
        *,
        messages: list[dict] | None = None,
        prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Generate a response from the LLM."""
        pass

    def _build_messages(
        self,
        messages: list[dict] | None = None,
        prompt: str | None = None,
    ) -> list[dict]:
        """
        Build message list from either messages or prompt.

        Args:
            messages: Pre-formatted message list
            prompt: Simple string prompt

        Returns:
            List of message dicts

        Raises:
            ValueError: If neither messages nor prompt provided
        """
        if messages is not None:
            return messages
        elif prompt is not None:
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics."""
        self._request_count += 1
        self._total_tokens_used += response.total_tokens

    @property
    def stats(self) -> dict:
        """Get client statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens_used": self._total_tokens_used,
        }

    async def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
