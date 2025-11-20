"""
Adapters package for external service integrations.
"""

from .llm import BaseLLMClient, LLMResponse, create_client

__all__ = ["create_client", "BaseLLMClient", "LLMResponse"]
