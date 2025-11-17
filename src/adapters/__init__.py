"""
Adapters package for external service integrations.
"""

from .llm import create_client, BaseLLMClient, LLMResponse

__all__ = ["create_client", "BaseLLMClient", "LLMResponse"]
