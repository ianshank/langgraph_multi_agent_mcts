"""
API module for LangGraph Multi-Agent MCTS Framework.

Provides:
- Authentication and authorization
- Rate limiting
- Error handling
- REST API endpoints
"""

from src.api.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    FrameworkError,
    LLMError,
    MCTSError,
    RAGError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)

__all__ = [
    "FrameworkError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "LLMError",
    "MCTSError",
    "RAGError",
    "TimeoutError",
    "ConfigurationError",
]
