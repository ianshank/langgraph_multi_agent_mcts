"""
Base classes and protocols for enterprise use cases.

This module defines the core abstractions that all enterprise use cases
must implement, ensuring consistency and interoperability across domains.
"""

from __future__ import annotations

from .use_case import (
    BaseDomainState,
    BaseUseCase,
    DomainAgentProtocol,
    RewardFunctionProtocol,
    UseCaseProtocol,
)

__all__ = [
    "BaseUseCase",
    "BaseDomainState",
    "DomainAgentProtocol",
    "RewardFunctionProtocol",
    "UseCaseProtocol",
]
