"""
Factory patterns for enterprise use case creation.

Follows the factory pattern established in src/framework/factories.py.
"""

from __future__ import annotations

from .use_case_factory import (
    EnterpriseUseCaseFactory,
    UseCaseFactory,
)

__all__ = [
    "EnterpriseUseCaseFactory",
    "UseCaseFactory",
]
