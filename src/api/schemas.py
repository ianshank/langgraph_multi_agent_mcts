"""
API Pydantic Schemas.
"""

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request to trigger MCTS search."""

    problem: str
    test_cases: list[str] = Field(default_factory=list)
    context: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response from MCTS search."""

    solution: str
    solution_value: float
    agent_used: str
    execution_time_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)
