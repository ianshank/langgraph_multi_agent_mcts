"""
API Pydantic Schemas.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """Request to trigger MCTS search."""
    problem: str
    test_cases: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    config_overrides: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    """Response from MCTS search."""
    solution: str
    solution_value: float
    agent_used: str
    execution_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
