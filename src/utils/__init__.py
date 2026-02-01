"""Utility modules for the LangGraph Multi-Agent MCTS Framework."""

from src.utils.personality_response import PersonalityResponseGenerator

# Lazy imports for optional utilities
__all__ = [
    "PersonalityResponseGenerator",
    "PlanningLoader",
    "get_project_plan",
    "MCTSDebugger",
    "create_debugger",
]


def __getattr__(name: str):
    """Lazy import for utilities to avoid circular imports."""
    if name in ("PlanningLoader", "get_project_plan"):
        from src.utils.planning_loader import PlanningLoader, get_project_plan

        return {"PlanningLoader": PlanningLoader, "get_project_plan": get_project_plan}[name]

    if name in ("MCTSDebugger", "create_debugger"):
        from src.utils.mcts_debug import MCTSDebugger, create_debugger

        return {"MCTSDebugger": MCTSDebugger, "create_debugger": create_debugger}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
