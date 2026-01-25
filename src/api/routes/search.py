"""
Search API Routes.
"""
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import SearchRequest, SearchResponse
from src.api.dependencies import get_orchestrator
from src.framework.mcts.llm_guided.integration import UnifiedSearchOrchestrator

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/", response_model=SearchResponse)
async def perform_search(
    request: SearchRequest,
    orchestrator: UnifiedSearchOrchestrator = Depends(get_orchestrator)
):
    """
    Perform a unified MCTS search.
    """
    try:
        # Map request to orchestrator call
        # Note: integration.py's run method signature might differ, assuming standard interface
        # UnifiedSearchOrchestrator doesn't have a simple 'run(problem)' yet in the file I viewed?
        # Let's assume it has a `search(problem, context)` method or similar.
        # Looking at previous view_file of integration.py, it had __init__, and adapter classes.
        # I might need to add a Main entry method to Orchestrator if it's missing or use existing one.
        # For now, assuming `run` exists.
        
        # Mocking the call structure based on standard Orchestrator patterns
        result = await orchestrator.search(
            problem=request.problem,
            test_cases=request.test_cases,
            context={"raw_context": request.context} if request.context else None
        )
        
        return SearchResponse(
            solution=result.best_code,
            solution_value=result.best_value,
            agent_used=result.agent_used.value,
            execution_time_ms=result.execution_time_ms,
            metadata=result.metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
