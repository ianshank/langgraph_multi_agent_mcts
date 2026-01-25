"""
Chess API Routes.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict

from src.training.system_config import SystemConfig
from src.api.dependencies import get_system_config
from src.games.chess.adapter import ChessAdapter

router = APIRouter(prefix="/chess", tags=["chess"])

# Singleton Logic (could be moved to dependencies)
_chess_adapter: Optional[ChessAdapter] = None

def get_chess_adapter(config: SystemConfig = Depends(get_system_config)) -> ChessAdapter:
    global _chess_adapter
    if _chess_adapter is None:
        # Check if we have a trained model in registry? 
        # For now, initialize with generic/random weights if no "best" model found.
        # Ideally, we query the ModelRegistry here.
        _chess_adapter = ChessAdapter(config)
    return _chess_adapter

class ChessSearchRequest(BaseModel):
    fen: str
    simulations: int = 800

class ChessSearchResponse(BaseModel):
    best_move: str
    root_value: float
    simulations: int
    policy: Dict[str, int]
    nodes_explored: int

@router.post("/search", response_model=ChessSearchResponse)
async def search_best_move(
    request: ChessSearchRequest,
    adapter: ChessAdapter = Depends(get_chess_adapter)
):
    try:
        result = adapter.search(request.fen, simulations=request.simulations)
        return ChessSearchResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chess search failed: {str(e)}")

class LegalMovesRequest(BaseModel):
    fen: str

@router.post("/legal_moves")
async def get_legal_moves(
    request: LegalMovesRequest,
    adapter: ChessAdapter = Depends(get_chess_adapter)
):
    try:
        moves = adapter.get_legal_moves(request.fen)
        return {"moves": moves}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

class GameFeedback(BaseModel):
    fen_history: list[str]
    policy_history: list[Dict[str, float]]
    outcome: float # 1.0 (White Win), -1.0 (Black Win), 0.0 (Draw)

@router.post("/feedback")
async def report_game_feedback(
    feedback: GameFeedback,
    adapter: ChessAdapter = Depends(get_chess_adapter)
):
    """
    Submit game results for continuous learning.
    """
    # In a full implementation, we would access the ContinuousLearningManager here
    # and push the (state, policy, value) tuples to the replay buffer.
    # For now, we will log it.
    from src.observability.logging import get_structured_logger
    logger = get_structured_logger(__name__)
    
    logger.info("Received game feedback", 
                extra={
                    "moves": len(feedback.fen_history),
                    "outcome": feedback.outcome
                })
    
    # save_to_replay_buffer(feedback) # Future wiring
    return {"status": "accepted"}
