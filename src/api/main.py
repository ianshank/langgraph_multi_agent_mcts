"""
API Main Entry Point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events."""
    logger.info("Starting MCTS API Service...")
    # Initialize global resources (Orchestrator, DB) here
    yield
    logger.info("Shutting down MCTS API Service...")


app = FastAPI(title="LangGraph MCTS Orchestrator", version="1.0.0", lifespan=lifespan)

from src.api.routes import chess, models, search

app.include_router(search.router)
app.include_router(models.router)
app.include_router(chess.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "mcts-orchestrator"}
