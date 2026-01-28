"""
Vertex AI Cloud Run Application

This FastAPI application serves the LangGraph Multi-Agent MCTS framework
on Google Cloud Run with Vertex AI integration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application version
APP_VERSION = "1.0.0"


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The query to process")
    controller_type: str = Field(default="rnn", description="Controller type: 'rnn' or 'bert'")
    use_vertex_ai: bool = Field(default=True, description="Use Vertex AI for LLM calls")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    response: str = Field(..., description="Agent response")
    agent: str = Field(..., description="Selected agent name")
    confidence: float = Field(..., description="Agent confidence score")
    routing_probabilities: dict[str, float] = Field(..., description="Routing probabilities")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    timestamp: str
    google_cloud_project: str | None
    vertex_ai_enabled: bool


# Global framework instance
_framework = None


def get_framework():
    """Get or initialize the framework."""
    global _framework
    if _framework is None:
        from src.agents.meta_controller.bert_controller_v2 import BERTMetaController
        from src.agents.meta_controller.rnn_controller import RNNMetaController

        _framework = {
            "rnn": RNNMetaController(name="RNNController", seed=42),
            "bert": BERTMetaController(name="BERTController", seed=42),
        }
        logger.info("Framework initialized successfully")
    return _framework


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting Vertex AI MCTS Application v{APP_VERSION}")
    logger.info(f"Google Cloud Project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'Not Set')}")
    logger.info(f"Location: {os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')}")

    # Initialize framework
    get_framework()

    yield

    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="LangGraph Multi-Agent MCTS API",
    description="Production API for the LangGraph Multi-Agent MCTS framework with Vertex AI integration",
    version=APP_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LangGraph Multi-Agent MCTS API",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Cloud Run."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        vertex_ai_enabled=os.getenv("ADK_BACKEND") == "vertex_ai",
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query using the multi-agent framework.

    The meta-controller routes the query to the optimal agent based on
    query characteristics and returns the processed response.
    """
    start_time = time.perf_counter()

    try:
        framework = get_framework()

        # Import feature extraction
        from src.agents.meta_controller.base import MetaControllerFeatures

        # Create simple features from query
        query_length = len(request.query)
        has_technical = any(
            word in request.query.lower()
            for word in ["algorithm", "code", "implement", "technical", "system"]
        )

        features = MetaControllerFeatures(
            hrm_confidence=0.5,
            trm_confidence=0.5,
            mcts_value=0.5,
            consensus_score=0.7,
            last_agent="none",
            iteration=0,
            query_length=query_length,
            has_rag_context=query_length > 50,
            rag_relevance_score=0.7 if query_length > 50 else 0.0,
            is_technical_query=has_technical,
        )

        # Get controller prediction
        controller = framework.get(request.controller_type, framework["rnn"])
        prediction = controller.predict(features)

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return QueryResponse(
            response=f"[{prediction.agent.upper()} Agent] Processing query: {request.query[:100]}...",
            agent=prediction.agent,
            confidence=prediction.confidence,
            routing_probabilities=prediction.probabilities,
            execution_time_ms=round(execution_time_ms, 2),
            metadata={
                "controller_type": request.controller_type,
                "vertex_ai_enabled": request.use_vertex_ai,
                "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT"),
            },
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vertex-ai/invoke")
async def vertex_ai_invoke(request: Request):
    """
    Invoke the framework using Vertex AI Agent Engine compatible format.

    This endpoint is designed for compatibility with Vertex AI Agent Engine
    deployment patterns.
    """
    try:
        body = await request.json()
        query = body.get("query", body.get("input", ""))

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Process using the query endpoint
        query_request = QueryRequest(query=query, use_vertex_ai=True)
        result = await process_query(query_request)

        return {
            "output": result.response,
            "metadata": {
                "agent": result.agent,
                "confidence": result.confidence,
                "routing_probabilities": result.routing_probabilities,
                "execution_time_ms": result.execution_time_ms,
            },
        }

    except Exception as e:
        logger.error(f"Vertex AI invoke error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List available agents and their capabilities."""
    return {
        "agents": [
            {
                "name": "hrm",
                "description": "Hierarchical Reasoning Module",
                "capabilities": ["complex decomposition", "multi-step reasoning"],
            },
            {
                "name": "trm",
                "description": "Task Refinement Module",
                "capabilities": ["iterative refinement", "solution optimization"],
            },
            {
                "name": "mcts",
                "description": "Monte Carlo Tree Search",
                "capabilities": ["strategic exploration", "optimization"],
            },
        ],
        "meta_controllers": [
            {
                "name": "rnn",
                "description": "GRU-based sequential pattern recognition",
            },
            {
                "name": "bert",
                "description": "Transformer-based text understanding with LoRA",
            },
        ],
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
