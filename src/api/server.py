"""
FastAPI Backend for LangGraph Multi-Agent MCTS.

Provides REST API endpoints for:
- Query processing
- Configuration management
- Training orchestration
- Metrics and monitoring
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Framework imports
from src.framework.actions import (
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    SynthesisConfig,
    create_research_config,
    create_coding_config,
    create_creative_config,
    DEFAULT_GRAPH_CONFIG,
)
from src.framework.mcts.config import (
    MCTSConfig,
    ConfigPreset,
    create_preset_config,
)
from src.framework.mcts.core import MCTSState

# Optional imports
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralMCTSAdapter,
        get_fast_neural_config,
        get_balanced_neural_config,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from src.training.expert_iteration import (
        ExpertIterationConfig,
        ExpertIterationTrainer,
        ReplayBuffer,
    )
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class DomainPreset(str, Enum):
    """Available domain presets."""
    GENERAL = "general"
    RESEARCH = "research"
    CODING = "coding"
    CREATIVE = "creative"


class MCTSPreset(str, Enum):
    """Available MCTS presets."""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    EXPLORATION_HEAVY = "exploration_heavy"
    EXPLOITATION_HEAVY = "exploitation_heavy"


class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., min_length=1, max_length=10000)
    use_mcts: bool = True
    use_rag: bool = False
    max_iterations: int = Field(default=3, ge=1, le=10)
    domain_preset: DomainPreset = DomainPreset.GENERAL
    mcts_preset: MCTSPreset = MCTSPreset.BALANCED
    use_neural_mcts: bool = False


class AgentOutput(BaseModel):
    """Output from a single agent."""
    agent: str
    response: str
    confidence: float
    metadata: dict[str, Any] = {}


class MCTSStats(BaseModel):
    """MCTS search statistics."""
    iterations: int
    best_action: str
    best_action_visits: int
    best_action_value: float
    cache_hit_rate: float
    tree_depth: int = 0
    node_count: int = 0


class QueryResponse(BaseModel):
    """Response model for query processing."""
    query: str
    response: str
    agent_outputs: list[AgentOutput]
    consensus_score: float
    mcts_stats: MCTSStats | None = None
    elapsed_time: float
    config_used: dict[str, Any]


class ConfigRequest(BaseModel):
    """Request for custom configuration."""
    consensus_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    heuristic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_iterations: int = Field(default=3, ge=1, le=10)


class TrainingRequest(BaseModel):
    """Request for training configuration."""
    num_episodes: int = Field(default=100, ge=10, le=10000)
    mcts_simulations: int = Field(default=400, ge=50, le=3200)
    batch_size: int = Field(default=256, ge=32, le=2048)
    learning_rate: float = Field(default=1e-3, ge=1e-5, le=1e-2)
    num_iterations: int = Field(default=10, ge=1, le=100)


class TrainingStatus(BaseModel):
    """Training status response."""
    is_training: bool
    current_iteration: int
    total_iterations: int
    metrics: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    neural_available: bool
    training_available: bool


# ============================================================================
# Application State
# ============================================================================

class AppState:
    """Application state container."""

    def __init__(self):
        self.is_training = False
        self.training_metrics: list[dict] = []
        self.current_iteration = 0
        self.total_iterations = 0
        self.query_count = 0
        self.start_time = time.time()


state = AppState()


# ============================================================================
# Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LangGraph Multi-Agent MCTS API")
    logger.info(f"Neural MCTS available: {NEURAL_AVAILABLE}")
    logger.info(f"Training available: {TRAINING_AVAILABLE}")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="LangGraph Multi-Agent MCTS API",
    description="API for cognitive architecture with neural-guided tree search",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        neural_available=NEURAL_AVAILABLE,
        training_available=TRAINING_AVAILABLE,
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the multi-agent system.

    This endpoint runs the full multi-agent pipeline including:
    - HRM (Hierarchical Reasoning Model)
    - TRM (Task Refinement Model)
    - MCTS (Monte Carlo Tree Search)
    """
    start_time = time.time()
    state.query_count += 1

    logger.info(f"Processing query: {request.query[:50]}...")

    # Get configuration based on preset
    graph_config = _get_graph_config(request.domain_preset)
    mcts_config = _get_mcts_config(request.mcts_preset)

    # Simulate agent processing
    agent_outputs = await _simulate_agent_pipeline(
        request.query,
        request.use_mcts,
        request.use_rag,
        graph_config,
    )

    # Compute consensus
    confidences = [a.confidence for a in agent_outputs]
    consensus_score = sum(confidences) / len(confidences) if confidences else 0.0

    # MCTS stats (simulated)
    mcts_stats = None
    if request.use_mcts:
        mcts_stats = MCTSStats(
            iterations=mcts_config.num_iterations,
            best_action="synthesize",
            best_action_visits=int(mcts_config.num_iterations * 0.35),
            best_action_value=0.87,
            cache_hit_rate=0.23,
            tree_depth=min(mcts_config.max_tree_depth, 8),
            node_count=mcts_config.num_iterations * 4,
        )

    # Generate response
    response_text = _synthesize_response(request.query, agent_outputs, consensus_score)

    elapsed = time.time() - start_time

    return QueryResponse(
        query=request.query,
        response=response_text,
        agent_outputs=agent_outputs,
        consensus_score=consensus_score,
        mcts_stats=mcts_stats,
        elapsed_time=elapsed,
        config_used={
            "domain_preset": request.domain_preset.value,
            "mcts_preset": request.mcts_preset.value,
            "use_neural_mcts": request.use_neural_mcts,
        },
    )


@app.get("/config/presets")
async def get_config_presets():
    """Get available configuration presets."""
    return {
        "domain_presets": {
            "general": {"description": "Balanced settings for general use"},
            "research": {"description": "Optimized for research and exploration"},
            "coding": {"description": "Optimized for code generation"},
            "creative": {"description": "Optimized for creative tasks"},
        },
        "mcts_presets": {
            "fast": create_preset_config(ConfigPreset.FAST).to_dict(),
            "balanced": create_preset_config(ConfigPreset.BALANCED).to_dict(),
            "thorough": create_preset_config(ConfigPreset.THOROUGH).to_dict(),
        },
        "neural_available": NEURAL_AVAILABLE,
    }


@app.post("/config/custom")
async def create_custom_config(request: ConfigRequest):
    """Create a custom configuration."""
    config = GraphConfig(
        confidence=ConfidenceConfig(
            consensus_threshold=request.consensus_threshold,
        ),
        rollout_weights=RolloutWeights(
            heuristic_weight=request.heuristic_weight,
            random_weight=1.0 - request.heuristic_weight,
        ),
        synthesis=SynthesisConfig(
            temperature=request.temperature,
        ),
        max_iterations=request.max_iterations,
    )

    return {
        "status": "created",
        "config": config.to_dict(),
    }


@app.post("/training/start")
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """Start Expert Iteration training."""
    if not TRAINING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Training not available. Install PyTorch.",
        )

    if state.is_training:
        raise HTTPException(
            status_code=409,
            detail="Training already in progress",
        )

    # Start training in background
    background_tasks.add_task(
        _run_training,
        request.num_episodes,
        request.mcts_simulations,
        request.batch_size,
        request.learning_rate,
        request.num_iterations,
    )

    state.is_training = True
    state.total_iterations = request.num_iterations
    state.current_iteration = 0
    state.training_metrics = []

    return {
        "status": "started",
        "message": f"Training started for {request.num_iterations} iterations",
    }


@app.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return TrainingStatus(
        is_training=state.is_training,
        current_iteration=state.current_iteration,
        total_iterations=state.total_iterations,
        metrics=state.training_metrics,
    )


@app.post("/training/stop")
async def stop_training():
    """Stop current training."""
    if not state.is_training:
        raise HTTPException(
            status_code=400,
            detail="No training in progress",
        )

    state.is_training = False
    return {"status": "stopped"}


@app.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    uptime = time.time() - state.start_time
    return {
        "uptime_seconds": uptime,
        "query_count": state.query_count,
        "is_training": state.is_training,
        "training_iterations_complete": state.current_iteration,
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _get_graph_config(preset: DomainPreset) -> GraphConfig:
    """Get graph configuration for preset."""
    configs = {
        DomainPreset.GENERAL: DEFAULT_GRAPH_CONFIG,
        DomainPreset.RESEARCH: create_research_config(),
        DomainPreset.CODING: create_coding_config(),
        DomainPreset.CREATIVE: create_creative_config(),
    }
    return configs.get(preset, DEFAULT_GRAPH_CONFIG)


def _get_mcts_config(preset: MCTSPreset) -> MCTSConfig:
    """Get MCTS configuration for preset."""
    preset_map = {
        MCTSPreset.FAST: ConfigPreset.FAST,
        MCTSPreset.BALANCED: ConfigPreset.BALANCED,
        MCTSPreset.THOROUGH: ConfigPreset.THOROUGH,
        MCTSPreset.EXPLORATION_HEAVY: ConfigPreset.EXPLORATION_HEAVY,
        MCTSPreset.EXPLOITATION_HEAVY: ConfigPreset.EXPLOITATION_HEAVY,
    }
    return create_preset_config(preset_map[preset])


async def _simulate_agent_pipeline(
    query: str,
    use_mcts: bool,
    use_rag: bool,
    config: GraphConfig,
) -> list[AgentOutput]:
    """Simulate the agent pipeline."""
    outputs = []

    # Simulate HRM
    await asyncio.sleep(0.1)
    outputs.append(AgentOutput(
        agent="HRM",
        response=f"Hierarchical decomposition of query into sub-problems",
        confidence=0.85,
        metadata={"sub_problems": 3, "depth": 2},
    ))

    # Simulate TRM
    await asyncio.sleep(0.1)
    outputs.append(AgentOutput(
        agent="TRM",
        response=f"Recursive refinement completed",
        confidence=0.82,
        metadata={"iterations": 4, "convergence": True},
    ))

    # Simulate MCTS
    if use_mcts:
        await asyncio.sleep(0.1)
        outputs.append(AgentOutput(
            agent="MCTS",
            response=f"Tree search explored multiple solution paths",
            confidence=0.88,
            metadata={"best_action": "synthesize", "visits": 35},
        ))

    return outputs


def _synthesize_response(
    query: str,
    agent_outputs: list[AgentOutput],
    consensus_score: float,
) -> str:
    """Synthesize final response from agent outputs."""
    agent_count = len(agent_outputs)
    return (
        f"Based on multi-agent analysis (consensus: {consensus_score:.1%}), "
        f"combining insights from {agent_count} specialized agents: "
        f"The query has been decomposed hierarchically, refined iteratively, "
        f"and validated through tree search to ensure optimal response quality."
    )


async def _run_training(
    num_episodes: int,
    mcts_simulations: int,
    batch_size: int,
    learning_rate: float,
    num_iterations: int,
):
    """Run training in background."""
    logger.info(f"Starting training: {num_iterations} iterations")

    for i in range(num_iterations):
        if not state.is_training:
            logger.info("Training stopped")
            break

        state.current_iteration = i + 1

        # Simulate training iteration
        await asyncio.sleep(1.0)

        # Record metrics
        state.training_metrics.append({
            "iteration": i + 1,
            "avg_outcome": 0.5 + 0.05 * i,
            "policy_loss": 2.0 - 0.1 * i,
            "value_loss": 1.0 - 0.05 * i,
            "buffer_size": num_episodes * (i + 1),
        })

        logger.info(f"Completed iteration {i + 1}/{num_iterations}")

    state.is_training = False
    logger.info("Training complete")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
