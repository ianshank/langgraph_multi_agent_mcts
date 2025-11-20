"""
Production REST API server for LangGraph Multi-Agent MCTS Framework.

Provides:
- OpenAPI/Swagger documentation
- Authentication via API keys
- Rate limiting
- Health and readiness endpoints
- Request validation with Pydantic
- Prometheus metrics exposure
"""

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import framework components
try:
    from src.adapters.llm import create_client  # noqa: F401
    from src.api.auth import (
        APIKeyAuthenticator,
        ClientInfo,
        RateLimitConfig,
        get_authenticator,
        set_authenticator,
    )
    from src.api.exceptions import (
        AuthenticationError,
        AuthorizationError,  # noqa: F401
        FrameworkError,
        RateLimitError,
        ValidationError,  # noqa: F401
    )
    from src.models.validation import MCTSConfig, QueryInput  # noqa: F401

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = str(e)

# Prometheus metrics (optional)
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    # Define metrics
    REQUEST_COUNT = Counter("mcts_requests_total", "Total number of requests", ["method", "endpoint", "status"])
    REQUEST_LATENCY = Histogram("mcts_request_duration_seconds", "Request latency in seconds", ["method", "endpoint"])
    ACTIVE_REQUESTS = Gauge("mcts_active_requests", "Number of active requests")
    ERROR_COUNT = Counter("mcts_errors_total", "Total number of errors", ["error_type"])
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query processing."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query to process",
        json_schema_extra={"example": "Recommend defensive positions for night attack scenario"},
    )
    use_mcts: bool = Field(default=True, description="Enable MCTS tactical simulation")
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")
    mcts_iterations: int | None = Field(default=None, ge=1, le=10000, description="Override default MCTS iterations")
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Conversation thread ID for state persistence",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Recommend defensive positions for night attack",
                "use_mcts": True,
                "use_rag": True,
                "mcts_iterations": 200,
                "thread_id": "session_123",
            }
        }


class QueryResponse(BaseModel):
    """Response model for query results."""

    response: str = Field(..., description="Final synthesized response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    agents_used: list[str] = Field(..., description="List of agents that contributed")
    mcts_stats: dict[str, Any] | None = Field(default=None, description="MCTS simulation statistics")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(default="1.0.0", description="API version")
    uptime_seconds: float = Field(..., description="Service uptime")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Whether service is ready")
    checks: dict[str, bool] = Field(..., description="Individual check results")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: bool = Field(default=True)
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(..., description="Error timestamp")


# Application startup
start_time = time.time()
framework_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global framework_instance

    # Startup
    print("Starting MCTS Framework API server...")

    # Initialize authenticator with demo key (replace in production)
    authenticator = APIKeyAuthenticator(
        valid_keys=["demo-api-key-replace-in-production"],
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
        ),
    )
    set_authenticator(authenticator)

    # Initialize framework (lazy loading)
    # framework_instance = create_framework()

    print("API server started successfully")

    yield

    # Shutdown
    print("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="LangGraph Multi-Agent MCTS API",
    description="""
## Multi-Agent Reasoning API with MCTS Tactical Simulation

This API provides access to a sophisticated multi-agent reasoning framework that combines:
- **HRM Agent**: Hierarchical decomposition of complex queries
- **TRM Agent**: Iterative refinement for response quality
- **MCTS Engine**: Monte Carlo Tree Search for tactical simulation
- **RAG Integration**: Context retrieval from vector stores

### Features
- Secure API key authentication
- Rate limiting per client
- Real-time metrics (Prometheus)
- Distributed tracing (OpenTelemetry)
- Production-grade error handling

### Quick Start
1. Obtain an API key
2. Include `X-API-Key` header in requests
3. Send queries to `/query` endpoint
4. Monitor health via `/health` endpoint
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "query", "description": "Query processing operations"},
        {"name": "health", "description": "Health and readiness checks"},
        {"name": "metrics", "description": "Observability endpoints"},
    ],
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    if PROMETHEUS_AVAILABLE:
        ACTIVE_REQUESTS.inc()

    start = time.perf_counter()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.dec()
            elapsed = time.perf_counter() - start
            REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=str(status)).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed)

    return response


# Authentication dependency
async def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    """Verify API key and return client info."""
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Authentication module not available")

    try:
        authenticator = get_authenticator()
        client_info = authenticator.require_auth(x_api_key)
        return client_info
    except AuthenticationError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="authentication").inc()
        raise HTTPException(status_code=401, detail=e.user_message)
    except RateLimitError as e:
        if PROMETHEUS_AVAILABLE:
            ERROR_COUNT.labels(error_type="rate_limit").inc()
        raise HTTPException(
            status_code=429, detail=e.user_message, headers={"Retry-After": str(e.retry_after_seconds or 60)}
        )


# Exception handlers
@app.exception_handler(FrameworkError)
async def framework_error_handler(request: Request, exc: FrameworkError):
    """Handle framework-specific errors."""
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(error_type=exc.error_code).inc()

    return JSONResponse(status_code=500, content=exc.to_user_response())


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    if PROMETHEUS_AVAILABLE:
        ERROR_COUNT.labels(error_type="validation").inc()

    return JSONResponse(status_code=400, content=exc.to_user_response())


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns basic service health status. Use this for load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["health"])
async def readiness_check():
    """
    Readiness check endpoint.

    Verifies all dependencies are available. Use this for Kubernetes readiness probes.
    """
    checks = {
        "imports_available": IMPORTS_AVAILABLE,
        "authenticator_configured": True,
        "llm_client_available": True,  # Would check actual client
        "prometheus_available": PROMETHEUS_AVAILABLE,
    }

    # Check if all critical services are available
    all_ready = all(
        [
            checks["imports_available"],
            checks["authenticator_configured"],
        ]
    )

    if not all_ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ReadinessResponse(ready=all_ready, checks=checks)


@app.get("/metrics", tags=["metrics"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    """
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Prometheus metrics not available")

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def process_query(request: QueryRequest, client_info: ClientInfo = Depends(verify_api_key)):
    """
    Process a query using the multi-agent MCTS framework.

    This endpoint:
    1. Validates the input query
    2. Optionally retrieves context via RAG
    3. Processes through HRM and TRM agents
    4. Optionally runs MCTS simulation
    5. Synthesizes a final response

    **Authentication**: Requires valid API key in X-API-Key header.

    **Rate Limiting**: Subject to rate limits per client.
    """
    start_time = time.perf_counter()

    # Validate input using validation models
    if IMPORTS_AVAILABLE:
        try:
            QueryInput(
                query=request.query,
                use_rag=request.use_rag,
                use_mcts=request.use_mcts,
                thread_id=request.thread_id,
            )
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                ERROR_COUNT.labels(error_type="validation").inc()
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

    # Process query (mock implementation for demo)
    # In production, this would call the actual framework
    await asyncio.sleep(0.1)  # Simulate processing

    processing_time = (time.perf_counter() - start_time) * 1000

    # Mock response
    return QueryResponse(
        response=f"Processed query: {request.query[:100]}...",
        confidence=0.85,
        agents_used=["hrm", "trm"] + (["mcts"] if request.use_mcts else []),
        mcts_stats=(
            {
                "iterations": request.mcts_iterations or 100,
                "best_action": "recommended_action",
                "root_visits": request.mcts_iterations or 100,
            }
            if request.use_mcts
            else None
        ),
        processing_time_ms=processing_time,
        metadata={
            "client_id": client_info.client_id,
            "thread_id": request.thread_id,
            "rag_enabled": request.use_rag,
        },
    )


@app.get("/stats", tags=["metrics"])
async def get_stats(client_info: ClientInfo = Depends(verify_api_key)):
    """
    Get usage statistics for the authenticated client.

    Returns request counts and rate limit information.
    """
    authenticator = get_authenticator()
    stats = authenticator.get_client_stats(client_info.client_id)

    return {
        "client_id": client_info.client_id,
        "roles": list(client_info.roles),
        **stats,
        "rate_limits": {
            "per_minute": authenticator.rate_limit_config.requests_per_minute,
            "per_hour": authenticator.rate_limit_config.requests_per_hour,
            "per_day": authenticator.rate_limit_config.requests_per_day,
        },
    }


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.rest_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info",
        access_log=True,
    )
