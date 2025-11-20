"""
Monitoring and observability module for LangGraph Multi-Agent MCTS Framework.

This module provides:
- Prometheus metrics collection
- OpenTelemetry distributed tracing
- Custom metrics for agents and MCTS operations
- Performance monitoring
"""

from .prometheus_metrics import (
    ACTIVE_OPERATIONS,
    AGENT_CONFIDENCE_SCORES,
    AGENT_REQUEST_LATENCY,
    AGENT_REQUESTS_TOTAL,
    LLM_REQUEST_ERRORS,
    MCTS_ITERATION_LATENCY,
    MCTS_ITERATIONS_TOTAL,
    setup_metrics,
)

try:
    from .otel_tracing import setup_tracing, trace_operation

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def setup_tracing(*args, **kwargs):
        """Dummy function when tracing is not available."""
        pass

    def trace_operation(*args, **kwargs):
        """Dummy decorator when tracing is not available."""

        def decorator(func):
            return func

        return decorator


__all__ = [
    "setup_metrics",
    "setup_tracing",
    "trace_operation",
    "AGENT_REQUESTS_TOTAL",
    "AGENT_REQUEST_LATENCY",
    "AGENT_CONFIDENCE_SCORES",
    "MCTS_ITERATIONS_TOTAL",
    "MCTS_ITERATION_LATENCY",
    "ACTIVE_OPERATIONS",
    "LLM_REQUEST_ERRORS",
    "TRACING_AVAILABLE",
]
