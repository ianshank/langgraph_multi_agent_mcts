# Observability Module
"""
Comprehensive observability infrastructure for multi-agent MCTS framework.

Includes:
- JSON structured logging with correlation IDs
- OpenTelemetry tracing with automatic span creation
- Metrics collection for MCTS and agent performance
- Debug utilities for MCTS tree visualization
- Performance profiling tools
"""

from .logging import setup_logging, get_logger, CorrelationIdFilter
from .tracing import TracingManager, trace_operation, get_tracer
from .metrics import MetricsCollector, mcts_metrics, agent_metrics
from .debug import MCTSDebugger, export_tree_to_dot, visualize_mcts_tree
from .profiling import profile_block, AsyncProfiler, MemoryProfiler, generate_performance_report

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "CorrelationIdFilter",
    # Tracing
    "TracingManager",
    "trace_operation",
    "get_tracer",
    # Metrics
    "MetricsCollector",
    "mcts_metrics",
    "agent_metrics",
    # Debug
    "MCTSDebugger",
    "export_tree_to_dot",
    "visualize_mcts_tree",
    # Profiling
    "profile_block",
    "AsyncProfiler",
    "MemoryProfiler",
    "generate_performance_report",
]
