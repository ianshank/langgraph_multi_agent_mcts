"""
Performance Optimization Module for LangGraph Multi-Agent MCTS.

Provides:
- Connection pool management for LLM clients
- Embedding cache for feature extraction
- Batched inference utilities
- Memory-efficient data structures
- Performance benchmarking utilities
"""

from __future__ import annotations

from .connection_pool import (
    ConnectionPool,
    ConnectionPoolConfig,
    create_connection_pool,
)
from .embedding_cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    create_embedding_cache,
)
from .optimization import (
    BatchProcessor,
    MemoryOptimizer,
    benchmark_iterations_per_second,
)

__all__ = [
    # Connection pooling
    "ConnectionPool",
    "ConnectionPoolConfig",
    "create_connection_pool",
    # Embedding cache
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    "create_embedding_cache",
    # Optimization utilities
    "BatchProcessor",
    "MemoryOptimizer",
    "benchmark_iterations_per_second",
]

__version__ = "1.0.0"
