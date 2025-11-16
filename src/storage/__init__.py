# Storage Module
"""
Storage infrastructure for multi-agent MCTS framework.

Includes:
- Async S3 client with retry strategies
- Content-hash based idempotent keys
- Compression support
- Pinecone vector storage for agent selection history
"""

from .s3_client import S3Config, S3StorageClient

# Pinecone integration (optional)
try:
    from .pinecone_store import (  # noqa: F401
        PINECONE_AVAILABLE,
        PineconeVectorStore,
    )

    _pinecone_exports = [
        "PineconeVectorStore",
        "PINECONE_AVAILABLE",
    ]
except ImportError:
    _pinecone_exports = []

__all__ = [
    "S3StorageClient",
    "S3Config",
] + _pinecone_exports
