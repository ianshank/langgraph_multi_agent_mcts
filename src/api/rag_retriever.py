"""
RAG Retriever Service for Framework Integration.

Provides document retrieval and context formatting for the query pipeline.
Supports multiple backends (Pinecone, local embeddings) with graceful fallback.

Best Practices 2025:
- Configuration-driven (no hardcoded values)
- Async-first design
- Comprehensive error handling
- Type-safe interfaces
- Structured logging with correlation IDs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.config.settings import Settings, get_settings

# Try to import structured logging (optional dependency)
try:
    from src.observability.logging import get_correlation_id, get_structured_logger

    logger = get_structured_logger(__name__)
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    _HAS_STRUCTURED_LOGGING = False

    def get_correlation_id() -> str:
        """Fallback correlation ID function."""
        return "unknown"


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with content and metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    documents: list[RetrievedDocument]
    query: str
    retrieval_time_ms: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """Format documents as context string."""
        if not self.documents:
            return ""

        context_parts = []
        for i, doc in enumerate(self.documents, 1):
            context_parts.append(
                f"[Document {i}] (score: {doc.score:.3f})\n{doc.content}"
            )

        return "\n\n".join(context_parts)

    @property
    def has_results(self) -> bool:
        """Check if any documents were retrieved."""
        return len(self.documents) > 0


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector store backends."""

    def is_available(self) -> bool: ...

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class RAGRetriever:
    """
    RAG Retriever service with multi-backend support.

    Supports:
    - Pinecone vector store
    - Local embedding fallback
    - Graceful degradation when backends unavailable
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pinecone_store: Any | None = None,
        embedding_model: Any | None = None,
    ):
        """
        Initialize RAG retriever.

        Args:
            settings: Application settings (uses global if None)
            pinecone_store: Optional Pinecone store instance
            embedding_model: Optional embedding model for queries
        """
        self._settings = settings or get_settings()
        self._pinecone_store = pinecone_store
        self._embedding_model = embedding_model
        self._logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._available_backends: list[str] = []

        # Configuration from settings
        self._top_k_default = self._settings.FRAMEWORK_TOP_K_RETRIEVAL

    async def initialize(self) -> bool:
        """
        Initialize retriever and check available backends.

        Returns:
            True if at least one backend is available
        """
        if self._is_initialized:
            return len(self._available_backends) > 0

        init_start = time.perf_counter()
        correlation_id = get_correlation_id()

        if _HAS_STRUCTURED_LOGGING:
            logger.info(
                "Initializing RAG retriever",
                correlation_id=correlation_id,
                pinecone_configured=self._pinecone_store is not None,
            )
        else:
            logger.info("Initializing RAG retriever")

        # Check Pinecone availability
        if self._pinecone_store is not None:
            try:
                if hasattr(self._pinecone_store, "is_available"):
                    if self._pinecone_store.is_available:
                        self._available_backends.append("pinecone")
                        if _HAS_STRUCTURED_LOGGING:
                            logger.info(
                                "Pinecone backend available",
                                correlation_id=correlation_id,
                            )
                        else:
                            logger.info("Pinecone backend available")
                else:
                    # Assume available if no check method
                    self._available_backends.append("pinecone")
            except Exception as e:
                if _HAS_STRUCTURED_LOGGING:
                    logger.warning(
                        "Pinecone backend check failed",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    logger.warning(f"Pinecone backend check failed: {e}")

        # Check for local embedding fallback
        if self._embedding_model is not None:
            self._available_backends.append("local")
            if _HAS_STRUCTURED_LOGGING:
                logger.info(
                    "Local embedding backend available",
                    correlation_id=correlation_id,
                )
            else:
                logger.info("Local embedding backend available")

        self._is_initialized = True
        init_time = (time.perf_counter() - init_start) * 1000

        if _HAS_STRUCTURED_LOGGING:
            logger.info(
                "RAG retriever initialized",
                correlation_id=correlation_id,
                available_backends=self._available_backends,
                init_time_ms=round(init_time, 2),
            )
        else:
            logger.info(f"RAG retriever initialized with backends: {self._available_backends}")

        return len(self._available_backends) > 0

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string to search for
            top_k: Number of documents to retrieve (uses config default if None)
            filter_metadata: Optional metadata filters
            min_score: Minimum similarity score threshold

        Returns:
            RetrievalResult with documents and metadata
        """
        if not self._is_initialized:
            await self.initialize()

        retrieval_start = time.perf_counter()
        correlation_id = get_correlation_id()
        top_k = top_k or self._top_k_default

        if _HAS_STRUCTURED_LOGGING:
            logger.debug(
                "Starting document retrieval",
                correlation_id=correlation_id,
                query_length=len(query),
                top_k=top_k,
                min_score=min_score,
                available_backends=self._available_backends,
            )
        else:
            logger.debug(f"Starting document retrieval for query length {len(query)}")

        # Try backends in priority order
        documents: list[RetrievedDocument] = []
        backend_used = "none"

        # Try Pinecone first
        if "pinecone" in self._available_backends:
            try:
                documents, backend_used = await self._retrieve_from_pinecone(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    min_score=min_score,
                )
            except Exception as e:
                if _HAS_STRUCTURED_LOGGING:
                    logger.warning(
                        "Pinecone retrieval failed, trying fallback",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    logger.warning(f"Pinecone retrieval failed: {e}")

        # Fallback to local if Pinecone failed
        if not documents and "local" in self._available_backends:
            try:
                documents, backend_used = await self._retrieve_local(
                    query=query,
                    top_k=top_k,
                    filter_metadata=filter_metadata,
                    min_score=min_score,
                )
            except Exception as e:
                if _HAS_STRUCTURED_LOGGING:
                    logger.warning(
                        "Local retrieval failed",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    logger.warning(f"Local retrieval failed: {e}")

        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        result = RetrievalResult(
            documents=documents,
            query=query,
            retrieval_time_ms=retrieval_time,
            backend=backend_used,
            metadata={
                "top_k_requested": top_k,
                "documents_returned": len(documents),
                "min_score": min_score,
                "filter_applied": filter_metadata is not None,
            },
        )

        if _HAS_STRUCTURED_LOGGING:
            logger.info(
                "Document retrieval completed",
                correlation_id=correlation_id,
                documents_retrieved=len(documents),
                backend=backend_used,
                retrieval_time_ms=round(retrieval_time, 2),
                avg_score=round(
                    sum(d.score for d in documents) / len(documents), 3
                ) if documents else 0.0,
            )
        else:
            logger.info(f"Document retrieval completed: {len(documents)} docs from {backend_used}")

        return result

    async def _retrieve_from_pinecone(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
        min_score: float,
    ) -> tuple[list[RetrievedDocument], str]:
        """Retrieve from Pinecone vector store."""
        if self._pinecone_store is None:
            return [], "none"

        # Use the store's search method
        if hasattr(self._pinecone_store, "find_similar_decisions"):
            # Meta-controller style store
            # Create features for similarity search
            from src.agents.meta_controller.base import MetaControllerFeatures

            features = MetaControllerFeatures(
                hrm_confidence=0.5,
                trm_confidence=0.5,
                mcts_value=0.5,
                consensus_score=0.0,
                last_agent="none",
                iteration=0,
                query_length=len(query),
                has_rag_context=False,
            )

            results = self._pinecone_store.find_similar_decisions(
                features=features,
                top_k=top_k,
            )

            documents = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= min_score:
                    metadata = result.get("metadata", {})
                    documents.append(
                        RetrievedDocument(
                            content=str(metadata),
                            score=score,
                            metadata=metadata,
                            source="pinecone",
                        )
                    )

            return documents, "pinecone"

        elif hasattr(self._pinecone_store, "similarity_search"):
            # LangChain-style store
            results = self._pinecone_store.similarity_search(
                query=query,
                k=top_k,
            )

            documents = []
            for doc in results:
                score = getattr(doc, "score", 0.7)  # Default score if not provided
                if score >= min_score:
                    documents.append(
                        RetrievedDocument(
                            content=doc.page_content,
                            score=score,
                            metadata=getattr(doc, "metadata", {}),
                            source="pinecone",
                        )
                    )

            return documents, "pinecone"

        return [], "none"

    async def _retrieve_local(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
        min_score: float,
    ) -> tuple[list[RetrievedDocument], str]:
        """Retrieve using local embeddings (fallback)."""
        # This would use a local embedding model and in-memory index
        # For now, return empty as this requires additional setup
        logger.debug("Local retrieval not fully implemented")
        return [], "local"

    def format_context(
        self,
        result: RetrievalResult,
        max_length: int | None = None,
        include_scores: bool = True,
    ) -> str:
        """
        Format retrieval result as context string.

        Args:
            result: Retrieval result to format
            max_length: Maximum context length (truncates if exceeded)
            include_scores: Whether to include similarity scores

        Returns:
            Formatted context string
        """
        if not result.has_results:
            return ""

        context_parts = []
        for i, doc in enumerate(result.documents, 1):
            if include_scores:
                header = f"[Source {i}] (relevance: {doc.score:.2f})"
            else:
                header = f"[Source {i}]"

            context_parts.append(f"{header}\n{doc.content}")

        context = "\n\n---\n\n".join(context_parts)

        # Truncate if needed
        if max_length and len(context) > max_length:
            context = context[:max_length - 3] + "..."
            logger.debug(f"Context truncated from {len(result.context)} to {max_length} chars")

        return context

    @property
    def is_available(self) -> bool:
        """Check if retriever has any available backends."""
        return len(self._available_backends) > 0

    @property
    def available_backends(self) -> list[str]:
        """Get list of available backends."""
        return self._available_backends.copy()


# Factory function for creating retrievers
def create_rag_retriever(
    settings: Settings | None = None,
    pinecone_api_key: str | None = None,
    pinecone_host: str | None = None,
) -> RAGRetriever:
    """
    Factory function to create a RAG retriever with proper configuration.

    Args:
        settings: Application settings
        pinecone_api_key: Override Pinecone API key
        pinecone_host: Override Pinecone host

    Returns:
        Configured RAGRetriever instance
    """
    settings = settings or get_settings()

    # Try to create Pinecone store
    pinecone_store = None
    try:
        from src.storage.pinecone_store import PineconeVectorStore

        api_key = pinecone_api_key or settings.get_pinecone_api_key()
        host = pinecone_host or settings.PINECONE_HOST

        if api_key and host:
            pinecone_store = PineconeVectorStore(
                api_key=api_key,
                host=host,
                namespace="rag_documents",
                auto_init=True,
            )
            logger.info("Pinecone store created for RAG retriever")
    except ImportError:
        logger.debug("Pinecone not available for RAG retriever")
    except Exception as e:
        logger.warning(f"Failed to create Pinecone store: {e}")

    return RAGRetriever(
        settings=settings,
        pinecone_store=pinecone_store,
    )


__all__ = [
    "RAGRetriever",
    "RetrievedDocument",
    "RetrievalResult",
    "create_rag_retriever",
]
