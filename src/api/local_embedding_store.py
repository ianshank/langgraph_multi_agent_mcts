"""
Local Embedding Store for RAG Fallback.

Provides local document storage and similarity search using sentence-transformers.
This enables RAG functionality even when cloud vector stores (Pinecone) are unavailable.

Best Practices 2025:
- No hardcoded model names (configurable)
- Graceful degradation without dependencies
- Efficient numpy-based similarity search
- Thread-safe document management
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import numpy - required for this module
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


@dataclass
class LocalDocument:
    """Document stored in local index."""

    id: str
    content: str
    embedding: Any | None = None  # np.ndarray when available
    metadata: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash based on document ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on document ID."""
        if not isinstance(other, LocalDocument):
            return False
        return self.id == other.id


class LocalEmbeddingStore:
    """
    Local embedding store using sentence-transformers and numpy.

    Provides:
    - Document embedding with sentence-transformers
    - Similarity search with cosine similarity
    - In-memory storage with optional persistence
    - Thread-safe operations

    Usage:
        store = LocalEmbeddingStore()
        if store.initialize():
            store.add_documents([{"content": "Hello world", "metadata": {"source": "test"}}])
            results = store.search("Hello", top_k=5)
    """

    # Default model for embeddings - small and fast
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # Embedding dimension for default model
    DEFAULT_EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        embedding_dim: int | None = None,
    ):
        """
        Initialize local embedding store.

        Args:
            model_name: Sentence transformer model name (defaults to all-MiniLM-L6-v2)
            cache_dir: Optional directory for caching models
            embedding_dim: Embedding dimension (auto-detected from model if not specified)
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._cache_dir = cache_dir
        self._embedding_dim = embedding_dim or self.DEFAULT_EMBEDDING_DIM
        self._model: Any | None = None
        self._documents: list[LocalDocument] = []
        self._embeddings: Any | None = None  # np.ndarray
        self._document_ids: set[str] = set()
        self._is_initialized = False
        self._lock = threading.RLock()

    def initialize(self) -> bool:
        """
        Initialize the embedding model.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._is_initialized:
            return True

        if not _HAS_NUMPY:
            logger.warning("numpy not installed, local embedding store unavailable")
            return False

        if not _HAS_SENTENCE_TRANSFORMERS:
            logger.warning("sentence-transformers not installed, local embedding store unavailable")
            return False

        with self._lock:
            try:
                self._model = SentenceTransformer(
                    self._model_name,
                    cache_folder=str(self._cache_dir) if self._cache_dir else None,
                )

                # Auto-detect embedding dimension
                test_embedding = self._model.encode("test", convert_to_numpy=True)
                self._embedding_dim = test_embedding.shape[0]

                self._is_initialized = True
                logger.info(
                    f"Initialized local embedding model: {self._model_name} "
                    f"(dim={self._embedding_dim})"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                return False

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """
        Add documents to the store.

        Args:
            documents: List of dicts with 'content' and optional 'metadata', 'id'
            batch_size: Batch size for embedding generation

        Returns:
            Number of documents added
        """
        if not self._is_initialized:
            if not self.initialize():
                logger.warning("Cannot add documents - store not initialized")
                return 0

        with self._lock:
            added = 0
            new_docs: list[LocalDocument] = []
            new_contents: list[str] = []

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                # Generate document ID if not provided
                doc_id = doc.get("id") or self._generate_doc_id(content)

                # Skip duplicates
                if doc_id in self._document_ids:
                    logger.debug(f"Skipping duplicate document: {doc_id}")
                    continue

                local_doc = LocalDocument(
                    id=doc_id,
                    content=content,
                    metadata=doc.get("metadata", {}),
                )
                new_docs.append(local_doc)
                new_contents.append(content)
                added += 1

            if not new_contents:
                return 0

            # Generate embeddings in batches
            try:
                all_embeddings = []
                for i in range(0, len(new_contents), batch_size):
                    batch = new_contents[i : i + batch_size]
                    batch_embeddings = self._model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,  # Pre-normalize for cosine similarity
                    )
                    all_embeddings.append(batch_embeddings)

                new_embeddings = np.vstack(all_embeddings)

                # Set embeddings on documents
                for doc, emb in zip(new_docs, new_embeddings):
                    doc.embedding = emb

                # Add to store
                self._documents.extend(new_docs)
                self._document_ids.update(doc.id for doc in new_docs)

                # Update embedding matrix
                if self._embeddings is None:
                    self._embeddings = new_embeddings
                else:
                    self._embeddings = np.vstack([self._embeddings, new_embeddings])

                logger.debug(f"Added {added} documents to local store (total: {len(self._documents)})")
                return added

            except Exception as e:
                logger.error(f"Failed to add documents: {e}")
                return 0

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query string
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filters (exact match)

        Returns:
            List of results with content, score, metadata, and id
        """
        if not self._is_initialized or self._embeddings is None:
            return []

        if len(self._documents) == 0:
            return []

        with self._lock:
            try:
                # Encode query (normalized for cosine similarity)
                query_embedding = self._model.encode(
                    query,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )

                # Compute cosine similarities (dot product since vectors are normalized)
                similarities = np.dot(self._embeddings, query_embedding)

                # Apply metadata filter if provided
                valid_indices = list(range(len(self._documents)))
                if filter_metadata:
                    valid_indices = [
                        i
                        for i in valid_indices
                        if self._matches_filter(self._documents[i].metadata, filter_metadata)
                    ]

                # Get top-k indices from valid documents
                if filter_metadata:
                    filtered_similarities = [(i, similarities[i]) for i in valid_indices]
                    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                    top_indices = [i for i, _ in filtered_similarities[:top_k]]
                else:
                    top_indices = np.argsort(similarities)[::-1][:top_k].tolist()

                results = []
                for idx in top_indices:
                    score = float(similarities[idx])
                    if score < min_score:
                        continue

                    doc = self._documents[idx]
                    results.append(
                        {
                            "content": doc.content,
                            "score": score,
                            "metadata": doc.metadata,
                            "id": doc.id,
                        }
                    )

                return results

            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document by ID.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False otherwise
        """
        if doc_id not in self._document_ids:
            return False

        with self._lock:
            # Find document index
            idx = None
            for i, doc in enumerate(self._documents):
                if doc.id == doc_id:
                    idx = i
                    break

            if idx is None:
                return False

            # Remove from documents list
            self._documents.pop(idx)
            self._document_ids.discard(doc_id)

            # Remove from embeddings matrix
            if self._embeddings is not None and len(self._documents) > 0:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)
            elif len(self._documents) == 0:
                self._embeddings = None

            return True

    def clear(self) -> None:
        """Clear all documents from the store."""
        with self._lock:
            self._documents.clear()
            self._document_ids.clear()
            self._embeddings = None
            logger.debug("Cleared local embedding store")

    def _generate_doc_id(self, content: str) -> str:
        """Generate a unique document ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _matches_filter(
        self,
        metadata: dict[str, Any],
        filter_metadata: dict[str, Any],
    ) -> bool:
        """Check if document metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    @property
    def is_available(self) -> bool:
        """Check if store is initialized."""
        return self._is_initialized

    @property
    def has_documents(self) -> bool:
        """Check if store has any documents."""
        return len(self._documents) > 0

    @property
    def document_count(self) -> int:
        """Get number of documents in store."""
        return len(self._documents)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


def create_local_embedding_store(
    model_name: str | None = None,
    cache_dir: str | None = None,
    auto_initialize: bool = True,
) -> LocalEmbeddingStore | None:
    """
    Factory function to create a local embedding store.

    Args:
        model_name: Sentence transformer model name
        cache_dir: Directory for caching models
        auto_initialize: Whether to initialize on creation

    Returns:
        LocalEmbeddingStore instance or None if initialization fails
    """
    if not _HAS_NUMPY or not _HAS_SENTENCE_TRANSFORMERS:
        logger.warning(
            "Cannot create local embedding store: "
            f"numpy={'yes' if _HAS_NUMPY else 'no'}, "
            f"sentence-transformers={'yes' if _HAS_SENTENCE_TRANSFORMERS else 'no'}"
        )
        return None

    store = LocalEmbeddingStore(
        model_name=model_name,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    if auto_initialize:
        if not store.initialize():
            return None

    return store


__all__ = [
    "LocalDocument",
    "LocalEmbeddingStore",
    "create_local_embedding_store",
]
