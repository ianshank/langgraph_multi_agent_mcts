"""
Integration layer for advanced embeddings with existing RAG system.

Provides drop-in replacement for VectorIndexBuilder with advanced embeddings.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.advanced_embeddings import BaseEmbedder, EmbedderFactory
from training.rag_builder import VectorIndexBuilder as BaseVectorIndexBuilder

logger = logging.getLogger(__name__)


class AdvancedVectorIndexBuilder(BaseVectorIndexBuilder):
    """
    Enhanced VectorIndexBuilder with advanced embeddings support.

    Drop-in replacement for VectorIndexBuilder that uses state-of-the-art embeddings.
    """

    def __init__(self, config: dict[str, Any], use_advanced: bool = True):
        """
        Initialize advanced vector index builder.

        Args:
            config: RAG configuration
            use_advanced: Whether to use advanced embeddings (True) or fallback to base (False)
        """
        # Initialize base class
        super().__init__(config)

        self.use_advanced = use_advanced
        self.advanced_embedder = None

        if use_advanced and "embeddings" in config:
            self._initialize_advanced_embeddings(config["embeddings"])

    def _initialize_advanced_embeddings(self, embeddings_config: dict[str, Any]) -> None:
        """
        Initialize advanced embedding system.

        Args:
            embeddings_config: Advanced embeddings configuration
        """
        logger.info("Initializing advanced embedding system...")

        # Build primary embedder config
        primary_config = {
            "model": embeddings_config.get("model"),
            "provider": embeddings_config.get("provider"),
            "dimension": embeddings_config.get("dimension", 1024),
            "batch_size": embeddings_config.get("batch_size", 32),
            "cache_dir": embeddings_config.get("cache_dir", "./cache/embeddings"),
            "cache_enabled": embeddings_config.get("cache_enabled", True),
        }

        # Add model-specific settings
        provider = embeddings_config.get("provider", "").lower()
        if provider in embeddings_config:
            primary_config.update(embeddings_config[provider])

        # Try to create primary embedder
        try:
            embedder = EmbedderFactory.create_embedder(primary_config)
            if embedder.is_available():
                self.advanced_embedder = embedder
                self.embedding_dim = embedder.dimension
                logger.info(f"Using advanced embedder: {embedder.model_name} (dim={embedder.dimension})")
                return
            else:
                logger.warning(f"Primary embedder {primary_config['model']} not available, trying fallbacks...")
        except Exception as e:
            logger.warning(f"Failed to initialize primary embedder: {e}")

        # Try fallback models
        fallback_models = embeddings_config.get("fallback_models", [])
        for fallback in fallback_models:
            try:
                fallback_config = {
                    "model": fallback["model"],
                    "provider": fallback.get("provider"),
                    "dimension": fallback["dimension"],
                    "batch_size": embeddings_config.get("batch_size", 32),
                    "cache_dir": embeddings_config.get("cache_dir", "./cache/embeddings"),
                    "cache_enabled": embeddings_config.get("cache_enabled", True),
                }

                embedder = EmbedderFactory.create_embedder(fallback_config)
                if embedder.is_available():
                    self.advanced_embedder = embedder
                    self.embedding_dim = embedder.dimension
                    logger.info(f"Using fallback embedder: {embedder.model_name} (dim={embedder.dimension})")
                    return
            except Exception as e:
                logger.warning(f"Failed to initialize fallback embedder {fallback['model']}: {e}")

        # No advanced embedder available, use base implementation
        logger.warning("All advanced embedders failed, using base sentence-transformers")
        self.use_advanced = False

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using advanced embeddings if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.use_advanced and self.advanced_embedder:
            try:
                # Use advanced embedder
                embeddings = self.advanced_embedder.embed([text])
                return embeddings[0]
            except Exception as e:
                logger.warning(f"Advanced embedding failed: {e}, falling back to base")
                self.use_advanced = False

        # Fallback to base implementation
        return super()._embed_text(text)

    def batch_embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Batch embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        if self.use_advanced and self.advanced_embedder:
            try:
                result = self.advanced_embedder.embed_with_cache(texts)
                logger.info(
                    f"Batch embedded {len(texts)} texts in {result.latency_ms:.2f}ms "
                    f"(cache hit rate: {result.metadata['cache_hit_rate']:.2%})"
                )
                return result.embeddings
            except Exception as e:
                logger.warning(f"Advanced batch embedding failed: {e}, falling back to base")
                self.use_advanced = False

        # Fallback: use base implementation one at a time
        embeddings = []
        for text in texts:
            embeddings.append(self._embed_text(text))
        return np.array(embeddings)

    def get_embedder_info(self) -> dict[str, Any]:
        """Get information about the current embedder."""
        if self.use_advanced and self.advanced_embedder:
            return {
                "type": "advanced",
                "model": self.advanced_embedder.model_name,
                "dimension": self.advanced_embedder.dimension,
                "batch_size": self.advanced_embedder.batch_size,
                "cache_enabled": self.advanced_embedder.cache_enabled,
            }
        else:
            return {
                "type": "base",
                "model": self.embedding_model_name,
                "dimension": self.embedding_dim,
            }


def create_rag_index_builder(config_path: str = "training/config.yaml", use_advanced: bool = True) -> AdvancedVectorIndexBuilder:
    """
    Factory function to create RAG index builder.

    Args:
        config_path: Path to configuration file
        use_advanced: Whether to use advanced embeddings

    Returns:
        Configured AdvancedVectorIndexBuilder
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"]
    return AdvancedVectorIndexBuilder(rag_config, use_advanced=use_advanced)


def migrate_embeddings(
    old_index_builder: BaseVectorIndexBuilder,
    new_embedder: BaseEmbedder,
    output_namespace: str | None = None,
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Migrate existing embeddings to new embedding model.

    Args:
        old_index_builder: Existing index builder with documents
        new_embedder: New embedder to use
        output_namespace: Output namespace (if None, uses same as input)
        batch_size: Batch size for re-embedding

    Returns:
        Migration statistics
    """
    logger.info("Starting embedding migration...")
    start_namespace = old_index_builder.namespace

    if output_namespace is None:
        output_namespace = f"{start_namespace}_migrated"

    # Get all chunks from old index
    chunks = old_index_builder.chunk_store
    total_chunks = len(chunks)

    if total_chunks == 0:
        logger.warning("No chunks to migrate")
        return {"status": "empty", "chunks_migrated": 0}

    logger.info(f"Migrating {total_chunks} chunks from {start_namespace} to {output_namespace}")

    # Create new index builder with new embedder
    new_config = old_index_builder.config.copy()
    new_config["namespace"] = output_namespace
    new_index_builder = AdvancedVectorIndexBuilder(new_config, use_advanced=False)
    new_index_builder.advanced_embedder = new_embedder
    new_index_builder.use_advanced = True
    new_index_builder.embedding_dim = new_embedder.dimension

    # Re-embed and upload in batches
    migrated_count = 0
    batch_vectors = []

    for i, chunk in enumerate(chunks):
        # Re-embed text
        embedding = new_embedder.embed([chunk["text"]])[0]

        # Prepare vector
        vector_id = chunk["id"]
        truncated_text = chunk["text"][:10000] if len(chunk["text"]) > 10000 else chunk["text"]

        metadata = {
            "doc_id": str(chunk["doc_id"]),
            "chunk_id": int(chunk["chunk_id"]),
            "text": truncated_text,
            "migrated": True,
            **chunk.get("metadata", {}),
        }

        batch_vectors.append({"id": vector_id, "values": embedding.tolist(), "metadata": metadata})

        # Upsert batch
        if len(batch_vectors) >= batch_size:
            new_index_builder._upsert_batch(batch_vectors)
            migrated_count += len(batch_vectors)
            batch_vectors = []
            logger.info(f"Migrated {migrated_count}/{total_chunks} chunks")

    # Upsert remaining
    if batch_vectors:
        new_index_builder._upsert_batch(batch_vectors)
        migrated_count += len(batch_vectors)

    # Update chunk store
    new_index_builder.chunk_store = chunks

    logger.info(f"Migration complete: {migrated_count} chunks migrated to {output_namespace}")

    return {
        "status": "success",
        "chunks_migrated": migrated_count,
        "old_namespace": start_namespace,
        "new_namespace": output_namespace,
        "old_model": old_index_builder.embedding_model_name,
        "new_model": new_embedder.model_name,
        "old_dimension": old_index_builder.embedding_dim,
        "new_dimension": new_embedder.dimension,
    }


def compare_embeddings_on_index(
    index_builder: BaseVectorIndexBuilder, embedders: list[BaseEmbedder], sample_size: int = 100
) -> dict[str, Any]:
    """
    Compare different embeddings on same index.

    Args:
        index_builder: Index builder with chunks
        embedders: List of embedders to compare
        sample_size: Number of chunks to sample for comparison

    Returns:
        Comparison results
    """
    import random

    chunks = index_builder.chunk_store
    if not chunks:
        return {"error": "No chunks in index"}

    # Sample chunks
    sample_chunks = random.sample(chunks, min(sample_size, len(chunks)))
    texts = [c["text"] for c in sample_chunks]

    results = {}

    for embedder in embedders:
        if not embedder.is_available():
            logger.warning(f"Embedder {embedder.model_name} not available, skipping")
            continue

        try:
            result = embedder.embed_with_cache(texts)
            results[embedder.model_name] = {
                "dimension": result.dimension,
                "latency_ms": result.latency_ms,
                "cache_hit_rate": result.metadata.get("cache_hit_rate", 0),
                "avg_latency_per_text": result.latency_ms / len(texts),
            }
            logger.info(f"{embedder.model_name}: {result.latency_ms:.2f}ms for {len(texts)} texts")
        except Exception as e:
            logger.error(f"Failed to embed with {embedder.model_name}: {e}")
            results[embedder.model_name] = {"error": str(e)}

    return {
        "sample_size": len(texts),
        "embedders": results,
    }


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Advanced Embedding Integration")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rag_config = config["rag"]

    # Test 1: Create advanced index builder
    logger.info("\n=== Test 1: Create Advanced Index Builder ===")
    builder = AdvancedVectorIndexBuilder(rag_config, use_advanced=True)
    info = builder.get_embedder_info()
    logger.info(f"Embedder info: {info}")

    # Test 2: Embed some texts
    logger.info("\n=== Test 2: Embed Test Texts ===")
    test_texts = [
        "This is a test document about cybersecurity.",
        "The MITRE ATT&CK framework provides comprehensive threat intelligence.",
        "Machine learning models require high-quality training data.",
    ]

    try:
        embeddings = builder.batch_embed_texts(test_texts)
        logger.info(f"Batch embedding result: shape={embeddings.shape}")
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")

    # Test 3: Factory function
    logger.info("\n=== Test 3: Factory Function ===")
    builder2 = create_rag_index_builder(config_path, use_advanced=True)
    info2 = builder2.get_embedder_info()
    logger.info(f"Factory-created embedder info: {info2}")

    logger.info("\nIntegration tests complete!")
