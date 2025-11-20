"""
Quick Start Example for Advanced Embeddings

Demonstrates basic usage of the advanced embedding system.
"""

import logging
import os

from training.advanced_embeddings import (
    CohereEmbedder,
    EmbedderFactory,
    EnsembleEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    VoyageEmbedder,
)
from training.embedding_integration import create_rag_index_builder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example 1: Basic embedding usage."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 1: Basic Embedding Usage")
    logger.info("=" * 80)

    test_texts = [
        "Cybersecurity threats are evolving rapidly.",
        "The MITRE ATT&CK framework helps classify attack techniques.",
        "Machine learning models require careful validation.",
    ]

    # Try Voyage AI (if API key available)
    if os.environ.get("VOYAGE_API_KEY"):
        logger.info("\nTesting Voyage AI...")
        config = {
            "model": "voyage-large-2-instruct",
            "dimension": 1024,
            "batch_size": 32,
            "cache_enabled": True,
            "cache_dir": "./cache/embeddings",
        }
        embedder = VoyageEmbedder(config)

        if embedder.is_available():
            result = embedder.embed_with_cache(test_texts)
            logger.info(f"  Embeddings shape: {result.embeddings.shape}")
            logger.info(f"  Dimension: {result.dimension}")
            logger.info(f"  Latency: {result.latency_ms:.2f}ms")
            logger.info(f"  Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")

            # Second call should hit cache
            result2 = embedder.embed_with_cache(test_texts)
            logger.info(f"  Second call cache hit rate: {result2.metadata['cache_hit_rate']:.2%}")
        else:
            logger.warning("  Voyage AI not available")
    else:
        logger.info("\nVoyage AI: Set VOYAGE_API_KEY to test")

    # Try Cohere (if API key available)
    if os.environ.get("COHERE_API_KEY"):
        logger.info("\nTesting Cohere...")
        config = {
            "model": "embed-english-v3.0",
            "dimension": 1024,
            "batch_size": 32,
            "cache_enabled": True,
            "cache_dir": "./cache/embeddings",
        }
        embedder = CohereEmbedder(config)

        if embedder.is_available():
            result = embedder.embed_with_cache(test_texts)
            logger.info(f"  Embeddings shape: {result.embeddings.shape}")
            logger.info(f"  Latency: {result.latency_ms:.2f}ms")
        else:
            logger.warning("  Cohere not available")
    else:
        logger.info("\nCohere: Set COHERE_API_KEY to test")

    # Sentence-Transformers (always available)
    logger.info("\nTesting Sentence-Transformers (fallback)...")
    config = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "cache_enabled": True,
        "cache_dir": "./cache/embeddings",
    }
    embedder = SentenceTransformerEmbedder(config)

    if embedder.is_available():
        result = embedder.embed_with_cache(test_texts)
        logger.info(f"  Embeddings shape: {result.embeddings.shape}")
        logger.info(f"  Latency: {result.latency_ms:.2f}ms")
    else:
        logger.warning("  Sentence-Transformers not available")


def example_factory_with_fallback():
    """Example 2: Using factory with automatic fallback."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Factory with Automatic Fallback")
    logger.info("=" * 80)

    # Define preference order
    configs = [
        {"model": "voyage-large-2-instruct", "provider": "voyage", "dimension": 1024},
        {"model": "embed-english-v3.0", "provider": "cohere", "dimension": 1024},
        {"model": "text-embedding-3-large", "provider": "openai", "dimension": 1024},
        {"model": "BAAI/bge-large-en-v1.5", "provider": "huggingface"},
        {"model": "sentence-transformers/all-MiniLM-L6-v2", "provider": "huggingface"},
    ]

    # Add common config
    for config in configs:
        config["batch_size"] = 32
        config["cache_enabled"] = True
        config["cache_dir"] = "./cache/embeddings"

    embedder = EmbedderFactory.create_with_fallback(configs)

    logger.info(f"\nSelected embedder: {embedder.model_name}")
    logger.info(f"Dimension: {embedder.dimension}")
    logger.info(f"Available: {embedder.is_available()}")

    # Test embedding
    test_texts = ["Test text for fallback mechanism."]
    result = embedder.embed_with_cache(test_texts)
    logger.info(f"Embeddings shape: {result.embeddings.shape}")


def example_ensemble():
    """Example 3: Ensemble embeddings."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Ensemble Embeddings")
    logger.info("=" * 80)

    # Create multiple embedders
    embedders = []

    # Add available embedders
    if os.environ.get("VOYAGE_API_KEY"):
        voyage = VoyageEmbedder({"model": "voyage-large-2-instruct", "dimension": 1024})
        if voyage.is_available():
            embedders.append(voyage)
            logger.info("  Added Voyage to ensemble")

    if os.environ.get("COHERE_API_KEY"):
        cohere = CohereEmbedder({"model": "embed-english-v3.0", "dimension": 1024})
        if cohere.is_available():
            embedders.append(cohere)
            logger.info("  Added Cohere to ensemble")

    # Always add sentence-transformers
    st = SentenceTransformerEmbedder({"model": "sentence-transformers/all-MiniLM-L6-v2"})
    if st.is_available():
        embedders.append(st)
        logger.info("  Added Sentence-Transformers to ensemble")

    if len(embedders) < 2:
        logger.warning("Need at least 2 embedders for ensemble. Skipping.")
        return

    # Create ensemble
    for method in ["mean", "concat"]:
        logger.info(f"\nTesting {method} ensemble...")
        config = {"combination_method": method}
        ensemble = EnsembleEmbedder(config, embedders[:2])  # Use first 2

        test_texts = ["Ensemble embedding test."]
        embeddings = ensemble.embed(test_texts)
        logger.info(f"  Result shape: {embeddings.shape}")
        logger.info(f"  Combined dimension: {ensemble.dimension}")


def example_rag_integration():
    """Example 4: Integration with RAG system."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: RAG System Integration")
    logger.info("=" * 80)

    try:
        # Create builder with advanced embeddings
        builder = create_rag_index_builder(config_path="training/config.yaml", use_advanced=True)

        # Get embedder info
        info = builder.get_embedder_info()
        logger.info("\nRAG Index Builder Info:")
        logger.info(f"  Type: {info['type']}")
        logger.info(f"  Model: {info['model']}")
        logger.info(f"  Dimension: {info['dimension']}")

        # Test batch embedding
        test_texts = [
            "Cybersecurity document 1",
            "Threat intelligence report",
            "Network security analysis",
        ]

        embeddings = builder.batch_embed_texts(test_texts)
        logger.info(f"\nBatch embedded {len(test_texts)} texts")
        logger.info(f"  Result shape: {embeddings.shape}")

    except Exception as e:
        logger.error(f"RAG integration error: {e}")


def example_matryoshka():
    """Example 5: Matryoshka embeddings (flexible dimensions)."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Matryoshka Embeddings")
    logger.info("=" * 80)

    test_texts = ["Test text for Matryoshka embeddings."]

    # Cohere supports Matryoshka
    if os.environ.get("COHERE_API_KEY"):
        logger.info("\nCohere Matryoshka:")
        for dim in [1024, 512, 256, 128]:
            config = {
                "model": "embed-english-v3.0",
                "dimension": dim,
                "cache_enabled": True,
                "cache_dir": "./cache/embeddings",
            }
            embedder = CohereEmbedder(config)

            if embedder.is_available():
                embeddings = embedder.embed(test_texts)
                logger.info(f"  Dimension {dim}: {embeddings.shape}")
    else:
        logger.info("\nCohere: Set COHERE_API_KEY to test Matryoshka")

    # OpenAI supports dimension reduction
    if os.environ.get("OPENAI_API_KEY"):
        logger.info("\nOpenAI dimension reduction:")
        for dim in [3072, 1024, 512]:
            config = {
                "model": "text-embedding-3-large",
                "dimension": dim,
                "cache_enabled": True,
                "cache_dir": "./cache/embeddings",
            }
            embedder = OpenAIEmbedder(config)

            if embedder.is_available():
                embeddings = embedder.embed(test_texts)
                logger.info(f"  Dimension {dim}: {embeddings.shape}")
    else:
        logger.info("\nOpenAI: Set OPENAI_API_KEY to test dimension reduction")


def example_performance_comparison():
    """Example 6: Compare performance of different embedders."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: Performance Comparison")
    logger.info("=" * 80)

    # Generate test texts
    test_texts = [f"Test document number {i} about cybersecurity." for i in range(10)]

    embedders_to_test = []

    # Add available embedders
    if os.environ.get("VOYAGE_API_KEY"):
        embedders_to_test.append(("Voyage", VoyageEmbedder({"model": "voyage-large-2-instruct"})))

    if os.environ.get("COHERE_API_KEY"):
        embedders_to_test.append(("Cohere", CohereEmbedder({"model": "embed-english-v3.0"})))

    if os.environ.get("OPENAI_API_KEY"):
        embedders_to_test.append(("OpenAI", OpenAIEmbedder({"model": "text-embedding-3-small"})))

    embedders_to_test.append(
        ("Sentence-T", SentenceTransformerEmbedder({"model": "sentence-transformers/all-MiniLM-L6-v2"}))
    )

    logger.info(f"\nComparing {len(embedders_to_test)} embedders on {len(test_texts)} texts:")
    logger.info(f"{'Embedder':<15} {'Dimension':<10} {'Latency':<12} {'Cache Hit':<12}")
    logger.info("-" * 50)

    for name, embedder in embedders_to_test:
        if not embedder.is_available():
            logger.info(f"{name:<15} {'N/A':<10} {'Not available'}")
            continue

        result = embedder.embed_with_cache(test_texts)
        logger.info(
            f"{name:<15} "
            f"{result.dimension:<10} "
            f"{result.latency_ms:<12.2f}ms "
            f"{result.metadata['cache_hit_rate']:<12.2%}"
        )


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED EMBEDDINGS - QUICK START EXAMPLES")
    logger.info("=" * 80)

    # Check API keys
    logger.info("\nAPI Key Status:")
    logger.info(f"  VOYAGE_API_KEY: {'✓' if os.environ.get('VOYAGE_API_KEY') else '✗'}")
    logger.info(f"  COHERE_API_KEY: {'✓' if os.environ.get('COHERE_API_KEY') else '✗'}")
    logger.info(f"  OPENAI_API_KEY: {'✓' if os.environ.get('OPENAI_API_KEY') else '✗'}")
    logger.info("\n(Set API keys as environment variables to enable all features)")

    # Run examples
    try:
        example_basic_usage()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")

    try:
        example_factory_with_fallback()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}")

    try:
        example_ensemble()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}")

    try:
        example_rag_integration()
    except Exception as e:
        logger.error(f"Example 4 failed: {e}")

    try:
        example_matryoshka()
    except Exception as e:
        logger.error(f"Example 5 failed: {e}")

    try:
        example_performance_comparison()
    except Exception as e:
        logger.error(f"Example 6 failed: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLES COMPLETE")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Set API keys for Voyage, Cohere, or OpenAI")
    logger.info("2. Run benchmarks: python training/embedding_benchmark.py --synthetic")
    logger.info("3. Integrate with RAG: See training/ADVANCED_EMBEDDINGS.md")
    logger.info("4. Migrate existing data: python training/migrate_embeddings.py --help")


if __name__ == "__main__":
    main()
