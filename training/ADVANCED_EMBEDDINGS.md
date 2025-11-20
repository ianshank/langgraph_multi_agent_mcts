# Advanced Embedding System Documentation

## Overview

The advanced embedding system provides state-of-the-art embedding models with production-ready features including caching, batching, async processing, and automatic fallbacks.

## Supported Models

### 1. Voyage AI (Top MTEB 2024)
- **Model**: `voyage-large-2-instruct`
- **Dimension**: 1024
- **Best for**: General-purpose embeddings, instruction-following
- **Setup**: `pip install voyageai` and set `VOYAGE_API_KEY`

### 2. Cohere Embed v3
- **Model**: `embed-english-v3.0`
- **Dimension**: 1024 (supports Matryoshka: 1024, 512, 256, 128)
- **Best for**: Multilingual, flexible dimensions
- **Setup**: `pip install cohere` and set `COHERE_API_KEY`

### 3. OpenAI
- **Model**: `text-embedding-3-large`
- **Dimension**: 3072 (can reduce to any dimension)
- **Best for**: High quality, flexible dimensions
- **Setup**: `pip install openai` and set `OPENAI_API_KEY`

### 4. BGE (HuggingFace)
- **Model**: `BAAI/bge-large-en-v1.5`
- **Dimension**: 1024
- **Best for**: Fine-tuning, local deployment
- **Setup**: `pip install sentence-transformers`

### 5. Sentence-Transformers (Fallback)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Best for**: Lightweight, no API required
- **Setup**: `pip install sentence-transformers`

## Installation

```bash
# Core dependencies
pip install sentence-transformers

# Optional: API-based models
pip install voyageai cohere openai

# For benchmarking
pip install numpy pyyaml
```

## Configuration

### Basic Configuration (`training/config.yaml`)

```yaml
rag:
  embeddings:
    # Primary model
    model: "voyage-large-2-instruct"
    provider: "voyage"
    dimension: 1024
    batch_size: 32
    cache_enabled: true
    cache_dir: "./cache/embeddings"

    # Fallback chain (tried in order)
    fallback_models:
      - model: "embed-english-v3.0"
        provider: "cohere"
        dimension: 1024
      - model: "BAAI/bge-large-en-v1.5"
        provider: "huggingface"
        dimension: 1024
```

### Environment Variables

```bash
export VOYAGE_API_KEY="your-voyage-key"
export COHERE_API_KEY="your-cohere-key"
export OPENAI_API_KEY="your-openai-key"
export PINECONE_API_KEY="your-pinecone-key"
```

## Usage

### 1. Basic Embedding

```python
from training.advanced_embeddings import VoyageEmbedder

config = {
    "model": "voyage-large-2-instruct",
    "dimension": 1024,
    "batch_size": 32,
    "cache_enabled": True,
    "cache_dir": "./cache/embeddings"
}

embedder = VoyageEmbedder(config)

texts = ["Document about cybersecurity", "Another document"]
result = embedder.embed_with_cache(texts)

print(f"Shape: {result.embeddings.shape}")
print(f"Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")
print(f"Latency: {result.latency_ms:.2f}ms")
```

### 2. Using Factory with Fallbacks

```python
from training.advanced_embeddings import EmbedderFactory

configs = [
    {"model": "voyage-large-2-instruct", "provider": "voyage"},
    {"model": "embed-english-v3.0", "provider": "cohere"},
    {"model": "sentence-transformers/all-MiniLM-L6-v2"}
]

embedder = EmbedderFactory.create_with_fallback(configs)
print(f"Using: {embedder.model_name}")
```

### 3. Integration with RAG System

```python
from training.embedding_integration import create_rag_index_builder

# Create builder with advanced embeddings
builder = create_rag_index_builder(
    config_path="training/config.yaml",
    use_advanced=True
)

# Build index as normal
from training.data_pipeline import PRIMUSProcessor

processor = PRIMUSProcessor(config)
stats = builder.build_index(processor.stream_documents())
print(f"Indexed {stats.total_chunks} chunks")

# Search
results = builder.search("cybersecurity threats", k=10)
```

### 4. Ensemble Embeddings

```python
from training.advanced_embeddings import EnsembleEmbedder, VoyageEmbedder, CohereEmbedder

voyage = VoyageEmbedder({"model": "voyage-large-2-instruct"})
cohere = CohereEmbedder({"model": "embed-english-v3.0"})

ensemble_config = {
    "combination_method": "mean",  # or "concat", "weighted"
    "weights": [0.6, 0.4]  # for weighted
}

ensemble = EnsembleEmbedder(ensemble_config, [voyage, cohere])
embeddings = ensemble.embed(texts)
```

### 5. Async Batch Processing

```python
import asyncio
from training.advanced_embeddings import AsyncEmbedder, VoyageEmbedder

embedder = VoyageEmbedder(config)
async_embedder = AsyncEmbedder(embedder)

async def process_batches():
    batches = [batch1, batch2, batch3]
    results = await async_embedder.embed_batch_async(batches)
    return results

results = asyncio.run(process_batches())
```

## Benchmarking

### Run Benchmark

```bash
# With synthetic dataset
python training/embedding_benchmark.py \
    --config training/config.yaml \
    --synthetic \
    --num-queries 50 \
    --num-docs 1000 \
    --output ./reports/embedding_benchmark.json

# With custom dataset
python training/embedding_benchmark.py \
    --config training/config.yaml \
    --dataset ./data/eval_dataset.json \
    --output ./reports/embedding_benchmark.json
```

### Custom Benchmark

```python
from training.embedding_benchmark import EmbeddingBenchmark

benchmark = EmbeddingBenchmark({"k_values": [1, 5, 10, 20]})

# Create synthetic dataset
benchmark.create_synthetic_dataset(
    num_queries=50,
    num_docs=1000,
    relevant_docs_per_query=5
)

# Compare embedders
configs = [
    {"model": "voyage-large-2-instruct", "provider": "voyage"},
    {"model": "embed-english-v3.0", "provider": "cohere"},
]

results = benchmark.compare_embedders(configs)
benchmark.generate_report(results, "./reports/benchmark.json")
```

### Evaluation Dataset Format

```json
{
  "queries": [
    {"id": "q1", "text": "How to detect malware?"},
    {"id": "q2", "text": "Best practices for network security"}
  ],
  "corpus": [
    {"id": "d1", "text": "Malware detection techniques..."},
    {"id": "d2", "text": "Network security guidelines..."}
  ],
  "ground_truth": {
    "q1": ["d1", "d3"],
    "q2": ["d2", "d5"]
  }
}
```

## Migration

### Migrate Existing Embeddings

```bash
# Migrate to new model
python training/migrate_embeddings.py \
    --config training/config.yaml \
    --source-namespace training \
    --target-namespace training_voyage \
    --target-model voyage-large-2-instruct \
    --batch-size 100

# Dry run first
python training/migrate_embeddings.py \
    --source-namespace training \
    --target-model voyage-large-2-instruct \
    --dry-run
```

### Programmatic Migration

```python
from training.embedding_integration import migrate_embeddings
from training.advanced_embeddings import VoyageEmbedder

# Load existing index
old_builder = create_rag_index_builder(use_advanced=False)
old_builder.load_index()

# Create new embedder
new_embedder = VoyageEmbedder({"model": "voyage-large-2-instruct"})

# Migrate
result = migrate_embeddings(
    old_index_builder=old_builder,
    new_embedder=new_embedder,
    output_namespace="training_migrated",
    batch_size=100
)

print(f"Migrated {result['chunks_migrated']} chunks")
```

## Matryoshka Embeddings

Matryoshka embeddings allow flexible dimension reduction without retraining.

### Cohere Example

```python
# Generate at different dimensions
dimensions = [1024, 512, 256, 128]

for dim in dimensions:
    config = {
        "model": "embed-english-v3.0",
        "dimension": dim
    }
    embedder = CohereEmbedder(config)
    embeddings = embedder.embed(texts)
    print(f"Dimension {dim}: {embeddings.shape}")
```

### OpenAI Example

```python
# Reduce dimension for cost/performance
config = {
    "model": "text-embedding-3-large",
    "dimension": 1024  # Down from 3072
}
embedder = OpenAIEmbedder(config)
```

## Performance Optimization

### 1. Caching

Enable caching to avoid re-embedding identical texts:

```python
config = {
    "cache_enabled": True,
    "cache_dir": "./cache/embeddings"
}
```

### 2. Batch Processing

Process multiple texts at once:

```python
# Good: Batch processing
embeddings = embedder.embed(texts)  # All at once

# Bad: One at a time
embeddings = [embedder.embed([text])[0] for text in texts]
```

### 3. Async Processing

For multiple independent batches:

```python
async def process_all():
    tasks = [async_embedder.embed_async(batch) for batch in batches]
    return await asyncio.gather(*tasks)
```

## Cost Estimation

| Model | Cost per 1M tokens | Dimension | Speed |
|-------|-------------------|-----------|-------|
| Voyage Large 2 Instruct | $0.12 | 1024 | Fast |
| Cohere Embed v3 | $0.10 | 1024 | Fast |
| OpenAI text-embedding-3-large | $0.13 | 3072 | Medium |
| BGE/Sentence-Transformers | Free | 384-1024 | Fast (local) |

## Best Practices

### 1. Choose the Right Model

- **High accuracy**: Voyage, OpenAI, Cohere
- **Cost-effective**: BGE, Sentence-Transformers
- **Multilingual**: Cohere
- **Fine-tuning**: BGE

### 2. Enable Caching

Always enable caching in production to reduce costs and latency.

### 3. Use Fallbacks

Configure fallback models to ensure service availability:

```yaml
fallback_models:
  - model: "voyage-large-2-instruct"  # Best
  - model: "embed-english-v3.0"       # Good
  - model: "sentence-transformers/all-MiniLM-L6-v2"  # Always works
```

### 4. Optimize Dimensions

Use Matryoshka embeddings to reduce storage and improve speed:

```python
# For production: balance quality and performance
dimension: 1024  # Good quality, reasonable size

# For development: faster, smaller
dimension: 512

# For maximum quality
dimension: 3072  # OpenAI only
```

### 5. Monitor Performance

Track cache hit rates, latency, and quality metrics:

```python
result = embedder.embed_with_cache(texts)
print(f"Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")
print(f"Latency: {result.latency_ms:.2f}ms")
```

## Troubleshooting

### API Key Issues

```python
import os
print(f"Voyage: {bool(os.environ.get('VOYAGE_API_KEY'))}")
print(f"Cohere: {bool(os.environ.get('COHERE_API_KEY'))}")
print(f"OpenAI: {bool(os.environ.get('OPENAI_API_KEY'))}")
```

### Model Availability

```python
embedder = VoyageEmbedder(config)
if not embedder.is_available():
    print("Voyage not available, check API key and installation")
```

### Dimension Mismatch

Ensure the dimension in config matches the model:

```python
# Voyage models
"voyage-large-2-instruct": 1024
"voyage-large-2": 1536

# Cohere (flexible via Matryoshka)
"embed-english-v3.0": 1024 (can reduce to 512, 256, 128)

# OpenAI (flexible)
"text-embedding-3-large": 3072 (can reduce to any)
```

## Examples

### Complete RAG Pipeline

```python
import yaml
from training.embedding_integration import create_rag_index_builder
from training.data_pipeline import PRIMUSProcessor

# 1. Load config
with open("training/config.yaml") as f:
    config = yaml.safe_load(f)

# 2. Create processor
processor = PRIMUSProcessor(config["data"]["primus_seed"])

# 3. Create builder with advanced embeddings
builder = create_rag_index_builder(use_advanced=True)

# 4. Build index
stats = builder.build_index(processor.stream_documents())
print(f"Built index with {stats.total_chunks} chunks")

# 5. Save index
builder.save_index()

# 6. Search
results = builder.search("MITRE ATT&CK techniques", k=10)
for result in results:
    print(f"{result.score:.3f}: {result.text[:100]}...")
```

### Benchmark Multiple Models

```python
from training.embedding_benchmark import EmbeddingBenchmark

# Create benchmark
benchmark = EmbeddingBenchmark({"k_values": [1, 5, 10]})
benchmark.create_synthetic_dataset(num_queries=100, num_docs=2000)

# Define models to compare
models = [
    {"model": "voyage-large-2-instruct", "dimension": 1024},
    {"model": "embed-english-v3.0", "provider": "cohere", "dimension": 1024},
    {"model": "text-embedding-3-large", "dimension": 1024},
    {"model": "BAAI/bge-large-en-v1.5"},
]

# Run benchmark
results = benchmark.compare_embedders(models)

# Generate report
benchmark.generate_report(results, "./reports/comparison.json")
```

## References

- [Voyage AI Documentation](https://docs.voyageai.com/)
- [Cohere Embed v3](https://docs.cohere.com/docs/embeddings)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [BGE Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
