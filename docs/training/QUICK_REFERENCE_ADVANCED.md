# Quick Reference: Advanced Features
## Modules 8-10 Command Cheat Sheet

**Last Updated:** 2024-11-20

---

## Table of Contents

1. [Advanced Embeddings](#advanced-embeddings)
2. [Hybrid Retrieval](#hybrid-retrieval)
3. [Research Corpus Building](#research-corpus-building)
4. [Synthetic Data Generation](#synthetic-data-generation)
5. [Self-Play Training](#self-play-training)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Common Configurations](#common-configurations)

---

## Advanced Embeddings

### Initialize Multi-Model Embedder with Fallback

```python
from training.advanced_embeddings import EmbedderFactory
import os

configs = [
    {"model": "voyage-large-2-instruct", "dimension": 1024, "cache_enabled": True},
    {"model": "embed-english-v3.0", "provider": "cohere", "dimension": 512},
    {"model": "BAAI/bge-large-en-v1.5"}  # Local fallback
]
embedder = EmbedderFactory.create_with_fallback(configs)
```

### Embed with Caching

```python
# First call - cache miss
result = embedder.embed_with_cache(documents)
print(f"Latency: {result.latency_ms:.0f}ms")

# Second call - cache hit
result = embedder.embed_with_cache(documents)  # Much faster!
print(f"Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")
```

### Matryoshka Dimension Reduction

```python
# Full 1024-dim
full_config = {"model": "embed-english-v3.0", "dimension": 1024}

# Reduced 256-dim (75% storage savings)
reduced_config = {"model": "embed-english-v3.0", "dimension": 256}
```

### Ensemble Embeddings

```python
from training.advanced_embeddings import EnsembleEmbedder

# Concatenate (2048-dim)
ensemble = EnsembleEmbedder(
    {"combination_method": "concat"},
    embedders=[voyage, cohere]
)

# Weighted average (1024-dim)
ensemble = EnsembleEmbedder(
    {"combination_method": "weighted", "weights": [0.6, 0.4]},
    embedders=[voyage, cohere]
)
```

---

## Hybrid Retrieval

### Basic Hybrid Retriever

```python
from rank_bm25 import BM25Okapi

# Build indices
tokenized = [doc["text"].lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized)

# Dense embeddings
dense_vectors = embedder.embed([doc["text"] for doc in documents])

# Hybrid search
def hybrid_search(query, alpha=0.5):
    # Dense scores (cosine similarity)
    query_emb = embedder.embed([query])[0]
    dense_scores = cosine_similarity(query_emb, dense_vectors)

    # Sparse scores (BM25)
    sparse_scores = bm25.get_scores(query.lower().split())

    # Normalize and combine
    dense_norm = normalize(dense_scores)
    sparse_norm = normalize(sparse_scores)
    hybrid = alpha * dense_norm + (1 - alpha) * sparse_norm

    return hybrid
```

### Cross-Encoder Re-Ranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Re-rank top-100 to top-10
candidates = retriever.retrieve(query, top_k=100)
pairs = [[query, doc["text"]] for doc in candidates]
scores = reranker.predict(pairs)

# Sort by re-rank score
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:10]
```

### Evaluate Retrieval Quality

```python
from training.benchmark_suite import RetrievalMetrics

metrics = RetrievalMetrics()
result = RetrievalResult(
    query=query,
    retrieved_docs=retrieved_ids,
    ground_truth_relevant=relevant_ids,
)

ndcg = metrics.ndcg_at_k(result, k=10)
recall = metrics.recall_at_k(result, k=100)
mrr = metrics.mean_reciprocal_rank(result)

print(f"nDCG@10: {ndcg:.3f}")
print(f"Recall@100: {recall:.3f}")
print(f"MRR: {mrr:.3f}")
```

---

## Research Corpus Building

### Fetch Papers from arXiv

```python
from training.research_corpus_builder import ResearchCorpusBuilder

config = {
    "categories": ["cs.AI", "cs.LG", "cs.CL"],
    "keywords": ["MCTS", "AlphaZero", "reinforcement learning"],
    "date_start": "2023-01-01",
    "date_end": "2024-12-31",
    "max_results": 100,
    "cache_dir": "./cache/research",
    "chunk_size": 512,
    "rate_limit_delay": 3.0,
}

builder = ResearchCorpusBuilder(config)

# Build corpus
for chunk in builder.build_corpus(mode="keywords", max_papers=100):
    print(f"Processed: {chunk.doc_id}")

# Get statistics
stats = builder.get_statistics()
print(f"Papers: {stats.total_papers_processed}")
print(f"Chunks: {stats.total_chunks_created}")
```

### Export Metadata

```python
from pathlib import Path

# Export paper metadata
builder.export_metadata(Path("./output/papers.json"))
```

---

## Synthetic Data Generation

### Generate Q&A Pairs

```python
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator
from src.adapters.llm import create_client

# Initialize
llm_client = create_client("openai", model="gpt-3.5-turbo")
generator = SyntheticKnowledgeGenerator(llm_client, output_dir="./synthetic_data")

# Generate pairs
pairs = await generator.generate_batch(
    num_samples=1000,
    categories=["mcts_algorithms", "langgraph_workflows"],
    batch_size=10
)

# Filter by quality
high_quality = generator.filter_by_quality(pairs, min_score=0.6)

# Save dataset
generator.save_dataset(high_quality, "qa_dataset.json", format="langsmith")

# Statistics
stats = generator.get_statistics()
print(f"Valid pairs: {stats['valid_pairs']}")
print(f"Avg quality: {stats['avg_quality_score']:.3f}")
print(f"Cost: ${stats['total_cost']:.2f}")
```

### Custom Question Templates

```python
custom_templates = {
    "my_category": [
        "Explain {concept} with examples",
        "How does {algorithm} handle {challenge}?",
    ]
}

generator.templates.update(custom_templates)
```

---

## Self-Play Training

### Initialize Self-Play Trainer

```python
from training.self_play_generator import SelfPlayTrainer

trainer = SelfPlayTrainer(
    hrm_agent=hrm_model,
    trm_agent=trm_model,
    config={
        "games_per_iteration": 1000,
        "batch_size": 64,
        "parallel_batch_size": 32,
        "max_buffer_size": 10000,
        "mcts": {"num_simulations": 100, "c_puct": 1.25},
    },
    device="cuda",
)
```

### Run Training Iteration

```python
# Single iteration
metrics = await trainer.iteration(iteration_num=0)

print(f"Episodes: {metrics['num_episodes']}")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Policy examples: {metrics['num_policy_examples']}")
print(f"Eval success: {metrics['eval_success_rate']:.2%}")

# Multi-iteration training
for i in range(10):
    metrics = await trainer.iteration(i)
    print(f"Iteration {i}: Success {metrics['success_rate']:.2%}")
```

### Generate Episodes

```python
from training.self_play_generator import (
    SelfPlayEpisodeGenerator,
    MathProblemGenerator
)

# Generate tasks
task_gen = MathProblemGenerator(difficulty_range=(0.1, 1.0))
tasks = task_gen.generate(num_tasks=100)

# Generate episodes
episode_gen = SelfPlayEpisodeGenerator(
    mcts_config={"num_simulations": 100}
)

episodes = []
for task in tasks:
    episode = await episode_gen.generate_episode(task, max_steps=50)
    episodes.append(episode)

# Analyze
success_count = sum(1 for ep in episodes if ep.outcome == "success")
print(f"Success rate: {success_count / len(episodes):.2%}")
```

### Extract Training Data

```python
from training.self_play_generator import TrainingDataExtractor

extractor = TrainingDataExtractor()
training_data = extractor.extract_examples(episodes)

print(f"Policy examples: {len(training_data['policy'])}")
print(f"Value examples: {len(training_data['value'])}")
print(f"Reasoning examples: {len(training_data['reasoning'])}")
```

---

## Monitoring & Metrics

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
query_counter = Counter('rag_queries_total', 'Total queries', ['status'])
query_counter.labels(status='success').inc()

# Histograms
latency = Histogram('query_latency_seconds', 'Query latency')
with latency.time():
    result = rag_pipeline(query)

# Gauges
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate')
cache_hit_rate.set(embedder.get_cache_stats()['hit_rate'])
```

### A/B Testing

```python
from training.continual_learning import ABTestFramework

ab = ABTestFramework({
    "traffic_split": 0.1,
    "min_samples": 1000,
    "confidence_level": 0.95
})

# Create test
test_id = ab.create_test("model_v2", model_a, model_b, metric_fn)

# Assign traffic
group = ab.assign_group(test_id, request_id)

# Record results
ab.record_result(test_id, group, input_data, output, success_metric)

# Check status
status = ab.get_test_status(test_id)
print(f"Status: {status['status']}")
print(f"Recommendation: {status['result']['recommendation']}")
```

### Drift Detection

```python
from training.continual_learning import DriftDetector
import numpy as np

detector = DriftDetector({
    "window_size": 1000,
    "threshold": 0.1,
    "detection_method": "kolmogorov_smirnov"
})

# Set baseline
reference = np.random.randn(1000, 5)
detector.set_reference_distribution(reference)

# Monitor production data
for sample in production_samples:
    drift_report = detector.add_sample(sample)
    if drift_report:
        print(f"Drift detected: {drift_report.severity:.3f}")
```

---

## Common Configurations

### Production RAG Config

```yaml
# config/production.yaml
embeddings:
  model: "voyage-large-2-instruct"
  dimension: 1024
  cache_enabled: true
  cache_dir: "./cache/embeddings"

retrieval:
  hybrid:
    alpha: 0.5
    dense_top_k: 100
    sparse_top_k: 100
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: 10

generation:
  model: "gpt-4"
  max_tokens: 500
  temperature: 0.7

monitoring:
  prometheus:
    enabled: true
    port: 9090
  logging:
    level: "INFO"
    file: "logs/rag.log"
```

### Self-Play Training Config

```yaml
# config/self_play.yaml
training:
  games_per_iteration: 1000
  batch_size: 64
  parallel_batch_size: 32
  max_buffer_size: 10000
  num_iterations: 100

mcts:
  num_simulations: 100
  c_puct: 1.25

tasks:
  generators:
    - type: "math"
      difficulty_range: [0.1, 1.0]
      weight: 0.4
    - type: "code"
      difficulty_range: [0.3, 0.8]
      weight: 0.3
    - type: "reasoning"
      difficulty_range: [0.2, 0.9]
      weight: 0.3

checkpointing:
  enabled: true
  directory: "./checkpoints"
  save_frequency: 5  # iterations
```

### Research Corpus Config

```yaml
# config/research_corpus.yaml
arxiv:
  categories:
    - "cs.AI"
    - "cs.LG"
    - "cs.CL"
  keywords:
    - "MCTS"
    - "AlphaZero"
    - "multi-agent"
    - "LLM reasoning"
  date_start: "2020-01-01"
  date_end: "2024-12-31"
  max_results: 1000
  rate_limit_delay: 3.0

processing:
  chunk_size: 512
  chunk_overlap: 50
  cache_dir: "./cache/research"
  skip_cached: true

embedding:
  model: "voyage-large-2-instruct"
  dimension: 1024
  batch_size: 100
```

---

## Command Line Tools

### Run Research Corpus Builder

```bash
# Build corpus
python -m training.research_corpus_builder \
  --config config/research_corpus.yaml \
  --output ./output/research_corpus \
  --max-papers 100

# With specific categories
python -m training.research_corpus_builder \
  --categories cs.AI cs.LG \
  --keywords "MCTS" "AlphaZero" \
  --date-start 2023-01-01 \
  --max-results 100
```

### Generate Synthetic Data

```bash
# Generate Q&A pairs
python -m training.synthetic_knowledge_generator \
  --num-samples 1000 \
  --provider openai \
  --model gpt-3.5-turbo \
  --output-dir ./synthetic_data \
  --min-quality 0.6

# With specific categories
python -m training.synthetic_knowledge_generator \
  --num-samples 1000 \
  --categories mcts_algorithms langgraph_workflows \
  --batch-size 10
```

### Run Self-Play Training

```bash
# Start training
python -m training.self_play_generator \
  --config config/self_play.yaml \
  --checkpoint-dir ./checkpoints \
  --num-iterations 100

# Resume from checkpoint
python -m training.self_play_generator \
  --config config/self_play.yaml \
  --resume ./checkpoints/iteration_50.pt
```

### Benchmark Embeddings

```bash
# Compare embedding models
python -m training.embedding_benchmark \
  --models voyage-large-2 cohere-v3 openai-3-large \
  --test-queries data/test_queries.json \
  --output ./benchmarks/results.json
```

---

## Environment Variables

### Required Keys

```bash
# LLM Providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Embedding Providers
export VOYAGE_API_KEY="pa-..."
export COHERE_API_KEY="..."

# Vector Database
export PINECONE_API_KEY="..."
export PINECONE_ENVIRONMENT="us-west1-gcp"

# Monitoring
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="advanced-training"
export WANDB_API_KEY="..."
```

### Load from .env

```python
from dotenv import load_dotenv
load_dotenv()

# Or specify file
load_dotenv(".env.production")
```

---

## Performance Tips

### Optimization Checklist

- [ ] Enable embedding cache (target: >80% hit rate)
- [ ] Use Matryoshka dimensions for storage savings
- [ ] Implement cross-encoder re-ranking
- [ ] Tune hybrid search alpha parameter
- [ ] Profile latency bottlenecks
- [ ] Use async/parallel processing
- [ ] Monitor memory usage
- [ ] Set up Prometheus metrics
- [ ] Configure A/B testing
- [ ] Enable drift detection

### Latency Targets

| Component | Target p95 |
|-----------|------------|
| Embedding | <100ms |
| Vector search | <50ms |
| BM25 search | <20ms |
| Re-ranking (10 docs) | <200ms |
| LLM generation | <2000ms |
| **Total pipeline** | **<500ms** |

### Cost Optimization

| Strategy | Savings |
|----------|---------|
| Embedding cache | 80-95% |
| Matryoshka (512 dim) | 50% storage |
| Two-stage synthetic gen | 60-80% |
| Batch API requests | 20-30% |

---

## Quick Troubleshooting

### Issue: Low quality scores
→ Use better model (GPT-4 vs GPT-3.5-turbo)
→ Improve prompts with examples
→ Generate more, filter aggressively

### Issue: High latency
→ Profile each pipeline stage
→ Enable caching
→ Use async/parallel processing
→ Reduce batch sizes

### Issue: API rate limits
→ Add exponential backoff
→ Enable caching
→ Reduce request rate

### Issue: Out of memory
→ Limit buffer sizes
→ Use smaller embedding dimensions
→ Clear caches periodically
→ Enable garbage collection

### Issue: Training not improving
→ Check learning rate
→ Verify training data diversity
→ Add exploration bonus
→ Use curriculum learning

---

## Additional Resources

- **Full Documentation:** [Module 8](MODULE_8_ADVANCED_RAG.md), [Module 9](MODULE_9_KNOWLEDGE_ENGINEERING.md), [Module 10](MODULE_10_SELF_IMPROVEMENT.md)
- **Troubleshooting:** [TROUBLESHOOTING_ADVANCED.md](TROUBLESHOOTING_ADVANCED.md)
- **Assessment:** [ASSESSMENT_ADVANCED.md](ASSESSMENT_ADVANCED.md)
- **Lab Exercises:** [LAB_EXERCISES_ADVANCED.md](LAB_EXERCISES_ADVANCED.md)
- **Code Examples:** `/labs/module_8/`, `/labs/module_9/`, `/labs/module_10/`

---

**Quick Reference v1.0** | Last Updated: 2024-11-20
