# Module 8: Advanced RAG Techniques

**Duration:** 12 hours (3 days)
**Format:** Workshop + Advanced Lab Session
**Difficulty:** Advanced
**Prerequisites:** Completed Modules 1-7, understanding of embeddings and vector databases

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement advanced embedding strategies** using state-of-the-art models
2. **Design hybrid retrieval systems** combining dense and sparse methods
3. **Optimize RAG performance** with ensemble embeddings and re-ranking
4. **Benchmark embedding models** using comprehensive metrics
5. **Deploy production-ready RAG** with caching and performance optimization

---

## Session 1: State-of-the-Art Embeddings (3 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [training/advanced_embeddings.py](../../training/advanced_embeddings.py) - Advanced embedding implementations
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Latest embedding benchmarks
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Flexible embedding dimensions

### Lecture: Modern Embedding Architectures (60 minutes)

#### Top Embedding Models (2024)

**Model Comparison:**

| Model | Dimension | MTEB Score | Use Case |
|-------|-----------|------------|----------|
| Voyage-large-2-instruct | 1024 | 69.1 | General purpose (Top MTEB 2024) |
| Cohere embed-v3 | 1024* | 68.3 | Matryoshka (flexible dims) |
| OpenAI text-embedding-3-large | 3072* | 64.6 | Good balance, supports dim reduction |
| BGE-large-en-v1.5 | 1024 | 64.2 | Fine-tuning, open source |

*Supports dimension reduction via Matryoshka embedding

#### Voyage AI Embeddings

**Why Voyage AI:**
- Ranked #1 on MTEB leaderboard (2024)
- Instruction-tuned for better query understanding
- Optimized for retrieval tasks
- Excellent for domain-specific tasks

**Implementation:**
```python
from training.advanced_embeddings import VoyageEmbedder

# Initialize Voyage embedder
config = {
    "model": "voyage-large-2-instruct",
    "dimension": 1024,
    "batch_size": 32,
    "input_type": "document",  # or "query" for queries
    "cache_enabled": True,
    "cache_dir": "./cache/embeddings"
}

embedder = VoyageEmbedder(config)

# Check availability
if embedder.is_available():
    # Embed documents
    docs = [
        "Monte Carlo Tree Search is a heuristic search algorithm",
        "MCTS uses random sampling to evaluate game states",
        "UCB1 balances exploration and exploitation in tree search"
    ]

    result = embedder.embed_with_cache(docs)

    print(f"Embeddings shape: {result.embeddings.shape}")
    print(f"Model: {result.model}")
    print(f"Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")
    print(f"Latency: {result.latency_ms:.2f}ms")
```

**Key Features:**
- Automatic caching to reduce API costs
- Batch processing for efficiency
- Support for document vs query input types
- Error handling and fallbacks

#### Cohere Embed v3 with Matryoshka

**Matryoshka Embeddings:**
- Single model supports multiple dimensions (1024, 512, 256, 128, 64)
- Truncate embeddings for lower memory/storage
- Maintains strong performance at reduced dimensions
- Ideal for resource-constrained deployments

**Implementation:**
```python
from training.advanced_embeddings import CohereEmbedder

# Full 1024-dimensional embeddings
full_config = {
    "model": "embed-english-v3.0",
    "dimension": 1024,
    "input_type": "search_document",
    "embedding_types": ["float"],
}

full_embedder = CohereEmbedder(full_config)
full_result = full_embedder.embed_with_cache(docs)

print(f"Full dimension: {full_result.dimension}")  # 1024

# Reduced 256-dimensional embeddings (4x smaller!)
reduced_config = {
    "model": "embed-english-v3.0",
    "dimension": 256,
    "input_type": "search_document",
    "embedding_types": ["float"],
}

reduced_embedder = CohereEmbedder(reduced_config)
reduced_result = reduced_embedder.embed_with_cache(docs)

print(f"Reduced dimension: {reduced_result.dimension}")  # 256
print(f"Storage savings: {(1 - 256/1024)*100:.1f}%")  # 75% smaller!
```

**Dimension Trade-offs:**

| Dimension | Storage | Search Speed | Recall@10 |
|-----------|---------|--------------|-----------|
| 1024 | 100% | Baseline | 98% |
| 512 | 50% | 1.9x faster | 96% |
| 256 | 25% | 3.8x faster | 93% |
| 128 | 12.5% | 7.5x faster | 88% |

**Use Cases for Different Dimensions:**
- 1024: Maximum accuracy, critical applications
- 512: Good balance for production
- 256: Fast retrieval, large-scale systems
- 128: Resource-constrained edge devices

#### OpenAI Embeddings with Dimension Reduction

**Implementation:**
```python
from training.advanced_embeddings import OpenAIEmbedder

# Full 3072-dimensional embeddings
full_config = {
    "model": "text-embedding-3-large",
    "dimension": 3072,  # Full dimension
}

# Reduced to 1024 dimensions (API-side reduction)
reduced_config = {
    "model": "text-embedding-3-large",
    "dimension": 1024,  # API truncates for you
}

embedder = OpenAIEmbedder(reduced_config)
result = embedder.embed_with_cache(docs)

print(f"Dimension: {result.dimension}")  # 1024
print(f"Cost savings: ~67% (fewer dimensions)")
```

**Advantages:**
- Reduction happens API-side (no local post-processing)
- Maintains quality at reduced dimensions
- Lower latency (less data transfer)
- Cost-effective for large-scale applications

### Lecture: Ensemble Embeddings (45 minutes)

#### Why Ensemble Embeddings?

**Benefits:**
1. **Robustness:** Multiple models reduce single-point failures
2. **Complementary Strengths:** Different models excel at different query types
3. **Improved Recall:** Captures diverse semantic relationships
4. **Better Coverage:** Handles domain shift and edge cases

**Ensemble Strategies:**

**1. Concatenation:**
```python
from training.advanced_embeddings import EnsembleEmbedder, VoyageEmbedder, CohereEmbedder

# Create individual embedders
voyage = VoyageEmbedder({"model": "voyage-large-2-instruct", "dimension": 1024})
cohere = CohereEmbedder({"model": "embed-english-v3.0", "dimension": 1024})

# Combine with concatenation
ensemble_config = {
    "combination_method": "concat",  # Concatenate embeddings
    "cache_enabled": True,
}

ensemble = EnsembleEmbedder(ensemble_config, embedders=[voyage, cohere])

result = ensemble.embed_with_cache(docs)
print(f"Combined dimension: {result.dimension}")  # 2048 (1024 + 1024)
```

**2. Weighted Average:**
```python
# Weighted combination (optimized weights from validation)
ensemble_config = {
    "combination_method": "weighted",
    "weights": [0.6, 0.4],  # Voyage 60%, Cohere 40%
}

ensemble = EnsembleEmbedder(ensemble_config, embedders=[voyage, cohere])
result = ensemble.embed_with_cache(docs)
print(f"Combined dimension: {result.dimension}")  # 1024 (averaged)
```

**3. Mean Pooling:**
```python
# Simple average (equal weights)
ensemble_config = {
    "combination_method": "mean",
}

ensemble = EnsembleEmbedder(ensemble_config, embedders=[voyage, cohere])
result = ensemble.embed_with_cache(docs)
```

**When to Use Each Strategy:**
- **Concat:** Maximum information retention, higher accuracy, more storage
- **Weighted:** Balanced approach, optimized for specific use case
- **Mean:** Simple, fast, good baseline for multi-model

### Hands-On Exercise: Implement Multi-Model Embedding (45 minutes)

**Exercise 8.1: Create Optimized Embedding Pipeline**

**Objective:** Build a production-ready embedding system with fallbacks and optimization.

**Requirements:**
1. Primary embedder: Voyage AI (if API key available)
2. Fallback embedder: Cohere or OpenAI
3. Final fallback: BGE (local model)
4. Caching enabled for all
5. Error handling and automatic fallback
6. Performance monitoring

**Template:**
```python
import os
from training.advanced_embeddings import EmbedderFactory

# Configuration with fallback chain
configs = [
    {
        "model": "voyage-large-2-instruct",
        "api_key": os.getenv("VOYAGE_API_KEY"),
        "dimension": 1024,
        "cache_enabled": True,
    },
    {
        "model": "embed-english-v3.0",
        "provider": "cohere",
        "api_key": os.getenv("COHERE_API_KEY"),
        "dimension": 512,  # Matryoshka reduction
        "cache_enabled": True,
    },
    {
        "model": "BAAI/bge-large-en-v1.5",
        "cache_enabled": True,
    }
]

# Create embedder with automatic fallback
embedder = EmbedderFactory.create_with_fallback(configs)

# Test documents
test_docs = [
    "What is the UCB1 formula in MCTS?",
    "How does LangGraph handle conditional branching?",
    "Explain the exploration-exploitation tradeoff",
]

# Embed with fallback protection
result = embedder.embed_with_cache(test_docs)

# Print results
print(f"Using model: {result.model}")
print(f"Dimension: {result.dimension}")
print(f"Cached: {result.cached}")
print(f"Cache hit rate: {result.metadata['cache_hit_rate']:.2%}")
print(f"Latency: {result.latency_ms:.2f}ms")

# Verify embeddings
print(f"Embeddings shape: {result.embeddings.shape}")
print(f"Sample embedding (first 10 dims): {result.embeddings[0][:10]}")
```

**Success Criteria:**
- System successfully embeds documents
- Fallback works if primary API unavailable
- Cache reduces repeated embedding costs
- Latency under 500ms for 10 documents

**Deliverable:** Working embedding pipeline with fallback chain

---

## Session 2: Hybrid Retrieval Systems (3 hours)

### Pre-Reading (30 minutes)

- Hybrid Search paper: [Dense-Sparse Fusion](https://arxiv.org/abs/2104.08663)
- BM25 algorithm overview
- Re-ranking strategies

### Lecture: Dense + Sparse Retrieval (60 minutes)

#### Why Hybrid Search?

**Dense Retrieval (Vector Search):**
- ✅ Captures semantic similarity
- ✅ Handles paraphrasing and synonyms
- ✅ Works across languages
- ❌ Struggles with exact keyword matching
- ❌ Poor for rare terms and acronyms

**Sparse Retrieval (BM25):**
- ✅ Excellent for exact keywords
- ✅ Fast and efficient
- ✅ Handles rare terms and acronyms
- ❌ No semantic understanding
- ❌ Fails with paraphrasing

**Hybrid = Best of Both Worlds**

#### BM25 Algorithm

**Formula:**
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

where:
- qi: query term i
- f(qi, D): frequency of qi in document D
- |D|: length of document D
- avgdl: average document length
- k1: term frequency saturation parameter (default: 1.5)
- b: length normalization parameter (default: 0.75)
- IDF(qi): inverse document frequency of qi
```

**Implementation with Pinecone:**
```python
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """Hybrid dense + sparse retrieval system."""

    def __init__(
        self,
        pinecone_index,
        embedder,
        documents: list[dict],
        alpha: float = 0.5
    ):
        """
        Initialize hybrid retriever.

        Args:
            pinecone_index: Pinecone index for dense retrieval
            embedder: Embedding model
            documents: List of documents with text and metadata
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
        """
        self.index = pinecone_index
        self.embedder = embedder
        self.documents = documents
        self.alpha = alpha

        # Build BM25 index
        tokenized_docs = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve using hybrid dense + sparse search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieved documents with scores
        """
        # Dense retrieval (vector search)
        query_embedding = self.embedder.embed([query])[0]
        dense_results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k * 2,  # Retrieve more for re-ranking
            include_metadata=True
        )

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        dense_scores = {
            match['id']: match['score']
            for match in dense_results['matches']
        }
        dense_scores = self._normalize_scores(dense_scores)

        bm25_scores_dict = {
            doc['id']: score
            for doc, score in zip(self.documents, bm25_scores)
        }
        bm25_scores_dict = self._normalize_scores(bm25_scores_dict)

        # Combine scores
        all_doc_ids = set(dense_scores.keys()) | set(bm25_scores_dict.keys())
        hybrid_scores = {}

        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            sparse_score = bm25_scores_dict.get(doc_id, 0.0)

            # Weighted combination
            hybrid_scores[doc_id] = (
                self.alpha * dense_score +
                (1 - self.alpha) * sparse_score
            )

        # Sort and return top-k
        sorted_docs = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            doc = next(d for d in self.documents if d['id'] == doc_id)
            results.append({
                'id': doc_id,
                'text': doc['text'],
                'score': score,
                'dense_score': dense_scores.get(doc_id, 0.0),
                'sparse_score': bm25_scores_dict.get(doc_id, 0.0),
            })

        return results

    def _normalize_scores(self, scores: dict) -> dict:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

# Usage example
retriever = HybridRetriever(
    pinecone_index=index,
    embedder=embedder,
    documents=documents,
    alpha=0.5  # Equal weight to dense and sparse
)

results = retriever.retrieve(
    query="What is UCB1 exploration constant?",
    top_k=10
)

for i, result in enumerate(results):
    print(f"{i+1}. Score: {result['score']:.3f} "
          f"(Dense: {result['dense_score']:.3f}, "
          f"Sparse: {result['sparse_score']:.3f})")
    print(f"   {result['text'][:100]}...")
```

### Lecture: Re-Ranking Strategies (45 minutes)

#### Cross-Encoder Re-Ranking

**Why Re-Rank:**
- Initial retrieval: Fast but approximate (bi-encoder)
- Re-ranking: Slow but accurate (cross-encoder)
- Two-stage pipeline: Best of both worlds

**Architecture:**
```
Query → Bi-Encoder → Top-100 results → Cross-Encoder → Top-10 results
        (Fast)                          (Slow but accurate)
```

**Implementation:**
```python
from sentence_transformers import CrossEncoder

class ReRanker:
    """Cross-encoder re-ranker for improved ranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder re-ranker."""
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 10
    ) -> list[dict]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of results to return

        Returns:
            Re-ranked documents
        """
        # Prepare input pairs
        pairs = [[query, doc['text']] for doc in documents]

        # Compute cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine scores with documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)

        # Sort by re-rank score
        reranked = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )

        return reranked[:top_k]

# Usage
reranker = ReRanker()

# Get initial candidates from hybrid retrieval
candidates = retriever.retrieve(query="UCB1 formula", top_k=100)

# Re-rank to get final top-10
final_results = reranker.rerank(
    query="UCB1 formula",
    documents=candidates,
    top_k=10
)

print(f"Initial top result score: {candidates[0]['score']:.3f}")
print(f"Re-ranked top result score: {final_results[0]['rerank_score']:.3f}")
```

### Hands-On Exercise: Build Hybrid Retrieval Pipeline (45 minutes)

**Exercise 8.2: Complete Hybrid RAG System**

**Objective:** Implement end-to-end hybrid retrieval with re-ranking.

**Requirements:**
1. Dense retrieval with Voyage/Cohere embeddings
2. Sparse retrieval with BM25
3. Hybrid score combination (alpha=0.5)
4. Cross-encoder re-ranking
5. Performance metrics (latency, recall@10)

**Template provided in:** `labs/module_8/exercise_8_2_hybrid_rag.py`

**Deliverable:** Working hybrid RAG system with benchmark results

---

## Session 3: Performance Optimization & Benchmarking (3 hours)

### Pre-Reading (30 minutes)

- [training/benchmark_suite.py](../../training/benchmark_suite.py) - Benchmarking framework
- [training/embedding_benchmark.py](../../training/embedding_benchmark.py) - Embedding benchmarks

### Lecture: Embedding Performance Benchmarking (60 minutes)

#### Key Metrics for RAG Evaluation

**Retrieval Metrics:**
1. **nDCG@k:** Normalized Discounted Cumulative Gain
2. **Recall@k:** Fraction of relevant docs in top-k
3. **MRR:** Mean Reciprocal Rank
4. **Precision@k:** Precision in top-k results

**Implementation:**
```python
from training.benchmark_suite import RetrievalMetrics, RetrievalResult

# Create retrieval result
result = RetrievalResult(
    query="What is MCTS UCB1 formula?",
    retrieved_docs=["doc_1", "doc_3", "doc_5", "doc_2", "doc_7"],
    relevance_scores=[0.95, 0.88, 0.82, 0.75, 0.70],
    ground_truth_relevant=["doc_1", "doc_2", "doc_4"],
    ground_truth_rankings={"doc_1": 1, "doc_2": 2, "doc_4": 3}
)

# Compute metrics
metrics = RetrievalMetrics()

ndcg_10 = metrics.ndcg_at_k(result, k=10)
recall_10 = metrics.recall_at_k(result, k=10)
mrr = metrics.mean_reciprocal_rank(result)
precision_5 = metrics.precision_at_k(result, k=5)

print(f"nDCG@10: {ndcg_10:.3f}")
print(f"Recall@10: {recall_10:.3f}")
print(f"MRR: {mrr:.3f}")
print(f"Precision@5: {precision_5:.3f}")
```

#### Embedding Model Comparison

**Benchmark Script:**
```python
from training.embedding_benchmark import EmbeddingBenchmark

# Define models to compare
models = [
    {
        "name": "voyage-large-2",
        "type": "voyage",
        "config": {"model": "voyage-large-2-instruct", "dimension": 1024}
    },
    {
        "name": "cohere-v3-1024",
        "type": "cohere",
        "config": {"model": "embed-english-v3.0", "dimension": 1024}
    },
    {
        "name": "cohere-v3-512",
        "type": "cohere",
        "config": {"model": "embed-english-v3.0", "dimension": 512}
    },
    {
        "name": "openai-3-large",
        "type": "openai",
        "config": {"model": "text-embedding-3-large", "dimension": 1024}
    },
]

# Run benchmark
benchmark = EmbeddingBenchmark(
    models=models,
    test_queries_path="data/test_queries.json",
    ground_truth_path="data/ground_truth.json"
)

results = benchmark.run_benchmark()

# Display results
benchmark.print_comparison_table(results)
```

**Expected Output:**
```
Embedding Model Benchmark Results
==================================
Model                 | nDCG@10 | Recall@10 | MRR   | Latency (ms) | Cost/1K
---------------------|---------|-----------|-------|--------------|----------
voyage-large-2       | 0.892   | 0.945     | 0.867 | 245          | $0.12
cohere-v3-1024       | 0.885   | 0.938     | 0.859 | 198          | $0.10
cohere-v3-512        | 0.871   | 0.925     | 0.845 | 112          | $0.05
openai-3-large       | 0.868   | 0.932     | 0.851 | 189          | $0.13

Recommendations:
- Best accuracy: voyage-large-2 (nDCG@10: 0.892)
- Best value: cohere-v3-512 (75% less storage, 54% faster)
- Best speed: cohere-v3-512 (112ms avg latency)
```

### Lecture: Caching and Cost Optimization (45 minutes)

#### Intelligent Caching Strategy

**Cache Levels:**
1. **Embedding Cache:** Cache embeddings by text hash
2. **Query Cache:** Cache retrieval results by query
3. **LRU Eviction:** Remove least recently used entries

**Implementation:**
```python
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path

class CachedEmbedder:
    """Embedder with disk-based cache."""

    def __init__(self, embedder, cache_dir: str = "./cache/embeddings"):
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        content = f"{self.embedder.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                # Load from cache
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                embeddings.append((i, embedding))
                self.cache_hits += 1
            else:
                # Need to embed
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.cache_misses += 1

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self.embedder.embed(uncached_texts)

            # Save to cache and add to results
            for idx, text, embedding in zip(
                uncached_indices, uncached_texts, new_embeddings
            ):
                cache_key = self._get_cache_key(text)
                cache_file = self.cache_dir / f"{cache_key}.pkl"

                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)

                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

# Usage
embedder = VoyageEmbedder(config)
cached_embedder = CachedEmbedder(embedder)

# First call - cache miss
docs = ["MCTS exploration", "UCB1 formula"]
embeddings1 = cached_embedder.embed(docs)

# Second call - cache hit
embeddings2 = cached_embedder.embed(docs)

stats = cached_embedder.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cost savings: ~{stats['hit_rate'] * 100:.0f}%")
```

**Cost Analysis:**

| Strategy | Embeddings/Month | Cost (Voyage) | With 90% Cache | Savings |
|----------|------------------|---------------|----------------|---------|
| No Cache | 1,000,000 | $120 | N/A | N/A |
| With Cache | 1,000,000 | $120 | $12 | $108/mo |

### Hands-On Exercise: Benchmark and Optimize (45 minutes)

**Exercise 8.3: Complete Performance Optimization**

**Objective:** Optimize RAG system for production deployment.

**Tasks:**
1. Run embedding benchmark on 3+ models
2. Implement caching layer
3. Measure cost savings
4. Optimize batch sizes
5. Profile memory usage

**Template provided in:** `labs/module_8/exercise_8_3_optimization.py`

**Deliverable:** Performance report with recommendations

---

## Session 4: Production Deployment (3 hours)

### Pre-Reading (30 minutes)

- [Pinecone Best Practices](https://docs.pinecone.io/docs/performance-tuning)
- Vector database optimization strategies

### Lecture: Production RAG Architecture (60 minutes)

#### Complete Production Stack

**Architecture:**
```
User Query
    ↓
API Gateway (FastAPI)
    ↓
Query Preprocessing
    ↓
┌─────────────────────────────────┐
│   Hybrid Retrieval Pipeline     │
│                                 │
│  ┌──────────────┐              │
│  │ Dense Search │ → Pinecone   │
│  └──────────────┘              │
│                                 │
│  ┌──────────────┐              │
│  │ Sparse Search│ → BM25       │
│  └──────────────┘              │
│                                 │
│  Score Fusion (alpha=0.5)      │
└─────────────────────────────────┘
    ↓
Re-Ranking (Cross-Encoder)
    ↓
LLM Generation (with context)
    ↓
Response + Citations
```

**FastAPI Implementation:**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

app = FastAPI(title="Advanced RAG API")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranking: bool = True
    alpha: float = 0.5  # Dense vs sparse weight

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_stats: dict

# Initialize components
embedder = initialize_embedder()
retriever = HybridRetriever(...)
reranker = ReRanker(...)
llm = initialize_llm()

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Process RAG query with hybrid retrieval."""
    try:
        # 1. Retrieve candidates
        candidates = retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,  # Retrieve more for re-ranking
            alpha=request.alpha
        )

        # 2. Re-rank if requested
        if request.use_reranking:
            results = reranker.rerank(
                query=request.query,
                documents=candidates,
                top_k=request.top_k
            )
        else:
            results = candidates[:request.top_k]

        # 3. Generate answer with LLM
        context = "\n\n".join([doc['text'] for doc in results])
        answer = llm.generate(
            prompt=f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:",
            max_tokens=500
        )

        # 4. Prepare response
        return QueryResponse(
            answer=answer,
            sources=[
                {
                    "text": doc['text'][:200],
                    "score": doc.get('rerank_score', doc['score']),
                    "id": doc['id']
                }
                for doc in results
            ],
            retrieval_stats={
                "num_candidates": len(candidates),
                "num_reranked": len(results),
                "avg_score": sum(r['score'] for r in results) / len(results)
            }
        )

    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "embedder": embedder.model_name,
        "cache_stats": embedder.get_cache_stats()
    }
```

### Lecture: Monitoring and Observability (45 minutes)

**Key Metrics to Track:**
1. **Latency:** p50, p95, p99 response times
2. **Quality:** nDCG@10, user feedback ratings
3. **Cost:** API calls, embedding costs
4. **Cache Performance:** Hit rates, storage usage

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
query_counter = Counter(
    'rag_queries_total',
    'Total RAG queries',
    ['status', 'model']
)

query_latency = Histogram(
    'rag_query_duration_seconds',
    'RAG query latency',
    ['stage']
)

cache_hit_rate = Gauge(
    'embedding_cache_hit_rate',
    'Embedding cache hit rate'
)

retrieval_quality = Histogram(
    'retrieval_ndcg_score',
    'Retrieval nDCG@10 score'
)

# Usage in query handler
with query_latency.labels(stage='retrieval').time():
    results = retriever.retrieve(query)

query_counter.labels(status='success', model=embedder.model_name).inc()
cache_hit_rate.set(embedder.get_cache_stats()['hit_rate'])
```

### Hands-On Exercise: Deploy Production RAG (45 minutes)

**Exercise 8.4: Complete Production Deployment**

**Objective:** Deploy fully-functional RAG API with monitoring.

**Requirements:**
1. FastAPI server with `/query` and `/health` endpoints
2. Hybrid retrieval pipeline
3. Prometheus metrics
4. Docker containerization
5. Health checks and graceful shutdown

**Deliverable:** Deployed RAG API with monitoring dashboard

---

## Module 8 Assessment

### Practical Assessment (100 points)

**Task:** Build and deploy advanced RAG system

**Requirements:**

1. **Embedding System (25 points)**
   - Implement multi-model with fallbacks
   - Enable caching
   - Performance benchmarks

2. **Hybrid Retrieval (25 points)**
   - Dense + sparse combination
   - Optimal alpha tuning
   - Re-ranking pipeline

3. **Optimization (25 points)**
   - Latency under 500ms p95
   - Cache hit rate >70%
   - Cost analysis

4. **Production Deployment (25 points)**
   - FastAPI server
   - Prometheus metrics
   - Docker deployment
   - Documentation

**Minimum Passing:** 70/100

**Submission:** GitHub repository with complete implementation

---

## Assessment Rubric

| Criteria | Excellent (90-100%) | Good (70-89%) | Needs Improvement (<70%) |
|----------|-------------------|--------------|------------------------|
| **Embeddings** | Multi-model with optimal fallbacks | Single model with fallback | No fallback handling |
| **Retrieval** | Hybrid with tuned alpha | Dense or sparse only | Basic retrieval |
| **Performance** | <500ms p95, >80% cache | <1s p95, >50% cache | >1s or poor caching |
| **Production** | Full API, metrics, docs | Working API | Incomplete implementation |

---

## Additional Resources

### Code References
- [training/advanced_embeddings.py](../../training/advanced_embeddings.py)
- [training/embedding_benchmark.py](../../training/embedding_benchmark.py)
- [training/benchmark_suite.py](../../training/benchmark_suite.py)

### Reading
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Matryoshka Embeddings Paper](https://arxiv.org/abs/2205.13147)
- [Hybrid Search Best Practices](https://www.pinecone.io/learn/hybrid-search/)

### Tools
- Voyage AI: https://www.voyageai.com/
- Cohere Embeddings: https://docs.cohere.com/docs/embeddings
- Pinecone: https://www.pinecone.io/

---

## What's Next?

**Congratulations!** You've mastered advanced RAG techniques. Continue to:

- **Module 9:** Knowledge Engineering & Continual Learning
- Build research corpus from arXiv papers
- Generate synthetic training data at scale
- Implement production feedback loops

---

**Module 8 Complete!** 🎯
