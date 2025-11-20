# Advanced Lab Exercises (Modules 8-10)
## LangGraph Multi-Agent MCTS Advanced Training

**Purpose:** Hands-on exercises for advanced RAG, knowledge engineering, and self-improving systems
**Format:** Progressive difficulty with complete code templates
**Time:** 40-50 hours total (self-paced)

---

## Table of Contents

1. [Module 8 Labs: Advanced RAG](#module-8-labs-advanced-rag)
2. [Module 9 Labs: Knowledge Engineering](#module-9-labs-knowledge-engineering)
3. [Module 10 Labs: Self-Improving AI](#module-10-labs-self-improving-ai)
4. [Integration Labs](#integration-labs)
5. [Capstone Project](#capstone-project)

---

## Module 8 Labs: Advanced RAG

### Lab 8.1: Multi-Model Embedding Pipeline (2 hours)

**Objective:** Build production-ready embedding system with fallbacks.

**Setup:**
```bash
cd langgraph_multi_agent_mcts
export VOYAGE_API_KEY="your-key-here"     # Optional
export COHERE_API_KEY="your-key-here"     # Optional
export OPENAI_API_KEY="your-key-here"     # Required
```

**Task 1: Configure Embedding Chain (30 minutes)**

Create `labs/module_8/lab_8_1_embeddings.py`:

```python
#!/usr/bin/env python3
"""Lab 8.1: Multi-model embedding pipeline with fallbacks."""

import os
import time
from training.advanced_embeddings import (
    EmbedderFactory,
    VoyageEmbedder,
    CohereEmbedder,
    OpenAIEmbedder,
    BGEEmbedder,
)

def main():
    # TODO: Configure fallback chain
    configs = [
        # Primary: Voyage (if available)
        {
            "model": "voyage-large-2-instruct",
            "api_key": os.getenv("VOYAGE_API_KEY"),
            "dimension": 1024,
            "cache_enabled": True,
            "cache_dir": "./cache/embeddings",
        },
        # Fallback 1: Cohere
        {
            "model": "embed-english-v3.0",
            "provider": "cohere",
            "api_key": os.getenv("COHERE_API_KEY"),
            "dimension": 512,  # Matryoshka reduction
            "cache_enabled": True,
        },
        # Fallback 2: OpenAI
        {
            "model": "text-embedding-3-large",
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "dimension": 1024,
            "cache_enabled": True,
        },
        # Final fallback: BGE (local)
        {
            "model": "BAAI/bge-large-en-v1.5",
            "cache_enabled": True,
        }
    ]

    # Create embedder with fallback
    print("Initializing embedder with fallback chain...")
    embedder = EmbedderFactory.create_with_fallback(configs)
    print(f"✓ Using model: {embedder.model_name}")

    # TODO: Test with sample documents
    test_docs = [
        "What is the UCB1 formula in Monte Carlo Tree Search?",
        "How does LangGraph handle conditional branching in workflows?",
        "Explain the exploration-exploitation tradeoff in MCTS",
        "What are the key components of the AlphaZero training algorithm?",
        "How do you implement task decomposition in multi-agent systems?",
    ]

    # First run - cache miss
    print("\nFirst embedding run (cache miss)...")
    start = time.time()
    result1 = embedder.embed_with_cache(test_docs)
    latency1 = (time.time() - start) * 1000
    print(f"✓ Embedded {len(test_docs)} documents")
    print(f"  Model: {result1.model}")
    print(f"  Dimension: {result1.dimension}")
    print(f"  Latency: {latency1:.2f}ms")
    print(f"  Cache hit rate: {result1.metadata['cache_hit_rate']:.2%}")

    # Second run - cache hit
    print("\nSecond embedding run (cache hit)...")
    start = time.time()
    result2 = embedder.embed_with_cache(test_docs)
    latency2 = (time.time() - start) * 1000
    print(f"✓ Embedded {len(test_docs)} documents")
    print(f"  Latency: {latency2:.2f}ms")
    print(f"  Cache hit rate: {result2.metadata['cache_hit_rate']:.2%}")
    print(f"  Speedup: {latency1/latency2:.1f}x faster")

    # TODO: Verify embeddings
    print("\nEmbedding verification:")
    print(f"  Shape: {result1.embeddings.shape}")
    print(f"  Sample (first 10 dims): {result1.embeddings[0][:10]}")

    # TODO: Test similarity
    print("\nSimilarity test:")
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    # Compare first two documents (both about MCTS)
    sim_related = cosine_similarity(result1.embeddings[0], result1.embeddings[2])
    print(f"  MCTS UCB1 vs MCTS exploration: {sim_related:.3f}")

    # Compare unrelated documents
    sim_unrelated = cosine_similarity(result1.embeddings[0], result1.embeddings[1])
    print(f"  MCTS vs LangGraph: {sim_unrelated:.3f}")

    print("\n✅ Lab 8.1 Complete!")
    print(f"\nNext: Implement hybrid retrieval in Lab 8.2")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Initializing embedder with fallback chain...
✓ Using model: voyage-large-2-instruct

First embedding run (cache miss)...
✓ Embedded 5 documents
  Model: voyage-large-2-instruct
  Dimension: 1024
  Latency: 245.34ms
  Cache hit rate: 0.00%

Second embedding run (cache hit)...
✓ Embedded 5 documents
  Latency: 2.15ms
  Cache hit rate: 100.00%
  Speedup: 114.1x faster

Embedding verification:
  Shape: (5, 1024)
  Sample (first 10 dims): [0.023, -0.145, 0.089, ...]

Similarity test:
  MCTS UCB1 vs MCTS exploration: 0.876
  MCTS vs LangGraph: 0.512

✅ Lab 8.1 Complete!
```

**Success Criteria:**
- ✅ Embedder initializes successfully
- ✅ Cache works (>90% hit rate on second run)
- ✅ Latency improvement >50x with cache
- ✅ Related documents have >0.7 similarity
- ✅ Unrelated documents have <0.6 similarity

**Troubleshooting:**
- **Error: API key not found** → Check environment variables
- **Error: Model not available** → Fallback should work automatically
- **Low similarity scores** → Verify embedding dimension matches across runs

---

### Lab 8.2: Hybrid Retrieval System (3 hours)

**Objective:** Implement dense + sparse hybrid search with re-ranking.

**Setup:**
```bash
pip install rank-bm25 sentence-transformers
```

**Task 1: Build Hybrid Retriever (90 minutes)**

Create `labs/module_8/lab_8_2_hybrid_search.py`:

```python
#!/usr/bin/env python3
"""Lab 8.2: Hybrid dense + sparse retrieval."""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from training.advanced_embeddings import EmbedderFactory
import os

# Sample document corpus
DOCUMENTS = [
    {
        "id": "doc_1",
        "text": "Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision processes. It uses random sampling to evaluate possible moves.",
    },
    {
        "id": "doc_2",
        "text": "The UCB1 formula balances exploration and exploitation: UCB1 = Q(a) + C * sqrt(ln(N) / n(a)) where Q(a) is the average reward.",
    },
    {
        "id": "doc_3",
        "text": "LangGraph is a framework for building stateful, multi-agent workflows with LLMs. It supports conditional branching and cycles.",
    },
    {
        "id": "doc_4",
        "text": "AlphaZero combines MCTS with deep neural networks for policy and value prediction. It learns entirely through self-play.",
    },
    {
        "id": "doc_5",
        "text": "PUCT (Predictor + UCT) extends UCB1 for neural MCTS by incorporating prior policy: PUCT = Q + U where U = c_puct * P * sqrt(N) / (1+n)",
    },
    {
        "id": "doc_6",
        "text": "Task decomposition in HRM breaks complex queries into subtasks. Each subtask can be solved independently.",
    },
    {
        "id": "doc_7",
        "text": "TRM (Task Refinement Module) iteratively improves solutions through multiple refinement rounds with feedback.",
    },
    {
        "id": "doc_8",
        "text": "MCTS exploration constant C controls the exploration-exploitation tradeoff. Higher C means more exploration.",
    },
]

class HybridRetriever:
    """Hybrid dense + sparse retriever."""

    def __init__(self, documents, embedder, alpha=0.5):
        self.documents = documents
        self.embedder = embedder
        self.alpha = alpha  # Weight for dense vs sparse

        # Build BM25 index
        print("Building BM25 index...")
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Build dense index (embed all documents)
        print("Building dense index...")
        texts = [doc["text"] for doc in documents]
        result = embedder.embed_with_cache(texts)
        self.dense_vectors = result.embeddings
        print(f"✓ Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 5):
        """Hybrid retrieval."""
        # Dense retrieval (cosine similarity)
        query_embedding = self.embedder.embed([query])[0]

        dense_scores = {}
        for i, doc in enumerate(self.documents):
            sim = np.dot(query_embedding, self.dense_vectors[i])
            sim = sim / (np.linalg.norm(query_embedding) * np.linalg.norm(self.dense_vectors[i]))
            dense_scores[doc["id"]] = float(sim)

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores_array = self.bm25.get_scores(tokenized_query)
        sparse_scores = {
            doc["id"]: score
            for doc, score in zip(self.documents, bm25_scores_array)
        }

        # Normalize scores
        def normalize(scores):
            vals = list(scores.values())
            min_val, max_val = min(vals), max(vals)
            if max_val == min_val:
                return {k: 1.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        dense_scores = normalize(dense_scores)
        sparse_scores = normalize(sparse_scores)

        # Combine scores
        hybrid_scores = {}
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        for doc_id in all_ids:
            d_score = dense_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)
            hybrid_scores[doc_id] = self.alpha * d_score + (1 - self.alpha) * s_score

        # Sort and return top-k
        sorted_ids = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in sorted_ids[:top_k]:
            doc = next(d for d in self.documents if d["id"] == doc_id)
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "score": score,
                "dense_score": dense_scores[doc_id],
                "sparse_score": sparse_scores[doc_id],
            })

        return results


def main():
    # Initialize embedder
    print("Initializing embedder...")
    embedder = EmbedderFactory.create_with_fallback([
        {
            "model": "text-embedding-3-large",
            "provider": "openai",
            "dimension": 1024,
            "cache_enabled": True,
        }
    ])

    # TODO: Create hybrid retriever
    retriever = HybridRetriever(
        documents=DOCUMENTS,
        embedder=embedder,
        alpha=0.5  # Equal weight
    )

    # TODO: Test queries
    queries = [
        "What is the UCB1 formula?",
        "How does MCTS exploration work?",
        "Explain LangGraph workflows",
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)

        results = retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.3f} "
                  f"(Dense: {result['dense_score']:.3f}, Sparse: {result['sparse_score']:.3f})")
            print(f"   ID: {result['id']}")
            print(f"   Text: {result['text'][:100]}...")

    print("\n✅ Lab 8.2 Complete!")
    print("Next: Add cross-encoder re-ranking for improved accuracy")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Initializing embedder...
Building BM25 index...
Building dense index...
✓ Indexed 8 documents

======================================================================
Query: What is the UCB1 formula?
======================================================================

1. Score: 0.925 (Dense: 0.891, Sparse: 0.958)
   ID: doc_2
   Text: The UCB1 formula balances exploration and exploitation: UCB1 = Q(a) + C * sqrt(ln(N) / n(a))...

2. Score: 0.743 (Dense: 0.678, Sparse: 0.809)
   ID: doc_5
   Text: PUCT (Predictor + UCT) extends UCB1 for neural MCTS by incorporating prior policy...

3. Score: 0.612 (Dense: 0.645, Sparse: 0.579)
   ID: doc_1
   Text: Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision processes...
```

**Success Criteria:**
- ✅ Hybrid retrieval combines dense + sparse
- ✅ Relevant documents ranked in top-3
- ✅ Hybrid scores > max(dense, sparse) for best results
- ✅ System handles various query types

---

### Lab 8.3: Performance Benchmarking (2 hours)

**Objective:** Benchmark embedding models and optimize performance.

See full implementation in `labs/module_8/lab_8_3_benchmarking.py`

**Key Tasks:**
1. Benchmark 3+ embedding models
2. Measure latency and accuracy
3. Optimize batch sizes
4. Profile memory usage
5. Generate performance report

---

## Module 9 Labs: Knowledge Engineering

### Lab 9.1: Build Research Corpus (3 hours)

**Objective:** Ingest 100+ papers from arXiv and build searchable corpus.

**Setup:**
```bash
pip install arxiv pyyaml
export LANGSMITH_PROJECT="research-corpus-lab"
```

**Task 1: Configure and Run Corpus Builder (90 minutes)**

Create `labs/module_9/lab_9_1_research_corpus.py`:

```python
#!/usr/bin/env python3
"""Lab 9.1: Build research corpus from arXiv papers."""

from training.research_corpus_builder import ResearchCorpusBuilder
from pathlib import Path
import json

def main():
    # TODO: Configure for your research area
    config = {
        "categories": [
            "cs.AI",  # Artificial Intelligence
            "cs.LG",  # Machine Learning
            "cs.CL",  # Computation and Language
        ],
        "keywords": [
            "MCTS",
            "Monte Carlo Tree Search",
            "AlphaZero",
            "reinforcement learning",
            "multi-agent systems",
            "LLM reasoning",
        ],
        "date_start": "2023-01-01",
        "date_end": "2024-12-31",
        "max_results": 100,
        "cache_dir": "./cache/research_corpus_lab",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "rate_limit_delay": 3.0,
    }

    # Initialize builder
    print("Initializing Research Corpus Builder...")
    builder = ResearchCorpusBuilder(config=config)

    # TODO: Build corpus
    print(f"\nFetching papers from arXiv...")
    print(f"Categories: {', '.join(config['categories'])}")
    print(f"Keywords: {', '.join(config['keywords'][:3])}...")
    print(f"Date range: {config['date_start']} to {config['date_end']}")
    print(f"Target: {config['max_results']} papers\n")

    chunk_count = 0
    paper_ids = set()

    for chunk in builder.build_corpus(mode="keywords", max_papers=100, skip_cached=True):
        chunk_count += 1
        paper_ids.add(chunk.doc_id)

        if chunk_count % 50 == 0:
            print(f"  Processed {chunk_count} chunks from {len(paper_ids)} papers...")

    # Get statistics
    stats = builder.get_statistics()

    print(f"\n{'='*70}")
    print("RESEARCH CORPUS STATISTICS")
    print('='*70)
    print(f"Papers fetched: {stats.total_papers_fetched}")
    print(f"Papers processed: {stats.total_papers_processed}")
    print(f"Chunks created: {stats.total_chunks_created}")
    print(f"Papers cached: {stats.papers_cached}")
    print(f"Papers skipped (cached): {stats.papers_skipped}")
    print(f"Errors: {stats.errors}")

    if stats.date_range[0] and stats.date_range[1]:
        print(f"\nDate range:")
        print(f"  Earliest: {stats.date_range[0].strftime('%Y-%m-%d')}")
        print(f"  Latest: {stats.date_range[1].strftime('%Y-%m-%d')}")

    print(f"\nCategory breakdown:")
    for category, count in sorted(stats.categories_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} papers")

    # TODO: Export metadata
    output_dir = Path("./output/research_corpus_lab")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "papers_metadata.json"
    builder.export_metadata(metadata_path)
    print(f"\n✓ Exported metadata to {metadata_path}")

    # TODO: Generate report
    report = {
        "lab": "9.1 - Research Corpus Building",
        "date": "2024-11-20",
        "statistics": {
            "papers_fetched": stats.total_papers_fetched,
            "papers_processed": stats.total_papers_processed,
            "chunks_created": stats.total_chunks_created,
            "avg_chunks_per_paper": stats.total_chunks_created / max(stats.total_papers_processed, 1),
        },
        "categories": stats.categories_breakdown,
        "success": stats.total_papers_processed >= 50,  # At least 50 papers
    }

    report_path = output_dir / "corpus_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Generated report: {report_path}")

    print("\n✅ Lab 9.1 Complete!")
    print(f"\nNext steps:")
    print(f"  1. Review papers in {metadata_path}")
    print(f"  2. Integrate with vector database (Lab 9.2)")
    print(f"  3. Build hybrid search system")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Initializing Research Corpus Builder...

Fetching papers from arXiv...
Categories: cs.AI, cs.LG, cs.CL
Keywords: MCTS, Monte Carlo Tree Search, AlphaZero...
Date range: 2023-01-01 to 2024-12-31
Target: 100 papers

  Processed 50 chunks from 12 papers...
  Processed 100 chunks from 24 papers...
  Processed 150 chunks from 35 papers...
  ...

======================================================================
RESEARCH CORPUS STATISTICS
======================================================================
Papers fetched: 98
Papers processed: 98
Chunks created: 412
Papers cached: 98
Papers skipped (cached): 0
Errors: 0

Date range:
  Earliest: 2023-01-15
  Latest: 2024-11-18

Category breakdown:
  cs.LG: 52 papers
  cs.AI: 38 papers
  cs.CL: 8 papers

✓ Exported metadata to ./output/research_corpus_lab/papers_metadata.json
✓ Generated report: ./output/research_corpus_lab/corpus_report.json

✅ Lab 9.1 Complete!
```

**Success Criteria:**
- ✅ Fetch 50+ papers successfully
- ✅ Generate 200+ chunks
- ✅ No API errors
- ✅ Metadata export complete
- ✅ Statistics report generated

---

### Lab 9.2: Generate Synthetic Training Data (2.5 hours)

**Objective:** Generate 1,000 high-quality Q&A pairs using LLMs.

See full implementation in `labs/module_9/lab_9_2_synthetic_data.py`

**Key Tasks:**
1. Configure LLM client
2. Generate 1,500 raw Q&A pairs
3. Apply quality filtering
4. Target 1,000+ high-quality pairs
5. Export in LangSmith format
6. Track costs and statistics

---

## Module 10 Labs: Self-Improving AI

### Lab 10.1: Self-Play Episode Generation (3 hours)

**Objective:** Generate 1,000 self-play episodes with MCTS.

See full implementation in `labs/module_10/lab_10_1_self_play.py`

**Key Features:**
- Task generation (math, code, reasoning)
- MCTS-guided episode execution
- Complete trace capture
- Training data extraction

---

### Lab 10.2: Training Loop Implementation (3 hours)

**Objective:** Run 10 iterations of self-play training.

**Key Components:**
- Episode generation
- Policy/value target extraction
- Model training
- Evaluation and comparison
- Checkpointing

---

## Integration Labs

### Integration Lab 1: End-to-End RAG System (4 hours)

**Objective:** Build complete RAG system integrating Modules 8 & 9.

**Components:**
1. Research corpus (100+ papers)
2. Hybrid retrieval (dense + sparse)
3. Cross-encoder re-ranking
4. LLM generation with citations
5. FastAPI deployment
6. Monitoring dashboard

---

## Capstone Project

### Final Integration: Self-Improving RAG System (8 hours)

**Objective:** Build complete self-improving knowledge system.

**Requirements:**
1. **Knowledge Base (25%)**
   - Research corpus (100+ papers)
   - Synthetic Q&A (1,000+ pairs)
   - Hybrid search with re-ranking

2. **Self-Improvement (25%)**
   - Self-play episode generation
   - Training loop (5+ iterations)
   - A/B testing framework

3. **Production Deployment (25%)**
   - FastAPI endpoints
   - Prometheus metrics
   - Grafana dashboard
   - Docker deployment

4. **Monitoring & Feedback (25%)**
   - Feedback collection
   - Quality metrics
   - Continual learning
   - Performance tracking

**Deliverables:**
- Complete GitHub repository
- Demo video (5-10 minutes)
- Technical documentation
- Performance report

---

## Troubleshooting Tips

### Common Issues

**Issue: API Rate Limits**
- Solution: Adjust rate_limit_delay in config
- Use caching to reduce API calls
- Implement exponential backoff

**Issue: Out of Memory**
- Solution: Reduce batch_size
- Use Matryoshka embeddings (lower dimensions)
- Process in smaller chunks

**Issue: Low Retrieval Quality**
- Solution: Tune alpha parameter (dense/sparse weight)
- Add re-ranking layer
- Increase chunk overlap

**Issue: Slow Training**
- Solution: Use parallel batch processing
- Enable GPU acceleration
- Optimize data loading

---

## Additional Resources

### Code Templates
- All lab code: `labs/module_8/`, `labs/module_9/`, `labs/module_10/`
- Helper scripts: `scripts/lab_helpers/`

### Documentation
- [Module 8: Advanced RAG](MODULE_8_ADVANCED_RAG.md)
- [Module 9: Knowledge Engineering](MODULE_9_KNOWLEDGE_ENGINEERING.md)
- [Module 10: Self-Improving AI](MODULE_10_SELF_IMPROVEMENT.md)

### Support
- Office hours: Schedule TBD
- Discussion forum: GitHub Discussions
- Bug reports: GitHub Issues

---

**Labs Complete!** 🎉

Continue to [Assessment](ASSESSMENT_ADVANCED.md) for certification.
