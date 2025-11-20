# Troubleshooting Guide: Advanced Features
## Modules 8-10 Common Issues and Solutions

**Last Updated:** 2024-11-20
**Applies to:** LangGraph Multi-Agent MCTS v1.0+

---

## Table of Contents

1. [Advanced Embeddings Issues](#advanced-embeddings-issues)
2. [Hybrid Retrieval Problems](#hybrid-retrieval-problems)
3. [Research Corpus Building](#research-corpus-building)
4. [Synthetic Data Generation](#synthetic-data-generation)
5. [Knowledge Graph Issues](#knowledge-graph-issues)
6. [Self-Play Training](#self-play-training)
7. [Production Deployment](#production-deployment)
8. [Performance Optimization](#performance-optimization)

---

## Advanced Embeddings Issues

### Issue: API Key Not Found

**Symptoms:**
```
Error: VOYAGE_API_KEY not found in environment
```

**Solutions:**

1. **Check environment variables:**
```bash
# List all API keys
env | grep API_KEY

# Set for current session
export VOYAGE_API_KEY="your-key-here"
export COHERE_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

2. **Use .env file:**
```bash
# Create .env file
cat > .env <<EOF
VOYAGE_API_KEY=your-key-here
COHERE_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

3. **Configure fallback chain:**
```python
# If primary API unavailable, fallback works automatically
configs = [
    {"model": "voyage-large-2-instruct", "api_key": os.getenv("VOYAGE_API_KEY")},
    {"model": "embed-english-v3.0", "provider": "cohere", "api_key": os.getenv("COHERE_API_KEY")},
    {"model": "BAAI/bge-large-en-v1.5"},  # Local fallback, no API needed
]
embedder = EmbedderFactory.create_with_fallback(configs)
```

---

### Issue: Rate Limit Errors

**Symptoms:**
```
RateLimitError: Too many requests (429)
```

**Solutions:**

1. **Implement exponential backoff:**
```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limited. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=5, base_delay=2)
def embed_with_retry(embedder, texts):
    return embedder.embed(texts)
```

2. **Reduce batch size:**
```python
# Instead of large batches
result = embedder.embed(1000_documents)  # May hit rate limit

# Use smaller batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    result = embedder.embed(batch)
    time.sleep(1)  # Add delay between batches
```

3. **Enable caching:**
```python
# Cache reduces API calls
config = {
    "model": "voyage-large-2-instruct",
    "cache_enabled": True,
    "cache_dir": "./cache/embeddings"
}
embedder = VoyageEmbedder(config)

# Second call uses cache - no API call
result1 = embedder.embed(texts)  # API call
result2 = embedder.embed(texts)  # Cache hit
```

---

### Issue: Dimension Mismatch

**Symptoms:**
```
ValueError: Embedding dimension mismatch: expected 1024, got 512
```

**Solutions:**

1. **Verify configuration:**
```python
# Check configured dimension
print(f"Config dimension: {config['dimension']}")

# Check actual embedding dimension
result = embedder.embed(["test"])
print(f"Actual dimension: {result.embeddings.shape[1]}")
```

2. **Fix Matryoshka dimension:**
```python
# Cohere embed-v3 supports multiple dimensions
config = {
    "model": "embed-english-v3.0",
    "dimension": 512,  # Valid: 1024, 512, 256, 128, 64
}
```

3. **Recreate vector index:**
```bash
# If vector DB dimension doesn't match embeddings
# Delete and recreate index
python scripts/recreate_index.py --dimension 1024
```

---

### Issue: Cache Not Working

**Symptoms:**
```
Cache hit rate: 0.00% (expected >50%)
```

**Solutions:**

1. **Verify cache directory:**
```python
import os
from pathlib import Path

cache_dir = Path("./cache/embeddings")
cache_dir.mkdir(parents=True, exist_ok=True)

# Check permissions
assert os.access(cache_dir, os.W_OK), "Cache dir not writable"
```

2. **Clear corrupted cache:**
```bash
# Remove cache and rebuild
rm -rf ./cache/embeddings/*
```

3. **Check cache key generation:**
```python
# Cache keys must be deterministic
# Same text → same cache key
text1 = "MCTS exploration"
text2 = "MCTS exploration"  # Identical

key1 = embedder._get_cache_key(text1)
key2 = embedder._get_cache_key(text2)
assert key1 == key2, "Cache keys don't match!"
```

---

## Hybrid Retrieval Problems

### Issue: Poor Hybrid Search Quality

**Symptoms:**
```
nDCG@10: 0.42 (expected >0.70)
Recall@100: 0.58 (expected >0.85)
```

**Solutions:**

1. **Tune alpha parameter:**
```python
# Test different alpha values
alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
best_alpha = None
best_ndcg = 0

for alpha in alphas:
    retriever = HybridRetriever(docs, embedder, alpha=alpha)
    ndcg = evaluate_retrieval(retriever, test_queries)

    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_alpha = alpha

print(f"Best alpha: {best_alpha} (nDCG: {best_ndcg:.3f})")
```

2. **Add re-ranking:**
```python
from sentence_transformers import CrossEncoder

# Initial hybrid retrieval (top-100)
candidates = hybrid_retriever.retrieve(query, top_k=100)

# Re-rank with cross-encoder (top-10)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc["text"]] for doc in candidates]
scores = reranker.predict(pairs)

# Sort by re-rank score
for doc, score in zip(candidates, scores):
    doc["rerank_score"] = score

reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:10]
```

3. **Improve BM25 tokenization:**
```python
# Better tokenization for technical content
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize_technical(text):
    # Preserve acronyms (MCTS, UCB1, etc)
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove stopwords but keep technical terms
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words or len(t) <= 4]

    return tokens

# Use custom tokenizer
tokenized = [tokenize_technical(doc["text"]) for doc in documents]
bm25 = BM25Okapi(tokenized)
```

---

### Issue: Slow Hybrid Search

**Symptoms:**
```
Retrieval latency: 2,500ms p95 (target: <500ms)
```

**Solutions:**

1. **Optimize vector search:**
```python
# Use approximate nearest neighbor search
# Pinecone automatically uses ANN

# For FAISS:
import faiss

# Create HNSW index (faster than flat)
d = embedding_dim
index = faiss.IndexHNSWFlat(d, 32)  # 32 = M parameter
index.add(embeddings)

# Search
D, I = index.search(query_embedding, k=100)
```

2. **Parallelize dense and sparse:**
```python
import asyncio

async def dense_search(query):
    query_emb = embedder.embed([query])[0]
    return pinecone_index.query(vector=query_emb, top_k=100)

async def sparse_search(query):
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    return scores

# Run in parallel
dense_task = asyncio.create_task(dense_search(query))
sparse_task = asyncio.create_task(sparse_search(query))

dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
```

3. **Cache search results:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieval(query: str, top_k: int):
    return hybrid_retriever.retrieve(query, top_k)

# Repeated queries use cache
results = cached_retrieval("What is UCB1?", 10)  # First call
results = cached_retrieval("What is UCB1?", 10)  # Cached
```

---

## Research Corpus Building

### Issue: arXiv API Errors

**Symptoms:**
```
arxiv.arxiv.HTTPError: 503 Service Unavailable
```

**Solutions:**

1. **Respect rate limits:**
```python
config = {
    "rate_limit_delay": 3.0,  # At least 3 seconds
    "max_retries": 5,
}

# Add backoff on errors
import time

for attempt in range(config["max_retries"]):
    try:
        papers = fetch_arxiv_papers(query)
        break
    except HTTPError as e:
        if attempt == config["max_retries"] - 1:
            raise
        wait_time = config["rate_limit_delay"] * (2 ** attempt)
        print(f"Error {e}. Retrying in {wait_time}s...")
        time.sleep(wait_time)
```

2. **Use caching to avoid re-fetching:**
```python
# Check cache before API call
cache_file = cache_dir / f"{arxiv_id}.json"
if cache_file.exists():
    with open(cache_file) as f:
        return json.load(f)

# Fetch only if not cached
paper = fetch_paper(arxiv_id)
with open(cache_file, "w") as f:
    json.dump(paper, f)
```

3. **Handle malformed responses:**
```python
try:
    papers = list(search.results())
except Exception as e:
    logger.warning(f"Failed to parse results: {e}")
    papers = []

# Validate paper data
valid_papers = []
for paper in papers:
    if paper.title and paper.pdf_url:
        valid_papers.append(paper)
    else:
        logger.warning(f"Skipping invalid paper: {paper.entry_id}")
```

---

### Issue: PDF Parsing Errors

**Symptoms:**
```
Error: Failed to extract text from PDF
```

**Solutions:**

1. **Use multiple PDF parsers:**
```python
# Try multiple libraries
def extract_pdf_text(pdf_path):
    # Try PyPDF2 first
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages)
            if text.strip():
                return text
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")

    # Fallback to pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
            if text.strip():
                return text
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    raise ValueError("All PDF parsers failed")
```

2. **Skip problematic papers:**
```python
# Continue on errors instead of failing entire corpus
for paper in papers:
    try:
        text = extract_pdf_text(paper.pdf_path)
        chunks = create_chunks(text)
    except Exception as e:
        logger.error(f"Failed to process {paper.entry_id}: {e}")
        stats.errors += 1
        continue  # Skip to next paper
```

---

## Synthetic Data Generation

### Issue: Low Quality Scores

**Symptoms:**
```
Average quality score: 0.38 (target: >0.60)
Valid pairs: 452/1000 (target: >800)
```

**Solutions:**

1. **Use better model:**
```python
# Instead of gpt-3.5-turbo
llm_client = create_client("openai", model="gpt-3.5-turbo")

# Use gpt-4 for better quality
llm_client = create_client("openai", model="gpt-4")

# Or use Claude Sonnet
llm_client = create_client("anthropic", model="claude-3-sonnet-20240229")
```

2. **Improve prompts:**
```python
def create_answer_prompt(question, difficulty="medium"):
    return f"""You are an expert in Monte Carlo Tree Search and multi-agent systems.

Question: {question}

Provide a detailed, technical answer that:
1. Explains concepts clearly with proper terminology
2. Includes specific examples or code snippets
3. Uses structured formatting (bullet points, numbered lists)
4. Provides step-by-step reasoning where appropriate
5. Is comprehensive (300-500 words)

Your answer:"""
```

3. **Filter aggressively:**
```python
# Generate more, keep best
raw_pairs = await generator.generate_batch(num_samples=2000)

# Filter by quality
high_quality = generator.filter_by_quality(raw_pairs, min_score=0.7)

# Take top N
final_pairs = sorted(high_quality, key=lambda p: p.quality_score, reverse=True)[:1000]
```

---

### Issue: High API Costs

**Symptoms:**
```
Generated 1,000 pairs
Estimated cost: $180 (budget: $50)
```

**Solutions:**

1. **Two-stage generation:**
```python
# Stage 1: Bulk generation with cheap model
cheap_client = create_client("openai", model="gpt-3.5-turbo")
cheap_gen = SyntheticKnowledgeGenerator(cheap_client)
bulk_pairs = await cheap_gen.generate_batch(2000)
# Cost: ~$4

# Stage 2: Enhance top pairs
filtered = cheap_gen.filter_by_quality(bulk_pairs, min_score=0.5)
top_500 = sorted(filtered, key=lambda p: p.quality_score, reverse=True)[:500]

# Add reasoning paths with GPT-4
expensive_client = create_client("openai", model="gpt-4")
for pair in top_500:
    pair.reasoning_paths = await generate_reasoning(expensive_client, pair.question)
# Cost: ~$15

# Total: ~$19 for 500 high-quality pairs
```

2. **Cache responses:**
```python
# Avoid regenerating similar questions
@lru_cache(maxsize=1000)
def generate_answer_cached(question_hash, model):
    return llm_client.generate(question)

# Use hash for caching
question_hash = hashlib.md5(question.encode()).hexdigest()
answer = generate_answer_cached(question_hash, model="gpt-3.5-turbo")
```

3. **Optimize token usage:**
```python
# Reduce max_tokens for answers
response = llm_client.generate(
    prompt=prompt,
    max_tokens=500,  # Instead of 2000
    temperature=0.7
)
```

---

## Self-Play Training

### Issue: Low Success Rate

**Symptoms:**
```
Episodes: 1000
Success rate: 8% (target: >25%)
```

**Solutions:**

1. **Start with easier tasks:**
```python
# Adjust difficulty range
easy_gen = MathProblemGenerator(
    difficulty_range=(0.1, 0.4),  # Start easy
    seed=42
)

# Gradually increase difficulty
for iteration in range(10):
    min_diff = 0.1 + (iteration * 0.05)
    max_diff = min_diff + 0.3
    gen = MathProblemGenerator(difficulty_range=(min_diff, max_diff))
    tasks = gen.generate(100)
```

2. **Increase MCTS simulations:**
```python
# More simulations = better decisions
mcts_config = {
    "num_simulations": 200,  # Increased from 100
    "c_puct": 1.25,
}
```

3. **Add curriculum learning:**
```python
# Train on solved tasks first
successful_tasks = [ep.task for ep in episodes if ep.outcome == "success"]

# Generate similar tasks
new_tasks = generate_similar_tasks(successful_tasks, num=100)

# Mix with new tasks
all_tasks = successful_tasks + new_tasks
random.shuffle(all_tasks)
```

---

### Issue: Training Not Improving

**Symptoms:**
```
Iteration 1: Success rate 22%
Iteration 5: Success rate 23%
Iteration 10: Success rate 21%
```

**Solutions:**

1. **Check learning rate:**
```python
# Too high or too low learning rate
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,  # Try different values: 1e-3, 1e-4, 1e-5
    weight_decay=0.01
)
```

2. **Verify training data quality:**
```python
# Check if examples are diverse
policy_examples = training_data["policy"]
print(f"Policy examples: {len(policy_examples)}")

# Check for duplicates
unique_states = set(hash(ex.state.tobytes()) for ex in policy_examples)
print(f"Unique states: {len(unique_states)}")

if len(unique_states) < len(policy_examples) * 0.5:
    print("WARNING: Too many duplicate states!")
```

3. **Add exploration bonus:**
```python
# Encourage trying new states
def select_action_with_bonus(actions, visit_counts, exploration_bonus=0.1):
    scores = []
    for action in actions:
        visits = visit_counts.get(action.action_id, 0)
        # Bonus for less-visited actions
        bonus = exploration_bonus / (visits + 1)
        scores.append(q_values[action.action_id] + bonus)

    return actions[np.argmax(scores)]
```

---

## Production Deployment

### Issue: High Latency

**Symptoms:**
```
p50 latency: 850ms
p95 latency: 2,340ms
p99 latency: 4,100ms
```

**Solutions:**

1. **Profile bottlenecks:**
```python
import time

def profile_rag_pipeline(query):
    metrics = {}

    start = time.time()
    embeddings = embedder.embed([query])
    metrics["embedding_ms"] = (time.time() - start) * 1000

    start = time.time()
    candidates = retriever.retrieve(query, top_k=100)
    metrics["retrieval_ms"] = (time.time() - start) * 1000

    start = time.time()
    results = reranker.rerank(query, candidates, top_k=10)
    metrics["reranking_ms"] = (time.time() - start) * 1000

    start = time.time()
    answer = llm.generate(context=results, query=query)
    metrics["generation_ms"] = (time.time() - start) * 1000

    print(f"Pipeline breakdown:")
    for stage, latency in metrics.items():
        print(f"  {stage}: {latency:.0f}ms")

    return metrics
```

2. **Optimize each stage:**
```python
# Embedding: Use cache + batch
embedder.cache_enabled = True

# Retrieval: Reduce top_k
candidates = retriever.retrieve(query, top_k=50)  # Instead of 100

# Re-ranking: Use smaller model
reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")  # Faster

# Generation: Reduce max_tokens
answer = llm.generate(context, query, max_tokens=300)  # Instead of 500
```

3. **Add async/parallel processing:**
```python
import asyncio

async def rag_pipeline_async(query):
    # Embed query
    query_emb = await embedder.embed_async([query])

    # Parallel retrieval
    dense_task = asyncio.create_task(dense_retrieval(query_emb))
    sparse_task = asyncio.create_task(sparse_retrieval(query))
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    # Combine and re-rank
    candidates = combine_results(dense_results, sparse_results)
    results = reranker.rerank(query, candidates, top_k=10)

    # Generate answer
    answer = await llm.generate_async(context=results, query=query)
    return answer
```

---

### Issue: Memory Leaks

**Symptoms:**
```
Memory usage increasing over time
8GB → 16GB → 24GB → OOM
```

**Solutions:**

1. **Clear caches periodically:**
```python
import gc

# Clear LRU caches
@app.get("/admin/clear-cache")
def clear_cache():
    embedder.clear_cache()
    retriever.clear_cache()
    gc.collect()
    return {"status": "cache_cleared"}

# Schedule periodic cleanup
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(clear_cache, 'interval', hours=1)
scheduler.start()
```

2. **Limit buffer sizes:**
```python
# Episode buffer with max size
self.episode_buffer = deque(maxlen=10000)  # Auto-evicts old episodes

# Feedback buffer with max size
self.feedback_buffer = deque(maxlen=100000)
```

3. **Profile memory usage:**
```python
import tracemalloc

tracemalloc.start()

# Run operation
result = rag_pipeline(query)

# Check memory
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.1f}MB")
print(f"Peak: {peak / 1024**2:.1f}MB")

tracemalloc.stop()
```

---

## Performance Optimization

### Best Practices Summary

1. **Embeddings:**
   - Enable caching (>80% hit rate)
   - Use Matryoshka dimensions when possible
   - Batch API requests
   - Implement fallback chain

2. **Retrieval:**
   - Tune hybrid alpha parameter
   - Add cross-encoder re-ranking
   - Use ANN for vector search
   - Cache popular queries

3. **Training:**
   - Start with curriculum learning
   - Use experience replay
   - Monitor for catastrophic forgetting
   - Save regular checkpoints

4. **Production:**
   - Profile all pipeline stages
   - Use async/parallel processing
   - Monitor memory usage
   - Implement health checks

---

## Getting Help

### When Issues Persist

1. **Check logs:**
```bash
# Application logs
tail -f logs/app.log

# Error logs
grep ERROR logs/app.log

# Performance logs
grep "latency" logs/app.log | awk '{print $NF}' | sort -n
```

2. **Enable debug mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. **Create minimal reproduction:**
```python
# Isolate the issue
def minimal_test():
    # Minimal code that reproduces issue
    embedder = create_embedder()
    result = embedder.embed(["test"])
    print(result)

minimal_test()
```

4. **Report issue:**
- GitHub Issues: Include error logs, config, minimal reproduction
- Discussion Forum: General questions and tips
- Stack Overflow: Tag with `langgraph-mcts`

---

**Still stuck?** Check [Quick Reference](QUICK_REFERENCE_ADVANCED.md) for common commands.
