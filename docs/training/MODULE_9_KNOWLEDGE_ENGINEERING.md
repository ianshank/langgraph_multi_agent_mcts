# Module 9: Knowledge Engineering & Continual Learning

**Duration:** 14 hours (3-4 days)
**Format:** Workshop + Production Implementation Lab
**Difficulty:** Advanced
**Prerequisites:** Completed Modules 1-8, understanding of vector databases and LLMs

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Automate knowledge acquisition** from arXiv papers and code repositories
2. **Generate synthetic training data** at scale using LLMs
3. **Build and query knowledge graphs** for enhanced retrieval
4. **Implement production feedback loops** for continual learning
5. **Scale knowledge bases** to 100,000+ documents efficiently

---

## Session 1: Research Paper Ingestion (3.5 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [training/research_corpus_builder.py](../../training/research_corpus_builder.py) - arXiv ingestion pipeline
- [arXiv API Documentation](https://info.arxiv.org/help/api/index.html)
- Research corpus building best practices

### Lecture: Automated Paper Processing (90 minutes)

#### Why Ingest Research Papers?

**Benefits:**
- **Stay current:** Automatically track latest AI/ML research
- **Domain expertise:** Build specialized knowledge bases
- **Citation chains:** Follow research threads and methodologies
- **Reproducibility:** Access methods and implementations

**arXiv.org Features:**
- 2M+ papers across all scientific domains
- Full-text PDF access
- Comprehensive metadata (authors, categories, dates)
- Daily updates (~500-1000 new papers/day in CS)
- Free API access

#### Research Corpus Builder Architecture

**Pipeline Overview:**
```
arXiv API Query
    ↓
Paper Fetching (with rate limiting)
    ↓
Metadata Extraction
    ↓
Abstract Processing
    ↓
Section-Aware Chunking
    ↓
Embedding Generation
    ↓
Vector Database Storage
    ↓
Deduplication & Caching
```

**Implementation:**
```python
from training.research_corpus_builder import ResearchCorpusBuilder

# Configuration
config = {
    "categories": [
        "cs.AI",  # Artificial Intelligence
        "cs.LG",  # Machine Learning
        "cs.CL",  # Computation and Language
        "cs.NE",  # Neural and Evolutionary Computing
    ],
    "keywords": [
        "MCTS",
        "AlphaZero",
        "MuZero",
        "reinforcement learning",
        "multi-agent",
        "LLM reasoning",
        "chain-of-thought",
        "tree-of-thought",
        "self-improvement",
        "Constitutional AI",
        "RLHF",
        "DPO",
    ],
    "date_start": "2020-01-01",
    "date_end": "2024-12-31",
    "max_results": 1000,
    "cache_dir": "./cache/research_corpus",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "rate_limit_delay": 3.0,  # Respect arXiv API limits
}

# Initialize builder
builder = ResearchCorpusBuilder(config=config)

# Build corpus by keywords
print("Building research corpus...")
for chunk in builder.build_corpus(mode="keywords", max_papers=100):
    # Each chunk is a DocumentChunk object
    print(f"Processed: {chunk.doc_id}, Chunk {chunk.chunk_id}")
    print(f"Section: {chunk.metadata.get('section')}")
    print(f"Preview: {chunk.text[:100]}...")

# Get statistics
stats = builder.get_statistics()
print(f"\nCorpus Statistics:")
print(f"  Papers fetched: {stats.total_papers_fetched}")
print(f"  Papers processed: {stats.total_papers_processed}")
print(f"  Chunks created: {stats.total_chunks_created}")
print(f"  Categories: {stats.categories_breakdown}")
```

#### Advanced Query Building

**Category-Based Queries:**
```python
# Fetch papers from specific category
for chunk in builder.build_corpus(mode="categories", max_papers=500):
    # Process each chunk
    pass
```

**Keyword-Based Queries:**
```python
# Fetch papers matching specific keywords
for chunk in builder.build_corpus(mode="keywords", max_papers=100):
    # Automatically deduplicates across keywords
    pass
```

**Date-Range Filtering:**
```python
config = {
    "date_start": "2023-01-01",
    "date_end": "2024-12-31",
    # Papers published in 2023-2024 only
}
```

#### Paper Chunking Strategy

**Chunk Types:**
1. **Metadata Chunk:** Title, authors, categories
2. **Abstract Chunks:** Split if >512 tokens
3. **Comment Chunks:** Paper comments/notes

**Example Chunks:**
```python
# Chunk 1: Metadata
{
    "doc_id": "arxiv_2301.00234",
    "chunk_id": 0,
    "text": """
        Title: AlphaZero-Style Self-Play for Language Model Alignment

        Authors: Smith, J., Johnson, A., Williams, R.

        Categories: cs.AI, cs.LG, cs.CL

        Published: 2023-01-15
    """,
    "metadata": {
        "section": "title",
        "chunk_type": "metadata",
        "arxiv_id": "2301.00234",
        "pdf_url": "https://arxiv.org/pdf/2301.00234.pdf"
    }
}

# Chunk 2: Abstract
{
    "doc_id": "arxiv_2301.00234",
    "chunk_id": 1,
    "text": """
        We present a novel approach to language model alignment
        using AlphaZero-style self-play training. Our method generates
        high-quality training data through iterative self-improvement,
        eliminating the need for large-scale human feedback...
    """,
    "metadata": {
        "section": "abstract",
        "chunk_type": "abstract"
    }
}
```

### Lecture: Integration with RAG Pipeline (45 minutes)

#### End-to-End Integration

**Complete Pipeline:**
```python
from training.research_corpus_builder import (
    ResearchCorpusBuilder,
    integrate_with_rag_pipeline
)
from training.rag_builder import VectorIndexBuilder
from training.advanced_embeddings import VoyageEmbedder

# 1. Initialize corpus builder
corpus_builder = ResearchCorpusBuilder(config=config)

# 2. Initialize RAG components
embedder = VoyageEmbedder({
    "model": "voyage-large-2-instruct",
    "dimension": 1024,
    "cache_enabled": True
})

index_builder = VectorIndexBuilder(
    embedder=embedder,
    index_name="research-corpus-mcts",
    dimension=1024
)

# 3. Integrate: Fetch papers → Chunk → Embed → Index
stats = integrate_with_rag_pipeline(
    corpus_builder=corpus_builder,
    rag_index_builder=index_builder,
    batch_size=100  # Process in batches
)

print(f"Integration Complete:")
print(f"  Papers processed: {stats['corpus']['papers_processed']}")
print(f"  Chunks indexed: {stats['index']['total_chunks']}")
print(f"  Index size: {stats['index']['index_size_mb']:.2f} MB")
print(f"  Categories covered: {stats['corpus']['categories']}")
```

**Incremental Updates:**
```python
# Daily update: Fetch only new papers
daily_config = config.copy()
daily_config["date_start"] = "2024-11-19"  # Yesterday
daily_config["date_end"] = "2024-11-20"    # Today

# Incremental build
for chunk in builder.build_corpus(mode="categories", skip_cached=True):
    # Only processes new papers (cache prevents re-processing)
    index_builder.add_chunk(chunk)

print(f"Added {builder.stats.total_papers_processed} new papers")
```

### Hands-On Exercise: Build Research Corpus (90 minutes)

**Exercise 9.1: Build Custom Research Corpus**

**Objective:** Build a 100-paper research corpus on your topic of interest.

**Requirements:**
1. Configure arXiv categories and keywords
2. Fetch 100+ papers from 2020-2024
3. Process and chunk papers
4. Generate embeddings
5. Index in Pinecone/local vector DB
6. Generate statistics report

**Template:**
```python
# labs/module_9/exercise_9_1_research_corpus.py

from training.research_corpus_builder import ResearchCorpusBuilder
from training.advanced_embeddings import EmbedderFactory

# TODO: Customize configuration
config = {
    "categories": ["cs.AI", "cs.LG"],  # Your categories
    "keywords": [
        "your",
        "keywords",
        "here",
    ],
    "date_start": "2020-01-01",
    "max_results": 100,
    "cache_dir": "./cache/my_research_corpus",
}

# Initialize builder
builder = ResearchCorpusBuilder(config=config)

# TODO: Build corpus
chunk_count = 0
for chunk in builder.build_corpus(mode="keywords", max_papers=100):
    chunk_count += 1
    if chunk_count % 50 == 0:
        print(f"Processed {chunk_count} chunks...")

# TODO: Generate report
stats = builder.get_statistics()
print(f"\n=== Research Corpus Report ===")
print(f"Papers fetched: {stats.total_papers_fetched}")
print(f"Papers processed: {stats.total_papers_processed}")
print(f"Chunks created: {stats.total_chunks_created}")
print(f"Date range: {stats.date_range}")
print(f"Categories: {stats.categories_breakdown}")

# TODO: Export metadata
builder.export_metadata(Path("./output/research_metadata.json"))
```

**Deliverable:** Research corpus with 100+ papers and statistics report

---

## Session 2: Synthetic Data Generation (4 hours)

### Pre-Reading (30 minutes)

- [training/synthetic_knowledge_generator.py](../../training/synthetic_knowledge_generator.py)
- [Synthetic Data for LLMs](https://arxiv.org/abs/2305.14233)

### Lecture: LLM-Based Q&A Generation (90 minutes)

#### Why Synthetic Data?

**Benefits:**
- **Scale:** Generate 10,000+ Q&A pairs automatically
- **Coverage:** Ensure comprehensive topic coverage
- **Quality Control:** Filter by quality metrics
- **Cost-Effective:** Cheaper than human annotation
- **Diversity:** Multiple reasoning paths per question

**Use Cases:**
- Training data for fine-tuning
- Evaluation datasets
- Few-shot examples
- User simulation for testing

#### Synthetic Knowledge Generator Architecture

**Pipeline:**
```
Question Templates
    ↓
Template Filling (with vocabularies)
    ↓
LLM Answer Generation
    ↓
Quality Validation
    ↓
Quality Scoring
    ↓
Context Extraction
    ↓
LangSmith Dataset Format
```

**Implementation:**
```python
from training.synthetic_knowledge_generator import (
    SyntheticKnowledgeGenerator,
    QAPair
)
from src.adapters.llm import create_client

# Initialize LLM client
llm_client = create_client(
    provider="openai",
    model="gpt-4",
    rate_limit_per_minute=60
)

# Initialize generator
generator = SyntheticKnowledgeGenerator(
    llm_client=llm_client,
    output_dir="training/synthetic_data",
    config={
        "min_question_length": 20,
        "min_answer_length": 100,
    }
)

# Generate Q&A pairs
pairs = await generator.generate_batch(
    num_samples=1000,
    categories=[
        "mcts_algorithms",
        "langgraph_workflows",
        "multi_agent_coordination"
    ],
    batch_size=10  # Concurrent generation
)

print(f"Generated {len(pairs)} Q&A pairs")
print(f"Average quality score: {sum(p.quality_score for p in pairs) / len(pairs):.3f}")
```

#### Question Templates

**Built-in Categories:**
```python
QUESTION_TEMPLATES = {
    "mcts_algorithms": [
        "Explain {algorithm} step by step with examples",
        "How does {algorithm} work in the context of {domain}?",
        "What are the key differences between {method_a} and {method_b}?",
        "Implement {algorithm} in Python with {constraints}",
    ],
    "exploration_exploitation": [
        "How does UCB1 balance exploration and exploitation in {context}?",
        "Explain the exploration constant C in {algorithm}",
        "What is the regret bound for {algorithm}?",
    ],
    "alphazero_neural": [
        "How does AlphaZero combine MCTS with neural networks?",
        "Explain the policy and value networks in AlphaZero",
        "What is PUCT and how is it different from UCB1?",
    ],
    "langgraph_workflows": [
        "How do you implement a {workflow_type} workflow in LangGraph?",
        "What are the best practices for error handling in LangGraph?",
        "How do you implement cycles and loops in LangGraph?",
    ],
    # ... 8 total categories
}
```

**Custom Templates:**
```python
# Add your own templates
custom_templates = {
    "my_domain": [
        "What is {concept} and how does it work?",
        "Compare {method_a} and {method_b} for {use_case}",
    ]
}

# Add to generator
generator.templates.update(custom_templates)
```

#### Quality Control

**Validation Rules:**
1. Minimum question length (20 chars)
2. Minimum answer length (100 chars)
3. Question ends with "?"
4. No placeholder text ({}, [TODO])
5. Answer doesn't just repeat question
6. Valid context extracted

**Quality Scoring (0.0 to 1.0):**
```python
def score_quality(qa_pair: QAPair) -> float:
    """
    Quality factors:
    - Answer length (up to 0.2)
    - Code examples (up to 0.15)
    - Structured formatting (up to 0.2)
    - Technical terms (up to 0.2)
    - Context quality (up to 0.2)
    - Multiple reasoning paths (bonus 0.1)
    """
    score = 0.0

    # Length
    if len(qa_pair.answer) >= 500:
        score += 0.2

    # Code blocks
    if "```" in qa_pair.answer:
        score += 0.15

    # Structure (bullets, numbers, headers)
    if has_structure(qa_pair.answer):
        score += 0.2

    # Technical content
    technical_terms = count_technical_terms(qa_pair.answer)
    score += min(0.2, technical_terms * 0.05)

    # Contexts
    if len(qa_pair.contexts) >= 3:
        score += 0.15

    return min(1.0, score)
```

**Filtering:**
```python
# Generate 1000 pairs
all_pairs = await generator.generate_batch(num_samples=1000)

# Filter by quality
high_quality = generator.filter_by_quality(all_pairs, min_score=0.7)
print(f"High quality pairs: {len(high_quality)} / {len(all_pairs)}")

# Save only high-quality pairs
generator.save_dataset(high_quality, "high_quality_qa.json")
```

### Lecture: Cost Optimization (45 minutes)

#### Token Usage and Cost Estimation

**Cost Analysis:**
```python
# Average costs per Q&A pair
costs = {
    "gpt-4": 0.03,           # $0.03 per Q&A pair
    "gpt-3.5-turbo": 0.002,  # $0.002 per Q&A pair
    "claude-3-opus": 0.015,   # $0.015 per Q&A pair
    "claude-3-sonnet": 0.003, # $0.003 per Q&A pair
}

# For 10,000 Q&A pairs
samples = 10_000
print(f"Cost for {samples:,} Q&A pairs:")
for model, cost in costs.items():
    total = samples * cost
    print(f"  {model}: ${total:.2f}")
```

**Output:**
```
Cost for 10,000 Q&A pairs:
  gpt-4: $300.00
  gpt-3.5-turbo: $20.00
  claude-3-opus: $150.00
  claude-3-sonnet: $30.00
```

**Optimization Strategies:**
1. **Use cheaper models:** GPT-3.5-turbo or Claude Sonnet for bulk generation
2. **Quality filtering:** Generate more, keep best
3. **Template efficiency:** Reuse answers for similar questions
4. **Batch processing:** Reduce API overhead
5. **Caching:** Avoid regenerating similar questions

**Practical Approach:**
```python
# Two-stage approach: Quantity then Quality
# Stage 1: Generate 5,000 pairs with GPT-3.5-turbo
cheap_client = create_client("openai", model="gpt-3.5-turbo")
cheap_generator = SyntheticKnowledgeGenerator(cheap_client)

pairs_stage1 = await cheap_generator.generate_batch(5000)
# Cost: $10

# Stage 2: Filter and enhance top 1,000 with GPT-4
filtered = cheap_generator.filter_by_quality(pairs_stage1, min_score=0.6)
top_1000 = filtered[:1000]

# Enhance with reasoning paths using GPT-4
expensive_client = create_client("openai", model="gpt-4")
for pair in top_1000:
    pair.reasoning_paths = await generate_reasoning_paths(
        expensive_client, pair.question, num_paths=2
    )
# Cost: ~$20

# Total: $30 for 1,000 high-quality pairs with reasoning
```

### Hands-On Exercise: Generate Synthetic Dataset (105 minutes)

**Exercise 9.2: Generate 1,000 Q&A Pairs**

**Objective:** Create a high-quality synthetic dataset for training/evaluation.

**Requirements:**
1. Generate 1,500 raw Q&A pairs
2. Apply quality filtering (min_score=0.5)
3. Target: 1,000+ high-quality pairs
4. Include diverse categories
5. Track costs and statistics
6. Export in LangSmith format

**Template:**
```python
# labs/module_9/exercise_9_2_synthetic_data.py

import asyncio
from training.synthetic_knowledge_generator import SyntheticKnowledgeGenerator
from src.adapters.llm import create_client

async def main():
    # TODO: Initialize LLM client
    llm_client = create_client(
        provider="openai",
        model="gpt-3.5-turbo",  # Cost-effective
        rate_limit_per_minute=60
    )

    # TODO: Create generator
    generator = SyntheticKnowledgeGenerator(
        llm_client=llm_client,
        output_dir="training/synthetic_data",
    )

    # TODO: Generate pairs
    pairs = await generator.generate_batch(
        num_samples=1500,  # Generate more to filter
        categories=[
            "mcts_algorithms",
            "langgraph_workflows",
            "multi_agent_coordination",
            "code_implementation",
        ],
        batch_size=10
    )

    # TODO: Filter by quality
    high_quality = generator.filter_by_quality(pairs, min_score=0.5)

    print(f"Generated {len(pairs)} total pairs")
    print(f"High quality: {len(high_quality)} pairs")

    # TODO: Save datasets
    generator.save_dataset(high_quality, "synthetic_qa_langsmith.json", format="langsmith")
    generator.save_dataset(high_quality, "synthetic_qa_raw.json", format="raw")

    # TODO: Print statistics
    stats = generator.get_statistics()
    print(f"\n=== Generation Statistics ===")
    print(f"Total generated: {stats['total_generated']}")
    print(f"Valid pairs: {stats['valid_pairs']}")
    print(f"Invalid pairs: {stats['invalid_pairs']}")
    print(f"Avg quality score: {stats['avg_quality_score']:.3f}")
    print(f"Total API calls: {stats['api_calls']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Estimated cost: ${stats['total_cost']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria:**
- Generate 1,000+ high-quality pairs
- Average quality score ≥ 0.6
- Diverse coverage across categories
- Cost under $25

**Deliverable:** Synthetic dataset with statistics report

---

## Session 3: Knowledge Graphs (3 hours)

### Pre-Reading (30 minutes)

- Knowledge graph fundamentals
- Neo4j or networkx for graph storage
- Entity and relationship extraction

### Lecture: Knowledge Graph Construction (90 minutes)

#### Why Knowledge Graphs?

**Benefits:**
- **Structured Knowledge:** Entities and relationships
- **Reasoning:** Multi-hop queries and inference
- **Explainability:** Traceable reasoning paths
- **Hybrid Search:** Combine with vector search

**Architecture:**
```
Research Papers / Documents
    ↓
Entity Extraction (NER)
    ↓
Relationship Extraction
    ↓
Knowledge Graph Construction
    ↓
Graph + Vector Hybrid Search
```

**Simple Implementation:**
```python
import networkx as nx
from typing import List, Dict, Any

class SimpleKnowledgeGraph:
    """Simple knowledge graph using NetworkX."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, entity_type: str, properties: dict):
        """Add entity node to graph."""
        self.graph.add_node(
            entity_id,
            entity_type=entity_type,
            **properties
        )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict = None
    ):
        """Add relationship edge to graph."""
        self.graph.add_edge(
            source_id,
            target_id,
            rel_type=rel_type,
            **(properties or {})
        )

    def query_neighbors(self, entity_id: str, rel_type: str = None) -> List[str]:
        """Get neighboring entities."""
        neighbors = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if rel_type is None or data.get('rel_type') == rel_type:
                neighbors.append(target)
        return neighbors

    def query_path(self, source_id: str, target_id: str) -> List[str]:
        """Find shortest path between entities."""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return []

# Usage
kg = SimpleKnowledgeGraph()

# Add entities
kg.add_entity("MCTS", "algorithm", {"name": "Monte Carlo Tree Search"})
kg.add_entity("UCB1", "formula", {"name": "Upper Confidence Bound 1"})
kg.add_entity("AlphaZero", "system", {"name": "AlphaZero"})

# Add relationships
kg.add_relationship("MCTS", "UCB1", "uses_formula")
kg.add_relationship("AlphaZero", "MCTS", "implements")

# Query
formulas = kg.query_neighbors("MCTS", rel_type="uses_formula")
print(f"MCTS uses: {formulas}")  # ['UCB1']

path = kg.query_path("AlphaZero", "UCB1")
print(f"Path from AlphaZero to UCB1: {path}")
# ['AlphaZero', 'MCTS', 'UCB1']
```

**LLM-Based Entity/Relationship Extraction:**
```python
async def extract_entities_relationships(text: str, llm_client) -> dict:
    """Extract entities and relationships using LLM."""
    prompt = f"""Extract entities and relationships from this text.

Text: {text}

Format:
Entities: [entity1, entity2, ...]
Relationships: [entity1 -> relation -> entity2, ...]

Extract:"""

    response = await llm_client.generate(prompt)

    # Parse response (simplified)
    lines = response.text.split('\n')
    entities = []
    relationships = []

    for line in lines:
        if line.startswith("Entities:"):
            entities = [e.strip() for e in line.split(':')[1].split(',')]
        elif line.startswith("Relationships:"):
            # Parse "entity1 -> relation -> entity2"
            rel_text = line.split(':')[1]
            # ... parsing logic

    return {"entities": entities, "relationships": relationships}
```

### Lecture: Hybrid Vector + Graph Search (45 minutes)

**Combining Vector Search and Knowledge Graphs:**

```python
class HybridKnowledgeRetriever:
    """Hybrid retrieval using vectors + knowledge graph."""

    def __init__(self, vector_index, knowledge_graph, embedder):
        self.vector_index = vector_index
        self.kg = knowledge_graph
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve using hybrid approach:
        1. Vector search for initial candidates
        2. Knowledge graph expansion for related entities
        3. Re-rank combined results
        """
        # 1. Vector search
        query_embedding = self.embedder.embed([query])[0]
        vector_results = self.vector_index.query(
            vector=query_embedding,
            top_k=top_k * 2
        )

        # 2. Expand with knowledge graph
        expanded_results = []
        for result in vector_results['matches']:
            entity_id = result['id']

            # Get direct result
            expanded_results.append({
                'id': entity_id,
                'score': result['score'],
                'text': result['metadata']['text'],
                'source': 'vector'
            })

            # Get related entities from KG
            neighbors = self.kg.query_neighbors(entity_id)
            for neighbor_id in neighbors[:3]:  # Top 3 neighbors
                if neighbor_id in self.vector_index:
                    neighbor_doc = self.vector_index.fetch([neighbor_id])
                    expanded_results.append({
                        'id': neighbor_id,
                        'score': result['score'] * 0.7,  # Discounted
                        'text': neighbor_doc['text'],
                        'source': 'graph_expansion',
                        'relation': 'related_to',
                        'parent': entity_id
                    })

        # 3. De-duplicate and sort
        seen_ids = set()
        final_results = []
        for result in sorted(expanded_results, key=lambda x: x['score'], reverse=True):
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                final_results.append(result)
                if len(final_results) >= top_k:
                    break

        return final_results
```

### Hands-On Exercise: Build Knowledge Graph (45 minutes)

**Exercise 9.3: Extract and Query Knowledge Graph**

**Objective:** Build a knowledge graph from research papers.

**Requirements:**
1. Extract entities and relationships from 20+ papers
2. Build knowledge graph
3. Implement hybrid search
4. Query multi-hop relationships

**Deliverable:** Knowledge graph with hybrid retrieval

---

## Session 4: Production Feedback Loop (3.5 hours)

### Pre-Reading (30 minutes)

- [training/continual_learning.py](../../training/continual_learning.py)
- Continual learning principles
- A/B testing best practices

### Lecture: Feedback Collection (90 minutes)

#### Production Feedback Architecture

**Pipeline:**
```
User Query → System Response
    ↓
User Feedback (thumbs up/down, corrections)
    ↓
Feedback Collection (sampling)
    ↓
Quality Filtering
    ↓
Training Sample Creation
    ↓
Incremental Model Update
    ↓
A/B Testing
    ↓
Deployment (if improved)
```

**Implementation:**
```python
from training.continual_learning import FeedbackCollector, FeedbackSample
import time

# Initialize feedback collector
collector = FeedbackCollector({
    "buffer_size": 100000,
    "sample_rate": 0.1,  # Collect 10% of interactions
})

# In your API endpoint
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Generate response
    response = await generate_response(request.query)

    # Return response
    return response

@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackRequest):
    """Collect user feedback on responses."""
    # Create feedback sample
    sample = FeedbackSample(
        sample_id=feedback.response_id,
        input_data={"query": feedback.query},
        model_output=feedback.original_response,
        user_feedback=feedback.rating,  # "positive", "negative", "neutral"
        corrected_output=feedback.correction if feedback.has_correction else None,
        timestamp=time.time(),
        metadata={
            "user_id": feedback.user_id,
            "session_id": feedback.session_id,
        }
    )

    # Add to collector
    collector.add_feedback(sample)

    return {"status": "feedback_recorded"}

# Periodic training data extraction
@app.post("/admin/extract-training-data")
async def extract_training_data():
    """Extract high-quality training samples from feedback."""
    samples = collector.get_training_samples(min_quality=0.5)

    # Save for retraining
    with open("training/feedback_samples.json", "w") as f:
        json.dump(samples, f)

    stats = collector.get_statistics()
    return {
        "status": "extracted",
        "num_samples": len(samples),
        "stats": stats
    }
```

### Lecture: Incremental Training (45 minutes)

#### Elastic Weight Consolidation (EWC)

**Problem:** Catastrophic forgetting when training on new data

**Solution:** EWC regularization preserves important weights

**Implementation:**
```python
from training.continual_learning import IncrementalTrainer

# Initialize trainer
trainer = IncrementalTrainer({
    "retrain_threshold": 1000,  # Retrain after 1K new samples
    "forgetting_prevention": "elastic_weight_consolidation",
    "ewc_lambda": 1000.0,  # EWC regularization strength
})

# Check if retraining needed
if trainer.should_retrain(len(new_samples)):
    print("Retraining triggered!")

    # Compute Fisher Information on old task
    trainer.compute_fisher_information(model, old_dataloader)

    # Incremental update with EWC
    metrics = trainer.incremental_update(
        model=model,
        new_dataloader=new_dataloader,
        old_dataloader=old_dataloader,
        num_epochs=3
    )

    print(f"Update complete. Loss: {metrics['losses'][-1]:.4f}")
    print(f"EWC loss: {metrics['ewc_losses'][-1]:.4f}")
```

### Lecture: A/B Testing Framework (45 minutes)

**Implementation:**
```python
from training.continual_learning import ABTestFramework

# Initialize A/B testing
ab_framework = ABTestFramework({
    "enabled": True,
    "traffic_split": 0.1,  # 10% to treatment
    "min_samples": 1000,
    "confidence_level": 0.95,
})

# Create test: current model vs new model
test_id = ab_framework.create_test(
    test_name="model_v2_test",
    model_a=current_model,
    model_b=new_model,
    metric_fn=lambda inp, out: compute_success_metric(inp, out)
)

# In production
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    # Assign to test group
    group = ab_framework.assign_group(test_id, request.request_id)

    # Use appropriate model
    model = current_model if group == "A" else new_model
    response = await model.generate(request.query)

    # Record result
    success = compute_success_metric(request.query, response)
    ab_framework.record_result(
        test_id=test_id,
        group=group,
        input_data=request.query,
        output=response,
        success_metric=success
    )

    return response

# Check test status
status = ab_framework.get_test_status(test_id)
if status["status"] == "analyzed":
    print(f"Result: {status['result']['recommendation']}")
    print(f"Improvement: {status['result']['improvement']:.2%}")
```

### Hands-On Exercise: Implement Feedback Loop (60 minutes)

**Exercise 9.4: Complete Production Feedback System**

**Objective:** Build end-to-end continual learning pipeline.

**Requirements:**
1. Feedback collection endpoint
2. Training sample extraction
3. Incremental training setup
4. A/B testing framework
5. Monitoring dashboard

**Deliverable:** Production feedback system with monitoring

---

## Module 9 Assessment

### Practical Assessment (100 points)

**Capstone Project: Build Custom Knowledge Base**

**Requirements:**

1. **Research Corpus (25 points)**
   - 100+ papers from arXiv
   - Proper chunking and metadata
   - Integrated with vector DB

2. **Synthetic Data (25 points)**
   - 1,000+ Q&A pairs
   - Quality score ≥ 0.6
   - Multiple categories

3. **Knowledge Graph (20 points)**
   - Entity extraction
   - Relationship mapping
   - Hybrid retrieval

4. **Feedback Loop (30 points)**
   - Collection system
   - Incremental training
   - A/B testing
   - Monitoring

**Minimum Passing:** 70/100

**Submission:** GitHub repository + demo video

---

## Additional Resources

### Code References
- [training/research_corpus_builder.py](../../training/research_corpus_builder.py)
- [training/synthetic_knowledge_generator.py](../../training/synthetic_knowledge_generator.py)
- [training/continual_learning.py](../../training/continual_learning.py)

### Reading
- [arXiv API Documentation](https://info.arxiv.org/help/api/index.html)
- [Synthetic Data for LLMs](https://arxiv.org/abs/2305.14233)
- [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)

---

## What's Next?

**Module 10:** Self-Improving AI Systems
- AlphaZero-style self-play training
- RLHF implementation
- Performance monitoring and evaluation

---

**Module 9 Complete!** 🎯
