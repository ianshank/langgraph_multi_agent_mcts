# Advanced Training Assessment
## Modules 8-10: RAG, Knowledge Engineering, and Self-Improving Systems

**Purpose:** Comprehensive assessment for Level 3 (Expert) certification
**Format:** Module quizzes + Practical projects + Capstone
**Passing Score:** 70% overall, minimum 60% per module

---

## Table of Contents

1. [Module 8 Assessment: Advanced RAG](#module-8-assessment)
2. [Module 9 Assessment: Knowledge Engineering](#module-9-assessment)
3. [Module 10 Assessment: Self-Improving AI](#module-10-assessment)
4. [Capstone Project](#capstone-project)
5. [Certification Criteria](#certification-criteria)

---

## Module 8 Assessment: Advanced RAG

### Quiz 8: Advanced Retrieval Techniques (30 minutes, 20 points)

**Instructions:** Select the best answer for each question.

**1. Matryoshka Embeddings (2 points)**

Matryoshka embeddings allow you to:
- A) Train multiple embedding models simultaneously
- B) Use a single model at multiple dimension sizes
- C) Combine embeddings from different providers
- D) Encrypt embeddings for secure storage

<details>
<summary>Answer</summary>
B) Use a single model at multiple dimension sizes - Matryoshka embeddings can be truncated to smaller dimensions while maintaining reasonable performance.
</details>

---

**2. Hybrid Search Alpha Parameter (2 points)**

In hybrid retrieval with `score = alpha * dense + (1-alpha) * sparse`, what does alpha=0.7 mean?
- A) 70% sparse, 30% dense
- B) 70% dense, 30% sparse
- C) 70% accuracy threshold
- D) 70% of documents should be retrieved

<details>
<summary>Answer</summary>
B) 70% dense, 30% sparse - Alpha weights the dense (vector) score higher.
</details>

---

**3. Cross-Encoder vs Bi-Encoder (2 points)**

Why use cross-encoders for re-ranking instead of initial retrieval?
- A) Cross-encoders are faster than bi-encoders
- B) Cross-encoders scale better to millions of documents
- C) Cross-encoders are more accurate but slower
- D) Cross-encoders don't require training

<details>
<summary>Answer</summary>
C) Cross-encoders are more accurate but slower - They process query+document together (O(n) comparisons) vs bi-encoders (O(1) after pre-computation).
</details>

---

**4. Embedding Cache Benefits (2 points)**

Which statement about embedding caching is FALSE?
- A) Reduces API costs by avoiding duplicate embeddings
- B) Improves latency for repeated queries
- C) Guarantees identical embeddings for same text
- D) Automatically updates when embedding models change

<details>
<summary>Answer</summary>
D) Automatically updates when embedding models change - Caches must be invalidated manually when models change.
</details>

---

**5. nDCG@10 Metric (2 points)**

nDCG@10 = 0.85 means:
- A) 85% of top-10 results are relevant
- B) Ranking quality is 85% of ideal ranking in top-10
- C) 85% precision in top-10 results
- D) 8.5 out of 10 results are correct

<details>
<summary>Answer</summary>
B) Ranking quality is 85% of ideal ranking in top-10 - nDCG measures ranking quality relative to perfect ranking.
</details>

---

**6. BM25 Parameters (2 points)**

In BM25, the parameter `b` controls:
- A) Boost factor for rare terms
- B) Document length normalization
- C) Query term weighting
- D) Number of results returned

<details>
<summary>Answer</summary>
B) Document length normalization - b=0 means no normalization, b=1 means full normalization.
</details>

---

**7. Voyage AI vs OpenAI Embeddings (2 points)**

According to MTEB leaderboard (2024), Voyage-large-2-instruct:
- A) Ranks lower than OpenAI text-embedding-3-large
- B) Ranks higher than OpenAI text-embedding-3-large
- C) Has the same performance as OpenAI
- D) Only works for code embeddings

<details>
<summary>Answer</summary>
B) Ranks higher than OpenAI text-embedding-3-large - Voyage-large-2 typically ranks #1 on MTEB.
</details>

---

**8. Ensemble Embeddings (2 points)**

Concatenating embeddings from two 1024-dim models results in:
- A) Better quality with 1024-dim output
- B) Better quality with 2048-dim output
- C) Same quality with averaged 1024-dim output
- D) Worse quality due to redundancy

<details>
<summary>Answer</summary>
B) Better quality with 2048-dim output - Concatenation preserves all information from both models but doubles dimension.
</details>

---

**9. Recall@100 vs Precision@10 (2 points)**

Which metric is more important for RAG initial retrieval?
- A) Precision@10 - want only relevant docs
- B) Recall@100 - want to capture all relevant docs
- C) Both are equally important
- D) Neither applies to RAG

<details>
<summary>Answer</summary>
B) Recall@100 - Initial retrieval should cast wide net; re-ranking improves precision later.
</details>

---

**10. Production RAG Latency (2 points)**

For production RAG, p95 latency under 500ms means:
- A) Average latency is 500ms
- B) 95% of requests complete in <500ms
- C) 5% of requests fail
- D) Maximum latency is 500ms

<details>
<summary>Answer</summary>
B) 95% of requests complete in <500ms - p95 is the 95th percentile latency.
</details>

---

### Practical Project 8: Advanced RAG System (80 points)

**Deliverable:** Complete RAG system with hybrid retrieval

**Requirements:**

**1. Multi-Model Embedding System (20 points)**
- [ ] Primary embedder with fallback (5 points)
- [ ] Caching enabled with >80% hit rate (5 points)
- [ ] Performance benchmarks on 3+ models (5 points)
- [ ] Documentation of model selection (5 points)

**2. Hybrid Retrieval Pipeline (25 points)**
- [ ] Dense retrieval implemented (8 points)
- [ ] Sparse (BM25) retrieval implemented (8 points)
- [ ] Score fusion with tuned alpha (6 points)
- [ ] Evaluation metrics (nDCG@10, Recall@100) (3 points)

**3. Re-Ranking Layer (15 points)**
- [ ] Cross-encoder implementation (8 points)
- [ ] Improved quality vs base retrieval (4 points)
- [ ] Performance profiling (3 points)

**4. Production Deployment (20 points)**
- [ ] FastAPI with /query endpoint (8 points)
- [ ] Prometheus metrics (6 points)
- [ ] Health checks and error handling (4 points)
- [ ] Documentation and examples (2 points)

**Grading Rubric:**

| Score | Criteria |
|-------|----------|
| 90-100 | All requirements met, excellent performance, comprehensive docs |
| 80-89 | All requirements met, good performance, adequate docs |
| 70-79 | Most requirements met, acceptable performance |
| <70 | Missing requirements or poor performance |

**Submission:**
- GitHub repository URL
- README with setup instructions
- Performance report with metrics

---

## Module 9 Assessment: Knowledge Engineering

### Quiz 9: Knowledge Acquisition & Management (30 minutes, 20 points)

**1. arXiv API Rate Limits (2 points)**

What is the recommended delay between arXiv API requests?
- A) No delay needed
- B) 1 second
- C) 3 seconds
- D) 10 seconds

<details>
<summary>Answer</summary>
C) 3 seconds - arXiv recommends at least 3 seconds between requests to avoid rate limiting.
</details>

---

**2. Synthetic Data Quality (2 points)**

Which factor does NOT contribute to synthetic Q&A quality score?
- A) Answer length
- B) Presence of code examples
- C) Question difficulty
- D) Structured formatting

<details>
<summary>Answer</summary>
C) Question difficulty - Quality score is based on answer characteristics, not question difficulty.
</details>

---

**3. Knowledge Graph Benefits (2 points)**

Knowledge graphs improve RAG by:
- A) Replacing vector search entirely
- B) Providing structured relationships for multi-hop queries
- C) Reducing storage costs
- D) Eliminating need for embeddings

<details>
<summary>Answer</summary>
B) Providing structured relationships for multi-hop queries - KGs complement vector search with explicit relationships.
</details>

---

**4. Elastic Weight Consolidation (2 points)**

EWC prevents catastrophic forgetting by:
- A) Freezing all model weights
- B) Penalizing changes to important weights
- C) Training only new layers
- D) Using larger learning rates

<details>
<summary>Answer</summary>
B) Penalizing changes to important weights - EWC adds regularization based on Fisher Information.
</details>

---

**5. Feedback Collection Sampling (2 points)**

If sample_rate=0.1, this means:
- A) Sample 10% of user interactions
- B) Sample every 10th interaction
- C) Collect feedback for 10 minutes
- D) Store 10% of responses

<details>
<summary>Answer</summary>
A) Sample 10% of user interactions - Random sampling at 10% rate balances data collection and storage.
</details>

---

**6. A/B Test Statistical Significance (2 points)**

With confidence_level=0.95, p_value=0.03, the result is:
- A) Not statistically significant
- B) Statistically significant
- C) Inconclusive
- D) Invalid test

<details>
<summary>Answer</summary>
B) Statistically significant - p_value (0.03) < significance level (1 - 0.95 = 0.05).
</details>

---

**7. Research Paper Chunking (2 points)**

What is the recommended chunk size for research papers?
- A) 128 tokens
- B) 256 tokens
- C) 512 tokens
- D) 1024 tokens

<details>
<summary>Answer</summary>
C) 512 tokens - Balances context size and retrieval granularity for academic papers.
</details>

---

**8. Synthetic Data Cost Optimization (2 points)**

To generate 10,000 Q&A pairs cost-effectively, best approach is:
- A) Use GPT-4 for all pairs
- B) Use GPT-3.5-turbo, filter by quality, enhance top pairs with GPT-4
- C) Use only free models
- D) Generate manually

<details>
<summary>Answer</summary>
B) Two-stage approach - Cheap model for bulk, expensive model for top quality. Balances cost and quality.
</details>

---

**9. Drift Detection (2 points)**

Kolmogorov-Smirnov test detects:
- A) Model accuracy changes
- B) Distribution changes in input data
- C) API latency increases
- D) Cache hit rate degradation

<details>
<summary>Answer</summary>
B) Distribution changes in input data - K-S test compares distributions to detect data drift.
</details>

---

**10. Continual Learning Threshold (2 points)**

If retrain_threshold=1000, retraining triggers when:
- A) Model accuracy drops below 1000
- B) 1000 new training samples accumulated
- C) 1000 users provide feedback
- D) System runs for 1000 hours

<details>
<summary>Answer</summary>
B) 1000 new training samples accumulated - Threshold counts accumulated samples since last training.
</details>

---

### Practical Project 9: Knowledge Base Construction (80 points)

**Deliverable:** Complete knowledge base with continual learning

**Requirements:**

**1. Research Corpus (20 points)**
- [ ] 100+ papers from arXiv (8 points)
- [ ] Proper chunking and metadata extraction (6 points)
- [ ] Integration with vector database (4 points)
- [ ] Statistics report (2 points)

**2. Synthetic Training Data (20 points)**
- [ ] 1,000+ Q&A pairs generated (8 points)
- [ ] Average quality score ≥ 0.6 (6 points)
- [ ] Multiple categories covered (4 points)
- [ ] LangSmith format export (2 points)

**3. Knowledge Graph (20 points)**
- [ ] Entity extraction from papers (8 points)
- [ ] Relationship mapping (6 points)
- [ ] Hybrid vector+graph retrieval (4 points)
- [ ] Query examples (2 points)

**4. Feedback Loop (20 points)**
- [ ] Collection system implemented (6 points)
- [ ] Incremental training with EWC (6 points)
- [ ] A/B testing framework (6 points)
- [ ] Monitoring dashboard (2 points)

**Submission:**
- GitHub repository
- Knowledge base statistics
- Sample queries with results
- Continual learning report

---

## Module 10 Assessment: Self-Improving AI

### Quiz 10: Self-Play and RLHF (30 minutes, 20 points)

**1. AlphaZero Training Loop (2 points)**

In AlphaZero, MCTS visit counts are used as:
- A) Value targets
- B) Improved policy targets
- C) Exploration bonuses
- D) Loss function weights

<details>
<summary>Answer</summary>
B) Improved policy targets - Visit count distribution represents better policy than raw network output.
</details>

---

**2. Self-Play Episode Structure (2 points)**

Which is NOT part of a self-play episode?
- A) Initial state
- B) Actions and states
- C) MCTS search trees
- D) Human expert moves

<details>
<summary>Answer</summary>
D) Human expert moves - Self-play learns without human data.
</details>

---

**3. Policy Loss vs Value Loss (2 points)**

In AlphaZero training, value loss measures:
- A) Accuracy of action selection
- B) Accuracy of outcome prediction
- C) MCTS search efficiency
- D) Training iteration speed

<details>
<summary>Answer</summary>
B) Accuracy of outcome prediction - Value network predicts final outcome/reward.
</details>

---

**4. Replay Buffer Purpose (2 points)**

Experience replay buffers help by:
- A) Storing human demonstrations
- B) Breaking correlation between consecutive samples
- C) Reducing memory usage
- D) Accelerating MCTS search

<details>
<summary>Answer</summary>
B) Breaking correlation between consecutive samples - Replay breaks temporal correlation for more stable training.
</details>

---

**5. PUCT Formula (2 points)**

PUCT extends UCB1 by adding:
- A) Prior policy from neural network
- B) Temporal difference updates
- C) Human preference weights
- D) Beam search expansion

<details>
<summary>Answer</summary>
A) Prior policy from neural network - P term in PUCT = Q + U where U incorporates prior P.
</details>

---

**6. RLHF Stages (2 points)**

Correct order of RLHF stages:
- A) Reward model → SFT → PPO
- B) PPO → Reward model → SFT
- C) SFT → Reward model → PPO
- D) SFT → PPO → Reward model

<details>
<summary>Answer</summary>
C) SFT → Reward model → PPO - Standard RLHF pipeline from InstructGPT.
</details>

---

**7. DPO vs RLHF (2 points)**

DPO simplifies RLHF by:
- A) Using larger models
- B) Eliminating separate reward model
- C) Requiring less data
- D) Training faster

<details>
<summary>Answer</summary>
B) Eliminating separate reward model - DPO optimizes directly from preferences.
</details>

---

**8. A/B Test Sample Size (2 points)**

If min_samples=1000 and traffic_split=0.1, minimum total traffic needed:
- A) 100 requests
- B) 1,000 requests
- C) 10,000 requests
- D) 11,000 requests

<details>
<summary>Answer</summary>
D) 11,000 requests - Need 1000 in control (90%) and 1000*0.1/0.9 ≈ 111 in treatment, total ~11,000.
</details>

---

**9. Episode Success Rate (2 points)**

Success rate = 0.35 means:
- A) 35% accuracy
- B) 35% of episodes solved task correctly
- C) 3.5/10 quality score
- D) 35 successful episodes

<details>
<summary>Answer</summary>
B) 35% of episodes solved task correctly - Fraction of episodes with outcome="success".
</details>

---

**10. Model Evaluation Metric (2 points)**

To decide if new model is better than old:
- A) Training loss is lower
- B) New model wins >55% in head-to-head
- C) Inference is faster
- D) Model size is smaller

<details>
<summary>Answer</summary>
B) New model wins >55% in head-to-head - Direct comparison on same tasks is most reliable.
</details>

---

### Practical Project 10: Self-Improving System (80 points)

**Deliverable:** Complete self-play training system

**Requirements:**

**1. Episode Generation (20 points)**
- [ ] Task generators (math, code, reasoning) (6 points)
- [ ] 1,000+ episodes generated (8 points)
- [ ] Success rate >20% (4 points)
- [ ] Complete trace capture (2 points)

**2. Training Loop (25 points)**
- [ ] Policy example extraction (8 points)
- [ ] Value example extraction (6 points)
- [ ] Model training (5+ iterations) (8 points)
- [ ] Checkpointing (3 points)

**3. Evaluation Framework (20 points)**
- [ ] Benchmark suite (6 points)
- [ ] Model comparison (6 points)
- [ ] A/B testing (6 points)
- [ ] Statistical analysis (2 points)

**4. Production System (15 points)**
- [ ] Monitoring dashboard (6 points)
- [ ] Performance metrics (4 points)
- [ ] Documentation (3 points)
- [ ] Demo video (2 points)

**Submission:**
- GitHub repository
- Training logs and metrics
- Demo video (5-10 min)
- Technical report

---

## Capstone Project (100 points)

### End-to-End Self-Improving Knowledge System

**Objective:** Integrate Modules 8, 9, and 10 into production system.

**Requirements:**

**1. Knowledge Base (25 points)**
- [ ] Research corpus (100+ papers) (8 points)
- [ ] Synthetic Q&A (1,000+ pairs) (8 points)
- [ ] Hybrid search with re-ranking (6 points)
- [ ] Knowledge graph integration (3 points)

**2. Self-Improvement (25 points)**
- [ ] Self-play episode generation (8 points)
- [ ] Training loop (5+ iterations) (10 points)
- [ ] Quality improvement demonstrated (5 points)
- [ ] A/B testing (2 points)

**3. Production Deployment (25 points)**
- [ ] FastAPI with full endpoints (10 points)
- [ ] Prometheus + Grafana monitoring (8 points)
- [ ] Docker deployment (5 points)
- [ ] Health checks and error handling (2 points)

**4. Continual Learning (25 points)**
- [ ] Feedback collection system (8 points)
- [ ] Drift detection (6 points)
- [ ] Incremental retraining (8 points)
- [ ] Performance tracking over time (3 points)

**Deliverables:**
1. **GitHub Repository** - Complete source code
2. **Demo Video** - 10-15 minute system walkthrough
3. **Technical Documentation** - Architecture, setup, usage
4. **Performance Report** - Metrics, benchmarks, improvements

**Timeline:** 2 weeks

**Grading Rubric:**

| Score | Criteria |
|-------|----------|
| 90-100 | Exceptional - All requirements exceeded, production-ready |
| 80-89 | Excellent - All requirements met, well-documented |
| 70-79 | Good - Most requirements met, functional system |
| 60-69 | Acceptable - Basic requirements met |
| <60 | Incomplete - Missing major components |

---

## Certification Criteria

### Level 3: Expert Developer

**Requirements:**
- Complete Modules 8, 9, 10
- Pass all module quizzes (≥70%)
- Complete all practical projects (≥70%)
- Complete capstone project (≥70%)
- Overall score ≥70%

**Certification Includes:**
- Digital certificate
- LinkedIn badge
- GitHub badge
- Portfolio showcase

**Score Calculation:**
```
Module 8: Quiz (20%) + Project (80%) = 100 points
Module 9: Quiz (20%) + Project (80%) = 100 points
Module 10: Quiz (20%) + Project (80%) = 100 points
Capstone: 100 points

Total: 400 points
Passing: 280 points (70%)
```

---

## Submission Guidelines

### GitHub Repository Structure

```
project-name/
├── README.md                  # Project overview
├── docs/
│   ├── architecture.md       # System architecture
│   ├── setup.md             # Setup instructions
│   └── usage.md             # Usage guide
├── src/
│   ├── embeddings/          # Embedding implementations
│   ├── retrieval/           # Retrieval systems
│   ├── knowledge/           # Knowledge management
│   └── self_play/           # Self-play training
├── tests/
│   └── test_*.py            # Unit tests
├── configs/
│   └── config.yaml          # Configuration
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
└── requirements.txt
```

### Demo Video Requirements

**Content:**
1. Introduction (1 min) - Project overview
2. System Architecture (2 min) - Component walkthrough
3. Live Demo (5-7 min) - Core functionality
4. Results & Metrics (2-3 min) - Performance data
5. Conclusion (1 min) - Key achievements

**Technical:**
- Format: MP4, 1080p
- Length: 10-15 minutes
- Upload: YouTube (unlisted) or Loom

### Report Template

```markdown
# Project Title

## Executive Summary
Brief overview (1 paragraph)

## Architecture
System design and components

## Implementation
Key technical decisions

## Results
### Performance Metrics
- Latency: X ms (p95)
- Accuracy: Y%
- Cost: $Z/1K queries

### Improvements
Before vs After metrics

## Challenges & Solutions
What problems encountered and how solved

## Future Work
Potential improvements

## Conclusion
Key learnings
```

---

## Assessment Schedule

**Week 1-2:** Module 8 (Quiz + Project)
**Week 3-4:** Module 9 (Quiz + Project)
**Week 5-6:** Module 10 (Quiz + Project)
**Week 7-8:** Capstone Project
**Week 9:** Final review and certification

---

## Support & Resources

### Office Hours
- Schedule: TBD
- Format: Virtual (Zoom/Google Meet)
- Topics: Technical questions, project reviews

### Peer Review
- Exchange projects with peers
- Provide constructive feedback
- Learn from different approaches

### Resources
- [Troubleshooting Guide](TROUBLESHOOTING_ADVANCED.md)
- [Quick Reference](QUICK_REFERENCE_ADVANCED.md)
- [Code Examples](../../labs/)

---

**Good luck on your certification journey!** 🎓
