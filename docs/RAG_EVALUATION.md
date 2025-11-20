# RAG Evaluation Guide

This guide covers the comprehensive RAG (Retrieval-Augmented Generation) evaluation system integrated into the LangGraph Multi-Agent MCTS Framework.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Metrics](#metrics)
4. [Running Evaluations](#running-evaluations)
5. [Monitoring and Dashboards](#monitoring-and-dashboards)
6. [CI/CD Integration](#cicd-integration)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The RAG evaluation system provides:

- **Automated evaluation** of RAG pipeline performance using RAGAS metrics
- **MCTS comparison** - evaluate RAG with and without MCTS enhancement
- **Real-time monitoring** - Prometheus metrics and Grafana dashboards
- **CI/CD integration** - automated evaluation on PRs and scheduled runs
- **LangSmith datasets** - curated evaluation datasets for MCTS topics

### Key Features

- **Faithfulness**: Measures how grounded answers are in retrieved context
- **Answer Relevancy**: Evaluates relevance of answers to questions
- **Context Precision**: Assesses quality of retrieved documents (low noise)
- **Context Recall**: Measures coverage of relevant information

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Evaluation Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  LangSmith   │──────▶  Evaluation  │──────▶   RAGAS    │ │
│  │   Dataset    │      │    Script    │      │  Metrics  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      ▼                     │       │
│         │              ┌──────────────┐              │       │
│         └─────────────▶│ LangGraph    │◀─────────────┘       │
│                        │  Framework   │                      │
│                        │  (RAG + MCTS)│                      │
│                        └──────────────┘                      │
│                                │                             │
│                ┌───────────────┴───────────────┐             │
│                ▼                               ▼             │
│         ┌──────────────┐               ┌──────────────┐     │
│         │   Weights &  │               │  Prometheus  │     │
│         │    Biases    │               │   Metrics    │     │
│         └──────────────┘               └──────────────┘     │
│                                                │             │
│                                                ▼             │
│                                        ┌──────────────┐     │
│                                        │   Grafana    │     │
│                                        │  Dashboard   │     │
│                                        └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Evaluation Flow

1. **Dataset Loading**: Load evaluation examples from LangSmith
2. **Pipeline Execution**: Process each query through RAG pipeline
3. **Metric Collection**: Record Prometheus metrics (latency, docs retrieved, etc.)
4. **RAGAS Evaluation**: Compute faithfulness, relevance, precision, recall
5. **Results Logging**: Save to CSV and upload to W&B
6. **Comparison**: Compare MCTS vs baseline performance

---

## Metrics

### Prometheus Metrics

#### Operational Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `mcts_rag_queries_total` | Counter | Total RAG queries by status |
| `mcts_rag_retrieval_latency_seconds` | Histogram | RAG retrieval latency |
| `mcts_rag_documents_retrieved` | Histogram | Number of documents retrieved |
| `mcts_rag_relevance_score` | Histogram | Relevance scores of retrieved docs |

#### RAGAS Metrics (Custom)

| Metric | Type | Description |
|--------|------|-------------|
| `ragas_faithfulness_score` | Gauge | RAGAS faithfulness score |
| `ragas_answer_relevancy_score` | Gauge | RAGAS answer relevancy |
| `ragas_context_precision_score` | Gauge | RAGAS context precision |
| `ragas_context_recall_score` | Gauge | RAGAS context recall |

### RAGAS Metrics Explained

#### Faithfulness (0.0 - 1.0)
- **What it measures**: How factually grounded the answer is in the retrieved context
- **High score means**: Answer contains only information present in the context
- **Low score means**: Answer contains hallucinations or unsupported claims
- **Target**: ≥ 0.80

#### Answer Relevancy (0.0 - 1.0)
- **What it measures**: How relevant the answer is to the question
- **High score means**: Answer directly addresses the question
- **Low score means**: Answer is off-topic or incomplete
- **Target**: ≥ 0.75

#### Context Precision (0.0 - 1.0)
- **What it measures**: Precision of retrieved documents (signal-to-noise ratio)
- **High score means**: Most retrieved docs are relevant to the question
- **Low score means**: Many irrelevant documents retrieved
- **Target**: ≥ 0.80

#### Context Recall (0.0 - 1.0)
- **What it measures**: Coverage of relevant information in retrieved docs
- **High score means**: All necessary information was retrieved
- **Low score means**: Missing relevant documents
- **Target**: ≥ 0.80

---

## Running Evaluations

### Prerequisites

```bash
# Set environment variables
export LANGSMITH_API_KEY=your_key_here
export WANDB_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here  # or ANTHROPIC_API_KEY
export LLM_PROVIDER=openai  # or anthropic, lmstudio
```

### Create Evaluation Dataset

```bash
# Create RAG evaluation dataset in LangSmith
python scripts/create_rag_eval_datasets.py
```

This creates a dataset with 25 MCTS-related Q&A examples covering:
- Basic MCTS concepts (selection, expansion, simulation, backpropagation)
- UCB1 and exploration/exploitation
- AlphaZero architecture
- Advanced techniques (RAVE, progressive widening, virtual loss)

### Run Baseline Evaluation

```bash
# Evaluate RAG without MCTS
python scripts/evaluate_rag.py \
  --dataset rag-eval-dataset \
  --limit 50 \
  --mcts-enabled false \
  --output results/baseline_eval.csv
```

### Run MCTS-Enhanced Evaluation

```bash
# Evaluate RAG with MCTS
python scripts/evaluate_rag.py \
  --dataset rag-eval-dataset \
  --limit 50 \
  --mcts-enabled true \
  --output results/mcts_eval.csv
```

### Compare Results

```bash
# View summary statistics
python -c "
import pandas as pd
baseline = pd.read_csv('results/baseline_eval.csv')
mcts = pd.read_csv('results/mcts_eval.csv')

print('=== Baseline (no MCTS) ===')
print(baseline[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].describe())

print('\n=== MCTS-Enhanced ===')
print(mcts[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].describe())

print('\n=== Improvement ===')
for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
    improvement = mcts[metric].mean() - baseline[metric].mean()
    print(f'{metric}: {improvement:+.3f} ({improvement/baseline[metric].mean()*100:+.1f}%)')
"
```

---

## Monitoring and Dashboards

### Grafana Dashboard

Access the RAG Evaluation dashboard at: `http://localhost:3000/d/rag-evaluation`

**Dashboard Panels**:

1. **Query Rate**: RAG queries per second
2. **Retrieval Latency**: P50, P95, P99 latency percentiles
3. **Documents Retrieved**: Distribution of documents per query
4. **Relevance Scores**: Distribution of document relevance scores
5. **Success Rate**: Percentage of successful RAG queries
6. **RAGAS Metrics**: Time-series of faithfulness, relevancy, precision, recall
7. **MCTS Comparison**: Side-by-side comparison of MCTS vs baseline

### Prometheus Alerts

The system includes the following alerts (defined in `monitoring/prometheus_alerts_rag.yml`):

#### Warning Alerts
- **HighRAGRetrievalLatency**: P95 latency > 1.0s for 5 minutes
- **LowRAGSuccessRate**: Success rate < 90% for 5 minutes
- **LowRAGRelevanceScore**: Average relevance < 0.60 for 10 minutes
- **TooFewDocumentsRetrieved**: Avg docs < 1.0 for 10 minutes
- **LowRAGASFaithfulness**: Faithfulness < 0.70 for 5 minutes
- **LowRAGASAnswerRelevancy**: Answer relevancy < 0.70 for 5 minutes
- **LowRAGASContextPrecision**: Context precision < 0.70 for 5 minutes
- **LowRAGASContextRecall**: Context recall < 0.70 for 5 minutes

#### Critical Alerts
- **CriticalRAGRetrievalLatency**: P95 latency > 2.0s for 3 minutes
- **CriticalRAGSuccessRate**: Success rate < 75% for 3 minutes

### View Metrics in Prometheus

```bash
# Query RAG metrics
curl 'http://localhost:9090/api/v1/query?query=rate(mcts_rag_queries_total[5m])'

# Query RAGAS metrics
curl 'http://localhost:9090/api/v1/query?query=ragas_faithfulness_score'
```

---

## CI/CD Integration

### Triggering RAG Evaluation

The RAG evaluation runs automatically on:

1. **Pull Requests** to `main` or `develop` branches
2. **Manual workflow dispatch** via GitHub Actions UI
3. **Scheduled runs** (if configured with cron schedule)

### Manual Trigger

```bash
# Via GitHub CLI
gh workflow run ci.yml --ref feature/my-branch

# Via GitHub UI
# Go to Actions → CI Pipeline → Run workflow
```

### Required Secrets

Configure these in GitHub Settings → Secrets:

- `LANGSMITH_API_KEY`: LangSmith API key for dataset access
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI)
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Claude)
- `LLM_PROVIDER`: LLM provider to use (openai, anthropic, or lmstudio)

### CI/CD Workflow Steps

1. **Setup**: Install dependencies and create results directory
2. **Create Dataset**: Initialize RAG evaluation dataset in LangSmith
3. **Baseline Evaluation**: Run RAG evaluation without MCTS (limit: 50 examples)
4. **MCTS Evaluation**: Run RAG evaluation with MCTS enabled
5. **Compare Results**: Display side-by-side comparison
6. **Upload Artifacts**: Save CSV results as GitHub artifacts
7. **Upload to Tracking**: Results automatically uploaded to W&B and LangSmith

### Viewing Results

- **GitHub Artifacts**: Download CSV files from workflow run
- **Weights & Biases**: https://wandb.ai → your project → rag-eval runs
- **LangSmith**: https://smith.langchain.com → datasets → rag-eval-dataset

---

## Troubleshooting

### Common Issues

#### 1. Dataset Not Found

**Error**: `Dataset 'rag-eval-dataset' not found in LangSmith`

**Solution**:
```bash
# Create the dataset
python scripts/create_rag_eval_datasets.py

# Verify it was created
python -c "
from langsmith import Client
client = Client()
datasets = list(client.list_datasets())
print([d.name for d in datasets])
"
```

#### 2. Missing API Keys

**Error**: `LANGSMITH_API_KEY not set`

**Solution**:
```bash
# Set in environment
export LANGSMITH_API_KEY=your_key_here

# Or create .env file
echo "LANGSMITH_API_KEY=your_key_here" >> .env
echo "WANDB_API_KEY=your_key_here" >> .env
```

#### 3. RAGAS Import Error

**Error**: `ImportError: No module named 'ragas'`

**Solution**:
```bash
pip install ragas>=0.1.0
```

#### 4. LLM Provider Connection Issues

**Error**: `Failed to initialize RAG pipeline`

**Solution**:
```bash
# For LM Studio
# 1. Start LM Studio server on localhost:1234
# 2. Load a model (e.g., llama-2-7b)

# For OpenAI
export OPENAI_API_KEY=your_key_here
export LLM_PROVIDER=openai

# For Anthropic
export ANTHROPIC_API_KEY=your_key_here
export LLM_PROVIDER=anthropic
```

#### 5. Low RAGAS Scores

**Problem**: Getting consistently low scores (< 0.5)

**Possible causes and solutions**:

1. **Poor retrieval quality**:
   - Check vector store is populated with relevant documents
   - Increase number of retrieved documents (`top_k_retrieval`)
   - Verify embeddings model is appropriate

2. **LLM quality issues**:
   - Try a more capable model (e.g., GPT-4 instead of GPT-3.5)
   - Adjust temperature and sampling parameters
   - Review prompt templates

3. **Dataset mismatch**:
   - Ensure evaluation dataset matches your document corpus
   - Check that ground truth answers are accurate
   - Verify context snippets are relevant

### Debugging Tips

#### Enable Debug Logging

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Run evaluation
python scripts/evaluate_rag.py --dataset rag-eval-dataset --limit 10
```

#### Inspect Individual Examples

```python
import pandas as pd

# Load results
df = pd.read_csv('results/baseline_eval.csv')

# Find low-scoring examples
low_faithfulness = df[df['faithfulness'] < 0.5]
print(low_faithfulness[['question', 'answer', 'faithfulness']])

# Inspect specific example
example = df.iloc[0]
print(f"Question: {example['question']}")
print(f"Answer: {example['answer']}")
print(f"Contexts: {example['contexts']}")
print(f"Scores: F={example['faithfulness']:.2f}, R={example['answer_relevancy']:.2f}")
```

#### Test RAG Pipeline Manually

```python
import asyncio
from src.config.settings import get_settings
from src.adapters.llm import create_client
from src.framework.graph import LangGraphMultiAgentFramework

async def test_rag():
    settings = get_settings()

    model_adapter = create_client(
        provider=settings.LLM_PROVIDER,
        model=settings.DEFAULT_MODEL,
    )

    framework = LangGraphMultiAgentFramework(
        model_adapter=model_adapter,
        mcts_iterations=0,  # Disable MCTS for testing
    )

    result = await framework.process(
        query="What is Monte Carlo Tree Search?",
        use_rag=True,
        use_mcts=False,
    )

    print(f"Response: {result['response']}")
    print(f"Retrieved docs: {len(result['metadata']['retrieved_docs'])}")
    for i, doc in enumerate(result['metadata']['retrieved_docs']):
        print(f"\nDoc {i+1}:")
        print(f"  Content: {doc['content'][:100]}...")
        print(f"  Score: {doc['metadata'].get('score', 'N/A')}")

asyncio.run(test_rag())
```

### Performance Optimization

#### Reduce Evaluation Time

```bash
# Use smaller sample size
python scripts/evaluate_rag.py --dataset rag-eval-dataset --limit 10

# Disable W&B logging
export WANDB_MODE=disabled
```

#### Improve Retrieval Quality

1. **Increase top-k**: Retrieve more documents for better coverage
2. **Reranking**: Add a reranking step after initial retrieval
3. **Hybrid search**: Combine dense and sparse retrieval
4. **Better embeddings**: Use domain-specific embedding models

#### Optimize Prometheus Metrics

```bash
# Reduce scrape interval for lower overhead
# Edit monitoring/prometheus.yml
scrape_interval: 30s  # instead of 10s
```

---

## Best Practices

### 1. Regular Evaluation

- Run evaluations **weekly** on production datasets
- Compare trends over time in W&B
- Set up alerts for metric degradation

### 2. Dataset Maintenance

- Keep evaluation dataset **up-to-date** with new topics
- Include **diverse question types** (factual, reasoning, multi-hop)
- Maintain **high-quality ground truth** answers

### 3. Monitoring

- Monitor **all four RAGAS metrics** (not just one)
- Track **retrieval latency** for performance regression
- Alert on **success rate drops**

### 4. A/B Testing

- Compare **MCTS vs baseline** regularly
- Test different **retrieval strategies**
- Experiment with **LLM models**

### 5. Documentation

- Document **expected metric ranges**
- Record **configuration changes**
- Maintain **troubleshooting runbook**

---

## Additional Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation)
- [Weights & Biases Guides](https://docs.wandb.ai/)
- [Prometheus Alerting](https://prometheus.io/docs/alerting/latest/overview/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review GitHub Issues
3. Contact the team via Slack/Email
4. File a new issue with:
   - Error logs
   - Configuration details
   - Steps to reproduce
