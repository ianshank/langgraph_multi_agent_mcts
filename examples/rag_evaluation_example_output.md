# RAG Evaluation Example Output

This document shows example outputs from the RAG evaluation system.

## Evaluation Run Output

```
$ python scripts/evaluate_rag.py --dataset rag-eval-dataset --limit 10 --mcts-enabled false

2025-11-19 14:30:15 - evaluate_rag - INFO - === RAG Evaluation Configuration ===
2025-11-19 14:30:15 - evaluate_rag - INFO - Dataset: rag-eval-dataset
2025-11-19 14:30:15 - evaluate_rag - INFO - Limit: 10
2025-11-19 14:30:15 - evaluate_rag - INFO - MCTS Enabled: False
2025-11-19 14:30:15 - evaluate_rag - INFO - MCTS Implementation: baseline
2025-11-19 14:30:15 - evaluate_rag - INFO - =====================================

2025-11-19 14:30:16 - evaluate_rag - INFO - LangSmith client initialized for project: langgraph-mcts
2025-11-19 14:30:17 - evaluate_rag - INFO - Loading dataset 'rag-eval-dataset'...
2025-11-19 14:30:18 - evaluate_rag - INFO - Loaded 10 examples from dataset 'rag-eval-dataset'

2025-11-19 14:30:19 - evaluate_rag - INFO - Evaluating 10 examples with ragas (mcts=False)
2025-11-19 14:30:19 - evaluate_rag - INFO - Initializing RAG pipeline...

2025-11-19 14:30:20 - evaluate_rag - INFO - Processing example 1/10: What is Monte Carlo Tree Search?...
2025-11-19 14:30:25 - evaluate_rag - INFO - Processing example 2/10: How does UCB1 balance exploration and exploitati...
2025-11-19 14:30:30 - evaluate_rag - INFO - Processing example 3/10: What is the difference between MCTS and minimax...
...
2025-11-19 14:32:15 - evaluate_rag - INFO - Processing example 10/10: What is temperature in the context of AlphaZer...

2025-11-19 14:32:20 - evaluate_rag - INFO - Running ragas evaluation...
2025-11-19 14:32:45 - evaluate_rag - INFO - Evaluation complete!

2025-11-19 14:32:45 - evaluate_rag - INFO -
=== Evaluation Results ===
2025-11-19 14:32:45 - evaluate_rag - INFO - Faithfulness: 0.842
2025-11-19 14:32:45 - evaluate_rag - INFO - Answer Relevancy: 0.789
2025-11-19 14:32:45 - evaluate_rag - INFO - Context Precision: 0.856
2025-11-19 14:32:45 - evaluate_rag - INFO - Context Recall: 0.812
2025-11-19 14:32:45 - evaluate_rag - INFO - ==========================

2025-11-19 14:32:50 - evaluate_rag - INFO - Results logged to W&B: https://wandb.ai/...
2025-11-19 14:32:51 - evaluate_rag - INFO - Results saved to results/baseline_eval.csv
```

## CSV Output Format

**File: results/baseline_eval.csv**

```csv
question,answer,contexts,ground_truth,faithfulness,answer_relevancy,context_precision,context_recall
"What is Monte Carlo Tree Search?","Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for decision-making. It combines tree search with random sampling through four phases: Selection, Expansion, Simulation, and Backpropagation. MCTS is particularly effective for games with large state spaces.","['Monte Carlo Tree Search (MCTS) is a heuristic search algorithm for decision-making problems...', 'MCTS operates by building a search tree incrementally...']","Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used for decision-making that combines tree search with random sampling.",0.92,0.88,0.95,0.87
"How does UCB1 balance exploration and exploitation in MCTS?","UCB1 uses the formula UCB1 = Q/N + C * sqrt(ln(N_parent)/N) to balance exploration and exploitation. The first term favors high-reward nodes (exploitation) while the second term encourages visiting less-explored nodes (exploration).","['UCB1 (Upper Confidence Bound 1) is a key formula in MCTS...', 'The UCB1 formula is: UCB1 = Q/N + C * sqrt(ln(N_parent)/N)...']","UCB1 balances exploration and exploitation through a two-term formula...",0.89,0.85,0.91,0.83
...
```

## Comparison Output

```
$ python -c "import pandas as pd; baseline = pd.read_csv('results/baseline_eval.csv'); mcts = pd.read_csv('results/mcts_eval.csv'); print('Baseline:', baseline[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean()); print('MCTS:', mcts[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']].mean())"

=== Baseline (no MCTS) ===
faithfulness           0.842
answer_relevancy       0.789
context_precision      0.856
context_recall         0.812
dtype: float64

=== MCTS-Enhanced ===
faithfulness           0.871
answer_relevancy       0.823
context_precision      0.869
context_recall         0.835
dtype: float64

=== Improvement ===
faithfulness: +0.029 (+3.4%)
answer_relevancy: +0.034 (+4.3%)
context_precision: +0.013 (+1.5%)
context_recall: +0.023 (+2.8%)
```

## Prometheus Metrics Output

Query RAG metrics:
```bash
$ curl 'http://localhost:9090/api/v1/query?query=rate(mcts_rag_queries_total[5m])'
```

Response:
```json
{
  "status": "success",
  "data": {
    "resultType": "vector",
    "result": [
      {
        "metric": {
          "__name__": "mcts_rag_queries_total",
          "status": "success"
        },
        "value": [1700412645, "0.0333"]
      }
    ]
  }
}
```

## Weights & Biases Dashboard

**Run Summary:**
- **Run Name**: `rag-eval-baseline-20251119-143015`
- **Dataset**: `rag-eval-dataset`
- **MCTS Enabled**: `false`
- **Num Examples**: `50`

**Metrics:**
| Metric | Value |
|--------|-------|
| mean_faithfulness | 0.842 |
| mean_answer_relevancy | 0.789 |
| mean_context_precision | 0.856 |
| mean_context_recall | 0.812 |
| num_examples | 50 |

**Charts:**
- Metric distribution histograms
- Per-question score plots
- Time-series of evaluation runs
- Comparison with previous runs

## Grafana Dashboard

### Panels Displayed:

1. **RAG Query Rate**
   - Current: 0.033 req/s
   - 5m average: 0.028 req/s

2. **RAG Retrieval Latency**
   - P50: 142ms
   - P95: 287ms
   - P99: 412ms

3. **Documents Retrieved**
   - P50: 5 docs
   - P95: 8 docs
   - Average: 5.2 docs

4. **Relevance Scores**
   - P50: 0.76
   - P95: 0.92
   - Average: 0.79

5. **Success Rate**
   - 98.7% (green background)

6. **RAGAS Metrics (Time Series)**
   - Faithfulness: 0.842 (trending up)
   - Answer Relevancy: 0.789 (stable)
   - Context Precision: 0.856 (trending up)
   - Context Recall: 0.812 (stable)

## CI/CD Workflow Output

**GitHub Actions Log:**

```
Run python scripts/evaluate_rag.py --dataset rag-eval-dataset --limit 50 --mcts-enabled false --output results/baseline_eval.csv
  python scripts/evaluate_rag.py --dataset rag-eval-dataset --limit 50 --mcts-enabled false --output results/baseline_eval.csv
  shell: /usr/bin/bash -e {0}
  env:
    WANDB_API_KEY: ***
    LANGSMITH_API_KEY: ***
    LANGCHAIN_TRACING_V2: true
    WANDB_MODE: online
    LLM_PROVIDER: openai
    OPENAI_API_KEY: ***
    LOG_LEVEL: INFO

2025-11-19T19:30:15.123Z INFO Loading dataset 'rag-eval-dataset'...
2025-11-19T19:30:16.234Z INFO Loaded 50 examples from dataset
2025-11-19T19:30:17.345Z INFO Evaluating 50 examples with ragas (mcts=False)
...
2025-11-19T19:35:42.678Z INFO Evaluation complete!
2025-11-19T19:35:42.679Z INFO Faithfulness: 0.842
2025-11-19T19:35:42.680Z INFO Answer Relevancy: 0.789
2025-11-19T19:35:42.681Z INFO Results saved to results/baseline_eval.csv
```

**Artifacts Uploaded:**
- `rag-eval-results/baseline_eval.csv` (12.3 KB)
- `rag-eval-results/mcts_eval.csv` (12.5 KB)

## Alert Notifications

**Example Alert: Low RAGAS Faithfulness**

```
Alert: LowRAGASFaithfulness
Severity: warning
Status: firing
Description: RAGAS faithfulness score is 0.68 for dataset rag-eval-dataset (threshold: 0.70)
Labels:
  - severity: warning
  - component: rag_evaluation
  - dataset: rag-eval-dataset
Started: 2025-11-19 14:35:00 UTC
```

**Resolution Actions:**
1. Check recent model/prompt changes
2. Review low-scoring examples in W&B
3. Validate retrieval quality
4. Consider model fine-tuning if persistent

## Summary Statistics

**Typical Evaluation Run (50 examples):**
- **Duration**: 5-7 minutes (baseline), 8-12 minutes (MCTS)
- **API Calls**: ~150-200 LLM calls
- **Token Usage**: ~50K-80K tokens
- **Cost**: $0.50-$1.50 (depending on model)

**Expected Score Ranges:**
- **Faithfulness**: 0.75-0.90 (good: ≥ 0.80)
- **Answer Relevancy**: 0.70-0.85 (good: ≥ 0.75)
- **Context Precision**: 0.75-0.90 (good: ≥ 0.80)
- **Context Recall**: 0.70-0.85 (good: ≥ 0.80)

**MCTS vs Baseline Improvement:**
- Typically +2% to +5% across all metrics
- More pronounced for complex multi-hop questions
- Diminishing returns on simple factual queries
